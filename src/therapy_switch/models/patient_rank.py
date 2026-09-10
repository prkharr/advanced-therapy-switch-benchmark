"""Models that target a capacity-constrained, distinct-patient list.

The EHR architectures are local synthetic implementations, not externally
pretrained medical checkpoints. All preprocessing is fitted on training data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from therapy_switch.features.history import HistoryFeatureEncoder
from therapy_switch.models.common import build_preprocessor
from therapy_switch.patient_lists import latest_patient_indices, patient_capture_metrics
from therapy_switch.research import fit_candidate


def selection_key(frame, probability):
    measured = patient_capture_metrics(frame, probability)
    return measured["recall"], measured["ap"]


@dataclass
class PatientModel:
    features: list
    estimator: object
    history: object = None
    sequence: bool = False

    def matrix(self, frame, events):
        matrix = frame[self.features].copy()
        if self.history is not None:
            matrix = pd.concat([matrix, self.history.transform(events, frame)], axis=1)
        return matrix

    def predict_scores(self, frame, events=None):
        matrix = self.matrix(frame, events)
        if self.sequence:
            return self.estimator.predict_scores(matrix, frame, events)
        return np.asarray(self.estimator.predict_proba(matrix)[:, 1], dtype=float)


class CapacityBoost:
    """Boosting with validation checkpoint selection by patient capture, then AP."""

    def __init__(self, family, options, random_state=42):
        self.family = family
        self.options = options
        self.random_state = random_state

    def fit(self, xt, yt, xv, yv, validation):
        options = copy.deepcopy(self.options)
        steps = int(options.pop("iterations", 600))
        power = float(options.pop("weight_power", 0.0))
        ratio = (len(yt) - np.sum(yt)) / np.sum(yt)
        self.preprocessor_ = build_preprocessor(xt, scale_numeric=False)
        at = np.asarray(self.preprocessor_.fit_transform(xt))
        av = np.asarray(self.preprocessor_.transform(xv))
        if self.family == "catboost":
            from catboost import CatBoostClassifier

            self.model_ = CatBoostClassifier(
                iterations=steps,
                learning_rate=0.03,
                loss_function="Logloss",
                random_seed=self.random_state,
                thread_count=2,
                verbose=False,
                allow_writing_files=False,
                scale_pos_weight=ratio**power,
                **options,
            ).fit(at, yt)
            scores = [
                (i, self.model_.predict_proba(av, ntree_end=i)[:, 1])
                for i in range(20, steps + 1, 20)
            ]
        elif self.family == "xgboost":
            from xgboost import XGBClassifier

            self.model_ = XGBClassifier(
                n_estimators=steps,
                learning_rate=0.03,
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=2,
                eval_metric="aucpr",
                scale_pos_weight=ratio**power,
                **options,
            ).fit(at, yt)
            scores = [
                (i, self.model_.predict_proba(av, iteration_range=(0, i))[:, 1])
                for i in range(20, steps + 1, 20)
            ]
        elif self.family == "lgbm_capacity":
            from lightgbm import LGBMClassifier

            self.model_ = LGBMClassifier(
                n_estimators=steps,
                learning_rate=0.025,
                n_jobs=2,
                verbosity=-1,
                random_state=self.random_state,
                deterministic=True,
                force_col_wise=True,
                scale_pos_weight=ratio**power,
                **options,
            ).fit(at, yt)
            scores = [
                (i, self.model_.predict_proba(av, num_iteration=i)[:, 1])
                for i in range(20, steps + 1, 20)
            ]
        else:
            raise ValueError(f"Unsupported capacity boosting family: {self.family}")
        if not scores:
            raise ValueError("At least 20 boosting iterations are required")
        self.best_iteration_, _ = max(scores, key=lambda pair: selection_key(validation, pair[1]))
        return self

    def predict_proba(self, frame):
        values = self.preprocessor_.transform(frame)
        if self.family == "catboost":
            return self.model_.predict_proba(values, ntree_end=self.best_iteration_)
        if self.family == "xgboost":
            return self.model_.predict_proba(values, iteration_range=(0, self.best_iteration_))
        return self.model_.predict_proba(values, num_iteration=self.best_iteration_)


class LambdaRankModel:
    """Quarter-grouped LambdaRank; scores mapped monotonically using training data.

    The sigmoid is a score transformation, not probability calibration.
    """

    def __init__(self, options, random_state=42):
        self.options = options
        self.random_state = random_state

    def fit(self, xt, yt, xv, yv, train, validation):
        from lightgbm import LGBMRanker

        options = copy.deepcopy(self.options)
        steps = int(options.pop("iterations", 600))
        self.preprocessor_ = build_preprocessor(xt)
        groups = pd.to_datetime(train.index_date).dt.to_period("Q").astype(str)
        order = np.argsort(groups.to_numpy(), kind="stable")
        at = np.asarray(self.preprocessor_.fit_transform(xt))
        av = np.asarray(self.preprocessor_.transform(xv))
        lengths = groups.iloc[order].groupby(groups.iloc[order], sort=False).size().to_numpy()
        self.model_ = LGBMRanker(
            objective="lambdarank",
            label_gain=[0, 1],
            n_estimators=steps,
            learning_rate=0.025,
            n_jobs=2,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=self.random_state,
            lambdarank_truncation_level=max(2, int(np.ceil(np.mean(lengths) * 0.1))),
            **options,
        ).fit(at[order], np.asarray(yt)[order], group=lengths)
        checkpoints = []
        for i in range(20, steps + 1, 20):
            train_scores = self.model_.predict(at, num_iteration=i)
            center, scale = np.mean(train_scores), max(np.std(train_scores), 1e-6)
            probability = expit((self.model_.predict(av, num_iteration=i) - center) / scale)
            checkpoints.append((selection_key(validation, probability), i, center, scale))
        _, self.best_iteration_, self.center_, self.scale_ = max(
            checkpoints, key=lambda row: row[0]
        )
        return self

    def predict_proba(self, frame):
        raw = self.model_.predict(
            self.preprocessor_.transform(frame), num_iteration=self.best_iteration_
        )
        p = expit((raw - self.center_) / self.scale_)
        return np.column_stack([1 - p, p])


def fit_patient_model(spec, train, validation, train_events, validation_events, features, *, seed):
    """Fit without a test argument; vocabulary and transformations use train only."""
    train = train.copy()
    validation = validation.copy()
    if spec.get("latest_only", False):
        train = train.iloc[latest_patient_indices(train)].copy()
    history = None
    xt, xv = train[features].copy(), validation[features].copy()
    if spec.get("history", False):
        history = HistoryFeatureEncoder().fit(train_events, train)
        xt = pd.concat([xt, history.transform(train_events, train)], axis=1)
        xv = pd.concat([xv, history.transform(validation_events, validation)], axis=1)
    family = spec["family"]
    options = copy.deepcopy(spec["options"])
    yt, yv = train.label.to_numpy(), validation.label.to_numpy()
    sequence = family in {"ehr_transformer", "retain", "rank_mlp"}
    if family in {"catboost", "xgboost", "lgbm_capacity"}:
        estimator = CapacityBoost(family, options, seed).fit(xt, yt, xv, yv, validation)
    elif family == "lambdarank":
        estimator = LambdaRankModel(options, seed).fit(xt, yt, xv, yv, train, validation)
    elif sequence:
        from therapy_switch.models.patient_neural import PatientNeuralEstimator

        estimator = PatientNeuralEstimator(family, options, seed).fit(
            xt, train, train_events, xv, validation, validation_events
        )
    else:
        # Legacy model families preserve their original training implementation.
        left, right = xt.copy(), xv.copy()
        left["label"], right["label"] = yt, yv
        estimator = fit_candidate(spec, left, right, list(xt), seed=seed)
    return PatientModel(list(features), estimator, history, sequence)


@dataclass
class PatientEnsemble:
    members: list
    weights: list

    def predict_scores(self, frame, events=None):
        if not self.members or len(self.members) != len(self.weights):
            raise ValueError("An ensemble requires aligned members and weights")
        weights = np.asarray(self.weights, dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Ensemble weights must be finite, nonnegative, and sum above zero")
        return np.average(
            [model.predict_scores(frame, events) for model in self.members], axis=0, weights=weights
        )
