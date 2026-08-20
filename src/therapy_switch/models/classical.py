"""Naive, interpretable, tree, and gradient-boosting model runners."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Mapping, Optional

import numpy as np

from .common import (
    DependencyUnavailable,
    average_precision,
    build_preprocessor,
    ensure_two_training_classes,
    fitted_pipeline,
    model_options,
    positive_class_weight,
    require_module,
    select_threshold,
)
from .contracts import SUCCESS, ModelResult, ModelRun
from .tuning import ParamSpec, TuningResult, tune_estimator

LOGGER = logging.getLogger(__name__)

CONTROL_OPTIONS = {
    "tune",
    "n_trials",
    "search_space",
    "prefer_optuna",
    "threshold_strategy",
    "fixed_threshold",
    "estimator_params",
    "early_stopping_rounds",
}


class BaseModelRunner(ABC):
    key: str
    model_name: str
    category: str
    requires_sequence: bool = False

    @abstractmethod
    def run(self, run: ModelRun) -> ModelResult:
        raise NotImplementedError

    def _failed(self, exc: Exception) -> ModelResult:
        if isinstance(exc, DependencyUnavailable):
            LOGGER.warning("%s is not applicable: %s", self.model_name, exc)
            return ModelResult.not_applicable(
                self.model_name, self.category, str(exc), dependency_missing=True
            )
        LOGGER.exception("%s model run failed", self.model_name)
        return ModelResult.failed(
            self.model_name,
            self.category,
            f"{type(exc).__name__}: {exc}",
        )


class NaiveBaselineRunner(BaseModelRunner):
    key = "naive_baseline"
    model_name = "Naive Baseline"
    category = "Baseline"

    def run(self, run: ModelRun) -> ModelResult:
        try:
            run.validate_tabular()
            y_train, y_val, _ = run.targets()
            options = run.options_for(self.key)
            prevalence = float(np.mean(y_train))
            started = perf_counter()
            val_probability = np.full(len(y_val), prevalence, dtype=float)
            test_probability = np.full(len(run.X_test), prevalence, dtype=float)
            inference_time = perf_counter() - started
            threshold = select_threshold(
                y_val,
                val_probability,
                strategy=str(options.get("threshold_strategy", "fixed")),
                fixed_threshold=float(options.get("fixed_threshold", 0.5)),
            )
            return ModelResult(
                model=self.model_name,
                category=self.category,
                status=SUCCESS,
                estimator={"strategy": "training_prevalence", "value": prevalence},
                validation_probabilities=val_probability,
                test_probabilities=test_probability,
                test_predictions=(test_probability >= threshold).astype(np.int8),
                threshold=threshold,
                validation_score=average_precision(y_val, val_probability),
                best_params={"constant_probability": prevalence},
                training_time_seconds=0.0,
                inference_time_seconds=inference_time,
                metadata={
                    "ranking_strategy": "constant; expected random lift is 1.0",
                    "positive_class_weight": (
                        None if prevalence in (0.0, 1.0) else (1.0 - prevalence) / prevalence
                    ),
                },
            )
        except Exception as exc:
            return self._failed(exc)


class TabularEstimatorRunner(BaseModelRunner):
    """Template implementing safe preprocessing, tuning and result capture."""

    scale_numeric = False
    default_search_space: Mapping[str, Any] = {}

    @abstractmethod
    def create_estimator(
        self,
        params: Mapping[str, Any],
        random_state: int,
        class_ratio: float,
        *,
        for_tuning: bool = False,
        early_stopping_rounds: int = 25,
    ) -> Any:
        raise NotImplementedError

    def fit_estimator(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int,
    ) -> Any:
        estimator.fit(X_train, y_train)
        return estimator

    def _estimator_params(self, options: Mapping[str, Any]) -> dict[str, Any]:
        nested = options.get("estimator_params", {})
        if not isinstance(nested, Mapping):
            raise TypeError("estimator_params must be a mapping")
        direct = model_options(options, CONTROL_OPTIONS)
        return {**direct, **dict(nested)}

    @staticmethod
    def _pipeline_search_space(search_space: Mapping[str, Any]) -> dict[str, Any]:
        return {
            (name if name.startswith("model__") else f"model__{name}"): spec
            for name, spec in search_space.items()
        }

    def _tune(
        self,
        run: ModelRun,
        estimator: Any,
        preprocessor: Any,
        options: Mapping[str, Any],
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> Optional[TuningResult]:
        if not bool(options.get("tune", False)):
            return None
        space = options.get("search_space", self.default_search_space)
        if not isinstance(space, Mapping) or not space:
            raise ValueError(f"no hyperparameter search space configured for {self.key}")
        pipeline = fitted_pipeline(preprocessor, estimator)
        return tune_estimator(
            pipeline,
            self._pipeline_search_space(space),
            run.X_train,
            y_train,
            run.X_val,
            y_val,
            n_trials=int(options.get("n_trials", 20)),
            random_state=run.random_state,
            prefer_optuna=bool(options.get("prefer_optuna", True)),
        )

    def run(self, run: ModelRun) -> ModelResult:
        try:
            run.validate_tabular()
            y_train, y_val, _ = run.targets()
            ensure_two_training_classes(y_train)
            class_ratio = positive_class_weight(y_train)
            options = run.options_for(self.key)
            estimator_params = self._estimator_params(options)
            early_stopping_rounds = int(options.get("early_stopping_rounds", 25))
            if early_stopping_rounds < 1:
                raise ValueError("early_stopping_rounds must be positive")

            started = perf_counter()
            # The search estimator intentionally omits early stopping because
            # RandomizedSearchCV owns its validation folds.
            tuning_preprocessor = build_preprocessor(run.X_train, scale_numeric=self.scale_numeric)
            search_estimator = self.create_estimator(
                estimator_params,
                run.random_state,
                class_ratio,
                for_tuning=True,
                early_stopping_rounds=early_stopping_rounds,
            )
            tuning = self._tune(
                run,
                search_estimator,
                tuning_preprocessor,
                options,
                y_train,
                y_val,
            )
            if tuning is not None:
                if tuning.status != SUCCESS:
                    reason = tuning.reason or "hyperparameter search did not complete"
                    return ModelResult.failed(self.model_name, self.category, reason)
                tuned = {
                    name.removeprefix("model__"): value
                    for name, value in tuning.best_params.items()
                }
                estimator_params.update(tuned)

            preprocessor = build_preprocessor(run.X_train, scale_numeric=self.scale_numeric)
            X_train = np.asarray(preprocessor.fit_transform(run.X_train))
            X_val = np.asarray(preprocessor.transform(run.X_val))
            estimator = self.create_estimator(
                estimator_params,
                run.random_state,
                class_ratio,
                for_tuning=False,
                early_stopping_rounds=early_stopping_rounds,
            )
            estimator = self.fit_estimator(
                estimator,
                X_train,
                y_train,
                X_val,
                y_val,
                early_stopping_rounds,
            )
            training_time = perf_counter() - started
            pipeline = fitted_pipeline(preprocessor, estimator)

            validation_probability = np.asarray(
                pipeline.predict_proba(run.X_val)[:, 1], dtype=float
            )
            threshold = select_threshold(
                y_val,
                validation_probability,
                strategy=str(options.get("threshold_strategy", "f1")),
                fixed_threshold=float(options.get("fixed_threshold", 0.5)),
            )
            inference_started = perf_counter()
            test_probability = np.asarray(pipeline.predict_proba(run.X_test)[:, 1], dtype=float)
            inference_time = perf_counter() - inference_started

            metadata: dict[str, Any] = {
                "n_transformed_features": int(X_train.shape[1]),
                "positive_class_weight": class_ratio,
                "threshold_selected_on": "validation",
            }
            best_params = dict(estimator_params)
            if tuning is not None:
                metadata["tuning"] = {
                    "backend": tuning.backend,
                    "n_trials": tuning.n_trials,
                    "duration_seconds": tuning.duration_seconds,
                    "best_validation_pr_auc": tuning.best_validation_score,
                    "trials": tuning.trials,
                }
            return ModelResult(
                model=self.model_name,
                category=self.category,
                status=SUCCESS,
                estimator=pipeline,
                validation_probabilities=validation_probability,
                test_probabilities=test_probability,
                test_predictions=(test_probability >= threshold).astype(np.int8),
                threshold=threshold,
                validation_score=average_precision(y_val, validation_probability),
                best_params=best_params,
                training_time_seconds=training_time,
                inference_time_seconds=inference_time,
                metadata=metadata,
            )
        except Exception as exc:
            return self._failed(exc)


class LogisticRegressionRunner(TabularEstimatorRunner):
    key = "logistic_regression"
    model_name = "Logistic Regression"
    category = "Classical ML"
    scale_numeric = True
    default_search_space = {
        "C": ParamSpec.floating(1e-3, 100.0, log=True),
        "penalty": ["l1", "l2"],
    }

    def create_estimator(
        self,
        params: Mapping[str, Any],
        random_state: int,
        class_ratio: float,
        **_: Any,
    ) -> Any:
        require_module("sklearn", "scikit-learn")
        from sklearn.linear_model import LogisticRegression

        defaults = {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": random_state,
            "solver": "liblinear",
        }
        defaults.update(params)
        return LogisticRegression(**defaults)


class RandomForestRunner(TabularEstimatorRunner):
    key = "random_forest"
    model_name = "Random Forest"
    category = "Classical ML"
    default_search_space = {
        "n_estimators": ParamSpec.integer(100, 500),
        "max_depth": [None, 6, 10, 16],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    }

    def create_estimator(
        self,
        params: Mapping[str, Any],
        random_state: int,
        class_ratio: float,
        **_: Any,
    ) -> Any:
        require_module("sklearn", "scikit-learn")
        from sklearn.ensemble import RandomForestClassifier

        defaults = {
            "n_estimators": 200,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": random_state,
            "min_samples_leaf": 2,
        }
        defaults.update(params)
        return RandomForestClassifier(**defaults)


class XGBoostRunner(TabularEstimatorRunner):
    key = "xgboost"
    model_name = "XGBoost"
    category = "Classical ML"
    default_search_space = {
        "n_estimators": ParamSpec.integer(100, 600),
        "max_depth": ParamSpec.integer(2, 8),
        "learning_rate": ParamSpec.floating(0.01, 0.2, log=True),
        "subsample": ParamSpec.floating(0.6, 1.0),
        "colsample_bytree": ParamSpec.floating(0.6, 1.0),
    }

    def create_estimator(
        self,
        params: Mapping[str, Any],
        random_state: int,
        class_ratio: float,
        *,
        for_tuning: bool = False,
        early_stopping_rounds: int = 25,
    ) -> Any:
        xgboost = require_module("xgboost")
        defaults = {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": class_ratio,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": random_state,
        }
        if not for_tuning:
            defaults["early_stopping_rounds"] = early_stopping_rounds
        defaults.update(params)
        return xgboost.XGBClassifier(**defaults)

    def fit_estimator(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int,
    ) -> Any:
        estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return estimator


class LightGBMRunner(TabularEstimatorRunner):
    key = "lightgbm"
    model_name = "LightGBM"
    category = "Classical ML"
    default_search_space = {
        "n_estimators": ParamSpec.integer(100, 600),
        "num_leaves": ParamSpec.integer(15, 63),
        "learning_rate": ParamSpec.floating(0.01, 0.2, log=True),
        "subsample": ParamSpec.floating(0.6, 1.0),
        "colsample_bytree": ParamSpec.floating(0.6, 1.0),
    }

    def create_estimator(
        self,
        params: Mapping[str, Any],
        random_state: int,
        class_ratio: float,
        **_: Any,
    ) -> Any:
        lightgbm = require_module("lightgbm")
        defaults = {
            "n_estimators": 400,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": class_ratio,
            "n_jobs": -1,
            "random_state": random_state,
            "verbosity": -1,
        }
        defaults.update(params)
        return lightgbm.LGBMClassifier(**defaults)

    def fit_estimator(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int,
    ) -> Any:
        lightgbm = require_module("lightgbm")
        estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[
                lightgbm.early_stopping(early_stopping_rounds, verbose=False),
                lightgbm.log_evaluation(period=0),
            ],
        )
        return estimator


class CatBoostRunner(TabularEstimatorRunner):
    key = "catboost"
    model_name = "CatBoost"
    category = "Classical ML"
    default_search_space = {
        "iterations": ParamSpec.integer(100, 600),
        "depth": ParamSpec.integer(3, 9),
        "learning_rate": ParamSpec.floating(0.01, 0.2, log=True),
        "l2_leaf_reg": ParamSpec.floating(1.0, 10.0, log=True),
    }

    def create_estimator(
        self,
        params: Mapping[str, Any],
        random_state: int,
        class_ratio: float,
        **_: Any,
    ) -> Any:
        catboost = require_module("catboost")
        defaults = {
            "iterations": 400,
            "depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": class_ratio,
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "random_seed": random_state,
            "verbose": False,
            "allow_writing_files": False,
        }
        defaults.update(params)
        return catboost.CatBoostClassifier(**defaults)

    def fit_estimator(
        self,
        estimator: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int,
    ) -> Any:
        estimator.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        return estimator
