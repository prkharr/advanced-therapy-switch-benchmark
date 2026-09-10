"""Optional modern tabular neural estimators with training-only preprocessing."""

from __future__ import annotations

import copy
from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .classical import BaseModelRunner
from .common import build_preprocessor, require_module, select_threshold
from .contracts import SUCCESS, ModelResult


def make_neural_preprocessor(frame, *, normalization="quantile", feature_count=None, seed=42):
    steps = [
        ("columns", build_preprocessor(frame, scale_numeric=False)),
        ("variance", VarianceThreshold()),
    ]
    if feature_count is not None:
        # Cap k after one-hot encoding and constant-column removal.
        steps.append(("selection", CappedFeatureSelector(int(feature_count))))
    if normalization == "quantile":
        steps.append(
            (
                "scale",
                QuantileTransformer(
                    n_quantiles=min(256, len(frame)),
                    output_distribution="normal",
                    random_state=seed,
                    subsample=None,
                ),
            )
        )
    elif normalization == "standard":
        steps.append(("scale", StandardScaler()))
    else:
        raise ValueError("normalization must be quantile or standard")
    return Pipeline(steps)


class CappedFeatureSelector(BaseEstimator):
    def __init__(self, k=30):
        self.k = k

    def fit(self, X, y):
        if self.k < 1:
            raise ValueError("feature count must be positive")
        self.selector_ = SelectKBest(f_classif, k=min(self.k, X.shape[1])).fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


class TabMEstimator(ClassifierMixin, BaseEstimator):
    """Official TabM architecture; member losses precede probability averaging.

    Numerical bins, imputation, categories, optional feature selection and scaling
    are all fitted on training rows. Validation AP selects the stopping epoch.
    No pretrained weights, network calls or test labels are needed.
    """

    def __init__(
        self,
        *,
        width=64,
        blocks=2,
        members=16,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_epochs=120,
        patience=16,
        batch_size=256,
        weight_power=0.0,
        normalization="quantile",
        feature_count=None,
        bins=0,
        embedding_dim=8,
        random_state=42,
        num_threads=2,
        architecture="tabm",
        linear_C=0.05,
    ):
        self.width = width
        self.blocks = blocks
        self.members = members
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.weight_power = weight_power
        self.normalization = normalization
        self.feature_count = feature_count
        self.bins = bins
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.num_threads = num_threads
        self.architecture = architecture
        self.linear_C = linear_C

    def fit(self, X, y, X_val, y_val):
        torch = require_module("torch")
        y, y_val = np.asarray(y, dtype=np.float32), np.asarray(y_val, dtype=np.float32)
        if set(np.unique(y)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
            raise ValueError("TabM requires both classes in training and validation")
        if min(self.width, self.blocks, self.members, self.max_epochs, self.patience) < 1:
            raise ValueError("TabM dimensions and training limits must be positive")
        if self.batch_size < 2 or not 0 <= self.weight_power <= 1:
            raise ValueError("Invalid batch size or class-weight power")
        torch.set_num_threads(self.num_threads)
        torch.manual_seed(self.random_state)
        self.preprocessor_ = make_neural_preprocessor(
            X,
            normalization=self.normalization,
            feature_count=self.feature_count,
            seed=self.random_state,
        )
        xt = torch.tensor(self.preprocessor_.fit_transform(X, y), dtype=torch.float32)
        xv = torch.tensor(self.preprocessor_.transform(X_val), dtype=torch.float32)
        yt = torch.tensor(y)
        numerical_embeddings = None
        if self.bins:
            rtdl = require_module("rtdl_num_embeddings")
            numerical_embeddings = rtdl.PiecewiseLinearEmbeddings(
                rtdl.compute_bins(xt, n_bins=self.bins),
                d_embedding=self.embedding_dim,
                activation=False,
                version="B",
            )
        if self.architecture == "tabm":
            tabm = require_module("tabm")
            self.network_ = tabm.TabM.make(
                n_num_features=xt.shape[1],
                d_out=1,
                num_embeddings=numerical_embeddings,
                d_block=self.width,
                n_blocks=self.blocks,
                k=self.members,
                dropout=self.dropout,
                arch_type="tabm",
            )
        elif self.architecture == "residual_mlp":
            from sklearn.linear_model import LogisticRegression

            from .architectures import ResidualTabularMLP

            ratio = float((1 - y).sum() / y.sum()) ** self.weight_power
            linear = LogisticRegression(
                C=self.linear_C, max_iter=1500, class_weight={0: 1, 1: ratio}
            ).fit(xt.numpy(), y)
            self.network_ = ResidualTabularMLP(xt.shape[1], self.width, self.blocks, self.dropout)
            with torch.no_grad():
                self.network_.linear.weight.copy_(torch.tensor(linear.coef_, dtype=torch.float32))
                self.network_.linear.bias.copy_(
                    torch.tensor(linear.intercept_, dtype=torch.float32)
                )
        else:
            raise ValueError("architecture must be tabm or residual_mlp")
        optimizer = torch.optim.AdamW(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        ratio = float((1 - y).sum() / y.sum()) ** self.weight_power
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio))
        self.history_ = []
        best, stale, state = -np.inf, 0, None
        if self.architecture == "residual_mlp":
            best = float(average_precision_score(y_val, self._predict_tensor(xv)))
            state = copy.deepcopy(self.network_.state_dict())
            self.best_epoch_ = 0
        generator = torch.Generator().manual_seed(self.random_state)
        for epoch in range(self.max_epochs):
            self.network_.train()
            order = torch.randperm(len(xt), generator=generator)
            losses = []
            for idx in order.split(self.batch_size):
                optimizer.zero_grad()
                logits = self.network_(xt[idx]).squeeze(-1)
                loss = criterion(logits, yt[idx, None].expand_as(logits))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network_.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            probability = self._predict_tensor(xv)
            score = float(average_precision_score(y_val, probability))
            self.history_.append(
                {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_ap": score}
            )
            if score > best + 1e-6:
                best, stale, state = score, 0, copy.deepcopy(self.network_.state_dict())
                self.best_epoch_ = epoch + 1
            else:
                stale += 1
            if stale >= self.patience:
                break
        self.network_.load_state_dict(state)
        self.network_.eval()
        self.classes_ = np.array([0, 1])
        self.validation_score_ = best
        return self

    def _predict_tensor(self, values):
        torch = require_module("torch")
        self.network_.eval()
        with torch.no_grad():
            parts = [
                torch.sigmoid(self.network_(batch).squeeze(-1)).mean(1).numpy()
                for batch in values.split(self.batch_size)
            ]
        return np.concatenate(parts).astype(float)

    def predict_proba(self, X):
        torch = require_module("torch")
        values = torch.tensor(self.preprocessor_.transform(X), dtype=torch.float32)
        positive = self._predict_tensor(values)
        return np.column_stack([1 - positive, positive])


class TabMRunner(BaseModelRunner):
    key = "tabm"
    model_name = "TabM"
    category = "Tabular DL"

    def run(self, run):
        try:
            run.validate_tabular()
            y_train, y_val, _ = run.targets()
            options = run.options_for(self.key)
            started = perf_counter()
            estimator = TabMEstimator(random_state=run.random_state, **options)
            estimator.fit(run.X_train, y_train, run.X_val, y_val)
            trained = perf_counter() - started
            val = estimator.predict_proba(run.X_val)[:, 1]
            threshold = select_threshold(y_val, val)
            started = perf_counter()
            test = estimator.predict_proba(run.X_test)[:, 1]
            return ModelResult(
                model=self.model_name,
                category=self.category,
                status=SUCCESS,
                estimator=estimator,
                validation_probabilities=val,
                test_probabilities=test,
                test_predictions=(test >= threshold).astype(np.int8),
                threshold=threshold,
                validation_score=float(average_precision_score(y_val, val)),
                best_params=estimator.get_params(),
                training_time_seconds=trained,
                inference_time_seconds=perf_counter() - started,
                history={"epochs": estimator.history_},
                metadata={"best_epoch": estimator.best_epoch_, "vocabulary_fit": "training only"},
            )
        except Exception as exc:
            return self._failed(exc)


class ExternalTabularEstimator(ClassifierMixin, BaseEstimator):
    """Optional local RealMLP, TabICL or TabPFN experiment.

    Pretrained models require explicit local checkpoints. TabICL automatic
    downloads are disabled; TabPFN verifies the file before importing its runtime.
    """

    def __init__(self, *, family="realmlp", feature_count=None, random_state=42, options=None):
        self.family = family
        self.feature_count = feature_count
        self.random_state = random_state
        self.options = options

    def fit(self, X, y, X_val, y_val):
        steps = [
            ("columns", build_preprocessor(X, scale_numeric=False)),
            ("variance", VarianceThreshold()),
        ]
        if self.feature_count is not None:
            steps.append(("selection", CappedFeatureSelector(self.feature_count)))
        self.preprocessor_ = Pipeline(steps)
        xt = np.asarray(self.preprocessor_.fit_transform(X, y), dtype=np.float32)
        xv = np.asarray(self.preprocessor_.transform(X_val), dtype=np.float32)
        options = {} if self.options is None else dict(self.options)
        if self.family == "realmlp":
            module = require_module("pytabkit")
            params = dict(
                device="cpu",
                random_state=self.random_state,
                n_cv=1,
                n_refit=0,
                n_epochs=128,
                n_threads=2,
                hidden_sizes=[128] * 3,
                val_metric_name="cross_entropy",
                use_ls=False,
                verbosity=0,
            )
            params.update(options)
            self.model_ = module.RealMLP_TD_Classifier(**params)
            self.model_.fit(xt, np.asarray(y), xv, np.asarray(y_val))
        elif self.family == "tabicl":
            module = require_module("tabicl")
            if not options.get("model_path"):
                raise ValueError("TabICL requires an explicitly supplied local checkpoint")
            if options.get("allow_auto_download", False):
                raise ValueError("Automatic checkpoint downloads are disabled")
            torch = require_module("torch")
            torch.set_num_threads(int(options.pop("num_threads", 2)))
            params = dict(
                device="cpu",
                n_estimators=4,
                batch_size=1,
                n_jobs=1,
                random_state=self.random_state,
                allow_auto_download=False,
                use_amp=False,
                use_fa3=False,
                offload_mode=False,
            )
            params.update(options)
            self.model_ = module.TabICLClassifier(**params)
            self.model_.fit(xt, np.asarray(y))
        elif self.family == "tabpfn":
            from pathlib import Path

            if not options.get("model_path") or not Path(options["model_path"]).is_file():
                raise ValueError("TabPFN requires an existing explicit local checkpoint")
            module = require_module("tabpfn")
            torch = require_module("torch")
            torch.set_num_threads(int(options.pop("num_threads", 2)))
            params = dict(
                device="cpu",
                n_estimators=2,
                random_state=self.random_state,
                n_preprocessing_jobs=1,
                ignore_pretraining_limits=True,
                memory_saving_mode=True,
            )
            params.update(options)
            self.model_ = module.TabPFNClassifier(**params)
            self.model_.fit(xt, np.asarray(y))
        else:
            raise ValueError("family must be realmlp, tabicl or tabpfn")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        values = np.asarray(self.preprocessor_.transform(X), dtype=np.float32)
        return self.model_.predict_proba(values)


class ExternalTabularRunner(BaseModelRunner):
    category = "Tabular DL"

    def run(self, run):
        try:
            run.validate_tabular()
            y_train, y_val, _ = run.targets()
            settings = run.options_for(self.key)
            feature_count = settings.pop("feature_count", None)
            started = perf_counter()
            estimator = ExternalTabularEstimator(
                family=self.key,
                feature_count=feature_count,
                random_state=run.random_state,
                options=settings,
            ).fit(run.X_train, y_train, run.X_val, y_val)
            val = estimator.predict_proba(run.X_val)[:, 1]
            trained = perf_counter() - started
            threshold = select_threshold(y_val, val)
            started = perf_counter()
            test = estimator.predict_proba(run.X_test)[:, 1]
            return ModelResult(
                model=self.model_name,
                category=self.category,
                status=SUCCESS,
                estimator=estimator,
                validation_probabilities=val,
                test_probabilities=test,
                test_predictions=(test >= threshold).astype(np.int8),
                threshold=threshold,
                validation_score=float(average_precision_score(y_val, val)),
                best_params=settings,
                training_time_seconds=trained,
                inference_time_seconds=perf_counter() - started,
                metadata={
                    "preprocessing_fit": "training only",
                    "stopping_metric": "validation cross entropy"
                    if self.key == "realmlp"
                    else "pretrained; no supervised stopping",
                },
            )
        except Exception as exc:
            return self._failed(exc)


class RealMLPRunner(ExternalTabularRunner):
    key = "realmlp"
    model_name = "RealMLP"


class TabICLRunner(ExternalTabularRunner):
    key = "tabicl"
    model_name = "TabICL"


class ProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Frozen nonnegative probability blend; selection belongs to validation."""

    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

    def predict_proba(self, X):
        weights = (
            np.ones(len(self.estimators)) if self.weights is None else np.asarray(self.weights)
        )
        if (
            not len(weights)
            or len(weights) != len(self.estimators)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError(
                "Ensemble weights must align, be finite/nonnegative and have positive sum"
            )
        probabilities = np.stack([m.predict_proba(X) for m in self.estimators])
        return np.average(probabilities, axis=0, weights=weights)


class ResidualMLPRunner(TabMRunner):
    key = "residual_mlp"
    model_name = "Residual MLP"

    def run(self, run):
        from dataclasses import replace

        options = {**run.options_for(self.key), "architecture": "residual_mlp"}
        return super().run(replace(run, params={**run.params, self.key: options}))


class TabPFNRunner(ExternalTabularRunner):
    key = "tabpfn"
    model_name = "TabPFN"


class NeuralEnsembleRunner(BaseModelRunner):
    key = "neural_ensemble"
    model_name = "Neural Ensemble"
    category = "Neural Ensemble"

    def run(self, run):
        try:
            from therapy_switch.research import fit_candidate

            run.validate_tabular()
            y_train, y_val, _ = run.targets()
            options = run.options_for(self.key)
            members = options.get("members", [])
            if not members or any(
                m["family"] not in {"tabm", "realmlp", "tabicl", "tabpfn", "mlp"} for m in members
            ):
                raise ValueError("A neural ensemble requires configured neural members")
            has_tabpfn = any(m["family"] == "tabpfn" for m in members)
            if has_tabpfn != (self.key == "tabpfn_ensemble"):
                raise ValueError("Use tabpfn_ensemble for ensembles containing TabPFN")
            train, validation = run.X_train.copy(), run.X_val.copy()
            train["label"], validation["label"] = y_train, y_val
            started = perf_counter()
            fitted = [
                fit_candidate(
                    spec, train, validation, list(run.X_train.columns), seed=run.random_state
                )
                for spec in members
            ]
            estimator = ProbabilityEnsemble(fitted, options.get("weights"))
            val = estimator.predict_proba(run.X_val)[:, 1]
            duration = perf_counter() - started
            threshold = select_threshold(y_val, val)
            started = perf_counter()
            test = estimator.predict_proba(run.X_test)[:, 1]
            return ModelResult(
                model=self.model_name,
                category=self.category,
                status=SUCCESS,
                estimator=estimator,
                validation_probabilities=val,
                test_probabilities=test,
                test_predictions=(test >= threshold).astype(np.int8),
                threshold=threshold,
                validation_score=float(average_precision_score(y_val, val)),
                best_params=options,
                training_time_seconds=duration,
                inference_time_seconds=perf_counter() - started,
                metadata={
                    "member_count": len(members),
                    "combination": "fixed nonnegative probability weights",
                    "members": [m["name"] for m in members],
                },
            )
        except Exception as exc:
            return self._failed(exc)


class TabPFNEnsembleRunner(NeuralEnsembleRunner):
    key = "tabpfn_ensemble"
    model_name = "TabPFN Neural Ensemble"
