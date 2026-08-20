"""PR-AUC-oriented hyperparameter search with an Optuna-first policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .common import DependencyUnavailable, require_module
from .contracts import FAILED, NOT_APPLICABLE, SUCCESS


@dataclass(frozen=True)
class ParamSpec:
    """Backend-neutral parameter distribution.

    Kinds are ``categorical``, ``int`` and ``float``. Integer and float ranges
    include both bounds; ``log=True`` requests logarithmic sampling.
    """

    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Sequence[Any]] = None
    log: bool = False

    @classmethod
    def categorical(cls, choices: Sequence[Any]) -> "ParamSpec":
        return cls(kind="categorical", choices=tuple(choices))

    @classmethod
    def integer(cls, low: int, high: int, log: bool = False) -> "ParamSpec":
        return cls(kind="int", low=low, high=high, log=log)

    @classmethod
    def floating(cls, low: float, high: float, log: bool = False) -> "ParamSpec":
        return cls(kind="float", low=low, high=high, log=log)


@dataclass
class TuningResult:
    status: str
    backend: str
    best_params: dict[str, Any] = field(default_factory=dict)
    best_validation_score: Optional[float] = None
    n_trials: int = 0
    duration_seconds: float = 0.0
    trials: list[dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None


def _optuna_value(trial: Any, name: str, spec: Any) -> Any:
    if isinstance(spec, ParamSpec):
        if spec.kind == "categorical":
            if not spec.choices:
                raise ValueError(f"empty categorical search space for {name}")
            return trial.suggest_categorical(name, list(spec.choices))
        if spec.low is None or spec.high is None:
            raise ValueError(f"search bounds are required for {name}")
        if spec.kind == "int":
            return trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
        if spec.kind == "float":
            return trial.suggest_float(name, float(spec.low), float(spec.high), log=spec.log)
        raise ValueError(f"unknown parameter kind {spec.kind!r} for {name}")
    if isinstance(spec, (list, tuple, np.ndarray)):
        if len(spec) == 0:
            raise ValueError(f"empty categorical search space for {name}")
        return trial.suggest_categorical(name, list(spec))
    raise TypeError(f"search space entry {name!r} must be ParamSpec or a finite sequence")


def _randomized_values(spec: Any) -> list[Any]:
    if isinstance(spec, ParamSpec):
        if spec.kind == "categorical":
            return list(spec.choices or ())
        if spec.low is None or spec.high is None:
            raise ValueError("numeric ParamSpec requires low/high")
        if spec.kind == "int":
            low, high = int(spec.low), int(spec.high)
            values = np.unique(np.linspace(low, high, min(20, high - low + 1)).round())
            return values.astype(int).tolist()
        if spec.kind == "float":
            if spec.log:
                return np.geomspace(float(spec.low), float(spec.high), 20).tolist()
            return np.linspace(float(spec.low), float(spec.high), 20).tolist()
        raise ValueError(f"unknown parameter kind {spec.kind!r}")
    if isinstance(spec, (list, tuple, np.ndarray)):
        return list(spec)
    raise TypeError("search entries must be ParamSpec or finite sequences")


def _optuna_search(
    estimator: Any,
    search_space: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[np.ndarray],
    n_trials: int,
    random_state: int,
) -> TuningResult:
    optuna = require_module("optuna")
    require_module("sklearn", "scikit-learn")
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    started = perf_counter()

    def objective(trial: Any) -> float:
        params = {name: _optuna_value(trial, name, spec) for name, spec in search_space.items()}
        candidate = clone(estimator).set_params(**params)
        if X_val is not None and y_val is not None:
            candidate.fit(X_train, y_train)
            probability = candidate.predict_proba(X_val)[:, 1]
            return float(average_precision_score(y_val, probability))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials = [
        {
            "number": trial.number,
            "score": trial.value,
            "params": dict(trial.params),
            "state": str(trial.state),
        }
        for trial in study.trials
    ]
    return TuningResult(
        status=SUCCESS,
        backend="optuna",
        best_params=dict(study.best_params),
        best_validation_score=float(study.best_value),
        n_trials=len(study.trials),
        duration_seconds=perf_counter() - started,
        trials=trials,
    )


def _randomized_search(
    estimator: Any,
    search_space: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[np.ndarray],
    n_trials: int,
    random_state: int,
) -> TuningResult:
    require_module("sklearn", "scikit-learn")
    from sklearn.model_selection import (
        PredefinedSplit,
        RandomizedSearchCV,
        StratifiedKFold,
    )

    started = perf_counter()
    distributions = {name: _randomized_values(spec) for name, spec in search_space.items()}
    if any(len(values) == 0 for values in distributions.values()):
        raise ValueError("hyperparameter search spaces cannot be empty")

    if X_val is not None and y_val is not None:
        X_search = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_search = np.concatenate([y_train, y_val])
        fold = np.r_[
            np.full(len(y_train), -1, dtype=int),
            np.zeros(len(y_val), dtype=int),
        ]
        cv: Any = PredefinedSplit(fold)
    else:
        X_search = X_train
        y_search = y_train
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=distributions,
        n_iter=n_trials,
        scoring="average_precision",
        n_jobs=1,
        cv=cv,
        refit=False,
        random_state=random_state,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_search, y_search)
    results = search.cv_results_
    trials = [
        {
            "number": index,
            "score": float(results["mean_test_score"][index]),
            "params": dict(results["params"][index]),
        }
        for index in range(len(results["params"]))
    ]
    best_index = int(np.nanargmax(results["mean_test_score"]))
    return TuningResult(
        status=SUCCESS,
        backend="randomized_search_cv",
        best_params=dict(results["params"][best_index]),
        best_validation_score=float(results["mean_test_score"][best_index]),
        n_trials=len(trials),
        duration_seconds=perf_counter() - started,
        trials=trials,
    )


def tune_estimator(
    estimator: Any,
    search_space: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[Sequence[int]] = None,
    *,
    n_trials: int = 20,
    random_state: int = 42,
    prefer_optuna: bool = True,
) -> TuningResult:
    """Tune an estimator for average precision (PR-AUC).

    Optuna is used when requested and installed. Otherwise the fallback is
    explicitly ``RandomizedSearchCV``. With a validation partition, a
    ``PredefinedSplit`` ensures candidates are fitted only on training rows and
    scored on validation rows.
    """

    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    y_train_array = np.asarray(y_train, dtype=int).reshape(-1)
    y_val_array = None if y_val is None else np.asarray(y_val, dtype=int).reshape(-1)
    if (X_val is None) != (y_val_array is None):
        raise ValueError("X_val and y_val must be provided together")

    if prefer_optuna:
        try:
            return _optuna_search(
                estimator,
                search_space,
                X_train,
                y_train_array,
                X_val,
                y_val_array,
                n_trials,
                random_state,
            )
        except DependencyUnavailable:
            # An absent Optuna package is the documented trigger for the
            # RandomizedSearchCV fallback; model/search failures still surface.
            pass

    try:
        return _randomized_search(
            estimator,
            search_space,
            X_train,
            y_train_array,
            X_val,
            y_val_array,
            n_trials,
            random_state,
        )
    except DependencyUnavailable as exc:
        return TuningResult(
            status=NOT_APPLICABLE,
            backend="unavailable",
            n_trials=0,
            reason=str(exc),
        )
    except Exception as exc:
        return TuningResult(
            status=FAILED,
            backend="randomized_search_cv",
            n_trials=0,
            reason=f"hyperparameter search failed: {type(exc).__name__}: {exc}",
        )
