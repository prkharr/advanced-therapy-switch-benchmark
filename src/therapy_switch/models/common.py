"""Dependency-light helpers shared by model runners."""

from __future__ import annotations

import importlib
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class DependencyUnavailable(ImportError):
    """A model dependency was not installed in the active environment."""


def require_module(module: str, install_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        hint = install_hint or module
        raise DependencyUnavailable(
            f"optional dependency {module!r} is unavailable; install {hint!r}"
        ) from exc


def ensure_two_training_classes(y: np.ndarray) -> None:
    if np.unique(y).size != 2:
        raise ValueError("training target must contain both negative and positive patients")


def positive_class_weight(y: np.ndarray) -> float:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("class weighting requires both target classes in training data")
    return negatives / positives


def average_precision(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Average precision with an sklearn implementation when available.

    The small local implementation preserves a functioning naive baseline in
    environments where the modeling extras have not been installed.
    """

    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = y[order]
    ranked_scores = scores[order]
    cumulative = np.cumsum(ranked)
    # Evaluate only at the end of each tied-score group. Treating arbitrary tie
    # order as rank information would give a constant naive predictor a false
    # AP above or below prevalence.
    group_ends = np.r_[np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]), len(y) - 1]
    true_positives = cumulative[group_ends]
    precision = true_positives / (group_ends + 1)
    recall = true_positives / positives
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    strategy: str = "f1",
    fixed_threshold: float = 0.5,
) -> float:
    """Select a decision threshold using validation data only."""

    if strategy == "fixed":
        if not 0 <= fixed_threshold <= 1:
            raise ValueError("fixed_threshold must be between zero and one")
        return float(fixed_threshold)
    if strategy != "f1":
        raise ValueError("threshold_strategy must be 'f1' or 'fixed'")

    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(scores) == 0:
        return float(fixed_threshold)
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best_threshold = float(fixed_threshold)
    best_f1 = -1.0
    for threshold in candidates:
        prediction = scores >= threshold
        tp = float(np.sum((prediction == 1) & (y == 1)))
        fp = float(np.sum((prediction == 1) & (y == 0)))
        fn = float(np.sum((prediction == 0) & (y == 1)))
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else 2 * tp / denominator
        # Prefer the larger threshold on ties to avoid unnecessary targeting.
        if f1 > best_f1 or (math.isclose(f1, best_f1) and threshold > best_threshold):
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def sklearn_components() -> Mapping[str, Any]:
    """Import sklearn lazily, keeping package import usable without its extras."""

    require_module("sklearn", "scikit-learn")
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return {
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def build_preprocessor(frame: pd.DataFrame, scale_numeric: bool = False) -> Any:
    components = sklearn_components()
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", components["SimpleImputer"](strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", components["StandardScaler"]()))

    try:
        encoder = components["OneHotEncoder"](handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = components["OneHotEncoder"](handle_unknown="ignore", sparse=False)

    transformers: list[tuple[str, Any, Iterable[Any]]] = []
    if numeric_columns:
        transformers.append(("numeric", components["Pipeline"](numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                components["Pipeline"](
                    [
                        (
                            "imputer",
                            components["SimpleImputer"](strategy="most_frequent"),
                        ),
                        ("one_hot", encoder),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("at least one model feature is required")
    return components["ColumnTransformer"](
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def fitted_pipeline(preprocessor: Any, estimator: Any) -> Any:
    pipeline = sklearn_components()["Pipeline"](
        [("preprocessor", preprocessor), ("model", estimator)]
    )
    return pipeline


def model_options(options: Mapping[str, Any], reserved: Iterable[str]) -> dict[str, Any]:
    reserved_set = set(reserved)
    return {key: value for key, value in options.items() if key not in reserved_set}


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
