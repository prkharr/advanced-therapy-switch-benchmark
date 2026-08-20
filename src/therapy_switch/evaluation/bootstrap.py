"""Stratified and paired bootstrap confidence intervals."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import top_fraction_metrics, validate_predictions

BOOTSTRAP_METRICS = (
    "ROC-AUC",
    "PR-AUC",
    "Recall@Top10%",
    "Recall@Top20%",
    "Lift@Top10%",
    "Lift@Top20%",
)


def _metric_values(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    top_10 = top_fraction_metrics(y_true, y_score, 0.10)
    top_20 = top_fraction_metrics(y_true, y_score, 0.20)
    return {
        "ROC-AUC": float(roc_auc_score(y_true, y_score)),
        "PR-AUC": float(average_precision_score(y_true, y_score)),
        "Recall@Top10%": float(top_10["recall"]),
        "Recall@Top20%": float(top_20["recall"]),
        "Lift@Top10%": float(top_10["lift"]),
        "Lift@Top20%": float(top_20["lift"]),
    }


def _stratified_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample each observed class with replacement and retain its class count."""

    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    sampled = np.concatenate(
        (
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        )
    )
    rng.shuffle(sampled)
    return sampled


def _validate_bootstrap_inputs(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    y, scores = validate_predictions(y_true, y_score)
    if np.unique(y).size < 2:
        raise ValueError("Bootstrap inference requires at least one patient per class.")
    if int(n_bootstrap) != n_bootstrap or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least 2.")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")
    return y, scores


def bootstrap_confidence_intervals(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    model: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return percentile CIs using class-stratified patient resampling."""

    y, scores = _validate_bootstrap_inputs(y_true, y_score, n_bootstrap, confidence_level)
    point = _metric_values(y, scores)
    samples = {metric: [] for metric in BOOTSTRAP_METRICS}
    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        indices = _stratified_indices(y, rng)
        values = _metric_values(y[indices], scores[indices])
        for metric in BOOTSTRAP_METRICS:
            if np.isfinite(values[metric]):
                samples[metric].append(values[metric])

    alpha = 1.0 - confidence_level
    rows: list[dict[str, float | int | str]] = []
    for metric in BOOTSTRAP_METRICS:
        distribution = np.asarray(samples[metric], dtype=float)
        rows.append(
            {
                "model": model,
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(np.quantile(distribution, alpha / 2.0)),
                "ci_upper": float(np.quantile(distribution, 1.0 - alpha / 2.0)),
                "standard_error": float(np.std(distribution, ddof=1)),
                "confidence_level": float(confidence_level),
                "successful_samples": int(distribution.size),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_all_models(
    y_true: Sequence[int] | np.ndarray,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate reproducible confidence intervals for each valid model."""

    frames = []
    for offset, (model, scores) in enumerate(predictions.items()):
        frames.append(
            bootstrap_confidence_intervals(
                y_true,
                scores,
                model=model,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state + offset,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_bootstrap_comparison(
    y_true: Sequence[int] | np.ndarray,
    classical_scores: Sequence[float] | np.ndarray,
    deep_learning_scores: Sequence[float] | np.ndarray,
    *,
    classical_model: str,
    deep_learning_model: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare models on identical resampled patients.

    Differences are defined as ``deep learning - classical``.  The two-sided
    bootstrap p-value tests whether zero lies away from the empirical
    difference distribution; confidence intervals are the primary result.
    """

    y, classical = _validate_bootstrap_inputs(
        y_true, classical_scores, n_bootstrap, confidence_level
    )
    _, deep_learning = validate_predictions(y, deep_learning_scores)
    classical_point = _metric_values(y, classical)
    dl_point = _metric_values(y, deep_learning)
    differences = {metric: [] for metric in BOOTSTRAP_METRICS}
    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        indices = _stratified_indices(y, rng)
        classical_values = _metric_values(y[indices], classical[indices])
        dl_values = _metric_values(y[indices], deep_learning[indices])
        for metric in BOOTSTRAP_METRICS:
            difference = dl_values[metric] - classical_values[metric]
            if np.isfinite(difference):
                differences[metric].append(difference)

    alpha = 1.0 - confidence_level
    rows: list[dict[str, float | int | bool | str]] = []
    for metric in BOOTSTRAP_METRICS:
        distribution = np.asarray(differences[metric], dtype=float)
        lower = float(np.quantile(distribution, alpha / 2.0))
        upper = float(np.quantile(distribution, 1.0 - alpha / 2.0))
        below_or_equal = (np.count_nonzero(distribution <= 0.0) + 1) / (distribution.size + 1)
        above_or_equal = (np.count_nonzero(distribution >= 0.0) + 1) / (distribution.size + 1)
        p_value = min(1.0, 2.0 * min(below_or_equal, above_or_equal))
        rows.append(
            {
                "metric": metric,
                "classical_model": classical_model,
                "deep_learning_model": deep_learning_model,
                "classical_estimate": classical_point[metric],
                "deep_learning_estimate": dl_point[metric],
                "difference_dl_minus_classical": (dl_point[metric] - classical_point[metric]),
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": float(confidence_level),
                "p_value_two_sided": float(p_value),
                "statistically_significant": bool(lower > 0.0 or upper < 0.0),
                "successful_samples": int(distribution.size),
            }
        )
    return pd.DataFrame(rows)


# Explicit aliases for callers that use "stratified" in the function name.
stratified_bootstrap_ci = bootstrap_confidence_intervals
paired_stratified_bootstrap = paired_bootstrap_comparison


__all__ = [
    "BOOTSTRAP_METRICS",
    "bootstrap_all_models",
    "bootstrap_confidence_intervals",
    "paired_bootstrap_comparison",
    "paired_stratified_bootstrap",
    "stratified_bootstrap_ci",
]
