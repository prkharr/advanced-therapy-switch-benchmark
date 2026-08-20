"""Evaluation metrics for imbalanced therapy-switch prediction.

The business-facing ranking metrics in this module use a deterministic top-k
definition: for a requested fraction ``p``, the highest-scoring
``ceil(n_patients * p)`` patients are selected (with stable input-order tie
breaking).  This makes very small cohorts usable and keeps the denominator
visible to callers.

Threshold selection is deliberately exposed only through
``tune_threshold_on_validation``.  A threshold returned by that function may
then be frozen and applied to a test population; the test labels must never be
used to choose it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_TOP_FRACTIONS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class ThresholdTuningResult:
    """A decision threshold selected using validation labels only."""

    threshold: float
    objective: str
    objective_value: float
    validation_sample_size: int
    candidate_count: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _as_binary_target(y_true: Sequence[int] | np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).reshape(-1)
    if y.size == 0:
        raise ValueError("y_true must contain at least one patient.")
    try:
        y_float = y.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must contain binary values 0 and 1.") from exc
    if not np.all(np.isfinite(y_float)):
        raise ValueError("y_true must not contain missing or infinite values.")
    unique = set(np.unique(y_float).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"y_true must be binary 0/1; observed values: {sorted(unique)}")
    return y_float.astype(np.int8)


def _as_scores(y_score: Sequence[float] | np.ndarray, expected_length: int) -> np.ndarray:
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if scores.size != expected_length:
        raise ValueError(f"y_score has {scores.size} rows but y_true has {expected_length}.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must contain only finite values.")
    return scores


def validate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    require_probabilities: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return one-dimensional binary labels and scores."""

    y = _as_binary_target(y_true)
    scores = _as_scores(y_score, len(y))
    if require_probabilities and np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Probability scores must be within [0, 1].")
    return y, scores


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if int(y_true.sum()) == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def top_k_count(n_patients: int, fraction: float) -> int:
    """Return the selected population size for a top-fraction metric."""

    if n_patients <= 0:
        raise ValueError("n_patients must be positive.")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    return min(n_patients, max(1, int(np.ceil(n_patients * float(fraction)))))


def top_fraction_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    """Compute capture, precision, and lift in the highest-scoring fraction.

    When the population contains no positive patients, recall and lift are
    undefined and returned as ``NaN``.  Precision is still well-defined (zero).
    """

    y, scores = validate_predictions(y_true, y_score)
    k = top_k_count(len(y), fraction)
    order = np.argsort(-scores, kind="stable")
    selected_y = y[order[:k]]
    selected_positives = int(selected_y.sum())
    total_positives = int(y.sum())
    prevalence = float(y.mean())
    precision = selected_positives / k
    recall = _safe_divide(selected_positives, total_positives)
    lift = _safe_divide(precision, prevalence)
    return {
        "fraction": float(fraction),
        "selected_count": k,
        "selected_positives": selected_positives,
        "total_positives": total_positives,
        "recall": recall,
        "precision": float(precision),
        "lift": lift,
    }


def ranking_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    fractions: Iterable[float] = DEFAULT_TOP_FRACTIONS,
) -> dict[str, float]:
    """Return Recall/Precision/Lift metrics at requested population fractions."""

    result: dict[str, float] = {}
    for fraction in fractions:
        values = top_fraction_metrics(y_true, y_score, float(fraction))
        label = f"{float(fraction) * 100:g}%"
        result[f"Recall@{label}"] = float(values["recall"])
        result[f"Precision@{label}"] = float(values["precision"])
        result[f"Lift@{label}"] = float(values["lift"])
    return result


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute probability and threshold-based binary classification metrics."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be within [0, 1].")
    y, scores = validate_predictions(y_true, y_score)
    predicted = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    specificity = _safe_divide(int(tn), int(tn + fp))
    # Passing explicit labels keeps log loss defined for single-class slices.
    return {
        "ROC-AUC": _safe_roc_auc(y, scores),
        "PR-AUC": _safe_average_precision(y, scores),
        "Precision": float(precision_score(y, predicted, zero_division=0)),
        "Recall": float(recall_score(y, predicted, zero_division=0)),
        "Specificity": specificity,
        "F1": float(f1_score(y, predicted, zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y, predicted)),
        "Log Loss": float(log_loss(y, scores, labels=[0, 1])),
        "Brier Score": float(brier_score_loss(y, scores)),
        "Threshold": float(threshold),
    }


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    fractions: Iterable[float] = DEFAULT_TOP_FRACTIONS,
) -> dict[str, float]:
    """Compute all standard and business-facing ranking metrics."""

    output = classification_metrics(y_true, y_score, threshold=threshold)
    output.update(ranking_metrics(y_true, y_score, fractions=fractions))
    return output


def class_imbalance_summary(
    y_true: Sequence[int] | np.ndarray,
) -> dict[str, float | int | str]:
    """Summarize class counts, prevalence, and positive-to-negative ratio."""

    y = _as_binary_target(y_true)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "positive_patients": positives,
        "negative_patients": negatives,
        "positive_prevalence": float(y.mean()),
        "positive_to_negative_ratio": (
            f"{positives}:{negatives}" if negatives else f"{positives}:0"
        ),
        "positive_negative_ratio_numeric": _safe_divide(positives, negatives),
    }


def _threshold_objective(
    name: str,
    y_true: np.ndarray,
    predicted: np.ndarray,
) -> float:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    normalized = {"max_f1": "f1", "max_recall": "recall"}.get(normalized, normalized)
    objectives: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "f1": lambda y, p: float(f1_score(y, p, zero_division=0)),
        "recall": lambda y, p: float(recall_score(y, p, zero_division=0)),
        "precision": lambda y, p: float(precision_score(y, p, zero_division=0)),
        "balanced_accuracy": lambda y, p: float(balanced_accuracy_score(y, p)),
    }
    if normalized not in objectives:
        raise ValueError("objective must be one of: f1, recall, precision, balanced_accuracy.")
    return objectives[normalized](y_true, predicted)


def tune_threshold_on_validation(
    y_validation: Sequence[int] | np.ndarray,
    validation_scores: Sequence[float] | np.ndarray,
    *,
    objective: str = "f1",
    thresholds: Iterable[float] | None = None,
    target_recall: float | None = None,
) -> ThresholdTuningResult:
    """Select a decision threshold using *validation* labels and scores only.

    Ties are resolved in favor of the higher threshold, which is deterministic
    and avoids silently increasing the number of positive predictions.
    """

    y, scores = validate_predictions(y_validation, validation_scores)
    normalized_objective = objective.lower().replace("-", "_").replace(" ", "_")
    if normalized_objective == "target_recall":
        if target_recall is None or not 0.0 < float(target_recall) <= 1.0:
            raise ValueError(
                "target_recall must be supplied in (0, 1] for objective='target_recall'."
            )
        if int(y.sum()) == 0:
            raise ValueError("target_recall tuning requires positive validation patients.")
    else:
        # Validate before the candidate loop so an empty/malformed candidate
        # collection cannot mask a misspelled objective.
        _threshold_objective(normalized_objective, y, np.zeros_like(y))
    if thresholds is None:
        candidates = np.unique(np.concatenate((np.array([0.0, 1.0]), scores.astype(float))))
    else:
        candidates = np.unique(np.asarray(list(thresholds), dtype=float))
        if candidates.size == 0:
            raise ValueError("thresholds must contain at least one value.")
        if not np.all(np.isfinite(candidates)) or np.any((candidates < 0.0) | (candidates > 1.0)):
            raise ValueError("All candidate thresholds must be finite and in [0, 1].")

    best_threshold = float(candidates[0])
    best_value = float("-inf")
    for threshold in np.sort(candidates):
        predicted = (scores >= threshold).astype(np.int8)
        if normalized_objective == "target_recall":
            achieved_recall = float(recall_score(y, predicted, zero_division=0))
            value = (
                float(precision_score(y, predicted, zero_division=0))
                if achieved_recall >= float(target_recall)
                else float("-inf")
            )
        else:
            value = _threshold_objective(objective, y, predicted)
        if value > best_value or (np.isclose(value, best_value) and threshold > best_threshold):
            best_value = value
            best_threshold = float(threshold)

    return ThresholdTuningResult(
        threshold=best_threshold,
        objective=(
            f"precision_at_recall_{float(target_recall):g}"
            if normalized_objective == "target_recall"
            else objective
        ),
        objective_value=float(best_value),
        validation_sample_size=len(y),
        candidate_count=len(candidates),
    )


# Friendly aliases used by some orchestration layers.
calculate_metrics = evaluate_predictions
calculate_ranking_metrics = ranking_metrics


__all__ = [
    "DEFAULT_TOP_FRACTIONS",
    "ThresholdTuningResult",
    "calculate_metrics",
    "calculate_ranking_metrics",
    "class_imbalance_summary",
    "classification_metrics",
    "evaluate_predictions",
    "ranking_metrics",
    "top_fraction_metrics",
    "top_k_count",
    "tune_threshold_on_validation",
    "validate_predictions",
]
