"""Decile and cumulative-gains analyses for propensity rankings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import top_k_count, validate_predictions

DECILE_COLUMNS = [
    "model",
    "decile",
    "patient_count",
    "actual_switchers",
    "switch_rate",
    "precision",
    "recall_in_decile",
    "cumulative_switchers",
    "cumulative_recall_pct",
    "lift",
    "cumulative_lift",
    "avg_predicted_probability",
]

GAINS_COLUMNS = [
    "model",
    "population_percentile",
    "population_targeted_pct",
    "patient_count",
    "cumulative_switchers",
    "actual_switchers_captured_pct",
    "cumulative_precision",
    "cumulative_lift",
]


def decile_analysis(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    model: str,
) -> pd.DataFrame:
    """Rank patients and return ten high-to-low propensity groups.

    Ten rows are always emitted.  For populations smaller than ten, trailing
    deciles are empty and their rate metrics are undefined (``NaN``).
    """

    y, scores = validate_predictions(y_true, y_score)
    ranked_indices = np.argsort(-scores, kind="stable")
    groups = np.array_split(ranked_indices, 10)
    total_switchers = int(y.sum())
    prevalence = float(y.mean())
    cumulative_switchers = 0
    cumulative_patients = 0
    rows: list[dict[str, float | int | str]] = []

    for decile_number, indices in enumerate(groups, start=1):
        patient_count = int(len(indices))
        switchers = int(y[indices].sum()) if patient_count else 0
        cumulative_switchers += switchers
        cumulative_patients += patient_count
        switch_rate = switchers / patient_count if patient_count else float("nan")
        recall_in_decile = switchers / total_switchers if total_switchers else float("nan")
        cumulative_recall = (
            cumulative_switchers / total_switchers if total_switchers else float("nan")
        )
        lift = switch_rate / prevalence if patient_count and prevalence > 0.0 else float("nan")
        cumulative_precision = (
            cumulative_switchers / cumulative_patients if cumulative_patients else float("nan")
        )
        cumulative_lift = (
            cumulative_precision / prevalence
            if cumulative_patients and prevalence > 0.0
            else float("nan")
        )
        rows.append(
            {
                "model": model,
                "decile": decile_number,
                "patient_count": patient_count,
                "actual_switchers": switchers,
                "switch_rate": switch_rate,
                # Precision within a disjoint decile is its observed switch rate.
                "precision": switch_rate,
                "recall_in_decile": recall_in_decile,
                "cumulative_switchers": cumulative_switchers,
                "cumulative_recall_pct": cumulative_recall,
                "lift": lift,
                "cumulative_lift": cumulative_lift,
                "avg_predicted_probability": (
                    float(scores[indices].mean()) if patient_count else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows, columns=DECILE_COLUMNS)


def all_models_decile_analysis(
    y_true: Sequence[int] | np.ndarray,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
) -> pd.DataFrame:
    """Create the requested decile output for every supplied model."""

    frames = [decile_analysis(y_true, scores, model=model) for model, scores in predictions.items()]
    if not frames:
        return pd.DataFrame(columns=DECILE_COLUMNS)
    return pd.concat(frames, ignore_index=True)[DECILE_COLUMNS]


def cumulative_gains(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    model: str,
    percentiles: Iterable[int] = range(1, 101),
) -> pd.DataFrame:
    """Calculate switcher capture at requested cumulative population percentiles."""

    y, scores = validate_predictions(y_true, y_score)
    ordered_y = y[np.argsort(-scores, kind="stable")]
    total_switchers = int(y.sum())
    prevalence = float(y.mean())
    cumulative_positive = np.cumsum(ordered_y)
    rows: list[dict[str, float | int | str]] = []
    seen: set[int] = set()
    for percentile in percentiles:
        percentile_int = int(percentile)
        if percentile_int != percentile or not 1 <= percentile_int <= 100:
            raise ValueError("percentiles must be unique integers from 1 through 100.")
        if percentile_int in seen:
            continue
        seen.add(percentile_int)
        k = top_k_count(len(y), percentile_int / 100.0)
        switchers = int(cumulative_positive[k - 1])
        precision = switchers / k
        rows.append(
            {
                "model": model,
                "population_percentile": percentile_int,
                "population_targeted_pct": k / len(y),
                "patient_count": k,
                "cumulative_switchers": switchers,
                "actual_switchers_captured_pct": (
                    switchers / total_switchers if total_switchers else float("nan")
                ),
                "cumulative_precision": precision,
                "cumulative_lift": (precision / prevalence if prevalence > 0.0 else float("nan")),
            }
        )
    return pd.DataFrame(rows, columns=GAINS_COLUMNS)


def all_models_cumulative_gains(
    y_true: Sequence[int] | np.ndarray,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    percentiles: Iterable[int] = range(1, 101),
) -> pd.DataFrame:
    percentile_values = tuple(percentiles)
    frames = [
        cumulative_gains(y_true, scores, model=model, percentiles=percentile_values)
        for model, scores in predictions.items()
    ]
    if not frames:
        return pd.DataFrame(columns=GAINS_COLUMNS)
    return pd.concat(frames, ignore_index=True)[GAINS_COLUMNS]


def save_decile_analysis(
    frame: pd.DataFrame,
    path: str | Path = "outputs/decile_analysis.csv",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def save_cumulative_gains(
    frame: pd.DataFrame,
    path: str | Path = "outputs/cumulative_gains.csv",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


# Backward-friendly verb spelling.
calculate_decile_analysis = decile_analysis
calculate_cumulative_gains = cumulative_gains


__all__ = [
    "DECILE_COLUMNS",
    "GAINS_COLUMNS",
    "all_models_cumulative_gains",
    "all_models_decile_analysis",
    "calculate_cumulative_gains",
    "calculate_decile_analysis",
    "cumulative_gains",
    "decile_analysis",
    "save_cumulative_gains",
    "save_decile_analysis",
]
