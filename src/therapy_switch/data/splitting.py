"""Patient-safe random and out-of-time dataset splitting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ._config import config_value


def _fractions(config: Any) -> tuple[float, float, float]:
    configured_train = config_value(
        config,
        "train_fraction",
        "split.train_fraction",
        "splitting.train_fraction",
        default=None,
    )
    validation = float(
        config_value(
            config,
            "validation_fraction",
            "val_fraction",
            "split.validation_fraction",
            "split.val_fraction",
            "splitting.validation_fraction",
            "splitting.val_fraction",
            default=0.15,
        )
    )
    test = float(
        config_value(
            config, "test_fraction", "split.test_fraction", "splitting.test_fraction", default=0.15
        )
    )
    train = 1.0 - validation - test if configured_train is None else float(configured_train)
    if min(train, validation, test) <= 0:
        raise ValueError("Train, validation, and test fractions must all be positive")
    if not np.isclose(train + validation + test, 1.0, atol=1e-8):
        raise ValueError("Train, validation, and test fractions must sum to 1")
    return train, validation, test


def _patient_summary(
    data: pd.DataFrame,
    patient_col: str,
    label_col: str,
    index_date_col: str | None = None,
) -> pd.DataFrame:
    required = {patient_col, label_col}
    if index_date_col is not None:
        required.add(index_date_col)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Data is missing split columns: {missing}")
    if data[patient_col].isna().any():
        raise ValueError(f"{patient_col} cannot contain missing values")
    label_counts = data.groupby(patient_col, dropna=False)[label_col].nunique(dropna=False)
    if (label_counts > 1).any():
        bad = label_counts.loc[label_counts > 1].index[:5].tolist()
        raise ValueError(f"Patients have inconsistent labels: {bad}")
    aggregations: dict[str, str] = {label_col: "first"}
    if index_date_col is not None:
        date_counts = data.groupby(patient_col, dropna=False)[index_date_col].nunique(dropna=False)
        if (date_counts > 1).any():
            bad = date_counts.loc[date_counts > 1].index[:5].tolist()
            raise ValueError(f"Patients have inconsistent index dates: {bad}")
        aggregations[index_date_col] = "first"
    return data.groupby(patient_col, as_index=False, dropna=False).agg(aggregations)


def _allocate_counts(n: int, fractions: tuple[float, float, float]) -> np.ndarray:
    raw = np.asarray(fractions) * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    for position in np.argsort(-(raw - counts))[:remainder]:
        counts[position] += 1
    if n >= 3:
        for empty_position in np.flatnonzero(counts == 0):
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[empty_position] += 1
    return counts


def validate_patient_disjoint(
    splits: Mapping[str, pd.DataFrame], patient_col: str = "patient_id"
) -> None:
    """Raise ``AssertionError`` if a patient occurs in more than one split."""

    names = list(splits)
    identifiers = {name: set(splits[name][patient_col].tolist()) for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = identifiers[left] & identifiers[right]
            if overlap:
                preview = sorted(str(item) for item in overlap)[:5]
                raise AssertionError(f"Patient overlap between {left} and {right}: {preview}")


def stratified_patient_split(
    data: pd.DataFrame,
    config: Mapping[str, Any] | Any | None = None,
    *,
    patient_col: str = "patient_id",
    label_col: str = "label",
) -> dict[str, pd.DataFrame]:
    """Split rows by unique patients while approximately preserving prevalence.

    Unlike a row-level train/test split, this remains safe when ``data`` contains
    multiple events per patient. Small classes are allocated to all three splits
    when at least three patients from that class are available.
    """

    config = {} if config is None else config
    fractions = _fractions(config)
    seed = int(
        config_value(
            config,
            "random_seed",
            "split.random_seed",
            "splitting.random_seed",
            "project.random_seed",
            default=42,
        )
    )
    summary = _patient_summary(data, patient_col, label_col)
    if summary.empty:
        raise ValueError("Cannot split an empty dataset")
    rng = np.random.default_rng(seed)
    assigned: dict[str, list[Any]] = {"train": [], "validation": [], "test": []}
    for _, stratum in summary.groupby(label_col, dropna=False, sort=True):
        patient_ids = stratum[patient_col].to_numpy(copy=True)
        rng.shuffle(patient_ids)
        train_count, validation_count, _ = _allocate_counts(len(patient_ids), fractions)
        assigned["train"].extend(patient_ids[:train_count].tolist())
        assigned["validation"].extend(
            patient_ids[train_count : train_count + validation_count].tolist()
        )
        assigned["test"].extend(patient_ids[train_count + validation_count :].tolist())

    result: dict[str, pd.DataFrame] = {}
    for split_name, patient_ids in assigned.items():
        subset = data.loc[data[patient_col].isin(patient_ids)].copy()
        # Deterministic ordering makes saved partitions reproducible without
        # imposing an order requirement on downstream learners.
        subset.sort_values(patient_col, kind="stable", inplace=True)
        subset.reset_index(drop=True, inplace=True)
        result[split_name] = subset
    validate_patient_disjoint(result, patient_col)
    return result


def temporal_patient_split(
    data: pd.DataFrame,
    config: Mapping[str, Any] | Any | None = None,
    *,
    patient_col: str = "patient_id",
    label_col: str = "label",
    index_date_col: str = "index_date",
) -> dict[str, pd.DataFrame]:
    """Create earlier-train/later-validation/most-recent-test partitions.

    All patients sharing an index date are kept in the same partition, avoiding
    an ambiguous same-day boundary. Consequently proportions may be approximate.
    No future labels or event dates are consulted when selecting cutoffs.
    """

    config = {} if config is None else config
    fractions = _fractions(config)
    summary = _patient_summary(data, patient_col, label_col, index_date_col)
    summary[index_date_col] = pd.to_datetime(summary[index_date_col], errors="coerce")
    if summary[index_date_col].isna().any():
        raise ValueError(f"{index_date_col} contains invalid dates")
    date_counts = summary.groupby(index_date_col).size().sort_index()
    if len(date_counts) < 3:
        raise ValueError("Temporal splitting requires at least three distinct index dates")

    cumulative = date_counts.cumsum().to_numpy()
    dates = date_counts.index
    n_patients = len(summary)
    train_target = fractions[0] * n_patients
    validation_target = (fractions[0] + fractions[1]) * n_patients
    train_position = int(np.searchsorted(cumulative, train_target, side="left"))
    train_position = int(np.clip(train_position, 0, len(dates) - 3))
    validation_position = int(np.searchsorted(cumulative, validation_target, side="left"))
    validation_position = int(np.clip(validation_position, train_position + 1, len(dates) - 2))
    train_cutoff = dates[train_position]
    validation_cutoff = dates[validation_position]

    patient_dates = summary.set_index(patient_col)[index_date_col]
    assignments = pd.Series("test", index=patient_dates.index, dtype="string")
    assignments.loc[patient_dates <= validation_cutoff] = "validation"
    assignments.loc[patient_dates <= train_cutoff] = "train"
    result = {
        split_name: data.loc[data[patient_col].map(assignments).eq(split_name)]
        .copy()
        .reset_index(drop=True)
        for split_name in ("train", "validation", "test")
    }
    if any(frame.empty for frame in result.values()):
        raise ValueError("Temporal boundaries produced an empty partition")
    validate_patient_disjoint(result, patient_col)
    if result["train"][index_date_col].max() > result["validation"][index_date_col].min():
        raise AssertionError("Training dates extend into validation")
    if result["validation"][index_date_col].max() > result["test"][index_date_col].min():
        raise AssertionError("Validation dates extend into test")
    return result


# Concise aliases are useful in configuration-driven orchestration code.
stratified_split = stratified_patient_split
temporal_split = temporal_patient_split
