"""Strict patient-disjoint temporal evaluation with label maturity purging."""

import numpy as np
import pandas as pd

from ._config import config_value


def validate_patient_disjoint(splits, patient_col="patient_id"):
    seen = set()
    for name, frame in splits.items():
        ids = set(frame[patient_col])
        if seen & ids:
            raise AssertionError(f"Patient overlap in {name}")
        seen |= ids


def _fractions(config):
    val = float(
        config_value(config, "validation_fraction", "splitting.validation_fraction", default=0.2)
    )
    test = float(config_value(config, "test_fraction", "splitting.test_fraction", default=0.2))
    if min(val, test) <= 0 or val + test >= 1:
        raise ValueError("Invalid split fractions")
    return 1 - val - test, val, test


def stratified_patient_split(data, config=None, *, patient_col="patient_id", label_col="label"):
    config = config or {}
    train, val, _ = _fractions(config)
    rng = np.random.default_rng(
        int(config_value(config, "random_seed", "project.random_seed", default=42))
    )
    summary = data.groupby(patient_col)[label_col].max()
    assigned = {}
    for _, group in summary.groupby(summary):
        ids = rng.permutation(group.index)
        a, b = int(len(ids) * train), int(len(ids) * (train + val))
        for name, values in [("train", ids[:a]), ("validation", ids[a:b]), ("test", ids[b:])]:
            assigned.update(dict.fromkeys(values, name))
    result = {
        name: data.loc[data[patient_col].map(assigned).eq(name)].copy().reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    if any(frame.empty for frame in result.values()):
        raise ValueError("Empty split; increase patient population")
    validate_patient_disjoint(result, patient_col)
    return result


def temporal_patient_split(
    data, config=None, *, patient_col="patient_id", label_col="label", index_date_col="index_date"
):
    config = config or {}
    train, val, _ = _fractions(config)
    if data.empty or data[patient_col].isna().any():
        raise ValueError("Cannot split empty or unkeyed data")
    if "label_available_date" not in data:
        raise ValueError("Temporal split requires label_available_date")
    summary = data.groupby(patient_col)[index_date_col].min().sort_values()
    dates = pd.to_datetime(summary)
    train_cutoff = pd.Timestamp(
        config_value(
            config, "splitting.train_end", default=dates.iloc[int((len(dates) - 1) * train)]
        )
    )
    val_cutoff = pd.Timestamp(
        config_value(
            config,
            "splitting.validation_end",
            default=dates.iloc[int((len(dates) - 1) * (train + val))],
        )
    )
    gap = pd.Timedelta(days=int(config_value(config, "splitting.temporal_gap_days", default=0)))
    if val_cutoff <= train_cutoff or gap.days < 0:
        raise ValueError("Invalid temporal boundaries or gap")
    assignments = pd.Series("test", index=dates.index)
    assignments.loc[dates <= val_cutoff] = "validation"
    assignments.loc[dates <= train_cutoff] = "train"
    result = {}
    for name in ("train", "validation", "test"):
        frame = data.loc[data[patient_col].map(assignments).eq(name)].copy()
        if name == "train":
            frame = frame.loc[frame.label_available_date <= train_cutoff]
        elif name == "validation":
            frame = frame.loc[
                (frame[index_date_col] > train_cutoff + gap)
                & (frame.label_available_date <= val_cutoff)
            ]
        else:
            frame = frame.loc[frame[index_date_col] > val_cutoff + gap]
        result[name] = frame.sort_values([index_date_col, patient_col]).reset_index(drop=True)
    if any(frame.empty for frame in result.values()):
        raise ValueError("Maturity purging left an empty split; widen temporal periods")
    validate_patient_disjoint(result, patient_col)
    if result["train"].label_available_date.max() >= result["validation"][index_date_col].min():
        raise AssertionError("Training labels unavailable at validation scoring start")
    if result["validation"].label_available_date.max() >= result["test"][index_date_col].min():
        raise AssertionError("Validation labels unavailable at test scoring start")
    return result


def split_manifest(data, splits):
    selected = {}
    for name, frame in splits.items():
        selected.update(dict.fromkeys(frame.snapshot_id, name))
    out = data[["snapshot_id", "patient_id", "index_date", "label_available_date", "label"]].copy()
    out["split"] = out.snapshot_id.map(selected).fillna("purged")
    out["reason"] = np.where(
        out.split.eq("purged"), "patient assignment, temporal gap or label maturity", "retained"
    )
    return out


stratified_split = stratified_patient_split
temporal_split = temporal_patient_split
