"""Structural safeguards supplement, but cannot replace, reviewed feature lineage."""

import re

import numpy as np
import pandas as pd

METADATA_COLUMNS = {
    "snapshot_id",
    "patient_id",
    "cohort_id",
    "start_dt",
    "index_date",
    "resp",
    "label",
    "outcome",
    "outcome_date",
    "feature_as_of",
    "feature_cutoff",
    "lookback_start",
    "prediction_end",
    "label_available_date",
    "eligible",
    "followup_complete",
    "covered_days",
    "maximum_gap_days",
    "split",
}
DEFAULT_FORBIDDEN_PATTERNS = (
    r"(^|_)(resp|label|outcome|target|future|post_index|split|partition|fold|cohort)($|_)",
    r"prediction_window",
    r"switched_after",
    r"(^|_)(patient|snapshot|hcp|provider)_id($|_)",
    r"(^|_)(index_date|start_dt|as_of|available_date|prediction_end)($|_)",
)


class LeakageError(AssertionError):
    pass


def validate_predictor_names(columns):
    bad = [
        c
        for c in columns
        if str(c).lower() in METADATA_COLUMNS
        or any(re.search(pattern, str(c), re.I) for pattern in DEFAULT_FORBIDDEN_PATTERNS)
    ]
    if bad:
        raise LeakageError(f"Forbidden target/identifier/split predictor columns: {bad}")


def find_post_index_events(
    events, cohort, *, date_col, patient_col="patient_id", index_date_col="index_date"
):
    key = "snapshot_id" if "snapshot_id" in events and "snapshot_id" in cohort else patient_col
    if cohort[key].duplicated().any():
        raise ValueError("Use snapshot_id to audit repeated patient snapshots")
    joined = events.drop(columns=[index_date_col], errors="ignore").merge(
        cohort[[key, index_date_col]], on=key, how="left", validate="many_to_one"
    )
    dates = pd.to_datetime(joined[date_col], errors="raise")
    indices = pd.to_datetime(joined[index_date_col], errors="raise")
    if dates.isna().any() or indices.isna().any():
        raise LeakageError("Unmatched or missing event/index dates")
    return joined.loc[dates > indices]


def assert_no_post_index_events(events, cohort, **kwargs):
    if not find_post_index_events(events, cohort, **kwargs).empty:
        raise LeakageError("Found post-index feature events")


def audit_feature_frame(features, cohort=None, *, raise_on_error=True, **kwargs):
    issues = []
    try:
        validate_predictor_names([c for c in features if c not in {"snapshot_id", "feature_as_of"}])
    except LeakageError as exc:
        issues.append({"check": "forbidden_feature_name", "detail": str(exc)})
    if "snapshot_id" not in features or features.snapshot_id.duplicated().any():
        issues.append(
            {"check": "duplicate_or_missing_snapshot", "detail": "Require unique snapshot_id"}
        )
    numeric = features.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        issues.append({"check": "infinite_value", "detail": "Numeric infinity"})
    report = pd.DataFrame(issues, columns=["check", "detail"])
    if raise_on_error and len(report):
        raise LeakageError(report.to_string(index=False))
    return report


def assert_no_feature_leakage(features, cohort=None, **kwargs):
    audit_feature_frame(features, cohort, **kwargs)


check_temporal_leakage = assert_no_post_index_events
