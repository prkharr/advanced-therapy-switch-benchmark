"""Automated checks for common temporal and target-leakage failures."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    """Raised when data supplied to a feature/model stage leaks future outcome."""


DEFAULT_FORBIDDEN_PATTERNS = (
    r"(^|_)outcome($|_)",
    r"(^|_)outcome_date($|_)",
    r"(^|_)target($|_)",
    r"(^|_)future($|_)",
    r"(^|_)post_index($|_)",
    r"prediction_window",
    r"future_advanced",
    r"switched_after",
)


def find_post_index_events(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    date_col: str,
    patient_col: str = "patient_id",
    index_date_col: str = "index_date",
) -> pd.DataFrame:
    """Return event rows occurring strictly after each patient's index date."""

    required_events = {patient_col, date_col}
    required_cohort = {patient_col, index_date_col}
    if not required_events.issubset(events.columns):
        raise ValueError(f"Events missing columns: {sorted(required_events - set(events.columns))}")
    if not required_cohort.issubset(cohort.columns):
        raise ValueError(f"Cohort missing columns: {sorted(required_cohort - set(cohort.columns))}")
    landmarks = cohort[[patient_col, index_date_col]].copy()
    if landmarks[patient_col].duplicated().any():
        raise ValueError("Cohort must contain one index date per patient")
    landmarks[index_date_col] = pd.to_datetime(landmarks[index_date_col], errors="coerce")
    tagged = events.drop(columns=[index_date_col], errors="ignore").copy()
    tagged[date_col] = pd.to_datetime(tagged[date_col], errors="coerce")
    if tagged[date_col].isna().any() and events[date_col].notna().any():
        raise ValueError(f"{date_col} contains invalid dates")
    tagged = tagged.merge(landmarks, on=patient_col, how="inner", validate="many_to_one")
    return tagged.loc[tagged[date_col] > tagged[index_date_col]].copy()


def assert_no_post_index_events(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    date_col: str,
    patient_col: str = "patient_id",
    index_date_col: str = "index_date",
) -> None:
    """Raise :class:`LeakageError` if a feature-source event is post-index."""

    leaked = find_post_index_events(
        events,
        cohort,
        date_col=date_col,
        patient_col=patient_col,
        index_date_col=index_date_col,
    )
    if not leaked.empty:
        preview_columns = [patient_col, date_col, index_date_col]
        preview = leaked[preview_columns].head(5).to_dict("records")
        raise LeakageError(f"Found {len(leaked)} post-index feature events; examples: {preview}")


def audit_feature_frame(
    features: pd.DataFrame,
    cohort: pd.DataFrame | None = None,
    *,
    patient_col: str = "patient_id",
    index_date_col: str = "index_date",
    label_col: str = "label",
    forbidden_patterns: Iterable[str] = DEFAULT_FORBIDDEN_PATTERNS,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Audit a model-ready feature frame and return a table of issues.

    This structural check catches future/outcome-named columns, duplicate patient
    snapshots, unexpected labels, and infinite numeric values. It complements
    :func:`assert_no_post_index_events`, which should be applied to event frames
    after temporal slicing.
    """

    issues: list[dict[str, str]] = []
    allowed = {patient_col, index_date_col, label_col}
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in forbidden_patterns]
    for column in features.columns:
        if column in allowed:
            continue
        for pattern in compiled:
            if pattern.search(str(column)):
                issues.append(
                    {
                        "check": "forbidden_feature_name",
                        "column": str(column),
                        "detail": f"matched {pattern.pattern!r}",
                    }
                )
                break
    if patient_col not in features:
        issues.append(
            {"check": "missing_patient_id", "column": patient_col, "detail": "column absent"}
        )
    elif features[patient_col].duplicated().any():
        issues.append(
            {
                "check": "duplicate_snapshot",
                "column": patient_col,
                "detail": f"{int(features[patient_col].duplicated().sum())} duplicate rows",
            }
        )

    numeric = features.select_dtypes(include=[np.number])
    for column in numeric.columns:
        if np.isinf(numeric[column].to_numpy(dtype=float, na_value=np.nan)).any():
            issues.append(
                {"check": "infinite_value", "column": column, "detail": "contains +/- infinity"}
            )

    if cohort is not None and patient_col in features and patient_col in cohort:
        if label_col not in features or label_col not in cohort:
            issues.append(
                {
                    "check": "missing_label",
                    "column": label_col,
                    "detail": "label absent from features or cohort",
                }
            )
            report = pd.DataFrame(issues, columns=["check", "column", "detail"])
            if raise_on_error:
                raise LeakageError("missing_label:label (label absent from features or cohort)")
            return report
        expected = cohort[[patient_col, label_col]].drop_duplicates(patient_col)
        actual = features[[patient_col, label_col]].merge(
            expected,
            on=patient_col,
            how="left",
            suffixes=("_feature", "_cohort"),
            validate="one_to_one",
        )
        missing = actual[f"{label_col}_cohort"].isna()
        if missing.any():
            issues.append(
                {
                    "check": "unknown_patient",
                    "column": patient_col,
                    "detail": f"{int(missing.sum())} feature patients absent from cohort",
                }
            )
        mismatch = (actual[f"{label_col}_feature"] != actual[f"{label_col}_cohort"]) & ~missing
        if mismatch.any():
            issues.append(
                {
                    "check": "label_mismatch",
                    "column": label_col,
                    "detail": f"{int(mismatch.sum())} labels differ from cohort",
                }
            )
    report = pd.DataFrame(issues, columns=["check", "column", "detail"])
    if raise_on_error and not report.empty:
        rendered = "; ".join(
            f"{row.check}:{row.column} ({row.detail})" for row in report.itertuples()
        )
        raise LeakageError(rendered)
    return report


def assert_no_feature_leakage(
    features: pd.DataFrame, cohort: pd.DataFrame | None = None, **kwargs: object
) -> None:
    """Compatibility wrapper that raises when :func:`audit_feature_frame` fails."""

    audit_feature_frame(features, cohort, raise_on_error=True, **kwargs)


check_temporal_leakage = assert_no_post_index_events
