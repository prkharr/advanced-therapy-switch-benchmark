"""Eligible-patient cohort construction with strict temporal semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ._config import config_value, therapy_definition, timeline_days

REQUIRED_PATIENT_COLUMNS = {
    "patient_id",
    "observation_start",
    "observation_end",
}
REQUIRED_PHARMACY_COLUMNS = {
    "patient_id",
    "fill_date",
    "drug_id",
    "therapy_class",
}


def _require_columns(frame: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _resolve_index_dates(
    patients: pd.DataFrame,
    conventional_claims: pd.DataFrame,
    config: Any,
    observation_days: int,
    prediction_days: int,
) -> pd.Series:
    strategy = str(
        config_value(
            config,
            "index_date_strategy",
            "cohort.index_date_strategy",
            "timeline.index_date_strategy",
            default="auto",
        )
    ).lower()
    index_column = str(
        config_value(config, "index_date_column", "cohort.index_date_column", default="index_date")
    )
    scalar_index = config_value(config, "cohort.index_date", default=None)

    if scalar_index is not None:
        return pd.Series(pd.Timestamp(scalar_index), index=patients.index, dtype="datetime64[ns]")
    if (
        strategy
        in {
            "auto",
            "patient_column",
            "provided",
            "conventional_claim_anchor",
        }
        and index_column in patients.columns
    ):
        index_dates = pd.to_datetime(patients[index_column], errors="coerce")
        if strategy != "auto" and index_dates.isna().any():
            raise ValueError(f"Patient index-date column {index_column!r} contains invalid dates")
        return index_dates
    if strategy in {"patient_column", "provided"}:
        raise ValueError(f"Configured index-date column {index_column!r} was not found")

    supported = {
        "auto",
        "first_eligible_conventional_fill",
        "last_eligible_conventional_fill",
        "first_conventional_fill",
        "last_conventional_fill",
        "conventional_claim_anchor",
    }
    if strategy not in supported:
        raise ValueError(
            f"Unsupported index_date_strategy {strategy!r}; expected one of {sorted(supported)}"
        )

    eligible = conventional_claims.merge(
        patients[["patient_id", "observation_start", "observation_end"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    eligible["earliest_index"] = eligible["observation_start"] + pd.to_timedelta(
        observation_days, unit="D"
    )
    eligible["latest_index"] = eligible["observation_end"] - pd.to_timedelta(
        prediction_days, unit="D"
    )
    eligible = eligible.loc[
        eligible["fill_date"].between(eligible["earliest_index"], eligible["latest_index"])
    ]
    configured_start = config_value(config, "timeline.index_date_start", default=None)
    configured_end = config_value(config, "timeline.index_date_end", default=None)
    if configured_start is not None:
        eligible = eligible.loc[eligible["fill_date"] >= pd.Timestamp(configured_start)]
    if configured_end is not None:
        eligible = eligible.loc[eligible["fill_date"] <= pd.Timestamp(configured_end)]
    choose_first = strategy in {
        "auto",
        "first_eligible_conventional_fill",
        "first_conventional_fill",
    }
    aggregate = "min" if choose_first else "max"
    by_patient = eligible.groupby("patient_id", sort=False)["fill_date"].agg(aggregate)
    return patients["patient_id"].map(by_patient)


def build_cohort(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any] | Any | None = None
) -> pd.DataFrame:
    """Build a leakage-safe conventional-therapy cohort and future outcome.

    Eligibility requires a complete lookback and prediction window, at least one
    conventional-therapy claim during lookback (configurable), and no advanced
    therapy on or before the index date. The outcome is an advanced-therapy claim
    strictly after index and on or before ``index + prediction_window_days``.

    The default ``auto`` index strategy uses ``patients.index_date`` when present
    (as in the synthetic data). Otherwise it uses the first conventional claim
    that permits complete lookback and follow-up. Strategies and therapy mappings
    are configuration-driven.

    Returns a frame with at least ``patient_id``, ``index_date``, ``outcome``, and
    ``label``. ``outcome_date`` is included for auditability but must never be a
    model feature.
    """

    config = {} if config is None else config
    if "patients" not in tables or "pharmacy_claims" not in tables:
        raise ValueError("tables must contain 'patients' and 'pharmacy_claims'")
    patients = tables["patients"].copy()
    pharmacy = tables["pharmacy_claims"].copy()
    _require_columns(patients, REQUIRED_PATIENT_COLUMNS, "patients")
    _require_columns(pharmacy, REQUIRED_PHARMACY_COLUMNS, "pharmacy_claims")
    if patients["patient_id"].duplicated().any():
        raise ValueError("patients.patient_id must be unique")

    patients["observation_start"] = pd.to_datetime(patients["observation_start"], errors="coerce")
    patients["observation_end"] = pd.to_datetime(patients["observation_end"], errors="coerce")
    pharmacy["fill_date"] = pd.to_datetime(pharmacy["fill_date"], errors="coerce")
    if patients[["observation_start", "observation_end"]].isna().any().any():
        raise ValueError("Patient observation bounds contain invalid dates")
    if pharmacy["fill_date"].isna().any():
        raise ValueError("pharmacy_claims.fill_date contains invalid dates")

    therapy = therapy_definition(config)
    conventional_mask = therapy.conventional_mask(pharmacy)
    advanced_mask = therapy.advanced_mask(pharmacy)
    if isinstance(conventional_mask, bool) or not conventional_mask.any():
        raise ValueError(
            "No conventional claims matched the configured therapy mapping; "
            "supply drug_ids and/or therapy_classes"
        )
    conventional = pharmacy.loc[conventional_mask].copy()
    advanced = (
        pharmacy.loc[advanced_mask].copy()
        if not isinstance(advanced_mask, bool)
        else pharmacy.iloc[0:0].copy()
    )

    observation_days, prediction_days = timeline_days(config)
    patients["index_date"] = _resolve_index_dates(
        patients, conventional, config, observation_days, prediction_days
    )
    candidate = patients.dropna(subset=["index_date"]).copy()
    candidate["index_date"] = pd.to_datetime(candidate["index_date"])
    candidate["lookback_start"] = candidate["index_date"] - pd.to_timedelta(
        observation_days, unit="D"
    )
    candidate["prediction_end"] = candidate["index_date"] + pd.to_timedelta(
        prediction_days, unit="D"
    )
    candidate = candidate.loc[
        (candidate["observation_start"] <= candidate["lookback_start"])
        & (candidate["observation_end"] >= candidate["prediction_end"])
    ].copy()

    minimum_conventional_claims = int(
        config_value(
            config,
            "minimum_conventional_claims",
            "cohort.minimum_conventional_claims",
            default=1,
        )
    )
    if minimum_conventional_claims < 0:
        raise ValueError("minimum_conventional_claims cannot be negative")
    require_conventional = bool(
        config_value(
            config,
            "require_conventional_exposure",
            "cohort.require_conventional_exposure",
            default=True,
        )
    )
    if require_conventional and minimum_conventional_claims == 0:
        minimum_conventional_claims = 1

    conventional_history = conventional.merge(
        candidate[["patient_id", "index_date", "lookback_start"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    conventional_history = conventional_history.loc[
        conventional_history["fill_date"].between(
            conventional_history["lookback_start"], conventional_history["index_date"]
        )
    ]
    conventional_counts = conventional_history.groupby("patient_id").size()
    if minimum_conventional_claims > 0:
        eligible_ids = conventional_counts.loc[
            conventional_counts >= minimum_conventional_claims
        ].index
        candidate = candidate.loc[candidate["patient_id"].isin(eligible_ids)].copy()

    require_on_index = bool(
        config_value(
            config,
            "require_conventional_on_index",
            "cohort.require_conventional_on_index",
            default=False,
        )
    )
    if require_on_index:
        index_claims = conventional.merge(
            candidate[["patient_id", "index_date"]],
            on="patient_id",
            how="inner",
            validate="many_to_one",
        )
        index_claim_ids = index_claims.loc[
            index_claims["fill_date"].eq(index_claims["index_date"]), "patient_id"
        ].unique()
        candidate = candidate.loc[candidate["patient_id"].isin(index_claim_ids)].copy()

    advanced_with_landmark = advanced.merge(
        candidate[["patient_id", "index_date", "prediction_end"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    prior_advanced_ids = advanced_with_landmark.loc[
        advanced_with_landmark["fill_date"] <= advanced_with_landmark["index_date"],
        "patient_id",
    ].unique()
    candidate = candidate.loc[~candidate["patient_id"].isin(prior_advanced_ids)].copy()

    advanced_with_landmark = advanced.merge(
        candidate[["patient_id", "index_date", "prediction_end"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    future_outcomes = advanced_with_landmark.loc[
        (advanced_with_landmark["fill_date"] > advanced_with_landmark["index_date"])
        & (advanced_with_landmark["fill_date"] <= advanced_with_landmark["prediction_end"])
    ]
    first_outcome = future_outcomes.groupby("patient_id")["fill_date"].min()

    result = candidate[["patient_id", "index_date"]].copy()
    result["outcome_date"] = result["patient_id"].map(first_outcome)
    result["label"] = result["outcome_date"].notna().astype("int8")
    result["outcome"] = result["label"]
    result = result[["patient_id", "index_date", "outcome", "label", "outcome_date"]]
    result.sort_values(["index_date", "patient_id"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def validate_cohort_timeline(
    cohort: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any] | Any | None = None,
) -> None:
    """Raise ``AssertionError`` when cohort labels violate temporal definitions."""

    rebuilt = build_cohort(tables, config)
    expected = rebuilt.set_index("patient_id")[["index_date", "label", "outcome_date"]]
    actual = cohort.set_index("patient_id")[["index_date", "label", "outcome_date"]]
    pd.testing.assert_frame_equal(
        actual.sort_index(), expected.loc[actual.index].sort_index(), check_dtype=False
    )
