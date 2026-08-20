"""Leakage-safe tabular feature engineering for longitudinal claims."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from therapy_switch.data._config import (
    as_tuple,
    config_value,
    therapy_definition,
    timeline_days,
)

from .leakage import audit_feature_frame

DEFAULT_WINDOWS = (30, 60, 90, 180, 365)


def _required(frame: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _safe_feature_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "unknown"


def _count_within(days_before: pd.Series, window: int) -> int:
    return int((days_before <= window).sum())


def _unique_within(frame: pd.DataFrame, days_column: str, value_column: str, window: int) -> int:
    if value_column not in frame:
        return 0
    return int(frame.loc[frame[days_column] <= window, value_column].dropna().nunique())


def _smoothed_ratio(numerator: float, denominator: float, epsilon: float = 1.0) -> float:
    """A finite rate ratio whose neutral value is one when both counts are zero."""

    return float((numerator + epsilon) / (denominator + epsilon))


def _days_since(dates: pd.Series, index_date: pd.Timestamp, censor_days: int) -> int:
    if dates.empty:
        return censor_days + 1
    return int((index_date - dates.max()).days)


def _treatment_change_dates(pharmacy: pd.DataFrame) -> pd.Series:
    if pharmacy.empty or "drug_id" not in pharmacy:
        return pd.Series([], dtype="datetime64[ns]")
    ordered = pharmacy.sort_values(
        ["fill_date", "claim_id"] if "claim_id" in pharmacy else ["fill_date"]
    )
    changes = ordered["drug_id"].astype("string").ne(ordered["drug_id"].astype("string").shift())
    if len(changes):
        changes.iloc[0] = False
    return ordered.loc[changes, "fill_date"]


def _proportion_days_covered(
    pharmacy: pd.DataFrame, history_start: pd.Timestamp, index_date: pd.Timestamp
) -> float:
    if pharmacy.empty or "days_supply" not in pharmacy:
        return 0.0
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for fill_date, days_supply in pharmacy[["fill_date", "days_supply"]].itertuples(index=False):
        if pd.isna(days_supply) or float(days_supply) <= 0:
            continue
        start = max(pd.Timestamp(fill_date), history_start)
        end = min(
            pd.Timestamp(fill_date) + pd.Timedelta(days=float(days_supply)),
            index_date + pd.Timedelta(days=1),
        )
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort(key=lambda item: item[0])
    merged: list[list[pd.Timestamp]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    covered = sum((end - start).days for start, end in merged)
    denominator = max(1, (index_date - history_start).days + 1)
    return float(min(1.0, covered / denominator))


def _history_by_patient(
    frame: pd.DataFrame,
    date_column: str,
    cohort: pd.DataFrame,
    observation_days: int,
) -> dict[Any, pd.DataFrame]:
    if frame.empty:
        return {}
    copy = frame.copy()
    copy[date_column] = pd.to_datetime(copy[date_column], errors="coerce")
    if copy[date_column].isna().any():
        raise ValueError(f"{date_column} contains invalid dates")
    landmarks = cohort[["patient_id", "index_date"]].copy()
    landmarks["index_date"] = pd.to_datetime(landmarks["index_date"])
    joined = copy.merge(landmarks, on="patient_id", how="inner", validate="many_to_one")
    joined["days_before_index"] = (joined["index_date"] - joined[date_column]).dt.days
    joined = joined.loc[joined["days_before_index"].between(0, observation_days)].copy()
    return {
        patient_id: group.drop(columns=["index_date"]).copy()
        for patient_id, group in joined.groupby("patient_id", sort=False)
    }


def build_tabular_features(
    tables: Mapping[str, pd.DataFrame],
    cohort: pd.DataFrame,
    config: Mapping[str, Any] | Any | None = None,
) -> pd.DataFrame:
    """Build one leakage-safe feature row per cohort patient.

    Medical and pharmacy events are independently restricted to the inclusive
    observation interval ``[index - observation_window_days, index]``. Default
    rolling windows are 30/60/90/180/365 days. Categorical fields are retained as
    strings for downstream preprocessing pipelines; numeric missing-value
    imputation is intentionally left to model pipelines.
    """

    config = {} if config is None else config
    required_tables = {"patients", "medical_claims", "pharmacy_claims"}
    missing_tables = sorted(required_tables - set(tables))
    if missing_tables:
        raise ValueError(f"Missing tables: {missing_tables}")
    patients = tables["patients"].copy()
    medical = tables["medical_claims"].copy()
    pharmacy = tables["pharmacy_claims"].copy()
    providers = tables.get("providers", pd.DataFrame()).copy()
    _required(patients, {"patient_id"}, "patients")
    _required(
        medical,
        {"patient_id", "claim_date", "diagnosis_code", "procedure_code", "place_of_service"},
        "medical_claims",
    )
    _required(
        pharmacy,
        {"patient_id", "fill_date", "drug_id", "therapy_class", "days_supply"},
        "pharmacy_claims",
    )
    _required(cohort, {"patient_id", "index_date", "label"}, "cohort")
    if patients["patient_id"].duplicated().any() or cohort["patient_id"].duplicated().any():
        raise ValueError("Patients and cohort must each contain one row per patient")

    observation_days, _ = timeline_days(config)
    configured_windows = config_value(
        config,
        "feature_windows_days",
        "features.windows_days",
        "features.rolling_windows_days",
        default=DEFAULT_WINDOWS,
    )
    windows = tuple(sorted({int(window) for window in configured_windows}))
    if any(window <= 0 or window > observation_days for window in windows):
        raise ValueError("Feature windows must be positive and no longer than observation window")
    # The required business windows are present whenever they fit the configured
    # lookback, even if callers add custom windows.
    windows = tuple(sorted(set(windows) | {w for w in DEFAULT_WINDOWS if w <= observation_days}))

    patient_lookup = patients.set_index("patient_id", drop=False)
    medical_history = _history_by_patient(medical, "claim_date", cohort, observation_days)
    pharmacy_history = _history_by_patient(pharmacy, "fill_date", cohort, observation_days)

    provider_specialty: dict[Any, str] = {}
    if not providers.empty and {"provider_id", "specialty"}.issubset(providers.columns):
        if providers["provider_id"].duplicated().any():
            raise ValueError("providers.provider_id must be unique")
        provider_specialty = (
            providers.set_index("provider_id")["specialty"].astype("string").to_dict()
        )

    configured_specialties = as_tuple(
        config_value(
            config, "specialist_specialties", "features.specialist_specialties", default=None
        )
    )
    general_specialties = {
        value.lower()
        for value in as_tuple(
            config_value(
                config,
                "features.generalist_specialties",
                default=("primary_care", "emergency_medicine", "hospitalist", "general_practice"),
            )
        )
    }
    disease_codes = as_tuple(
        config_value(
            config,
            "disease_indicator_codes",
            "features.disease_indicator_codes",
            default=("DX_DISEASE", "DX_SEVERE"),
        )
    )
    severity_codes = set(
        as_tuple(
            config_value(
                config,
                "severity_diagnosis_codes",
                "features.severity_diagnosis_codes",
                default=("DX_SEVERE",),
            )
        )
    )
    configured_comorbidities = as_tuple(
        config_value(config, "comorbidity_codes", "features.comorbidity_codes", default=None)
    )
    discontinuation_days = int(
        config_value(
            config, "discontinuation_gap_days", "features.discontinuation_gap_days", default=60
        )
    )
    ratio_epsilon = float(
        config_value(config, "safe_ratio_epsilon", "features.safe_ratio_epsilon", default=1.0)
    )
    if ratio_epsilon <= 0:
        raise ValueError("safe_ratio_epsilon must be positive")
    include_geography = bool(
        config_value(config, "include_geography", "features.include_geography", default=True)
    )
    missing_recency_days = int(
        config_value(
            config,
            "missing_recency_value_days",
            "features.missing_recency_value_days",
            default=observation_days + 1,
        )
    )
    therapy = therapy_definition(config)

    rows: list[dict[str, Any]] = []
    empty_medical = medical.iloc[0:0].assign(days_before_index=pd.Series(dtype="int64"))
    empty_pharmacy = pharmacy.iloc[0:0].assign(days_before_index=pd.Series(dtype="int64"))
    for cohort_row in cohort[["patient_id", "index_date", "label"]].itertuples(index=False):
        patient_id = cohort_row.patient_id
        if patient_id not in patient_lookup.index:
            raise ValueError(f"Cohort patient {patient_id!r} is absent from patients table")
        index_date = pd.Timestamp(cohort_row.index_date)
        history_start = index_date - pd.Timedelta(days=observation_days)
        patient = patient_lookup.loc[patient_id]
        patient_medical = medical_history.get(patient_id, empty_medical).copy()
        patient_pharmacy = pharmacy_history.get(patient_id, empty_pharmacy).copy()

        if "provider_id" in patient_medical:
            patient_medical["provider_specialty"] = (
                patient_medical["provider_id"]
                .map(provider_specialty)
                .fillna("unknown")
                .astype("string")
            )
        else:
            patient_medical["provider_specialty"] = "unknown"
        if configured_specialties:
            specialist_mask = patient_medical["provider_specialty"].isin(configured_specialties)
        else:
            specialist_mask = ~patient_medical["provider_specialty"].str.lower().isin(
                general_specialties | {"unknown"}
            )
        patient_medical["is_specialist"] = specialist_mask.astype(bool)

        conventional_mask = therapy.conventional_mask(patient_pharmacy)
        conventional_rx = (
            patient_pharmacy.loc[conventional_mask].copy()
            if not isinstance(conventional_mask, bool)
            else patient_pharmacy.iloc[0:0].copy()
        )
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "index_date": index_date,
            "label": int(cohort_row.label),
            "gender": str(patient.get("gender", "unknown")),
        }
        if include_geography:
            row["geography"] = str(patient.get("geography", "unknown"))
        birth_year = patient.get("birth_year", np.nan)
        if pd.notna(birth_year):
            row["age"] = int(index_date.year - int(birth_year))
        else:
            row["age"] = float(patient.get("age", np.nan))

        medical_days = patient_medical["days_before_index"]
        pharmacy_days = patient_pharmacy["days_before_index"]
        for window in windows:
            med_window = patient_medical.loc[medical_days <= window]
            rx_window = patient_pharmacy.loc[pharmacy_days <= window]
            row[f"diagnosis_count_{window}d"] = int(med_window["diagnosis_code"].notna().sum())
            row[f"unique_diagnosis_count_{window}d"] = int(
                med_window["diagnosis_code"].dropna().nunique()
            )
            row[f"procedure_count_{window}d"] = int(med_window["procedure_code"].notna().sum())
            row[f"visits_{window}d"] = int(len(med_window))
            place = med_window["place_of_service"].astype("string").str.lower()
            row[f"outpatient_visits_{window}d"] = int(place.isin(["outpatient", "office"]).sum())
            row[f"inpatient_visits_{window}d"] = int(place.eq("inpatient").sum())
            row[f"er_visits_{window}d"] = int(place.isin(["er", "emergency"]).sum())
            row[f"specialist_visits_{window}d"] = int(med_window["is_specialist"].sum())
            row[f"rx_claims_{window}d"] = int(len(rx_window))
            row[f"unique_therapies_{window}d"] = int(rx_window["drug_id"].dropna().nunique())
            row[f"refill_frequency_{window}d"] = float(len(rx_window) * 30.0 / window)

        diagnoses = patient_medical["diagnosis_code"].dropna().astype("string")
        row["diagnosis_count"] = int(len(diagnoses))
        row["unique_diagnosis_count"] = int(diagnoses.nunique())
        if configured_comorbidities:
            comorbidity_mask = diagnoses.isin(configured_comorbidities)
        else:
            comorbidity_mask = diagnoses.str.startswith("DX_COMORBID")
        row["comorbidity_count"] = int(diagnoses.loc[comorbidity_mask].nunique())
        row["severity_proxy_count"] = int(diagnoses.isin(severity_codes).sum())
        for diagnosis in disease_codes:
            row[f"has_diagnosis_{_safe_feature_name(diagnosis)}"] = int(
                diagnoses.eq(diagnosis).any()
            )

        ordered_rx = conventional_rx.sort_values(
            ["fill_date", "claim_id"] if "claim_id" in conventional_rx else ["fill_date"]
        )
        row["current_therapy"] = (
            str(ordered_rx.iloc[-1]["drug_id"]) if not ordered_rx.empty else "unknown"
        )
        unique_therapy_count = int(ordered_rx["drug_id"].dropna().nunique())
        row["number_previous_therapies"] = max(0, unique_therapy_count - 1)
        change_dates = _treatment_change_dates(ordered_rx)
        row["number_treatment_changes"] = int(len(change_dates))
        row["therapy_duration_days"] = (
            int((index_date - ordered_rx["fill_date"].min()).days) + 1
            if not ordered_rx.empty
            else 0
        )
        row["proportion_days_covered"] = _proportion_days_covered(
            ordered_rx, history_start, index_date
        )

        diagnosis_dates = patient_medical.loc[
            patient_medical["diagnosis_code"].notna(), "claim_date"
        ]
        rx_dates = patient_pharmacy["fill_date"]
        specialist_dates = patient_medical.loc[patient_medical["is_specialist"], "claim_date"]
        row["days_since_last_diagnosis"] = _days_since(
            diagnosis_dates, index_date, missing_recency_days - 1
        )
        row["days_since_last_rx"] = _days_since(rx_dates, index_date, missing_recency_days - 1)
        row["days_since_last_specialist_visit"] = _days_since(
            specialist_dates, index_date, missing_recency_days - 1
        )
        row["days_since_last_treatment_change"] = _days_since(
            change_dates, index_date, missing_recency_days - 1
        )
        row["discontinuation_proxy"] = int(row["days_since_last_rx"] > discontinuation_days)

        visits_recent = int((medical_days <= 30).sum())
        visits_previous = int(medical_days.between(31, 60).sum())
        row["visits_last_30d_over_previous_30d"] = _smoothed_ratio(
            visits_recent, visits_previous, ratio_epsilon
        )
        diagnosis_rows = patient_medical["diagnosis_code"].notna()
        diagnoses_recent = int((diagnosis_rows & (medical_days <= 90)).sum())
        diagnoses_previous = int((diagnosis_rows & medical_days.between(91, 180)).sum())
        row["diagnoses_last_90d_minus_previous_90d"] = diagnoses_recent - diagnoses_previous
        rx_recent = int((pharmacy_days <= min(90, observation_days)).sum())
        historical_span = max(1, observation_days - min(90, observation_days))
        rx_historical = int((pharmacy_days > min(90, observation_days)).sum())
        recent_rate = rx_recent / max(1, min(90, observation_days))
        historical_rate = rx_historical / historical_span
        row["rx_frequency_recent_vs_historical"] = float(
            (recent_rate + 1.0 / 90.0) / (historical_rate + 1.0 / historical_span)
        )
        specialist_recent = int((patient_medical["is_specialist"] & (medical_days <= 90)).sum())
        specialist_previous = int(
            (patient_medical["is_specialist"] & medical_days.between(91, 180)).sum()
        )
        row["specialist_visit_acceleration"] = specialist_recent - specialist_previous
        visits_last_90 = int((medical_days <= 90).sum())
        visits_previous_90 = int(medical_days.between(91, 180).sum())
        row["utilization_trend"] = float((visits_last_90 - visits_previous_90) / 3.0)
        rows.append(row)

    result = pd.DataFrame.from_records(rows)
    result.sort_values("patient_id", inplace=True)
    result.reset_index(drop=True, inplace=True)
    result.attrs["observation_window_days"] = observation_days
    result.attrs["feature_windows_days"] = windows
    audit_feature_frame(result, cohort, raise_on_error=True)
    return result


# A compact alias for callers that use a generic feature-builder convention.
build_features = build_tabular_features
