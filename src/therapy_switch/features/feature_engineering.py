"""Point-in-time wide features linked one-to-one to eligible snapshots."""

from __future__ import annotations

import pandas as pd

from therapy_switch.data._config import therapy_definition
from therapy_switch.data.availability import (
    coverage_days,
    known_claims,
    maximum_gap,
    valid_exposures,
)
from therapy_switch.data.event_sequences import build_event_frame


def build_tabular_features(tables, cohort, config=None):
    config = config or {}
    settings = config.get("features", {})
    windows = settings.get("rolling_windows_days", [30, 60, 90, 180, 365])
    history = int(config.get("timeline", {}).get("observation_window_days", 365))
    lag = int(config.get("timeline", {}).get("claims_lag_days", 0))
    if any(w <= 0 or w > history for w in windows):
        raise ValueError("Feature windows must be positive and fit the observation window")
    events = build_event_frame(tables, cohort, config)
    event_groups = dict(tuple(events.groupby("snapshot_id", sort=False)))
    rx_groups = dict(tuple(tables["pharmacy_claims"].groupby("patient_id", sort=False)))
    patients = tables["patients"].set_index("patient_id")
    enroll_groups = dict(tuple(tables["enrollment"].groupby("patient_id", sort=False)))
    plan_map = tables["plans"].set_index("plan_id").payer_type
    therapy = therapy_definition(config)
    all_rx = tables["pharmacy_claims"]
    # This history contains claim records, never snapshot labels. Every lookup
    # below applies the focal snapshot's availability cutoff and excludes itself.
    hcp_groups = dict(tuple(all_rx.groupby("prescriber_id", sort=False)))
    rows = []
    for snap in cohort.itertuples(index=False):
        pid, index, start = snap.patient_id, snap.index_date, snap.lookback_start
        patient = patients.loc[pid]
        ev = event_groups.get(snap.snapshot_id, events.iloc[:0])
        raw_rx = rx_groups.get(pid, all_rx.iloc[:0])
        available_rx = known_claims(raw_rx, "fill_date", index, lag)
        paid = valid_exposures(available_rx)
        conventional = paid.loc[therapy.conventional_mask(paid)].sort_values(
            ["fill_date", "claim_id"]
        )
        covered = coverage_days(conventional, start, index)
        recent_conv = conventional.loc[conventional.fill_date.ge(start)]
        changes = recent_conv.drug_id.ne(recent_conv.drug_id.shift())
        if len(changes):
            changes.iloc[0] = False
        change_dates = recent_conv.loc[changes, "fill_date"]
        valid = ev.loc[ev.status.isin(["paid", "final"])]
        diagnosis = valid.loc[valid.event_type.str.startswith("diagnosis")]
        specialist = diagnosis.loc[
            diagnosis.provider_specialty.isin(
                settings.get(
                    "specialist_specialties",
                    ["sleep_medicine", "neurology", "pulmonology", "psychiatry"],
                )
            )
        ]
        row = {
            "snapshot_id": snap.snapshot_id,
            "feature_as_of": index,
            "age": index.year - int(patient.birth_year),
            "gender": patient.gender,
            "current_therapy": conventional.iloc[-1].drug_id if len(conventional) else "unknown",
            "proportion_days_covered": float(covered.mean()),
            "maximum_therapy_gap_days": maximum_gap(covered),
            "therapy_changes": int(changes.sum()),
            "therapies_tried": int(recent_conv.drug_id.nunique()),
            "therapy_duration_days": int((index - conventional.fill_date.min()).days)
            if len(conventional)
            else 0,
        }
        if settings.get("include_geography", True):
            row["geography"] = patient.geography
        enrollment = enroll_groups.get(pid, tables["enrollment"].iloc[:0])
        enrollment = enrollment.loc[
            enrollment.coverage_start.le(index) & enrollment.coverage_end.ge(index)
        ]
        if enrollment.plan_id.nunique() > 1:
            raise ValueError("Concurrent plans require an approved primary-plan mapping")
        row["payer_type"] = (
            plan_map.get(enrollment.iloc[0].plan_id, "unknown") if len(enrollment) else "unknown"
        )
        for window in windows:
            period = ev.loc[ev.days_before_index <= window]
            for kind in ("diagnosis", "procedure", "pharmacy"):
                row[f"{kind}_count_{window}d"] = int(period.event_type.str.startswith(kind).sum())
            row[f"specialist_visits_{window}d"] = int(
                (specialist.days_before_index <= window).sum()
            )
            acute = diagnosis.loc[diagnosis.days_before_index.le(window)]
            row[f"emergency_visits_{window}d"] = int(acute.place_of_service.eq("emergency").sum())
            row[f"urgent_visits_{window}d"] = int(acute.place_of_service.eq("urgent_care").sum())
            rx_period = available_rx.loc[
                available_rx.fill_date.ge(index - pd.Timedelta(days=window))
            ]
            row[f"rejected_rx_{window}d"] = int(rx_period.status.eq("rejected").sum())
            row[f"reversed_rx_{window}d"] = int(rx_period.status.eq("reversed").sum())
            row[f"patient_cost_{window}d"] = float(rx_period.patient_cost.sum())
        sentinel = int(settings.get("missing_recency_value_days", history + 1))
        for name, dates in [
            ("diagnosis", diagnosis.event_date),
            ("specialist", specialist.event_date),
            ("rx", recent_conv.fill_date),
            ("treatment_change", change_dates),
        ]:
            row[f"days_since_last_{name}"] = (
                int((index - dates.max()).days) if len(dates) else sentinel
            )
        for name, frame in [("diagnosis", diagnosis), ("specialist", specialist)]:
            recent = int((frame.days_before_index <= 90).sum())
            prior = int(frame.days_before_index.between(91, 180).sum())
            row[f"{name}_acceleration"] = recent - prior
            row[f"{name}_recent_prior_ratio"] = (recent + 1) / (prior + 1)
        row["recent_coverage_90d"] = float(
            coverage_days(conventional, index - pd.Timedelta(days=89), index).mean()
        )
        row["previous_coverage_90d"] = float(
            coverage_days(
                conventional, index - pd.Timedelta(days=179), index - pd.Timedelta(days=90)
            ).mean()
        )
        row["coverage_change"] = row["recent_coverage_90d"] - row["previous_coverage_90d"]
        row["symptom_count"] = int(
            diagnosis.code.isin(settings.get("symptom_codes", ["SYN_SYMPTOM"])).sum()
        )
        row["comorbidity_count"] = int(
            diagnosis.code.isin(settings.get("comorbidity_codes", ["SYN_COMORBID"])).sum()
        )
        hcp = conventional.iloc[-1].prescriber_id if len(conventional) else None
        hcp_history = hcp_groups.get(hcp, all_rx.iloc[:0])
        hcp_history = valid_exposures(known_claims(hcp_history, "fill_date", index, lag, start))
        hcp_history = hcp_history.loc[hcp_history.patient_id.ne(pid)]
        row["hcp_historical_advanced_patients"] = int(
            hcp_history.loc[therapy.advanced_mask(hcp_history)].patient_id.nunique()
        )
        row["hcp_historical_patients"] = int(hcp_history.patient_id.nunique())
        rows.append(row)
    result = pd.DataFrame(rows)
    from .leakage import validate_predictor_names

    validate_predictor_names([c for c in result if c not in {"snapshot_id", "feature_as_of"}])
    return result


build_features = build_tabular_features
