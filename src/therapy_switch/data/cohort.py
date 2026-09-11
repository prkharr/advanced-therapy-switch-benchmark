"""Monthly eligibility and future labels, keyed by snapshot rather than patient."""

from __future__ import annotations

import pandas as pd

from ._config import therapy_definition, timeline_days
from .availability import coverage_days, known_claims, maximum_gap, valid_exposures


def _enrolled(intervals, start, end):
    cursor = pd.Timestamp(start)
    for row in intervals.sort_values("coverage_start").itertuples(index=False):
        left, right = pd.Timestamp(row.coverage_start), pd.Timestamp(row.coverage_end)
        if right < cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right + pd.Timedelta(days=1))
        if cursor > end:
            return True
    return False


def snapshot_candidates(tables, config):
    patients = tables["patients"]
    timeline = config.get("timeline", {})
    strategy = timeline.get("index_date_strategy", "monthly")
    count = int(timeline.get("snapshots_per_patient", 3))
    first = timeline.get("index_date_start")
    last = timeline.get("index_date_end")
    records = []
    if "snapshot_candidates" in tables:
        candidates = tables["snapshot_candidates"][["patient_id", "index_date"]].copy()
    else:
        for patient in patients.itertuples(index=False):
            if strategy in {"provided", "monthly"} and hasattr(patient, "index_date"):
                dates = [
                    pd.Timestamp(patient.index_date) + pd.DateOffset(months=i) for i in range(count)
                ]
            elif strategy == "monthly" and first and last:
                dates = pd.date_range(first, last, freq="MS")
            else:
                raise ValueError(
                    "Provide snapshot_candidates, patient index_date, or a monthly calendar"
                )
            records.extend({"patient_id": patient.patient_id, "index_date": date} for date in dates)
        candidates = pd.DataFrame(records)
    candidates["index_date"] = pd.to_datetime(candidates.index_date, errors="raise")
    if candidates[["patient_id", "index_date"]].isna().any().any() or candidates.duplicated().any():
        raise ValueError("Candidate patient/index pairs must be non-null and unique")
    if first:
        candidates = candidates.loc[candidates.index_date >= pd.Timestamp(first)]
    if last:
        candidates = candidates.loc[candidates.index_date <= pd.Timestamp(last)]
    return candidates.sort_values(["index_date", "patient_id"]).reset_index(drop=True)


def build_cohort(tables, config=None, *, labelled=True):
    """Build mature training snapshots or label-free, as-known scoring snapshots.

    Scoring requires observation only through index and excludes every advanced
    exposure already known at index, including within the predictor service lag.
    """
    config = config or {}
    observation, horizon = timeline_days(config)
    timeline, settings = config.get("timeline", {}), config.get("cohort", {})
    lag = int(timeline.get("claims_lag_days", 0))
    runout = int(timeline.get("label_runout_days", 30))
    if "as_of_date" not in config.get("data", {}):
        raise ValueError("data.as_of_date is required to establish label maturity")
    as_of = pd.Timestamp(config["data"]["as_of_date"])
    history_min = int(timeline.get("minimum_history_days", observation))
    followup_min = max(horizon, int(timeline.get("minimum_followup_days", horizon)))
    coverage_window = int(settings.get("coverage_window_days", 270))
    coverage_min = int(settings.get("minimum_covered_days", 135))
    diagnosis_codes = settings["diagnosis_codes"]
    confirm_codes = settings["confirmation_codes"]
    separation = int(settings.get("diagnosis_separation_days", 90))
    therapy = therapy_definition(config)
    patients = tables["patients"].set_index("patient_id")
    rx_groups = dict(tuple(tables["pharmacy_claims"].groupby("patient_id", sort=False)))
    med_groups = dict(tuple(tables["medical_claims"].groupby("patient_id", sort=False)))
    enroll_groups = dict(tuple(tables["enrollment"].groupby("patient_id", sort=False)))
    empty_rx, empty_med = tables["pharmacy_claims"].iloc[:0], tables["medical_claims"].iloc[:0]
    rows, exclusions = [], []
    for candidate in snapshot_candidates(tables, config).itertuples(index=False):
        pid, index = candidate.patient_id, pd.Timestamp(candidate.index_date)
        if pid not in patients.index:
            raise ValueError("Candidate contains unknown patient")
        p = patients.loc[pid]
        start, end = index - pd.Timedelta(days=observation), index + pd.Timedelta(days=horizon)
        label_available = end + pd.Timedelta(days=runout)
        reason = None
        enrollment = enroll_groups.get(pid, tables["enrollment"].iloc[:0])
        required_start = index - pd.Timedelta(days=max(observation, history_min, coverage_window))
        required_end = index + pd.Timedelta(days=followup_min) if labelled else index
        if index > as_of:
            reason = "future_index"
        elif labelled and label_available > as_of:
            reason = "immature_label"
        elif (
            pd.Timestamp(p.observation_start) > required_start
            or pd.Timestamp(p.observation_end) < required_end
            or not _enrolled(enrollment, required_start, required_end)
        ):
            reason = "insufficient_observation_or_enrollment"
        rx_all = rx_groups.get(pid, empty_rx)
        rx_known = valid_exposures(known_claims(rx_all, "fill_date", index, lag))
        conventional = rx_known.loc[therapy.conventional_mask(rx_known)]
        med_known = valid_exposures(
            known_claims(med_groups.get(pid, empty_med), "claim_date", index, lag)
        )
        first_dx = med_known.loc[med_known.diagnosis_code.isin(diagnosis_codes), "claim_date"].min()
        confirms = med_known.loc[med_known.diagnosis_code.isin(confirm_codes), "claim_date"]
        confirmed = (
            pd.notna(first_dx) and (confirms > first_dx + pd.Timedelta(days=separation)).any()
        )
        covered = coverage_days(conventional, index - pd.Timedelta(days=coverage_window - 1), index)
        if (
            reason is None
            and settings.get("require_diagnosis_confirmation", True)
            and not confirmed
        ):
            reason = "diagnosis_not_confirmed"
        if reason is None and conventional.empty:
            reason = "no_conventional_exposure"
        if reason is None and int(covered.sum()) < coverage_min:
            reason = "insufficient_conventional_coverage"
        if (
            reason is None
            and settings.get("require_conventional_on_index", False)
            and not covered[-1]
        ):
            reason = "no_conventional_coverage_on_index"
        maximum_allowed = settings.get("maximum_gap_days")
        if (
            reason is None
            and maximum_allowed is not None
            and maximum_gap(covered) > int(maximum_allowed)
        ):
            reason = "conventional_gap"
        eligibility_rx = (
            rx_known if labelled else valid_exposures(known_claims(rx_all, "fill_date", index, 0))
        )
        if reason is None and therapy.advanced_mask(eligibility_rx).any():
            reason = "known_prior_advanced"
        sid = f"{pid}__{index:%Y%m%d}"
        if reason:
            exclusions.append(
                {"snapshot_id": sid, "patient_id": pid, "index_date": index, "reason": reason}
            )
            continue
        outcome_date = pd.NaT
        if labelled:
            observed_rx = valid_exposures(known_claims(rx_all, "fill_date", label_available))
            future = observed_rx.loc[
                therapy.advanced_mask(observed_rx)
                & observed_rx.fill_date.gt(index)
                & observed_rx.fill_date.le(end)
            ]
            outcome_date = future.fill_date.min()
        first_fill = conventional.fill_date.min()
        rows.append(
            {
                "snapshot_id": sid,
                "patient_id": pid,
                "cohort_id": settings["cohort_id"],
                "start_dt": first_fill,
                "index_date": index,
                "feature_cutoff": index - pd.Timedelta(days=lag),
                "lookback_start": start,
                "prediction_end": end,
                "label_available_date": label_available,
                "resp": int(pd.notna(outcome_date)),
                "label": int(pd.notna(outcome_date)),
                "outcome_date": outcome_date,
                "eligible": True,
                "followup_complete": True,
                "covered_days": int(covered.sum()),
                "maximum_gap_days": maximum_gap(covered),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        if labelled:
            raise ValueError("No eligible mature snapshots; inspect dates and cohort rules")
        result = pd.DataFrame(
            columns=[
                "snapshot_id",
                "patient_id",
                "cohort_id",
                "start_dt",
                "index_date",
                "feature_cutoff",
                "lookback_start",
                "prediction_end",
                "eligible",
                "covered_days",
                "maximum_gap_days",
            ]
        )
    if not labelled:
        result = result.drop(
            columns=["label", "resp", "outcome_date", "label_available_date", "followup_complete"],
            errors="ignore",
        )
    for column in ("index_date", "feature_cutoff", "lookback_start", "prediction_end", "start_dt"):
        result[column] = pd.to_datetime(result[column])
    result.attrs["exclusions"] = exclusions
    return result.sort_values(["index_date", "patient_id"]).reset_index(drop=True)


def validate_cohort_timeline(cohort, tables, config=None):
    expected = build_cohort(tables, config).set_index("snapshot_id")
    actual = cohort.set_index("snapshot_id")
    columns = ["patient_id", "index_date", "label", "outcome_date", "label_available_date"]
    pd.testing.assert_frame_equal(
        actual[columns].sort_index(), expected[columns].sort_index(), check_dtype=False
    )
