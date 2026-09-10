"""Direct prepared snapshot/wide/event inputs; no raw-claims re-engineering."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from therapy_switch.features.leakage import (
    LeakageError,
    audit_feature_frame,
    validate_predictor_names,
)

from .event_sequences import normalize_events


@dataclass
class PreparedInputs:
    snapshots: pd.DataFrame
    wide: pd.DataFrame
    events: pd.DataFrame
    feature_columns: tuple[str, ...]

    def modeling_frame(self):
        return self.snapshots.merge(
            self.wide.drop(columns="feature_as_of"),
            on="snapshot_id",
            how="left",
            validate="one_to_one",
        )


def validate_prepared_inputs(inputs, config, *, require_lineage=True):
    snapshots, wide, events = inputs.snapshots, inputs.wide, inputs.events
    required = {
        "snapshot_id",
        "patient_id",
        "cohort_id",
        "start_dt",
        "index_date",
        "resp",
        "label",
        "outcome_date",
        "lookback_start",
        "feature_cutoff",
        "prediction_end",
        "label_available_date",
        "eligible",
        "followup_complete",
    }
    if not required.issubset(snapshots):
        raise ValueError(f"Snapshot columns missing: {sorted(required - set(snapshots))}")
    if snapshots.empty or snapshots.snapshot_id.isna().any() or not snapshots.snapshot_id.is_unique:
        raise ValueError("Snapshots require nonempty unique non-null snapshot_id")
    date_fields = [
        "start_dt",
        "index_date",
        "lookback_start",
        "feature_cutoff",
        "prediction_end",
        "label_available_date",
    ]
    if snapshots[date_fields + ["cohort_id"]].isna().any().any():
        raise ValueError("Snapshot timeline and cohort identity cannot be missing")
    if (snapshots.start_dt > snapshots.feature_cutoff).any():
        raise LeakageError("Conventional therapy start exceeds feature cutoff")
    if (
        snapshots.patient_id.isna().any()
        or snapshots.duplicated(["patient_id", "index_date"]).any()
    ):
        raise ValueError("Duplicate or null patient/index snapshots")
    if not snapshots.label.isin([0, 1]).all() or not snapshots.label.eq(snapshots.resp).all():
        raise ValueError("RESP/label must be aligned binary values")
    if not snapshots.eligible.eq(True).all() or not snapshots.followup_complete.eq(True).all():
        raise ValueError("Benchmark snapshots must be eligible with complete follow-up")
    timeline = config["timeline"]
    horizon = int(timeline["prediction_window_days"])
    expected_end = snapshots.index_date + pd.Timedelta(days=horizon)
    if not snapshots.prediction_end.eq(expected_end).all():
        raise LeakageError("Prediction end does not match horizon")
    if (
        snapshots.feature_cutoff
        > snapshots.index_date - pd.Timedelta(days=int(timeline.get("claims_lag_days", 0)))
    ).any():
        raise LeakageError("Feature cutoff exceeds configured claims lag")
    if (snapshots.lookback_start > snapshots.feature_cutoff).any():
        raise ValueError("Inverted feature history")
    earliest_maturity = expected_end + pd.Timedelta(days=int(timeline.get("label_runout_days", 30)))
    if (snapshots.label_available_date < earliest_maturity).any():
        raise LeakageError("Labels claim maturity before prediction end and runout")
    if (snapshots.label_available_date > pd.Timestamp(config["data"]["as_of_date"])).any():
        raise LeakageError("Immature future labels")
    positives = snapshots.label.eq(1)
    if not snapshots.outcome_date.notna().eq(positives).all():
        raise ValueError("Outcome date must be present exactly for positives")
    if (
        (snapshots.loc[positives, "outcome_date"] <= snapshots.loc[positives, "index_date"])
        | (snapshots.loc[positives, "outcome_date"] > snapshots.loc[positives, "prediction_end"])
    ).any():
        raise LeakageError("Outcome outside strictly future window")
    if (
        wide.snapshot_id.isna().any()
        or not wide.snapshot_id.is_unique
        or set(wide.snapshot_id) != set(snapshots.snapshot_id)
    ):
        raise ValueError("Wide inputs must align one-to-one with every snapshot")
    validate_predictor_names(inputs.feature_columns)
    if set(wide) != {"snapshot_id", "feature_as_of", *inputs.feature_columns}:
        raise LeakageError("Wide table contains columns outside the explicit feature allowlist")
    audit_feature_frame(wide)
    wide_check = wide[["snapshot_id", "feature_as_of"]].merge(
        snapshots[["snapshot_id", "index_date"]]
    )
    if (
        wide_check.feature_as_of.isna().any()
        or (wide_check.feature_as_of > wide_check.index_date).any()
    ):
        raise LeakageError("Wide feature availability is missing or post-index")
    if require_lineage:
        lineage = config["data"].get("feature_lineage", {})
        for feature in inputs.feature_columns:
            record = lineage.get(feature, {})
            if record.get("available_at_index") is not True or not record.get("definition"):
                raise ValueError(f"Reviewed point-in-time lineage required for feature: {feature}")
    required_events = {
        "snapshot_id",
        "patient_id",
        "event_id",
        "event_date",
        "available_date",
        "event_type",
        "code_system",
        "code",
        "hcp_id",
        "product_id",
        "therapy_class",
        "provider_specialty",
        "status",
    }
    if not required_events.issubset(events):
        raise ValueError(f"Event columns missing: {sorted(required_events - set(events))}")
    if (
        events[
            [
                "snapshot_id",
                "patient_id",
                "event_id",
                "event_date",
                "available_date",
                "code",
                "code_system",
                "event_type",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Event identity, dates and tokens cannot be null")
    if events.duplicated(["snapshot_id", "event_id"]).any():
        raise ValueError("Duplicate snapshot event IDs")
    checked = events.drop(columns=["index_date"], errors="ignore").merge(
        snapshots[["snapshot_id", "patient_id", "index_date", "feature_cutoff", "lookback_start"]],
        on="snapshot_id",
        how="left",
        suffixes=("", "_snapshot"),
        validate="many_to_one",
    )
    if (
        checked.index_date.isna().any()
        or not checked.patient_id.eq(checked.patient_id_snapshot).all()
    ):
        raise ValueError("Events are orphaned or linked to a different patient")
    if (checked.event_date > checked.feature_cutoff).any() or (
        checked.available_date > checked.index_date
    ).any():
        raise LeakageError("Post-index or unavailable sequence events")
    if (checked.event_date < checked.lookback_start).any() or (
        checked.available_date < checked.event_date
    ).any():
        raise LeakageError("Invalid event history/availability interval")
    for col in inputs.feature_columns:
        if pd.api.types.is_datetime64_any_dtype(wide[col]):
            raise LeakageError(f"Unreviewed raw date predictor: {col}")
        if (
            pd.api.types.is_numeric_dtype(wide[col])
            and np.isinf(wide[col].to_numpy(dtype=float)).any()
        ):
            raise ValueError("Infinite wide features")
    return inputs


def load_prepared_inputs(tables, config):
    mapping = config["data"].get("tables", {})
    frames = {}
    for name in ("snapshots", "wide", "events"):
        frame = tables[name].rename(columns=mapping.get(name, {}).get("columns", {})).copy()
        frame.columns = [str(c).lower() for c in frame]
        if not frame.columns.is_unique:
            raise ValueError("Duplicate prepared columns after canonicalization")
        for col in (
            "start_dt",
            "index_date",
            "outcome_date",
            "lookback_start",
            "feature_cutoff",
            "prediction_end",
            "label_available_date",
            "feature_as_of",
            "event_date",
            "available_date",
        ):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], errors="raise")
                if col != "outcome_date" and frame[col].isna().any():
                    raise ValueError(f"Missing required date: {col}")
        frames[name] = frame
    if "label" not in frames["snapshots"] and "resp" in frames["snapshots"]:
        frames["snapshots"]["label"] = frames["snapshots"].resp
    for col in ("resp", "label"):
        frames["snapshots"][col] = pd.to_numeric(frames["snapshots"][col], errors="raise")
    for col in ("covered_days", "maximum_gap_days"):
        if col in frames["snapshots"]:
            frames["snapshots"][col] = pd.to_numeric(frames["snapshots"][col], errors="raise")
    for col in ("eligible", "followup_complete"):
        frames["snapshots"][col] = frames["snapshots"][col].map(
            lambda value: {"true": True, "false": False, "1": True, "0": False}.get(
                str(value).lower(), value
            )
        )
    columns = tuple(config["data"].get("feature_columns", []))
    if not columns:
        raise ValueError("Prepared input requires an explicit feature_columns allowlist")
    for col in columns:
        dtype = config["data"].get("feature_lineage", {}).get(col, {}).get("dtype", "numeric")
        if dtype == "numeric":
            frames["wide"][col] = pd.to_numeric(frames["wide"][col], errors="raise")
        elif dtype != "categorical":
            raise ValueError(f"Unknown feature dtype for {col}")
    result = PreparedInputs(frames["snapshots"], frames["wide"], frames["events"], columns)
    validate_prepared_inputs(result, config)
    events = (
        frames["events"]
        .drop(columns="index_date", errors="ignore")
        .merge(
            frames["snapshots"][["snapshot_id", "index_date"]],
            on="snapshot_id",
            validate="many_to_one",
        )
    )
    result.events = normalize_events(events)
    return result
