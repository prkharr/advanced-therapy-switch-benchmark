from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from therapy_switch.data.event_sequences import build_event_frame
from therapy_switch.data.prepared_input_adapter import (
    load_prepared_inputs,
    validate_prepared_inputs,
)
from therapy_switch.features import build_tabular_features
from therapy_switch.features.leakage import LeakageError, validate_predictor_names


@pytest.mark.parametrize(
    "column",
    [
        "RESP",
        "OUTCOME_DATE",
        "future_rx",
        "post_index_cost",
        "label_rate",
        "patient_id",
        "snapshot_id",
        "split",
        "target_encoding",
        "hcp_id",
        "cohort_id",
    ],
)
def test_target_and_identifier_predictors_are_forbidden(column):
    with pytest.raises(LeakageError):
        validate_predictor_names(["age", column])


def test_future_and_late_claims_cannot_change_predictors_or_sequences(prepared_data):
    config, tables, inputs = prepared_data
    snapshot = inputs.snapshots.iloc[:1]
    index = snapshot.index_date.iloc[0]
    changed = {key: frame.copy() for key, frame in tables.items()}
    med = changed["medical_claims"]
    rx = changed["pharmacy_claims"]
    for frame, date_col in ((med, "claim_date"), (rx, "fill_date")):
        future = frame[date_col].gt(index) | frame.available_date.gt(index)
        frame.loc[future, "patient_cost"] = 9e8
        frame.loc[future, "status"] = "rejected"
    med.loc[med.claim_date.gt(index), "diagnosis_code"] = "FUTURE_INFORMATION"
    rx.loc[rx.fill_date.gt(index), "therapy_class"] = "advanced"
    expected = build_tabular_features(tables, snapshot, config)
    actual = build_tabular_features(changed, snapshot, config)
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(
        build_event_frame(tables, snapshot, config), build_event_frame(changed, snapshot, config)
    )


def test_relabeling_cannot_change_features(prepared_data):
    config, tables, inputs = prepared_data
    snapshots = inputs.snapshots.iloc[:3].copy()
    other = snapshots.assign(
        label=1 - snapshots.label, resp=1 - snapshots.resp, outcome_date=pd.NaT
    )
    pd.testing.assert_frame_equal(
        build_tabular_features(tables, snapshots, config),
        build_tabular_features(tables, other, config),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "future_event",
        "late_arrival",
        "orphan",
        "different_patient",
        "extra_predictor",
        "outcome_outside",
        "immature",
        "missing_date",
        "wrong_lag",
        "duplicate_event",
        "unreviewed_feature",
        "future_wide",
    ],
)
def test_prepared_contract_fails_closed(prepared_data, prepared_config, mutation):
    _, _, original = prepared_data
    frames = {name: getattr(original, name).copy() for name in ("snapshots", "wide", "events")}
    config = deepcopy(prepared_config)
    index = frames["snapshots"].set_index("snapshot_id").index_date
    sid = frames["events"].snapshot_id.iloc[0]
    if mutation == "future_event":
        frames["events"].loc[0, "event_date"] = index[sid] + pd.Timedelta(days=1)
    elif mutation == "late_arrival":
        frames["events"].loc[0, "available_date"] = index[sid] + pd.Timedelta(days=1)
    elif mutation == "orphan":
        frames["events"].loc[0, "snapshot_id"] = "unknown"
    elif mutation == "different_patient":
        frames["events"].loc[0, "patient_id"] = "different"
    elif mutation == "extra_predictor":
        frames["wide"]["unapproved"] = 1
    elif mutation == "outcome_outside":
        row = frames["snapshots"].index[frames["snapshots"].label.eq(1)][0]
        frames["snapshots"].loc[row, "outcome_date"] = frames["snapshots"].loc[row, "index_date"]
    elif mutation == "immature":
        frames["snapshots"]["label_available_date"] = pd.Timestamp("2099-01-01")
    elif mutation == "missing_date":
        frames["snapshots"].loc[0, "feature_cutoff"] = pd.NaT
    elif mutation == "wrong_lag":
        frames["snapshots"]["feature_cutoff"] = frames["snapshots"].index_date
    elif mutation == "duplicate_event":
        frames["events"] = pd.concat(
            [frames["events"], frames["events"].iloc[:1]], ignore_index=True
        )
    elif mutation == "unreviewed_feature":
        config["data"]["feature_lineage"] = {}
    else:
        frames["wide"]["feature_as_of"] = pd.Timestamp("2099-01-01")
    with pytest.raises((ValueError, LeakageError)):
        load_prepared_inputs(frames, config)


def test_direct_prepared_validation_rejects_nat(prepared_data):
    config, _, original = prepared_data
    snapshots = original.snapshots.copy()
    snapshots.loc[0, "start_dt"] = pd.NaT
    with pytest.raises(ValueError):
        validate_prepared_inputs(
            replace(original, snapshots=snapshots), config, require_lineage=False
        )
