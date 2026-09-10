from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from therapy_switch.data.availability import (
    coverage_days,
    known_claims,
    maximum_gap,
    valid_exposures,
)
from therapy_switch.data.cohort import build_cohort, snapshot_candidates
from therapy_switch.data.event_sequences import encode_event_sequences, fit_sequence_vocabularies
from therapy_switch.data.generate_synthetic_claims import generate_synthetic_claims
from therapy_switch.data.raw_source_adapter import canonicalize_raw_tables
from therapy_switch.data.splitting import (
    split_manifest,
    stratified_patient_split,
    temporal_patient_split,
)
from therapy_switch.schemas import SchemaError, validate_tables


def test_synthetic_contract_seed_and_nontrivial_outcomes(prepared_data):
    config, tables, inputs = prepared_data
    validate_tables(tables)
    assert set(tables) == {
        "patients",
        "medical_claims",
        "pharmacy_claims",
        "providers",
        "plans",
        "enrollment",
        "therapy_mapping",
    }
    again = canonicalize_raw_tables(generate_synthetic_claims(config), config)
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], again[name])
        assert not {"label", "resp", "latent_risk"}.intersection(tables[name])
    assert 0.02 < inputs.snapshots.label.mean() < 0.30
    assert tables["patients"].patient_id.str.startswith("SYN_").all()
    assert tables["medical_claims"].place_of_service.eq("urgent_care").any()
    assert tables["pharmacy_claims"].status.eq("rejected").any()
    assert tables["pharmacy_claims"].reversal_date.notna().any()


def test_repeated_snapshots_and_labels(prepared_data):
    config, tables, inputs = prepared_data
    snapshots = inputs.snapshots
    assert snapshots.snapshot_id.is_unique
    assert snapshots.groupby("patient_id").size().max() == 3
    positives = snapshots.loc[snapshots.label.eq(1)]
    assert positives.outcome_date.gt(positives.index_date).all()
    assert positives.outcome_date.le(positives.prediction_end).all()
    assert snapshots.loc[snapshots.label.eq(0), "outcome_date"].isna().all()
    assert snapshots.label_available_date.le(pd.Timestamp(config["data"]["as_of_date"])).all()
    assert snapshots.covered_days.ge(config["cohort"]["minimum_covered_days"]).all()


def test_interval_union_carry_in_overlap_and_gaps():
    fills = pd.DataFrame(
        {
            "fill_date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-14"]),
            "days_supply": [7, 5, 2],
        }
    )
    coverage = coverage_days(fills, "2024-01-05", "2024-01-15")
    np.testing.assert_array_equal(coverage, [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    assert maximum_gap(coverage) == 5


def test_claim_availability_lag_and_reversal_as_of():
    frame = pd.DataFrame(
        {
            "fill_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-09"]),
            "available_date": pd.to_datetime(["2024-01-02", "2024-02-01", "2024-01-09"]),
            "reversal_date": pd.to_datetime(["2024-01-20", None, None]),
            "status": ["paid", "paid", "paid"],
        }
    )
    known = known_claims(frame, "fill_date", "2024-01-10", lag_days=3)
    assert list(known.index) == [0]
    assert len(valid_exposures(known)) == 1
    assert valid_exposures(known_claims(frame, "fill_date", "2024-01-21")).index.tolist() == [2]
    assert frame.status.eq("paid").all()


def _single_patient(tables, patient_id):
    subset = {name: value.copy() for name, value in tables.items()}
    for name in ("patients", "medical_claims", "pharmacy_claims", "enrollment"):
        subset[name] = subset[name].loc[subset[name].patient_id.eq(patient_id)].copy()
    return subset


def test_cohort_maturity_coverage_and_diagnosis_rules(prepared_data):
    config, tables, inputs = prepared_data
    pid = inputs.snapshots.patient_id.iloc[0]
    subset = _single_patient(tables, pid)
    rows = build_cohort(subset, config)
    invalid = deepcopy(config)
    invalid["data"]["as_of_date"] = str(rows.index_date.min().date())
    with pytest.raises(ValueError, match="No eligible"):
        build_cohort(subset, invalid)
    subset["enrollment"]["coverage_end"] = rows.index_date.min()
    with pytest.raises(ValueError, match="No eligible"):
        build_cohort(subset, config)
    subset = _single_patient(tables, pid)
    subset["medical_claims"]["diagnosis_code"] = "SYN_UNRELATED"
    with pytest.raises(ValueError, match="No eligible"):
        build_cohort(subset, config)
    subset = _single_patient(tables, pid)
    subset["pharmacy_claims"]["days_supply"] = 0
    with pytest.raises(ValueError, match="No eligible"):
        build_cohort(subset, config)


def test_known_prior_advanced_excludes_and_late_future_claim_is_not_a_label(prepared_data):
    config, tables, inputs = prepared_data
    positive = inputs.snapshots.loc[inputs.snapshots.label.eq(1)].iloc[0]
    subset = _single_patient(tables, positive.patient_id)
    subset["snapshot_candidates"] = pd.DataFrame([positive[["patient_id", "index_date"]]])
    rx = subset["pharmacy_claims"]
    advanced = rx.therapy_class.eq("advanced")
    rx.loc[advanced, "available_date"] = positive.label_available_date + pd.Timedelta(days=1)
    assert build_cohort(subset, config).label.item() == 0
    rx.loc[advanced, "fill_date"] = positive.index_date - pd.Timedelta(days=20)
    rx.loc[advanced, "available_date"] = positive.index_date - pd.Timedelta(days=10)
    rx.loc[advanced, "reversal_date"] = pd.NaT
    rx.loc[advanced, "status"] = "paid"
    with pytest.raises(ValueError, match="No eligible"):
        build_cohort(subset, config)


def test_strict_splits_preserve_patients_time_and_maturity(prepared_data):
    config, _, inputs = prepared_data
    frame = inputs.modeling_frame()
    for splitter in (stratified_patient_split, temporal_patient_split):
        split = splitter(frame, config)
        sets = [set(part.patient_id) for part in split.values()]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        repeated = splitter(frame, config)
        for name in split:
            pd.testing.assert_frame_equal(split[name], repeated[name])
    assert split["train"].label_available_date.max() < split["validation"].index_date.min()
    assert split["validation"].label_available_date.max() < split["test"].index_date.min()
    manifest = split_manifest(frame, split)
    assert manifest.snapshot_id.is_unique
    assert manifest.split.eq("purged").any()
    relabeled = frame.assign(label=1 - frame.label)
    for name, part in temporal_patient_split(relabeled, config).items():
        assert part.snapshot_id.tolist() == split[name].snapshot_id.tolist()


def test_sequence_training_vocabulary_unknown_truncation_padding_and_empty(prepared_data):
    config, _, inputs = prepared_data
    snaps = inputs.snapshots.iloc[:3].copy()
    events = inputs.events.loc[inputs.events.snapshot_id.isin(snaps.snapshot_id)].copy()
    train_events = events.loc[events.snapshot_id.eq(snaps.snapshot_id.iloc[0])]
    vocab = fit_sequence_vocabularies(train_events)
    events.loc[events.snapshot_id.eq(snaps.snapshot_id.iloc[1]), "code_token"] = "heldout::unseen"
    events = events.loc[events.snapshot_id.ne(snaps.snapshot_id.iloc[2])]
    config = deepcopy(config)
    config["features"]["sequence"]["max_length"] = 8
    encoded = encode_event_sequences(events.sample(frac=1, random_state=1), snaps, config, vocab)
    assert encoded.lengths.tolist() == [8, 8, 0]
    assert (encoded.code_ids[1] == 1).all()
    assert (encoded.code_ids[2] == 0).all()
    assert not encoded.attention_mask[2].any()
    assert encoded.time_delta_days[:, 0].tolist() == [0, 0, 0]
    expected = train_events.sort_values(["event_date", "event_id"], kind="stable").tail(8)
    np.testing.assert_array_equal(encoded.event_dates[0], expected.event_date.to_numpy())
    assert "heldout::unseen" not in vocab["code"]
    encoded.to_sequence_split().validated(3)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "orphan", "negative_supply", "future_availability"]
)
def test_raw_schema_rejects_invalid_claims(prepared_data, mutation):
    _, tables, _ = prepared_data
    raw = {key: frame.copy() for key, frame in tables.items()}
    rx = raw["pharmacy_claims"]
    if mutation == "duplicate":
        raw["pharmacy_claims"] = pd.concat([rx, rx.iloc[:1]], ignore_index=True)
    elif mutation == "orphan":
        rx.loc[0, "patient_id"] = "UNKNOWN"
    elif mutation == "negative_supply":
        rx.loc[0, "days_supply"] = -1
    else:
        rx.loc[0, "available_date"] = rx.loc[0, "fill_date"] - pd.Timedelta(days=1)
    with pytest.raises(SchemaError):
        validate_tables(raw)


def test_candidate_duplicates_rejected(prepared_data):
    config, tables, inputs = prepared_data
    raw = dict(tables)
    row = inputs.snapshots[["patient_id", "index_date"]].iloc[:1]
    raw["snapshot_candidates"] = pd.concat([row, row])
    with pytest.raises(ValueError, match="unique"):
        snapshot_candidates(raw, config)
