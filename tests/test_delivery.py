"""End-to-end delivery contracts, current eligibility and HCP-only exports."""

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from therapy_switch.data.cohort import build_cohort
from therapy_switch.data.prepared_input_adapter import PreparedInputs
from therapy_switch.delivery.adapters import load_prepared, load_raw
from therapy_switch.delivery.contracts import DeliveryInputs, ScoringBatch
from therapy_switch.delivery.pipeline import DeliveryModel, load_delivery_config, run_delivery
from therapy_switch.delivery.report import write_client_report
from therapy_switch.delivery.targeting import FIELD_COLUMNS, build_field_targets, csv_safe
from therapy_switch.models.contracts import LeakageError
from therapy_switch.patient_lists import rank_patients


class ExampleModel:
    def __init__(self, reverse=False):
        self.reverse = reverse

    def predict_scores(self, frame, events):
        assert not {"label", "resp", "outcome_date"} & set(frame)
        values = frame.x.to_numpy(dtype=float)
        return 1 - values if self.reverse else values


def fit_example(train, validation, train_events, validation_events, features, *, settings, seed):
    assert "label" in train and "label" in validation
    assert not set(train.patient_id) & set(validation.patient_id)
    assert set(train_events.snapshot_id) <= set(train.snapshot_id)
    return ExampleModel(settings.get("reverse", False))


def example_adapter(config, *, include_training, expected_features=None, session=None):
    batch = make_batch()
    training = None
    if include_training:
        snapshots = batch.snapshots.copy()
        snapshots["index_date"] = pd.date_range("2018-01-01", periods=len(snapshots), freq="90D")
        snapshots["feature_cutoff"] = snapshots.index_date - pd.Timedelta(days=7)
        snapshots["lookback_start"] = snapshots.index_date - pd.Timedelta(days=365)
        snapshots["label_available_date"] = snapshots.index_date + pd.Timedelta(days=120)
        snapshots["label"] = np.arange(len(snapshots)) % 2
        snapshots["resp"] = snapshots.label
        snapshots["start_dt"] = snapshots.lookback_start
        snapshots["cohort_id"] = "SYN_TEST"
        snapshots["followup_complete"] = True
        snapshots["prediction_end"] = snapshots.index_date + pd.Timedelta(days=90)
        snapshots["outcome_date"] = (snapshots.index_date + pd.Timedelta(days=30)).where(
            snapshots.label.eq(1)
        )
        wide = batch.wide.copy()
        wide["feature_as_of"] = snapshots.index_date
        events = batch.events.copy()
        events["event_date"] = snapshots.index_date - pd.Timedelta(days=15)
        events["available_date"] = snapshots.index_date - pd.Timedelta(days=12)
        events["product_id"] = "SYN_CONV_A"
        training = PreparedInputs(snapshots, wide, events, ("x",))
    return DeliveryInputs(batch, training, {"source": "test fixture"})


def make_batch():
    n = 20
    snapshots = pd.DataFrame(
        {
            "patient_id": [f"patient_{i:02}" for i in range(n)],
            "snapshot_id": [f"assessment_{i:02}" for i in range(n)],
            "index_date": pd.Timestamp("2024-12-01"),
            "feature_cutoff": pd.Timestamp("2024-11-24"),
            "lookback_start": pd.Timestamp("2023-12-02"),
            "eligible": True,
        }
    )
    wide = pd.DataFrame(
        {
            "snapshot_id": snapshots.snapshot_id,
            "feature_as_of": pd.Timestamp("2024-12-01"),
            "x": np.arange(n) / n,
        }
    )
    events = pd.DataFrame(
        {
            "snapshot_id": snapshots.snapshot_id,
            "patient_id": snapshots.patient_id,
            "event_id": [f"event_{i:02}" for i in range(n)],
            "event_date": pd.Timestamp("2024-11-20"),
            "available_date": pd.Timestamp("2024-11-22"),
            "event_type": "pharmacy_paid",
            "code_token": "pharmacy::SYN_CONV_A",
            "code_system": "pharmacy",
            "code": "SYN_CONV_A",
            "status": "paid",
            "hcp_id": ["HCP_A"] * 10 + ["HCP_B"] * 10,
            "provider_specialty": "neurology",
            "therapy_class": "conventional",
        }
    )
    providers = pd.DataFrame(
        {
            "provider_id": ["HCP_A", "HCP_B"],
            "specialty": ["neurology", "sleep_medicine"],
            "geography": ["NORTH", "SOUTH"],
            "organization": ["ORG_A", "ORG_B"],
        }
    )
    return ScoringBatch(snapshots, wide, events, providers, ("x",)).validate("2024-12-01")


@pytest.fixture
def delivery_config(tmp_path):
    config = load_delivery_config("configs/delivery_demo.yaml")
    config["delivery"].update(
        adapter="test_delivery:example_adapter",
        output_dir=str(tmp_path / "outputs"),
        artifact_dir=str(tmp_path / "artifacts"),
        model={"trainer": "test_delivery:fit_example", "evidence_status": "experimental"},
    )
    return config


def test_train_score_then_reuse_model_and_replace_model(delivery_config):
    result = run_delivery(delivery_config)
    assert result["eligible_patients"] == 20 and result["priority_patients"] == 2
    field = pd.read_csv(result["hcp_csv"])
    assert field.hcp_id.tolist() == ["HCP_B"]
    assert field.priority_patient_count.tolist() == [2]
    assert not {"patient_id", "snapshot_id", "risk_score", "label"} & set(field)
    audit = json.loads(Path(result["manifest"]).read_text())
    assert audit["status"] == "COMPLETED" and audit["coverage"]["exported_priority_patients"] == 2
    assert audit["historical_evaluation"]["test"]["selected"] == 1
    replay_config = deepcopy(delivery_config)
    # Score-only must not resolve or invoke a trainer.
    replay_config["delivery"]["model"]["trainer"] = "does_not_exist:never_train"
    replay = run_delivery(replay_config, mode="score", model_artifact=result["model_artifact"])
    pd.testing.assert_frame_equal(field, pd.read_csv(replay["hcp_csv"]))
    delivery_config["delivery"]["model"]["reverse"] = True
    reverse = run_delivery(delivery_config)
    assert pd.read_csv(reverse["hcp_csv"]).hcp_id.tolist() == ["HCP_A"]
    html = Path(result["html_report"]).read_text(encoding="utf-8")
    assert "significant improvement has not been established" in html
    assert "<script src=" not in html and "patient_19" not in html
    assert "20" in html and "HCP_B" in html


def test_prepared_adapter_replaces_raw_without_changing_orchestrator(delivery_config, tmp_path):
    first = run_delivery(delivery_config)
    batch = make_batch()
    folder = tmp_path / "prepared"
    folder.mkdir()
    for name in ("snapshots", "wide", "events", "providers"):
        getattr(batch, name).to_csv(folder / f"{name}.csv", index=False)
    delivery_config["delivery"].update(
        adapter="therapy_switch.delivery.adapters:load_prepared",
        prepared={"scoring_dir": str(folder), "format": "csv"},
    )
    delivery_config["data"]["feature_columns"] = ["x"]
    delivery_config["data"]["feature_lineage"] = {
        "x": {
            "available_at_index": True,
            "definition": "Synthetic signal at index",
            "dtype": "numeric",
        }
    }
    prepared = load_prepared(delivery_config, include_training=False)
    assert (prepared.scoring.events.days_before_index == 11).all()
    result = run_delivery(delivery_config, mode="score", model_artifact=first["model_artifact"])
    pd.testing.assert_frame_equal(pd.read_csv(first["hcp_csv"]), pd.read_csv(result["hcp_csv"]))


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("outcome", "outcome fields"),
        ("future_wide", "unavailable"),
        ("duplicate", "one unique"),
        ("wrong_patient", "different patient"),
        ("extra_feature", "allowlist"),
        ("missing_hcp_field", "attribution fields"),
    ],
)
def test_scoring_contract_rejects_invalid_inputs(mutation, message):
    batch = make_batch()
    if mutation == "outcome":
        batch.snapshots["label"] = 0
    elif mutation == "future_wide":
        batch.wide["feature_as_of"] = pd.Timestamp("2025-01-01")
    elif mutation == "duplicate":
        batch.snapshots.loc[0, "patient_id"] = batch.snapshots.loc[1, "patient_id"]
    elif mutation == "wrong_patient":
        batch.events.loc[0, "patient_id"] = "another_patient"
    elif mutation == "extra_feature":
        batch.wide["hidden_feature"] = 10
    else:
        batch.events = batch.events.drop(columns="hcp_id")
    with pytest.raises(ValueError, match=message):
        batch.validate("2024-12-01")


@pytest.mark.parametrize("field", ["event_date", "available_date"])
def test_current_events_cannot_use_future_information(field):
    batch = make_batch()
    batch.events.loc[0, field] = pd.Timestamp("2024-12-02")
    with pytest.raises(LeakageError):
        batch.validate("2024-12-01")


def test_artifact_rejects_schema_change_and_backdated_scoring():
    batch = make_batch()
    artifact = DeliveryModel(ExampleModel(), ("x",), "2024-01-01", "experimental", None, "example")
    with pytest.raises(ValueError, match="unavailable"):
        artifact.predict(batch, "2023-12-01")
    artifact.features = ("new_feature",)
    with pytest.raises(ValueError, match="schema changed"):
        artifact.predict(batch, "2024-12-01")


def test_missing_model_leaves_a_failed_audit(delivery_config):
    with pytest.raises(ValueError, match="trusted local"):
        run_delivery(delivery_config, mode="score")
    files = list(Path(delivery_config["delivery"]["artifact_dir"]).glob("*/run_manifest.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["status"] == "FAILED"


def test_targeting_counts_and_unassigned_patients():
    batch = make_batch()
    patients = rank_patients(batch.snapshots, batch.wide.x)
    # The highest-ranked patient has no known eligible prescriber.
    batch.events.loc[19, "hcp_id"] = None
    targets, assignments, coverage = build_field_targets(
        patients, batch.events, batch.providers, {}, "2024-12-01"
    )
    assert targets.hcp_id.tolist() == ["HCP_B"]
    assert targets.priority_patient_count.tolist() == [1]
    assert targets.eligible_patient_count.tolist() == [9]
    assert coverage["unattributed_priority_patients"] == 1
    assert coverage["priority_patients"] == 2
    assert assignments.patient_id.is_unique
    empty, _, coverage = build_field_targets(
        patients, batch.events, batch.providers, {"minimum_priority_patients": 3}, "2024-12-01"
    )
    assert empty.empty and list(empty) == FIELD_COLUMNS
    assert coverage["withheld_priority_patients"] == 1


def test_targeting_ties_are_deterministic_and_counts_do_not_sum_scores():
    batch = make_batch()
    scores = np.zeros(20)
    scores[[0, 19]] = [0.91, 0.92]
    patients = rank_patients(batch.snapshots, scores)
    targets, _, _ = build_field_targets(patients, batch.events, batch.providers, {}, "2024-12-01")
    assert targets.hcp_id.tolist() == ["HCP_A", "HCP_B"]
    assert targets.priority_patient_count.tolist() == [1, 1]
    assert "expected_switchers" not in targets


def test_empty_batch_exports_headers_and_explanatory_report(delivery_config, monkeypatch):
    batch = make_batch()
    batch.snapshots = batch.snapshots.iloc[:0]
    batch.wide = batch.wide.iloc[:0]
    batch.events = batch.events.iloc[:0]
    import test_delivery

    monkeypatch.setattr(
        test_delivery, "example_adapter", lambda *args, **kwargs: DeliveryInputs(batch)
    )
    artifact = Path(delivery_config["delivery"]["artifact_dir"]) / "empty_model.joblib"
    artifact.parent.mkdir(parents=True)
    joblib.dump(
        DeliveryModel(ExampleModel(), ("x",), "2024-01-01", "experimental", None, "example"),
        artifact,
    )
    result = run_delivery(delivery_config, mode="score", model_artifact=artifact)
    assert result["hcp_targets"] == 0 and result["eligible_patients"] == 0
    assert pd.read_csv(result["hcp_csv"]).empty
    assert "No HCPs match this view" in Path(result["html_report"]).read_text(encoding="utf-8")


def test_client_files_escape_formulas_and_html(tmp_path):
    batch = make_batch()
    batch.providers.loc[1, "organization"] = '=1+1 </script><script>alert("x")</script>'
    targets, _, coverage = build_field_targets(
        rank_patients(batch.snapshots, batch.wide.x),
        batch.events,
        batch.providers,
        {},
        "2024-12-01",
    )
    assert csv_safe(targets).organization.iloc[0].startswith("'=")
    assert csv_safe(targets).priority_patient_count.iloc[0] == 2
    manifest = {
        "patient_fraction": 0.1,
        "model_evidence": "experimental",
        "data_scope": "Synthetic",
        "scoring_date": "2024-12-01",
        "run_id": "test",
    }
    path = tmp_path / "report.html"
    write_client_report(targets, coverage, manifest, path)
    html = path.read_text(encoding="utf-8")
    assert '<script>alert("x")</script>' not in html
    assert "\\u003c/script>" in html
    assert "<noscript>" in html and "<script src=" not in html


def test_live_raw_cohort_needs_no_future_labels_or_future_observation(prepared_data):
    config, raw, prepared = prepared_data
    row = prepared.snapshots.iloc[0]
    date = pd.Timestamp(row.index_date)
    config = deepcopy(config)
    config["data"]["as_of_date"] = str(date.date())
    tables = {key: frame.copy() for key, frame in raw.items()}
    tables["snapshot_candidates"] = pd.DataFrame(
        {"patient_id": [row.patient_id], "index_date": [date]}
    )
    tables["patients"]["observation_end"] = date
    tables["enrollment"]["coverage_end"] = date
    live = build_cohort(tables, config, labelled=False)
    assert len(live) == 1 and live.patient_id.item() == row.patient_id
    assert not {"label", "resp", "outcome_date", "label_available_date", "followup_complete"} & set(
        live
    )
    with pytest.raises(ValueError, match="mature"):
        build_cohort(tables, config)
    # Remove every claim unavailable at assessment: eligibility must be unchanged.
    trimmed = dict(tables)
    for name, day in [("medical_claims", "claim_date"), ("pharmacy_claims", "fill_date")]:
        trimmed[name] = tables[name].loc[
            tables[name][day].le(date) & tables[name].available_date.le(date)
        ]
    pd.testing.assert_frame_equal(live, build_cohort(trimmed, config, labelled=False))
    # Already-switched patients must be excluded even within the seven-day predictor lag.
    exposure = tables["pharmacy_claims"].iloc[[0]].copy()
    exposure["patient_id"] = row.patient_id
    exposure["fill_date"] = date - pd.Timedelta(days=1)
    exposure["available_date"] = date
    exposure["reversal_date"] = pd.NaT
    exposure["drug_id"] = config["therapy_mapping"]["advanced"][0]
    exposure["therapy_class"] = "advanced"
    exposure["status"] = "paid"
    tables["pharmacy_claims"] = pd.concat([tables["pharmacy_claims"], exposure], ignore_index=True)
    excluded = build_cohort(tables, config, labelled=False)
    assert excluded.empty and excluded.attrs["exclusions"][0]["reason"] == "known_prior_advanced"


def test_raw_adapter_current_features_ignore_future_claims_and_handles_empty(
    prepared_data, monkeypatch
):
    config, raw, prepared = prepared_data
    config = deepcopy(config)
    config["delivery"] = {"scoring_date": "2024-06-01"}
    import therapy_switch.delivery.adapters as adapters

    monkeypatch.setattr(adapters, "_raw_tables", lambda *_: raw)
    first = load_raw(config, include_training=False, expected_features=prepared.feature_columns)
    assert len(first.scoring.snapshots) > 0
    date = pd.Timestamp(config["delivery"]["scoring_date"])
    trimmed = {key: frame.copy() for key, frame in raw.items()}
    for name, day in [("medical_claims", "claim_date"), ("pharmacy_claims", "fill_date")]:
        trimmed[name] = raw[name].loc[raw[name][day].le(date) & raw[name].available_date.le(date)]
    monkeypatch.setattr(adapters, "_raw_tables", lambda *_: trimmed)
    second = load_raw(config, include_training=False, expected_features=prepared.feature_columns)
    pd.testing.assert_frame_equal(first.scoring.wide, second.scoring.wide)
    pd.testing.assert_frame_equal(first.scoring.events, second.scoring.events)
    config["data"]["as_of_date"] = config["delivery"]["scoring_date"] = "2026-01-01"
    empty = load_raw(config, include_training=False, expected_features=prepared.feature_columns)
    assert empty.scoring.snapshots.empty and empty.scoring.events.empty


def test_config_is_independent_of_working_directory(tmp_path, monkeypatch):
    source = Path("configs/delivery_demo.yaml").resolve()
    expected = load_delivery_config(source)
    monkeypatch.chdir(tmp_path)
    actual = load_delivery_config(source)
    assert actual["delivery"]["model"]["recipe"] == expected["delivery"]["model"]["recipe"]
    assert Path(actual["delivery"]["output_dir"]).is_absolute()


def test_changed_input_meaning_requires_refit(delivery_config):
    first = run_delivery(delivery_config)
    delivery_config["delivery"]["feature_contract_version"] = "changed-feature-semantics"
    with pytest.raises(ValueError, match="rules changed"):
        run_delivery(delivery_config, mode="score", model_artifact=first["model_artifact"])


def test_timestamp_exclusion_metadata_does_not_break_artifact_export(delivery_config, monkeypatch):
    import test_delivery

    original = example_adapter

    def with_metadata(*args, **kwargs):
        inputs = original(*args, **kwargs)
        inputs.scoring.snapshots.attrs["exclusions"] = [{"index_date": pd.Timestamp("2024-12-01")}]
        return inputs

    monkeypatch.setattr(test_delivery, "example_adapter", with_metadata)
    result = run_delivery(delivery_config)
    assert result["status"] == "COMPLETED"
    saved = pd.read_parquet(Path(result["manifest"]).parent / "patient_scores.parquet")
    assert len(saved) == 20 and saved.attrs == {}
