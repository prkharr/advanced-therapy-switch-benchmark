"""Manual-folder ingestion, export integrity, timing, and baseline boundaries."""

import json
from pathlib import Path

import pandas as pd
import pytest

from therapy_switch.config import validate_config
from therapy_switch.data.cohort import build_cohort
from therapy_switch.delivery.raw_export import export_raw_data
from therapy_switch.features.feature_engineering import build_tabular_features
from therapy_switch.io import load_claims_directory
from therapy_switch.models.baselines import fit_baselines
from therapy_switch.profiling import raw_profile


def test_downloaded_folder_needs_no_export_manifest(config, raw, tmp_path):
    validate_config(config)
    for name, frame in raw.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    config["data"]["input_dir"] = str(tmp_path)
    tables = load_claims_directory(config)
    assert tables["pharmacy_claims"].drug_id.iloc[0] == "00000000001"
    cohort = build_cohort(tables, config)
    assert cohort.set_index("patient_id").label.to_dict() == {"p1": 1, "p2": 0}
    features = build_tabular_features(tables, cohort, config)
    assert len(features.columns) - 2 == 70
    assert features.pharmacy_count_30d.eq(1).all()  # July outcome cannot enter June history.
    assert raw_profile(tables)["patients"]["distinct_patients"] == 2
    tables["medical_claims"].loc[1, "available_date"] = pd.Timestamp("2024-07-02")
    assert set(build_cohort(tables, config).patient_id) == {"p2"}


class Frame:
    def __init__(self, data):
        self.data = data
        self.columns = list(data)

    def limit(self, n):
        return Frame(self.data.iloc[:n])

    def to_pandas_batches(self):
        yield self.data


class Session:
    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def sql(self, query):
        self.calls.append(query)
        return Frame(self.tables[query.split()[-1]])


def test_export_integrity_and_query_placeholder(config, raw, tmp_path):
    sql = tmp_path / "queries"
    sql.mkdir()
    for name in raw:
        (sql / f"{name}.sql").write_text(f"SELECT * FROM {name};")
    session = Session(raw)
    result = export_raw_data(session, config, sql, tmp_path / "export")
    config["data"]["input_dir"] = result["raw_dir"]
    assert len(load_claims_directory(config)["patients"]) == 2
    path = Path(result["raw_dir"]) / "patients.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after export"):
        load_claims_directory(config)
    (sql / "patients.sql").write_text("SELECT * FROM {{UNRESOLVED}}")
    before = len(session.calls)
    with pytest.raises(ValueError, match="placeholders"):
        export_raw_data(session, config, sql, tmp_path / "export")
    assert len(session.calls) == before


def test_oversize_export_is_failed(config, raw, tmp_path):
    sql = tmp_path / "queries"
    sql.mkdir()
    for name in raw:
        (sql / f"{name}.sql").write_text(f"SELECT * FROM {name}")
    with pytest.raises(ValueError, match="exceeds"):
        export_raw_data(Session(raw), config, sql, tmp_path / "export", max_rows=1)
    manifest = next((tmp_path / "export").glob("*/export_manifest.json"))
    assert json.loads(manifest.read_text())["status"] == "FAILED"


def test_independent_models_fit_without_test_inputs():
    # Minimal numerical classifier case; no source claims or empirical performance claims.
    train = pd.DataFrame({"patient_id": [f"a{i}" for i in range(40)],
                          "snapshot_id": [f"a{i}" for i in range(40)],
                          "index_date": pd.Timestamp("2022-01-01"),
                          "x": [0., 1.] * 20, "label": [0, 1] * 20})
    validation = train.iloc[:10].copy()
    validation["patient_id"] = [f"b{i}" for i in range(10)]
    validation["snapshot_id"] = validation.patient_id
    validation["x"] += 100  # Held-out distribution must not influence preprocessing.
    model = fit_baselines(train, validation, None, None, ["x"], settings={}, seed=42)
    assert set(model.candidates) == {"logistic_regression", "lightgbm"}
    assert len(model.predict_scores(validation)) == 10
    assert model.parameters["logistic_regression"]["class_weight"] is None
    assert model.parameters["lightgbm"]["n_estimators"] == 100
    for candidate in model.candidates.values():
        numeric = candidate.named_steps["preprocessor"].named_transformers_["numeric"]
        assert numeric.named_steps["imputer"].statistics_[0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="disjoint"):
        fit_baselines(train, train.iloc[:10], None, None, ["x"], settings={}, seed=42)


def test_raw_folder_to_hcp_outputs_and_heldout_comparison(config, raw, tmp_path):
    """Exercise the full interface using three small independent calendar cases."""
    import yaml

    from therapy_switch.delivery.pipeline import run_delivery
    from therapy_switch.schemas import CANONICAL_SCHEMAS

    parts = {name: [] for name in raw}
    for cohort_number, months in enumerate((-18, -12, -6)):
        for copy_number in range(4):
            prefix = f"g{cohort_number}_{copy_number}_"
            for name, original in raw.items():
                if name in {"providers", "plans", "therapy_mapping"}:
                    continue
                frame = original.copy()
                for column in ("patient_id", "claim_id"):
                    if column in frame:
                        frame[column] = prefix + frame[column]
                for column in CANONICAL_SCHEMAS[name].date_columns:
                    frame[column] = pd.to_datetime(frame[column]) + pd.DateOffset(months=months)
                if name == "patients":
                    frame["index_date"] = pd.Timestamp("2024-07-01") + pd.DateOffset(months=months)
                parts[name].append(frame)
    for name in raw:
        table = raw[name] if not parts[name] else pd.concat(parts[name], ignore_index=True)
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    config["data"].update(input_dir=str(tmp_path), as_of_date="2024-02-15")
    config["timeline"].update(index_date_start="2023-01-01", index_date_end="2024-01-01",
                              index_date_strategy="provided", snapshots_per_patient=1)
    config["splitting"].update(train_end="2023-05-01", validation_end="2023-11-01")
    config["delivery"] = yaml.safe_load(Path("configs/delivery_real.example.yaml").read_text())["delivery"]
    config["delivery"].update(scoring_date="2024-02-15", output_dir=str(tmp_path / "out"),
                              artifact_dir=str(tmp_path / "artifacts"))
    config["delivery"]["hcp"]["relevant_specialties"] = ["neurology"]
    result = run_delivery(config)
    assert result["status"] == "COMPLETED"
    assert result["eligible_patients"] == 4
    assert result["priority_patients"] == 1
    assert result["hcp_targets"] == 1
    targets = pd.read_csv(result["hcp_csv"])
    assert "patient_id" not in targets
    assert Path(result["html_report"]).name == "field_report.html"
    directory = Path(result["model_artifact"]).parent
    comparison = pd.read_csv(directory / "baseline_comparison.csv")
    assert len(comparison) == 4
    assert set(comparison.split) == {"validation", "test"}
    audit = json.loads((directory / "training_audit.json").read_text())
    assert audit["reload_verified"]
    assert audit["split_counts"]["train"]["patients"] == 8
    assert (directory / "raw_profile.json").exists()


def test_setup_migrates_paths_and_preserves_rules(tmp_path):
    import importlib.util
    import shutil

    import yaml

    spec = importlib.util.spec_from_file_location("setup_real_data", "setup_real_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "configs").mkdir()
    for file in ("real_data.example.yaml", "delivery_real.example.yaml"):
        shutil.copy(Path("configs") / file, tmp_path / "configs" / file)
    module.setup(tmp_path)
    path = tmp_path / "configs/private/benchmark.yaml"
    saved = yaml.safe_load(path.read_text())
    saved["cohort"]["diagnosis_codes"] = ["REVIEWED_CODE"]
    saved["data"]["input_dir"] = "old/path"
    path.write_text(yaml.safe_dump(saved))
    module.setup(tmp_path)
    new = yaml.safe_load(path.read_text())
    assert new["data"]["input_dir"] == "../../actual_raw_data"
    assert new["cohort"]["diagnosis_codes"] == ["REVIEWED_CODE"]
    assert path.with_suffix(".yaml.bak").exists()
    assert list((tmp_path / "actual_raw_data").iterdir()) == []
