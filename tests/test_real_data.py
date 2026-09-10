"""Real-mode boundaries tested with generated fixtures, never client records."""

import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_raw_export import Session, queries

from therapy_switch.config import load_config, validate_config
from therapy_switch.delivery import pipeline
from therapy_switch.delivery.raw_export import export_raw_data
from therapy_switch.io import load_claims_directory


@pytest.fixture
def actual_mode(prepared_data):
    config, raw, _ = prepared_data
    config = deepcopy(config)
    # Test-only code identifiers; these are not clinical mappings.
    def replace(value):
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        return value.replace("SYN_", "FIXTURE_") if isinstance(value, str) else value
    config = replace(config)
    config["data"].pop("synthetic", None)
    config["data"].update(kind="real", source="files", require_export_manifest=True,
                          extract_version="test_fixture_v1", file_format="csv")
    config["timeline"].update(index_date_start="2020-01-01", index_date_end="2022-01-01")
    config["source_definitions"] = dict.fromkeys(
        ["cohort", "target", "availability", "status", "observation", "therapy_mapping",
         "diagnosis_normalization"], "Explicitly generated fixture semantics for tests only",
    )
    config["delivery"] = {
        "scoring_date": "2024-12-01", "hcp": {"relevant_specialties": ["neurology"]},
    }
    raw = {k: v.replace(r"^SYN_", "FIXTURE_", regex=True) for k, v in raw.items()}
    return config, raw


def test_real_template_reports_missing_definitions():
    with pytest.raises(ValueError) as exc:
        load_config("configs/real_data.example.yaml")
    assert "data.as_of_date" in str(exc.value)
    assert "source_definitions.availability" in str(exc.value)
    assert "cohort.diagnosis_codes" in str(exc.value)


def test_real_mode_rejects_demo_codes(actual_mode):
    config, _ = actual_mode
    validate_config(config)
    config["cohort"]["diagnosis_codes"] = ["SYN_NT1"]
    with pytest.raises(ValueError, match="demonstration"):
        validate_config(config)


def test_verified_raw_identity_and_tampering(actual_mode, tmp_path):
    config, raw = actual_mode
    result = export_raw_data(Session(raw), config, queries(tmp_path), tmp_path / "raw")
    config["data"]["input_dir"] = result["raw_dir"]
    assert len(load_claims_directory(config)["patients"]) == len(raw["patients"])
    config["data"]["extract_version"] = "different"
    with pytest.raises(ValueError, match="version"):
        load_claims_directory(config)
    config["data"]["extract_version"] = "test_fixture_v1"
    path = Path(result["raw_dir"]) / "patients.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after export"):
        load_claims_directory(config)


def test_legacy_or_synthetic_model_rejected(actual_mode):
    config, _ = actual_mode
    model = pipeline.DeliveryModel(None, (), "2020-01-01", "experimental", None, "fixture")
    with pytest.raises(ValueError, match="trained in real-data"):
        model.predict(None, "2024-12-01", config=config)


def test_actual_setup_preserves_user_edits(tmp_path):
    import shutil

    shutil.copytree("configs", tmp_path / "configs", ignore=shutil.ignore_patterns("private"))
    shutil.copytree("sql/raw_extract", tmp_path / "sql/raw_extract")
    setup = runpy.run_path("setup_real_data.py")["setup"]
    setup(tmp_path)
    path = tmp_path / "configs/private/delivery.yaml"
    path.write_text("user changes")
    setup(tmp_path)
    assert path.read_text() == "user changes"
    assert len(list((tmp_path / "configs/private/sql").glob("*.sql"))) == 7


def test_extract_cli_connects_once_then_consumes_files(actual_mode, tmp_path, monkeypatch):
    config, raw = actual_mode
    config["extraction"] = {"sql_dir": str(queries(tmp_path)), "output_dir": str(tmp_path / "raw"),
                            "connection_name": "fixture"}
    session = Session(raw)
    closed = []
    session.close = lambda: closed.append(True)
    builder = SimpleNamespace(config=lambda *a: SimpleNamespace(create=lambda: session))
    monkeypatch.setitem(sys.modules, "snowflake.snowpark", SimpleNamespace(Session=SimpleNamespace(builder=builder)))
    monkeypatch.setattr(pipeline, "load_delivery_config", lambda *a, **k: deepcopy(config))
    seen = []
    def run(loaded, **kwargs):
        seen.append(loaded)
        load_claims_directory(loaded)
        return {"status": "COMPLETED"}
    monkeypatch.setattr(pipeline, "run_delivery", run)
    assert pipeline.main(["--extract", "--check"]) == 0
    assert session.calls == [] and seen == []
    assert pipeline.main(["--extract"]) == 0
    assert len(session.calls) == 7 and closed == [True]
    manifest = json.loads((Path(seen[0]["data"]["input_dir"]) / "export_manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and manifest["data_kind"] == "real"


def test_no_config_does_not_start_synthetic_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "run_delivery", lambda *a, **k: pytest.fail("must not run"))
    assert pipeline.main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_schema_inspection_reads_only_metadata(tmp_path):
    import pandas as pd

    inspect = runpy.run_path("inspect_snowflake.py")["inspect_sources"]
    calls = []
    def sql(query, params):
        calls.append((query, params))
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame({
            "TABLE_NAME": ["MEDICAL_EVENTS_LATEST"], "COLUMN_NAME": ["PATIENT_ID"],
        }))
    path = inspect(SimpleNamespace(sql=sql), "EXAMPLE_DB", "claims", tmp_path)
    assert path.exists() and len(calls) == 1
    assert "INFORMATION_SCHEMA.COLUMNS" in calls[0][0]
    assert calls[0][1][0] == "CLAIMS"
    with pytest.raises(ValueError, match="identifier"):
        inspect(SimpleNamespace(sql=sql), "DB; DROP TABLE x", "claims", tmp_path)
    assert len(calls) == 1
