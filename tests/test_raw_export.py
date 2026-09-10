"""Raw export failure boundaries and compatibility with the file adapter."""

import json
from copy import deepcopy

import pytest

from therapy_switch.delivery.adapters import _raw_tables
from therapy_switch.delivery.raw_export import export_raw_data
from therapy_switch.schemas import CANONICAL_SCHEMAS


class Frame:
    def __init__(self, data):
        self.data = data.rename(columns=str.upper)
        self.columns = self.data.columns.tolist()

    def limit(self, count):
        return Frame(self.data.iloc[:count])

    def to_pandas_batches(self):
        for start in range(0, len(self.data), 17):
            yield self.data.iloc[start : start + 17]


class Session:
    def __init__(self, tables):
        self.tables = tables
        self.calls = []

    def sql(self, query):
        self.calls.append(query)
        return Frame(self.tables[query.split()[-1]])


def queries(tmp_path):
    directory = tmp_path / "sql"
    directory.mkdir()
    for name in CANONICAL_SCHEMAS:
        (directory / f"{name}.sql").write_text(f"SELECT * FROM {name};")
    return directory


def test_export_batches_reload_and_preserve_identifiers(prepared_data, tmp_path):
    config, raw, _ = prepared_data
    session = Session(raw)
    result = export_raw_data(session, config, queries(tmp_path), tmp_path / "raw")
    from pathlib import Path

    directory = Path(result["raw_dir"])
    manifest = json.loads((directory / "export_manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and len(session.calls) == 7
    file_config = deepcopy(config)
    file_config["data"].update(source="files", input_dir=str(directory), file_format="csv")
    loaded = _raw_tables(file_config, None)
    for name in CANONICAL_SCHEMAS:
        assert len(loaded[name]) == len(raw[name]) == manifest["tables"][name]["rows"]
        assert not (directory / f"{name}.csv.partial").exists()
    assert loaded["patients"].patient_id.tolist() == raw["patients"].patient_id.tolist()


def test_placeholder_rejected_before_query(prepared_data, tmp_path):
    config, raw, _ = prepared_data
    directory = queries(tmp_path)
    (directory / "patients.sql").write_text("SELECT * FROM {{PATIENTS_VIEW}};")
    session = Session(raw)
    with pytest.raises(ValueError, match="placeholders"):
        export_raw_data(session, config, directory, tmp_path / "raw")
    assert session.calls == [] and not (tmp_path / "raw").exists()


@pytest.mark.parametrize("failure", ["oversize", "bad_status", "missing_column"])
def test_failed_export_cannot_be_consumed(prepared_data, tmp_path, failure):
    config, raw, _ = prepared_data
    raw = {name: frame.copy() for name, frame in raw.items()}
    if failure == "bad_status":
        raw["medical_claims"]["status"] = "unmapped"
    if failure == "missing_column":
        raw["medical_claims"] = raw["medical_claims"].drop(columns="available_date")
    with pytest.raises(ValueError):
        export_raw_data(
            Session(raw), config, queries(tmp_path), tmp_path / "raw",
            max_rows=1 if failure == "oversize" else 1_000_000,
        )
    directory = next((tmp_path / "raw").iterdir())
    assert json.loads((directory / "export_manifest.json").read_text())["status"] == "FAILED"
    file_config = deepcopy(config)
    file_config["data"].update(source="files", input_dir=str(directory), file_format="csv")
    with pytest.raises(ValueError, match="incomplete or failed"):
        _raw_tables(file_config, None)


def test_raw_dir_cli_overrides_demo_source(monkeypatch, tmp_path):
    from therapy_switch.delivery import pipeline

    seen = {}
    monkeypatch.setattr(pipeline, "run_delivery", lambda config, **kwargs: seen.update(config) or {})
    assert pipeline.main(["--config", "configs/delivery_demo.yaml", "--raw-dir", str(tmp_path)]) == 0
    assert seen["data"]["source"] == "files"
    assert seen["data"]["input_dir"] == str(tmp_path.resolve())
    assert seen["data"]["file_format"] == "csv"
