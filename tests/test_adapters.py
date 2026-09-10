import importlib
import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

from therapy_switch.data.prepared_input_adapter import load_prepared_inputs
from therapy_switch.data.raw_source_adapter import canonicalize_raw_tables
from therapy_switch.data.snowflake_adapter import INTEGRATION_STATUS, SnowflakeAdapter, capabilities
from therapy_switch.io import _read_frame
from therapy_switch.pipeline import prepare_inputs
from therapy_switch.schemas import SchemaError


def test_prepared_input_matches_raw_without_reengineering(
    prepared_data, prepared_config, monkeypatch
):
    _, _, original = prepared_data
    frames = {name: getattr(original, name).copy() for name in ("snapshots", "wide", "events")}
    monkeypatch.setattr(
        "therapy_switch.pipeline.prepare_raw_inputs",
        lambda *args: pytest.fail("Prepared data must bypass raw preparation"),
    )
    _, loaded = prepare_inputs(prepared_config, tables=frames)
    pd.testing.assert_frame_equal(loaded.modeling_frame(), original.modeling_frame())
    pd.testing.assert_frame_equal(loaded.events, original.events, check_like=True)


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_prepared_file_roundtrip_with_uppercase_aliases(
    prepared_data, prepared_config, tmp_path, extension
):
    _, _, original = prepared_data
    frames = {}
    for name in ("snapshots", "wide", "events"):
        frame = getattr(original, name).copy()
        frame.attrs = {}
        frame.columns = frame.columns.str.upper()
        path = tmp_path / f"{name}.{extension}"
        if extension == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False)
        frames[name] = _read_frame(path, extension)
    loaded = load_prepared_inputs(frames, prepared_config)
    pd.testing.assert_frame_equal(
        loaded.modeling_frame(), original.modeling_frame(), check_dtype=False
    )
    assert loaded.feature_columns == original.feature_columns


def test_raw_mapping_preserves_identifier_zeros(prepared_data):
    config, tables, _ = prepared_data
    frames = {name: frame.copy() for name, frame in tables.items()}
    ids = {value: f"{i:010d}" for i, value in enumerate(frames["patients"].patient_id)}
    for frame in frames.values():
        if "patient_id" in frame:
            frame["patient_id"] = frame.patient_id.map(ids)
        frame.columns = frame.columns.str.upper()
    normalized = canonicalize_raw_tables(frames, config)
    assert normalized["patients"].patient_id.iloc[0] == "0000000000"


def test_effective_mapping_overlap_fails(prepared_data):
    config, tables, _ = prepared_data
    frames = dict(tables)
    frames["therapy_mapping"] = pd.concat(
        [tables["therapy_mapping"], tables["therapy_mapping"].iloc[:1]]
    )
    with pytest.raises(SchemaError):
        canonicalize_raw_tables(frames, config)


def test_no_snowflake_required_and_no_implicit_session(monkeypatch):
    monkeypatch.setitem(sys.modules, "snowflake", None)
    import therapy_switch.data.snowflake_adapter as module

    importlib.reload(module)
    assert capabilities()["environment_verification"] == "NOT VERIFIED IN SENTINEL"
    assert capabilities()["snowpark_installed"] is False
    with pytest.raises(ValueError, match="Supply a session"):
        SnowflakeAdapter()


def test_supplied_snowpark_session_is_bounded_and_identifiers_validated():
    session = MagicMock()
    session.table.return_value.limit.return_value.to_pandas.return_value = pd.DataFrame(
        {"A": [1, 2]}
    )
    adapter = SnowflakeAdapter(session, max_rows=2)
    frames = adapter.read_tables({"patients": "YOUR_DATABASE.YOUR_SCHEMA.YOUR_TABLE"})
    assert len(frames["patients"]) == 2
    session.table.return_value.limit.assert_called_with(3)
    with pytest.raises(ValueError, match="identifier"):
        adapter.read_tables({"patients": "TABLE; DROP TABLE OTHER"})
    session.table.return_value.limit.return_value.to_pandas.return_value = pd.DataFrame(
        {"A": [1, 2, 3]}
    )
    with pytest.raises(ValueError, match="exceeds"):
        adapter.read_tables({"patients": "YOUR_TABLE"})
    assert INTEGRATION_STATUS["implementation"] == "IMPLEMENTED"


def test_active_session_is_explicit_and_mocked(monkeypatch):
    import types

    context = types.ModuleType("snowflake.snowpark.context")
    sentinel = object()
    context.get_active_session = lambda: sentinel
    monkeypatch.setitem(sys.modules, "snowflake.snowpark.context", context)
    assert SnowflakeAdapter(use_active_session=True).session is sentinel
