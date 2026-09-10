"""Built-in extraction stages; custom extractors return the same DeliveryInputs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

from therapy_switch.data.cohort import build_cohort
from therapy_switch.data.event_sequences import build_event_frame
from therapy_switch.data.generate_synthetic_claims import generate_synthetic_claims
from therapy_switch.data.prepared_input_adapter import load_prepared_inputs
from therapy_switch.data.raw_source_adapter import canonicalize_raw_tables, prepare_raw_inputs
from therapy_switch.features.feature_engineering import build_tabular_features
from therapy_switch.io import _read_frame, load_claims_directory

from .contracts import DeliveryInputs, ScoringBatch


def _table_hash(frame):
    hashed = hashlib.sha256()
    hashed.update("|".join(map(str, frame.columns)).encode())
    hashed.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return hashed.hexdigest()


def _raw_tables(config, session):
    source = config["data"]["source"]
    if source == "synthetic":
        return generate_synthetic_claims(config)
    if source == "files":
        manifest = Path(config["data"]["input_dir"]) / "export_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["status"] != "COMPLETED":
            raise ValueError("Raw export is incomplete or failed; use a completed extract")
        return load_claims_directory(config)
    if source == "snowflake_raw":
        from therapy_switch.data.snowflake_adapter import SnowflakeAdapter

        settings = config["data"]["snowflake"]
        return SnowflakeAdapter(
            session,
            use_active_session=settings.get("use_active_session", False),
            max_rows=int(settings.get("max_rows_per_table", 1_000_000)),
        ).read_tables(settings["tables"])
    raise ValueError(
        "Raw delivery supports synthetic, files, or explicitly configured snowflake_raw"
    )


def load_raw(config, *, include_training, expected_features=None, session=None):
    """Read seven raw tables once, then build mature training and live scoring batches."""
    raw = _raw_tables(config, session)
    raw_csv_dir = None
    if config["data"]["source"] == "synthetic" and config["delivery"].get("synthetic_csv_dir"):
        directory = Path(config["delivery"]["synthetic_csv_dir"])
        directory.mkdir(parents=True, exist_ok=False)
        for name, frame in raw.items():
            frame.to_csv(directory / f"{name}.csv", index=False)
        raw_csv_dir = str(directory.resolve())
    canonical = canonicalize_raw_tables(raw, config)
    date = pd.Timestamp(config["delivery"]["scoring_date"])
    if date > pd.Timestamp(config["data"]["as_of_date"]):
        raise ValueError("Scoring date exceeds the extract's as-of date")
    training = None
    if include_training:
        training_config = copy.deepcopy(config)
        training_config["data"]["as_of_date"] = str(date.date())
        _, training = prepare_raw_inputs(canonical, training_config)
        expected_features = training.feature_columns
    scoring_config = copy.deepcopy(config)
    scoring_config["data"]["as_of_date"] = str(date.date())
    scoring_config["timeline"].pop("index_date_start", None)
    scoring_config["timeline"].pop("index_date_end", None)
    current_tables = dict(canonical)
    current_tables["snapshot_candidates"] = pd.DataFrame(
        {"patient_id": canonical["patients"].patient_id, "index_date": date}
    )
    snapshots = build_cohort(current_tables, scoring_config, labelled=False)
    exclusions = snapshots.attrs.get("exclusions", [])
    if len(snapshots):
        wide = build_tabular_features(canonical, snapshots, scoring_config)
        columns = tuple(c for c in wide if c not in {"snapshot_id", "feature_as_of"})
    else:
        if expected_features is None:
            raise ValueError("An empty scoring cohort requires a supplied model feature schema")
        columns = tuple(expected_features)
        wide = pd.DataFrame(columns=["snapshot_id", "feature_as_of", *columns])
    events = build_event_frame(canonical, snapshots, scoring_config)
    batch = ScoringBatch(snapshots, wide, events, canonical["providers"], columns).validate(date)
    return DeliveryInputs(
        batch,
        training,
        {
            "source": config["data"]["source"],
            "raw_csv_dir": raw_csv_dir,
            "raw_counts": {name: len(frame) for name, frame in canonical.items()},
            "raw_hashes": {name: _table_hash(frame) for name, frame in canonical.items()},
            "excluded_patient_counts": pd.Series([row["reason"] for row in exclusions], dtype=str)
            .value_counts()
            .to_dict(),
            "training_labels_mature_by": str(date.date()) if include_training else None,
            "scoring_uses_future_labels": False,
        },
    )


def load_prepared(config, *, include_training, expected_features=None, session=None):
    """Example replacement: reviewed prepared local frames, without changing orchestration."""
    del expected_features, session
    settings = config["delivery"]["prepared"]
    extension = settings.get("format", "parquet")
    root = Path(settings["scoring_dir"])
    frames = {
        name: _read_frame(root / f"{name}.{extension}", extension)
        for name in ("snapshots", "wide", "events", "providers")
    }
    snapshots, wide, events = frames["snapshots"], frames["wide"], frames["events"]
    for name in ("index_date", "feature_cutoff", "lookback_start"):
        snapshots[name] = pd.to_datetime(snapshots[name])
    snapshots["eligible"] = snapshots.eligible.astype(str).str.lower().isin(["true", "1"])
    wide["feature_as_of"] = pd.to_datetime(wide.feature_as_of)
    features = tuple(config["data"]["feature_columns"])
    lineage = config["data"].get("feature_lineage", {})
    for column in features:
        item = lineage.get(column, {})
        if item.get("available_at_index") is not True or not item.get("definition"):
            raise ValueError(f"Reviewed feature lineage is required for {column}")
        if item.get("dtype") == "numeric":
            wide[column] = pd.to_numeric(wide[column], errors="raise")
    from therapy_switch.data.event_sequences import normalize_events

    events["event_date"] = pd.to_datetime(events.event_date)
    events["available_date"] = pd.to_datetime(events.available_date)
    events = events.drop(columns="index_date", errors="ignore").merge(
        snapshots[["snapshot_id", "index_date"]],
        on="snapshot_id",
        how="left",
        validate="many_to_one",
    )
    events = normalize_events(events)
    batch = ScoringBatch(snapshots, wide, events, frames["providers"], features).validate(
        config["delivery"]["scoring_date"]
    )
    training = None
    if include_training:
        root = Path(settings["training_dir"])
        train_frames = {
            name: _read_frame(root / f"{name}.{extension}", extension)
            for name in ("snapshots", "wide", "events")
        }
        training_config = copy.deepcopy(config)
        training_config["data"]["as_of_date"] = str(config["delivery"]["scoring_date"])
        training = load_prepared_inputs(train_frames, training_config)
    return DeliveryInputs(
        batch, training, {"source": "prepared local adapter", "scoring_uses_future_labels": False}
    )
