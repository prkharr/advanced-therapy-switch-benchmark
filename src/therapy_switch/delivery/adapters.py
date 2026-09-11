"""Built-in extraction stages; custom extractors return the same DeliveryInputs."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pandas as pd

from therapy_switch.data.cohort import build_cohort
from therapy_switch.data.event_sequences import build_event_frame
from therapy_switch.data.raw_source_adapter import canonicalize_raw_tables, prepare_raw_inputs
from therapy_switch.features.feature_engineering import build_tabular_features
from therapy_switch.io import load_claims_directory
from therapy_switch.profiling import raw_profile

from .contracts import DeliveryInputs, ScoringBatch


def _table_hash(frame):
    hashed = hashlib.sha256()
    hashed.update("|".join(map(str, frame.columns)).encode())
    hashed.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return hashed.hexdigest()


def _raw_tables(config, session):
    del session
    if config["data"]["source"] != "files":
        raise ValueError("Load the seven real-data CSV files from actual_raw_data")
    return load_claims_directory(config)


def load_raw(config, *, include_training, expected_features=None, session=None):
    """Read seven raw tables once, then build mature training and live scoring batches."""
    raw = _raw_tables(config, session)
    raw_csv_dir = str(Path(config["data"]["input_dir"]).resolve())
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
            "raw_profile": raw_profile(canonical),
            "raw_hashes": {name: _table_hash(frame) for name, frame in canonical.items()},
            "excluded_patient_counts": pd.Series([row["reason"] for row in exclusions], dtype=str)
            .value_counts()
            .to_dict(),
            "training_labels_mature_by": str(date.date()) if include_training else None,
            "scoring_uses_future_labels": False,
        },
    )
