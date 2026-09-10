"""Shared synthetic cohort; no proprietary fixtures."""

from copy import deepcopy

import pytest

from therapy_switch.config import load_config
from therapy_switch.data.generate_synthetic_claims import generate_synthetic_claims
from therapy_switch.data.raw_source_adapter import prepare_raw_inputs


@pytest.fixture(scope="session")
def prepared_data():
    config = load_config("configs/quickstart.yaml")
    config["data"]["synthetic"]["n_patients"] = 120
    config["data"]["synthetic"]["n_providers"] = 25
    config["data"]["persist_generated_data"] = False
    raw = generate_synthetic_claims(config)
    canonical, inputs = prepare_raw_inputs(raw, config)
    return config, canonical, inputs


@pytest.fixture
def prepared_config(prepared_data):
    config, _, inputs = prepared_data
    config = deepcopy(config)
    config["data"]["source"] = "prepared_files"
    config["data"]["feature_columns"] = list(inputs.feature_columns)
    config["data"]["feature_lineage"] = {
        c: {
            "available_at_index": True,
            "definition": f"Reviewed fixture feature {c}",
            "dtype": "numeric" if inputs.wide[c].dtype.kind in "iufb" else "categorical",
        }
        for c in inputs.feature_columns
    }
    return config
