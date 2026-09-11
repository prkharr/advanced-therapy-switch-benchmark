"""Small hand-specified contract cases; no input dataset is distributed."""

from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture
def config():
    c = yaml.safe_load(Path("configs/real_data.example.yaml").read_text())
    c["data"].update(as_of_date="2024-08-31", extract_version="unit_case", input_dir="unused")
    c["timeline"].update(observation_window_days=365, prediction_window_days=30,
                         minimum_history_days=30, minimum_followup_days=30,
                         claims_lag_days=2, label_runout_days=1,
                         index_date_start="2024-07-01", index_date_end="2024-07-01")
    c["cohort"].update(cohort_id="unit_case", require_diagnosis_confirmation=True,
                      diagnosis_codes=["DX"], confirmation_codes=["DX"],
                      diagnosis_separation_days=5, coverage_window_days=30,
                      minimum_covered_days=10, require_conventional_on_index=False)
    c["features"]["specialist_specialties"] = ["neurology"]
    c["source_definitions"] = dict.fromkeys(c["source_definitions"], "Hand-specified contract case")
    return c


@pytest.fixture
def raw():
    from therapy_switch.schemas import CANONICAL_SCHEMAS

    records = {
        "patients": [["p1", 1980, "F", "g", "2023-01-01", "2024-12-31"],
                     ["p2", 1970, "M", "g", "2023-01-01", "2024-12-31"]],
        "medical_claims": [
            ["m1", "p1", "2024-06-05", "2024-06-06", "final", "DX", None, "h1", "office"],
            ["m2", "p1", "2024-06-15", "2024-06-16", "final", "DX", None, "h1", "office"],
            ["m3", "p2", "2024-06-05", "2024-06-06", "final", "DX", None, "h1", "office"],
            ["m4", "p2", "2024-06-15", "2024-06-16", "final", "DX", None, "h1", "office"],
        ],
        "pharmacy_claims": [
            ["r1", "p1", "2024-06-01", "2024-06-02", "paid", "00000000001", "conventional", 90, 90, "h1", "a", 1],
            ["r2", "p2", "2024-06-01", "2024-06-02", "paid", "00000000001", "conventional", 90, 90, "h1", "a", 1],
            ["r3", "p1", "2024-07-10", "2024-07-11", "paid", "00000000002", "advanced", 30, 30, "h1", "a", 2],
        ],
        "providers": [["h1", "neurology", "g", "o"]],
        "plans": [["a", "commercial"]],
        "enrollment": [["p1", "2023-01-01", "2024-12-31", "a"],
                       ["p2", "2023-01-01", "2024-12-31", "a"]],
        "therapy_mapping": [["00000000001", "conventional", "2020-01-01", "2030-01-01"],
                            ["00000000002", "advanced", "2020-01-01", "2030-01-01"]],
    }
    return {name: pd.DataFrame(rows, columns=CANONICAL_SCHEMAS[name].required_columns)
            for name, rows in records.items()}
