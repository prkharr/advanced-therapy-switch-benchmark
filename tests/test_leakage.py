from __future__ import annotations

import pandas as pd
import pytest

from therapy_switch.data import build_cohort, generate_synthetic_claims
from therapy_switch.features import (
    LeakageError,
    assert_no_post_index_events,
    audit_feature_frame,
    build_tabular_features,
)


@pytest.fixture(scope="module")
def leakage_data():
    config = {
        "data": {"synthetic": {"n_patients": 55, "target_prevalence": 0.1}},
        "project": {"random_seed": 5},
        "timeline": {"observation_window_days": 365, "prediction_window_days": 90},
        "therapy_mapping": {
            "conventional": ["CONV_A", "CONV_B"],
            "advanced": ["ADV_A"],
        },
    }
    tables = generate_synthetic_claims(config)
    cohort = build_cohort(tables, config)
    return config, tables, cohort


def test_post_index_claims_do_not_change_features(leakage_data) -> None:
    config, tables, cohort = leakage_data
    expected = build_tabular_features(tables, cohort, config)
    future = cohort[["patient_id", "index_date"]].copy()
    future["claim_id"] = [f"FUTURE{i:05d}" for i in range(len(future))]
    future["claim_date"] = future["index_date"] + pd.Timedelta(days=10)
    future["diagnosis_code"] = "DX_SEVERE"
    future["procedure_code"] = "PROC_INFUSION_EVAL"
    future["provider_id"] = tables["providers"].iloc[0]["provider_id"]
    future["place_of_service"] = "inpatient"
    future = future[tables["medical_claims"].columns]
    mutated = {name: frame.copy() for name, frame in tables.items()}
    mutated["medical_claims"] = pd.concat([mutated["medical_claims"], future], ignore_index=True)
    actual = build_tabular_features(mutated, cohort, config)
    pd.testing.assert_frame_equal(actual, expected)


def test_post_index_event_audit_raises(leakage_data) -> None:
    _, _, cohort = leakage_data
    event = pd.DataFrame(
        {
            "patient_id": [cohort.iloc[0]["patient_id"]],
            "event_date": [cohort.iloc[0]["index_date"] + pd.Timedelta(days=1)],
        }
    )
    with pytest.raises(LeakageError, match="post-index"):
        assert_no_post_index_events(event, cohort, date_col="event_date")


def test_structural_audit_rejects_outcome_features(leakage_data) -> None:
    _, tables, cohort = leakage_data
    features = build_tabular_features(tables, cohort, {})
    features["future_advanced_therapy"] = cohort["label"].to_numpy()
    report = audit_feature_frame(features, cohort, raise_on_error=False)
    assert "forbidden_feature_name" in set(report["check"])
    with pytest.raises(LeakageError):
        audit_feature_frame(features, cohort)
