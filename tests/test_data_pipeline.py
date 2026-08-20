from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from therapy_switch.data import (
    build_cohort,
    build_event_sequences,
    generate_synthetic_claims,
    stratified_patient_split,
    temporal_patient_split,
    validate_patient_disjoint,
)
from therapy_switch.features import build_tabular_features


@pytest.fixture(scope="module")
def pipeline_data() -> tuple[dict, dict[str, pd.DataFrame], pd.DataFrame]:
    config = {
        "project": {"random_seed": 19},
        "data": {
            "synthetic": {
                "n_patients": 120,
                "n_providers": 35,
                "target_prevalence": 0.10,
                "signal_strength": 0.8,
                "noise_scale": 1.0,
            }
        },
        "timeline": {"observation_window_days": 365, "prediction_window_days": 90},
        "therapy_mapping": {
            "conventional": ["CONV_A", "CONV_B", "CONV_C"],
            "advanced": ["ADV_A", "ADV_B"],
        },
        "cohort": {"require_conventional_on_index": True},
        "splitting": {"validation_fraction": 0.15, "test_fraction": 0.20},
        "features": {
            "rolling_windows_days": [30, 60, 90, 180, 365],
            "sequence": {"max_events": 24},
        },
    }
    tables = generate_synthetic_claims(config)
    cohort = build_cohort(tables, config)
    return config, tables, cohort


def test_synthetic_claims_follow_contract_and_prevalence(pipeline_data) -> None:
    config, tables, cohort = pipeline_data
    assert set(tables) == {"patients", "medical_claims", "pharmacy_claims", "providers"}
    assert len(tables["patients"]) == 120
    assert not tables["patients"]["patient_id"].duplicated().any()
    assert {
        "claim_id",
        "patient_id",
        "claim_date",
        "diagnosis_code",
        "procedure_code",
        "provider_id",
        "place_of_service",
    }.issubset(tables["medical_claims"].columns)
    assert {
        "claim_id",
        "patient_id",
        "fill_date",
        "drug_id",
        "therapy_class",
        "quantity",
        "days_supply",
        "prescriber_id",
    }.issubset(tables["pharmacy_claims"].columns)
    assert cohort["label"].mean() == pytest.approx(
        config["data"]["synthetic"]["target_prevalence"], abs=0.01
    )


def test_cohort_outcomes_are_strictly_future_and_in_window(pipeline_data) -> None:
    config, tables, cohort = pipeline_data
    positive = cohort.loc[cohort["label"].eq(1)]
    assert not positive.empty
    assert (positive["outcome_date"] > positive["index_date"]).all()
    assert (positive["outcome_date"] <= positive["index_date"] + pd.Timedelta(days=90)).all()

    pharmacy = tables["pharmacy_claims"]
    index_claims = pharmacy.merge(
        cohort[["patient_id", "index_date"]], on="patient_id", validate="many_to_one"
    )
    conventional = index_claims["drug_id"].isin(config["therapy_mapping"]["conventional"])
    assert set(cohort["patient_id"]) == set(
        index_claims.loc[
            conventional & index_claims["fill_date"].eq(index_claims["index_date"]), "patient_id"
        ]
    )


def test_preindex_advanced_therapy_excludes_patient(pipeline_data) -> None:
    config, tables, cohort = pipeline_data
    patient = cohort.iloc[0]
    mutated = {name: frame.copy() for name, frame in tables.items()}
    extra = mutated["pharmacy_claims"].iloc[[0]].copy()
    extra["claim_id"] = "RX_PRIOR_ADVANCED_TEST"
    extra["patient_id"] = patient["patient_id"]
    extra["fill_date"] = patient["index_date"] - pd.Timedelta(days=1)
    extra["drug_id"] = "ADV_A"
    extra["therapy_class"] = "advanced"
    mutated["pharmacy_claims"] = pd.concat([mutated["pharmacy_claims"], extra], ignore_index=True)
    rebuilt = build_cohort(mutated, config)
    assert patient["patient_id"] not in set(rebuilt["patient_id"])


def test_tabular_features_include_windows_recency_and_trends(pipeline_data) -> None:
    config, tables, cohort = pipeline_data
    features = build_tabular_features(tables, cohort, config)
    assert len(features) == len(cohort)
    assert features["patient_id"].is_unique
    expected = {
        "diagnosis_count_30d",
        "diagnosis_count_60d",
        "specialist_visits_90d",
        "rx_claims_180d",
        "procedure_count_365d",
        "days_since_last_diagnosis",
        "days_since_last_rx",
        "days_since_last_specialist_visit",
        "days_since_last_treatment_change",
        "visits_last_30d_over_previous_30d",
        "diagnoses_last_90d_minus_previous_90d",
        "rx_frequency_recent_vs_historical",
        "specialist_visit_acceleration",
        "utilization_trend",
    }
    assert expected.issubset(features.columns)
    numeric = features.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_patient_safe_random_and_temporal_splits(pipeline_data) -> None:
    config, _, cohort = pipeline_data
    random_splits = stratified_patient_split(cohort, config)
    validate_patient_disjoint(random_splits)
    assert sum(len(split) for split in random_splits.values()) == len(cohort)
    overall = cohort["label"].mean()
    for split in random_splits.values():
        assert abs(split["label"].mean() - overall) < 0.08

    temporal_splits = temporal_patient_split(cohort, config)
    validate_patient_disjoint(temporal_splits)
    assert (
        temporal_splits["train"]["index_date"].max()
        <= temporal_splits["validation"]["index_date"].min()
    )
    assert (
        temporal_splits["validation"]["index_date"].max()
        <= temporal_splits["test"]["index_date"].min()
    )


def test_event_sequences_are_padded_recent_and_preindex(pipeline_data) -> None:
    config, tables, cohort = pipeline_data
    sequences = build_event_sequences(tables, cohort, config)
    assert len(sequences) == len(cohort)
    assert sequences.event_type_ids.shape == (len(cohort), 24)
    assert np.array_equal(sequences.attention_mask.sum(axis=1), sequences.lengths)
    assert (sequences.lengths > 0).all()
    valid_dates = sequences.event_dates[sequences.attention_mask]
    repeated_index = np.repeat(sequences.index_dates[:, None], 24, axis=1)
    assert (valid_dates <= repeated_index[sequences.attention_mask]).all()
    assert sequences.events["days_before_index"].between(0, 365).all()
    model_split = sequences.to_sequence_split().validated(expected_rows=len(cohort))
    assert model_split.values.shape == (len(cohort), 24, 5)
