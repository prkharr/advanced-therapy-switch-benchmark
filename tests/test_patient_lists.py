"""Capacity evaluation and history leakage regression tests."""

import numpy as np
import pandas as pd
import pytest

from therapy_switch.patient_lists import (
    latest_patient_indices,
    patient_capture_metrics,
    rank_patients,
)


@pytest.fixture
def assessments():
    return pd.DataFrame(
        {
            "patient_id": ["a", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "snapshot_id": [f"s{i}" for i in range(11)],
            "index_date": pd.to_datetime(["2024-01-01"] + ["2024-02-01"] * 10),
            "feature_cutoff": pd.Timestamp("2024-01-25"),
            "lookback_start": pd.Timestamp("2023-02-01"),
            "label": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            "x": np.arange(11) / 10,
        }
    )


@pytest.fixture
def events(assessments):
    frame = assessments.iloc[1:]
    return pd.DataFrame(
        {
            "snapshot_id": frame.snapshot_id.to_numpy(),
            "event_id": [f"e{i}" for i in range(10)],
            "event_date": pd.Timestamp("2024-01-10"),
            "available_date": pd.Timestamp("2024-01-12"),
            "event_type": "diagnosis",
            "code_token": ["known"] * 9 + ["heldout"],
        }
    )


def test_list_uses_latest_assessment_before_ranking(assessments):
    scores = np.array([0.999, 0.1, 0.9] + [0.2] * 8)
    ranked = rank_patients(assessments, scores)
    assert len(ranked) == 10 and ranked.selected.sum() == 1
    assert ranked.iloc[0].patient_id == "b"
    assert ranked.loc[ranked.patient_id == "a", "snapshot_id"].item() == "s1"
    metrics = patient_capture_metrics(assessments, scores)
    assert metrics["positives"] == 2 and metrics["captured"] == 1
    assert metrics["recall"] == 0.5 and metrics["precision"] == 1
    changed = assessments.copy()
    changed["label"] = 1 - changed.label
    assert np.array_equal(latest_patient_indices(assessments), latest_patient_indices(changed))


def test_cutoff_ties_and_ambiguous_assessments(assessments):
    ranked = rank_patients(assessments, np.ones(11))
    assert ranked.iloc[0].patient_id == "a"
    assert latest_patient_indices(assessments, as_of="2024-01-15").tolist() == [0]
    duplicate = pd.concat([assessments, assessments.iloc[[1]].assign(snapshot_id="other")])
    with pytest.raises(ValueError, match="Ambiguous"):
        latest_patient_indices(duplicate)
