"""Capacity evaluation and history leakage regression tests."""

import joblib
import numpy as np
import pandas as pd
import pytest

from therapy_switch.features.history import HistoryFeatureEncoder, eligible_events
from therapy_switch.models.contracts import LeakageError
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


@pytest.mark.parametrize(
    "column,value",
    [
        ("event_date", "2024-01-26"),
        ("available_date", "2024-02-02"),
        ("event_date", "2023-01-01"),
        ("available_date", None),
    ],
)
def test_history_rejects_unavailable_events(assessments, events, column, value):
    events.loc[events.index[0], column] = pd.Timestamp(value) if value else pd.NaT
    with pytest.raises(LeakageError):
        eligible_events(events, assessments.iloc[1:])


def test_history_vocabulary_train_only_and_serializable(assessments, events, tmp_path):
    train = assessments.iloc[1:-1]
    encoder = HistoryFeatureEncoder().fit(events, train)
    assert encoder.vocabulary_ == ["diagnosis::known"]
    expected = encoder.transform(events, assessments.iloc[-1:])
    assert expected["history_code_0_count_30d"].item() == 0
    path = tmp_path / "history.joblib"
    joblib.dump(encoder, path)
    pd.testing.assert_frame_equal(
        joblib.load(path).transform(events, assessments.iloc[-1:]), expected
    )
    changed = assessments.copy()
    changed["label"] = 1 - changed.label
    pd.testing.assert_frame_equal(
        encoder.transform(events, train), encoder.transform(events, changed.iloc[1:-1])
    )


@pytest.mark.parametrize(
    "family,pretrain", [("rank_mlp", 0), ("ehr_transformer", 1), ("retain", 0)]
)
def test_neural_roundtrip_and_unlabelled_scoring(assessments, events, tmp_path, family, pretrain):
    pytest.importorskip("torch")
    from therapy_switch.models.patient_rank import fit_patient_model

    frame = assessments.iloc[1:].copy()
    spec = {
        "name": family,
        "family": family,
        "history": True,
        "latest_only": True,
        "options": {
            "width": 8,
            "max_length": 8,
            "max_epochs": 2,
            "pretrain_epochs": pretrain,
            "batch_size": 10,
            "rank_weight": 1.0,
            "patience": 2,
        },
    }
    model = fit_patient_model(spec, frame, frame, events, events, ["x"], seed=1)
    expected = model.predict_scores(frame.drop(columns="label"), events)
    assert np.isfinite(expected).all() and np.all((expected >= 0) & (expected <= 1))
    path = tmp_path / f"{family}.joblib"
    joblib.dump(model, path)
    actual = joblib.load(path).predict_scores(frame, events)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    # Prediction of one patient cannot depend on other patients in the batch.
    single = model.predict_scores(frame.iloc[[0]], events)[0]
    assert single == pytest.approx(expected[0], abs=1e-6)
