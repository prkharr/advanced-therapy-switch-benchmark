"""Regression checks for patient-level confirmation and the test-data barrier."""

import json

import numpy as np
import pandas as pd
import pytest

from therapy_switch import patient_study
from therapy_switch.evaluation.patient_confirmation import confirm_patient_capture


def confirmation_frame():
    rng = np.random.default_rng(12)
    parts = []
    for cohort in [1, 2, 3]:
        y = np.r_[np.ones(40), np.zeros(360)]
        p = y * 0.7 + rng.random(400) * 0.3
        parts.append(
            pd.DataFrame(
                {
                    "cohort": cohort,
                    "patient_id": np.arange(400),
                    "label": y,
                    "candidate": p,
                    "baseline": rng.random(400),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_paired_patient_capture_detects_gain_and_zero_difference():
    frame = confirmation_frame()
    result = confirm_patient_capture(
        frame, candidate="candidate", references=["baseline"], n_bootstrap=200
    )
    assert result["passes"] and result["comparisons"][0]["ci_lower"] > 0
    frame["baseline"] = frame["candidate"]
    same = confirm_patient_capture(
        frame, candidate="candidate", references=["baseline"], n_bootstrap=200
    )
    assert not same["passes"]
    assert same["comparisons"][0]["ci_lower"] == same["comparisons"][0]["ci_upper"] == 0


def test_confirmation_rejects_duplicate_patients_and_invalid_probabilities():
    frame = confirmation_frame()
    with pytest.raises(ValueError, match="one assessment"):
        confirm_patient_capture(
            pd.concat([frame, frame.iloc[[0]]]),
            candidate="candidate",
            references=["baseline"],
            n_bootstrap=100,
        )
    frame.loc[0, "candidate"] = np.nan
    with pytest.raises(ValueError):
        confirm_patient_capture(
            frame, candidate="candidate", references=["baseline"], n_bootstrap=100
        )


def test_test_files_remain_unread_when_any_reserved_fit_is_missing(tmp_path, monkeypatch):
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text("{}")
    spec = {"name": "example", "family": "logistic_regression", "options": {}}
    recipe = {
        "protocol": {"fresh_confirmation_seeds": [1, 2]},
        "selected": {"members": [spec], "weights": [1]},
        "controls": {},
    }
    monkeypatch.setattr(patient_study, "check_recipe", lambda *args: recipe)
    folder = tmp_path / "confirmation" / "1"
    folder.mkdir(parents=True)
    (folder / "fits.json").write_text(
        json.dumps({"recipe_sha256": patient_study.digest(recipe_path), "fits": {}})
    )

    def forbidden(*args):
        pytest.fail("A test partition was read before all reserved fits were sealed")

    monkeypatch.setattr(patient_study, "load_partition", forbidden)
    with pytest.raises(ValueError, match="Every reserved fit"):
        patient_study.evaluate_confirmation(tmp_path, recipe_path)
    assert not (tmp_path / "confirmation" / "test_evaluation_started.json").exists()


def test_prepare_rejects_non_synthetic_sources_before_pipeline(tmp_path, monkeypatch):
    import therapy_switch.config

    monkeypatch.setattr(
        therapy_switch.config, "load_config", lambda *args: {"data": {"source": "prepared"}}
    )
    with pytest.raises(ValueError, match="only accepts synthetic"):
        patient_study.prepare(tmp_path / "cohort", 1)


def test_small_bank_freeze_fit_evaluate_and_replay(tmp_path, monkeypatch):
    from sklearn.dummy import DummyClassifier

    from therapy_switch.models.patient_rank import PatientModel
    from therapy_switch.patient_lists import patient_capture_metrics
    from therapy_switch.research import source_fingerprint

    # No optional learner is needed; this checks orchestration and artifact guards.
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-runtime")
    bank = [
        {"name": "lightgbm_reference", "family": "lightgbm", "options": {}},
        {"name": "logistic_reference", "family": "logistic_regression", "options": {}},
        {"name": "tuned_logistic", "family": "logistic_regression", "options": {"C": 0.1}},
        {"name": "ehr", "family": "ehr_transformer", "options": {}},
    ]
    for seed in [1, 2, 3, 4]:
        folder = tmp_path / "datasets" / str(seed)
        folder.mkdir(parents=True)
        hashes = {}
        for split in ["train", "validation", "test"]:
            frame = pd.DataFrame(
                {
                    "patient_id": [f"{split}_{i:03d}" for i in range(40)],
                    "snapshot_id": [f"{split}_{i:03d}" for i in range(40)],
                    "index_date": pd.Timestamp("2024-02-01"),
                    "feature_cutoff": pd.Timestamp("2024-01-25"),
                    "lookback_start": pd.Timestamp("2023-02-01"),
                    "label": np.tile([0, 0, 0, 1], 10),
                    "prior_count": np.arange(40),
                }
            )
            for name, data in [
                (split, frame),
                (split + "_events", pd.DataFrame({"snapshot_id": []})),
            ]:
                data.to_parquet(folder / f"{name}.parquet", index=False)
                hashes[name] = patient_study.digest(folder / f"{name}.parquet")
        (folder / "ready.json").write_text(
            json.dumps({"seed": seed, "features": ["prior_count"], "hashes": hashes})
        )
        if seed in [1, 2]:
            val = pd.read_parquet(folder / "validation.parquet")
            for spec in bank:
                out = tmp_path / "discovery" / str(seed) / spec["name"]
                out.mkdir(parents=True)
                p = np.full(len(val), 0.25)
                np.save(out / "validation_scores.npy", p)
                (out / "result.json").write_text(
                    json.dumps(
                        {
                            "spec": spec,
                            "status": "COMPLETED",
                            "implementation": patient_study.implementation_digest(),
                            "input_manifest": patient_study.digest(folder / "ready.json"),
                            "scores_sha256": patient_study.digest(out / "validation_scores.npy"),
                            "metrics": patient_capture_metrics(val, p),
                            "seconds": 0,
                            "origin": "unit fixture",
                        }
                    )
                )
    old_path = tmp_path / "previous.json"
    old_path.write_text(json.dumps({"selected": {"members": [bank[-1]], "weights": [1]}}))
    protocol = {
        "development_seeds": [1, 2],
        "fresh_confirmation_seeds": [3, 4],
        "capacity": 0.1,
        "material_gain": 0.03,
        "bootstrap_draws": 100,
        "family_alpha": 0.05,
    }
    recipe_path = tmp_path / "frozen_recipe.json"
    recipe = patient_study.freeze(tmp_path, bank, protocol, old_path, recipe_path)
    assert recipe["source_fingerprint"] == source_fingerprint()
    with pytest.raises(ValueError, match="cannot be overwritten"):
        patient_study.freeze(tmp_path, bank, protocol, old_path, recipe_path)
    reads = []
    original = pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        if str(path).endswith("test.parquet"):
            assert all(
                (tmp_path / "confirmation" / str(seed) / "fits.json").exists() for seed in [3, 4]
            )
            reads.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)

    def fake_fit(spec, train, validation, train_events, validation_events, features, *, seed):
        estimator = DummyClassifier(strategy="prior").fit(train[features], train.label)
        return PatientModel(features, estimator)

    monkeypatch.setattr(patient_study, "fit_patient_model", fake_fit)
    for seed in [3, 4]:
        patient_study.fit_confirmation(tmp_path, recipe_path, seed)
    assert reads == []
    report = patient_study.evaluate_confirmation(tmp_path, recipe_path)
    assert len(reads) == 2 and not report["passes"]
    assert patient_study.evaluate_confirmation(tmp_path, recipe_path) == report
    (tmp_path / "confirmation" / "3" / "lightgbm_reference.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Every reserved fit"):
        patient_study.evaluate_confirmation(tmp_path, recipe_path)
