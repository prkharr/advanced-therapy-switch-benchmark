"""Selection can operate while test partitions are absent."""

import json

import numpy as np
import pandas as pd
import pytest

from therapy_switch.research import (
    confirm_frozen_recipe,
    digest,
    run_validation_search,
    select_frozen_recipe,
    write_json,
)


def dataset(root, seed):
    folder = root / "datasets" / str(seed)
    folder.mkdir(parents=True)
    for split, count in [("train", 40), ("validation", 20)]:
        y = np.tile([0, 0, 0, 1], count // 4)
        data = pd.DataFrame(
            {
                "prior_count": y + np.random.default_rng(seed).normal(0, 0.4, count),
                "label": y,
                "patient_id": [f"{split}_{i}" for i in range(count)],
            }
        )
        data.to_parquet(folder / f"{split}.parquet", index=False)
    write_json(
        folder / "ready.json",
        {
            "seed": seed,
            "features": ["prior_count"],
            "hashes": {s: digest(folder / f"{s}.parquet") for s in ("train", "validation")},
        },
    )
    return folder


def test_validation_search_never_requires_test_file(tmp_path):
    folder = dataset(tmp_path, 17)
    bank = [
        {
            "name": "logistic_reference",
            "family": "logistic_regression",
            "options": {"C": 0.2, "max_iter": 100},
        }
    ]
    rows = run_validation_search(folder, bank, tmp_path / "discovery" / "17")
    assert rows[0]["status"] == "COMPLETED"
    assert not (folder / "test.parquet").exists()
    with open(folder / "train.parquet", "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        run_validation_search(folder, bank, tmp_path / "other")


def test_frozen_selection_requires_independent_seeds_and_guards_recipe(tmp_path, monkeypatch):
    bank = [
        {"name": "neural", "family": "tabm", "options": {}},
        {"name": "lightgbm_reference", "family": "lightgbm", "options": {}},
        {"name": "logistic_reference", "family": "logistic_regression", "options": {}},
    ]
    for seed in (11, 12):
        folder = dataset(tmp_path, seed)
        y = pd.read_parquet(folder / "validation.parquet").label.to_numpy()
        for spec in bank:
            out = tmp_path / "discovery" / str(seed) / spec["name"]
            out.mkdir(parents=True)
            score = 0.1 + 0.8 * y if spec["name"] == "neural" else np.repeat(0.25, len(y))
            np.save(out / "validation_scores.npy", score)
            write_json(
                out / "result.json",
                {
                    **spec,
                    "status": "COMPLETED",
                    "validation_ap": 1.0 if spec["name"] == "neural" else 0.25,
                },
            )
    for seed in (21, 22):
        dataset(tmp_path, seed)
    with pytest.raises(ValueError, match="independent"):
        select_frozen_recipe(tmp_path, bank, [11, 12], [12, 13])
    recipe = select_frozen_recipe(tmp_path, bank, [11, 12], [21, 22])
    assert recipe["selected"]["members"][0]["name"] == "neural"
    assert set(recipe["dataset_manifests"]) == {"11", "12", "21", "22"}
    with pytest.raises(FileExistsError, match="already frozen"):
        select_frozen_recipe(tmp_path, bank, [11, 12], [21, 22])
    with monkeypatch.context() as patch:
        patch.setattr("therapy_switch.research.source_fingerprint", lambda: "changed")
        with pytest.raises(ValueError, match="source changed"):
            confirm_frozen_recipe(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr("therapy_switch.research.runtime_versions", lambda: {})
        with pytest.raises(ValueError, match="Runtime versions changed"):
            confirm_frozen_recipe(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(
            "therapy_switch.research.checkpoint_manifest", lambda _: {"weights": "changed"}
        )
        with pytest.raises(ValueError, match="checkpoint changed"):
            confirm_frozen_recipe(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr("therapy_switch.research.dataset_manifests", lambda *_: {})
        with pytest.raises(ValueError, match="Dataset manifest changed"):
            confirm_frozen_recipe(tmp_path)
    path = tmp_path / "frozen_recipe.json"
    parsed = json.loads(path.read_text())
    parsed["selected"]["weights"] = [0]
    write_json(path, parsed)
    with pytest.raises(ValueError, match="hash mismatch"):
        confirm_frozen_recipe(tmp_path)


def test_confirmation_fits_every_cohort_before_test_read(tmp_path, monkeypatch):
    from sklearn.dummy import DummyClassifier

    from therapy_switch.research import dataset_manifests, runtime_versions, source_fingerprint

    members = [{"name": "neural", "family": "tabm", "options": {}}]
    refs = [
        {"name": "lightgbm_reference", "family": "lightgbm", "options": {}},
        {"name": "logistic_reference", "family": "logistic_regression", "options": {}},
    ]
    for seed in (1, 2, 3):
        folder = dataset(tmp_path, seed)
        test = pd.read_parquet(folder / "validation.parquet")
        test["snapshot_id"] = [f"s{i}" for i in range(len(test))]
        test.to_parquet(folder / "test.parquet", index=False)
        meta = json.loads((folder / "ready.json").read_text())
        meta["hashes"]["test"] = digest(folder / "test.parquet")
        write_json(folder / "ready.json", meta)
    recipe = {
        "source_fingerprint": source_fingerprint(),
        "runtime_versions": runtime_versions(),
        "checkpoints": {},
        "dataset_manifests": dataset_manifests(tmp_path, [1, 2, 3]),
        "development_seeds": [1],
        "confirmation_seeds": [2, 3],
        "selected": {"members": members, "weights": [1]},
        "original_references": refs,
        "tuned_references": dict(zip(["lightgbm", "logistic_regression"], refs)),
        "bootstrap_iterations": 100,
        "family_alpha": 0.05,
        "minimum_gain": 0.02,
    }
    path = tmp_path / "frozen_recipe.json"
    write_json(path, recipe)
    write_json(path.with_suffix(".sha256.json"), {"sha256": digest(path)})
    fitted = []

    def fake_fit(spec, train, validation, features, *, seed):
        fitted.append((seed, spec["name"]))
        return DummyClassifier(strategy="prior").fit(train[features], train.label)

    original_read = pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        if str(path).endswith("test.parquet"):
            assert len(fitted) == 6
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr("therapy_switch.research.fit_candidate", fake_fit)
    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    report = confirm_frozen_recipe(tmp_path)
    assert not report["passes"]
    assert len(report["cohorts"]) == 2
    assert confirm_frozen_recipe(tmp_path) == report
    assert len(fitted) == 6
