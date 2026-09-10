"""Reproducible validation search and frozen-candidate synthetic confirmation.

Search reads train/validation files only. Confirmation evaluates one recipe on
predeclared independent cohorts after fitting every model without test inputs.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .evaluation.confirmation import confirm_average_precision
from .evaluation.metrics import top_fraction_metrics
from .models.advanced_tabular import ExternalTabularEstimator, TabMEstimator
from .models.classical import LightGBMRunner, LogisticRegressionRunner
from .models.contracts import ModelRun
from .models.neural import MLPRunner


def source_fingerprint():
    """Hash normalized package sources; independent of checkout line endings."""
    package = Path(__file__).resolve().parent
    hashed = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        hashed.update(path.relative_to(package).as_posix().encode())
        hashed.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return hashed.hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "joblib",
        "pyarrow",
        "scikit-learn",
        "torch",
        "lightgbm",
        "tabm",
        "rtdl_num_embeddings",
        "pytabkit",
        "tabicl",
        "tabpfn",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def checkpoint_manifest(specs):
    result = {}
    for spec in specs:
        if spec["family"] in {"tabicl", "tabpfn"}:
            path = spec["options"]["options"]["model_path"]
            result[path] = digest(path)
    return result


def dataset_manifests(root, seeds):
    """Bind prepared input manifests without opening any test partition."""
    return {str(seed): digest(Path(root) / "datasets" / str(seed) / "ready.json") for seed in seeds}


def load_development(dataset):
    dataset = Path(dataset)
    meta = json.loads((dataset / "ready.json").read_text())
    for split in ("train", "validation"):
        if digest(dataset / f"{split}.parquet") != meta["hashes"][split]:
            raise ValueError(f"{split} input hash differs from the prepared manifest")
    train = pd.read_parquet(dataset / "train.parquet")
    validation = pd.read_parquet(dataset / "validation.parquet")
    return meta, train, validation


def fit_candidate(spec, train, validation, features, *, seed):
    """Fit a configured estimator; no test argument exists in this interface."""
    family, options = spec["family"], copy.deepcopy(spec["options"])
    xt, xv = train[features], validation[features]
    yt, yv = train.label.to_numpy(), validation.label.to_numpy()
    ModelRun(xt, yt, xv, yv, xv, yv).validate_tabular()
    if family == "tabm":
        estimator = TabMEstimator(random_state=seed, **options).fit(xt, yt, xv, yv)
    elif family in {"realmlp", "tabicl", "tabpfn"}:
        estimator = ExternalTabularEstimator(family=family, random_state=seed, **options).fit(
            xt, yt, xv, yv
        )
    else:
        runners = {
            "lightgbm": LightGBMRunner,
            "logistic_regression": LogisticRegressionRunner,
            "mlp": MLPRunner,
        }
        if family not in runners:
            raise ValueError(f"Unknown research family {family}")
        # Only validation is supplied to the legacy runner's scoring slot.
        run = ModelRun(xt, yt, xv, yv, xv, yv, params={family: options}, random_state=seed)
        result = runners[family]().run(run)
        if not result.succeeded:
            raise RuntimeError(result.reason)
        estimator = result.estimator
    return estimator


def run_validation_search(dataset, candidates, output):
    meta, train, validation = load_development(dataset)
    output = Path(output)
    summaries = []
    for spec in candidates:
        directory = output / spec["name"]
        result_path = directory / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if any(result.get(key) != spec[key] for key in ("name", "family", "options")):
                raise ValueError("Cached result belongs to a different candidate specification")
            summaries.append(result)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        print("Training", meta["seed"], spec["name"], flush=True)
        try:
            estimator = fit_candidate(spec, train, validation, meta["features"], seed=meta["seed"])
            probability = estimator.predict_proba(validation[meta["features"]])[:, 1]
            joblib.dump(estimator, directory / "model.joblib")
            np.save(directory / "validation_scores.npy", probability)
            write_json(directory / "history.json", getattr(estimator, "history_", []))
            result = {
                **spec,
                "seed": meta["seed"],
                "status": "COMPLETED",
                "validation_ap": float(average_precision_score(validation.label, probability)),
                "seconds": perf_counter() - started,
                "best_epoch": getattr(estimator, "best_epoch_", None),
                "scope": "development validation only; no test partition opened",
                "train_sha256": meta["hashes"]["train"],
                "validation_sha256": meta["hashes"]["validation"],
            }
            del estimator
        except Exception as exc:
            result = {
                **spec,
                "seed": meta["seed"],
                "status": "FAILED",
                "reason": f"{type(exc).__name__}: {exc}",
                "seconds": perf_counter() - started,
            }
        write_json(result_path, result)
        print(spec["name"], result["status"], result.get("validation_ap"), flush=True)
        summaries.append(result)
        gc.collect()
    return summaries


def select_frozen_recipe(
    root, candidates, development_seeds, confirmation_seeds, *, minimum_gain=0.02, output=None
):
    """Compare fixed single models and equal-weight neural ensembles on validation."""
    root = Path(root)
    if set(development_seeds) & set(confirmation_seeds):
        raise ValueError("Confirmation cohorts must be independent of development cohorts")
    if not development_seeds or not confirmation_seeds:
        raise ValueError("Both development and confirmation seeds are required")
    successful = {}
    scores = {}
    labels = {}
    for seed in development_seeds:
        _, _, validation = load_development(root / "datasets" / str(seed))
        labels[seed] = validation.label.to_numpy()
        successful[seed] = {}
        scores[seed] = {}
        for spec in candidates:
            directory = root / "discovery" / str(seed) / spec["name"]
            path = directory / "result.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if result["status"] == "COMPLETED":
                if any(result.get(key) != spec[key] for key in ("name", "family", "options")):
                    raise ValueError("Development result differs from candidate specification")
                successful[seed][spec["name"]] = result["validation_ap"]
                scores[seed][spec["name"]] = np.load(directory / "validation_scores.npy")
                measured = average_precision_score(labels[seed], scores[seed][spec["name"]])
                if not np.isclose(measured, result["validation_ap"], atol=1e-12, rtol=0):
                    raise ValueError("Development score differs from saved predictions")
    common = set.intersection(*(set(v) for v in successful.values()))
    neural = [
        c
        for c in candidates
        if c["name"] in common and c["family"] in {"tabm", "realmlp", "tabicl", "tabpfn", "mlp"}
    ]
    if not neural:
        raise ValueError("No neural candidate completed on every development cohort")
    means = {
        c["name"]: float(np.mean([successful[s][c["name"]] for s in development_seeds]))
        for c in neural
    }
    ranked = sorted(neural, key=lambda c: (-means[c["name"]], c["name"]))
    proposals = [{"members": [c], "weights": [1.0]} for c in ranked]
    proposals.extend(
        {"members": ranked[:k], "weights": [1 / k] * k} for k in (2, 3, 5) if len(ranked) >= k
    )
    # A bounded, reproducible validation-only search of convex probability blends.
    # No classical reference predictions or reserved labels enter these weights.
    pool = ranked[:10]
    if len(pool) > 1:
        rng = np.random.default_rng(20260910)
        for concentration in (0.2, 1.0):
            for weights in rng.dirichlet(np.repeat(concentration, len(pool)), size=256):
                weights[weights < 0.025] = 0
                weights /= weights.sum()
                active = [(member, float(w)) for member, w in zip(pool, weights) if w > 0]
                proposals.append(
                    {
                        "members": [member for member, _ in active],
                        "weights": [w for _, w in active],
                    }
                )
    diagnostics = []
    for proposal in proposals:
        aps = []
        for seed in development_seeds:
            probability = np.average(
                [scores[seed][m["name"]] for m in proposal["members"]],
                axis=0,
                weights=proposal["weights"],
            )
            aps.append(float(average_precision_score(labels[seed], probability)))
        proposal["development_ap"] = dict(zip(map(str, development_seeds), aps))
        proposal["mean_development_ap"] = float(np.mean(aps))
        diagnostics.append(proposal)
    selected = max(diagnostics, key=lambda p: p["mean_development_ap"])
    reference_names = {"lightgbm_reference", "logistic_reference"}
    reference_specs = [c for c in candidates if c["name"] in reference_names]
    if len(reference_specs) != 2:
        raise ValueError("Both original references must be included")
    tuned = {}
    for family in ("lightgbm", "logistic_regression"):
        pool = [c for c in candidates if c["family"] == family and c["name"] in common]
        tuned[family] = max(
            pool, key=lambda c: np.mean([successful[s][c["name"]] for s in development_seeds])
        )
    frozen = {
        "version": 1,
        "source_fingerprint": source_fingerprint(),
        "runtime_versions": runtime_versions(),
        "checkpoints": checkpoint_manifest(selected["members"]),
        "dataset_manifests": dataset_manifests(root, [*development_seeds, *confirmation_seeds]),
        "development_seeds": list(development_seeds),
        "confirmation_seeds": list(confirmation_seeds),
        "selected": selected,
        "original_references": reference_specs,
        "tuned_references": tuned,
        "minimum_gain": minimum_gain,
        "family_alpha": 0.05,
        "bootstrap_iterations": 2000,
        "selection": "mean development validation AP; individual recipes, top-2/3/5 equal ensembles and 512 convex blends of the top 10 neural recipes",
        "blend_search": {
            "seed": 20260910,
            "dirichlet_concentrations": [0.2, 1.0],
            "draws_each": 256,
            "weight_floor": 0.025,
        },
        "test_access": "none during selection",
    }
    output = Path(output or root / "frozen_recipe.json")
    if output.exists():
        raise FileExistsError("A recipe is already frozen; do not replace it after test access")
    write_json(root / "development_selection.json", diagnostics)
    write_json(output, frozen)
    write_json(output.with_suffix(".sha256.json"), {"sha256": digest(output)})
    return frozen


def confirm_frozen_recipe(root, *, recipe_path=None):
    root = Path(root)
    path = Path(recipe_path or root / "frozen_recipe.json")
    recorded = json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
    if digest(path) != recorded:
        raise ValueError("Frozen recipe hash mismatch")
    recipe = json.loads(path.read_text())
    if recipe.get("source_fingerprint") != source_fingerprint():
        raise ValueError("Research source changed after recipe freezing")
    if recipe.get("runtime_versions") != runtime_versions():
        raise ValueError("Runtime versions changed after recipe freezing")
    if recipe.get("checkpoints") != checkpoint_manifest(recipe["selected"]["members"]):
        raise ValueError("Pretrained checkpoint changed after recipe freezing")
    if recipe.get("dataset_manifests") != dataset_manifests(
        root, [*recipe["development_seeds"], *recipe["confirmation_seeds"]]
    ):
        raise ValueError("Dataset manifest changed after recipe freezing")
    output = root / "confirmation"
    marker = output / "started.json"
    if marker.exists() and json.loads(marker.read_text())["recipe_sha256"] != recorded:
        raise ValueError("Confirmation has already started with a different recipe")
    write_json(marker, {"recipe_sha256": recorded, "test_selection_permitted": False})
    report_path = output / "confirmation_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    specs = {s["name"]: s for s in recipe["selected"]["members"]}
    specs.update({s["name"]: s for s in recipe["original_references"]})
    specs.update({s["name"]: s for s in recipe["tuned_references"].values()})
    # All validation fitting finishes before any confirmation test file is opened.
    for seed in recipe["confirmation_seeds"]:
        meta, train, validation = load_development(root / "datasets" / str(seed))
        folder = output / str(seed)
        folder.mkdir(parents=True, exist_ok=True)
        for name, spec in specs.items():
            model_path = folder / f"{name}.joblib"
            fit_path = folder / f"{name}.fit.json"
            binding = {
                "recipe_sha256": recorded,
                "train_sha256": meta["hashes"]["train"],
                "validation_sha256": meta["hashes"]["validation"],
            }
            if model_path.exists():
                expected = {**binding, "model_sha256": digest(model_path)}
                if not fit_path.exists() or json.loads(fit_path.read_text()) != expected:
                    raise ValueError("Cached confirmation model has no matching fit manifest")
            else:
                print("Fit frozen recipe", seed, name, flush=True)
                model = fit_candidate(spec, train, validation, meta["features"], seed=seed)
                joblib.dump(model, model_path)
                write_json(fit_path, {**binding, "model_sha256": digest(model_path)})
                del model
                gc.collect()
    model_hashes = {str(p.relative_to(output)): digest(p) for p in output.glob("*/*.joblib")}
    write_json(output / "fitted_models.json", {"recipe_sha256": recorded, "models": model_hashes})
    predictions = []
    capacity = []
    for seed in recipe["confirmation_seeds"]:
        dataset = root / "datasets" / str(seed)
        meta = json.loads((dataset / "ready.json").read_text())
        if digest(dataset / "test.parquet") != meta["hashes"]["test"]:
            raise ValueError("Confirmation test hash mismatch")
        test = pd.read_parquet(dataset / "test.parquet")
        x = test[meta["features"]]
        folder = output / str(seed)
        predicted = {}
        for name in specs:
            print("Score frozen model", seed, name, flush=True)
            model = joblib.load(folder / f"{name}.joblib")
            predicted[name] = model.predict_proba(x)[:, 1]
            del model
            gc.collect()
        frame = test[["snapshot_id", "patient_id", "label"]].copy()
        frame["cohort"] = seed
        frame["candidate"] = np.average(
            [predicted[m["name"]] for m in recipe["selected"]["members"]],
            axis=0,
            weights=recipe["selected"]["weights"],
        )
        for spec in recipe["original_references"]:
            frame[spec["name"]] = predicted[spec["name"]]
        for family, spec in recipe["tuned_references"].items():
            frame[f"{family}_tuned"] = predicted[spec["name"]]
        for name in (
            "candidate",
            "lightgbm_reference",
            "logistic_reference",
            "lightgbm_tuned",
            "logistic_regression_tuned",
        ):
            m = top_fraction_metrics(frame.label, frame[name], 0.1)
            capacity.append(
                {
                    "cohort": seed,
                    "model": name,
                    "ap": float(average_precision_score(frame.label, frame[name])),
                    **m,
                }
            )
        predictions.append(frame)
    combined = pd.concat(predictions, ignore_index=True)
    combined.to_parquet(output / "test_predictions.parquet", index=False)
    pd.DataFrame(capacity).to_csv(output / "capacity.csv", index=False)
    report = confirm_average_precision(
        combined,
        candidate="candidate",
        references=["lightgbm_reference", "logistic_reference"],
        n_bootstrap=recipe["bootstrap_iterations"],
        family_alpha=recipe["family_alpha"],
        minimum_gain=recipe["minimum_gain"],
    )
    report["tuned_control_comparison"] = confirm_average_precision(
        combined,
        candidate="candidate",
        references=["lightgbm_tuned", "logistic_regression_tuned"],
        n_bootstrap=recipe["bootstrap_iterations"],
        family_alpha=recipe["family_alpha"],
        minimum_gain=recipe["minimum_gain"],
    )
    report["recipe_sha256"] = recorded
    write_json(report_path, report)
    return report


def prepare_synthetic_cohort(config, root, seed):
    """Prepare the unchanged configured generator into separately stored partitions."""
    import inspect

    from .data import generate_synthetic_claims
    from .data.splitting import split_manifest, temporal_patient_split
    from .pipeline import prepare_inputs

    config = copy.deepcopy(config)
    if config["data"]["source"] != "synthetic":
        raise ValueError("This confirmation study accepts synthetic generation only")
    config["project"]["random_seed"] = seed
    directory = Path(root) / "datasets" / str(seed)
    ready = directory / "ready.json"
    if ready.exists():
        return json.loads(ready.read_text())
    print("Prepare cohort", seed, flush=True)
    _, inputs = prepare_inputs(config)
    frame = inputs.modeling_frame()
    parts = temporal_patient_split(frame, config)
    directory.mkdir(parents=True, exist_ok=True)
    hashes, counts = {}, {}
    for split, part in parts.items():
        part.attrs = {}
        part.to_parquet(directory / f"{split}.parquet", index=False)
        hashes[split] = digest(directory / f"{split}.parquet")
        counts[split] = {
            "patients": int(part.patient_id.nunique()),
            "snapshots": len(part),
            "positives": int(part.label.sum()),
        }
    split_manifest(frame, parts).to_csv(directory / "split_manifest.csv", index=False)
    meta = {
        "seed": seed,
        "config": config,
        "features": list(inputs.feature_columns),
        "hashes": hashes,
        "counts": counts,
        "generator_sha256": digest(inspect.getfile(generate_synthetic_claims)),
    }
    write_json(ready, meta)
    print("Prepared", seed, counts, flush=True)
    return meta
