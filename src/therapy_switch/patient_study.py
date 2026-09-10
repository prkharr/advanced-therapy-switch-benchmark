"""Reproducible patient-capacity research with separate search and confirmation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from therapy_switch.models.patient_rank import fit_patient_model
from therapy_switch.patient_lists import patient_capture_metrics
from therapy_switch.research import digest, write_json


def implementation_digest():
    package = Path(__file__).resolve().parent
    paths = [package / "patient_lists.py", package / "research.py"]
    paths += list((package / "models").glob("*.py"))
    paths += list((package / "features").glob("*.py"))
    hashed = hashlib.sha256()
    for path in sorted(paths):
        hashed.update(path.relative_to(package).as_posix().encode())
        hashed.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return hashed.hexdigest()


def load_partition(dataset, split):
    from therapy_switch.features.leakage import validate_predictor_names
    from therapy_switch.models.contracts import LeakageError

    dataset = Path(dataset)
    meta = json.loads((dataset / "ready.json").read_text())
    parts = []
    for name in (split, split + "_events"):
        path = dataset / f"{name}.parquet"
        if digest(path) != meta["hashes"][name]:
            raise ValueError(f"Changed prepared input: {name}")
        parts.append(pd.read_parquet(path))
    frame = parts[0]
    validate_predictor_names(meta["features"])
    dates = frame[["index_date", "feature_cutoff", "lookback_start"]].apply(pd.to_datetime)
    if (
        dates.isna().any().any()
        or (dates.feature_cutoff > dates.index_date).any()
        or (dates.lookback_start > dates.feature_cutoff).any()
    ):
        raise LeakageError("Invalid snapshot observation boundaries")
    if (
        frame.patient_id.isna().any()
        or frame.snapshot_id.isna().any()
        or frame.snapshot_id.duplicated().any()
    ):
        raise ValueError("Prepared assessment keys must be present and unique")
    return meta, *parts


def prepare(dataset, seed, config_path="configs/default.yaml"):
    """Generate the unchanged synthetic benchmark and seal split files."""
    from therapy_switch.config import load_config
    from therapy_switch.data.splitting import temporal_patient_split
    from therapy_switch.pipeline import prepare_inputs

    dataset = Path(dataset)
    if (dataset / "ready.json").exists():
        raise ValueError("Prepared cohort already exists")
    dataset.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    if config["data"]["source"] != "synthetic":
        raise ValueError("This study preparer only accepts synthetic input")
    config["project"]["random_seed"] = seed
    _, inputs = prepare_inputs(config)
    partitions = temporal_patient_split(inputs.modeling_frame(), config)
    meta = {
        "seed": seed,
        "config": config,
        "features": list(inputs.feature_columns),
        "hashes": {},
        "counts": {},
    }
    for split, frame in partitions.items():
        events = inputs.events.loc[inputs.events.snapshot_id.isin(frame.snapshot_id)].copy()
        for key, value in ((split, frame), (split + "_events", events)):
            value.attrs = {}
            value.to_parquet(dataset / f"{key}.parquet", index=False)
            meta["hashes"][key] = digest(dataset / f"{key}.parquet")
        meta["counts"][split] = {
            "patients": int(frame.patient_id.nunique()),
            "snapshots": len(frame),
            "positives": int(frame.label.sum()),
        }
    write_json(dataset / "ready.json", meta)
    return meta


def search(dataset, candidates, output, legacy_cache=None, retry_failed=False):
    meta, train, train_events = load_partition(dataset, "train")
    _, validation, validation_events = load_partition(dataset, "validation")
    output = Path(output)
    source = implementation_digest()
    rows = []
    for spec in candidates:
        folder = output / spec["name"]
        folder.mkdir(parents=True, exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous["spec"] != spec or previous["input_manifest"] != digest(
                Path(dataset) / "ready.json"
            ):
                raise ValueError("Cached result does not match specification or inputs")
            if previous.get("implementation") != source:
                raise ValueError("Model implementation changed since this candidate was fitted")
            if previous["status"] != "FAILED" or not retry_failed:
                rows.append(previous)
                continue
        started = perf_counter()
        print("Training", meta["seed"], spec["name"], flush=True)
        try:
            legacy = Path(legacy_cache) / spec["name"] if legacy_cache else None
            if spec.get("reuse_legacy") and legacy and (legacy / "result.json").exists():
                old = json.loads((legacy / "result.json").read_text())
                if any(old[key] != spec[key] for key in ("name", "family", "options")):
                    raise ValueError("Legacy specification mismatch")
                legacy_dataset = Path(legacy_cache).parent.parent / "datasets" / str(meta["seed"])
                if any(
                    digest(legacy_dataset / f"{key}.parquet") != meta["hashes"][key]
                    for key in ("train", "validation")
                ):
                    raise ValueError("Legacy development partition mismatch")
                if old["status"] != "COMPLETED":
                    raise ValueError("Legacy fit did not complete")
                shutil.copyfile(legacy / "validation_scores.npy", folder / "validation_scores.npy")
                scores = np.load(folder / "validation_scores.npy")
                from sklearn.metrics import average_precision_score

                if not np.isclose(
                    average_precision_score(validation.label, scores),
                    old["validation_ap"],
                    rtol=0,
                    atol=1e-12,
                ):
                    raise ValueError("Legacy predictions do not match their recorded AP")
                model_file = None
                origin = "reused exact prior development predictions"
                duration = old["seconds"]
            else:
                model = fit_patient_model(
                    spec,
                    train,
                    validation,
                    train_events,
                    validation_events,
                    meta["features"],
                    seed=meta["seed"],
                )
                scores = model.predict_scores(validation, validation_events)
                model_file = folder / "model.joblib"
                joblib.dump(model, model_file)
                np.save(folder / "validation_scores.npy", scores)
                write_json(folder / "history.json", getattr(model.estimator, "history_", []))
                del model
                origin = "new fit"
                duration = perf_counter() - started
            result = {
                "spec": spec,
                "status": "COMPLETED",
                "seed": meta["seed"],
                "metrics": patient_capture_metrics(validation, scores),
                "seconds": duration,
                "origin": origin,
                "scores_sha256": digest(folder / "validation_scores.npy"),
                "model_sha256": digest(model_file) if model_file else None,
            }
        except Exception as exc:
            result = {
                "spec": spec,
                "status": "FAILED",
                "seed": meta["seed"],
                "reason": f"{type(exc).__name__}: {exc}",
                "seconds": perf_counter() - started,
            }
        result.update(implementation=source, input_manifest=digest(Path(dataset) / "ready.json"))
        write_json(result_path, result)
        print(
            spec["name"], result["status"], result.get("metrics", result.get("reason")), flush=True
        )
        rows.append(result)
        gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--config", default="configs/default.yaml")
    run = sub.add_parser("search")
    run.add_argument("--dataset", required=True)
    run.add_argument("--candidates", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--legacy-cache")
    run.add_argument("--retry-failed", action="store_true")
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--root", required=True)
    frozen.add_argument("--candidates", required=True)
    frozen.add_argument("--protocol", required=True)
    frozen.add_argument("--previous-recipe", required=True)
    frozen.add_argument("--output", required=True)
    fit = sub.add_parser("fit-confirmation")
    fit.add_argument("--root", required=True)
    fit.add_argument("--recipe", required=True)
    fit.add_argument("--seed", type=int, required=True)
    evaluate = sub.add_parser("evaluate-confirmation")
    evaluate.add_argument("--root", required=True)
    evaluate.add_argument("--recipe", required=True)
    score = sub.add_parser("score")
    score.add_argument("--model", required=True)
    score.add_argument("--snapshots", required=True)
    score.add_argument("--events")
    score.add_argument("--output", required=True)
    score.add_argument("--fraction", type=float, default=0.1)
    score.add_argument("--as-of")
    fit_list = sub.add_parser("fit-list")
    fit_list.add_argument("--dataset", required=True)
    fit_list.add_argument("--recipe", required=True)
    fit_list.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.dataset, args.seed, args.config)
    elif args.command == "search":
        search(
            args.dataset,
            json.loads(Path(args.candidates).read_text()),
            args.output,
            args.legacy_cache,
            args.retry_failed,
        )
    elif args.command == "freeze":
        freeze(
            args.root,
            json.loads(Path(args.candidates).read_text()),
            json.loads(Path(args.protocol).read_text()),
            args.previous_recipe,
            args.output,
        )
    elif args.command == "fit-confirmation":
        fit_confirmation(args.root, args.recipe, args.seed)
    elif args.command == "evaluate-confirmation":
        report = evaluate_confirmation(args.root, args.recipe)
        print(json.dumps(report, indent=2))
    elif args.command == "fit-list":
        fit_list_model(args.dataset, args.recipe, args.output)
    elif args.command == "score":
        from therapy_switch.patient_lists import rank_patients

        model = joblib.load(args.model)
        frame = pd.read_parquet(args.snapshots)
        events = pd.read_parquet(args.events) if args.events else None
        probability = model.predict_scores(frame.drop(columns="label", errors="ignore"), events)
        ranked = rank_patients(frame, probability, fraction=args.fraction, as_of=args.as_of)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.output, index=False)
        ranked.loc[ranked.selected].to_csv(
            Path(args.output).with_suffix(".selected.csv"), index=False
        )
        print(f"Scored {len(ranked)} distinct patients; selected {int(ranked.selected.sum())}")


def _canonical_json(path, value):
    """Stable LF JSON suitable for checked-in recipe hashes on every platform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, default=str) + "\n").encode())


def _patient_metrics(y, scores, fraction=0.1):
    from sklearn.metrics import average_precision_score

    k = int(np.ceil(len(y) * fraction))
    capture = int(y[np.argsort(-scores, kind="stable")[:k]].sum())
    return {"recall": capture / int(y.sum()), "ap": float(average_precision_score(y, scores))}


def freeze(root, candidates, protocol, previous_recipe, output):
    """Select fixed members/weights from development; never open confirmation test files."""
    from therapy_switch.patient_lists import latest_patient_indices
    from therapy_switch.research import checkpoint_manifest, runtime_versions, source_fingerprint

    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("A frozen recipe cannot be overwritten")
    development = protocol["development_seeds"]
    confirmation = protocol["fresh_confirmation_seeds"]
    if set(development) & set(confirmation) or not development or not confirmation:
        raise ValueError("Development and confirmation cohorts must be nonempty and disjoint")
    if len(confirmation) != len(set(confirmation)):
        raise ValueError("Confirmation seeds must be unique")
    if protocol["capacity"] != 0.1:
        raise ValueError("This candidate bank selects checkpoints for top-10% capacity")
    source = implementation_digest()
    loaded, labels, rows = {}, {}, []
    manifests = {
        str(seed): digest(root / "datasets" / str(seed) / "ready.json")
        for seed in [*development, *confirmation]
    }
    for seed in development:
        meta, validation, _ = load_partition(root / "datasets" / str(seed), "validation")
        ids = latest_patient_indices(validation)
        labels[seed] = validation.label.to_numpy()[ids]
        loaded[seed] = {}
        for spec in candidates:
            directory = root / "discovery" / str(seed) / spec["name"]
            path = directory / "result.json"
            if not path.exists():
                raise ValueError(f"Candidate search incomplete: {seed} {spec['name']}")
            result = json.loads(path.read_text())
            if result["spec"] != spec or result["implementation"] != source:
                raise ValueError("Development specification or model source changed")
            if result["input_manifest"] != manifests[str(seed)]:
                raise ValueError("Development inputs changed")
            if result["status"] != "COMPLETED":
                raise ValueError(f"Resolve failed candidate before freezing: {spec['name']}")
            prediction_path = directory / "validation_scores.npy"
            if digest(prediction_path) != result["scores_sha256"]:
                raise ValueError("Development predictions changed")
            full_scores = np.load(prediction_path)
            measured = patient_capture_metrics(validation, full_scores)
            if any(
                not np.isclose(measured[k], result["metrics"][k], rtol=0, atol=1e-12)
                for k in measured
            ):
                raise ValueError("Recorded development metrics do not match predictions")
            loaded[seed][spec["name"]] = full_scores[ids]
            rows.append(
                {
                    "seed": seed,
                    "name": spec["name"],
                    "family": spec["family"],
                    "history": spec.get("history", False),
                    "latest_only": spec.get("latest_only", False),
                    "seconds": result["seconds"],
                    "origin": result["origin"],
                    **measured,
                }
            )
    summary = pd.DataFrame(rows).groupby("name", sort=True)[["recall", "ap"]].mean()
    ranked = sorted(
        candidates,
        key=lambda c: (-summary.loc[c["name"], "recall"], -summary.loc[c["name"], "ap"], c["name"]),
    )
    proposals = [{"members": [c], "weights": [1.0]} for c in ranked]
    for count in (2, 3, 5, 10):
        if len(ranked) >= count:
            proposals.append({"members": ranked[:count], "weights": [1 / count] * count})
    pool = ranked[:12]
    # Include the strongest representative of each family, even if outside top 12.
    for family in sorted(set(c["family"] for c in ranked)):
        candidate = next(c for c in ranked if c["family"] == family)
        if candidate not in pool:
            pool.append(candidate)
    for i, left in enumerate(pool):
        for right in pool[i + 1 :]:
            for weight in (0.25, 0.5, 0.75):
                proposals.append({"members": [left, right], "weights": [weight, 1 - weight]})
    rng = np.random.default_rng(20260910)
    for concentration in (0.2, 1.0):
        for weights in rng.dirichlet(np.repeat(concentration, len(pool)), size=512):
            weights[weights < 0.04] = 0
            weights /= weights.sum()
            active = [
                (member, float(weight)) for member, weight in zip(pool, weights) if weight > 0
            ]
            proposals.append({"members": [m for m, _ in active], "weights": [w for _, w in active]})
    for proposal in proposals:
        metrics = []
        for seed in development:
            scores = np.average(
                [loaded[seed][m["name"]] for m in proposal["members"]],
                axis=0,
                weights=proposal["weights"],
            )
            metrics.append(_patient_metrics(labels[seed], scores))
        proposal["development_metrics"] = dict(zip(map(str, development), metrics))
        proposal["mean_recall"] = float(np.mean([m["recall"] for m in metrics]))
        proposal["mean_ap"] = float(np.mean([m["ap"] for m in metrics]))
    selected = max(proposals, key=lambda p: (p["mean_recall"], p["mean_ap"]))
    original_names = ("lightgbm_reference", "logistic_reference")
    original = [next(c for c in candidates if c["name"] == name) for name in original_names]
    previous = json.loads(Path(previous_recipe).read_text())
    previous_members = [
        next(c for c in candidates if c["name"] == old["name"])
        for old in previous["selected"]["members"]
    ]
    for current, old in zip(previous_members, previous["selected"]["members"]):
        if any(current[k] != old[k] for k in ("name", "family", "options")):
            raise ValueError("The previous neural recipe specification changed")
    classical = {
        "lightgbm",
        "logistic_regression",
        "catboost",
        "xgboost",
        "lgbm_capacity",
        "lambdarank",
    }
    tuned = next(c for c in ranked if c["family"] in classical and c["name"] not in original_names)
    ehr = next(c for c in ranked if c["family"] in {"ehr_transformer", "retain"})
    controls = {
        "original_lightgbm": {"members": [original[0]], "weights": [1.0]},
        "original_logistic": {"members": [original[1]], "weights": [1.0]},
        "previous_neural": {
            "members": previous_members,
            "weights": previous["selected"]["weights"],
        },
        "tuned_classical": {"members": [tuned], "weights": [1.0]},
        "best_ehr": {"members": [ehr], "weights": [1.0]},
    }
    recipe = {
        "version": 1,
        "protocol": protocol,
        "selected": selected,
        "controls": controls,
        "primary_references": ["original_lightgbm", "original_logistic", "previous_neural"],
        "secondary_controls": ["tuned_classical", "best_ehr"],
        "candidate_count": len(candidates),
        "proposal_count": len(proposals),
        "candidate_bank_sha256": hashlib.sha256(
            json.dumps(candidates, sort_keys=True).encode()
        ).hexdigest(),
        "implementation": source,
        "source_fingerprint": source_fingerprint(),
        "runtime_versions": runtime_versions(),
        "dataset_manifests": manifests,
        "checkpoints": checkpoint_manifest(candidates),
        "previous_neural_recipe_sha256": digest(previous_recipe),
    }
    from importlib.metadata import version

    recipe["runtime_versions"].update({k: version(k) for k in ("catboost", "xgboost")})
    pd.DataFrame(rows).to_csv(output.with_name("development_results.csv"), index=False)
    _canonical_json(output.with_name("selection_proposals.json"), proposals)
    _canonical_json(output, recipe)
    _canonical_json(output.with_suffix(".sha256.json"), {"sha256": digest(output)})
    return recipe


def check_recipe(recipe_path, root):
    from importlib.metadata import version

    from therapy_switch.research import runtime_versions, source_fingerprint

    path, root = Path(recipe_path), Path(root)
    expected = json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
    if digest(path) != expected:
        raise ValueError("Frozen recipe bytes changed")
    recipe = json.loads(path.read_text())
    if (
        recipe["implementation"] != implementation_digest()
        or recipe["source_fingerprint"] != source_fingerprint()
    ):
        raise ValueError("Source changed after freezing; use the frozen source revision")
    versions = runtime_versions()
    versions.update({k: version(k) for k in ("catboost", "xgboost")})
    if versions != recipe["runtime_versions"]:
        raise ValueError("Runtime changed after freezing")
    for seed, expected_hash in recipe["dataset_manifests"].items():
        if digest(root / "datasets" / seed / "ready.json") != expected_hash:
            raise ValueError("Prepared input manifest changed after freezing")
    for checkpoint, expected_hash in recipe["checkpoints"].items():
        if digest(checkpoint) != expected_hash:
            raise ValueError("Pretrained checkpoint changed after freezing")
    return recipe


def _all_specs(recipe):
    by_name = {}
    for group in [recipe["selected"], *recipe["controls"].values()]:
        for spec in group["members"]:
            if spec["name"] in by_name and by_name[spec["name"]] != spec:
                raise ValueError("Conflicting member specifications")
            by_name[spec["name"]] = spec
    return list(by_name.values())


def fit_confirmation(root, recipe_path, seed):
    """Fit a reserved cohort's frozen models using train/validation, without test reads."""
    root = Path(root)
    recipe = check_recipe(recipe_path, root)
    if seed not in recipe["protocol"]["fresh_confirmation_seeds"]:
        raise ValueError("Cohort was not reserved")
    dataset = root / "datasets" / str(seed)
    meta, train, train_events = load_partition(dataset, "train")
    _, validation, validation_events = load_partition(dataset, "validation")
    output = root / "confirmation" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "fits.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["recipe_sha256"] != digest(recipe_path):
            raise ValueError("Confirmation fits belong to a different recipe")
    else:
        manifest = {"recipe_sha256": digest(recipe_path), "seed": seed, "fits": {}}
    for spec in _all_specs(recipe):
        name = spec["name"]
        model_path = output / f"{name}.joblib"
        if name in manifest["fits"]:
            if digest(model_path) != manifest["fits"][name]["sha256"]:
                raise ValueError("Saved confirmation model changed")
            continue
        print("Confirmation training", seed, name, flush=True)
        started = perf_counter()
        model = fit_patient_model(
            spec, train, validation, train_events, validation_events, meta["features"], seed=seed
        )
        scores = model.predict_scores(validation, validation_events)
        joblib.dump(model, model_path)
        reloaded = joblib.load(model_path)
        np.testing.assert_allclose(
            reloaded.predict_scores(validation, validation_events), scores, rtol=0, atol=1e-7
        )
        manifest["fits"][name] = {
            "sha256": digest(model_path),
            "seconds": perf_counter() - started,
            "validation": patient_capture_metrics(validation, scores),
            "reload_verified": True,
            "train_sha256": meta["hashes"]["train"],
            "validation_sha256": meta["hashes"]["validation"],
            "train_events_sha256": meta["hashes"]["train_events"],
            "validation_events_sha256": meta["hashes"]["validation_events"],
            "best_epoch": getattr(model.estimator, "best_epoch_", None),
            "best_iteration": getattr(model.estimator, "best_iteration_", None),
        }
        _canonical_json(manifest_path, manifest)
        del model, reloaded
        gc.collect()
    return manifest


def evaluate_confirmation(root, recipe_path):
    """Unlock test scoring only after every reserved model fit is sealed."""
    from therapy_switch.evaluation.patient_confirmation import confirm_patient_capture
    from therapy_switch.models.patient_rank import PatientEnsemble
    from therapy_switch.patient_lists import latest_patient_indices, rank_patients

    root = Path(root)
    recipe = check_recipe(recipe_path, root)
    seeds = recipe["protocol"]["fresh_confirmation_seeds"]
    specs = _all_specs(recipe)
    output = root / "confirmation"
    for seed in seeds:
        folder = output / str(seed)
        manifest = json.loads((folder / "fits.json").read_text())
        if manifest["recipe_sha256"] != digest(recipe_path):
            raise ValueError("Confirmation fits belong to a different recipe")
        for spec in specs:
            recorded = manifest["fits"].get(spec["name"])
            if recorded is None or digest(folder / f"{spec['name']}.joblib") != recorded["sha256"]:
                raise ValueError("Every reserved fit must complete before test evaluation")
    # This marker is written before the first test read. Replays use the same recipe.
    lock = output / "test_evaluation_started.json"
    if lock.exists() and json.loads(lock.read_text())["recipe_sha256"] != digest(recipe_path):
        raise ValueError("These reserved tests have already been evaluated under another recipe")
    _canonical_json(
        lock, {"recipe_sha256": digest(recipe_path), "status": "test evaluation started"}
    )
    rows, metrics, timings = [], [], []
    for seed in seeds:
        _, test, events = load_partition(root / "datasets" / str(seed), "test")
        folder = output / str(seed)
        models, member_scores = {}, {}
        for spec in specs:
            name = spec["name"]
            models[name] = joblib.load(folder / f"{name}.joblib")
            started = perf_counter()
            member_scores[name] = models[name].predict_scores(test.drop(columns="label"), events)
            timings.append(
                {"seed": seed, "name": name, "inference_seconds": perf_counter() - started}
            )
        evaluated = {"candidate": recipe["selected"], **recipe["controls"]}
        full = test[["patient_id", "snapshot_id", "index_date", "label"]].copy()
        for name, group in evaluated.items():
            full[name] = np.average(
                [member_scores[m["name"]] for m in group["members"]],
                axis=0,
                weights=group["weights"],
            )
            for capacity in (0.05, 0.1, 0.2):
                metrics.append(
                    {
                        "cohort": seed,
                        "model": name,
                        "capacity": capacity,
                        **patient_capture_metrics(test, full[name], fraction=capacity),
                    }
                )
        ids = latest_patient_indices(full)
        one = full.iloc[ids].copy()
        one["cohort"] = seed
        rows.append(one)
        champion = PatientEnsemble(
            [models[m["name"]] for m in recipe["selected"]["members"]],
            recipe["selected"]["weights"],
        )
        joblib.dump(champion, folder / "champion.joblib")
        rank_patients(test, full.candidate, fraction=recipe["protocol"]["capacity"]).to_csv(
            folder / "ranked_patients.csv", index=False
        )
        del models, member_scores, champion
        gc.collect()
    frame = pd.concat(rows, ignore_index=True)
    frame.to_parquet(output / "patient_scores.parquet", index=False)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(timings).to_csv(output / "inference_times.csv", index=False)
    protocol = recipe["protocol"]
    report = confirm_patient_capture(
        frame,
        candidate="candidate",
        references=recipe["primary_references"],
        fraction=protocol["capacity"],
        n_bootstrap=protocol["bootstrap_draws"],
        family_alpha=protocol["family_alpha"],
        minimum_gain=protocol["material_gain"],
    )
    report["recipe_sha256"] = digest(recipe_path)
    report["secondary_comparisons"] = confirm_patient_capture(
        frame,
        candidate="candidate",
        references=recipe["secondary_controls"],
        fraction=protocol["capacity"],
        n_bootstrap=protocol["bootstrap_draws"],
        family_alpha=protocol["family_alpha"],
        minimum_gain=protocol["material_gain"],
    )
    _canonical_json(output / "report.json", report)
    return report


def fit_list_model(dataset, recipe_path, output):
    """Refit fixed selected members on approved local train/validation partitions."""
    from therapy_switch.models.patient_rank import PatientEnsemble
    from therapy_switch.research import source_fingerprint

    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new model output path; existing models are not overwritten")
    meta, train, train_events = load_partition(dataset, "train")
    _, validation, validation_events = load_partition(dataset, "validation")
    if set(train.patient_id) & set(validation.patient_id):
        raise ValueError("Training and validation patients must be disjoint")
    recipe = json.loads(Path(recipe_path).read_text())
    selected = recipe["selected"]
    members = [
        fit_patient_model(
            spec,
            train,
            validation,
            train_events,
            validation_events,
            meta["features"],
            seed=meta["seed"],
        )
        for spec in selected["members"]
    ]
    ensemble = PatientEnsemble(members, selected["weights"])
    scores = ensemble.predict_scores(validation.drop(columns="label"), validation_events)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, output)
    reloaded = joblib.load(output)
    np.testing.assert_allclose(
        reloaded.predict_scores(validation, validation_events), scores, rtol=0, atol=1e-7
    )
    _canonical_json(
        output.with_suffix(".manifest.json"),
        {
            "model_sha256": digest(output),
            "recipe_sha256": digest(recipe_path),
            "input_manifest_sha256": digest(Path(dataset) / "ready.json"),
            "source_fingerprint": source_fingerprint(),
            "reload_verified": True,
            "validation": patient_capture_metrics(validation, scores),
            "scope": "Fixed-recipe local fit; no test partition opened; scores are ranking signals",
        },
    )
    return ensemble


if __name__ == "__main__":
    main()
