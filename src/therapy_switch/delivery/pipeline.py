"""Single-command raw-to-HCP delivery orchestration.

Stages communicate through DeliveryInputs and a model exposing predict_scores.
Custom adapters/trainers are configured module:function references.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import yaml

from therapy_switch.config import _deep_merge, load_config
from therapy_switch.data.prepared_input_adapter import validate_prepared_inputs
from therapy_switch.data.splitting import temporal_patient_split
from therapy_switch.io import write_json
from therapy_switch.patient_lists import (
    latest_patient_indices,
    patient_capture_metrics,
    rank_patients,
)
from therapy_switch.provenance import runtime_versions, source_fingerprint

from .contracts import resolve_callable
from .targeting import build_field_targets, csv_safe


def input_policy_hash(config):
    """Version input meaning, independently of paths, transport and scoring date."""
    timeline = config["timeline"]
    policy = {key: config.get(key, {}) for key in ("cohort", "features", "therapy_mapping")}
    if config["data"].get("kind") == "real":
        policy["data_kind"] = "real"
        policy["source_definitions"] = config.get("source_definitions", {})
    policy["feature_contract_version"] = config["delivery"].get(
        "feature_contract_version", "claims-v1"
    )
    policy["timeline"] = {
        key: timeline.get(key)
        for key in (
            "observation_window_days",
            "prediction_window_days",
            "minimum_history_days",
            "claims_lag_days",
            "label_runout_days",
            "minimum_followup_days",
        )
    }
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


@dataclass
class DeliveryModel:
    estimator: object
    features: tuple[str, ...]
    training_label_cutoff: str
    evidence_status: str
    recipe_sha256: str | None
    trainer: str
    input_policy_sha256: str | None = None
    training_data_kind: str = "unverified"

    def predict(self, batch, scoring_date, *, config=None):
        if config and config["data"].get("kind") == "real":
            if getattr(self, "training_data_kind", "unverified") != "real":
                raise ValueError("Real-data scoring requires a model trained in real-data mode")
        if self.input_policy_sha256 is not None:
            if config is None or input_policy_hash(config) != self.input_policy_sha256:
                raise ValueError(
                    "Feature, cohort or timing rules changed; refit a compatible model"
                )
        if tuple(batch.feature_columns) != self.features:
            raise ValueError("Scoring feature schema changed; refit a compatible model")
        if pd.Timestamp(scoring_date) < pd.Timestamp(self.training_label_cutoff):
            raise ValueError("Model used labels unavailable at the requested scoring date")
        if batch.snapshots.empty:
            return np.array([], dtype=float)
        values = np.asarray(
            self.estimator.predict_scores(batch.modeling_frame(), batch.events), dtype=float
        )
        if values.shape != (len(batch.snapshots),) or not np.isfinite(values).all():
            raise ValueError("Model plugin must return one finite risk score per assessment")
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Model plugin ranking scores must lie in [0, 1]")
        return values


def load_delivery_config(path, *, overrides=None):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Delivery configuration not found: {path}. Run python setup_real_data.py first."
        )
    settings = yaml.safe_load(path.read_text()) or {}
    if "benchmark_config" not in settings or "delivery" not in settings:
        raise ValueError("Delivery config requires benchmark_config and delivery sections")
    base_path = Path(settings["benchmark_config"])
    config = load_config(
        base_path if base_path.is_absolute() else path.parent / base_path,
        settings.get("overrides", {}),
    )
    config["delivery"] = settings["delivery"]
    if overrides:
        config = _deep_merge(config, {"delivery": overrides})
    delivery = config["delivery"]
    if config["data"].get("kind") == "real":
        validate_real_delivery(config)
    if not 0 < float(delivery.get("patient_fraction", 0.1)) <= 1:
        raise ValueError("Patient fraction must lie in (0, 1]")
    scoring = pd.Timestamp(delivery["scoring_date"])
    if pd.isna(scoring) or scoring != scoring.normalize():
        raise ValueError("A calendar scoring date is required")
    hcp = delivery.get("hcp", {})
    tiers = hcp.get("tier_rank_percentiles", [0.2, 0.5])
    if (
        len(tiers) != 2
        or not 0 < tiers[0] < tiers[1] <= 1
        or int(hcp.get("minimum_priority_patients", 1)) < 1
    ):
        raise ValueError("Invalid HCP tier or minimum-patient settings")
    # Every delivery-config path is relative to that config file, independent of cwd.
    for owner, keys in [
        (delivery, ["output_dir", "artifact_dir", "model_artifact"]),
        (delivery.get("model", {}), ["recipe"]),
        (delivery.get("prepared", {}), ["training_dir", "scoring_dir"]),
    ]:
        for key in keys:
            if owner.get(key):
                value = Path(owner[key])
                owner[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    if config["data"].get("source") == "files":
        value = Path(config["data"]["input_dir"])
        config["data"]["input_dir"] = str(
            value if value.is_absolute() else (path.parent / value).resolve()
        )
    config["_delivery_config_path"] = str(path)
    config["extraction"] = copy.deepcopy(settings.get("extraction", {}))
    for key in ("sql_dir", "output_dir"):
        if config["extraction"].get(key):
            value = Path(config["extraction"][key])
            config["extraction"][key] = str(
                value if value.is_absolute() else (path.parent / value).resolve()
            )
    return config


def validate_real_delivery(config):
    from therapy_switch.real_data import validate_real_data

    validate_real_data(config)
    if config["data"].get("kind") != "real":
        return
    settings = config["delivery"]
    scoring = pd.to_datetime(settings.get("scoring_date"), errors="coerce")
    if pd.isna(scoring):
        raise ValueError("Set delivery.scoring_date to the actual assessment date")
    if scoring > pd.Timestamp(config["data"]["as_of_date"]):
        raise ValueError("Scoring date exceeds the extract as-of date")
    horizon = int(config["timeline"]["prediction_window_days"])
    runout = int(config["timeline"]["label_runout_days"])
    if pd.Timestamp(config["timeline"]["index_date_end"]) + pd.Timedelta(days=horizon + runout) > scoring:
        raise ValueError("Historical index_date_end needs mature outcomes by scoring_date")
    if not settings.get("hcp", {}).get("relevant_specialties"):
        raise ValueError("Set delivery.hcp.relevant_specialties using actual provider categories")


def _train_model(inputs, config, artifact_directory):
    training = inputs.training
    if training is None:
        raise ValueError("The adapter did not provide training data for train-score mode")
    validate_prepared_inputs(training, config, require_lineage=False)
    features = tuple(training.feature_columns)
    if features != tuple(inputs.scoring.feature_columns):
        raise ValueError("Training and scoring adapters produced different feature schemas")
    frame = training.modeling_frame()
    if frame.empty or frame.label_available_date.isna().any() or not frame.label.isin([0, 1]).all():
        raise ValueError("Training requires nonempty, binary labelled data with maturity dates")
    if pd.to_datetime(frame.label_available_date).max() > pd.Timestamp(
        config["delivery"]["scoring_date"]
    ):
        raise ValueError("Training input contains outcomes unavailable at scoring date")
    splits = temporal_patient_split(frame, config)
    if any(
        set(splits[a].patient_id) & set(splits[b].patient_id)
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Historical partitions must be patient-disjoint")
    settings = copy.deepcopy(config["delivery"]["model"])
    settings["patient_fraction"] = config["delivery"].get("patient_fraction", 0.1)
    trainer = resolve_callable(settings["trainer"])
    train, val = splits["train"], splits["validation"]
    train_events = training.events.loc[training.events.snapshot_id.isin(train.snapshot_id)]
    val_events = training.events.loc[training.events.snapshot_id.isin(val.snapshot_id)]
    started = perf_counter()
    estimator = trainer(
        train,
        val,
        train_events,
        val_events,
        list(features),
        settings=copy.deepcopy(settings),
        seed=int(config["project"]["random_seed"]),
    )
    if not callable(getattr(estimator, "predict_scores", None)):
        raise TypeError(
            "A model trainer must return an estimator exposing predict_scores(frame, events)"
        )
    recipe_hash = None
    if settings.get("recipe"):
        recipe_hash = hashlib.sha256(Path(settings["recipe"]).read_bytes()).hexdigest()
    cutoff = max(
        pd.to_datetime(train.label_available_date).max(),
        pd.to_datetime(val.label_available_date).max(),
    )
    model = DeliveryModel(
        estimator,
        features,
        str(cutoff.date()),
        getattr(estimator, "evidence_status", settings.get("evidence_status", "experimental")),
        recipe_hash,
        settings["trainer"],
        input_policy_hash(config),
        config["data"]["kind"],
    )
    path = artifact_directory / "model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    original = estimator.predict_scores(
        val.drop(columns=["label", "resp", "outcome_date"], errors="ignore"), val_events
    )
    replay = reloaded.estimator.predict_scores(
        val.drop(columns=["label", "resp", "outcome_date"], errors="ignore"), val_events
    )
    np.testing.assert_allclose(original, replay, rtol=0, atol=1e-7)
    audit = {
        "training_seconds": perf_counter() - started,
        "reload_verified": True,
        "split_counts": {},
        "model_selection": "validation recall at capacity, then average precision; test is held out",
        "runtime_versions": runtime_versions(),
        "selected_baseline": getattr(estimator, "selected_name", None),
        "baseline_parameters": getattr(estimator, "parameters", {}),
        "baseline_comparison": {},
        "historical_metrics": {},
    }
    for split, part in splits.items():
        audit["split_counts"][split] = {
            "patients": int(part.patient_id.nunique()),
            "snapshots": len(part),
            "positive_snapshots": int(part.label.sum()),
        }
    for split in ("validation", "test"):
        part = splits[split]
        events = training.events.loc[training.events.snapshot_id.isin(part.snapshot_id)]
        scores = estimator.predict_scores(
            part.drop(columns=["label", "resp", "outcome_date"], errors="ignore"), events
        )
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(part),) or not np.isfinite(scores).all():
            raise ValueError("Historical predictions must be finite and aligned")
        if part.iloc[latest_patient_indices(part)].label.nunique() == 2:
            audit["historical_metrics"][split] = patient_capture_metrics(
                part, scores, fraction=config["delivery"].get("patient_fraction", 0.1)
            )
        else:
            audit["historical_metrics"][split] = {
                "status": "not_estimable",
                "reason": "Latest patient assessments require both outcome classes",
            }
    for name, candidate in getattr(estimator, "candidates", {}).items():
        audit["baseline_comparison"][name] = {}
        for split in ("validation", "test"):
            part = splits[split]
            scores = candidate.predict_proba(part[list(features)])[:, 1]
            if part.iloc[latest_patient_indices(part)].label.nunique() == 2:
                audit["baseline_comparison"][name][split] = patient_capture_metrics(
                    part, scores, fraction=config["delivery"].get("patient_fraction", 0.1)
                )
            else:
                audit["baseline_comparison"][name][split] = {"status": "not_estimable"}
    comparison_rows = [dict(model=name, split=split, **metrics)
                       for name, parts in audit["baseline_comparison"].items()
                       for split, metrics in parts.items()]
    pd.DataFrame(comparison_rows).to_csv(artifact_directory / "baseline_comparison.csv", index=False)
    write_json(audit, artifact_directory / "training_audit.json")
    return model, path, audit


def run_delivery(config, *, mode="train-score", model_artifact=None, session=None):
    """Acquire -> prepare -> fit/load -> score -> attribute -> CSV/HTML; no external publishing."""
    from .report import write_field_report

    if mode not in {"train-score", "score"}:
        raise ValueError("Mode must be train-score or score")
    config = copy.deepcopy(config)
    validate_real_delivery(config)
    settings = config["delivery"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = Path(settings["output_dir"]) / run_id
    artifacts = Path(settings["artifact_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "status": "RUNNING",
        "mode": mode,
        "data_kind": config["data"].get("kind", "unverified"),
        "scoring_date": str(pd.Timestamp(settings["scoring_date"]).date()),
        "stages": {},
    }
    write_json(manifest, artifacts / "run_manifest.json")
    write_json(config, artifacts / "resolved_config.json")
    try:
        model = None
        if mode == "score":
            source = model_artifact or settings.get("model_artifact")
            if not source:
                raise ValueError("Score mode requires a trusted local model artifact")
            model_path = Path(source)
            model = joblib.load(model_path)
            if not isinstance(model, DeliveryModel):
                raise TypeError("Use a model artifact produced by this delivery pipeline")
        adapter = resolve_callable(settings["adapter"])
        started = perf_counter()
        inputs = adapter(
            config,
            include_training=mode == "train-score",
            expected_features=None if model is None else model.features,
            session=session,
        )
        inputs.scoring.validate(settings["scoring_date"])
        write_json(inputs.provenance.get("raw_profile", {}), artifacts / "raw_profile.json")
        snapshots = inputs.scoring.snapshots
        lag = pd.Timedelta(days=int(config["timeline"].get("claims_lag_days", 0)))
        lookback = pd.Timedelta(days=int(config["timeline"]["observation_window_days"]))
        if (
            snapshots.feature_cutoff > snapshots.index_date - lag
        ).any() or not snapshots.lookback_start.eq(snapshots.index_date - lookback).all():
            raise ValueError("Scoring batch does not satisfy configured feature timing rules")
        manifest["stages"]["extract_prepare_seconds"] = perf_counter() - started
        audit = {}
        if mode == "train-score":
            model, model_path, audit = _train_model(inputs, config, artifacts)
        started = perf_counter()
        scores = model.predict(inputs.scoring, settings["scoring_date"], config=config)
        fraction = float(settings.get("patient_fraction", 0.1))
        if len(scores):
            patients = rank_patients(inputs.scoring.snapshots, scores, fraction=fraction)
        else:
            patients = pd.DataFrame(
                columns=[
                    "patient_id",
                    "snapshot_id",
                    "index_date",
                    "risk_score",
                    "rank",
                    "selected",
                ]
            )
            patients["selected"] = patients.selected.astype(bool)
        manifest["stages"]["score_seconds"] = perf_counter() - started
        targets, attribution, coverage = build_field_targets(
            patients,
            inputs.scoring.events,
            inputs.scoring.providers,
            settings.get("hcp", {}),
            settings["scoring_date"],
        )
        scope = settings.get("data_scope", "unverified local data")
        targets["data_scope"], targets["model_evidence"] = scope, model.evidence_status
        # Patient-level records stay in the restricted artifact directory.
        for name, frame in [
            ("patient_scores", patients),
            ("patient_hcp_attribution", attribution),
            ("scoring_cohort", inputs.scoring.snapshots),
        ]:
            saved = frame.copy()
            saved.attrs = {}  # Exclusion summaries live in the manifest, not Parquet metadata.
            saved.to_parquet(artifacts / f"{name}.parquet", index=False)
        csv_path = output / "hcp_targets.csv"
        csv_safe(targets).to_csv(csv_path, index=False, encoding="utf-8-sig")
        schema = {
            "grain": "one HCP at one scoring date",
            "columns": list(targets),
            "identifier_columns": ["hcp_id"],
            "patient_identifiers_included": False,
            "ranking": "priority patient count descending; priority share descending; HCP ID ascending",
            "risk_interpretation": "counts of top-capacity patients; not expected switcher counts",
            "patient_fraction": fraction,
            "assessment_date": manifest["scoring_date"],
        }
        write_json(schema, output / "hcp_targets.schema.json")
        manifest.update(
            status="COMPLETED",
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            model_recipe_sha256=model.recipe_sha256,
            input_policy_sha256=model.input_policy_sha256,
            model_evidence=model.evidence_status,
            model_artifact=str(model_path.resolve()),
            feature_columns=list(model.features),
            source_fingerprint=source_fingerprint(),
            adapter=settings["adapter"],
            trainer=model.trainer,
            input_provenance=inputs.provenance,
            coverage=coverage,
            hcp_targets=len(targets),
            patient_fraction=fraction,
            data_scope=scope,
            historical_evaluation=audit.get("historical_metrics", {}),
            output_columns=list(targets),
            patient_data_location="restricted artifact directory",
        )
        write_field_report(targets, coverage, manifest, output / "field_report.html")
        write_json(manifest, artifacts / "run_manifest.json")
        return {
            "status": "COMPLETED",
            "run_id": run_id,
            "hcp_targets": len(targets),
            "eligible_patients": len(patients),
            "priority_patients": int(patients.selected.sum()),
            "hcp_csv": str(csv_path.resolve()),
            "html_report": str((output / "field_report.html").resolve()),
            "model_artifact": str(model_path.resolve()),
            "manifest": str((artifacts / "run_manifest.json").resolve()),
            "raw_csv_dir": inputs.provenance.get("raw_csv_dir"),
        }
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_json(manifest, artifacts / "run_manifest.json")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[3] / "configs" / "private" / "delivery.yaml"),
    )
    parser.add_argument("--mode", choices=["train-score", "score"], default="train-score")
    parser.add_argument("--model-artifact")
    parser.add_argument("--model-recipe")
    parser.add_argument("--output-dir")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--raw-dir", help="Read the seven canonical CSV files from this folder")
    inputs.add_argument("--extract", action="store_true", help="Export Snowflake CSVs before running")
    parser.add_argument("--connection-name", help="Override the configured named Snowflake connection")
    parser.add_argument("--check", action="store_true", help="Validate setup/files without training")
    args = parser.parse_args(argv)
    overrides = {}
    if args.model_recipe:
        overrides["model"] = {"recipe": str(Path(args.model_recipe).resolve())}
    if args.output_dir:
        overrides["output_dir"] = str(Path(args.output_dir).resolve())
    session = None
    try:
        config = load_delivery_config(args.config, overrides=overrides)
        if args.raw_dir:
            config["data"].update(
                source="files", input_dir=str(Path(args.raw_dir).resolve()), file_format="csv"
            )
        if args.extract:
            from .raw_export import export_raw_data, read_export_queries

            if config["data"].get("kind") != "real":
                raise ValueError("Use the real-data configuration with --extract")
            extraction = config["extraction"]
            read_export_queries(extraction["sql_dir"])
            connection = args.connection_name or extraction.get("connection_name")
            if not connection:
                raise ValueError("Set extraction.connection_name or supply --connection-name")
            if args.check:
                print(json.dumps({"status": "CONFIGURATION_READY", "queries": 7,
                                  "snowflake_execution": "not performed"}, indent=2))
                return 0
            from snowflake.snowpark import Session

            session = Session.builder.config("connection_name", connection).create()
            exported = export_raw_data(
                session, config, extraction["sql_dir"], extraction["output_dir"],
                max_rows=int(extraction.get("max_rows_per_table", 1_000_000)),
            )
            config["data"]["input_dir"] = exported["raw_dir"]
        if args.check:
            from therapy_switch.io import load_claims_directory

            if config["data"]["source"] != "files":
                raise ValueError("--check requires a real/raw file configuration")
            from therapy_switch.data.raw_source_adapter import canonicalize_raw_tables

            tables = canonicalize_raw_tables(load_claims_directory(config), config)
            from therapy_switch.profiling import save_profile

            report = save_profile(tables, Path(config["delivery"]["artifact_dir"]) / "input_check")
            result = {"status": "RAW_CONTRACT_VALID", "rows": {k: len(v) for k, v in tables.items()},
                      "evidence_file": str(report.resolve())}
        else:
            result = run_delivery(config, mode=args.mode, model_artifact=args.model_artifact)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc))
        return 2
    finally:
        if session is not None:
            session.close()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
