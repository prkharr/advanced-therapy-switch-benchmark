"""Offline snapshot benchmark shared by raw and prepared input paths."""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from therapy_switch.config import load_config, validate_config
from therapy_switch.data import generate_synthetic_claims
from therapy_switch.data.event_sequences import encode_event_sequences, fit_sequence_vocabularies
from therapy_switch.data.prepared_input_adapter import load_prepared_inputs
from therapy_switch.data.raw_source_adapter import prepare_raw_inputs
from therapy_switch.data.splitting import (
    split_manifest,
    stratified_patient_split,
    temporal_patient_split,
)
from therapy_switch.evaluation import (
    all_models_cumulative_gains,
    all_models_decile_analysis,
    benchmark_row,
    bootstrap_all_models,
    build_executive_benchmark,
    build_model_benchmark,
    calibration_analysis,
    evaluate_predictions,
    explain_model,
    paired_bootstrap_comparison,
    plot_benchmark_comparison,
    plot_calibration_curves,
    plot_lift_and_gains,
    plot_roc_pr_curves,
    save_explanation_outputs,
    tune_threshold_on_validation,
)
from therapy_switch.evaluation.bootstrap import align_external_scores
from therapy_switch.evaluation.calibration import select_crossfit_calibrator
from therapy_switch.hcp.hcp_prioritization import build_snapshot_hcp_output
from therapy_switch.io import _read_frame, load_claims_directory, save_claims_directory, write_json
from therapy_switch.manifest import build_run_manifest, write_run_manifest
from therapy_switch.models import ModelRun, model_registry
from therapy_switch.reporting import Recommendation, write_recommendation_report
from therapy_switch.utils import configure_logging, set_global_seed


@dataclass
class ExperimentResult:
    name: str
    benchmark: pd.DataFrame
    executive_benchmark: pd.DataFrame
    results: dict
    predictions: dict
    validation_scores: dict
    paired_comparison: pd.DataFrame
    recommendation: Recommendation
    splits: dict
    sequence_splits: dict
    output_dir: Path


@dataclass
class PipelineResult:
    config: dict
    tables: dict
    cohort: pd.DataFrame
    features: pd.DataFrame
    experiments: dict
    primary_experiment: str
    output_dir: Path
    artifact_dir: Path


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _read_prepared_files(config):
    settings = config["data"]
    root = Path(settings["input_dir"])
    extension = settings.get("file_format", "parquet")
    result = {}
    for name in ("snapshots", "wide", "events"):
        path = Path(settings.get("tables", {}).get(name, {}).get("file", f"{name}.{extension}"))
        result[name] = _read_frame(path if path.is_absolute() else root / path, extension)
    return result


def prepare_inputs(config, *, session=None, tables=None):
    source = config["data"].get("source", "synthetic")
    if tables is None:
        if source == "synthetic":
            tables = generate_synthetic_claims(config)
            if config["data"].get("persist_generated_data", False):
                save_claims_directory(
                    tables, config["data"].get("synthetic_output_dir", "data/synthetic")
                )
        elif source == "files":
            tables = load_claims_directory(config)
        elif source == "prepared_files":
            tables = _read_prepared_files(config)
        elif source in {"snowflake_raw", "snowflake_prepared"}:
            from therapy_switch.data.snowflake_adapter import SnowflakeAdapter

            settings = config["data"]["snowflake"]
            adapter = SnowflakeAdapter(
                session,
                use_active_session=settings.get("use_active_session", False),
                max_rows=int(settings.get("max_rows_per_table", 1_000_000)),
            )
            tables = adapter.read_tables(settings["tables"])
        else:
            raise ValueError("Unsupported data.source")
    if source in {"prepared_files", "snowflake_prepared"}:
        return dict(tables), load_prepared_inputs(tables, config)
    return prepare_raw_inputs(tables, config)


def _model_options(config, key):
    models = config["models"]
    base = (
        "lightgbm"
        if key == "lightgbm_wide"
        else "gru"
        if key in {"gru_no_time", "gru_shuffled"}
        else key
    )
    options = {**models.get(base, {}), **models.get(key, {})}
    options.pop("enabled", None)
    if key in {
        "mlp",
        "lstm",
        "gru",
        "bilstm",
        "transformer",
        "hybrid",
        "gru_no_time",
        "gru_shuffled",
    }:
        options = {**models.get("sequence_training", {}), **options}
    if key in {"gru_no_time", "gru_shuffled"}:
        options["use_time"] = False
    if key == "gru_shuffled":
        options["shuffle_order"] = True
    if key in {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "lightgbm_wide",
    }:
        options["tune"] = config.get("tuning", {}).get("enabled", False)
        options["n_trials"] = config.get("tuning", {}).get("n_trials", 5)
    return options


def _columns_for(key, columns):
    if key != "lightgbm_wide":
        return list(columns)
    pattern = r"(_(30|60|90|180)d$|days_since|acceleration|ratio|recent_|previous_|coverage_change)"
    retained = [c for c in columns if not re.search(pattern, c)]
    if not retained:
        raise ValueError("No coarse wide features remain for the recency ablation")
    return retained


def _sequence_occlusion(result, sequence, vocabulary, maximum):
    from dataclasses import replace

    count = min(maximum, len(sequence.values))
    subset = replace(
        sequence,
        values=sequence.values[:count].copy(),
        mask=sequence.mask[:count].copy(),
        times=sequence.times[:count].copy(),
        event_dates=sequence.event_dates[:count].copy(),
        available_dates=sequence.available_dates[:count].copy(),
        index_dates=sequence.index_dates[:count],
        wide=None if sequence.wide is None else sequence.wide.iloc[:count],
    )
    baseline = result.estimator.predict_proba(subset)[:, 1]
    rows = []
    for family in ("all_codes", "pharmacy", "diagnosis", "procedure"):
        altered = replace(subset, values=subset.values.copy())
        positions = subset.mask.copy()
        if family != "all_codes":
            token_ids = [
                value for name, value in vocabulary["event_type"].items() if name.startswith(family)
            ]
            positions &= np.isin(subset.values[:, :, 0], token_ids)
        altered.values[:, :, 1][positions] = 1
        delta = baseline - result.estimator.predict_proba(altered)[:, 1]
        rows.append(
            {
                "model": result.model,
                "occluded": family,
                "rows": count,
                "mean_score_change": float(delta.mean()),
                "mean_absolute_score_change": float(np.abs(delta).mean()),
                "interpretation": "Code replacement sensitivity; association only, not causality or attention attribution",
            }
        )
    return rows


def _run_experiment(name, tables, inputs, config, output_root, artifact_root, logger):
    output, artifacts = output_root / "experiments" / name, artifact_root / "experiments" / name
    output.mkdir(parents=True, exist_ok=True)
    (artifacts / "models").mkdir(parents=True, exist_ok=True)
    frame = inputs.modeling_frame()
    splitter = temporal_patient_split if name == "temporal" else stratified_patient_split
    splits = splitter(frame, config)
    manifest = split_manifest(frame, splits)
    manifest.to_csv(output / "split_manifest.csv", index=False)
    summary = [
        {
            "split": key,
            "snapshots": len(value),
            "patients": value.patient_id.nunique(),
            "positives": int(value.label.sum()),
            "prevalence": float(value.label.mean()),
            "index_min": value.index_date.min(),
            "index_max": value.index_date.max(),
            "last_label_available": value.label_available_date.max(),
        }
        for key, value in splits.items()
    ]
    pd.DataFrame(summary).to_csv(output / "split_summary.csv", index=False)
    if any(value.label.nunique() != 2 for value in splits.values()):
        raise ValueError(
            "Each benchmark partition requires both classes; increase population or widen dates"
        )
    training_events = inputs.events.loc[inputs.events.snapshot_id.isin(splits["train"].snapshot_id)]
    vocabulary = fit_sequence_vocabularies(
        training_events, config["features"]["sequence"].get("vocab_min_frequency", 1)
    )
    write_json(vocabulary, artifacts / "sequence_vocabularies.json")
    sequences = {
        key: encode_event_sequences(inputs.events, value, config, vocabulary)
        .to_sequence_split()
        .validated(len(value))
        for key, value in splits.items()
    }
    results, model_inputs, rows, predictions, val_scores = {}, {}, {}, {}, {}
    seed = int(config["project"]["random_seed"])
    registry = model_registry()
    for key, runner in registry.items():
        if not config["models"].get(key, {}).get("enabled", False):
            rows[key] = benchmark_row(
                model=runner.model_name,
                category=runner.category,
                status="NOT APPLICABLE",
                reason="Disabled by configuration",
            )
            continue
        logger.info("Training %s on %s", runner.model_name, name)
        columns = _columns_for(key, inputs.feature_columns)
        X = {split: values[columns].copy() for split, values in splits.items()}
        options = _model_options(config, key)
        run = ModelRun(
            X["train"],
            splits["train"].label,
            X["validation"],
            splits["validation"].label,
            X["test"],
            splits["test"].label,
            sequence_train=copy.copy(sequences["train"]),
            sequence_val=copy.copy(sequences["validation"]),
            sequence_test=copy.copy(sequences["test"]),
            params={key: options},
            random_state=seed,
            artifact_dir=artifacts,
        )
        result = runner.run(run)
        results[key] = result
        if not result.succeeded:
            rows[key] = benchmark_row(
                model=result.model,
                category=result.category,
                status=result.status,
                reason=result.reason,
            )
            continue
        strategy = config["evaluation"].get("threshold_strategy", "max_f1")
        if strategy == "fixed":
            threshold = float(config["evaluation"].get("fixed_threshold", 0.5))
        else:
            threshold = tune_threshold_on_validation(
                splits["validation"].label,
                result.validation_probabilities,
                objective="target_recall" if strategy == "target_recall" else "f1",
                target_recall=config["evaluation"].get("target_recall", 0.7),
            ).threshold
        result.threshold = threshold
        result.test_predictions = (result.test_probabilities >= threshold).astype(np.int8)
        test_input = run.sequence_test if runner.requires_sequence else X["test"]
        model_inputs[key] = test_input
        # Reload verification is mandatory for every fitted estimator.
        path = artifacts / "models" / f"{key}.joblib"
        joblib.dump(result.estimator, path)
        if key != "naive_baseline":
            reloaded = joblib.load(path)
            actual = reloaded.predict_proba(test_input)[:, 1]
            np.testing.assert_allclose(actual, result.test_probabilities, rtol=1e-6, atol=1e-7)
        write_json(
            {
                "features": columns,
                "sequence_max_length": config["features"]["sequence"]["max_length"],
                "categorical_sizes": sequences["train"].categorical_sizes,
                "options": options,
                "threshold": threshold,
                "reload_verified": key != "naive_baseline",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            artifacts / "models" / f"{key}_metadata.json",
        )
        predictions[result.model] = result.test_probabilities
        val_scores[result.model] = float(result.validation_score)
        metrics = evaluate_predictions(
            splits["test"].label,
            result.test_probabilities,
            threshold=threshold,
            fractions=config["evaluation"]["top_fractions"],
        )
        rows[key] = benchmark_row(
            model=result.model,
            category=result.category,
            metrics=metrics,
            training_time=result.training_time_seconds,
            inference_time=result.inference_time_seconds,
        )
        rows[key].update(metrics)
        logger.info("%s completed; validation AP %.4f", result.model, result.validation_score)
    # Freeze selected model solely from validation evidence before reporting test comparisons.
    eligible = {k: r for k, r in results.items() if r.succeeded and k != "naive_baseline"}
    if not eligible:
        raise RuntimeError("No predictive model completed")
    winner = max(eligible, key=lambda k: (eligible[k].validation_score, k))
    selected = eligible[winner]
    recommendation = Recommendation(
        selected.model,
        "VALIDATION SELECTED CANDIDATE",
        "Candidate selected by validation average precision. Held-out comparisons are descriptive and do not change this choice.",
        max(
            (r for r in eligible.values() if r.category == "Classical ML"),
            key=lambda r: r.validation_score,
            default=selected,
        ).model,
        max(
            (r for r in eligible.values() if r.category == "Longitudinal DL"),
            key=lambda r: r.validation_score,
            default=selected,
        ).model,
        "PR-AUC",
        float(config.get("recommendation", {}).get("material_pr_auc_gain", 0.01)),
        "SYNTHETIC BENCHMARK RESULTS"
        if config["data"]["source"] == "synthetic"
        else "Held-out prepared/raw benchmark",
    )
    benchmark = build_model_benchmark(
        rows.values(), expected_models=[r.model_name for r in registry.values()]
    )
    detailed = pd.DataFrame(rows.values())
    detailed.to_csv(output / "model_metrics_detailed.csv", index=False)
    benchmark.to_csv(output / "model_benchmark.csv", index=False)
    executive = build_executive_benchmark(benchmark, recommended_model=selected.model)
    executive.to_csv(output / "executive_benchmark.csv", index=False)
    write_recommendation_report(recommendation, benchmark, output)
    write_json({k: r.summary() for k, r in results.items()}, artifacts / "model_runs.json")
    write_json(
        {
            "model": selected.model,
            "selected_on": "validation_average_precision",
            "scores": val_scores,
        },
        artifacts / "model_selection.json",
    )
    y_test = splits["test"].label.to_numpy()
    test_scores = splits["test"][["snapshot_id", "patient_id", "index_date", "label"]].copy()
    for model, scores in predictions.items():
        test_scores[_slug(model)] = scores
    test_scores.to_csv(output / "patient_test_predictions.csv", index=False)
    deciles = all_models_decile_analysis(y_test, predictions)
    gains = all_models_cumulative_gains(y_test, predictions)
    deciles.to_csv(output / "decile_analysis.csv", index=False)
    gains.to_csv(output / "cumulative_gains.csv", index=False)
    bootstrap_settings = dict(
        n_bootstrap=int(config["evaluation"].get("bootstrap_iterations", 100)),
        confidence_level=float(config["evaluation"].get("confidence_level", 0.95)),
        random_state=seed,
        patient_ids=splits["test"].patient_id.to_numpy(),
    )
    logger.info("Calculating patient-cluster uncertainty for %s", name)
    bootstrap_all_models(y_test, predictions, **bootstrap_settings).to_csv(
        output / "bootstrap_confidence_intervals.csv", index=False
    )
    reference = predictions.get("LightGBM")
    reference_name = "LightGBM"
    external = config["evaluation"].get("external_reference_file")
    if external:
        reference = align_external_scores(
            splits["test"], pd.read_csv(external, dtype={"snapshot_id": str})
        )
        reference_name = "External Reference"
    paired_frames = []
    if reference is not None:
        for model, scores in predictions.items():
            if model not in {reference_name, "Naive Baseline"}:
                paired_frames.append(
                    paired_bootstrap_comparison(
                        y_test,
                        reference,
                        scores,
                        classical_model=reference_name,
                        deep_learning_model=model,
                        **bootstrap_settings,
                    )
                )
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    paired.to_csv(output / "paired_model_comparison.csv", index=False)
    if len(paired):
        paired.loc[paired.metric.str.startswith("TruePositives")].to_csv(
            output / "additional_true_positives.csv", index=False
        )
    calibrator, calibration_selection = select_crossfit_calibrator(
        splits["validation"].label,
        selected.validation_probabilities,
        splits["validation"].patient_id,
        methods=config["evaluation"].get("calibrators", ["none"]),
        random_state=seed,
    )
    calibration_selection.to_csv(output / "calibration_selection.csv", index=False)
    calibrated = (
        selected.test_probabilities
        if calibrator is None
        else calibrator.transform(selected.test_probabilities)
    )
    if calibrator is not None:
        joblib.dump(calibrator, artifacts / "calibrator.joblib")
        np.testing.assert_allclose(
            joblib.load(artifacts / "calibrator.joblib").transform(selected.test_probabilities),
            calibrated,
        )
    calibration_rows, curves = [], []
    for model, scores in {
        **predictions,
        f"{selected.model} Selected Calibration": calibrated,
    }.items():
        summary, curve = calibration_analysis(
            y_test, scores, model=model, n_bins=config["evaluation"].get("calibration_bins", 10)
        )
        calibration_rows.append(summary)
        curves.append(curve)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_summary.csv", index=False)
    pd.concat(curves).to_csv(output / "calibration_curve.csv", index=False)
    patient_scores = splits["test"][["snapshot_id", "patient_id", "index_date"]].copy()
    patient_scores["raw_propensity_score"] = selected.test_probabilities
    patient_scores["advanced_therapy_propensity_score"] = calibrated
    patient_scores["model"], patient_scores["score_scope"] = selected.model, "held_out_test"
    patient_scores.to_csv(output / "patient_propensity_scores.csv", index=False)
    targeting, attribution = build_snapshot_hcp_output(
        patient_scores, inputs.events, config.get("hcp", {})
    )
    targeting.to_csv(output / "hcp_targeting_output.csv", index=False)
    attribution.to_csv(output / "patient_hcp_attribution.csv", index=False)
    if config.get("explainability", {}).get("enabled", True):
        logger.info("Generating post-hoc explanations for %s", name)
        settings = config["explainability"]
        statuses, occlusion = [], []
        for key, result in eligible.items():
            if registry[key].requires_sequence:
                occlusion.extend(
                    _sequence_occlusion(
                        result, model_inputs[key], vocabulary, settings.get("max_samples", 100)
                    )
                )
                statuses.append(
                    {"model": result.model, "status": "COMPLETED", "method": "code occlusion"}
                )
            else:
                explanation = explain_model(
                    result.estimator,
                    model_inputs[key],
                    y_test,
                    max_samples=settings.get("max_samples", 100),
                    prefer_shap=settings.get("prefer_shap", True),
                    n_permutation_repeats=settings.get("permutation_repeats", 2),
                    random_state=seed,
                )
                save_explanation_outputs(
                    explanation, model_name=result.model, output_dir=output / "explainability"
                )
                statuses.append(
                    {
                        "model": result.model,
                        "status": explanation.status,
                        "method": explanation.method,
                        "reason": explanation.reason,
                    }
                )
        pd.DataFrame(statuses).to_csv(output / "explainability_status.csv", index=False)
        pd.DataFrame(occlusion).to_csv(output / "sequence_occlusion.csv", index=False)
    if config.get("visualizations", {}).get("enabled", True):
        figures = output / "figures"
        plot_benchmark_comparison(benchmark, output_path=figures / "model_benchmark.png")
        plot_roc_pr_curves(
            y_test, predictions, roc_path=figures / "roc.png", pr_path=figures / "pr.png"
        )
        plot_lift_and_gains(y_test, predictions, output_path=figures / "gains.png")
        plot_calibration_curves(y_test, predictions, output_path=figures / "calibration.png")
    manifest = build_run_manifest(
        config, tables, inputs.snapshots, experiment=name, split_frames=splits
    )
    manifest["split_manifest_sha256"] = hashlib.sha256(
        (output / "split_manifest.csv").read_bytes()
    ).hexdigest()
    manifest["feature_columns"] = list(inputs.feature_columns)
    manifest["sequence_preprocessing"] = {
        "vocabulary_fit": "training only",
        "max_length": config["features"]["sequence"]["max_length"],
        "truncation": "most recent, stable chronological ties",
        "padding": "right",
        "unknown_token": 1,
    }
    manifest["integration"] = {
        "implementation": "IMPLEMENTED",
        "environment_verification": "NOT VERIFIED IN SENTINEL",
    }
    write_run_manifest(manifest, artifacts / "run_manifest.json")
    return ExperimentResult(
        name,
        benchmark,
        executive,
        results,
        predictions,
        val_scores,
        paired,
        recommendation,
        splits,
        sequences,
        output,
    )


def run_pipeline(config_or_path, *, session=None, tables=None):
    config = (
        copy.deepcopy(dict(config_or_path))
        if isinstance(config_or_path, dict)
        else load_config(config_or_path)
    )
    validate_config(config)
    output, artifacts = (
        Path(config["project"]["output_dir"]),
        Path(config["project"]["artifact_dir"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output)
    set_global_seed(int(config["project"]["random_seed"]))
    write_json(config, artifacts / "resolved_config.json")
    try:
        logger.info("Preparing %s data", config["data"]["source"])
        raw, inputs = prepare_inputs(config, session=session, tables=tables)
        for name, frame in [
            ("snapshots", inputs.snapshots),
            ("wide_features", inputs.wide),
            ("event_history", inputs.events),
        ]:
            persisted = frame.copy()
            persisted.attrs = {}
            persisted.to_parquet(artifacts / f"{name}.parquet", index=False)
        exclusions = pd.DataFrame(inputs.snapshots.attrs.get("exclusions", []))
        exclusions.to_csv(output / "cohort_exclusions.csv", index=False)
        write_json(
            {
                "source": config["data"]["source"],
                "raw_rows": {k: len(v) for k, v in raw.items()},
                "snapshots": len(inputs.snapshots),
                "patients": inputs.snapshots.patient_id.nunique(),
                "positive_snapshots": int(inputs.snapshots.label.sum()),
                "prevalence": float(inputs.snapshots.label.mean()),
                "feature_count": len(inputs.feature_columns),
                "event_rows": len(inputs.events),
                "missing_wide": inputs.wide.isna().sum().to_dict(),
                "excluded_snapshots": len(exclusions),
            },
            output / "data_quality_report.json",
        )
        pd.DataFrame(
            [
                {
                    "patients": inputs.snapshots.patient_id.nunique(),
                    "snapshots": len(inputs.snapshots),
                    "positives": int(inputs.snapshots.label.sum()),
                    "prevalence": float(inputs.snapshots.label.mean()),
                }
            ]
        ).to_csv(output / "cohort_summary.csv", index=False)
        experiments = {}
        for name in config["splitting"]["experiments"]:
            experiments[name] = _run_experiment(
                name, raw, inputs, config, output, artifacts, logger
            )
        primary = config["splitting"]["primary_experiment"]
        for path in experiments[primary].output_dir.glob("*.csv"):
            (output / path.name).write_bytes(path.read_bytes())
        for filename in ("model_recommendation.json", "model_recommendation.md"):
            (output / filename).write_bytes(
                (experiments[primary].output_dir / filename).read_bytes()
            )
        (output / "run_manifest.json").write_bytes(
            (artifacts / "experiments" / primary / "run_manifest.json").read_bytes()
        )
        logger.info("Synthetic/offline benchmark completed")
        return PipelineResult(
            config, raw, inputs.snapshots, inputs.wide, experiments, primary, output, artifacts
        )
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
