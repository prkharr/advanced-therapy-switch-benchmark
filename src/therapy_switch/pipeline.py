"""End-to-end orchestration for leakage-safe claims model benchmarking."""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import numpy as np
import pandas as pd

from therapy_switch.config import load_config, validate_config
from therapy_switch.data import (
    build_cohort,
    build_event_frame,
    build_event_sequences,
    fit_sequence_vocabularies,
    generate_synthetic_claims,
    stratified_patient_split,
    temporal_patient_split,
    validate_cohort_timeline,
)
from therapy_switch.evaluation import (
    ProbabilityCalibrator,
    all_models_cumulative_gains,
    all_models_decile_analysis,
    benchmark_row,
    bootstrap_all_models,
    build_executive_benchmark,
    build_model_benchmark,
    calibration_analysis,
    class_imbalance_summary,
    evaluate_predictions,
    explain_model,
    not_applicable_row,
    paired_bootstrap_comparison,
    plot_benchmark_comparison,
    plot_calibration_curves,
    plot_decile_performance,
    plot_lift_and_gains,
    plot_roc_pr_curves,
    plot_top_fraction_lift,
    save_explanation_outputs,
)
from therapy_switch.features import build_tabular_features
from therapy_switch.hcp import (
    AttributionConfig,
    HCPScoringConfig,
    OpportunityConfig,
    build_hcp_targeting_output,
)
from therapy_switch.io import load_claims_directory, save_claims_directory, write_json
from therapy_switch.manifest import build_run_manifest, write_run_manifest
from therapy_switch.models import ModelResult, model_registry, run_benchmark
from therapy_switch.reporting import (
    CLASSICAL_MODELS,
    SEQUENCE_DL_MODELS,
    Recommendation,
    make_recommendation,
    write_recommendation_report,
)
from therapy_switch.schemas import validate_tables
from therapy_switch.utils import configure_logging, set_global_seed

ID_COLUMNS = {"patient_id", "index_date", "label", "outcome", "outcome_date"}
SEQUENCE_MODEL_NAMES = set(SEQUENCE_DL_MODELS)


@dataclass
class ExperimentResult:
    name: str
    benchmark: pd.DataFrame
    executive_benchmark: pd.DataFrame
    results: Dict[str, ModelResult]
    predictions: Dict[str, np.ndarray]
    validation_scores: Dict[str, float]
    paired_comparison: pd.DataFrame
    recommendation: Recommendation
    splits: Dict[str, pd.DataFrame]
    sequence_splits: Dict[str, Any]
    full_sequence: Any | None
    selected_calibrator: ProbabilityCalibrator | None
    output_dir: Path


@dataclass
class PipelineResult:
    config: Dict[str, Any]
    tables: Dict[str, pd.DataFrame]
    cohort: pd.DataFrame
    features: pd.DataFrame
    experiments: Dict[str, ExperimentResult]
    primary_experiment: str
    output_dir: Path
    artifact_dir: Path


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _load_or_generate_tables(config: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    source = str(config["data"].get("source", "synthetic")).lower()
    if source == "synthetic":
        tables = generate_synthetic_claims(config)
        if bool(config["data"].get("persist_generated_data", False)):
            synthetic_dir = Path(config["data"].get("synthetic_output_dir", "data/synthetic"))
            save_claims_directory(
                tables,
                synthetic_dir,
                file_format=str(config["data"].get("file_format", "csv")),
            )
    elif source == "files":
        tables = load_claims_directory(config)
    else:
        raise ValueError("data.source must be 'synthetic' or 'files'.")
    validate_tables(tables)
    return tables


def _split_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    splitting = dict(config["splitting"])
    validation_fraction = float(splitting["validation_fraction"])
    test_fraction = float(splitting["test_fraction"])
    return {
        "train_fraction": 1.0 - validation_fraction - test_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "random_seed": int(config["project"].get("random_seed", 42)),
    }


def _make_splits(
    features: pd.DataFrame, experiment: str, config: Mapping[str, Any]
) -> Dict[str, pd.DataFrame]:
    split_settings = _split_config(config)
    if experiment == "stratified":
        return stratified_patient_split(features, split_settings)
    if experiment == "temporal":
        return temporal_patient_split(features, split_settings)
    raise ValueError(f"Unknown split experiment: {experiment}")


def _model_features(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in frame.columns if column not in ID_COLUMNS]
    if not columns:
        raise ValueError("No model features remain after identifier/target removal.")
    return frame[columns].copy()


def _model_configuration(config: Mapping[str, Any]) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    registry = model_registry()
    model_section = config["models"]
    evaluation = config["evaluation"]
    tuning = config.get("tuning", {})
    sequence_training = dict(model_section.get("sequence_training", {}))
    threshold_strategy = str(evaluation.get("threshold_strategy", "max_f1")).lower()
    if threshold_strategy in {"max_f1", "maximum_f1"}:
        runner_threshold = "f1"
    elif threshold_strategy == "fixed":
        runner_threshold = "fixed"
    else:
        # Runner-level selection intentionally supports only deterministic F1 or
        # fixed thresholds. More specialized operating points remain available
        # through evaluation.tune_threshold_on_validation.
        runner_threshold = "f1"

    params: Dict[str, Dict[str, Any]] = {}
    enabled: list[str] = []
    for key in registry:
        settings = dict(model_section.get(key, {}))
        is_enabled = bool(settings.pop("enabled", True))
        if key in {"lstm", "gru", "bilstm", "transformer"}:
            settings = {**sequence_training, **settings}
        settings["threshold_strategy"] = runner_threshold
        settings["fixed_threshold"] = float(evaluation.get("fixed_threshold", 0.5))
        if key not in {"naive_baseline", "mlp", "lstm", "gru", "bilstm", "transformer"}:
            settings["tune"] = bool(tuning.get("enabled", False))
            settings["n_trials"] = int(tuning.get("n_trials", 25))
            settings["prefer_optuna"] = str(tuning.get("engine", "auto")) != "randomized_search"
        params[key] = settings
        if is_enabled:
            enabled.append(key)
    return params, enabled


def _build_sequence_inputs(
    tables: Mapping[str, pd.DataFrame],
    splits: Mapping[str, pd.DataFrame],
    full_features: pd.DataFrame,
    config: Mapping[str, Any],
    artifact_dir: Path,
) -> tuple[Dict[str, Any], Any | None, str | None]:
    """Fit token vocabularies on training only and audit every sequence split."""

    try:
        train_cohort = splits["train"][["patient_id", "index_date", "label"]].copy()
        training_events = build_event_frame(tables, train_cohort, config)
        if training_events.empty:
            return {}, None, "No pre-index training events were available."
        vocabularies = fit_sequence_vocabularies(training_events)
        write_json(vocabularies, artifact_dir / "sequence_vocabularies.json")
        sequence_inputs: Dict[str, Any] = {}
        for split_name in ("train", "validation", "test"):
            split_cohort = splits[split_name][["patient_id", "index_date", "label"]]
            dataset = build_event_sequences(tables, split_cohort, config, vocabularies=vocabularies)
            sequence = dataset.to_sequence_split().validated(expected_rows=len(split_cohort))
            sequence_inputs[split_name] = sequence
        full_cohort = full_features[["patient_id", "index_date", "label"]]
        full_dataset = build_event_sequences(tables, full_cohort, config, vocabularies=vocabularies)
        full_sequence = full_dataset.to_sequence_split().validated(expected_rows=len(full_cohort))
        return sequence_inputs, full_sequence, None
    except Exception as exc:
        return {}, None, f"Sequence construction failed: {type(exc).__name__}: {exc}"


def _save_model_artifacts(
    results: Mapping[str, ModelResult], artifact_dir: Path, logger: logging.Logger
) -> None:
    model_dir = artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Any] = {}
    for key, result in results.items():
        summaries[key] = result.summary()
        if not result.succeeded or result.estimator is None:
            continue
        path = model_dir / f"{_safe_slug(result.model)}.joblib"
        try:
            joblib.dump(result.estimator, path)
        except Exception as exc:
            logger.warning("Could not serialize %s: %s", result.model, exc)
            summaries[key]["Artifact serialization"] = f"FAILED: {type(exc).__name__}: {exc}"
    write_json(summaries, artifact_dir / "model_runs.json")


def _empty_paired_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "metric",
            "classical_model",
            "deep_learning_model",
            "classical_estimate",
            "deep_learning_estimate",
            "difference_dl_minus_classical",
            "ci_lower",
            "ci_upper",
            "confidence_level",
            "p_value_two_sided",
            "statistically_significant",
            "successful_samples",
        ]
    )


def _validation_best(names: tuple[str, ...], validation_scores: Mapping[str, float]) -> str | None:
    candidates = [
        (name, float(validation_scores.get(name, np.nan)))
        for name in names
        if np.isfinite(float(validation_scores.get(name, np.nan)))
    ]
    return max(candidates, key=lambda item: (item[1], item[0]))[0] if candidates else None


def _calibration_outputs(
    selected_model: str | None,
    results: Mapping[str, ModelResult],
    y_validation: np.ndarray,
    y_test: np.ndarray,
    config: Mapping[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> ProbabilityCalibrator | None:
    if selected_model is None:
        pd.DataFrame(columns=["model", "method", "validation_brier", "status", "reason"]).to_csv(
            output_dir / "calibration_selection.csv", index=False
        )
        return None
    result = next((item for item in results.values() if item.model == selected_model), None)
    if result is None or not result.succeeded:
        return None
    raw_validation = np.asarray(result.validation_probabilities, dtype=float)
    raw_test = np.asarray(result.test_probabilities, dtype=float)
    methods = [
        str(method)
        for method in config["evaluation"].get("calibrators", ["none", "sigmoid", "isotonic"])
    ]
    rows: list[dict[str, Any]] = [
        {
            "model": selected_model,
            "method": "none",
            "validation_brier": float(np.mean((raw_validation - y_validation) ** 2)),
            "status": "COMPLETED",
            "reason": "Uncalibrated reference",
        }
    ]
    fitted: Dict[str, ProbabilityCalibrator] = {}
    validation_outputs: Dict[str, np.ndarray] = {"none": raw_validation}
    test_outputs: Dict[str, np.ndarray] = {"uncalibrated": raw_test}
    for configured_method in methods:
        if configured_method.lower() in {"none", "uncalibrated"}:
            continue
        try:
            calibrator = ProbabilityCalibrator(method=configured_method).fit(
                y_validation, raw_validation, dataset_role="validation"
            )
            validation_probability = calibrator.transform(raw_validation)
            test_probability = calibrator.transform(raw_test)
            method = calibrator.method
            fitted[method] = calibrator
            validation_outputs[method] = validation_probability
            test_outputs[method] = test_probability
            rows.append(
                {
                    "model": selected_model,
                    "method": method,
                    "validation_brier": float(
                        np.mean((validation_probability - y_validation) ** 2)
                    ),
                    "status": "COMPLETED",
                    "reason": "Fit and selected using validation data only",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": selected_model,
                    "method": configured_method,
                    "validation_brier": np.nan,
                    "status": "NOT APPLICABLE",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    selection = pd.DataFrame(rows)
    completed = selection.loc[selection["status"].eq("COMPLETED")]
    selected_method = str(completed.sort_values("validation_brier").iloc[0]["method"])
    selection["selected"] = selection["method"].eq(selected_method)
    selection.to_csv(output_dir / "calibration_selection.csv", index=False)

    summaries: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    for method, probabilities in test_outputs.items():
        summary, curve = calibration_analysis(
            y_test,
            probabilities,
            model=f"{selected_model} ({method})",
            n_bins=int(config["evaluation"].get("calibration_bins", 10)),
        )
        summaries.append(summary)
        curves.append(curve)
    pd.DataFrame(summaries).to_csv(output_dir / "calibration_summary.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output_dir / "calibration_curve.csv", index=False)
    try:
        plot_calibration_curves(
            y_test,
            test_outputs,
            n_bins=int(config["evaluation"].get("calibration_bins", 10)),
            output_path=output_dir / "figures" / "calibration_curves.png",
        )
    except Exception as exc:
        logger.warning("Calibration plot unavailable: %s", exc)
    return fitted.get(selected_method)


def _run_explainability(
    results: Mapping[str, ModelResult],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    config: Mapping[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    settings = config.get("explainability", {})
    if not bool(settings.get("enabled", True)):
        return
    rows: list[dict[str, Any]] = []
    for result in results.values():
        if not result.succeeded:
            rows.append(
                {
                    "model": result.model,
                    "status": result.status,
                    "method": "none",
                    "reason": result.reason,
                }
            )
            continue
        if result.model in SEQUENCE_MODEL_NAMES:
            rows.append(
                {
                    "model": result.model,
                    "status": "NOT APPLICABLE",
                    "method": "sequence-specific attribution not configured",
                    "reason": (
                        "The generic tabular explainer is invalid for event tensors; use an "
                        "approved integrated-gradients/attention analysis before production."
                    ),
                }
            )
            continue
        if result.model == "Naive Baseline":
            continue
        explanation = explain_model(
            result.estimator,
            X_test,
            y_test,
            prefer_shap=bool(settings.get("prefer_shap", True)),
            max_samples=int(settings.get("max_samples", 1000)),
            n_permutation_repeats=int(settings.get("permutation_repeats", 5)),
            random_state=int(config["project"].get("random_seed", 42)),
        )
        try:
            save_explanation_outputs(
                explanation,
                model_name=result.model,
                output_dir=output_dir / "explainability",
            )
        except Exception as exc:
            logger.warning("Could not save explanation for %s: %s", result.model, exc)
        rows.append(
            {
                "model": result.model,
                "status": explanation.status,
                "method": explanation.method,
                "reason": explanation.reason,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "explainability_status.csv", index=False)


def _run_visualizations(
    y_test: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    benchmark: pd.DataFrame,
    deciles: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if not predictions:
        return
    figures = output_dir / "figures"
    calls = [
        lambda: plot_roc_pr_curves(
            y_test,
            predictions,
            roc_path=figures / "roc_curves.png",
            pr_path=figures / "pr_curves.png",
        ),
        lambda: plot_lift_and_gains(
            y_test, predictions, output_path=figures / "lift_and_gains.png"
        ),
        lambda: plot_top_fraction_lift(
            y_test, predictions, output_path=figures / "top_fraction_lift.png"
        ),
        lambda: plot_decile_performance(deciles, output_path=figures / "decile_performance.png"),
        lambda: plot_benchmark_comparison(benchmark, output_path=figures / "model_benchmark.png"),
    ]
    for call in calls:
        try:
            call()
        except Exception as exc:
            logger.warning("Visualization unavailable: %s", exc)


def _run_experiment(
    name: str,
    tables: Mapping[str, pd.DataFrame],
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    config: Mapping[str, Any],
    output_root: Path,
    artifact_root: Path,
    logger: logging.Logger,
) -> ExperimentResult:
    output_dir = output_root / "experiments" / name
    artifact_dir = artifact_root / "experiments" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting %s experiment", name)
    splits = _make_splits(features, name, config)
    split_summary = []
    for split_name, frame in splits.items():
        summary = class_imbalance_summary(frame["label"])
        summary.update(
            {
                "split": split_name,
                "index_date_min": pd.to_datetime(frame["index_date"]).min(),
                "index_date_max": pd.to_datetime(frame["index_date"]).max(),
            }
        )
        split_summary.append(summary)
    pd.DataFrame(split_summary).to_csv(output_dir / "split_summary.csv", index=False)

    sequence_splits, full_sequence, sequence_reason = _build_sequence_inputs(
        tables, splits, features, config, artifact_dir
    )
    if sequence_reason:
        logger.warning("%s", sequence_reason)
    params, enabled_keys = _model_configuration(config)
    y_train = splits["train"]["label"].to_numpy(dtype=int)
    y_validation = splits["validation"]["label"].to_numpy(dtype=int)
    y_test = splits["test"]["label"].to_numpy(dtype=int)
    X_train = _model_features(splits["train"])
    X_validation = _model_features(splits["validation"])
    X_test = _model_features(splits["test"])

    results = run_benchmark(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        sequence_train=sequence_splits.get("train"),
        sequence_val=sequence_splits.get("validation"),
        sequence_test=sequence_splits.get("test"),
        params=params,
        model_keys=enabled_keys,
        random_state=int(config["project"].get("random_seed", 42)),
        artifact_dir=artifact_dir,
    )
    _save_model_artifacts(results, artifact_dir, logger)

    registry = model_registry()
    rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    predictions: Dict[str, np.ndarray] = {}
    validation_scores: Dict[str, float] = {}
    thresholds: list[dict[str, Any]] = []
    for key, runner in registry.items():
        result = results.get(key)
        if result is None:
            rows.append(
                not_applicable_row(
                    runner.model_name,
                    "Disabled by configuration.",
                    category=runner.category,
                )
            )
            continue
        logger.info("%s [%s]: %s", result.model, name, result.status)
        if result.succeeded:
            scores = np.asarray(result.test_probabilities, dtype=float)
            metrics = evaluate_predictions(
                y_test,
                scores,
                threshold=float(result.threshold),
                fractions=config["evaluation"].get("top_fractions", [0.01, 0.05, 0.1, 0.2, 0.3]),
            )
            rows.append(
                benchmark_row(
                    model=result.model,
                    category=result.category,
                    metrics=metrics,
                    training_time=result.training_time_seconds,
                    inference_time=result.inference_time_seconds,
                    status=result.status,
                )
            )
            detailed_rows.append({"Model": result.model, **metrics})
            predictions[result.model] = scores
            validation_scores[result.model] = float(result.validation_score)
            thresholds.append(
                {
                    "Model": result.model,
                    "threshold": result.threshold,
                    "selected_on": "validation",
                    "strategy": params[key].get("threshold_strategy"),
                }
            )
        else:
            rows.append(
                benchmark_row(
                    model=result.model,
                    category=result.category,
                    status=result.status,
                    reason=result.reason or "Model did not produce valid predictions.",
                )
            )
    benchmark = build_model_benchmark(rows)
    pd.DataFrame(detailed_rows).to_csv(output_dir / "model_metrics_detailed.csv", index=False)
    pd.DataFrame(thresholds).to_csv(output_dir / "selected_thresholds.csv", index=False)

    deciles = all_models_decile_analysis(y_test, predictions)
    gains = all_models_cumulative_gains(y_test, predictions)
    deciles.to_csv(output_dir / "decile_analysis.csv", index=False)
    gains.to_csv(output_dir / "cumulative_gains.csv", index=False)

    test_scores = splits["test"][["patient_id", "index_date", "label"]].copy()
    for model, scores in predictions.items():
        test_scores[_safe_slug(model)] = scores
    test_scores.to_csv(output_dir / "patient_test_predictions.csv", index=False)

    bootstrap_iterations = int(config["evaluation"].get("bootstrap_iterations", 1000))
    confidence_level = float(config["evaluation"].get("confidence_level", 0.95))
    if predictions and np.unique(y_test).size == 2:
        confidence = bootstrap_all_models(
            y_test,
            predictions,
            n_bootstrap=bootstrap_iterations,
            confidence_level=confidence_level,
            random_state=int(config["project"].get("random_seed", 42)),
        )
    else:
        confidence = pd.DataFrame()
    confidence.to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)

    best_classical = _validation_best(CLASSICAL_MODELS, validation_scores)
    best_dl = _validation_best(SEQUENCE_DL_MODELS, validation_scores)
    paired = _empty_paired_frame()
    if best_classical and best_dl and np.unique(y_test).size == 2:
        paired = paired_bootstrap_comparison(
            y_test,
            predictions[best_classical],
            predictions[best_dl],
            classical_model=best_classical,
            deep_learning_model=best_dl,
            n_bootstrap=bootstrap_iterations,
            confidence_level=confidence_level,
            random_state=int(config["project"].get("random_seed", 42)),
        )
    paired.to_csv(output_dir / "paired_model_comparison.csv", index=False)

    recommendation = make_recommendation(
        benchmark,
        validation_scores,
        paired,
        material_pr_auc_gain=float(
            config.get("recommendation", {}).get("material_pr_auc_gain", 0.01)
        ),
        data_source=str(config["data"].get("source", "synthetic")),
    )
    executive = build_executive_benchmark(benchmark, recommended_model=recommendation.model)
    benchmark.to_csv(output_dir / "model_benchmark.csv", index=False)
    executive.to_csv(output_dir / "executive_benchmark.csv", index=False)
    write_recommendation_report(recommendation, benchmark, output_dir)

    tabular_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "MLP",
    ]
    benchmark.loc[benchmark["Model"].isin(tabular_names)].to_csv(
        output_dir / "tabular_comparison.csv", index=False
    )
    longitudinal_names = [name for name in [best_classical, *SEQUENCE_DL_MODELS] if name]
    benchmark.loc[benchmark["Model"].isin(longitudinal_names)].to_csv(
        output_dir / "longitudinal_comparison.csv", index=False
    )

    selected_calibrator = _calibration_outputs(
        recommendation.model,
        results,
        y_validation,
        y_test,
        config,
        output_dir,
        logger,
    )
    _run_explainability(results, X_test, y_test, config, output_dir, logger)
    if bool(config.get("visualizations", {}).get("enabled", True)):
        _run_visualizations(y_test, predictions, benchmark, deciles, output_dir, logger)

    manifest = build_run_manifest(
        config,
        tables,
        cohort,
        experiment=name,
        split_frames=splits,
    )
    if sequence_reason:
        manifest["sequence_status"] = {"status": "NOT APPLICABLE", "reason": sequence_reason}
    else:
        manifest["sequence_status"] = {"status": "AVAILABLE", "reason": None}
    write_run_manifest(manifest, artifact_dir / "run_manifest.json")
    logger.info("Completed %s experiment", name)
    return ExperimentResult(
        name=name,
        benchmark=benchmark,
        executive_benchmark=executive,
        results=results,
        predictions=predictions,
        validation_scores=validation_scores,
        paired_comparison=paired,
        recommendation=recommendation,
        splits=splits,
        sequence_splits=sequence_splits,
        full_sequence=full_sequence,
        selected_calibrator=selected_calibrator,
        output_dir=output_dir,
    )


def _provider_events(
    tables: Mapping[str, pd.DataFrame],
    cohort: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Build pre-index provider interactions for the separate HCP layer."""

    observation_days = int(config["timeline"]["observation_window_days"])
    landmarks = cohort[["patient_id", "index_date"]].copy()
    providers = tables["providers"][["provider_id", "specialty"]].drop_duplicates("provider_id")
    medical = tables["medical_claims"].merge(
        landmarks, on="patient_id", how="inner", validate="many_to_one"
    )
    medical["event_date"] = pd.to_datetime(medical["claim_date"])
    medical["days_before"] = (medical["index_date"] - medical["event_date"]).dt.days
    medical = medical.loc[medical["days_before"].between(0, observation_days)].copy()
    medical = medical.merge(providers, on="provider_id", how="left")
    medical["role"] = "visit"
    relevant_specialties = {
        str(value).lower() for value in config.get("hcp", {}).get("relevant_specialties", [])
    }
    medical["is_relevant"] = (
        medical["specialty"].fillna("").astype(str).str.lower().isin(relevant_specialties)
        if relevant_specialties
        else True
    )

    pharmacy = tables["pharmacy_claims"].merge(
        landmarks, on="patient_id", how="inner", validate="many_to_one"
    )
    pharmacy["event_date"] = pd.to_datetime(pharmacy["fill_date"])
    pharmacy["days_before"] = (pharmacy["index_date"] - pharmacy["event_date"]).dt.days
    pharmacy = pharmacy.loc[pharmacy["days_before"].between(0, observation_days)].copy()
    pharmacy = pharmacy.rename(columns={"prescriber_id": "provider_id"})
    pharmacy = pharmacy.merge(providers, on="provider_id", how="left")
    pharmacy["role"] = "prescriber"
    conventional = set(config["therapy_mapping"]["conventional"])
    pharmacy["is_relevant"] = pharmacy["drug_id"].isin(conventional)
    columns = [
        "patient_id",
        "provider_id",
        "event_date",
        "specialty",
        "role",
        "is_relevant",
    ]
    return pd.concat([medical[columns], pharmacy[columns]], ignore_index=True)


def _score_all_patients(
    experiment: ExperimentResult,
    features: pd.DataFrame,
) -> pd.DataFrame | None:
    selected_model = experiment.recommendation.model
    if selected_model is None:
        return None
    result = next(
        (item for item in experiment.results.values() if item.model == selected_model), None
    )
    if result is None or not result.succeeded or result.estimator is None:
        return None
    if selected_model in SEQUENCE_MODEL_NAMES:
        if experiment.full_sequence is None:
            return None
        raw_scores = result.estimator.predict_proba(experiment.full_sequence)[:, 1]
    else:
        raw_scores = result.estimator.predict_proba(_model_features(features))[:, 1]
    final_scores = np.asarray(raw_scores, dtype=float)
    if experiment.selected_calibrator is not None:
        final_scores = experiment.selected_calibrator.transform(final_scores)
    output = features[["patient_id", "index_date", "label"]].copy()
    output["raw_propensity_score"] = np.asarray(raw_scores, dtype=float)
    output["advanced_therapy_propensity_score"] = final_scores
    output["model"] = selected_model
    output["calibrated"] = experiment.selected_calibrator is not None
    return output


def _run_hcp_layer(
    tables: Mapping[str, pd.DataFrame],
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    experiment: ExperimentResult,
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    patient_scores = _score_all_patients(experiment, features)
    if patient_scores is None:
        write_json(
            {
                "status": "NOT APPLICABLE",
                "reason": "No deployable patient model was available for full-cohort scoring.",
            },
            output_dir / "hcp_status.json",
        )
        return
    patient_scores.to_csv(output_dir / "patient_propensity_scores.csv", index=False)
    hcp_config = config.get("hcp", {})
    events = _provider_events(tables, cohort, config)
    percentile = float(hcp_config.get("high_propensity_threshold_percentile", 90))
    high_threshold = float(
        np.percentile(patient_scores["advanced_therapy_propensity_score"], percentile)
    )
    weights_config = hcp_config.get("opportunity_score", {})
    weights = {
        "expected_switchers": float(weights_config.get("expected_switchers_weight", 0.60)),
        "high_propensity_patient_count": float(
            weights_config.get("high_propensity_patients_weight", 0.25)
        ),
        "eligible_patient_count": float(weights_config.get("eligible_patients_weight", 0.15)),
    }
    targeting, attribution = build_hcp_targeting_output(
        patient_scores,
        events,
        attribution_config=AttributionConfig(
            method=str(hcp_config.get("attribution_rule", "most_recent_relevant_prescriber")),
            hcp_col="provider_id",
            date_col="event_date",
            specialty_col="specialty",
            relevant_col="is_relevant",
            role_col="role",
            specialist_specialties=tuple(hcp_config.get("relevant_specialties", [])),
        ),
        opportunity_config=OpportunityConfig(high_propensity_threshold=high_threshold),
        scoring_config=HCPScoringConfig(weights=weights, normalization="percentile"),
    )
    provider_details = tables["providers"].rename(columns={"provider_id": "hcp_id"})
    targeting = targeting.merge(provider_details, on="hcp_id", how="left")
    targeting.to_csv(output_dir / "hcp_targeting_output.csv", index=False)
    attribution.to_csv(output_dir / "patient_hcp_attribution.csv", index=False)
    write_json(
        {
            "status": "COMPLETED",
            "attribution_rule": hcp_config.get("attribution_rule"),
            "eligible_patients": len(patient_scores),
            "attributed_patients": len(attribution),
            "attribution_coverage": len(attribution) / max(1, len(patient_scores)),
            "high_propensity_threshold": high_threshold,
            "opportunity_score_weights": weights,
            "opportunity_score_normalization": "percentile",
            "formula": "weighted sum of normalized HCP opportunity components",
        },
        output_dir / "hcp_status.json",
    )


def _write_cross_experiment_outputs(
    experiments: Mapping[str, ExperimentResult], output_dir: Path
) -> None:
    frames = []
    for name, result in experiments.items():
        frame = result.benchmark.copy()
        frame.insert(0, "Experiment", name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "random_vs_temporal_benchmark.csv", index=False)
    metrics = ["PR-AUC", "ROC-AUC", "Recall@10%", "Recall@20%", "Lift@10%", "Lift@20%"]
    completed = combined.loc[combined["Status"].eq("COMPLETED")]
    stability_rows: list[dict[str, Any]] = []
    if {"stratified", "temporal"}.issubset(experiments):
        for model, group in completed.groupby("Model"):
            by_experiment = group.set_index("Experiment")
            if not {"stratified", "temporal"}.issubset(by_experiment.index):
                continue
            row: dict[str, Any] = {"Model": model}
            for metric in metrics:
                random_value = float(by_experiment.loc["stratified", metric])
                temporal_value = float(by_experiment.loc["temporal", metric])
                row[f"Random {metric}"] = random_value
                row[f"OOT {metric}"] = temporal_value
                row[f"OOT minus Random {metric}"] = temporal_value - random_value
            stability_rows.append(row)
    pd.DataFrame(stability_rows).to_csv(output_dir / "model_stability.csv", index=False)


def run_pipeline(config_or_path: Mapping[str, Any] | str | Path) -> PipelineResult:
    """Run configured data preparation, experiments, and HCP prioritization."""

    if isinstance(config_or_path, Mapping):
        config = copy.deepcopy(dict(config_or_path))
        validate_config(config)
    else:
        config = load_config(config_or_path)
    output_dir = Path(config["project"].get("output_dir", "outputs"))
    artifact_dir = Path(config["project"].get("artifact_dir", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    seed = int(config["project"].get("random_seed", 42))
    set_global_seed(seed)
    write_json(config, artifact_dir / "resolved_config.json")

    logger.info("Loading %s claims data", config["data"].get("source"))
    tables = _load_or_generate_tables(config)
    cohort = build_cohort(tables, config)
    if cohort.empty:
        raise ValueError("Cohort construction produced no eligible patients.")
    validate_cohort_timeline(cohort, tables, config)
    features = build_tabular_features(tables, cohort, config)
    if len(features) != len(cohort):
        raise AssertionError("Feature matrix must contain exactly one row per cohort patient.")
    cohort_profile = class_imbalance_summary(cohort["label"])
    cohort_profile.update(
        {
            "index_date_min": pd.to_datetime(cohort["index_date"]).min(),
            "index_date_max": pd.to_datetime(cohort["index_date"]).max(),
            "feature_count": len(_model_features(features).columns),
        }
    )
    pd.DataFrame([cohort_profile]).to_csv(output_dir / "cohort_summary.csv", index=False)

    configured_experiments = [str(value).lower() for value in config["splitting"]["experiments"]]
    experiments: Dict[str, ExperimentResult] = {}
    for experiment in configured_experiments:
        experiments[experiment] = _run_experiment(
            experiment,
            tables,
            cohort,
            features,
            config,
            output_dir,
            artifact_dir,
            logger,
        )
    primary = str(config["splitting"].get("primary_experiment", "temporal")).lower()
    if primary not in experiments:
        raise ValueError("splitting.primary_experiment must be listed in splitting.experiments.")
    primary_result = experiments[primary]

    # Exact top-level filenames are aliases of the configured primary experiment.
    primary_files = [
        "model_benchmark.csv",
        "executive_benchmark.csv",
        "decile_analysis.csv",
        "cumulative_gains.csv",
        "bootstrap_confidence_intervals.csv",
        "paired_model_comparison.csv",
        "tabular_comparison.csv",
        "longitudinal_comparison.csv",
        "model_recommendation.json",
        "model_recommendation.md",
    ]
    for filename in primary_files:
        source = primary_result.output_dir / filename
        if source.exists():
            (output_dir / filename).write_bytes(source.read_bytes())
    _run_hcp_layer(tables, cohort, features, primary_result, config, output_dir)
    _write_cross_experiment_outputs(experiments, output_dir)
    logger.info("Benchmark complete. Primary experiment: %s", primary)
    return PipelineResult(
        config=config,
        tables=dict(tables),
        cohort=cohort,
        features=features,
        experiments=experiments,
        primary_experiment=primary,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
    )


__all__ = ["ExperimentResult", "PipelineResult", "run_pipeline"]
