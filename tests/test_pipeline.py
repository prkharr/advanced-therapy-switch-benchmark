import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from therapy_switch.pipeline import run_pipeline


def test_pipeline_outputs_reload_and_reproducibility(tmp_path, prepared_data, prepared_config):
    _, _, inputs = prepared_data
    config = deepcopy(prepared_config)
    config["splitting"]["experiments"] = ["stratified"]
    config["splitting"]["primary_experiment"] = "stratified"
    config["evaluation"]["bootstrap_iterations"] = 5
    config["visualizations"]["enabled"] = False
    config["explainability"]["enabled"] = False
    for key, settings in config["models"].items():
        if "enabled" in settings:
            settings["enabled"] = key in {"naive_baseline", "logistic_regression"}
    frames = {name: getattr(inputs, name) for name in ("snapshots", "wide", "events")}
    runs = []
    for attempt in (1, 2):
        config["project"]["output_dir"] = str(tmp_path / f"outputs_{attempt}")
        config["project"]["artifact_dir"] = str(tmp_path / f"artifacts_{attempt}")
        result = run_pipeline(config, tables=frames)
        runs.append(result)
        output = Path(config["project"]["output_dir"])
        for filename in (
            "model_benchmark.csv",
            "executive_benchmark.csv",
            "decile_analysis.csv",
            "cumulative_gains.csv",
            "patient_propensity_scores.csv",
            "hcp_targeting_output.csv",
            "bootstrap_confidence_intervals.csv",
            "calibration_selection.csv",
            "split_manifest.csv",
            "data_quality_report.json",
            "run_manifest.json",
        ):
            assert (output / filename).exists(), filename
        benchmark = pd.read_csv(output / "model_benchmark.csv")
        assert len(benchmark) == 15
        assert benchmark.Status.eq("COMPLETED").sum() == 2
        assert not benchmark.Status.eq("FAILED").any()
        assert result.experiments["stratified"].recommendation.model == "Logistic Regression"
        artifact = Path(config["project"]["artifact_dir"]) / "experiments/stratified"
        metadata = json.loads((artifact / "models/logistic_regression_metadata.json").read_text())
        assert metadata["reload_verified"]
        assert metadata["sha256"]
        assert result.cohort.patient_id.nunique() == 120
        assert len(result.cohort) > 120
    first, second = [run.experiments["stratified"] for run in runs]
    for model, probabilities in first.predictions.items():
        np.testing.assert_array_equal(probabilities, second.predictions[model])
    for name in first.splits:
        pd.testing.assert_frame_equal(first.splits[name], second.splits[name])
    pd.testing.assert_frame_equal(
        pd.read_csv(runs[0].output_dir / "patient_propensity_scores.csv"),
        pd.read_csv(runs[1].output_dir / "patient_propensity_scores.csv"),
    )
