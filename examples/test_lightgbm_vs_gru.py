"""Small proof-of-concept test: compare LightGBM with a GRU on one cohort.

The script uses synthetic claims so it can be shared and run without patient data.
It reuses the repository's cohort, 90-day label, temporal split, metrics, and
leakage checks. Replace the synthetic data settings with the approved real-data
configuration only after the benchmark design is frozen.

Run from the repository root:

    python -m pip install -e ".[all,dev]"
    python examples/test_lightgbm_vs_gru.py
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from therapy_switch.config import load_config
from therapy_switch.pipeline import run_pipeline


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_test_config(work_directory: Path) -> dict:
    """Return a fast, leakage-safe LightGBM-versus-GRU configuration."""

    config = copy.deepcopy(load_config(REPOSITORY_ROOT / "configs" / "quickstart.yaml"))
    config["project"]["output_dir"] = str(work_directory / "outputs")
    config["project"]["artifact_dir"] = str(work_directory / "artifacts")

    # Use a single out-of-time experiment so both models face the same patients.
    config["splitting"]["experiments"] = ["temporal"]
    config["splitting"]["primary_experiment"] = "temporal"

    # Keep only the current tree benchmark, one sequence model, and a prevalence baseline.
    for model_name, options in config["models"].items():
        if isinstance(options, dict) and model_name not in {
            "naive_baseline",
            "lightgbm",
            "gru",
            "sequence_training",
        }:
            options["enabled"] = False
    config["models"]["naive_baseline"]["enabled"] = True
    config["models"]["lightgbm"]["enabled"] = True
    config["models"]["gru"]["enabled"] = True

    # Small settings make this a test, not a final model-training run.
    config["data"]["synthetic"]["n_patients"] = 1_000
    config["models"]["lightgbm"]["n_estimators"] = 100
    config["models"]["gru"].update({"hidden_size": 32, "num_layers": 1})
    config["models"]["sequence_training"].update(
        {"max_epochs": 8, "patience": 2, "batch_size": 64}
    )
    config["evaluation"]["bootstrap_iterations"] = 100
    config["visualizations"]["enabled"] = False
    config["explainability"]["enabled"] = False
    return config


def main() -> None:
    with TemporaryDirectory(prefix="tak861_dl_test_") as temporary_directory:
        test_directory = Path(temporary_directory)
        result = run_pipeline(build_test_config(test_directory))
        experiment = result.experiments["temporal"]
        benchmark = experiment.benchmark

        # Basic test conditions for a valid patient-switch experiment.
        assert len(result.cohort) > 0
        assert set(result.cohort["label"].unique()).issubset({0, 1})
        assert result.config["timeline"]["prediction_window_days"] == 90
        assert {"LightGBM", "GRU"}.issubset(set(benchmark["Model"]))
        assert (experiment.output_dir / "model_benchmark.csv").exists()
        assert (experiment.output_dir / "patient_test_predictions.csv").exists()

        columns = [
            column
            for column in [
                "Model",
                "Status",
                "PR-AUC",
                "ROC-AUC",
                "Lift at 10%",
                "Recall at 10%",
                "Training Time (s)",
            ]
            if column in benchmark.columns
        ]
        print("\nLightGBM versus GRU test result")
        print(benchmark.loc[benchmark["Model"].isin(["LightGBM", "GRU"]), columns].to_string(index=False))
        print(f"\nEligible patients: {len(result.cohort):,}")
        print(f"Positive-label rate: {result.cohort['label'].mean():.2%}")
        print(f"Temporary benchmark files: {experiment.output_dir}")

        # Close the pipeline file handler before Windows removes the temporary folder.
        logging.shutdown()


if __name__ == "__main__":
    main()
