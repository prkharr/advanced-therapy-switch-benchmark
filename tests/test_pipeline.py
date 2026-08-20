from pathlib import Path

import pandas as pd

from therapy_switch.config import load_config
from therapy_switch.pipeline import run_pipeline


def test_end_to_end_pipeline_writes_required_outputs(tmp_path: Path):
    config = load_config("configs/quickstart.yaml")
    config["project"]["output_dir"] = str(tmp_path / "outputs")
    config["project"]["artifact_dir"] = str(tmp_path / "artifacts")
    config["data"]["synthetic"]["n_patients"] = 120
    config["data"]["synthetic"]["n_providers"] = 30
    config["data"]["synthetic"]["target_prevalence"] = 0.10
    config["splitting"]["experiments"] = ["stratified"]
    config["splitting"]["primary_experiment"] = "stratified"
    config["evaluation"]["bootstrap_iterations"] = 10
    config["visualizations"]["enabled"] = False
    config["explainability"]["enabled"] = False
    for model in (
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
        "lstm",
        "gru",
        "bilstm",
        "transformer",
    ):
        config["models"][model]["enabled"] = False

    result = run_pipeline(config)

    assert result.primary_experiment == "stratified"
    assert len(result.cohort) == 120
    output_dir = Path(config["project"]["output_dir"])
    for filename in (
        "model_benchmark.csv",
        "executive_benchmark.csv",
        "decile_analysis.csv",
        "cumulative_gains.csv",
        "patient_propensity_scores.csv",
        "hcp_targeting_output.csv",
    ):
        assert (output_dir / filename).exists(), filename

    benchmark = pd.read_csv(output_dir / "model_benchmark.csv")
    assert len(benchmark) == 11
    assert (
        benchmark.loc[benchmark["Model"].eq("Logistic Regression"), "Status"].item() == "COMPLETED"
    )
    assert benchmark.loc[benchmark["Model"].eq("LSTM"), "Status"].item() == ("NOT APPLICABLE")
    assert benchmark.loc[benchmark["Model"].eq("LSTM"), "Reason"].notna().all()

    split = result.experiments["stratified"].splits
    train_ids = set(split["train"]["patient_id"])
    validation_ids = set(split["validation"]["patient_id"])
    test_ids = set(split["test"]["patient_id"])
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
