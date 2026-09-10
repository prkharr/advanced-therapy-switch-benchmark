"""Tests for evaluation metrics, reporting schemas, and validation safeguards."""

from __future__ import annotations

import numpy as np
import pytest

from therapy_switch.evaluation.benchmark import (
    EXPECTED_MODELS,
    benchmark_row,
    build_executive_benchmark,
    build_model_benchmark,
)
from therapy_switch.evaluation.bootstrap import paired_bootstrap_comparison
from therapy_switch.evaluation.calibration import ProbabilityCalibrator
from therapy_switch.evaluation.decile_analysis import (
    DECILE_COLUMNS,
    cumulative_gains,
    decile_analysis,
)
from therapy_switch.evaluation.metrics import (
    evaluate_predictions,
    top_fraction_metrics,
    top_k_count,
    tune_threshold_on_validation,
)


def test_ranking_metrics_capture_perfect_top_decile() -> None:
    y_true = np.r_[np.ones(10, dtype=int), np.zeros(90, dtype=int)]
    y_score = np.linspace(1.0, 0.0, 100)

    metrics = evaluate_predictions(y_true, y_score)

    assert metrics["ROC-AUC"] == pytest.approx(1.0)
    assert metrics["PR-AUC"] == pytest.approx(1.0)
    assert metrics["Recall@10%"] == pytest.approx(1.0)
    assert metrics["Precision@10%"] == pytest.approx(1.0)
    assert metrics["Lift@10%"] == pytest.approx(10.0)
    assert metrics["Recall@5%"] == pytest.approx(0.5)


def test_top_fraction_uses_ceiling_and_stable_tie_order() -> None:
    assert top_k_count(11, 0.10) == 2
    result = top_fraction_metrics([1, 0, 0], [0.5, 0.5, 0.1], 0.10)
    assert result["selected_count"] == 1
    assert result["selected_positives"] == 1


def test_validation_threshold_tuning_is_deterministic() -> None:
    result = tune_threshold_on_validation([0, 0, 1, 1], [0.10, 0.40, 0.35, 0.80], objective="f1")
    assert result.threshold == pytest.approx(0.35)
    assert result.objective_value == pytest.approx(0.8)
    assert result.validation_sample_size == 4


def test_target_recall_threshold_strategy_optimizes_precision_subject_to_recall() -> None:
    result = tune_threshold_on_validation(
        [0, 0, 0, 1, 1],
        [0.05, 0.25, 0.45, 0.40, 0.80],
        objective="target_recall",
        target_recall=0.5,
    )
    assert result.threshold == pytest.approx(0.80)
    assert result.objective_value == pytest.approx(1.0)


def test_decile_analysis_has_exact_schema_and_highest_decile_first() -> None:
    y_true = np.r_[np.ones(10, dtype=int), np.zeros(90, dtype=int)]
    y_score = np.linspace(1.0, 0.0, 100)
    frame = decile_analysis(y_true, y_score, model="perfect")

    assert frame.columns.tolist() == DECILE_COLUMNS
    assert len(frame) == 10
    assert frame.iloc[0]["decile"] == 1
    assert frame.iloc[0]["actual_switchers"] == 10
    assert frame.iloc[0]["cumulative_recall_pct"] == pytest.approx(1.0)
    assert frame.iloc[0]["lift"] == pytest.approx(10.0)


def test_cumulative_gains_answers_capacity_cutoff() -> None:
    y_true = np.r_[np.ones(10, dtype=int), np.zeros(90, dtype=int)]
    y_score = np.linspace(1.0, 0.0, 100)
    frame = cumulative_gains(y_true, y_score, model="perfect", percentiles=[10, 20])
    assert frame.loc[0, "population_targeted_pct"] == pytest.approx(0.10)
    assert frame.loc[0, "actual_switchers_captured_pct"] == pytest.approx(1.0)
    assert frame.loc[0, "cumulative_lift"] == pytest.approx(10.0)


def test_paired_bootstrap_identical_models_has_zero_difference() -> None:
    y_true = np.r_[np.ones(20, dtype=int), np.zeros(80, dtype=int)]
    scores = np.linspace(1.0, 0.0, 100)
    comparison = paired_bootstrap_comparison(
        y_true,
        scores,
        scores,
        classical_model="classical",
        deep_learning_model="dl",
        n_bootstrap=30,
        random_state=7,
    )
    assert np.allclose(comparison["difference_dl_minus_classical"], 0.0)
    assert np.allclose(comparison["ci_lower"], 0.0)
    assert np.allclose(comparison["ci_upper"], 0.0)
    assert not comparison["statistically_significant"].any()


def test_calibrator_rejects_test_role() -> None:
    calibrator = ProbabilityCalibrator(method="platt")
    with pytest.raises(ValueError, match="validation data only"):
        calibrator.fit(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.7, 0.8],
            dataset_role="test",
        )


def test_benchmark_marks_missing_models_not_applicable_without_metrics() -> None:
    completed = benchmark_row(
        model="Logistic Regression",
        metrics={"PR-AUC": 0.4, "ROC-AUC": 0.7},
        training_time=1.2,
        inference_time=0.1,
    )
    benchmark = build_model_benchmark([completed])

    assert benchmark["Model"].tolist() == list(EXPECTED_MODELS)
    missing = benchmark[benchmark["Model"] == "Transformer"].iloc[0]
    assert missing["Status"] == "NOT APPLICABLE"
    assert isinstance(missing["Reason"], str) and missing["Reason"]
    assert np.isnan(missing["PR-AUC"])

    executive = build_executive_benchmark(benchmark, recommended_model="Logistic Regression")
    recommended = executive[executive["Recommended?"] == "Yes"]
    assert recommended["Model"].tolist() == ["Logistic Regression"]


def test_benchmark_normalizes_model_runner_success_status() -> None:
    benchmark = build_model_benchmark(
        [
            {
                "Model": "Logistic Regression",
                "Category": "Classical",
                "Status": "SUCCESS",
                "Reason": "",
                "PR-AUC": 0.4,
            }
        ],
        add_missing_models=False,
    )
    assert benchmark.loc[0, "Status"] == "COMPLETED"


def test_patient_cluster_bootstrap_preserves_all_snapshots_and_multiplicity():
    from collections import Counter

    from therapy_switch.evaluation.bootstrap import _bootstrap_indices

    ids = np.array(["A", "A", "B", "B", "B", "C"])
    y = np.array([0, 1, 0, 1, 0, 1])
    indices = _bootstrap_indices(y, np.random.default_rng(11), ids)
    counts = Counter(indices)
    assert counts[0] == counts[1]
    assert counts[2] == counts[3] == counts[4]
    assert sum(counts[loc] for loc in (0, 2, 5)) == 3
    same = paired_bootstrap_comparison(
        y,
        np.arange(6) / 6,
        np.arange(6) / 6,
        classical_model="reference",
        deep_learning_model="challenger",
        patient_ids=ids,
        n_bootstrap=30,
    )
    assert same.sampling_unit.eq("patient_cluster").all()
    assert same.difference_dl_minus_classical.eq(0).all()
    assert same.ci_lower.eq(0).all() and same.ci_upper.eq(0).all()


def test_external_reference_alignment_and_capacity_tp():
    import pandas as pd

    from therapy_switch.evaluation.bootstrap import align_external_scores

    snaps = pd.DataFrame({"snapshot_id": ["01", "02", "03"], "label": [1, 0, 1]})
    scores = pd.DataFrame({"snapshot_id": ["03", "01", "02"], "reference_score": [0.7, 0.9, 0.2]})
    np.testing.assert_array_equal(align_external_scores(snaps, scores), [0.9, 0.2, 0.7])
    with pytest.raises(ValueError, match="exactly"):
        align_external_scores(snaps, scores.iloc[:2])
    metrics = evaluate_predictions([1, 0, 1, 0, 0], [0.8, 0.7, 0.3, 0.2, 0.1], fractions=[0.2])
    assert metrics["TruePositives@20%"] == 1


def test_recommendation_never_selects_from_test_or_test_bootstrap():
    import pandas as pd

    from therapy_switch.reporting import make_recommendation

    benchmark = build_model_benchmark(
        [
            benchmark_row(model="LightGBM", metrics={"PR-AUC": 0.1}),
            benchmark_row(model="GRU", metrics={"PR-AUC": 0.9}),
            benchmark_row(model="MLP", metrics={"PR-AUC": 0.8}),
        ]
    )
    assert make_recommendation(benchmark, {}).model is None
    result = make_recommendation(
        benchmark,
        {"LightGBM": 0.5, "GRU": 0.2, "MLP": 0.6},
        pd.DataFrame({"metric": ["PR-AUC"], "ci_lower": [1]}),
    )
    assert result.model == "MLP"


def test_crossfit_calibration_keeps_patients_in_one_fold():
    from therapy_switch.evaluation.calibration import select_crossfit_calibrator

    y = np.tile([0, 1, 0, 1], 6)
    ids = np.repeat(np.arange(8), 3)
    scores = np.where(y, 0.7, 0.2)
    calibrator, selection = select_crossfit_calibrator(y, scores, ids, methods=["none", "sigmoid"])
    assert len(selection) == 2
    assert np.isfinite(selection.loc[selection.status.eq("COMPLETED"), "validation_brier"]).all()


def test_permutation_explains_frozen_predictor_without_fit_interface():
    import pandas as pd

    from therapy_switch.evaluation.explainability import explain_model

    class FrozenPredictor:
        def predict_proba(self, frame):
            positive = frame["signal"].to_numpy()
            return np.column_stack([1 - positive, positive])

    y = np.tile([0, 1], 30)
    X = pd.DataFrame({"signal": np.where(y, 0.8, 0.2), "irrelevant": np.arange(len(y))})
    explanation = explain_model(FrozenPredictor(), X, y, prefer_shap=False, n_permutation_repeats=4)
    assert explanation.available, explanation.reason
    assert explanation.method == "permutation_importance_pr_auc"
    importance = explanation.feature_importance.set_index("feature")
    assert importance.loc["signal", "permutation_ap_drop"] > 0.2
    assert importance.loc["irrelevant", "permutation_ap_drop"] == 0
