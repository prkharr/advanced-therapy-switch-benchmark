"""Ranking and capacity evaluation checks."""
import numpy as np
import pytest

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
