from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from therapy_switch.models import (
    FAILED,
    NOT_APPLICABLE,
    SUCCESS,
    LeakageError,
    LogisticRegressionRunner,
    LSTMRunner,
    MLPRunner,
    ModelRun,
    NaiveBaselineRunner,
    SequenceSplit,
    model_registry,
    run_benchmark,
)


def _frames() -> tuple[
    pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray
]:
    rng = np.random.default_rng(7)

    def frame(rows: int) -> pd.DataFrame:
        severity = rng.normal(size=rows)
        return pd.DataFrame(
            {
                "age": rng.integers(25, 82, rows),
                "severity_proxy": severity,
                "specialist_visits_90d": np.maximum(
                    0, np.rint(severity + rng.normal(1.5, 1.0, rows))
                ),
                "region": rng.choice(["north", "south", "west"], rows),
            }
        )

    X_train, X_val, X_test = frame(48), frame(16), frame(16)
    y_train = np.array([0] * 36 + [1] * 12)
    y_val = np.array([0] * 12 + [1] * 4)
    y_test = np.array([0] * 12 + [1] * 4)
    # Remove any ordering signal from the fixture target.
    for X, y in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        order = rng.permutation(len(y))
        X.iloc[:] = X.iloc[order].to_numpy()
        y[:] = y[order]
    return X_train, y_train, X_val, y_val, X_test, y_test


def _run(**overrides: object) -> ModelRun:
    X_train, y_train, X_val, y_val, X_test, y_test = _frames()
    values: dict[str, object] = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    values.update(overrides)
    return ModelRun(**values)  # type: ignore[arg-type]


def test_registry_contains_every_required_model_in_display_order() -> None:
    registry = model_registry()
    assert list(registry) == [
        "naive_baseline",
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "mlp",
        "lstm",
        "gru",
        "bilstm",
        "transformer",
        "hybrid",
        "lightgbm_wide",
        "gru_no_time",
        "gru_shuffled",
    ]
    assert [runner.model_name for runner in registry.values()] == [
        "Naive Baseline",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "MLP",
        "LSTM",
        "GRU",
        "BiLSTM",
        "Transformer",
        "Hybrid GRU Wide",
        "LightGBM Without Recency",
        "GRU Without Time",
        "GRU Shuffled Without Time",
    ]


def test_naive_baseline_uses_training_prevalence_and_uniform_contract() -> None:
    run = _run()
    result = NaiveBaselineRunner().run(run)
    expected = float(np.mean(run.y_train))
    assert result.status == SUCCESS
    assert result.succeeded
    assert np.allclose(result.validation_probabilities, expected)
    assert np.allclose(result.test_probabilities, expected)
    assert result.test_predictions is not None
    assert len(result.test_predictions) == len(run.y_test)
    assert result.training_time_seconds == 0.0
    assert result.validation_score == pytest.approx(np.mean(run.y_val))
    assert result.reason is None


def test_target_like_feature_is_rejected_as_leakage() -> None:
    run = _run()
    run.X_train["outcome"] = run.y_train
    run.X_val["outcome"] = run.y_val
    run.X_test["outcome"] = run.y_test
    result = NaiveBaselineRunner().run(run)
    assert result.status == FAILED
    assert result.reason is not None
    assert "target-like" in result.reason


def test_sequence_contract_rejects_post_index_events() -> None:
    sequence = SequenceSplit(
        values=np.ones((2, 2, 3), dtype=np.float32),
        mask=np.ones((2, 2), dtype=bool),
        event_dates=np.array([["2026-01-01", "2026-01-11"], ["2026-01-01", "2026-01-02"]]),
        index_dates=np.array(["2026-01-10", "2026-01-10"]),
    )
    with pytest.raises(LeakageError, match="post-index event"):
        sequence.validated(expected_rows=2)


def test_sequence_contract_requires_auditable_pre_index_assertion() -> None:
    unverified = SequenceSplit(values=np.ones((2, 2, 1), dtype=np.float32))
    with pytest.raises(LeakageError, match="unverified"):
        unverified.validated()

    verified = SequenceSplit(values=np.ones((2, 2, 1), dtype=np.float32), pre_index_verified=True)
    assert verified.validated().mask is not None


def test_sequence_contract_requires_right_padding() -> None:
    sequence = SequenceSplit(
        values=np.ones((1, 3, 1), dtype=np.float32),
        mask=np.array([[True, False, True]]),
        pre_index_verified=True,
    )
    with pytest.raises(ValueError, match="right padded"):
        sequence.validated()


def test_sequence_runner_is_explicitly_not_applicable_without_event_data() -> None:
    result = LSTMRunner().run(_run())
    assert result.status == NOT_APPLICABLE
    assert result.reason is not None
    assert "sequence" in result.reason


def test_sequence_runner_never_masks_a_leakage_failure_as_missing_dependency() -> None:
    run = _run()

    def sequence(rows: int, future: bool = False) -> SequenceSplit:
        event_dates = np.full((rows, 2), "2026-01-01", dtype="U10")
        if future:
            event_dates[0, 1] = "2026-01-11"
        return SequenceSplit(
            values=np.ones((rows, 2, 2), dtype=np.float32),
            mask=np.ones((rows, 2), dtype=bool),
            event_dates=event_dates,
            index_dates=np.full(rows, "2026-01-10", dtype="U10"),
        )

    run.sequence_train = sequence(len(run.y_train), future=True)
    run.sequence_val = sequence(len(run.y_val))
    run.sequence_test = sequence(len(run.y_test))
    result = LSTMRunner().run(run)
    assert result.status == FAILED
    assert result.reason is not None
    assert "post-index event" in result.reason


def test_convenience_api_returns_requested_models_only() -> None:
    X_train, y_train, X_val, y_val, X_test, y_test = _frames()
    results = run_benchmark(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        model_keys=["naive_baseline", "lstm"],
    )
    assert list(results) == ["naive_baseline", "lstm"]
    assert results["naive_baseline"].status == SUCCESS
    assert results["lstm"].status == NOT_APPLICABLE


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn modeling dependency is not installed",
)
def test_logistic_regression_pipeline_handles_numeric_and_categorical_features() -> None:
    result = LogisticRegressionRunner().run(_run())
    assert result.status == SUCCESS, result.reason
    assert result.estimator.named_steps["preprocessor"] is not None
    assert result.estimator.named_steps["model"] is not None
    assert result.test_probabilities is not None
    assert np.isfinite(result.test_probabilities).all()
    assert ((result.test_probabilities >= 0) & (result.test_probabilities <= 1)).all()
    assert result.metadata["positive_class_weight"] == pytest.approx(3.0)


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn modeling dependency is not installed",
)
def test_logistic_tuning_falls_back_and_records_pr_auc_trials() -> None:
    run = _run(
        params={
            "logistic_regression": {
                "tune": True,
                "prefer_optuna": False,
                "n_trials": 2,
                "search_space": {"C": [0.1, 1.0]},
            }
        }
    )
    result = LogisticRegressionRunner().run(run)
    assert result.status == SUCCESS, result.reason
    tuning = result.metadata["tuning"]
    assert tuning["backend"] == "randomized_search_cv"
    assert tuning["n_trials"] == 2
    assert np.isfinite(tuning["best_validation_pr_auc"])


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("sklearn") is None,
    reason="PyTorch/scikit-learn neural dependencies are not installed",
)
def test_mlp_records_bce_training_history() -> None:
    run = _run(
        params={
            "mlp": {
                "losses": ["bce"],
                "hidden_dims": [8],
                "batch_size": 16,
                "max_epochs": 2,
                "patience": 1,
            }
        }
    )
    result = MLPRunner().run(run)
    assert result.status == SUCCESS, result.reason
    assert result.history["selected_loss"] == "bce"
    assert result.history["runs"]["bce"]["epochs_trained"] >= 1
    assert result.test_probabilities is not None
    assert len(result.test_probabilities) == len(run.y_test)


@pytest.mark.parametrize("model_key", ["gru", "transformer", "hybrid"])
def test_sequence_models_embeddings_empty_padding_and_reload(tmp_path, model_key):
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    import joblib

    run = _run(
        params={
            model_key: {
                "max_epochs": 2,
                "patience": 1,
                "batch_size": 16,
                "hidden_size": 8,
                "d_model": 8,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 16,
                "losses": ["bce"],
                "dropout": 0,
            }
        }
    )
    rng = np.random.default_rng(41)

    def sequence(rows):
        values = np.zeros((rows, 6, 5), dtype=np.float32)
        values[:, :, :4] = rng.integers(1, 6, (rows, 6, 4))
        values[:, :, 4] = np.arange(6, 0, -1)
        mask = np.ones((rows, 6), dtype=bool)
        mask[0] = False
        mask[1, 3:] = False
        values[~mask] = 0
        return SequenceSplit(
            values=values,
            mask=mask,
            times=np.ones((rows, 6)),
            categorical_sizes=(6, 6, 6, 6),
            pre_index_verified=True,
        )

    run.sequence_train = sequence(len(run.y_train))
    run.sequence_val = sequence(len(run.y_val))
    run.sequence_test = sequence(len(run.y_test))
    result = model_registry()[model_key].run(run)
    assert result.succeeded, result.reason
    assert np.isfinite(result.test_probabilities).all()
    assert any(isinstance(m, torch.nn.Embedding) for m in result.estimator.network.modules())
    path = tmp_path / f"{model_key}.joblib"
    joblib.dump(result.estimator, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(
        loaded.predict_proba(run.sequence_test)[:, 1], result.test_probabilities
    )
    altered = replace(run.sequence_test, values=run.sequence_test.values.copy())
    altered.values[~altered.mask] = 4
    np.testing.assert_allclose(loaded.predict_proba(altered)[:, 1], result.test_probabilities)
    if model_key == "hybrid":
        changed = replace(run.sequence_test, wide=run.X_test.copy())
        changed.wide["age"] = changed.wide.age + 100
        assert not np.allclose(loaded.predict_proba(changed)[:, 1], result.test_probabilities)


def test_time_and_order_ablations_remove_timing_and_preserve_event_multiset():
    from therapy_switch.models.neural import SequenceScaler

    values = np.zeros((2, 10, 5), dtype=np.float32)
    values[:, :, :4] = np.arange(2, 12)[None, :, None]
    values[:, :, 4] = np.arange(10, 0, -1)
    split = SequenceSplit(
        values=values,
        mask=np.ones((2, 10), dtype=bool),
        times=np.ones((2, 10)),
        categorical_sizes=(12, 12, 12, 12),
        pre_index_verified=True,
    )
    ordered = SequenceScaler.fit(split, options={"use_time": False}, random_state=42).transform(
        split
    )
    shuffled = SequenceScaler.fit(
        split, options={"use_time": False, "shuffle_order": True}, random_state=42
    ).transform(split)
    assert (shuffled[0][:, :, 4] == 0).all() and (shuffled[2] == 0).all()
    assert not np.array_equal(ordered[0][0, :, 0], shuffled[0][0, :, 0])
    np.testing.assert_array_equal(np.sort(ordered[0], axis=1), np.sort(shuffled[0], axis=1))
    reversed_rows = SequenceSplit(
        values=values[::-1].copy(),
        mask=split.mask[::-1].copy(),
        times=split.times[::-1].copy(),
        categorical_sizes=split.categorical_sizes,
        pre_index_verified=True,
    )
    scaler = SequenceScaler.fit(
        split, options={"use_time": False, "shuffle_order": True}, random_state=42
    )
    np.testing.assert_array_equal(scaler.transform(reversed_rows)[0][::-1], shuffled[0])


def test_optional_torch_failure_is_explicit(monkeypatch):
    import therapy_switch.models.neural as neural

    monkeypatch.setattr(neural, "TORCH_AVAILABLE", False)
    result = MLPRunner().run(_run())
    assert result.status == NOT_APPLICABLE
    assert "torch" in result.reason


def test_sequence_available_date_cutoff():
    seq = SequenceSplit(
        values=np.ones((1, 1, 1)),
        mask=np.ones((1, 1), dtype=bool),
        event_dates=np.array([["2024-01-01"]]),
        available_dates=np.array([["2024-03-01"]]),
        index_dates=np.array(["2024-02-01"]),
    )
    with pytest.raises(LeakageError):
        seq.validated()
