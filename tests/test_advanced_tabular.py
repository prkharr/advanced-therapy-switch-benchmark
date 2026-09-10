"""Optional neural integration, train-only preprocessing and artifact checks."""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from therapy_switch.models.advanced_tabular import (
    ExternalTabularEstimator,
    ProbabilityEnsemble,
    TabMEstimator,
    TabMRunner,
)
from therapy_switch.models.contracts import ModelRun

TABM_AVAILABLE = importlib.util.find_spec("tabm") is not None


def data():
    rng = np.random.default_rng(51)
    frame = pd.DataFrame(
        {
            "prior_claims": rng.gamma(2, 3, 100),
            "therapy_days": rng.uniform(0, 365, 100),
            "payer": rng.choice(["a", "b"], 100),
            "constant": np.ones(100),
        }
    )
    frame.loc[81:, "payer"] = "unseen"
    frame.loc[2, "therapy_days"] = np.nan
    target = np.tile([0, 0, 0, 1], 25)
    return frame, target


@pytest.mark.skipif(not TABM_AVAILABLE, reason="optional tabm dependency")
def test_tabm_reload_test_labels_and_batch_invariance(tmp_path):
    x, y = data()
    options = {
        "width": 16,
        "blocks": 2,
        "members": 4,
        "max_epochs": 3,
        "patience": 2,
        "batch_size": 16,
        "feature_count": 2,
    }
    run = ModelRun(
        x.iloc[:64],
        y[:64],
        x.iloc[64:84],
        y[64:84],
        x.iloc[84:],
        y[84:],
        params={"tabm": options},
        random_state=7,
    )
    first = TabMRunner().run(run)
    assert first.succeeded, first.reason
    altered = TabMRunner().run(replace(run, y_test=1 - y[84:]))
    assert altered.succeeded, altered.reason
    np.testing.assert_allclose(first.test_probabilities, altered.test_probabilities)
    np.testing.assert_allclose(first.validation_probabilities, altered.validation_probabilities)
    path = tmp_path / "model.joblib"
    joblib.dump(first.estimator, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict_proba(x.iloc[84:])[:, 1], first.test_probabilities)
    np.testing.assert_allclose(
        restored.predict_proba(x.iloc[84:89])[:, 1], first.test_probabilities[:5], atol=1e-7
    )
    encoder = (
        restored.preprocessor_.named_steps["columns"]
        .named_transformers_["categorical"]
        .named_steps["one_hot"]
    )
    assert "unseen" not in encoder.categories_[0]


@pytest.mark.skipif(not TABM_AVAILABLE, reason="optional tabm dependency")
def test_tabm_bins_fit_with_constant_removed():
    x, y = data()
    model = TabMEstimator(width=16, members=4, bins=4, max_epochs=2, patience=1)
    model.fit(x.iloc[:64], y[:64], x.iloc[64:84], y[64:84])
    probability = model.predict_proba(x.iloc[84:])
    assert probability.shape == (16, 2)
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(1), 1)


def test_tabicl_cannot_implicitly_download():
    pytest.importorskip("tabicl")
    x, y = data()
    with pytest.raises(ValueError, match="local checkpoint"):
        ExternalTabularEstimator(family="tabicl").fit(x[:64], y[:64], x[64:84], y[64:84])
    with pytest.raises(ValueError, match="downloads are disabled"):
        ExternalTabularEstimator(
            family="tabicl", options={"model_path": "missing.ckpt", "allow_auto_download": True}
        ).fit(x[:64], y[:64], x[64:84], y[64:84])


class Constant:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))


def test_frozen_probability_ensemble_weights():
    x = np.ones((3, 2))
    model = ProbabilityEnsemble([Constant(0.2), Constant(0.8)], [3, 1])
    np.testing.assert_allclose(model.predict_proba(x)[:, 1], 0.35)
    for weights in ([1, -1], [np.nan, 1], [0, 0], [1]):
        with pytest.raises(ValueError, match="weights"):
            ProbabilityEnsemble(model.estimators, weights).predict_proba(x)


def test_tabm_leakage_rejected_before_fit():
    x, y = data()
    x["resp"] = y
    run = ModelRun(x[:64], y[:64], x[64:84], y[64:84], x[84:], y[84:])
    result = TabMRunner().run(run)
    assert not result.succeeded
    assert "resp" in result.reason


def test_tabpfn_requires_existing_checkpoint_without_import():
    x, y = data()
    with pytest.raises(ValueError, match="existing explicit local checkpoint"):
        ExternalTabularEstimator(family="tabpfn").fit(x[:64], y[:64], x[64:84], y[64:84])


@pytest.mark.skipif(not TABM_AVAILABLE, reason="optional tabm dependency")
def test_neural_ensemble_runner_reload_and_member_guard(tmp_path):
    from therapy_switch.models.advanced_tabular import NeuralEnsembleRunner

    x, y = data()
    spec = {
        "name": "small_tabm",
        "family": "tabm",
        "options": {"max_epochs": 2, "patience": 1, "members": 4, "width": 16, "num_threads": 2},
    }
    run = ModelRun(
        x[:64],
        y[:64],
        x[64:84],
        y[64:84],
        x[84:],
        y[84:],
        params={"neural_ensemble": {"members": [spec], "weights": [1.0]}},
    )
    result = NeuralEnsembleRunner().run(run)
    assert result.succeeded, result.reason
    path = tmp_path / "ensemble.joblib"
    joblib.dump(result.estimator, path)
    np.testing.assert_allclose(
        joblib.load(path).predict_proba(x[84:])[:, 1], result.test_probabilities
    )
    invalid = replace(
        run,
        params={
            "neural_ensemble": {"members": [{"name": "tree", "family": "lightgbm", "options": {}}]}
        },
    )
    assert not NeuralEnsembleRunner().run(invalid).succeeded
