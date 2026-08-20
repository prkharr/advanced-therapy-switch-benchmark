"""Probability calibration fitted exclusively on validation predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .metrics import validate_predictions


@dataclass
class ProbabilityCalibrator:
    """A score calibrator with an explicit validation-only fit contract."""

    method: str = "platt"
    random_state: int = 42

    def __post_init__(self) -> None:
        normalized = self.method.lower().replace("-", "_")
        aliases = {"sigmoid": "platt", "logistic": "platt"}
        self.method = aliases.get(normalized, normalized)
        if self.method not in {"platt", "isotonic"}:
            raise ValueError("method must be 'platt' (sigmoid) or 'isotonic'.")
        self._model: LogisticRegression | IsotonicRegression | None = None
        self.fitted_on_validation_: bool = False
        self.validation_sample_size_: int | None = None

    def fit(
        self,
        y_validation: Sequence[int] | np.ndarray,
        validation_scores: Sequence[float] | np.ndarray,
        *,
        dataset_role: str = "validation",
    ) -> "ProbabilityCalibrator":
        """Fit on validation data; other dataset roles are rejected."""

        if dataset_role.lower() not in {"validation", "val"}:
            raise ValueError(
                "Calibration must be fit on validation data only; final test labels "
                "must remain untouched."
            )
        y, scores = validate_predictions(y_validation, validation_scores)
        if np.unique(y).size < 2:
            raise ValueError("Calibration requires both outcome classes in validation data.")
        if self.method == "platt":
            model: LogisticRegression | IsotonicRegression = LogisticRegression(
                solver="lbfgs", random_state=self.random_state
            )
            model.fit(scores.reshape(-1, 1), y)
        else:
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(scores, y)
        self._model = model
        self.fitted_on_validation_ = True
        self.validation_sample_size_ = len(y)
        return self

    def transform(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        if not self.fitted_on_validation_ or self._model is None:
            raise RuntimeError("Calibrator must be fit on validation data before transform.")
        values = np.asarray(scores, dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("scores must be finite probabilities within [0, 1].")
        if self.method == "platt":
            calibrated = self._model.predict_proba(values.reshape(-1, 1))[:, 1]
        else:
            calibrated = self._model.predict(values)
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)

    def predict_proba(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        probabilities = self.transform(scores)
        return np.column_stack((1.0 - probabilities, probabilities))


def fit_calibrator_on_validation(
    y_validation: Sequence[int] | np.ndarray,
    validation_scores: Sequence[float] | np.ndarray,
    *,
    method: str = "platt",
    random_state: int = 42,
) -> ProbabilityCalibrator:
    """Construct and fit a validation-only probability calibrator."""

    return ProbabilityCalibrator(method=method, random_state=random_state).fit(
        y_validation, validation_scores, dataset_role="validation"
    )


def calibration_analysis(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    model: str,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Return calibration summary metrics and reliability-curve coordinates."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")
    y, scores = validate_predictions(y_true, probabilities)
    observed, predicted = calibration_curve(y, scores, n_bins=n_bins, strategy=strategy)
    summary: dict[str, float | int | str] = {
        "model": model,
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "n_patients": len(y),
        "n_bins_requested": n_bins,
        "n_bins_observed": len(observed),
    }
    curve = pd.DataFrame(
        {
            "model": model,
            "bin": np.arange(1, len(observed) + 1),
            "mean_predicted_probability": predicted,
            "observed_switch_rate": observed,
        }
    )
    return summary, curve


def compare_calibration_methods(
    y_validation: Sequence[int] | np.ndarray,
    validation_scores: Sequence[float] | np.ndarray,
    y_evaluation: Sequence[int] | np.ndarray,
    evaluation_scores: Sequence[float] | np.ndarray,
    *,
    model: str,
    methods: Sequence[str] = ("platt", "isotonic"),
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Fit candidates on validation and evaluate them on an untouched population."""

    y_eval, raw = validate_predictions(y_evaluation, evaluation_scores)
    summaries = []
    curves = []
    outputs: dict[str, np.ndarray] = {"uncalibrated": raw}
    summary, curve = calibration_analysis(
        y_eval, raw, model=f"{model} (uncalibrated)", n_bins=n_bins
    )
    summaries.append(summary)
    curves.append(curve)
    for method in methods:
        calibrator = fit_calibrator_on_validation(y_validation, validation_scores, method=method)
        calibrated = calibrator.transform(raw)
        outputs[calibrator.method] = calibrated
        summary, curve = calibration_analysis(
            y_eval,
            calibrated,
            model=f"{model} ({calibrator.method})",
            n_bins=n_bins,
        )
        summaries.append(summary)
        curves.append(curve)
    return (
        pd.DataFrame(summaries),
        pd.concat(curves, ignore_index=True),
        outputs,
    )


__all__ = [
    "ProbabilityCalibrator",
    "calibration_analysis",
    "compare_calibration_methods",
    "fit_calibrator_on_validation",
]
