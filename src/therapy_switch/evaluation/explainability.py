"""Model explainability with optional SHAP and deterministic fallbacks.

All outputs describe predictive associations.  They must not be interpreted as
causal effects, treatment effects, or treatment recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

NON_CAUSAL_DISCLAIMER = (
    "Attributions describe predictive associations in this model and dataset; "
    "they do not establish causality or support treatment recommendations."
)

IMPORTANCE_COLUMNS = [
    "feature",
    "mean_abs_attribution",
    "mean_attribution",
    "direction",
    "rank",
    "interpretation",
]


@dataclass
class ExplanationResult:
    status: str
    method: str
    reason: str
    feature_importance: pd.DataFrame
    attribution_values: np.ndarray | None = None
    explained_data: np.ndarray | None = None
    feature_names: list[str] | None = None

    @property
    def available(self) -> bool:
        return self.status == "COMPLETED"


def _feature_names(X: Any, feature_names: Sequence[str] | None) -> list[str]:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("X must have a two-dimensional tabular shape.")
    n_features = int(shape[1])
    if feature_names is not None:
        names = [str(name) for name in feature_names]
    elif hasattr(X, "columns"):
        names = [str(name) for name in X.columns]
    else:
        names = [f"feature_{index}" for index in range(n_features)]
    if len(names) != n_features:
        raise ValueError("feature_names length does not match the model input width.")
    return names


def _sample_rows(X: Any, max_samples: int, random_state: int) -> Any:
    if max_samples < 1:
        raise ValueError("max_samples must be positive.")
    n_rows = len(X)
    if n_rows <= max_samples:
        return X
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(n_rows, size=max_samples, replace=False))
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return np.asarray(X)[indices]


def _normalize_attributions(values: Any, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    # SHAP may return (samples, features, classes) for a binary classifier.
    if array.ndim == 3:
        if array.shape[1] == n_features:
            array = array[:, :, -1]
        elif array.shape[2] == n_features:
            array = array[-1, :, :]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != n_features:
        raise ValueError(f"Unexpected attribution shape {array.shape}; expected (*, {n_features}).")
    return array.astype(float)


def _importance_frame(
    names: Sequence[str],
    absolute: np.ndarray,
    signed: np.ndarray | None,
) -> pd.DataFrame:
    absolute_values = np.asarray(absolute, dtype=float).reshape(-1)
    if signed is None:
        signed_values = np.full(len(absolute_values), np.nan)
        direction = np.repeat("not signed", len(absolute_values))
    else:
        signed_values = np.asarray(signed, dtype=float).reshape(-1)
        direction = np.where(
            signed_values > 0.0,
            "positive association",
            np.where(signed_values < 0.0, "negative association", "neutral"),
        )
    frame = pd.DataFrame(
        {
            "feature": list(names),
            "mean_abs_attribution": absolute_values,
            "mean_attribution": signed_values,
            "direction": direction,
            "interpretation": NON_CAUSAL_DISCLAIMER,
        }
    ).sort_values("mean_abs_attribution", ascending=False, kind="stable")
    frame.insert(4, "rank", np.arange(1, len(frame) + 1))
    return frame.reset_index(drop=True)[IMPORTANCE_COLUMNS]


def _try_shap(
    model: Any,
    X: Any,
    names: Sequence[str],
) -> ExplanationResult:
    try:
        import shap  # type: ignore
    except ImportError:
        return ExplanationResult(
            status="UNAVAILABLE",
            method="SHAP",
            reason="Optional dependency 'shap' is not installed.",
            feature_importance=pd.DataFrame(columns=IMPORTANCE_COLUMNS),
        )
    try:
        # TreeExplainer is fast and exact for supported tree implementations.
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            raw_values = explainer.shap_values(X)
        else:
            explainer = shap.Explainer(model, X)
            explanation = explainer(X)
            raw_values = explanation.values
        values = _normalize_attributions(raw_values, len(names))
        frame = _importance_frame(names, np.mean(np.abs(values), axis=0), np.mean(values, axis=0))
        return ExplanationResult(
            status="COMPLETED",
            method="SHAP",
            reason="",
            feature_importance=frame,
            attribution_values=values,
            explained_data=np.asarray(X),
            feature_names=list(names),
        )
    except Exception as exc:  # optional integrations fail differently by model/version
        return ExplanationResult(
            status="UNAVAILABLE",
            method="SHAP",
            reason=f"SHAP could not explain this model: {type(exc).__name__}: {exc}",
            feature_importance=pd.DataFrame(columns=IMPORTANCE_COLUMNS),
        )


def _native_importance(model: Any, names: Sequence[str]) -> ExplanationResult:
    candidate = model
    if hasattr(model, "named_steps"):
        candidate = list(model.named_steps.values())[-1]
    if hasattr(candidate, "feature_importances_"):
        values = np.asarray(candidate.feature_importances_, dtype=float).reshape(-1)
        if len(values) == len(names):
            return ExplanationResult(
                status="COMPLETED",
                method="native_feature_importance",
                reason="SHAP unavailable or disabled; using model-native importance.",
                feature_importance=_importance_frame(names, np.abs(values), None),
            )
    if hasattr(candidate, "coef_"):
        coefficients = np.asarray(candidate.coef_, dtype=float)
        if coefficients.ndim > 1:
            coefficients = coefficients[-1]
        coefficients = coefficients.reshape(-1)
        if len(coefficients) == len(names):
            return ExplanationResult(
                status="COMPLETED",
                method="native_coefficients",
                reason="SHAP unavailable or disabled; using fitted coefficients.",
                feature_importance=_importance_frame(names, np.abs(coefficients), coefficients),
            )
    return ExplanationResult(
        status="UNAVAILABLE",
        method="native",
        reason="Model exposes neither compatible feature_importances_ nor coef_.",
        feature_importance=pd.DataFrame(columns=IMPORTANCE_COLUMNS),
    )


def _permutation_fallback(
    model: Any,
    X: Any,
    y: Sequence[int] | np.ndarray | None,
    names: Sequence[str],
    *,
    random_state: int,
    n_repeats: int,
) -> ExplanationResult:
    if y is None:
        return ExplanationResult(
            status="UNAVAILABLE",
            method="permutation_importance",
            reason="Permutation fallback requires evaluation labels.",
            feature_importance=pd.DataFrame(columns=IMPORTANCE_COLUMNS),
        )
    try:
        result = permutation_importance(
            model,
            X,
            np.asarray(y),
            scoring="average_precision",
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )
        values = np.asarray(result.importances_mean, dtype=float)
        return ExplanationResult(
            status="COMPLETED",
            method="permutation_importance_pr_auc",
            reason=(
                "SHAP and compatible native importance unavailable or disabled; "
                "using held-out PR-AUC permutation importance."
            ),
            # A negative permutation delta does not mean a negative patient-level
            # association, so do not label it as a signed clinical driver.
            feature_importance=_importance_frame(names, np.abs(values), None),
        )
    except Exception as exc:
        return ExplanationResult(
            status="UNAVAILABLE",
            method="permutation_importance",
            reason=(f"Permutation fallback failed: {type(exc).__name__}: {exc}"),
            feature_importance=pd.DataFrame(columns=IMPORTANCE_COLUMNS),
        )


def explain_model(
    model: Any,
    X: Any,
    y: Sequence[int] | np.ndarray | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    prefer_shap: bool = True,
    max_samples: int = 1000,
    n_permutation_repeats: int = 5,
    random_state: int = 42,
) -> ExplanationResult:
    """Explain a fitted model using SHAP, native, then permutation importance."""

    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return ExplanationResult(
            status="UNAVAILABLE",
            method="none",
            reason=(
                "Generic attribution requires a 2D tabular input. Supply a "
                "sequence-specific explainer/adapter for longitudinal tensors."
            ),
            feature_importance=pd.DataFrame(columns=IMPORTANCE_COLUMNS),
        )
    names = _feature_names(X, feature_names)
    sampled_X = _sample_rows(X, max_samples, random_state)
    sampled_y = None
    if y is not None:
        values_y = np.asarray(y).reshape(-1)
        if len(values_y) != len(X):
            raise ValueError("y length must match X.")
        if len(X) > len(sampled_X):
            # Recreate the same deterministic sample indexes.
            rng = np.random.default_rng(random_state)
            indices = np.sort(rng.choice(len(X), size=len(sampled_X), replace=False))
            sampled_y = values_y[indices]
        else:
            sampled_y = values_y

    shap_reason = "SHAP disabled by configuration."
    if prefer_shap:
        shap_result = _try_shap(model, sampled_X, names)
        if shap_result.available:
            return shap_result
        shap_reason = shap_result.reason
    native = _native_importance(model, names)
    if native.available:
        native.reason = f"{shap_reason} {native.reason}".strip()
        return native
    permutation = _permutation_fallback(
        model,
        sampled_X,
        sampled_y,
        names,
        random_state=random_state,
        n_repeats=n_permutation_repeats,
    )
    if permutation.available:
        permutation.reason = f"{shap_reason} {permutation.reason}".strip()
        return permutation
    permutation.reason = (f"{shap_reason} {native.reason} {permutation.reason}").strip()
    return permutation


def top_predictive_drivers(
    result: ExplanationResult, *, n: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top positive and negative predictive associations, when signed."""

    if n < 1:
        raise ValueError("n must be positive.")
    frame = result.feature_importance
    if frame.empty or frame["mean_attribution"].isna().all():
        empty = pd.DataFrame(columns=IMPORTANCE_COLUMNS)
        return empty.copy(), empty.copy()
    positive = frame[frame["mean_attribution"] > 0].nlargest(n, "mean_attribution")
    negative = frame[frame["mean_attribution"] < 0].nsmallest(n, "mean_attribution")
    return positive.reset_index(drop=True), negative.reset_index(drop=True)


def save_explanation_outputs(
    result: ExplanationResult,
    *,
    model_name: str,
    output_dir: str | Path = "outputs/explainability",
) -> dict[str, Path]:
    """Persist importance and signed-driver tables; save SHAP summary if possible."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_name = "_".join(model_name.lower().split())
    paths: dict[str, Path] = {}
    importance_path = output / f"{safe_name}_feature_importance.csv"
    result.feature_importance.to_csv(importance_path, index=False)
    paths["feature_importance"] = importance_path
    positive, negative = top_predictive_drivers(result)
    positive_path = output / f"{safe_name}_positive_drivers.csv"
    negative_path = output / f"{safe_name}_negative_drivers.csv"
    positive.to_csv(positive_path, index=False)
    negative.to_csv(negative_path, index=False)
    paths["positive_drivers"] = positive_path
    paths["negative_drivers"] = negative_path

    status_path = output / f"{safe_name}_explainability_status.txt"
    status_path.write_text(
        "\n".join(
            (
                f"status: {result.status}",
                f"method: {result.method}",
                f"reason: {result.reason or 'None'}",
                f"interpretation: {NON_CAUSAL_DISCLAIMER}",
            )
        ),
        encoding="utf-8",
    )
    paths["status"] = status_path

    if (
        result.method == "SHAP"
        and result.attribution_values is not None
        and result.explained_data is not None
    ):
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import shap  # type: ignore

            shap.summary_plot(
                result.attribution_values,
                result.explained_data,
                feature_names=result.feature_names,
                show=False,
            )
            summary_path = output / f"{safe_name}_shap_summary.png"
            plt.tight_layout()
            plt.savefig(summary_path, dpi=160, bbox_inches="tight")
            plt.close()
            paths["shap_summary"] = summary_path
        except Exception as exc:
            result.reason = (
                f"SHAP attribution values were generated, but the summary plot "
                f"could not be saved: {type(exc).__name__}: {exc}"
            )
            status_path.write_text(
                "\n".join(
                    (
                        f"status: {result.status}",
                        f"method: {result.method}",
                        f"reason: {result.reason}",
                        f"interpretation: {NON_CAUSAL_DISCLAIMER}",
                    )
                ),
                encoding="utf-8",
            )
    return paths


# Tree and tabular-neural models share the fallback chain.
explain_tree_model = explain_model
explain_neural_model = explain_model


__all__ = [
    "ExplanationResult",
    "IMPORTANCE_COLUMNS",
    "NON_CAUSAL_DISCLAIMER",
    "explain_model",
    "explain_neural_model",
    "explain_tree_model",
    "save_explanation_outputs",
    "top_predictive_drivers",
]
