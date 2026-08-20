"""Headless, file-oriented benchmark visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .decile_analysis import all_models_cumulative_gains
from .metrics import top_fraction_metrics, validate_predictions


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Visualizations require the optional 'matplotlib' dependency.") from exc
    return plt


def _output_path(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_roc_pr_curves(
    y_true: Sequence[int] | np.ndarray,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    roc_path: str | Path = "outputs/figures/roc_curves.png",
    pr_path: str | Path = "outputs/figures/pr_curves.png",
) -> dict[str, Path]:
    """Save separate ROC and precision-recall comparison plots."""

    if not predictions:
        raise ValueError("predictions must contain at least one model.")
    plt = _pyplot()
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    prevalence: float | None = None
    for model, raw_scores in predictions.items():
        y, scores = validate_predictions(y_true, raw_scores)
        if np.unique(y).size < 2:
            raise ValueError("ROC/PR plots require both outcome classes.")
        prevalence = float(y.mean())
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        ax_roc.plot(fpr, tpr, label=f"{model} (AUC={roc_auc_score(y, scores):.3f})")
        ax_pr.plot(
            recall,
            precision,
            label=f"{model} (AP={average_precision_score(y, scores):.3f})",
        )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax_roc.legend(loc="lower right", fontsize="small")
    ax_roc.grid(alpha=0.2)
    if prevalence is not None:
        ax_pr.axhline(prevalence, linestyle="--", color="grey", label="Prevalence")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    ax_pr.legend(loc="best", fontsize="small")
    ax_pr.grid(alpha=0.2)
    roc_output, pr_output = _output_path(roc_path), _output_path(pr_path)
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(roc_output, dpi=160, bbox_inches="tight")
    fig_pr.savefig(pr_output, dpi=160, bbox_inches="tight")
    plt.close(fig_roc)
    plt.close(fig_pr)
    return {"roc": roc_output, "precision_recall": pr_output}


def plot_lift_gains(
    y_true: Sequence[int] | np.ndarray,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    output_path: str | Path = "outputs/figures/lift_and_gains.png",
) -> Path:
    """Save cumulative lift and gains curves on aligned population percentiles."""

    if not predictions:
        raise ValueError("predictions must contain at least one model.")
    gains = all_models_cumulative_gains(y_true, predictions)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model, frame in gains.groupby("model", sort=False):
        x = frame["population_targeted_pct"] * 100.0
        axes[0].plot(x, frame["actual_switchers_captured_pct"] * 100.0, label=model)
        axes[1].plot(x, frame["cumulative_lift"], label=model)
    axes[0].plot([0, 100], [0, 100], linestyle="--", color="grey", label="Random")
    axes[0].set(
        xlabel="Population targeted (%)",
        ylabel="Actual switchers captured (%)",
        title="Cumulative gains",
    )
    axes[1].axhline(1.0, linestyle="--", color="grey", label="Random")
    axes[1].set(
        xlabel="Population targeted (%)",
        ylabel="Cumulative lift",
        title="Cumulative lift",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    output = _output_path(output_path)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_calibration_curves(
    y_true: Sequence[int] | np.ndarray,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
    output_path: str | Path = "outputs/figures/calibration_curves.png",
) -> Path:
    """Save reliability curves with the perfect-calibration reference line."""

    if not predictions:
        raise ValueError("predictions must contain at least one model.")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, raw_scores in predictions.items():
        y, scores = validate_predictions(y_true, raw_scores)
        observed, predicted = calibration_curve(y, scores, n_bins=n_bins, strategy=strategy)
        ax.plot(predicted, observed, marker="o", label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed switch rate",
        title="Probability calibration",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize="small")
    output = _output_path(output_path)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_decile_performance(
    deciles: pd.DataFrame,
    *,
    output_path: str | Path = "outputs/figures/decile_performance.png",
) -> Path:
    """Plot observed switch rate and cumulative recall by propensity decile."""

    required = {"model", "decile", "switch_rate", "cumulative_recall_pct"}
    missing = required - set(deciles.columns)
    if missing:
        raise ValueError(f"deciles is missing columns: {sorted(missing)}")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model, frame in deciles.groupby("model", sort=False):
        axes[0].plot(frame["decile"], frame["switch_rate"], marker="o", label=model)
        axes[1].plot(
            frame["decile"],
            frame["cumulative_recall_pct"] * 100.0,
            marker="o",
            label=model,
        )
    axes[0].set(xlabel="Decile (1 = highest score)", ylabel="Switch rate", title="Decile response")
    axes[1].set(
        xlabel="Decile targeted through",
        ylabel="Cumulative switchers captured (%)",
        title="Cumulative decile capture",
    )
    for axis in axes:
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    output = _output_path(output_path)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_benchmark_comparison(
    benchmark: pd.DataFrame,
    *,
    output_path: str | Path = "outputs/figures/model_benchmark.png",
) -> Path:
    """Plot the most decision-relevant metrics for successfully run models."""

    metrics = ["PR-AUC", "ROC-AUC", "Recall@10%", "Lift@10%"]
    required = {"Model", *metrics}
    missing = required - set(benchmark.columns)
    if missing:
        raise ValueError(f"benchmark is missing columns: {sorted(missing)}")
    frame = benchmark.copy()
    if "Status" in frame:
        frame = frame[frame["Status"] == "COMPLETED"]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=metrics, how="all")
    if frame.empty:
        raise ValueError("No completed numeric benchmark results are available to plot.")
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, metric in zip(axes.ravel(), metrics):
        values = frame[metric]
        axis.barh(frame["Model"], values)
        axis.set_title(metric)
        axis.grid(axis="x", alpha=0.2)
    output = _output_path(output_path)
    fig.suptitle("Therapy-switch model benchmark")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_top_fraction_lift(
    y_true: Sequence[int] | np.ndarray,
    predictions: Mapping[str, Sequence[float] | np.ndarray],
    *,
    output_path: str | Path = "outputs/figures/top_fraction_lift.png",
) -> Path:
    """Plot Lift@1/5/10/20/30% as a compact targeting comparison."""

    fractions = (0.01, 0.05, 0.10, 0.20, 0.30)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, scores in predictions.items():
        lifts = [top_fraction_metrics(y_true, scores, fraction)["lift"] for fraction in fractions]
        ax.plot([f * 100 for f in fractions], lifts, marker="o", label=model)
    ax.axhline(1.0, linestyle="--", color="grey", label="Random")
    ax.set(xlabel="Population targeted (%)", ylabel="Lift", title="Lift at targeting cutoffs")
    ax.grid(alpha=0.2)
    ax.legend(fontsize="small")
    output = _output_path(output_path)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


# Common alternate pluralized import name is provided via visualizations.py.
plot_lift_and_gains = plot_lift_gains


__all__ = [
    "plot_benchmark_comparison",
    "plot_calibration_curves",
    "plot_decile_performance",
    "plot_lift_and_gains",
    "plot_lift_gains",
    "plot_roc_pr_curves",
    "plot_top_fraction_lift",
]
