"""Conservative, auditable benchmark recommendation reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from therapy_switch.io import write_json

CLASSICAL_MODELS = (
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
)
TABULAR_DL_MODELS = ("MLP",)
SEQUENCE_DL_MODELS = (
    "LSTM",
    "GRU",
    "BiLSTM",
    "Transformer",
    "Hybrid GRU Wide",
    "GRU Without Time",
    "GRU Shuffled Without Time",
)


@dataclass(frozen=True)
class Recommendation:
    model: str | None
    decision: str
    rationale: str
    best_classical_model: str | None
    best_longitudinal_dl_model: str | None
    validation_metric: str
    materiality_threshold: float
    evidence_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _completed(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    candidates = frame.loc[frame["Model"].isin(names) & frame["Status"].eq("COMPLETED")].copy()
    return candidates


def _best_by_validation(
    frame: pd.DataFrame,
    names: tuple[str, ...],
    validation_scores: Mapping[str, float],
) -> str | None:
    candidates = _completed(frame, names)
    if candidates.empty:
        return None
    scored = [
        (name, float(validation_scores.get(name, np.nan)))
        for name in candidates["Model"].astype(str)
    ]
    finite = [(name, score) for name, score in scored if np.isfinite(score)]
    if finite:
        return max(finite, key=lambda pair: (pair[1], pair[0]))[0]
    return None


def make_recommendation(
    benchmark: pd.DataFrame,
    validation_scores: Mapping[str, float],
    paired_comparison: pd.DataFrame | None = None,
    *,
    material_pr_auc_gain: float = 0.01,
    data_source: str = "synthetic",
) -> Recommendation:
    """Freeze the candidate using validation AP only.

    Paired test comparisons are accepted for API compatibility but cannot alter
    selection. Missing validation evidence never falls back to test performance.
    """
    names = tuple(benchmark.loc[benchmark.Model.ne("Naive Baseline"), "Model"])
    chosen = _best_by_validation(benchmark, names, validation_scores)
    return Recommendation(
        model=chosen,
        decision="VALIDATION SELECTED CANDIDATE" if chosen else "NO MODEL RECOMMENDED",
        rationale=(
            "Candidate selected by validation average precision. Test comparisons are "
            "descriptive; they do not change the frozen choice."
            if chosen
            else "No completed predictive model has finite validation evidence."
        ),
        best_classical_model=_best_by_validation(benchmark, CLASSICAL_MODELS, validation_scores),
        best_longitudinal_dl_model=_best_by_validation(
            benchmark, SEQUENCE_DL_MODELS, validation_scores
        ),
        validation_metric="PR-AUC",
        materiality_threshold=material_pr_auc_gain,
        evidence_scope=(
            "SYNTHETIC BENCHMARK RESULTS; no real-data performance conclusion."
            if data_source == "synthetic"
            else "Held-out claims benchmark."
        ),
    )


def write_recommendation_report(
    recommendation: Recommendation,
    benchmark: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write machine-readable and executive-readable recommendation artifacts."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "model_recommendation.json"
    markdown_path = target / "model_recommendation.md"
    write_json(recommendation.to_dict(), json_path)

    completed = benchmark.loc[benchmark["Status"].eq("COMPLETED")].copy()
    display_columns = [
        "Model",
        "Category",
        "PR-AUC",
        "ROC-AUC",
        "Recall@10%",
        "Lift@10%",
        "Training Time",
        "Inference Time",
    ]
    if completed.empty:
        table = "No model completed successfully."
    else:
        table = completed[display_columns].to_markdown(index=False, floatfmt=".4f")
    markdown = f"""# Model benchmark recommendation

**Decision:** {recommendation.decision}

**Recommended model:** {recommendation.model or "None"}

{recommendation.rationale}

**Evidence scope:** {recommendation.evidence_scope}

## Held-out benchmark snapshot

{table}

The score predicts association with a future advanced-therapy claim for commercial
analytics. It is not a treatment recommendation and must not be interpreted causally.
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path
