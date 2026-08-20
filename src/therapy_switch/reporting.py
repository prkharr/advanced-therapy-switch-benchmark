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
SEQUENCE_DL_MODELS = ("LSTM", "GRU", "BiLSTM", "Transformer")


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
    candidates = candidates.dropna(subset=["PR-AUC"])
    if candidates.empty:
        return None
    return str(candidates.sort_values(["PR-AUC", "Model"]).iloc[-1]["Model"])


def make_recommendation(
    benchmark: pd.DataFrame,
    validation_scores: Mapping[str, float],
    paired_comparison: pd.DataFrame | None = None,
    *,
    material_pr_auc_gain: float = 0.01,
    data_source: str = "synthetic",
) -> Recommendation:
    """Recommend complexity only when longitudinal DL has material evidence.

    Candidate ordering is determined on validation PR-AUC. Test paired-bootstrap
    evidence is then used as a gate for accepting the additional operational
    complexity of longitudinal DL. This is intentionally conservative.
    """

    best_classical = _best_by_validation(benchmark, CLASSICAL_MODELS, validation_scores)
    best_sequence = _best_by_validation(benchmark, SEQUENCE_DL_MODELS, validation_scores)
    evidence_scope = (
        "Synthetic development benchmark; no production performance conclusion is valid."
        if data_source == "synthetic"
        else "Held-out claims benchmark; deployment still requires governance review."
    )

    if best_classical is None and best_sequence is None:
        return Recommendation(
            model=None,
            decision="NO MODEL RECOMMENDED",
            rationale="No classical or longitudinal model completed successfully.",
            best_classical_model=None,
            best_longitudinal_dl_model=None,
            validation_metric="PR-AUC",
            materiality_threshold=material_pr_auc_gain,
            evidence_scope=evidence_scope,
        )
    if best_classical is None:
        return Recommendation(
            model=best_sequence,
            decision="PROVISIONAL LONGITUDINAL MODEL",
            rationale=(
                "No classical comparator completed; treat this as provisional and repair the "
                "controlled comparison before deployment."
            ),
            best_classical_model=None,
            best_longitudinal_dl_model=best_sequence,
            validation_metric="PR-AUC",
            materiality_threshold=material_pr_auc_gain,
            evidence_scope=evidence_scope,
        )
    if best_sequence is None:
        return Recommendation(
            model=best_classical,
            decision="RECOMMEND CLASSICAL MODEL",
            rationale=(
                "The validation-selected classical model completed, while no valid longitudinal "
                "DL result was available. This does not establish that sequence learning fails."
            ),
            best_classical_model=best_classical,
            best_longitudinal_dl_model=None,
            validation_metric="PR-AUC",
            materiality_threshold=material_pr_auc_gain,
            evidence_scope=evidence_scope,
        )

    paired = pd.DataFrame() if paired_comparison is None else paired_comparison
    pr_row = paired.loc[
        paired.get("metric", pd.Series(dtype=str)).eq("PR-AUC")
        & paired.get("classical_model", pd.Series(dtype=str)).eq(best_classical)
        & paired.get("deep_learning_model", pd.Series(dtype=str)).eq(best_sequence)
    ]
    supported = False
    observed_difference = np.nan
    ci_lower = np.nan
    if not pr_row.empty:
        evidence = pr_row.iloc[0]
        observed_difference = float(evidence["difference_dl_minus_classical"])
        ci_lower = float(evidence["ci_lower"])
        supported = observed_difference >= material_pr_auc_gain and ci_lower > 0.0

    if supported:
        return Recommendation(
            model=best_sequence,
            decision="RECOMMEND LONGITUDINAL DL",
            rationale=(
                f"{best_sequence} exceeded {best_classical} by {observed_difference:.4f} PR-AUC; "
                f"the paired 95% interval lower bound was {ci_lower:.4f}, clearing the configured "
                f"{material_pr_auc_gain:.4f} materiality threshold. Review cost, stability, and "
                "explainability before deployment."
            ),
            best_classical_model=best_classical,
            best_longitudinal_dl_model=best_sequence,
            validation_metric="PR-AUC",
            materiality_threshold=material_pr_auc_gain,
            evidence_scope=evidence_scope,
        )
    return Recommendation(
        model=best_classical,
        decision="RECOMMEND CLASSICAL MODEL",
        rationale=(
            f"{best_sequence} did not show a statistically supported, material PR-AUC gain over "
            f"{best_classical}. Prefer the lower-complexity classical model. This conclusion is "
            "specific to the tested sequence representation and data snapshot, not a blanket "
            "claim that deep learning does not work."
        ),
        best_classical_model=best_classical,
        best_longitudinal_dl_model=best_sequence,
        validation_metric="PR-AUC",
        materiality_threshold=material_pr_auc_gain,
        evidence_scope=evidence_scope,
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
