"""Schema-safe benchmark table construction and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

EXPECTED_MODELS = (
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
)

MODEL_BENCHMARK_COLUMNS = [
    "Model",
    "Category",
    "ROC-AUC",
    "PR-AUC",
    "Recall",
    "Precision",
    "F1",
    "Recall@5%",
    "Recall@10%",
    "Recall@20%",
    "Recall@30%",
    "Lift@5%",
    "Lift@10%",
    "Lift@20%",
    "Lift@30%",
    "Brier Score",
    "Training Time",
    "Inference Time",
    "Status",
    "Reason",
]

EXECUTIVE_BENCHMARK_COLUMNS = [
    "Model",
    "PR-AUC",
    "ROC-AUC",
    "Recall@Top10%",
    "Recall@Top20%",
    "Lift@Top10%",
    "Lift@Top20%",
    "Complexity",
    "Recommended?",
    "Status",
    "Reason",
]

DEFAULT_CATEGORIES = {
    "Naive Baseline": "Baseline",
    "Logistic Regression": "Classical - Linear",
    "Random Forest": "Classical - Ensemble",
    "XGBoost": "Classical - Gradient Boosting",
    "LightGBM": "Classical - Gradient Boosting",
    "CatBoost": "Classical - Gradient Boosting",
    "MLP": "Deep Learning - Tabular",
    "LSTM": "Deep Learning - Longitudinal",
    "GRU": "Deep Learning - Longitudinal",
    "BiLSTM": "Deep Learning - Longitudinal",
    "Transformer": "Deep Learning - Longitudinal",
}

DEFAULT_COMPLEXITY = {
    "Naive Baseline": "Low",
    "Logistic Regression": "Low",
    "Random Forest": "Medium",
    "XGBoost": "Medium",
    "LightGBM": "Medium",
    "CatBoost": "Medium",
    "MLP": "High",
    "LSTM": "High",
    "GRU": "High",
    "BiLSTM": "High",
    "Transformer": "High",
}


def _normalize_status(status: object) -> str:
    normalized = str(status).upper().strip().replace("_", " ")
    return {"SUCCESS": "COMPLETED", "NOT APPLICABLE": "NOT APPLICABLE"}.get(normalized, normalized)


def benchmark_row(
    *,
    model: str,
    category: str | None = None,
    metrics: Mapping[str, float] | None = None,
    training_time: float | None = None,
    inference_time: float | None = None,
    status: str = "COMPLETED",
    reason: str = "",
) -> dict[str, object]:
    """Create one benchmark row without inventing unavailable values."""

    normalized_status = _normalize_status(status)
    if normalized_status not in {"COMPLETED", "FAILED", "NOT APPLICABLE"}:
        raise ValueError("status must be COMPLETED, FAILED, or NOT APPLICABLE.")
    if normalized_status != "COMPLETED" and not reason.strip():
        raise ValueError("A reason is required for failed or not-applicable models.")
    values = dict(metrics or {})
    row: dict[str, object] = {
        "Model": model,
        "Category": category or DEFAULT_CATEGORIES.get(model, "Unspecified"),
        "ROC-AUC": values.get("ROC-AUC", np.nan),
        "PR-AUC": values.get("PR-AUC", np.nan),
        "Recall": values.get("Recall", np.nan),
        "Precision": values.get("Precision", np.nan),
        "F1": values.get("F1", np.nan),
        "Recall@5%": values.get("Recall@5%", np.nan),
        "Recall@10%": values.get("Recall@10%", np.nan),
        "Recall@20%": values.get("Recall@20%", np.nan),
        "Recall@30%": values.get("Recall@30%", np.nan),
        "Lift@5%": values.get("Lift@5%", np.nan),
        "Lift@10%": values.get("Lift@10%", np.nan),
        "Lift@20%": values.get("Lift@20%", np.nan),
        "Lift@30%": values.get("Lift@30%", np.nan),
        "Brier Score": values.get("Brier Score", np.nan),
        "Training Time": training_time if training_time is not None else np.nan,
        "Inference Time": inference_time if inference_time is not None else np.nan,
        "Status": normalized_status,
        "Reason": reason,
    }
    if normalized_status != "COMPLETED":
        for column in MODEL_BENCHMARK_COLUMNS[2:18]:
            row[column] = np.nan
    return row


def not_applicable_row(
    model: str,
    reason: str,
    *,
    category: str | None = None,
) -> dict[str, object]:
    """Create an explicit NOT APPLICABLE row with no fabricated metrics."""

    return benchmark_row(
        model=model,
        category=category,
        status="NOT APPLICABLE",
        reason=reason,
    )


def build_model_benchmark(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_models: Sequence[str] = EXPECTED_MODELS,
    add_missing_models: bool = True,
) -> pd.DataFrame:
    """Build the primary table and explicitly mark models without a result."""

    supplied: dict[str, dict[str, object]] = {}
    for raw_row in rows:
        model = str(raw_row.get("Model", "")).strip()
        if not model:
            raise ValueError("Every benchmark row requires a non-empty Model value.")
        if model in supplied:
            raise ValueError(f"Duplicate benchmark result for model: {model}")
        row = {column: raw_row.get(column, np.nan) for column in MODEL_BENCHMARK_COLUMNS}
        row["Model"] = model
        row["Category"] = raw_row.get("Category", DEFAULT_CATEGORIES.get(model, "Unspecified"))
        row["Status"] = _normalize_status(raw_row.get("Status", "COMPLETED"))
        if row["Status"] not in {"COMPLETED", "FAILED", "NOT APPLICABLE"}:
            raise ValueError(f"Unsupported status for {model}: {row['Status']}")
        row["Reason"] = str(raw_row.get("Reason", ""))
        if row["Status"] != "COMPLETED" and not row["Reason"].strip():
            raise ValueError(f"{model} requires a failure/not-applicable reason.")
        if row["Status"] != "COMPLETED":
            for column in MODEL_BENCHMARK_COLUMNS[2:18]:
                row[column] = np.nan
        supplied[model] = row

    ordered: list[dict[str, object]] = []
    for model in expected_models:
        if model in supplied:
            ordered.append(supplied.pop(model))
        elif add_missing_models:
            ordered.append(
                not_applicable_row(
                    model,
                    "No valid prediction output was supplied for this benchmark run.",
                )
            )
    ordered.extend(supplied.values())
    return pd.DataFrame(ordered, columns=MODEL_BENCHMARK_COLUMNS)


def build_executive_benchmark(
    model_benchmark: pd.DataFrame,
    *,
    recommended_model: str | None = None,
    complexity: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Create the presentation table; recommendation remains an explicit decision."""

    missing = set(MODEL_BENCHMARK_COLUMNS) - set(model_benchmark.columns)
    if missing:
        raise ValueError(f"model_benchmark is missing columns: {sorted(missing)}")
    if recommended_model is not None and recommended_model not in set(model_benchmark["Model"]):
        raise ValueError(f"recommended_model not present in benchmark: {recommended_model}")
    if recommended_model is not None:
        recommended_status = model_benchmark.loc[
            model_benchmark["Model"].eq(recommended_model), "Status"
        ].iloc[0]
        if recommended_status != "COMPLETED":
            raise ValueError("recommended_model must have Status='COMPLETED'.")
    complexity_map = {**DEFAULT_COMPLEXITY, **dict(complexity or {})}
    rows = []
    for _, row in model_benchmark.iterrows():
        rows.append(
            {
                "Model": row["Model"],
                "PR-AUC": row["PR-AUC"],
                "ROC-AUC": row["ROC-AUC"],
                "Recall@Top10%": row["Recall@10%"],
                "Recall@Top20%": row["Recall@20%"],
                "Lift@Top10%": row["Lift@10%"],
                "Lift@Top20%": row["Lift@20%"],
                "Complexity": complexity_map.get(str(row["Model"]), "Unspecified"),
                "Recommended?": ("Yes" if recommended_model == row["Model"] else "No"),
                "Status": row["Status"],
                "Reason": row["Reason"],
            }
        )
    return pd.DataFrame(rows, columns=EXECUTIVE_BENCHMARK_COLUMNS)


def save_benchmark_tables(
    model_benchmark: pd.DataFrame,
    executive_benchmark: pd.DataFrame,
    *,
    output_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary_path = output / "model_benchmark.csv"
    executive_path = output / "executive_benchmark.csv"
    model_benchmark.to_csv(primary_path, index=False)
    executive_benchmark.to_csv(executive_path, index=False)
    return primary_path, executive_path


__all__ = [
    "DEFAULT_CATEGORIES",
    "DEFAULT_COMPLEXITY",
    "EXECUTIVE_BENCHMARK_COLUMNS",
    "EXPECTED_MODELS",
    "MODEL_BENCHMARK_COLUMNS",
    "benchmark_row",
    "build_executive_benchmark",
    "build_model_benchmark",
    "not_applicable_row",
    "save_benchmark_tables",
]
