"""Reproducibility and data-lineage manifest generation."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from therapy_switch.io import write_json

TRACKED_PACKAGES = (
    "advanced-therapy-switch-benchmark",
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "optuna",
    "shap",
)


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_value(arguments: list[str], workspace: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_run_manifest(
    config: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    cohort: pd.DataFrame,
    *,
    experiment: str,
    split_frames: Mapping[str, pd.DataFrame] | None = None,
    workspace: str | Path = ".",
) -> dict[str, Any]:
    """Build a PHI-safe manifest containing counts, settings, and code state."""

    workspace_path = Path(workspace).resolve()
    labels = cohort["label"] if "label" in cohort else pd.Series(dtype=int)
    split_summary: dict[str, Any] = {}
    for name, frame in (split_frames or {}).items():
        split_labels = frame["label"] if "label" in frame else pd.Series(dtype=int)
        split_summary[name] = {
            "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else len(frame),
            "positive_patients": int(split_labels.sum()) if len(split_labels) else None,
            "prevalence": float(split_labels.mean()) if len(split_labels) else None,
            "index_date_min": (
                pd.to_datetime(frame["index_date"]).min()
                if "index_date" in frame and len(frame)
                else None
            ),
            "index_date_max": (
                pd.to_datetime(frame["index_date"]).max()
                if "index_date" in frame and len(frame)
                else None
            ),
        }
    configuration = {key: value for key, value in config.items() if not str(key).startswith("_")}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "configuration": configuration,
        "table_row_counts": {name: int(len(frame)) for name, frame in tables.items()},
        "cohort": {
            "patients": int(cohort["patient_id"].nunique())
            if "patient_id" in cohort
            else len(cohort),
            "positive_patients": int(labels.sum()) if len(labels) else None,
            "negative_patients": int((1 - labels).sum()) if len(labels) else None,
            "prevalence": float(labels.mean()) if len(labels) else None,
        },
        "splits": split_summary,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {package: _version(package) for package in TRACKED_PACKAGES},
        },
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"], workspace_path),
            "branch": _git_value(["branch", "--show-current"], workspace_path),
            "status_porcelain": _git_value(["status", "--porcelain"], workspace_path),
        },
        "safety": {
            "contains_row_level_claims": False,
            "intended_use": "commercial analytics and HCP opportunity prioritization",
            "prohibited_use": "treatment recommendation or causal inference",
        },
    }


def write_run_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    write_json(manifest, destination)
    return destination
