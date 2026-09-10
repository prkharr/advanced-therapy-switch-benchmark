"""Capacity-based patient lists from supplied eligible snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .evaluation.metrics import top_k_count, validate_predictions


def latest_patient_indices(frame, *, as_of=None):
    """One latest supplied eligible assessment per patient; never inspect outcomes.

    Callers must establish eligibility at the intended scoring date upstream.
    Historical snapshots alone cannot establish whether a patient remains eligible.
    """
    required = {"patient_id", "snapshot_id", "index_date"}
    if not required.issubset(frame):
        raise ValueError(f"Missing patient-list keys: {sorted(required - set(frame))}")
    if frame[list(required)].isna().any().any() or frame.snapshot_id.duplicated().any():
        raise ValueError("Patient-list keys must be non-null and snapshots unique")
    work = frame[["patient_id", "snapshot_id", "index_date"]].copy().reset_index(drop=True)
    work["position"] = np.arange(len(work))
    work["index_date"] = pd.to_datetime(work.index_date, errors="raise")
    if work.duplicated(["patient_id", "index_date"]).any():
        raise ValueError("Ambiguous duplicate patient/index assessments")
    if as_of is not None:
        work = work.loc[work.index_date <= pd.Timestamp(as_of)]
    if work.empty:
        raise ValueError("No eligible assessments before the scoring cutoff")
    return (
        work.sort_values(["index_date", "snapshot_id"], kind="stable")
        .drop_duplicates("patient_id", keep="last")
        .sort_values("patient_id")
        .position.to_numpy()
    )


def rank_patients(frame, scores, *, fraction=0.1, as_of=None):
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("Patient scores must be finite and aligned with snapshots")
    ids = latest_patient_indices(frame, as_of=as_of)
    out = frame.iloc[ids][["patient_id", "snapshot_id", "index_date"]].copy()
    out["risk_score"] = values[ids]
    out = out.sort_values(
        ["risk_score", "patient_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["selected"] = out["rank"] <= top_k_count(len(out), fraction)
    return out


def patient_capture_metrics(frame, scores, *, fraction=0.1):
    y, score = validate_predictions(frame.label, scores)
    ids = latest_patient_indices(frame)
    y, score = y[ids], score[ids]
    if np.unique(y).size != 2:
        raise ValueError("Patient evaluation requires both classes")
    k = top_k_count(len(y), fraction)
    # ids are ordered by patient ID, giving outcome-independent stable tie handling.
    selected = np.argsort(-score, kind="stable")[:k]
    true_positives = int(y[selected].sum())
    return {
        "patients": len(y),
        "positives": int(y.sum()),
        "selected": k,
        "captured": true_positives,
        "recall": true_positives / int(y.sum()),
        "precision": true_positives / k,
        "lift": (true_positives / k) / float(y.mean()),
        "ap": float(average_precision_score(y, score)),
    }
