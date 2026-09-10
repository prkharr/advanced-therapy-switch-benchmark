"""Paired percentile intervals with patient-cluster resampling for repeated snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import top_fraction_metrics, validate_predictions

BOOTSTRAP_METRICS = (
    "ROC-AUC",
    "PR-AUC",
    *[
        f"{name}@Top{p}%"
        for p in (5, 10, 20)
        for name in ("Recall", "Precision", "Lift", "TruePositives")
    ],
)


def _metric_values(y, scores):
    out = {
        "ROC-AUC": float(roc_auc_score(y, scores)),
        "PR-AUC": float(average_precision_score(y, scores)),
    }
    for p in (5, 10, 20):
        m = top_fraction_metrics(y, scores, p / 100)
        out.update(
            {
                f"{k}@Top{p}%": float(m[v])
                for k, v in [
                    ("Recall", "recall"),
                    ("Precision", "precision"),
                    ("Lift", "lift"),
                    ("TruePositives", "selected_positives"),
                ]
            }
        )
    return out


def _bootstrap_indices(y, rng, patient_ids=None):
    if patient_ids is None:
        out = np.concatenate(
            [
                rng.choice(np.flatnonzero(y == c), size=int((y == c).sum()), replace=True)
                for c in (0, 1)
            ]
        )
        rng.shuffle(out)
        return out
    ids = np.asarray(patient_ids)
    if len(ids) != len(y) or pd.isna(ids).any():
        raise ValueError("Patient cluster identifiers must align and cannot be missing")
    groups = {pid: np.flatnonzero(ids == pid) for pid in pd.unique(ids)}
    keys = list(groups)
    draws = rng.integers(0, len(keys), size=len(keys))
    return np.concatenate([groups[keys[i]] for i in draws])


def _inputs(y, scores, n_bootstrap, confidence_level, patient_ids):
    y, scores = validate_predictions(y, scores)
    if np.unique(y).size != 2 or n_bootstrap < 2 or not 0 < confidence_level < 1:
        raise ValueError(
            "Bootstrap requires two classes, at least two draws and confidence in (0,1)"
        )
    if patient_ids is not None and (len(patient_ids) != len(y) or pd.isna(patient_ids).any()):
        raise ValueError("Patient cluster identifiers must align and cannot be missing")
    return y, scores


def paired_bootstrap_comparison(
    y_true,
    classical_scores,
    deep_learning_scores,
    *,
    classical_model,
    deep_learning_model,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42,
    patient_ids=None,
):
    y, left = _inputs(y_true, classical_scores, n_bootstrap, confidence_level, patient_ids)
    _, right = validate_predictions(y, deep_learning_scores)
    lp, rp = _metric_values(y, left), _metric_values(y, right)
    differences = []
    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        ids = _bootstrap_indices(y, rng, patient_ids)
        if np.unique(y[ids]).size < 2:
            continue
        a, b = _metric_values(y[ids], left[ids]), _metric_values(y[ids], right[ids])
        differences.append([b[k] - a[k] for k in BOOTSTRAP_METRICS])
    if len(differences) < 2:
        raise ValueError("Too few valid bootstrap draws")
    distribution = np.asarray(differences)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(distribution, [alpha, 1 - alpha], axis=0)
    return pd.DataFrame(
        [
            {
                "metric": k,
                "classical_model": classical_model,
                "deep_learning_model": deep_learning_model,
                "classical_estimate": lp[k],
                "deep_learning_estimate": rp[k],
                "difference_dl_minus_classical": rp[k] - lp[k],
                "ci_lower": lower[i],
                "ci_upper": upper[i],
                "confidence_level": confidence_level,
                "statistically_significant": bool(lower[i] > 0 or upper[i] < 0),
                "successful_samples": len(distribution),
                "requested_samples": n_bootstrap,
                "sampling_unit": "patient_cluster" if patient_ids is not None else "stratified_row",
                "multiplicity_adjusted": False,
            }
            for i, k in enumerate(BOOTSTRAP_METRICS)
        ]
    )


def bootstrap_confidence_intervals(
    y_true,
    y_score,
    *,
    model,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42,
    patient_ids=None,
):
    y, scores = _inputs(y_true, y_score, n_bootstrap, confidence_level, patient_ids)
    point, rng, samples = _metric_values(y, scores), np.random.default_rng(random_state), []
    for _ in range(n_bootstrap):
        ids = _bootstrap_indices(y, rng, patient_ids)
        if np.unique(y[ids]).size == 2:
            values = _metric_values(y[ids], scores[ids])
            samples.append([values[k] for k in BOOTSTRAP_METRICS])
    if len(samples) < 2:
        raise ValueError("Too few valid bootstrap draws")
    samples = np.asarray(samples)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha], axis=0)
    return pd.DataFrame(
        [
            {
                "model": model,
                "metric": k,
                "estimate": point[k],
                "ci_lower": lower[i],
                "ci_upper": upper[i],
                "standard_error": samples[:, i].std(ddof=1),
                "confidence_level": confidence_level,
                "successful_samples": len(samples),
                "requested_samples": n_bootstrap,
                "sampling_unit": "patient_cluster" if patient_ids is not None else "stratified_row",
            }
            for i, k in enumerate(BOOTSTRAP_METRICS)
        ]
    )


def bootstrap_all_models(y_true, predictions, **kwargs):
    frames = [
        bootstrap_confidence_intervals(y_true, scores, model=name, **kwargs)
        for name, scores in predictions.items()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def align_external_scores(test_snapshots, external_scores, score_column="reference_score"):
    """Fail closed on population mismatches; caller must verify model provenance/split."""
    required = {"snapshot_id", score_column}
    if not required.issubset(external_scores) or external_scores.snapshot_id.duplicated().any():
        raise ValueError("Reference predictions require one score per snapshot")
    if set(external_scores.snapshot_id) != set(test_snapshots.snapshot_id):
        raise ValueError(
            "Reference predictions must cover exactly the held-out snapshot population"
        )
    aligned = test_snapshots[["snapshot_id", "label"]].merge(
        external_scores, on="snapshot_id", validate="one_to_one"
    )
    _, scores = validate_predictions(aligned.label, aligned[score_column])
    return scores


stratified_bootstrap_ci = bootstrap_confidence_intervals
paired_stratified_bootstrap = paired_bootstrap_comparison
