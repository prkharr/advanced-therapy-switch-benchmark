"""Confirm a frozen candidate across independent cohorts without test selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from .metrics import validate_predictions


def confirm_average_precision(
    frame,
    *,
    candidate,
    references,
    cohort_column="cohort",
    patient_column="patient_id",
    label_column="label",
    n_bootstrap=2000,
    family_alpha=0.05,
    minimum_gain=0.02,
    seed=91,
):
    """Mean cohort AP differences with paired, cohort-stratified patient draws.

    The candidate and reference list must be frozen before viewing this frame.
    Bonferroni intervals cover all reference comparisons in the supplied family.
    Intervals condition on the trained models; they do not estimate retraining
    uncertainty or correct for selecting candidates on these same test outcomes.
    """
    references = list(references)
    if not references or len(references) != len(set(references)) or candidate in references:
        raise ValueError("Provide unique, distinct reference columns")
    if not 0 < family_alpha < 1 or minimum_gain < 0 or n_bootstrap < 100:
        raise ValueError("Invalid confirmation limits")
    if frame.empty or frame[[cohort_column, patient_column]].isna().any().any():
        raise ValueError("Cohort and patient keys must be present")
    cohorts = []
    rows = []
    columns = [candidate, *references]
    for cohort, part in frame.groupby(cohort_column, sort=True):
        y = np.asarray(part[label_column], dtype=int)
        scores = {}
        for name in columns:
            checked_y, checked_scores = validate_predictions(part[label_column], part[name])
            if np.unique(checked_y).size != 2:
                raise ValueError("Each confirmation cohort requires both classes")
            scores[name] = checked_scores
        groups = list(
            part.reset_index(drop=True).groupby(patient_column, sort=False).indices.values()
        )
        estimates = {
            name: float(average_precision_score(y, values)) for name, values in scores.items()
        }
        rows.append(
            {
                "cohort": cohort,
                "snapshots": len(part),
                "patients": len(groups),
                "positives": int(y.sum()),
                **estimates,
            }
        )
        cohorts.append((y, scores, groups))
    observed = np.array([np.mean([r[candidate] - r[ref] for r in rows]) for ref in references])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        differences = []
        for y, scores, groups in cohorts:
            choices = rng.integers(0, len(groups), len(groups))
            ids = np.concatenate([groups[i] for i in choices])
            if np.unique(y[ids]).size != 2:
                break
            right = average_precision_score(y[ids], scores[candidate][ids])
            differences.append(
                [right - average_precision_score(y[ids], scores[ref][ids]) for ref in references]
            )
        if len(differences) == len(cohorts):
            draws.append(np.mean(differences, axis=0))
    if len(draws) < max(100, n_bootstrap // 2):
        raise ValueError("Too few valid patient-cluster draws")
    alpha = family_alpha / len(references)
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    comparisons = []
    for i, ref in enumerate(references):
        comparisons.append(
            {
                "candidate": candidate,
                "reference": ref,
                "candidate_mean_ap": float(np.mean([r[candidate] for r in rows])),
                "reference_mean_ap": float(np.mean([r[ref] for r in rows])),
                "mean_ap_gain": float(observed[i]),
                "ci_lower": float(lower[i]),
                "ci_upper": float(upper[i]),
                "confidence_level": 1 - alpha,
                "family_alpha": family_alpha,
                "multiplicity_adjustment": "Bonferroni across reference comparisons",
                "minimum_gain": minimum_gain,
                "passes": bool(observed[i] >= minimum_gain and lower[i] > 0),
                "all_cohort_gains_positive": bool(all(r[candidate] > r[ref] for r in rows)),
            }
        )
    return {
        "passes": all(row["passes"] for row in comparisons),
        "sampling": "patient clusters resampled within each cohort; equal cohort weight",
        "requested_samples": n_bootstrap,
        "successful_samples": len(draws),
        "comparisons": comparisons,
        "cohorts": rows,
        "scope": "conditional on frozen fitted models; not real-data validation",
    }
