"""Paired uncertainty for distinct-patient capture at a fixed list capacity."""

from __future__ import annotations

import numpy as np

from therapy_switch.evaluation.metrics import top_k_count, validate_predictions


def _recall(y, scores, fraction):
    chosen = np.argsort(-scores, kind="stable")[: top_k_count(len(y), fraction)]
    return float(y[chosen].sum() / y.sum())


def confirm_patient_capture(
    frame,
    *,
    candidate,
    references,
    fraction=0.1,
    n_bootstrap=3000,
    family_alpha=0.05,
    minimum_gain=0.03,
    seed=919,
):
    """Paired patient bootstrap, separately within each cohort, with equal cohort weight.

    Recompute each model's top-capacity list inside every bootstrap sample.
    The data must already contain one latest eligible assessment per patient.
    These intervals condition on the fitted models, not on model retraining.
    """
    references = list(references)
    if not references or len(set(references)) != len(references) or candidate in references:
        raise ValueError("Distinct candidate and unique references are required")
    if n_bootstrap < 100 or not 0 < family_alpha < 1 or minimum_gain < 0:
        raise ValueError("Invalid confirmation settings")
    if frame.empty or frame[["cohort", "patient_id"]].isna().any().any():
        raise ValueError("Nonempty cohort and patient keys required")
    if frame.duplicated(["cohort", "patient_id"]).any():
        raise ValueError("Confirmation requires one assessment per patient")
    cohorts, rows = [], []
    names = [candidate, *references]
    for cohort, part in frame.groupby("cohort", sort=True):
        part = part.sort_values("patient_id", kind="stable")
        scores = {}
        for name in names:
            y, scores[name] = validate_predictions(part.label, part[name])
        if np.unique(y).size != 2:
            raise ValueError("Every cohort requires both classes")
        estimated = {name: _recall(y, p, fraction) for name, p in scores.items()}
        rows.append(
            {
                "cohort": cohort,
                "patients": len(y),
                "positives": int(y.sum()),
                "selected": top_k_count(len(y), fraction),
                **estimated,
            }
        )
        cohorts.append((y, scores))
    observed = np.array([np.mean([r[candidate] - r[ref] for r in rows]) for ref in references])
    rng, draws = np.random.default_rng(seed), []
    for _ in range(n_bootstrap):
        differences = []
        for y, scores in cohorts:
            # Sorting preserves patient-ID tie breaks, independently of outcomes.
            ids = np.sort(rng.integers(0, len(y), len(y)))
            if not y[ids].sum():
                break
            challenger = _recall(y[ids], scores[candidate][ids], fraction)
            differences.append(
                [challenger - _recall(y[ids], scores[r][ids], fraction) for r in references]
            )
        if len(differences) == len(cohorts):
            draws.append(np.mean(differences, axis=0))
    if len(draws) < max(100, n_bootstrap // 2):
        raise ValueError("Too few valid patient bootstrap draws")
    alpha = family_alpha / len(references)
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    comparisons = []
    for i, reference in enumerate(references):
        comparisons.append(
            {
                "candidate": candidate,
                "reference": reference,
                "candidate_mean_recall": float(np.mean([r[candidate] for r in rows])),
                "reference_mean_recall": float(np.mean([r[reference] for r in rows])),
                "mean_recall_gain": float(observed[i]),
                "ci_lower": float(lower[i]),
                "ci_upper": float(upper[i]),
                "confidence_level": 1 - alpha,
                "minimum_gain": minimum_gain,
                "passes": bool(observed[i] >= minimum_gain and lower[i] > 0),
                "all_cohort_gains_positive": all(r[candidate] > r[reference] for r in rows),
            }
        )
    return {
        "passes": all(r["passes"] for r in comparisons),
        "comparisons": comparisons,
        "cohorts": rows,
        "successful_samples": len(draws),
        "requested_samples": n_bootstrap,
        "capacity": fraction,
        "family_alpha": family_alpha,
        "multiplicity_adjustment": "Bonferroni across prespecified reference comparisons",
        "sampling": "paired patients within cohort, re-ranked each draw; equal cohort weight",
        "scope": "conditional on frozen fitted models; synthetic historical assessments",
    }
