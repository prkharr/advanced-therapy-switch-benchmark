"""Synthetic claims; noisy future outcomes are drawn from already generated history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._config import config_value, therapy_definition, timeline_days


def generate_synthetic_claims(config=None):
    config = config or {}
    if config_value(config, "data.kind") == "real":
        raise ValueError("Real-data configuration cannot generate synthetic claims")

    def option(key, default):
        return config_value(config, key, f"data.synthetic.{key}", default=default)

    n = int(option("n_patients", 3000))
    rng = np.random.default_rng(
        int(config_value(config, "random_seed", "project.random_seed", default=42))
    )
    prevalence, noise, strength = (
        float(option(k, d))
        for k, d in [("target_prevalence", 0.10), ("noise_scale", 0.8), ("signal_strength", 1.0)]
    )
    if n < 2 or not 0 < prevalence < 1 or noise <= 0 or strength < 0:
        raise ValueError("Invalid synthetic size, prevalence, noise or signal strength")
    history, horizon = timeline_days(config)
    count = int(option("n_providers", 100))
    if count < 1:
        raise ValueError("n_providers must be positive")
    therapy = therapy_definition(config)
    conv, adv = therapy.conventional_drug_ids, therapy.advanced_drug_ids
    providers = pd.DataFrame(
        {
            "provider_id": [f"SYN_HCP_{i:05d}" for i in range(count)],
            "specialty": rng.choice(
                ["primary_care", "sleep_medicine", "neurology", "pulmonology", "psychiatry"], count
            ),
            "geography": rng.choice(["SYN_NORTH", "SYN_SOUTH", "SYN_WEST"], count),
            "organization": [f"SYN_ORG_{i % 15}" for i in range(count)],
        }
    )
    plans = pd.DataFrame(
        {
            "plan_id": ["SYN_PLAN_A", "SYN_PLAN_B", "SYN_PLAN_C"],
            "payer_type": ["commercial", "public", "other"],
        }
    )
    start, end = (
        pd.Timestamp(option("start_date", "2020-01-01")),
        pd.Timestamp(option("end_date", "2025-12-31")),
    )
    months = int(config_value(config, "timeline.snapshots_per_patient", default=3))
    first = (start + pd.Timedelta(days=history + 90)).to_period("M").to_timestamp() + pd.DateOffset(
        months=1
    )
    last = (end - pd.Timedelta(days=horizon + 60 + 31 * months)).to_period("M").to_timestamp()
    anchors = pd.date_range(first, last, freq="MS")
    if len(anchors) < 12 or months < 1:
        raise ValueError("Synthetic dates must allow history, follow-up and 12 index months")
    patient_rows, med_rows, rx_rows, risks, landmarks = [], [], [], [], []
    pids = [f"SYN_PAT_{i:07d}" for i in range(n)]
    for pid in pids:
        anchor = pd.Timestamp(rng.choice(anchors))
        landmark = anchor + pd.DateOffset(months=months - 1)
        landmarks.append(landmark)
        begin = anchor - pd.Timedelta(days=history + 90)
        severity, slope = rng.normal(size=2)
        hcp, plan = str(rng.choice(providers.provider_id)), str(rng.choice(plans.plan_id))
        patient_rows.append(
            {
                "patient_id": pid,
                "birth_year": int(rng.integers(1940, 2003)),
                "gender": str(rng.choice(["F", "M", "U"])),
                "geography": str(rng.choice(providers.geography)),
                "observation_start": begin,
                "observation_end": end,
                "index_date": anchor,
                "plan_id": plan,
            }
        )
        patient_med, patient_rx = [], []
        # Artificial confirmation codes exercise a configurable rule; no clinical mapping.
        for offset in (history + 40, history - 80):
            patient_med.append((anchor - pd.Timedelta(days=offset), "SYN_NT1", None, "office", hcp))
        for date in pd.date_range(begin, landmark, freq="7D"):
            recent = (date - begin).days / max(1, (landmark - begin).days)
            rate = np.clip(np.exp(-1.3 + 0.45 * severity + 0.85 * slope * recent), 0.08, 3)
            for _ in range(rng.poisson(rate)):
                code = str(
                    rng.choice(
                        ["SYN_NT1", "SYN_SYMPTOM", "SYN_COMORBID", "SYN_OTHER"],
                        p=[0.2, 0.35, 0.2, 0.25],
                    )
                )
                patient_med.append(
                    (
                        date,
                        code,
                        "SYN_TEST" if rng.random() < 0.3 else None,
                        str(
                            rng.choice(
                                ["office", "outpatient", "emergency", "urgent_care"],
                                p=[0.50, 0.35, 0.10, 0.05],
                            )
                        ),
                        str(rng.choice(providers.provider_id)),
                    )
                )
        cursor, drug = begin, str(rng.choice(conv))
        while cursor <= landmark:
            if rng.random() < np.clip(0.1 + 0.12 * max(slope, 0), 0.05, 0.5):
                drug = str(rng.choice(conv))
            status = str(rng.choice(["paid", "rejected"], p=[0.93, 0.07]))
            delay = int(rng.choice([0, 3, 10, 45], p=[0.5, 0.3, 0.15, 0.05]))
            reversal = (
                cursor + pd.Timedelta(days=60)
                if status == "paid" and rng.random() < 0.03
                else pd.NaT
            )
            patient_rx.append(
                {
                    "patient_id": pid,
                    "fill_date": cursor,
                    "drug_id": drug,
                    "therapy_class": "conventional",
                    "quantity": 30,
                    "days_supply": 30,
                    "prescriber_id": hcp,
                    "plan_id": plan,
                    "status": status,
                    "available_date": cursor + pd.Timedelta(days=delay),
                    "reversal_date": reversal,
                    "patient_cost": float(rng.gamma(2, 10)),
                }
            )
            recent = (cursor - begin).days / max(1, (landmark - begin).days)
            cursor += pd.Timedelta(
                days=int(np.clip(rng.normal(27 + max(slope, 0) * recent * 8, 6), 14, 65))
            )
        # Deliberately retain post-index raw events to exercise leakage filters.
        for _ in range(rng.poisson(3)):
            patient_med.append(
                (
                    landmark + pd.Timedelta(days=int(rng.integers(1, horizon + 1))),
                    "SYN_SYMPTOM",
                    "SYN_TEST",
                    "office",
                    hcp,
                )
            )
        recent_med = sum(0 <= (landmark - r[0]).days <= 90 for r in patient_med)
        prior_med = sum(90 < (landmark - r[0]).days <= 180 for r in patient_med)
        recent_rx = [r for r in patient_rx if (landmark - r["fill_date"]).days <= 180]
        changes = sum(a["drug_id"] != b["drug_id"] for a, b in zip(recent_rx, recent_rx[1:]))
        rejected = sum(r["status"] == "rejected" for r in recent_rx)
        last_gap = (landmark - patient_rx[-1]["fill_date"]).days
        risks.append(
            strength
            * (
                0.18 * recent_med
                + 0.14 * (recent_med - prior_med)
                + 0.32 * changes
                + 0.18 * rejected
                + 0.012 * last_gap
            )
            + rng.normal(0, noise)
        )
        for date, diagnosis, procedure, place, provider in patient_med:
            med_rows.append(
                {
                    "patient_id": pid,
                    "claim_date": date,
                    "diagnosis_code": diagnosis,
                    "procedure_code": procedure,
                    "provider_id": provider,
                    "place_of_service": place,
                    "status": "final",
                    "available_date": date + pd.Timedelta(days=int(rng.choice([0, 3, 10]))),
                    "reversal_date": pd.NaT,
                }
            )
        rx_rows.extend(patient_rx)
    # Calibrate expected prevalence, then make independent Bernoulli draws.
    # Cohort filtering and repeated windows may change realized snapshot prevalence.
    risk, lo, hi = np.asarray(risks), -40.0, 40.0
    for _ in range(80):
        intercept = (lo + hi) / 2
        probability = 1 / (1 + np.exp(-np.clip(risk + intercept, -40, 40)))
        if probability.mean() > prevalence:
            hi = intercept
        else:
            lo = intercept
    outcomes = rng.random(n) < probability
    for pid, landmark, positive in zip(pids, landmarks, outcomes):
        if positive or rng.random() < 0.025:
            day = int(rng.integers(1, horizon + 1)) if positive else horizon + 20
            date = landmark + pd.Timedelta(days=day)
            rx_rows.append(
                {
                    "patient_id": pid,
                    "fill_date": date,
                    "drug_id": str(rng.choice(adv)),
                    "therapy_class": "advanced",
                    "quantity": 30,
                    "days_supply": 30,
                    "prescriber_id": str(rng.choice(providers.provider_id)),
                    "plan_id": "SYN_PLAN_A",
                    "status": "paid",
                    "available_date": date + pd.Timedelta(days=5),
                    "reversal_date": pd.NaT,
                    "patient_cost": float(rng.gamma(2, 30)),
                }
            )
    medical, pharmacy = pd.DataFrame(med_rows), pd.DataFrame(rx_rows)
    for frame, prefix, date_col in ((medical, "MED", "claim_date"), (pharmacy, "RX", "fill_date")):
        frame["claim_id"] = [f"SYN_{prefix}_{i:09d}" for i in range(len(frame))]
        frame.sort_values(["patient_id", date_col, "claim_id"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    patients = pd.DataFrame(patient_rows)
    enrollment = patients[["patient_id", "observation_start", "observation_end", "plan_id"]].rename(
        columns={"observation_start": "coverage_start", "observation_end": "coverage_end"}
    )
    mapping = pd.DataFrame(
        {
            "drug_id": list(conv) + list(adv),
            "therapy_class": ["conventional"] * len(conv) + ["advanced"] * len(adv),
            "effective_start": start,
            "effective_end": end,
        }
    )
    return {
        "patients": patients,
        "medical_claims": medical,
        "pharmacy_claims": pharmacy,
        "providers": providers,
        "plans": plans,
        "enrollment": enrollment,
        "therapy_mapping": mapping,
    }
