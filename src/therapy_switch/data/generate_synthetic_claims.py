"""Generate fully synthetic longitudinal medical and pharmacy claims.

The generator creates learnable but noisy pre-index patterns and a configurable,
imbalanced future advanced-therapy outcome. It does not reproduce, infer, or
reverse-engineer any proprietary data source.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ._config import config_value, therapy_definition, timeline_days


def _date_between(rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    if end < start:
        return start
    return start + pd.Timedelta(days=int(rng.integers(0, (end - start).days + 1)))


def _choose_provider(
    rng: np.random.Generator,
    provider_groups: dict[str, np.ndarray],
    specialty: str,
) -> str:
    choices = provider_groups.get(specialty)
    if choices is None or len(choices) == 0:
        choices = provider_groups["__all__"]
    return str(rng.choice(choices))


def _safe_positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def generate_synthetic_claims(
    config: Mapping[str, Any] | Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Create realistic, entirely synthetic claims tables.

    Parameters
    ----------
    config:
        Mapping or config object. Common options are ``n_patients`` (default
        1,000), ``random_seed`` (42), ``target_prevalence`` (0.08), timeline
        windows, ``start_date``/``end_date``, ``n_providers``, and
        ``therapy_mapping``. Nested ``synthetic.*`` forms are also accepted.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``patients``, ``medical_claims``, ``pharmacy_claims``, and ``providers``.
        The synthetic patients table includes an ``index_date`` landmark used by
        the default cohort strategy. Latent variables and synthetic labels are
        intentionally omitted; outcomes must be derived from future claims.
    """

    config = {} if config is None else config
    n_patients = _safe_positive_int(
        config_value(
            config,
            "n_patients",
            "synthetic.n_patients",
            "data.synthetic.n_patients",
            default=1000,
        ),
        "n_patients",
    )
    if n_patients < 2:
        raise ValueError("n_patients must be at least 2 to create a binary outcome")
    seed = int(
        config_value(
            config,
            "random_seed",
            "seed",
            "synthetic.random_seed",
            "data.synthetic.random_seed",
            "project.random_seed",
            default=42,
        )
    )
    target_prevalence = float(
        config_value(
            config,
            "target_prevalence",
            "synthetic.target_prevalence",
            "data.synthetic.target_prevalence",
            default=0.08,
        )
    )
    if not 0 < target_prevalence < 1:
        raise ValueError("target_prevalence must be strictly between zero and one")
    observation_days, prediction_days = timeline_days(config)
    rng = np.random.default_rng(seed)
    therapies = therapy_definition(config)

    configured_start = pd.Timestamp(
        config_value(
            config,
            "start_date",
            "synthetic.start_date",
            "data.synthetic.start_date",
            default="2019-01-01",
        )
    ).normalize()
    configured_end = pd.Timestamp(
        config_value(
            config,
            "end_date",
            "synthetic.end_date",
            "data.synthetic.end_date",
            default="2024-12-31",
        )
    ).normalize()
    minimum_span = observation_days + prediction_days + 120
    if (configured_end - configured_start).days < minimum_span:
        configured_end = configured_start + pd.Timedelta(days=minimum_span)

    n_providers = _safe_positive_int(
        config_value(
            config,
            "n_providers",
            "synthetic.n_providers",
            "data.synthetic.n_providers",
            default=max(30, min(250, n_patients // 6)),
        ),
        "n_providers",
    )
    geographies = np.array(
        config_value(
            config,
            "synthetic.geographies",
            "data.synthetic.geographies",
            default=("NORTHEAST", "MIDWEST", "SOUTH", "WEST"),
        ),
        dtype=object,
    )
    specialties = np.array(
        config_value(
            config,
            "synthetic.specialties",
            "data.synthetic.specialties",
            default=(
                "primary_care",
                "rheumatology",
                "dermatology",
                "gastroenterology",
                "emergency_medicine",
                "hospitalist",
            ),
        ),
        dtype=object,
    )
    if geographies.size == 0 or specialties.size == 0:
        raise ValueError("Synthetic geographies and specialties cannot be empty")

    provider_ids = np.array([f"HCP{i:05d}" for i in range(1, n_providers + 1)])
    provider_specialties = rng.choice(
        specialties,
        size=n_providers,
        p=_normalized_weights([0.32, 0.17, 0.14, 0.12, 0.10, 0.15], len(specialties)),
    )
    providers = pd.DataFrame(
        {
            "provider_id": provider_ids,
            "specialty": provider_specialties,
            "geography": rng.choice(geographies, size=n_providers),
            "organization": [
                f"ORG{i:03d}" for i in rng.integers(1, max(5, n_providers // 4) + 1, n_providers)
            ],
        }
    )
    provider_groups = {
        specialty: providers.loc[providers["specialty"] == specialty, "provider_id"].to_numpy()
        for specialty in specialties
    }
    provider_groups["__all__"] = provider_ids

    patient_ids = np.array([f"PAT{i:07d}" for i in range(1, n_patients + 1)])
    ages = np.clip(np.rint(rng.normal(52, 15, n_patients)), 18, 89).astype(int)
    genders = rng.choice(np.array(["F", "M", "U"]), n_patients, p=[0.53, 0.45, 0.02])
    patient_geo = rng.choice(geographies, n_patients)

    earliest_index = configured_start + pd.Timedelta(days=observation_days + 35)
    latest_index = configured_end - pd.Timedelta(days=prediction_days + 35)
    index_dates = pd.to_datetime(
        [_date_between(rng, earliest_index, latest_index) for _ in range(n_patients)]
    )
    history_buffer = rng.integers(35, 181, n_patients)
    followup_buffer = rng.integers(35, 181, n_patients)
    observation_starts = pd.to_datetime(index_dates) - pd.to_timedelta(
        observation_days + history_buffer, unit="D"
    )
    observation_ends = pd.to_datetime(index_dates) + pd.to_timedelta(
        prediction_days + followup_buffer, unit="D"
    )
    reference_year = int(configured_end.year)
    patients = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "birth_year": reference_year - ages,
            "age": ages,
            "gender": genders,
            "geography": patient_geo,
            "observation_start": observation_starts,
            "observation_end": observation_ends,
            "index_date": index_dates,
        }
    )

    # Risk is only used to shape histories and generate outcomes. It is never
    # returned, preventing a synthetic-only shortcut feature.
    baseline_severity = rng.normal(0, 1, n_patients)
    escalation = rng.normal(0, 1, n_patients)
    specialist_affinity = rng.normal(0, 1, n_patients)
    signal_strength = float(
        config_value(
            config,
            "signal_strength",
            "synthetic.signal_strength",
            "data.synthetic.signal_strength",
            default=0.75,
        )
    )
    noise_scale = float(
        config_value(
            config,
            "noise_scale",
            "synthetic.noise_scale",
            "data.synthetic.noise_scale",
            default=1.0,
        )
    )
    if signal_strength < 0 or noise_scale <= 0:
        raise ValueError("signal_strength must be non-negative and noise_scale positive")
    risk_logit = signal_strength * (
        0.65 * baseline_severity
        + 0.85 * escalation
        + 0.35 * specialist_affinity
        + 0.20 * ((ages - 50) / 15)
        + 0.10 * (genders == "F")
    ) + rng.normal(0, 0.45 * noise_scale, n_patients)
    # Gumbel noise makes the exact-prevalence outcome stochastic and prevents a
    # deterministic feature rule, while preserving useful signal on average.
    n_positive = int(np.clip(round(target_prevalence * n_patients), 1, n_patients - 1))
    noisy_outcome_score = risk_logit + rng.gumbel(0, 0.90 * noise_scale, n_patients)
    positive_indices = np.argpartition(noisy_outcome_score, -n_positive)[-n_positive:]
    labels = np.zeros(n_patients, dtype=np.int8)
    labels[positive_indices] = 1

    conventional_ids = np.array(therapies.conventional_drug_ids, dtype=object)
    advanced_ids = np.array(therapies.advanced_drug_ids, dtype=object)
    conventional_class = therapies.conventional_classes[0]
    advanced_class = therapies.advanced_classes[0]
    specialist_names = [
        name
        for name in ("rheumatology", "dermatology", "gastroenterology")
        if name in set(specialties)
    ]
    if not specialist_names:
        specialist_names = [str(specialties[0])]

    medical_rows: list[dict[str, Any]] = []
    pharmacy_rows: list[dict[str, Any]] = []
    medical_claim_number = 1
    pharmacy_claim_number = 1

    for position, patient_id in enumerate(patient_ids):
        index_date = pd.Timestamp(index_dates[position])
        history_start = index_date - pd.Timedelta(days=observation_days)
        patient_severity = baseline_severity[position]
        patient_escalation = escalation[position]

        # Medical intensity is generated in monthly blocks. Positive time slopes
        # are correlated with, but by no means determine, the future outcome.
        n_blocks = int(np.ceil(observation_days / 30))
        for block in range(n_blocks):
            block_end_offset = block * 30
            block_start_offset = min(observation_days, block_end_offset + 29)
            recency = 1.0 - block / max(1, n_blocks - 1)
            intensity = np.exp(
                -0.05
                + 0.40 * patient_severity
                + 0.50 * patient_escalation * recency
                + rng.normal(0, 0.18)
            )
            event_count = int(rng.poisson(np.clip(intensity, 0.15, 7.0)))
            for _ in range(event_count):
                days_before = int(rng.integers(block_end_offset, block_start_offset + 1))
                claim_date = index_date - pd.Timedelta(days=days_before)
                specialist_probability = float(
                    np.clip(
                        0.21 + 0.14 * patient_severity + 0.18 * patient_escalation * recency,
                        0.06,
                        0.72,
                    )
                )
                is_specialist = rng.random() < specialist_probability
                if is_specialist:
                    specialty = str(rng.choice(specialist_names))
                    place = "outpatient"
                else:
                    place = str(
                        rng.choice(
                            ["outpatient", "inpatient", "emergency", "office"],
                            p=[0.42, 0.10, 0.10, 0.38],
                        )
                    )
                    specialty = {
                        "inpatient": "hospitalist",
                        "emergency": "emergency_medicine",
                    }.get(place, "primary_care")
                provider_id = _choose_provider(rng, provider_groups, specialty)
                severe_probability = float(np.clip(0.13 + 0.10 * patient_severity, 0.03, 0.34))
                diagnosis_code = str(
                    rng.choice(
                        ["DX_DISEASE", "DX_SEVERE", "DX_COMORBID_A", "DX_COMORBID_B", "DX_OTHER"],
                        p=[0.36, severe_probability, 0.14, 0.12, 0.38 - severe_probability],
                    )
                )
                procedure_probability = float(
                    np.clip(
                        0.11 + 0.08 * patient_severity + 0.12 * patient_escalation * recency,
                        0.03,
                        0.40,
                    )
                )
                procedure_code = (
                    str(rng.choice(["PROC_LAB", "PROC_IMAGING", "PROC_INFUSION_EVAL"]))
                    if rng.random() < procedure_probability
                    else pd.NA
                )
                medical_rows.append(
                    {
                        "claim_id": f"MED{medical_claim_number:010d}",
                        "patient_id": patient_id,
                        "claim_date": claim_date,
                        "diagnosis_code": diagnosis_code,
                        "procedure_code": procedure_code,
                        "provider_id": provider_id,
                        "place_of_service": place,
                    }
                )
                medical_claim_number += 1

        # A modest amount of post-index utilization is included deliberately so
        # leakage tests exercise the temporal filters.
        future_medical_count = int(rng.poisson(np.exp(-0.1 + 0.15 * patient_severity)))
        for _ in range(future_medical_count):
            claim_date = index_date + pd.Timedelta(days=int(rng.integers(1, prediction_days + 1)))
            provider_id = _choose_provider(rng, provider_groups, "primary_care")
            medical_rows.append(
                {
                    "claim_id": f"MED{medical_claim_number:010d}",
                    "patient_id": patient_id,
                    "claim_date": claim_date,
                    "diagnosis_code": str(rng.choice(["DX_DISEASE", "DX_OTHER"])),
                    "procedure_code": pd.NA,
                    "provider_id": provider_id,
                    "place_of_service": "outpatient",
                }
            )
            medical_claim_number += 1

        # Conventional fills span the history. More severe/escalating patients
        # are somewhat more likely to have changes and imperfect refill gaps.
        base_days_supply = int(rng.choice([28, 30, 60, 90], p=[0.12, 0.55, 0.13, 0.20]))
        fill_cursor = history_start + pd.Timedelta(days=int(rng.integers(0, 45)))
        current_drug = str(rng.choice(conventional_ids))
        change_probability = float(
            np.clip(
                0.025 + 0.050 * max(patient_severity, 0) + 0.080 * max(patient_escalation, 0),
                0.01,
                0.25,
            )
        )
        while fill_cursor <= index_date:
            if len(conventional_ids) > 1 and rng.random() < change_probability:
                alternatives = conventional_ids[conventional_ids != current_drug]
                current_drug = str(rng.choice(alternatives))
            prescriber_specialty = (
                str(rng.choice(specialist_names))
                if rng.random() < float(np.clip(0.30 + 0.08 * patient_severity, 0.10, 0.70))
                else "primary_care"
            )
            pharmacy_rows.append(
                {
                    "claim_id": f"RX{pharmacy_claim_number:010d}",
                    "patient_id": patient_id,
                    "fill_date": fill_cursor,
                    "drug_id": current_drug,
                    "therapy_class": conventional_class,
                    "quantity": int(rng.choice([28, 30, 60, 90])),
                    "days_supply": base_days_supply,
                    "prescriber_id": _choose_provider(rng, provider_groups, prescriber_specialty),
                }
            )
            pharmacy_claim_number += 1
            gap = int(
                np.clip(
                    base_days_supply + rng.normal(4 - 2 * patient_severity, 9),
                    14,
                    120,
                )
            )
            fill_cursor += pd.Timedelta(days=gap)

        # Place an index-date conventional claim to make the landmark clinically
        # interpretable without revealing the subsequent outcome.
        pharmacy_rows.append(
            {
                "claim_id": f"RX{pharmacy_claim_number:010d}",
                "patient_id": patient_id,
                "fill_date": index_date,
                "drug_id": current_drug,
                "therapy_class": conventional_class,
                "quantity": 30,
                "days_supply": 30,
                "prescriber_id": _choose_provider(
                    rng, provider_groups, str(rng.choice(specialist_names))
                ),
            }
        )
        pharmacy_claim_number += 1

        if labels[position] == 1:
            switch_day = int(rng.integers(5, prediction_days + 1))
            pharmacy_rows.append(
                {
                    "claim_id": f"RX{pharmacy_claim_number:010d}",
                    "patient_id": patient_id,
                    "fill_date": index_date + pd.Timedelta(days=switch_day),
                    "drug_id": str(rng.choice(advanced_ids)),
                    "therapy_class": advanced_class,
                    "quantity": 28,
                    "days_supply": int(rng.choice([28, 30])),
                    "prescriber_id": _choose_provider(
                        rng, provider_groups, str(rng.choice(specialist_names))
                    ),
                }
            )
            pharmacy_claim_number += 1
        elif rng.random() < 0.025 and followup_buffer[position] > 15:
            # Some controls switch only after the prediction window, which makes
            # the prediction-window definition observable in tests.
            switch_day = prediction_days + int(rng.integers(5, followup_buffer[position] + 1))
            pharmacy_rows.append(
                {
                    "claim_id": f"RX{pharmacy_claim_number:010d}",
                    "patient_id": patient_id,
                    "fill_date": index_date + pd.Timedelta(days=switch_day),
                    "drug_id": str(rng.choice(advanced_ids)),
                    "therapy_class": advanced_class,
                    "quantity": 28,
                    "days_supply": 28,
                    "prescriber_id": _choose_provider(
                        rng, provider_groups, str(rng.choice(specialist_names))
                    ),
                }
            )
            pharmacy_claim_number += 1

    medical_claims = pd.DataFrame.from_records(medical_rows)
    pharmacy_claims = pd.DataFrame.from_records(pharmacy_rows)
    for frame, date_column in (
        (medical_claims, "claim_date"),
        (pharmacy_claims, "fill_date"),
    ):
        frame[date_column] = pd.to_datetime(frame[date_column])
        frame.sort_values(["patient_id", date_column, "claim_id"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    patients.sort_values("patient_id", inplace=True)
    patients.reset_index(drop=True, inplace=True)
    providers.sort_values("provider_id", inplace=True)
    providers.reset_index(drop=True, inplace=True)
    return {
        "patients": patients,
        "medical_claims": medical_claims,
        "pharmacy_claims": pharmacy_claims,
        "providers": providers,
    }


def _normalized_weights(default: list[float], size: int) -> np.ndarray:
    """Resize a default categorical distribution without hiding assumptions."""

    if size <= len(default):
        weights = np.asarray(default[:size], dtype=float)
    else:
        weights = np.concatenate(
            [np.asarray(default, dtype=float), np.ones(size - len(default), dtype=float)]
        )
    return weights / weights.sum()
