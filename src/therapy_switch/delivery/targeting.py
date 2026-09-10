"""Aggregate top-capacity patient counts into an HCP-only field file."""

from __future__ import annotations

import numpy as np
import pandas as pd

from therapy_switch.hcp.hcp_prioritization import AttributionConfig, attribute_patients_to_hcp

FIELD_COLUMNS = [
    "hcp_rank",
    "hcp_id",
    "specialty",
    "geography",
    "organization",
    "eligible_patient_count",
    "priority_patient_count",
    "priority_share_percent",
    "priority_tier",
    "targeting_reason",
    "assessment_date",
]


def build_field_targets(patient_list, events, providers, settings, scoring_date):
    linked = events.loc[events.snapshot_id.isin(patient_list.snapshot_id)].copy()
    linked = linked.rename(columns={"hcp_id": "provider_id", "provider_specialty": "specialty"})
    linked["role"] = np.where(linked.event_type.str.startswith("pharmacy"), "prescriber", "visit")
    specialties = settings.get("relevant_specialties", ["sleep_medicine", "neurology"])
    linked["is_relevant"] = linked.status.isin(["paid", "final"]) & (
        linked.therapy_class.eq("conventional") | linked.specialty.isin(specialties)
    )
    attribution = attribute_patients_to_hcp(
        linked,
        config=AttributionConfig(
            method=settings.get("attribution_rule", "most_recent_relevant_prescriber"),
            date_col="event_date",
            relevant_col="is_relevant",
            role_col="role",
            specialist_specialties=tuple(specialties),
        ),
    )
    known = set(providers.provider_id.dropna())
    if not set(attribution.hcp_id.dropna()).issubset(known):
        raise ValueError("Attribution contains an HCP absent from the provider directory")
    patients = patient_list.merge(attribution, on="patient_id", how="left", validate="one_to_one")
    assigned = patients.loc[patients.hcp_id.notna()]
    coverage = {
        "exported_priority_patients": 0,
        "withheld_priority_patients": 0,
        "eligible_patients": len(patients),
        "priority_patients": int(patients.selected.sum()),
        "attributed_patients": len(assigned),
        "unattributed_patients": int(patients.hcp_id.isna().sum()),
        "unattributed_priority_patients": int(
            patients.loc[patients.hcp_id.isna(), "selected"].sum()
        ),
    }
    if assigned.empty:
        return pd.DataFrame(columns=FIELD_COLUMNS), patients, coverage
    hcp = (
        assigned.groupby("hcp_id", sort=True)
        .agg(
            eligible_patient_count=("patient_id", "nunique"),
            priority_patient_count=("selected", "sum"),
        )
        .reset_index()
    )
    hcp["priority_patient_count"] = hcp.priority_patient_count.astype(int)
    hcp["priority_share_percent"] = (
        100 * hcp.priority_patient_count / hcp.eligible_patient_count
    ).round(2)
    hcp = hcp.loc[hcp.priority_patient_count >= int(settings.get("minimum_priority_patients", 1))]
    coverage["exported_priority_patients"] = int(hcp.priority_patient_count.sum())
    coverage["withheld_priority_patients"] = (
        int(assigned.selected.sum()) - coverage["exported_priority_patients"]
    )
    if hcp.empty:
        return pd.DataFrame(columns=FIELD_COLUMNS), patients, coverage
    hcp = hcp.merge(
        providers.rename(columns={"provider_id": "hcp_id"}),
        on="hcp_id",
        how="left",
        validate="one_to_one",
    )
    hcp = hcp.sort_values(
        ["priority_patient_count", "priority_share_percent", "hcp_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    hcp["hcp_rank"] = np.arange(1, len(hcp) + 1)
    tiers = settings.get("tier_rank_percentiles", [0.2, 0.5])
    hcp["priority_tier"] = np.select(
        [
            hcp.hcp_rank <= max(1, int(np.ceil(len(hcp) * tiers[0]))),
            hcp.hcp_rank <= max(1, int(np.ceil(len(hcp) * tiers[1]))),
        ],
        ["Tier 1", "Tier 2"],
        default="Tier 3",
    )
    hcp["targeting_reason"] = hcp.apply(
        lambda row: (
            f"{int(row.priority_patient_count)} priority patients among {int(row.eligible_patient_count)} attributed eligible patients"
        ),
        axis=1,
    )
    hcp["assessment_date"] = str(pd.Timestamp(scoring_date).date())
    return hcp[FIELD_COLUMNS], patients, coverage


def csv_safe(frame):
    """Keep spreadsheet software from executing text fields as formulas."""
    out = frame.copy()
    for name in out.select_dtypes(include=["object", "string"]).columns:
        out[name] = out[name].map(
            lambda v: (
                "'" + v
                if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
                else v
            )
        )
    return out
