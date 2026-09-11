"""Tests for auditable HCP attribution and opportunity scoring."""

from __future__ import annotations

import pandas as pd
import pytest

from therapy_switch.hcp.hcp_prioritization import (
    AttributionConfig,
    HCPScoringConfig,
    OpportunityConfig,
    attribute_patients_to_hcp,
    calculate_hcp_opportunity_metrics,
    score_hcp_opportunities,
)


@pytest.fixture
def provider_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p2", "p2", "p3"],
            "provider_id": ["A", "A", "B", "C", "D", "E"],
            "claim_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-15",
                    "2024-02-01",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-03-01",
                ]
            ),
            "specialty": [
                "Rheumatology",
                "Rheumatology",
                "Primary Care",
                "Rheumatology",
                "Rheumatology",
                "Rheumatology",
            ],
            "is_relevant": [True, True, True, True, True, True],
        }
    )


def test_recent_and_plurality_attribution_are_distinct_and_deterministic(
    provider_events: pd.DataFrame,
) -> None:
    recent = attribute_patients_to_hcp(
        provider_events,
        config=AttributionConfig(
            method="most_recent_relevant_prescriber", relevant_col="is_relevant"
        ),
    ).set_index("patient_id")
    plurality = attribute_patients_to_hcp(
        provider_events,
        config=AttributionConfig(method="plurality_of_relevant_claims", relevant_col="is_relevant"),
    ).set_index("patient_id")

    assert recent.loc["p1", "hcp_id"] == "B"
    assert plurality.loc["p1", "hcp_id"] == "A"
    # p2 is tied on count, so the most recent claim wins.
    assert plurality.loc["p2", "hcp_id"] == "D"


def test_frequent_specialist_respects_configured_specialty(
    provider_events: pd.DataFrame,
) -> None:
    mapping = attribute_patients_to_hcp(
        provider_events,
        config=AttributionConfig(
            method="most_frequent_relevant_specialist",
            relevant_col="is_relevant",
            specialist_specialties=("Rheumatology",),
        ),
    ).set_index("patient_id")
    assert mapping.loc["p1", "hcp_id"] == "A"


def test_existing_business_attribution_takes_precedence(
    provider_events: pd.DataFrame,
) -> None:
    existing = pd.DataFrame({"patient_id": ["p1"], "attributed_provider_id": ["BUSINESS_HCP"]})
    mapping = attribute_patients_to_hcp(
        provider_events,
        config=AttributionConfig(method="plurality_of_relevant_claims"),
        existing_attribution=existing,
    ).set_index("patient_id")
    assert mapping.loc["p1", "hcp_id"] == "BUSINESS_HCP"
    assert mapping.loc["p1", "attribution_method"] == "existing_attribution"


def test_hcp_metrics_and_default_score_are_probability_sums() -> None:
    scores = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "advanced_therapy_propensity_score": [0.90, 0.80, 0.20, 0.10],
        }
    )
    attribution = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "hcp_id": ["h1", "h1", "h2", "h2"],
        }
    )
    metrics = calculate_hcp_opportunity_metrics(
        scores,
        attribution,
        config=OpportunityConfig(high_propensity_threshold=0.75),
    ).set_index("hcp_id")

    assert metrics.loc["h1", "eligible_patient_count"] == 2
    assert metrics.loc["h1", "high_propensity_patient_count"] == 2
    assert metrics.loc["h1", "patients_top_5pct"] == 1
    assert metrics.loc["h1", "patients_top_10pct"] == 1
    assert metrics.loc["h1", "expected_switchers"] == pytest.approx(1.7)
    assert metrics.loc["h2", "expected_switchers"] == pytest.approx(0.3)

    prioritized = score_hcp_opportunities(metrics.reset_index()).set_index("hcp_id")
    assert prioritized.loc["h1", "hcp_opportunity_score"] == pytest.approx(1.7)
    assert prioritized.loc["h1", "hcp_rank"] == 1
    assert "expected_switchers" in prioritized.loc["h1", "opportunity_score_formula"]


def test_high_propensity_count_can_use_configured_percentile() -> None:
    scores = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "advanced_therapy_propensity_score": [0.9, 0.8, 0.2, 0.1],
        }
    )
    attribution = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "hcp_id": ["h1", "h1", "h2", "h2"],
        }
    )
    metrics = calculate_hcp_opportunity_metrics(
        scores,
        attribution,
        config=OpportunityConfig(
            high_propensity_threshold=None,
            high_propensity_threshold_percentile=75,
        ),
    ).set_index("hcp_id")
    assert metrics["high_propensity_patient_count"].sum() == 1
    assert metrics.loc["h1", "high_propensity_patient_count"] == 1


def test_weighted_score_exposes_each_component() -> None:
    metrics = pd.DataFrame(
        {
            "hcp_id": ["h1", "h2"],
            "expected_switchers": [2.0, 1.0],
            "patients_top_10pct": [1, 3],
        }
    )
    scored = score_hcp_opportunities(
        metrics,
        config=HCPScoringConfig(
            weights={"expected_switchers": 0.6, "patients_top_10pct": 0.4},
            normalization="minmax",
        ),
    )
    assert "score_component__expected_switchers" in scored.columns
    assert "score_component__patients_top_10pct" in scored.columns
    assert "opportunity_score_formula" in scored.columns
