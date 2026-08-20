"""Patient-to-HCP attribution and transparent HCP prioritization."""

from .hcp_prioritization import (
    ATTRIBUTION_METHODS,
    AttributionConfig,
    HCPScoringConfig,
    OpportunityConfig,
    aggregate_hcp_opportunity,
    attribute_patients,
    attribute_patients_to_hcp,
    build_hcp_targeting_output,
    calculate_hcp_opportunity_metrics,
    prioritize_hcps,
    save_hcp_targeting_output,
    score_hcp_opportunities,
)

__all__ = [
    "ATTRIBUTION_METHODS",
    "AttributionConfig",
    "HCPScoringConfig",
    "OpportunityConfig",
    "aggregate_hcp_opportunity",
    "attribute_patients",
    "attribute_patients_to_hcp",
    "build_hcp_targeting_output",
    "calculate_hcp_opportunity_metrics",
    "prioritize_hcps",
    "save_hcp_targeting_output",
    "score_hcp_opportunities",
]
