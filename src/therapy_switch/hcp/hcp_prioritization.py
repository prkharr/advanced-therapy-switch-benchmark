"""Configurable patient-to-HCP attribution and opportunity prioritization.

Patient prediction and HCP prioritization are deliberately separate layers.
This module consumes already-generated patient probabilities; it never trains
or changes the patient model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ATTRIBUTION_METHODS = {
    "most_recent_relevant_prescriber",
    "most_frequent_relevant_specialist",
    "plurality_of_relevant_claims",
    "existing_attribution",
}

ATTRIBUTION_ALIASES = {
    "recent_prescriber": "most_recent_relevant_prescriber",
    "most_recent_prescriber": "most_recent_relevant_prescriber",
    "frequent_specialist": "most_frequent_relevant_specialist",
    "most_frequent_specialist": "most_frequent_relevant_specialist",
    "plurality": "plurality_of_relevant_claims",
    "existing": "existing_attribution",
}


@dataclass(frozen=True)
class AttributionConfig:
    """Column mappings and the business-approved attribution rule."""

    method: str = "most_recent_relevant_prescriber"
    patient_col: str = "patient_id"
    hcp_col: str = "provider_id"
    date_col: str = "claim_date"
    specialty_col: str = "specialty"
    relevant_col: str | None = None
    role_col: str | None = None
    prescriber_roles: tuple[str, ...] = ("prescriber",)
    specialist_specialties: tuple[str, ...] = ()
    existing_hcp_col: str = "attributed_provider_id"
    prefer_existing_attribution: bool = True

    @property
    def normalized_method(self) -> str:
        normalized = self.method.lower().strip().replace(" ", "_")
        return ATTRIBUTION_ALIASES.get(normalized, normalized)

    def validate(self) -> None:
        if self.normalized_method not in ATTRIBUTION_METHODS:
            raise ValueError(
                f"Unknown attribution method '{self.method}'. Valid methods: "
                f"{sorted(ATTRIBUTION_METHODS)}"
            )


@dataclass(frozen=True)
class OpportunityConfig:
    """Definitions used to aggregate patient probabilities by attributed HCP."""

    patient_col: str = "patient_id"
    score_col: str = "advanced_therapy_propensity_score"
    hcp_col: str = "hcp_id"
    high_propensity_threshold: float | None = 0.5
    high_propensity_threshold_percentile: float | None = None
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20)

    def validate(self) -> None:
        if self.high_propensity_threshold is not None and not (
            0.0 <= self.high_propensity_threshold <= 1.0
        ):
            raise ValueError("high_propensity_threshold must be in [0, 1].")
        if self.high_propensity_threshold_percentile is not None and not (
            0.0 <= self.high_propensity_threshold_percentile <= 100.0
        ):
            raise ValueError("high_propensity_threshold_percentile must be in [0, 100].")
        if (
            self.high_propensity_threshold is None
            and self.high_propensity_threshold_percentile is None
        ):
            raise ValueError(
                "Set either high_propensity_threshold or high_propensity_threshold_percentile."
            )
        if not self.top_fractions:
            raise ValueError("top_fractions must not be empty.")
        if any(not 0.0 < value <= 1.0 for value in self.top_fractions):
            raise ValueError("Every top fraction must be in (0, 1].")
        if len(set(self.top_fractions)) != len(self.top_fractions):
            raise ValueError("top_fractions must be unique.")


@dataclass(frozen=True)
class HCPScoringConfig:
    """Transparent weighted formula for an HCP opportunity score.

    The default deliberately preserves ``expected_switchers`` as the score.
    Alternative multi-component formulas must be chosen explicitly by changing
    ``weights`` and, where units differ, ``normalization``.
    """

    weights: Mapping[str, float] = field(default_factory=lambda: {"expected_switchers": 1.0})
    normalization: str = "none"

    def validate(self) -> None:
        if not self.weights:
            raise ValueError("At least one score-component weight is required.")
        values = np.asarray(list(self.weights.values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Score weights must be finite and non-negative.")
        if float(values.sum()) <= 0.0:
            raise ValueError("At least one score weight must be positive.")
        if self.normalization not in {"none", "minmax", "percentile"}:
            raise ValueError("normalization must be none, minmax, or percentile.")


def _required_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _truthy_relevance(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0.0)
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .isin({"1", "true", "yes", "y", "relevant"})
    )


def _standardize_existing_attribution(
    existing: pd.DataFrame,
    config: AttributionConfig,
) -> pd.DataFrame:
    hcp_candidates = (config.existing_hcp_col, config.hcp_col, "hcp_id")
    hcp_source = next(
        (
            column
            for column in hcp_candidates
            if column in existing.columns and existing[column].notna().any()
        ),
        None,
    )
    if hcp_source is None:
        raise ValueError(
            "existing_attribution is missing an HCP column; expected one of "
            f"{list(dict.fromkeys(hcp_candidates))}."
        )
    _required_columns(existing, [config.patient_col, hcp_source], "existing_attribution")
    mapping = existing[[config.patient_col, hcp_source]].dropna().drop_duplicates()
    conflicts = mapping.groupby(config.patient_col)[hcp_source].nunique()
    if (conflicts > 1).any():
        examples = conflicts[conflicts > 1].index.astype(str).tolist()[:5]
        raise ValueError(
            f"Existing attribution contains multiple HCPs for a patient; examples: {examples}"
        )
    mapping = mapping.drop_duplicates(config.patient_col).rename(
        columns={config.patient_col: "patient_id", hcp_source: "hcp_id"}
    )
    mapping["attribution_method"] = "existing_attribution"
    mapping["attribution_claim_count"] = pd.Series(
        pd.array([pd.NA] * len(mapping), dtype="Int64"), index=mapping.index
    )
    mapping["attribution_last_date"] = pd.NaT
    return mapping.reset_index(drop=True)


def _filter_events(events: pd.DataFrame, config: AttributionConfig) -> pd.DataFrame:
    required = [config.patient_col, config.hcp_col]
    if config.normalized_method != "existing_attribution":
        required.append(config.date_col)
    _required_columns(events, required, "provider_events")
    selected = events.copy()
    selected = selected.dropna(subset=[config.patient_col, config.hcp_col])
    if config.relevant_col is not None:
        _required_columns(selected, [config.relevant_col], "provider_events")
        selected = selected[_truthy_relevance(selected[config.relevant_col])]

    if (
        config.normalized_method == "most_recent_relevant_prescriber"
        and config.role_col is not None
    ):
        _required_columns(selected, [config.role_col], "provider_events")
        allowed = {role.lower() for role in config.prescriber_roles}
        selected = selected[
            selected[config.role_col].fillna("").astype(str).str.lower().isin(allowed)
        ]

    if config.normalized_method == "most_frequent_relevant_specialist":
        _required_columns(selected, [config.specialty_col], "provider_events")
        if config.specialist_specialties:
            allowed_specialties = {specialty.lower() for specialty in config.specialist_specialties}
            selected = selected[
                selected[config.specialty_col]
                .fillna("")
                .astype(str)
                .str.lower()
                .isin(allowed_specialties)
            ]
        else:
            # Without a supplied specialty list, only rows explicitly carrying a
            # specialty are candidates; no proprietary specialty is assumed.
            selected = selected[selected[config.specialty_col].notna()]
    return selected


def _rule_based_attribution(events: pd.DataFrame, config: AttributionConfig) -> pd.DataFrame:
    selected = _filter_events(events, config)
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "hcp_id",
                "attribution_method",
                "attribution_claim_count",
                "attribution_last_date",
            ]
        )
    selected = selected.copy()
    selected["_event_date"] = pd.to_datetime(selected[config.date_col], errors="coerce")
    selected = selected.dropna(subset=["_event_date"])
    if selected.empty:
        raise ValueError(f"No valid dates remain in provider_events['{config.date_col}'].")
    selected["_hcp_tie_key"] = selected[config.hcp_col].astype(str)
    grouped = (
        selected.groupby(
            [config.patient_col, config.hcp_col, "_hcp_tie_key"],
            dropna=False,
            sort=False,
        )
        .agg(
            attribution_claim_count=(config.hcp_col, "size"),
            attribution_last_date=("_event_date", "max"),
        )
        .reset_index()
    )
    if config.normalized_method == "most_recent_relevant_prescriber":
        ordered = grouped.sort_values(
            [
                config.patient_col,
                "attribution_last_date",
                "attribution_claim_count",
                "_hcp_tie_key",
            ],
            ascending=[True, False, False, True],
            kind="stable",
        )
    else:
        ordered = grouped.sort_values(
            [
                config.patient_col,
                "attribution_claim_count",
                "attribution_last_date",
                "_hcp_tie_key",
            ],
            ascending=[True, False, False, True],
            kind="stable",
        )
    winners = ordered.drop_duplicates(config.patient_col, keep="first").rename(
        columns={config.patient_col: "patient_id", config.hcp_col: "hcp_id"}
    )
    winners["attribution_method"] = config.normalized_method
    return winners[
        [
            "patient_id",
            "hcp_id",
            "attribution_method",
            "attribution_claim_count",
            "attribution_last_date",
        ]
    ].reset_index(drop=True)


def attribute_patients_to_hcp(
    provider_events: pd.DataFrame,
    *,
    config: AttributionConfig | None = None,
    existing_attribution: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attribute each patient using an explicit, configurable rule.

    If an existing business-defined attribution is provided (or present in the
    configured existing-HCP column), it takes precedence by default.  A
    rule-based mapping fills only patients not covered by that existing mapping.
    Set ``prefer_existing_attribution=False`` to apply the configured rule to all
    patients explicitly.
    """

    resolved = config or AttributionConfig()
    resolved.validate()

    existing: pd.DataFrame | None = existing_attribution
    if (
        existing is None
        and resolved.existing_hcp_col in provider_events.columns
        and provider_events[resolved.existing_hcp_col].notna().any()
    ):
        existing = provider_events

    if resolved.normalized_method == "existing_attribution":
        if existing is None:
            raise ValueError(
                "method='existing_attribution' requires existing_attribution or "
                f"the '{resolved.existing_hcp_col}' column."
            )
        return _standardize_existing_attribution(existing, resolved)

    rule_based = _rule_based_attribution(provider_events, resolved)
    if existing is None or not resolved.prefer_existing_attribution:
        return rule_based

    existing_mapping = _standardize_existing_attribution(existing, resolved)
    uncovered = rule_based[~rule_based["patient_id"].isin(existing_mapping["patient_id"])]
    return pd.concat([existing_mapping, uncovered], ignore_index=True)


def _top_flag_name(fraction: float) -> str:
    percentage = fraction * 100.0
    label = f"{percentage:g}".replace(".", "_p")
    return f"patients_top_{label}pct"


def calculate_hcp_opportunity_metrics(
    patient_scores: pd.DataFrame,
    attribution: pd.DataFrame,
    *,
    config: OpportunityConfig | None = None,
) -> pd.DataFrame:
    """Aggregate eligible patients and probability-derived opportunity by HCP."""

    resolved = config or OpportunityConfig()
    resolved.validate()
    _required_columns(
        patient_scores,
        [resolved.patient_col, resolved.score_col],
        "patient_scores",
    )
    _required_columns(attribution, ["patient_id", resolved.hcp_col], "attribution")
    if patient_scores[resolved.patient_col].duplicated().any():
        raise ValueError("patient_scores must contain exactly one row per patient.")
    if attribution["patient_id"].duplicated().any():
        raise ValueError("attribution must contain at most one HCP per patient.")

    patients = patient_scores[[resolved.patient_col, resolved.score_col]].copy()
    patients = patients.rename(columns={resolved.patient_col: "patient_id"})
    patients[resolved.score_col] = pd.to_numeric(patients[resolved.score_col], errors="coerce")
    scores = patients[resolved.score_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"{resolved.score_col} must contain finite probabilities within [0, 1].")
    patients["_patient_tie_key"] = patients["patient_id"].astype(str)
    ranked = patients.sort_values(
        [resolved.score_col, "_patient_tie_key"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    for fraction in resolved.top_fractions:
        k = min(len(ranked), max(1, int(np.ceil(len(ranked) * fraction))))
        selected_ids = set(ranked.iloc[:k]["patient_id"].tolist())
        patients[_top_flag_name(fraction)] = patients["patient_id"].isin(selected_ids).astype(int)
    if resolved.high_propensity_threshold_percentile is not None:
        high_threshold = float(np.percentile(scores, resolved.high_propensity_threshold_percentile))
    else:
        assert resolved.high_propensity_threshold is not None
        high_threshold = float(resolved.high_propensity_threshold)
    patients["_high_propensity"] = (patients[resolved.score_col] >= high_threshold).astype(int)

    mapping = attribution[["patient_id", resolved.hcp_col]].dropna().copy()
    if resolved.hcp_col != "hcp_id":
        mapping = mapping.rename(columns={resolved.hcp_col: "hcp_id"})
    merged = mapping.merge(patients, on="patient_id", how="inner", validate="one_to_one")
    top_columns = [_top_flag_name(value) for value in resolved.top_fractions]
    output_columns = [
        "hcp_id",
        "eligible_patient_count",
        "high_propensity_patient_count",
        *top_columns,
        "mean_patient_propensity",
        "max_patient_propensity",
        "expected_switchers",
    ]
    if merged.empty:
        return pd.DataFrame(columns=output_columns)

    named_aggregations: dict[str, tuple[str, str]] = {
        "eligible_patient_count": ("patient_id", "nunique"),
        "high_propensity_patient_count": ("_high_propensity", "sum"),
        "mean_patient_propensity": (resolved.score_col, "mean"),
        "max_patient_propensity": (resolved.score_col, "max"),
        "expected_switchers": (resolved.score_col, "sum"),
    }
    for column in top_columns:
        named_aggregations[column] = (column, "sum")
    aggregated = merged.groupby("hcp_id", dropna=False).agg(**named_aggregations).reset_index()
    count_columns = [
        "eligible_patient_count",
        "high_propensity_patient_count",
        *top_columns,
    ]
    aggregated[count_columns] = aggregated[count_columns].astype(int)
    aggregated["_hcp_tie_key"] = aggregated["hcp_id"].astype(str)
    return (
        aggregated[output_columns + ["_hcp_tie_key"]]
        .sort_values(
            ["expected_switchers", "_hcp_tie_key"],
            ascending=[False, True],
            kind="stable",
        )
        .drop(columns="_hcp_tie_key")
        .reset_index(drop=True)
    )


def _normalize_component(series: pd.Series, method: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"Score component '{series.name}' contains missing values.")
    if method == "none":
        return values.astype(float)
    if method == "minmax":
        minimum, maximum = float(values.min()), float(values.max())
        if np.isclose(minimum, maximum):
            return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
        return (values - minimum) / (maximum - minimum)
    # Average percentile rank makes ties transparent and stable.
    return values.rank(method="average", pct=True).astype(float)


def score_hcp_opportunities(
    hcp_metrics: pd.DataFrame,
    *,
    config: HCPScoringConfig | None = None,
) -> pd.DataFrame:
    """Apply a documented weighted formula and return score components and rank."""

    resolved = config or HCPScoringConfig()
    resolved.validate()
    _required_columns(hcp_metrics, ["hcp_id", *resolved.weights], "hcp_metrics")
    output = hcp_metrics.copy()
    total_weight = float(sum(resolved.weights.values()))
    formula_terms = []
    score = pd.Series(np.zeros(len(output)), index=output.index, dtype=float)
    for metric, raw_weight in resolved.weights.items():
        weight = float(raw_weight) / total_weight
        normalized = _normalize_component(output[metric], resolved.normalization)
        component_column = f"score_component__{metric}"
        output[component_column] = weight * normalized
        score += output[component_column]
        formula_terms.append(f"{weight:.6g}*{resolved.normalization}({metric})")
    output["hcp_opportunity_score"] = score
    output["hcp_rank"] = score.rank(method="min", ascending=False).astype(int)
    output["opportunity_score_formula"] = " + ".join(formula_terms)
    output["_hcp_tie_key"] = output["hcp_id"].astype(str)
    return (
        output.sort_values(["hcp_rank", "_hcp_tie_key"], ascending=[True, True], kind="stable")
        .drop(columns="_hcp_tie_key")
        .reset_index(drop=True)
    )


def build_hcp_targeting_output(
    patient_scores: pd.DataFrame,
    provider_events: pd.DataFrame,
    *,
    attribution_config: AttributionConfig | None = None,
    opportunity_config: OpportunityConfig | None = None,
    scoring_config: HCPScoringConfig | None = None,
    existing_attribution: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run attribution, aggregate opportunity, and score HCPs.

    Returns ``(hcp_targeting_output, patient_hcp_attribution)`` so attribution
    coverage and business-rule choices remain auditable.
    """

    attribution = attribute_patients_to_hcp(
        provider_events,
        config=attribution_config,
        existing_attribution=existing_attribution,
    )
    opportunity = calculate_hcp_opportunity_metrics(
        patient_scores, attribution, config=opportunity_config
    )
    targeting = score_hcp_opportunities(opportunity, config=scoring_config)
    return targeting, attribution


def save_hcp_targeting_output(
    frame: pd.DataFrame,
    path: str | Path = "outputs/hcp_targeting_output.csv",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


# Concise aliases for orchestration and backwards-compatible naming.
attribute_patients = attribute_patients_to_hcp
aggregate_hcp_opportunity = calculate_hcp_opportunity_metrics
prioritize_hcps = score_hcp_opportunities


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
