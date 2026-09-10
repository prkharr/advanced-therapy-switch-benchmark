"""Stable interfaces between extraction, modeling and HCP delivery."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module

import numpy as np
import pandas as pd

from therapy_switch.features.history import eligible_events
from therapy_switch.features.leakage import validate_predictor_names


@dataclass
class ScoringBatch:
    snapshots: pd.DataFrame
    wide: pd.DataFrame
    events: pd.DataFrame
    providers: pd.DataFrame
    feature_columns: tuple[str, ...]

    def modeling_frame(self):
        return self.snapshots.merge(
            self.wide.drop(columns="feature_as_of"),
            on="snapshot_id",
            validate="one_to_one",
            how="left",
        )

    def validate(self, scoring_date):
        snapshots = self.snapshots
        required = {
            "patient_id",
            "snapshot_id",
            "index_date",
            "feature_cutoff",
            "lookback_start",
            "eligible",
        }
        if not required.issubset(snapshots):
            raise ValueError("Scoring snapshots are missing required keys or timing fields")
        if {"label", "resp", "outcome_date", "label_available_date"} & (
            set(snapshots) | set(self.events)
        ):
            raise ValueError("Live scoring snapshots must not contain outcome fields")
        if (
            snapshots[["patient_id", "snapshot_id"]].isna().any().any()
            or snapshots.patient_id.duplicated().any()
            or snapshots.snapshot_id.duplicated().any()
        ):
            raise ValueError("A scoring batch must contain one unique assessment per patient")
        dates = snapshots[["index_date", "feature_cutoff", "lookback_start"]].apply(pd.to_datetime)
        if dates.isna().any().any() or not dates.index_date.eq(pd.Timestamp(scoring_date)).all():
            raise ValueError("Every scoring assessment must use the declared scoring date")
        if (dates.feature_cutoff > dates.index_date).any() or (
            dates.lookback_start > dates.feature_cutoff
        ).any():
            raise ValueError("Invalid scoring observation window")
        if not snapshots.eligible.eq(True).all():
            raise ValueError("Only currently eligible patients may enter the scoring batch")
        self.snapshots = snapshots = snapshots.assign(**{key: dates[key] for key in dates})
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Scoring feature allowlist must be nonempty and unique")
        validate_predictor_names(self.feature_columns)
        if set(self.wide) != {"snapshot_id", "feature_as_of", *self.feature_columns}:
            raise ValueError("Scoring wide schema differs from its predictor allowlist")
        if self.wide.snapshot_id.duplicated().any() or set(self.wide.snapshot_id) != set(
            snapshots.snapshot_id
        ):
            raise ValueError("Wide features must align with every scoring assessment")
        available = pd.to_datetime(self.wide.feature_as_of)
        if available.isna().any() or (available > pd.Timestamp(scoring_date)).any():
            raise ValueError("Wide features are unavailable at scoring date")
        if any(pd.api.types.is_datetime64_any_dtype(self.wide[c]) for c in self.feature_columns):
            raise ValueError("Raw datetime predictors are not allowed")
        if self.wide.select_dtypes("number").isin([np.inf, -np.inf]).any().any():
            raise ValueError("Scoring features contain infinity")
        event_fields = {
            "snapshot_id",
            "patient_id",
            "event_id",
            "event_date",
            "available_date",
            "event_type",
            "hcp_id",
            "provider_specialty",
            "therapy_class",
            "status",
            "code_token",
        }
        if not event_fields.issubset(self.events):
            raise ValueError("Scoring events are missing history or HCP attribution fields")
        if self.events[["snapshot_id", "patient_id", "event_id"]].isna().any().any():
            raise ValueError("Scoring event keys must be non-null")
        checked = eligible_events(self.events, snapshots)
        if (
            pd.to_datetime(self.events.available_date) < pd.to_datetime(self.events.event_date)
        ).any():
            raise ValueError("Event availability precedes its service date")
        if len(checked) != len(self.events):
            raise ValueError("Scoring events contain unmatched assessments")
        if self.events.duplicated(["snapshot_id", "event_id"]).any():
            raise ValueError("Duplicate events within a scoring assessment")
        joined = self.events[["snapshot_id", "patient_id"]].merge(
            snapshots[["snapshot_id", "patient_id"]], on="snapshot_id", suffixes=("", "_expected")
        )
        if not joined.patient_id.eq(joined.patient_id_expected).all():
            raise ValueError("Scoring event belongs to a different patient")
        if not {"provider_id", "specialty", "geography", "organization"}.issubset(self.providers):
            raise ValueError("Provider directory is missing canonical fields")
        if self.providers.provider_id.isna().any() or self.providers.provider_id.duplicated().any():
            raise ValueError("Provider directory keys must be non-null and unique")
        return self


@dataclass
class DeliveryInputs:
    scoring: ScoringBatch
    training: object = None
    provenance: dict = field(default_factory=dict)


def resolve_callable(reference):
    """Load an explicitly configured, trusted installed Python plugin."""
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("A plugin must be an installed module:function reference")
    module, name = reference.split(":")
    value = getattr(import_module(module), name)
    if not callable(value):
        raise ValueError("Configured plugin is not callable")
    return value
