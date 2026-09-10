"""Training-vocabulary claims history counts and recency features."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from therapy_switch.models.contracts import LeakageError


def eligible_events(events, snapshots):
    required = {
        "snapshot_id",
        "event_id",
        "event_date",
        "available_date",
        "event_type",
        "code_token",
    }
    if not required.issubset(events):
        raise ValueError(f"Missing history columns: {sorted(required - set(events))}")
    dates = snapshots[["snapshot_id", "index_date", "feature_cutoff", "lookback_start"]]
    if dates.snapshot_id.duplicated().any():
        raise ValueError("Duplicate snapshot keys")
    matched = events.loc[events.snapshot_id.isin(dates.snapshot_id)].copy()
    matched = matched.drop(
        columns=["index_date", "feature_cutoff", "lookback_start"], errors="ignore"
    ).merge(dates, on="snapshot_id", validate="many_to_one")
    for key in ("event_date", "available_date", "index_date", "feature_cutoff", "lookback_start"):
        matched[key] = pd.to_datetime(matched[key], errors="raise")
    if (
        matched[["event_date", "available_date", "index_date", "feature_cutoff", "lookback_start"]]
        .isna()
        .any()
        .any()
    ):
        raise LeakageError("Missing history timing")
    if (
        (matched.event_date > matched.feature_cutoff).any()
        or (matched.available_date > matched.index_date).any()
        or (matched.event_date < matched.lookback_start).any()
    ):
        raise LeakageError("History contains an unavailable or out-of-window event")
    matched["days_before_index"] = (matched.index_date - matched.event_date).dt.days
    matched["history_token"] = (
        matched.event_type.astype(str) + "::" + matched.code_token.astype(str)
    )
    return matched.sort_values(["snapshot_id", "event_date", "event_id"], kind="stable")


class HistoryFeatureEncoder(BaseEstimator):
    """Fixed training-derived vocabulary; no outcome fields are consumed."""

    def __init__(self, max_codes=24, windows=(14, 30, 90, 365)):
        self.max_codes = max_codes
        self.windows = windows

    def fit(self, events, snapshots):
        data = eligible_events(events, snapshots)
        counts = Counter(data.history_token)
        self.vocabulary_ = [
            code
            for code, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
                : self.max_codes
            ]
        ]
        return self

    def transform(self, events, snapshots):
        data = eligible_events(events, snapshots)
        index = pd.Index(snapshots.snapshot_id)
        values = {}
        for i, code in enumerate(self.vocabulary_):
            part = data.loc[data.history_token.eq(code)]
            for window in self.windows:
                counts = part.loc[part.days_before_index <= window].groupby("snapshot_id").size()
                values[f"history_code_{i}_count_{window}d"] = np.log1p(
                    counts.reindex(index, fill_value=0).to_numpy()
                )
            recent = part.groupby("snapshot_id").days_before_index.min()
            values[f"history_code_{i}_recency"] = recent.reindex(index, fill_value=366).to_numpy()
        for window in self.windows:
            part = data.loc[data.days_before_index <= window]
            visits = part.groupby("snapshot_id").event_date.nunique()
            codes = part.groupby("snapshot_id").history_token.nunique()
            values[f"history_visit_dates_{window}d"] = np.log1p(
                visits.reindex(index, fill_value=0).to_numpy()
            )
            values[f"history_code_diversity_{window}d"] = codes.reindex(
                index, fill_value=0
            ).to_numpy()
        # Gaps between observed visit dates, available at the snapshot.
        visits = data[["snapshot_id", "event_date"]].drop_duplicates()
        visits["gap"] = visits.groupby("snapshot_id").event_date.diff().dt.days
        grouped = visits.groupby("snapshot_id").gap
        values["history_visit_gap_mean"] = grouped.mean().reindex(index).fillna(366).to_numpy()
        values["history_visit_gap_std"] = grouped.std().reindex(index).fillna(0).to_numpy()
        values["history_visit_gap_last"] = grouped.last().reindex(index).fillna(366).to_numpy()
        return pd.DataFrame(values, index=snapshots.index, dtype=float)

    def fit_transform(self, events, snapshots):
        return self.fit(events, snapshots).transform(events, snapshots)
