"""Point-in-time claims selection and interval-union coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def known_claims(frame, date_column, index_date, lag_days=0, start_date=None):
    """Select service and availability cutoffs independently, without mutating input.

    A reversal changes exposure validity only when its availability date has passed.
    The adapter requires immutable claim lines and a separate reversal timestamp.
    """
    index = pd.Timestamp(index_date)
    cutoff = index - pd.Timedelta(days=int(lag_days))
    dates = pd.to_datetime(frame[date_column], errors="raise")
    available = pd.to_datetime(frame["available_date"], errors="raise")
    selected = (dates <= cutoff) & (available <= index)
    if start_date is not None:
        selected &= dates >= pd.Timestamp(start_date)
    out = frame.loc[selected].copy()
    if "reversal_date" in out:
        reversed_by_now = pd.to_datetime(out["reversal_date"]).le(index)
        out.loc[reversed_by_now, "status"] = "reversed"
    return out


def valid_exposures(frame):
    return frame.loc[frame["status"].isin(["paid", "final"])].copy()


def coverage_days(frame, start_date, end_date):
    """Covered day union, inclusive calendar interval, no stockpiling assumption."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    covered = np.zeros(max(0, (end - start).days + 1), dtype=bool)
    for row in frame.itertuples(index=False):
        supply = max(0, int(row.days_supply))
        left = max(0, (pd.Timestamp(row.fill_date) - start).days)
        right = min(len(covered), (pd.Timestamp(row.fill_date) - start).days + supply)
        if right > left:
            covered[left:right] = True
    return covered


def maximum_gap(covered):
    padded = np.r_[True, covered, True]
    edges = np.flatnonzero(padded)
    return int(np.diff(edges).max() - 1) if len(edges) > 1 else 0
