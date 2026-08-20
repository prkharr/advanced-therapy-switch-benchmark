"""Leakage-safe feature construction and auditing."""

from .feature_engineering import build_features, build_tabular_features
from .leakage import (
    LeakageError,
    assert_no_feature_leakage,
    assert_no_post_index_events,
    audit_feature_frame,
    check_temporal_leakage,
    find_post_index_events,
)

__all__ = [
    "LeakageError",
    "assert_no_feature_leakage",
    "assert_no_post_index_events",
    "audit_feature_frame",
    "build_features",
    "build_tabular_features",
    "check_temporal_leakage",
    "find_post_index_events",
]
