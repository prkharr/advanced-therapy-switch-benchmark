"""Data generation, cohorting, splitting, and sequence construction APIs."""

from .cohort import build_cohort, validate_cohort_timeline
from .event_sequences import (
    EventSequenceDataset,
    build_event_frame,
    build_event_sequences,
    fit_sequence_vocabularies,
)
from .generate_synthetic_claims import generate_synthetic_claims
from .splitting import (
    stratified_patient_split,
    stratified_split,
    temporal_patient_split,
    temporal_split,
    validate_patient_disjoint,
)

__all__ = [
    "EventSequenceDataset",
    "build_cohort",
    "build_event_frame",
    "build_event_sequences",
    "fit_sequence_vocabularies",
    "generate_synthetic_claims",
    "stratified_patient_split",
    "stratified_split",
    "temporal_patient_split",
    "temporal_split",
    "validate_cohort_timeline",
    "validate_patient_disjoint",
]
