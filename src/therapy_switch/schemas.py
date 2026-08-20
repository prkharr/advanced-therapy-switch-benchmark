"""Canonical table contracts and validation for claims adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class TableSchema:
    name: str
    required_columns: Sequence[str]
    date_columns: Sequence[str] = ()


CANONICAL_SCHEMAS: Dict[str, TableSchema] = {
    "patients": TableSchema(
        "patients",
        ("patient_id", "gender", "geography", "observation_start", "observation_end"),
        ("observation_start", "observation_end"),
    ),
    "medical_claims": TableSchema(
        "medical_claims",
        (
            "claim_id",
            "patient_id",
            "claim_date",
            "diagnosis_code",
            "procedure_code",
            "provider_id",
            "place_of_service",
        ),
        ("claim_date",),
    ),
    "pharmacy_claims": TableSchema(
        "pharmacy_claims",
        (
            "claim_id",
            "patient_id",
            "fill_date",
            "drug_id",
            "therapy_class",
            "quantity",
            "days_supply",
            "prescriber_id",
        ),
        ("fill_date",),
    ),
    "providers": TableSchema(
        "providers", ("provider_id", "specialty", "geography", "organization")
    ),
}


class SchemaError(ValueError):
    """Raised when an input table cannot satisfy the canonical contract."""


def validate_tables(
    tables: Mapping[str, pd.DataFrame], required: Iterable[str] | None = None
) -> None:
    """Validate canonical input names, required fields, identifiers, and dates."""

    required_names = tuple(required or CANONICAL_SCHEMAS.keys())
    for name in required_names:
        if name not in tables:
            raise SchemaError(f"Missing required table: {name}")
        frame = tables[name]
        if not isinstance(frame, pd.DataFrame):
            raise SchemaError(f"Table {name!r} must be a pandas DataFrame.")
        schema = CANONICAL_SCHEMAS[name]
        missing = sorted(set(schema.required_columns).difference(frame.columns))
        if missing:
            raise SchemaError(f"Table {name!r} is missing columns: {missing}")
        if "patient_id" in frame.columns and frame["patient_id"].isna().any():
            raise SchemaError(f"Table {name!r} contains null patient_id values.")
        for column in schema.date_columns:
            converted = pd.to_datetime(frame[column], errors="coerce")
            if converted.isna().any() and frame[column].notna().any():
                raise SchemaError(f"Table {name!r}.{column} contains invalid dates.")
