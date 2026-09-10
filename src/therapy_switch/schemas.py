"""Canonical raw-table validation. All dates are timezone-naive calendar dates."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TableSchema:
    name: str
    required_columns: tuple
    date_columns: tuple = ()


CANONICAL_SCHEMAS = {
    "patients": TableSchema(
        "patients",
        ("patient_id", "birth_year", "gender", "geography", "observation_start", "observation_end"),
        ("observation_start", "observation_end"),
    ),
    "medical_claims": TableSchema(
        "medical_claims",
        (
            "claim_id",
            "patient_id",
            "claim_date",
            "available_date",
            "status",
            "diagnosis_code",
            "procedure_code",
            "provider_id",
            "place_of_service",
        ),
        ("claim_date", "available_date"),
    ),
    "pharmacy_claims": TableSchema(
        "pharmacy_claims",
        (
            "claim_id",
            "patient_id",
            "fill_date",
            "available_date",
            "status",
            "drug_id",
            "therapy_class",
            "quantity",
            "days_supply",
            "prescriber_id",
            "plan_id",
            "patient_cost",
        ),
        ("fill_date", "available_date"),
    ),
    "providers": TableSchema(
        "providers", ("provider_id", "specialty", "geography", "organization")
    ),
    "plans": TableSchema("plans", ("plan_id", "payer_type")),
    "enrollment": TableSchema(
        "enrollment",
        ("patient_id", "coverage_start", "coverage_end", "plan_id"),
        ("coverage_start", "coverage_end"),
    ),
    "therapy_mapping": TableSchema(
        "therapy_mapping",
        ("drug_id", "therapy_class", "effective_start", "effective_end"),
        ("effective_start", "effective_end"),
    ),
}


class SchemaError(ValueError):
    pass


def validate_tables(tables, required=None):
    for name in required or CANONICAL_SCHEMAS:
        if name not in tables:
            raise SchemaError(f"Missing required table: {name}")
        frame, schema = tables[name], CANONICAL_SCHEMAS[name]
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise SchemaError(f"{name} must be a dataframe with unique columns")
        missing = set(schema.required_columns) - set(frame.columns)
        if missing:
            raise SchemaError(f"{name} missing columns: {sorted(missing)}")
        keys = {
            "patients": "patient_id",
            "providers": "provider_id",
            "plans": "plan_id",
            "medical_claims": "claim_id",
            "pharmacy_claims": "claim_id",
        }
        key = keys.get(name)
        if key and (frame[key].isna().any() or frame[key].duplicated().any()):
            raise SchemaError(
                f"{name}.{key} must be non-null and unique; canonicalize duplicate source lines"
            )
        if "patient_id" in frame and frame.patient_id.isna().any():
            raise SchemaError(f"{name}: null patient_id")
        for col in schema.date_columns:
            dates = pd.to_datetime(frame[col], errors="coerce")
            if dates.isna().any():
                raise SchemaError(f"{name}.{col} contains null or invalid dates")
            if getattr(dates.dt, "tz", None) is not None:
                raise SchemaError("Normalize source dates to timezone-naive calendar dates")
        if (
            "status" in frame
            and not frame.status.isin(["paid", "final", "rejected", "reversed"]).all()
        ):
            raise SchemaError(f"{name}: unsupported status")
        date_col = "claim_date" if name == "medical_claims" else "fill_date"
        if "available_date" in frame:
            if (pd.to_datetime(frame.available_date) < pd.to_datetime(frame[date_col])).any():
                raise SchemaError(f"{name}: availability precedes service")
            if "reversal_date" in frame:
                reversed_at = pd.to_datetime(frame.reversal_date, errors="raise")
                if (reversed_at < pd.to_datetime(frame.available_date)).any():
                    raise SchemaError(f"{name}: reversal precedes original availability")
        if name == "pharmacy_claims":
            for col in ("days_supply", "quantity", "patient_cost"):
                values = pd.to_numeric(frame[col], errors="coerce")
                if values.isna().any() or (values < 0).any():
                    raise SchemaError(f"{name}.{col}: nonnegative values required")
            if (frame.days_supply % 1 != 0).any():
                raise SchemaError("days_supply must be whole days")
        for begin, end in [
            ("observation_start", "observation_end"),
            ("coverage_start", "coverage_end"),
            ("effective_start", "effective_end"),
        ]:
            if begin in frame and (pd.to_datetime(frame[begin]) > pd.to_datetime(frame[end])).any():
                raise SchemaError(f"{name}: inverted date interval")
    if required is None:
        pids = set(tables["patients"].patient_id)
        for name in ("medical_claims", "pharmacy_claims", "enrollment"):
            if not set(tables[name].patient_id).issubset(pids):
                raise SchemaError(f"{name}: unknown patient")
        providers, plans = set(tables["providers"].provider_id), set(tables["plans"].plan_id)
        for name, col in (("medical_claims", "provider_id"), ("pharmacy_claims", "prescriber_id")):
            if not set(tables[name][col].dropna()).issubset(providers):
                raise SchemaError(f"{name}: unknown provider")
        for name in ("pharmacy_claims", "enrollment"):
            if not set(tables[name].plan_id.dropna()).issubset(plans):
                raise SchemaError(f"{name}: unknown plan")
