"""Normalize source frames into canonical raw claims, with explicit mappings."""

import pandas as pd

from therapy_switch.schemas import CANONICAL_SCHEMAS, SchemaError, validate_tables


def canonicalize_raw_tables(tables, config):
    result = {}
    mappings = config.get("data", {}).get("tables", {})
    for name, schema in CANONICAL_SCHEMAS.items():
        if name not in tables:
            raise SchemaError(f"Missing required table: {name}")
        frame = tables[name].rename(columns=mappings.get(name, {}).get("columns", {})).copy()
        # Source mappings are explicit. Canonical uppercase aliases are accepted.
        frame.columns = [str(c).lower() for c in frame.columns]
        if not frame.columns.is_unique:
            raise SchemaError(f"{name}: duplicate columns after renaming")
        for col in (*schema.date_columns, "reversal_date"):
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], errors="raise")
        if "status" in frame:
            frame["status"] = frame.status.astype(str).str.lower()
        for col in ("birth_year", "quantity", "days_supply", "patient_cost"):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="raise")
        result[name] = frame
    validate_tables(result)
    mapping = result["therapy_mapping"]
    for _, group in mapping.groupby("drug_id"):
        ordered = group.sort_values("effective_start")
        if (
            ordered.effective_start.iloc[1:].reset_index(drop=True)
            <= ordered.effective_end.iloc[:-1].reset_index(drop=True)
        ).any():
            raise SchemaError("Overlapping therapy mapping effective intervals")
    pharmacy = result["pharmacy_claims"]
    tagged = pharmacy.drop(columns="therapy_class").merge(mapping, on="drug_id", how="left")
    tagged = tagged.loc[
        tagged.fill_date.ge(tagged.effective_start) & tagged.fill_date.le(tagged.effective_end)
    ]
    if len(tagged) != len(pharmacy) or tagged.claim_id.duplicated().any():
        raise SchemaError("Each pharmacy claim requires exactly one effective therapy mapping")
    expected = pharmacy.set_index("claim_id").therapy_class
    if not tagged.therapy_class.eq(tagged.claim_id.map(expected)).all():
        raise SchemaError("Pharmacy therapy class conflicts with effective mapping")
    result["pharmacy_claims"] = (
        tagged[pharmacy.columns]
        .sort_values(["patient_id", "fill_date", "claim_id"])
        .reset_index(drop=True)
    )
    if "snapshot_candidates" in tables:
        result["snapshot_candidates"] = tables["snapshot_candidates"].copy()
        result["snapshot_candidates"].columns = [
            str(c).lower() for c in result["snapshot_candidates"]
        ]
    return result


def prepare_raw_inputs(tables, config):
    from therapy_switch.features import build_tabular_features

    from .cohort import build_cohort
    from .event_sequences import build_event_frame
    from .prepared_input_adapter import PreparedInputs, validate_prepared_inputs

    canonical = canonicalize_raw_tables(tables, config)
    snapshots = build_cohort(canonical, config)
    wide = build_tabular_features(canonical, snapshots, config)
    events = build_event_frame(canonical, snapshots, config)
    inputs = PreparedInputs(
        snapshots, wide, events, tuple(c for c in wide if c not in {"snapshot_id", "feature_as_of"})
    )
    validate_prepared_inputs(inputs, config, require_lineage=False)
    return canonical, inputs
