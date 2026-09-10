"""Export source column metadata (no patient rows) to private local CSV."""

import argparse
import re
from pathlib import Path

SOURCE_FAMILIES = (
    "MEDICAL_EVENTS_LATEST", "PHARMACY_EVENTS_LATEST", "PATIENT_DEMOGRAPHICS_LATEST",
    "PATIENT_ENROLLMENT_LATEST", "PATIENT_GEOGRAPHY_LATEST", "PLANS_LATEST", "PROVIDERS_LATEST",
)


def inspect_sources(session, database, schema, output_dir):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", database):
        raise ValueError("Use one unquoted database identifier")
    placeholders = ", ".join("?" for _ in SOURCE_FAMILIES)
    query = f"""SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, ORDINAL_POSITION
    FROM {database}.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME IN ({placeholders})
    ORDER BY TABLE_NAME, ORDINAL_POSITION"""
    frame = session.sql(query, params=[schema.upper(), *SOURCE_FAMILIES]).to_pandas()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "source_columns.csv"
    frame.to_csv(path, index=False)
    found = set(frame["TABLE_NAME"])
    print(f"Saved {len(frame)} column definitions to {path.resolve()}")
    print("Not found or not visible: " + ", ".join(sorted(set(SOURCE_FAMILIES) - found)))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection-name", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output-dir", default="configs/private/discovery")
    args = parser.parse_args()
    from snowflake.snowpark import Session

    session = Session.builder.config("connection_name", args.connection_name).create()
    try:
        inspect_sources(session, args.database, args.schema, args.output_dir)
    finally:
        session.close()


if __name__ == "__main__":
    main()
