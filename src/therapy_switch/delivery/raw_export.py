"""Export approved Snowflake SELECTs to a validated, versioned raw CSV folder."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from therapy_switch.io import load_claims_directory, write_json
from therapy_switch.schemas import CANONICAL_SCHEMAS


def read_export_queries(sql_dir):
    """Check every query offline before opening a connection or writing files."""
    queries = {}
    for name in CANONICAL_SCHEMAS:
        query = (Path(sql_dir) / f"{name}.sql").read_text(encoding="utf-8")
        clean = re.sub(r"/\*.*?\*/|--[^\n]*", "", query, flags=re.S).strip().rstrip(";")
        if not re.match(r"^(SELECT|WITH)\b", clean, re.I):
            raise ValueError(f"{name}.sql must contain one reviewed SELECT query")
        if ";" in clean or "{{" in clean or re.search(r"\b(YOUR_|REPLACE_)", clean, re.I):
            raise ValueError(f"{name}.sql contains unresolved placeholders or multiple statements")
        queries[name] = clean
    return queries


def export_raw_data(session, config, sql_dir, output_dir="data/raw", *, max_rows=1_000_000):
    """Export consistently versioned canonical SELECTs, then validate all seven files."""
    from therapy_switch.real_data import validate_real_data

    validate_real_data(config)
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    queries = read_export_queries(sql_dir)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    directory = Path(output_dir).resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "RUNNING", "run_id": run_id, "source": "snowflake_selects",
        "extract_as_of_date": str(config["data"]["as_of_date"]), "tables": {},
        "data_kind": config["data"].get("kind", "unverified"),
        "extract_version": config["data"].get("extract_version"),
        "clinical_logic_review": "Requires upstream review; not established by export",
    }
    manifest_path = directory / "export_manifest.json"
    write_json(manifest, manifest_path)
    try:
        for name, query in queries.items():
            frame = session.sql(query)
            columns = [str(c).lower() for c in frame.columns]
            missing = set(CANONICAL_SCHEMAS[name].required_columns) - set(columns)
            if missing or len(columns) != len(set(columns)):
                raise ValueError(f"{name}: missing canonical columns {sorted(missing)} or duplicates")
            temporary = directory / f"{name}.csv.partial"
            count = 0
            with temporary.open("w", encoding="utf-8", newline="") as output:
                csv.writer(output).writerow(columns)
                # The extra row detects oversize extracts; it is never silently retained.
                for batch in frame.limit(max_rows + 1).to_pandas_batches():
                    count += len(batch)
                    if count > max_rows:
                        raise ValueError(f"{name} exceeds the {max_rows:,}-row local export limit")
                    if [str(c).lower() for c in batch.columns] != columns:
                        raise ValueError(f"{name}: returned columns differ from query schema")
                    batch.to_csv(output, index=False, header=False)
            path = temporary.with_suffix("")
            temporary.rename(path)
            digest = hashlib.sha256()
            with path.open("rb") as raw:
                for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest["tables"][name] = {
                "rows": count, "sha256": digest.hexdigest(),
                "sql_sha256": hashlib.sha256(query.encode()).hexdigest(),
            }
            write_json(manifest, manifest_path)

        validation = copy.deepcopy(config)
        validation["data"].update(source="files", input_dir=str(directory), file_format="csv")
        load_claims_directory(validation, check_export=False)
        manifest.update(status="COMPLETED", raw_dir=str(directory), contract_validation="passed")
        write_json(manifest, manifest_path)
        return {"raw_dir": str(directory), "manifest": str(manifest_path), "status": "COMPLETED"}
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write_json(manifest, manifest_path)
        raise


def main(argv=None):
    from therapy_switch.delivery.pipeline import load_delivery_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Reviewed delivery YAML")
    parser.add_argument("--sql-dir", required=True, help="Directory of seven reviewed SELECT files")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--connection-name", required=True, help="Existing approved named connection")
    args = parser.parse_args(argv)
    from snowflake.snowpark import Session

    config = load_delivery_config(args.config)
    read_export_queries(args.sql_dir)
    session = Session.builder.config("connection_name", args.connection_name).create()
    try:
        result = export_raw_data(
            session, config, args.sql_dir, args.output_dir, max_rows=args.max_rows
        )
        print(json.dumps(result, indent=2))
    finally:
        session.close()
    return 0
