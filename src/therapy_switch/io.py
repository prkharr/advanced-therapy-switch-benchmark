"""I/O adapters for canonical CSV/Parquet claims tables.

This module is deliberately vendor-neutral. A Komodo delivery, enterprise data
mart, or another claims source can be adapted by configuring file names and
column renames; modeling code only sees canonical fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from therapy_switch.schemas import CANONICAL_SCHEMAS, validate_tables

DEFAULT_FILE_NAMES = {name: name for name in CANONICAL_SCHEMAS}


def _read_frame(path: Path, file_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Claims input not found: {path}")
    if file_format == "csv":
        return pd.read_csv(path, dtype=str)
    if file_format == "parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet requires pyarrow or fastparquet. Install one or use CSV."
            ) from exc
    raise ValueError(f"Unsupported file_format={file_format!r}; expected 'csv' or 'parquet'.")


def verify_export_manifest(config):
    """Reject failed exports and verify real extract identity and file hashes."""
    data = config["data"]
    directory = Path(data["input_dir"])
    path = directory / "export_manifest.json"
    required = data.get("require_export_manifest", False)
    if not path.exists():
        if required:
            raise ValueError("A completed export_manifest.json is required in the raw folder")
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError("Raw export is incomplete or failed; use a completed extract")
    if required:
        if manifest.get("data_kind") != "real":
            raise ValueError("Real-data mode requires an export identified as real")
        if str(manifest.get("extract_as_of_date")) != str(data["as_of_date"]):
            raise ValueError("Raw export as-of date does not match the configuration")
        if manifest.get("extract_version") != data.get("extract_version"):
            raise ValueError("Raw export version does not match the configuration")
        if data.get("tables"):
            raise ValueError("Verified raw exports use canonical filenames and columns")
        for name in CANONICAL_SCHEMAS:
            digest = hashlib.sha256()
            with (directory / f"{name}.csv").open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != manifest.get("tables", {}).get(name, {}).get("sha256"):
                raise ValueError(f"Raw file {name}.csv changed after export; re-extract")
    return manifest


def load_claims_directory(config: Mapping[str, Any], *, check_export=True) -> Dict[str, pd.DataFrame]:
    """Load and canonicalize the seven claims input tables.

    Optional configuration::

        data:
          tables:
            patients:
              file: member_dimension
              columns: {member_token: patient_id}
    """

    data_config = config["data"]
    if check_export:
        verify_export_manifest(config)
    input_dir = Path(data_config["input_dir"])
    file_format = str(data_config.get("file_format", "parquet")).lower()
    extension = ".csv" if file_format == "csv" else ".parquet"
    table_config = data_config.get("tables", {})
    tables: Dict[str, pd.DataFrame] = {}

    for canonical_name, default_stem in DEFAULT_FILE_NAMES.items():
        settings = table_config.get(canonical_name, {})
        configured_file = str(settings.get("file", default_stem))
        candidate = Path(configured_file)
        if not candidate.suffix:
            candidate = candidate.with_suffix(extension)
        if not candidate.is_absolute():
            candidate = input_dir / candidate
        frame = _read_frame(candidate, file_format)
        rename_map = settings.get("columns", {})
        if rename_map:
            frame = frame.rename(columns=rename_map)
        for date_column in CANONICAL_SCHEMAS[canonical_name].date_columns:
            if date_column in frame:
                frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
        tables[canonical_name] = frame

    from therapy_switch.data.raw_source_adapter import canonicalize_raw_tables

    return canonicalize_raw_tables(tables, config)


def save_claims_directory(
    tables: Mapping[str, pd.DataFrame], directory: str | Path, file_format: str = "csv"
) -> None:
    """Persist canonical tables for development only.

    Production claims should remain in governed storage; this helper is mainly
    intended for synthetic datasets and integration fixtures.
    """

    validate_tables(tables)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if name not in CANONICAL_SCHEMAS:
            continue
        if file_format == "csv":
            frame.to_csv(target / f"{name}.csv", index=False)
        elif file_format == "parquet":
            try:
                frame.to_parquet(target / f"{name}.parquet", index=False)
            except ImportError as exc:
                raise RuntimeError("Writing Parquet requires pyarrow or fastparquet.") from exc
        else:
            raise ValueError("file_format must be 'csv' or 'parquet'.")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(payload: Any, path: str | Path) -> None:
    """Atomically write stable, human-readable JSON metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
