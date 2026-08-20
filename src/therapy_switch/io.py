"""I/O adapters for canonical CSV/Parquet claims tables.

This module is deliberately vendor-neutral. A Komodo delivery, enterprise data
mart, or another claims source can be adapted by configuring file names and
column renames; modeling code only sees canonical fields.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from therapy_switch.schemas import CANONICAL_SCHEMAS, validate_tables

DEFAULT_FILE_NAMES = {
    "patients": "patients",
    "medical_claims": "medical_claims",
    "pharmacy_claims": "pharmacy_claims",
    "providers": "providers",
}


def _read_frame(path: Path, file_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Claims input not found: {path}")
    if file_format == "csv":
        return pd.read_csv(path)
    if file_format == "parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet requires pyarrow or fastparquet. Install one or use CSV."
            ) from exc
    raise ValueError(f"Unsupported file_format={file_format!r}; expected 'csv' or 'parquet'.")


def load_claims_directory(config: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load and canonicalize the four claims input tables.

    Optional configuration::

        data:
          tables:
            patients:
              file: member_dimension
              columns: {member_token: patient_id}
    """

    data_config = config["data"]
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

    validate_tables(tables)
    return tables


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
