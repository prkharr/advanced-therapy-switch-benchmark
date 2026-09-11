"""Save reproducible aggregate evidence from the actual input files."""

from pathlib import Path

from therapy_switch.io import write_json
from therapy_switch.schemas import CANONICAL_SCHEMAS


def raw_profile(tables):
    result = {}
    for name, frame in tables.items():
        if name not in CANONICAL_SCHEMAS:
            continue
        row = {
            "rows": len(frame),
            "nulls": {column: int(value) for column, value in frame.isna().sum().items()},
            "date_ranges": {},
        }
        if "patient_id" in frame:
            row["distinct_patients"] = int(frame.patient_id.nunique())
        if "status" in frame:
            row["status_counts"] = frame.status.value_counts().to_dict()
        for column in CANONICAL_SCHEMAS[name].date_columns:
            row["date_ranges"][column] = {
                "min": str(frame[column].min()), "max": str(frame[column].max()),
            }
        result[name] = row
    return result


def save_profile(tables, output_dir):
    path = Path(output_dir) / "raw_profile.json"
    write_json(raw_profile(tables), path)
    return path
