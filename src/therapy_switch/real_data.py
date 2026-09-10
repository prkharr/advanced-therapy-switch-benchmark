"""Explicit real-data configuration; no clinical defaults or synthetic fallback."""

from collections.abc import Mapping

import pandas as pd


def real_data_issues(config):
    if config.get("data", {}).get("kind") != "real":
        return []
    issues = []

    def get(path):
        value = config
        for key in path.split("."):
            value = value.get(key) if isinstance(value, Mapping) else None
        return value

    required = [
        "data.as_of_date", "data.extract_version", "cohort.cohort_id",
        "timeline.observation_window_days", "timeline.prediction_window_days",
        "timeline.minimum_history_days", "timeline.minimum_followup_days",
        "timeline.claims_lag_days", "timeline.label_runout_days",
        "timeline.index_date_start", "timeline.index_date_end",
        "cohort.require_diagnosis_confirmation", "cohort.diagnosis_separation_days",
        "cohort.coverage_window_days", "cohort.minimum_covered_days",
        "cohort.require_conventional_on_index",
        "source_definitions.cohort", "source_definitions.target",
        "source_definitions.availability", "source_definitions.status",
        "source_definitions.observation", "source_definitions.therapy_mapping",
        "source_definitions.diagnosis_normalization",
    ]
    for path in required:
        if get(path) is None or get(path) == "":
            issues.append(f"Set {path}")
    for path in ("data.as_of_date", "timeline.index_date_start", "timeline.index_date_end"):
        value = pd.to_datetime(get(path), errors="coerce")
        if get(path) is not None and (
            pd.isna(value) or value.tzinfo is not None or value != value.normalize()
        ):
            issues.append(f"Use a valid calendar date for {path}")
    for path in ("cohort.require_diagnosis_confirmation", "cohort.require_conventional_on_index"):
        if get(path) is not None and not isinstance(get(path), bool):
            issues.append(f"Use YAML true or false for {path}")
    for path in ("cohort.diagnosis_codes", "cohort.confirmation_codes",
                 "features.specialist_specialties"):
        if not get(path):
            issues.append(f"Populate {path} with actual source values")
    if get("data.source") != "files":
        issues.append("Real raw-folder workflow requires data.source: files")
    if get("data.require_export_manifest") is not True:
        issues.append("Real raw-folder workflow requires data.require_export_manifest: true")
    if "synthetic" in config.get("data", {}):
        issues.append("Remove data.synthetic from the real-data configuration")

    def check_tokens(value, path):
        if isinstance(value, Mapping):
            for key, child in value.items():
                check_tokens(child, f"{path}.{key}")
        elif isinstance(value, list):
            for child in value:
                check_tokens(child, path)
        elif isinstance(value, str) and (
            value.upper().startswith(("SYN_", "YOUR_", "REPLACE_")) or "{{" in value
        ):
            issues.append(f"Replace demonstration/placeholder value in {path}")

    for section in ("cohort", "therapy_mapping", "features", "source_definitions"):
        check_tokens(config.get(section, {}), section)
    start = pd.to_datetime(get("timeline.index_date_start"), errors="coerce")
    end = pd.to_datetime(get("timeline.index_date_end"), errors="coerce")
    if pd.notna(start) and pd.notna(end):
        if start > end or len(pd.date_range(start, end, freq="MS")) == 0:
            issues.append("Historical index calendar must include at least one month start")
    return list(dict.fromkeys(issues))


def validate_real_data(config):
    issues = real_data_issues(config)
    if issues:
        raise ValueError("Real-data setup is incomplete:\n- " + "\n- ".join(issues))
