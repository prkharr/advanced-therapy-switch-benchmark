"""Configuration loading and validation utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


class ConfigError(ValueError):
    """Raised when a benchmark configuration is internally inconsistent."""


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Load YAML configuration, optionally deep-merging programmatic overrides."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError("Top-level YAML value must be a mapping.")
    if overrides:
        config = _deep_merge(config, overrides)
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def require_keys(mapping: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"Missing {context} configuration key(s): {', '.join(missing)}")


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail early on settings that could invalidate temporal evaluation."""

    require_keys(
        config,
        ["project", "data", "timeline", "therapy_mapping", "splitting", "models", "evaluation"],
        "top-level",
    )
    timeline = config["timeline"]
    require_keys(timeline, ["observation_window_days", "prediction_window_days"], "timeline")
    if int(timeline["observation_window_days"]) <= 0:
        raise ConfigError("observation_window_days must be positive.")
    if int(timeline["prediction_window_days"]) <= 0:
        raise ConfigError("prediction_window_days must be positive.")

    mappings = config["therapy_mapping"]
    require_keys(mappings, ["conventional", "advanced"], "therapy_mapping")
    conventional = set(mappings["conventional"])
    advanced = set(mappings["advanced"])
    overlap = conventional.intersection(advanced)
    if overlap:
        raise ConfigError(f"Conventional and advanced therapy mappings overlap: {sorted(overlap)}")
    if not conventional or not advanced:
        raise ConfigError("Both conventional and advanced therapy mappings must be non-empty.")

    split = config["splitting"]
    require_keys(split, ["validation_fraction", "test_fraction"], "splitting")
    validation_fraction = float(split["validation_fraction"])
    test_fraction = float(split["test_fraction"])
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ConfigError("validation_fraction and test_fraction must be > 0 and sum to < 1.")

    top_fractions = config["evaluation"].get("top_fractions", [])
    if not top_fractions or any(float(value) <= 0 or float(value) > 1 for value in top_fractions):
        raise ConfigError("evaluation.top_fractions must contain values in (0, 1].")
