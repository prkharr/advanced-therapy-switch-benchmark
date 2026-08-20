"""Small configuration helpers shared by the data pipeline.

The project deliberately does not require a particular configuration framework.
Plain dictionaries, dataclasses, and objects such as ``SimpleNamespace`` are all
accepted by public functions in :mod:`therapy_switch.data`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_MISSING = object()


def config_value(config: Any, *paths: str, default: Any = None) -> Any:
    """Return the first configured value found at one of ``paths``.

    A path uses dot notation, for example ``"timeline.observation_window_days"``.
    Mapping keys take precedence over object attributes at each level.
    """

    for path in paths:
        current = config
        found = True
        for part in path.split("."):
            if current is None:
                found = False
                break
            if isinstance(current, Mapping):
                current = current.get(part, _MISSING)
            else:
                current = getattr(current, part, _MISSING)
            if current is _MISSING:
                found = False
                break
        if found:
            return current
    return default


def as_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a scalar or iterable configuration value to strings."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


@dataclass(frozen=True)
class TherapyDefinition:
    """Non-proprietary, configuration-driven therapy identifiers."""

    conventional_drug_ids: tuple[str, ...]
    advanced_drug_ids: tuple[str, ...]
    conventional_classes: tuple[str, ...]
    advanced_classes: tuple[str, ...]

    def conventional_mask(self, frame: Any) -> Any:
        """Return a boolean mask selecting conventional-therapy rows."""

        mask = False
        if "drug_id" in frame:
            mask = frame["drug_id"].astype("string").isin(self.conventional_drug_ids)
        if "therapy_class" in frame:
            class_mask = frame["therapy_class"].astype("string").isin(self.conventional_classes)
            mask = class_mask if isinstance(mask, bool) else mask | class_mask
        return mask

    def advanced_mask(self, frame: Any) -> Any:
        """Return a boolean mask selecting advanced-therapy rows."""

        mask = False
        if "drug_id" in frame:
            mask = frame["drug_id"].astype("string").isin(self.advanced_drug_ids)
        if "therapy_class" in frame:
            class_mask = frame["therapy_class"].astype("string").isin(self.advanced_classes)
            mask = class_mask if isinstance(mask, bool) else mask | class_mask
        return mask


def _mapping_section(config: Any, arm: str) -> Any:
    return config_value(
        config,
        f"therapy_mapping.{arm}",
        f"therapy_mappings.{arm}",
        f"therapies.{arm}",
        default=None,
    )


def therapy_definition(config: Any) -> TherapyDefinition:
    """Resolve conventional and advanced therapy identifiers from configuration.

    Supported examples include::

        therapy_mapping:
          conventional: [CONV_A, CONV_B]  # interpreted as drug identifiers
          advanced: [ADV_A]

    and the more explicit::

        therapy_mapping:
          conventional:
            drug_ids: [CONV_A]
            therapy_classes: [conventional]
          advanced:
            drug_ids: [ADV_A]
            therapy_classes: [advanced]

    Generic synthetic defaults are provided so the example pipeline can run from
    scratch. Production mappings should always be supplied by configuration.
    """

    def resolve(arm: str, default_drugs: tuple[str, ...], default_class: str) -> tuple:
        section = _mapping_section(config, arm)
        if isinstance(section, Mapping):
            drug_ids = as_tuple(section.get("drug_ids", section.get("drugs")))
            classes = as_tuple(
                section.get("therapy_classes", section.get("classes", section.get("class")))
            )
        elif section is not None:
            drug_ids = as_tuple(section)
            classes = ()
        else:
            drug_ids = as_tuple(
                config_value(
                    config,
                    f"{arm}_drug_ids",
                    f"therapy_mapping.{arm}_drug_ids",
                    default=default_drugs,
                )
            )
            classes = ()

        explicit_classes = as_tuple(
            config_value(
                config,
                f"{arm}_therapy_classes",
                f"therapy_mapping.{arm}_therapy_classes",
                default=None,
            )
        )
        if explicit_classes:
            classes = explicit_classes
        if not classes:
            classes = (default_class,)
        if not drug_ids:
            drug_ids = default_drugs
        return drug_ids, classes

    conventional_ids, conventional_classes = resolve(
        "conventional", ("SYN_CONV_A", "SYN_CONV_B", "SYN_CONV_C"), "conventional"
    )
    advanced_ids, advanced_classes = resolve("advanced", ("SYN_ADV_A", "SYN_ADV_B"), "advanced")
    overlap = set(conventional_ids) & set(advanced_ids)
    if overlap:
        raise ValueError(f"Therapy drug mappings overlap: {sorted(overlap)}")
    class_overlap = set(conventional_classes) & set(advanced_classes)
    if class_overlap:
        raise ValueError(f"Therapy class mappings overlap: {sorted(class_overlap)}")
    return TherapyDefinition(
        conventional_drug_ids=conventional_ids,
        advanced_drug_ids=advanced_ids,
        conventional_classes=conventional_classes,
        advanced_classes=advanced_classes,
    )


def timeline_days(config: Any) -> tuple[int, int]:
    """Return validated observation and prediction window lengths."""

    observation = int(
        config_value(
            config,
            "observation_window_days",
            "timeline.observation_window_days",
            "cohort.observation_window_days",
            default=365,
        )
    )
    prediction = int(
        config_value(
            config,
            "prediction_window_days",
            "timeline.prediction_window_days",
            "cohort.prediction_window_days",
            default=90,
        )
    )
    if observation <= 0 or prediction <= 0:
        raise ValueError("Observation and prediction windows must be positive")
    return observation, prediction
