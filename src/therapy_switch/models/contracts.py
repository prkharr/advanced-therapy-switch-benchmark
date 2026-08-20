"""Shared input and output contracts for model benchmarking.

The contracts deliberately contain no orchestration assumptions.  A caller can
construct :class:`ModelRun` from in-memory pandas objects and optionally attach
event tensors for the longitudinal models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# ``COMPLETED`` matches the benchmark-table schema.  ``SUCCESS`` remains the
# ergonomic Python constant used by runners and callers.
SUCCESS = "COMPLETED"
FAILED = "FAILED"
NOT_APPLICABLE = "NOT APPLICABLE"


class LeakageError(ValueError):
    """Raised when a sequence contains information after its index date."""


def _as_binary_target(values: Sequence[Any], split: str) -> np.ndarray:
    target = np.asarray(values).reshape(-1)
    if target.size == 0:
        raise ValueError(f"{split} target is empty")
    if pd.isna(target).any():
        raise ValueError(f"{split} target contains missing values")
    unique = set(np.unique(target).tolist())
    if not unique.issubset({0, 1, False, True}):
        raise ValueError(f"{split} target must be binary 0/1; observed values: {sorted(unique)!r}")
    return target.astype(np.int64, copy=False)


@dataclass
class SequenceSplit:
    """A padded, pre-index event representation for one data partition.

    ``values`` is shaped ``(patients, events, event_features)``.  ``mask`` is
    shaped ``(patients, events)`` and uses ``True`` for real events. ``times``
    may contain non-negative elapsed-time values (for example, days since the
    previous event) and is used as temporal encoding by neural sequence models.

    Leakage safety must be auditable. Supply ``event_dates`` and ``index_dates``
    so it can be checked automatically, or set ``pre_index_verified=True`` only
    after the upstream cohort builder has enforced that invariant.
    """

    values: np.ndarray
    mask: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    event_dates: Optional[np.ndarray] = None
    index_dates: Optional[Sequence[Any]] = None
    pre_index_verified: bool = False

    def validated(self, expected_rows: Optional[int] = None) -> "SequenceSplit":
        values = np.asarray(self.values)
        if values.ndim != 3:
            raise ValueError("sequence values must have shape (patients, events, features)")
        if expected_rows is not None and values.shape[0] != expected_rows:
            raise ValueError(f"sequence has {values.shape[0]} patients; expected {expected_rows}")
        if values.shape[1] < 1 or values.shape[2] < 1:
            raise ValueError("sequence must contain at least one event and feature")

        mask = (
            np.ones(values.shape[:2], dtype=bool)
            if self.mask is None
            else np.asarray(self.mask, dtype=bool)
        )
        if mask.shape != values.shape[:2]:
            raise ValueError("sequence mask must match the patient/event dimensions")
        if np.any(mask.sum(axis=1) == 0):
            raise ValueError("every patient must have at least one pre-index event")
        if not np.isfinite(values[mask]).all():
            raise ValueError("valid sequence event features must be finite")
        # Packed recurrent sequences require real events followed by right
        # padding; accepting holes would silently discard later valid events.
        for row, row_mask in enumerate(mask):
            first_padding = np.flatnonzero(~row_mask)
            if first_padding.size and row_mask[first_padding[0] :].any():
                raise ValueError(f"sequence mask for patient row {row} is not right padded")

        times = None if self.times is None else np.asarray(self.times, dtype=float)
        if times is not None:
            if times.shape != values.shape[:2]:
                raise ValueError("sequence times must match the patient/event dimensions")
            if not np.isfinite(times[mask]).all():
                raise ValueError("valid sequence times must be finite")
            if np.any(times[mask] < 0):
                raise ValueError("valid sequence times must be non-negative")

        has_event_dates = self.event_dates is not None
        has_index_dates = self.index_dates is not None
        if has_event_dates != has_index_dates:
            raise ValueError("event_dates and index_dates must be supplied together")
        if not has_event_dates and not self.pre_index_verified:
            raise LeakageError(
                "pre-index sequence safety is unverified; provide event_dates and "
                "index_dates or set pre_index_verified=True after upstream validation"
            )

        if has_event_dates:
            raw_event_dates = np.asarray(self.event_dates)
            if raw_event_dates.shape != values.shape[:2]:
                raise ValueError("event_dates must match the patient/event dimensions")
            raw_index_dates = np.asarray(self.index_dates)
            if raw_index_dates.reshape(-1).shape[0] != values.shape[0]:
                raise ValueError("index_dates must contain one date per patient")

            event_dates = pd.to_datetime(raw_event_dates.reshape(-1), errors="coerce")
            event_dates = np.asarray(event_dates).reshape(values.shape[:2])
            index_dates = np.asarray(pd.to_datetime(raw_index_dates.reshape(-1), errors="coerce"))
            if pd.isna(index_dates).any():
                raise ValueError("index_dates contains missing or invalid dates")
            if pd.isna(event_dates[mask]).any():
                raise ValueError("valid events contain missing or invalid dates")
            future = mask & (event_dates > index_dates[:, None])
            if future.any():
                first = np.argwhere(future)[0]
                raise LeakageError(
                    "post-index event detected at patient row "
                    f"{int(first[0])}, event position {int(first[1])}"
                )

            # Events must be in chronological order for temporal encoders.
            for row in range(values.shape[0]):
                valid_dates = event_dates[row, mask[row]]
                if valid_dates.size > 1 and np.any(valid_dates[1:] < valid_dates[:-1]):
                    raise ValueError(
                        f"events for patient row {row} are not chronologically ordered"
                    )

        # Store normalized arrays so downstream code has one representation.
        self.values = values.astype(np.float32, copy=False)
        self.mask = mask
        self.times = times
        return self


@dataclass
class ModelRun:
    """Data and configuration for one or more model runners."""

    X_train: pd.DataFrame
    y_train: Sequence[Any]
    X_val: pd.DataFrame
    y_val: Sequence[Any]
    X_test: pd.DataFrame
    y_test: Sequence[Any]
    sequence_train: Optional[SequenceSplit] = None
    sequence_val: Optional[SequenceSplit] = None
    sequence_test: Optional[SequenceSplit] = None
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    random_state: int = 42
    artifact_dir: Optional[Path | str] = None

    def validate_tabular(self) -> "ModelRun":
        frames = {
            "train": self.X_train,
            "validation": self.X_val,
            "test": self.X_test,
        }
        for split, frame in frames.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"X_{split} must be a pandas DataFrame")
            if frame.empty:
                raise ValueError(f"X_{split} is empty")

        expected_columns = list(self.X_train.columns)
        for split, frame in (("validation", self.X_val), ("test", self.X_test)):
            if list(frame.columns) != expected_columns:
                raise ValueError(f"{split} feature columns/order differ from the training frame")

        forbidden = {
            "target",
            "label",
            "outcome",
            "y",
            "advanced_therapy_switch",
            "advanced_therapy_initiation",
            "switch_outcome",
        }
        leaked = [
            str(column) for column in expected_columns if str(column).strip().lower() in forbidden
        ]
        if leaked:
            raise LeakageError(
                "target-like columns are not permitted in model features: " + ", ".join(leaked)
            )

        targets = self.targets()
        for split, frame, target in (
            ("train", self.X_train, targets[0]),
            ("validation", self.X_val, targets[1]),
            ("test", self.X_test, targets[2]),
        ):
            if len(frame) != len(target):
                raise ValueError(f"X_{split} has {len(frame)} rows but y_{split} has {len(target)}")
        return self

    def validate_sequences(self) -> "ModelRun":
        sequences = (self.sequence_train, self.sequence_val, self.sequence_test)
        if any(sequence is None for sequence in sequences):
            raise ValueError("train, validation, and test sequence inputs are all required")
        targets = self.targets()
        for sequence, target in zip(sequences, targets):
            assert sequence is not None
            sequence.validated(expected_rows=len(target))
        input_dims = {sequence.values.shape[2] for sequence in sequences if sequence}
        if len(input_dims) != 1:
            raise ValueError("all sequence splits must use the same event feature size")
        return self

    def targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            _as_binary_target(self.y_train, "training"),
            _as_binary_target(self.y_val, "validation"),
            _as_binary_target(self.y_test, "test"),
        )

    def options_for(self, key: str) -> dict[str, Any]:
        options = self.params.get(key, {})
        if not isinstance(options, Mapping):
            raise TypeError(f"params[{key!r}] must be a mapping")
        return dict(options)

    @property
    def artifacts(self) -> Optional[Path]:
        return None if self.artifact_dir is None else Path(self.artifact_dir)


@dataclass
class ModelResult:
    """Uniform result returned by every benchmark runner."""

    model: str
    category: str
    status: str
    reason: Optional[str] = None
    estimator: Any = None
    validation_probabilities: Optional[np.ndarray] = None
    test_probabilities: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None
    threshold: float = 0.5
    validation_score: Optional[float] = None
    best_params: dict[str, Any] = field(default_factory=dict)
    training_time_seconds: Optional[float] = None
    inference_time_seconds: Optional[float] = None
    history: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {SUCCESS, FAILED, NOT_APPLICABLE}:
            raise ValueError(
                f"invalid model status {self.status!r}; expected COMPLETED, FAILED, "
                "or NOT APPLICABLE"
            )
        if self.status != SUCCESS:
            if not self.reason:
                raise ValueError("failed/not-applicable results require a reason")
            return
        if self.test_probabilities is None or self.validation_probabilities is None:
            raise ValueError("completed model results require validation/test probabilities")
        validation = np.asarray(self.validation_probabilities, dtype=float).reshape(-1)
        test = np.asarray(self.test_probabilities, dtype=float).reshape(-1)
        if not np.isfinite(validation).all() or not np.isfinite(test).all():
            raise ValueError("completed model probabilities must be finite")
        if np.any((validation < 0) | (validation > 1)) or np.any((test < 0) | (test > 1)):
            raise ValueError("completed model probabilities must be between zero and one")
        self.validation_probabilities = validation
        self.test_probabilities = test
        if self.test_predictions is not None:
            predictions = np.asarray(self.test_predictions, dtype=np.int8).reshape(-1)
            if len(predictions) != len(test):
                raise ValueError("test predictions and probabilities must be aligned")
            self.test_predictions = predictions

    @property
    def succeeded(self) -> bool:
        return self.status == SUCCESS

    @property
    def probabilities(self) -> Optional[np.ndarray]:
        """Alias for held-out test probabilities used by evaluators."""

        return self.test_probabilities

    @property
    def predictions(self) -> Optional[np.ndarray]:
        """Alias for held-out thresholded predictions used by evaluators."""

        return self.test_predictions

    @classmethod
    def not_applicable(
        cls, model: str, category: str, reason: str, **metadata: Any
    ) -> "ModelResult":
        return cls(
            model=model,
            category=category,
            status=NOT_APPLICABLE,
            reason=reason,
            metadata=dict(metadata),
        )

    @classmethod
    def failed(cls, model: str, category: str, reason: str, **metadata: Any) -> "ModelResult":
        return cls(
            model=model,
            category=category,
            status=FAILED,
            reason=reason,
            metadata=dict(metadata),
        )

    def summary(self) -> dict[str, Any]:
        """Return a serialization-friendly record without patient predictions."""

        return {
            "Model": self.model,
            "Category": self.category,
            "Status": self.status,
            "Reason": self.reason,
            "Validation PR-AUC": self.validation_score,
            "Threshold": self.threshold,
            "Training Time": self.training_time_seconds,
            "Inference Time": self.inference_time_seconds,
            "Best Parameters": self.best_params,
            **self.metadata,
        }
