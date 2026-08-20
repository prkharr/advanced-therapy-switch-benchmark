"""Model registry and a small in-memory benchmark integration API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .classical import (
    BaseModelRunner,
    CatBoostRunner,
    LightGBMRunner,
    LogisticRegressionRunner,
    NaiveBaselineRunner,
    RandomForestRunner,
    XGBoostRunner,
)
from .contracts import ModelResult, ModelRun, SequenceSplit
from .neural import (
    BiLSTMRunner,
    GRURunner,
    LSTMRunner,
    MLPRunner,
    TransformerRunner,
)

RUNNER_TYPES: tuple[type[BaseModelRunner], ...] = (
    NaiveBaselineRunner,
    LogisticRegressionRunner,
    RandomForestRunner,
    XGBoostRunner,
    LightGBMRunner,
    CatBoostRunner,
    MLPRunner,
    LSTMRunner,
    GRURunner,
    BiLSTMRunner,
    TransformerRunner,
)


def model_registry() -> dict[str, BaseModelRunner]:
    """Return fresh, stateless runner objects in benchmark display order."""

    return {runner.key: runner for runner in (kind() for kind in RUNNER_TYPES)}


def run_models(
    run: ModelRun,
    model_keys: Optional[Iterable[str]] = None,
) -> dict[str, ModelResult]:
    """Execute selected runners while retaining explicit unavailable results."""

    registry = model_registry()
    selected = list(registry) if model_keys is None else list(model_keys)
    unknown = [key for key in selected if key not in registry]
    if unknown:
        raise KeyError(f"unknown model keys {unknown!r}; available keys are {list(registry)!r}")
    return {key: registry[key].run(run) for key in selected}


def run_benchmark(
    X_train: pd.DataFrame,
    y_train: Sequence[Any],
    X_val: pd.DataFrame,
    y_val: Sequence[Any],
    X_test: pd.DataFrame,
    y_test: Sequence[Any],
    *,
    sequence_train: Optional[SequenceSplit | np.ndarray] = None,
    sequence_val: Optional[SequenceSplit | np.ndarray] = None,
    sequence_test: Optional[SequenceSplit | np.ndarray] = None,
    sequence_train_mask: Optional[np.ndarray] = None,
    sequence_val_mask: Optional[np.ndarray] = None,
    sequence_test_mask: Optional[np.ndarray] = None,
    sequence_train_times: Optional[np.ndarray] = None,
    sequence_val_times: Optional[np.ndarray] = None,
    sequence_test_times: Optional[np.ndarray] = None,
    sequences_pre_index_verified: bool = False,
    params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    model_keys: Optional[Iterable[str]] = None,
    random_state: int = 42,
    artifact_dir: Optional[Path | str] = None,
) -> dict[str, ModelResult]:
    """Convenience API accepting pandas partitions and optional event tensors.

    Sequence inputs may be fully auditable :class:`SequenceSplit` instances or
    raw ``(patient, event, feature)`` arrays with corresponding masks/times. For
    raw arrays, ``sequences_pre_index_verified=True`` is an explicit assertion
    that the upstream event builder removed every post-index event.
    """

    def coerce_sequence(
        value: Optional[SequenceSplit | np.ndarray],
        mask: Optional[np.ndarray],
        times: Optional[np.ndarray],
    ) -> Optional[SequenceSplit]:
        if value is None:
            if mask is not None or times is not None:
                raise ValueError("a sequence mask/time array was supplied without values")
            return None
        if isinstance(value, SequenceSplit):
            if mask is not None or times is not None:
                raise ValueError("mask/time arguments cannot accompany an existing SequenceSplit")
            return value
        return SequenceSplit(
            values=np.asarray(value),
            mask=mask,
            times=times,
            pre_index_verified=sequences_pre_index_verified,
        )

    contract = ModelRun(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        sequence_train=coerce_sequence(sequence_train, sequence_train_mask, sequence_train_times),
        sequence_val=coerce_sequence(sequence_val, sequence_val_mask, sequence_val_times),
        sequence_test=coerce_sequence(sequence_test, sequence_test_mask, sequence_test_times),
        params={} if params is None else params,
        random_state=random_state,
        artifact_dir=artifact_dir,
    )
    return run_models(contract, model_keys=model_keys)
