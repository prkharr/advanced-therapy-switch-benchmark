"""Consistent model runners for therapy-switch benchmarking.

The public API is dependency-aware: importing this module does not require the
optional gradient-boosting or deep-learning packages.
"""

from .architectures import (
    FocalLoss,
    RecurrentSequenceClassifier,
    TabularMLP,
    TemporalTransformerClassifier,
)
from .classical import (
    BaseModelRunner,
    CatBoostRunner,
    LightGBMRunner,
    LogisticRegressionRunner,
    NaiveBaselineRunner,
    RandomForestRunner,
    XGBoostRunner,
)
from .contracts import (
    FAILED,
    NOT_APPLICABLE,
    SUCCESS,
    LeakageError,
    ModelResult,
    ModelRun,
    SequenceSplit,
)
from .neural import (
    BiLSTMRunner,
    GRURunner,
    LSTMRunner,
    MLPRunner,
    TransformerRunner,
)
from .registry import model_registry, run_benchmark, run_models
from .tuning import ParamSpec, TuningResult, tune_estimator

__all__ = [
    "BaseModelRunner",
    "BiLSTMRunner",
    "CatBoostRunner",
    "FAILED",
    "FocalLoss",
    "GRURunner",
    "LSTMRunner",
    "LeakageError",
    "LightGBMRunner",
    "LogisticRegressionRunner",
    "MLPRunner",
    "ModelResult",
    "ModelRun",
    "NOT_APPLICABLE",
    "NaiveBaselineRunner",
    "ParamSpec",
    "RandomForestRunner",
    "RecurrentSequenceClassifier",
    "SUCCESS",
    "SequenceSplit",
    "TabularMLP",
    "TemporalTransformerClassifier",
    "TransformerRunner",
    "TuningResult",
    "XGBoostRunner",
    "model_registry",
    "run_benchmark",
    "run_models",
    "tune_estimator",
]
