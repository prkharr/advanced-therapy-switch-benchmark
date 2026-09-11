"""Independent baselines fitted from raw-derived features on the same partitions."""

from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from therapy_switch.patient_lists import latest_patient_indices, patient_capture_metrics

from .common import build_preprocessor, fitted_pipeline


@dataclass
class BaselineComparison:
    candidates: dict
    selected_name: str
    features: tuple
    validation_metrics: dict
    parameters: dict
    evidence_status: str = "independent_baseline"

    def predict_scores(self, frame, events=None):
        return self.candidates[self.selected_name].predict_proba(frame[list(self.features)])[:, 1]


def fit_baselines(train, validation, train_events, validation_events, features, *, settings, seed):
    """Fit two fresh estimators; validation selects, test data is never accepted here."""
    del train_events, validation_events
    if set(train.patient_id) & set(validation.patient_id):
        raise ValueError("Baseline training and validation patients must be disjoint")
    if train.label.nunique() != 2 or validation.iloc[latest_patient_indices(validation)].label.nunique() != 2:
        raise ValueError("Both baseline partitions need positive and negative outcomes")
    fraction = float(settings.get("patient_fraction", 0.1))
    candidates, metrics, parameters = {}, {}, {}
    # Generic library defaults, with explicit reproducibility/convergence/runtime settings.
    # Any later tuning must be documented and use only development partitions.
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "lightgbm": LGBMClassifier(random_state=seed, n_jobs=2, verbosity=-1),
    }
    for name, estimator in estimators.items():
        options = settings.get("parameters", {}).get(name, {})
        estimator.set_params(**options)
        model = fitted_pipeline(build_preprocessor(train[features], scale_numeric=name == "logistic_regression"), estimator)
        model.fit(train[features], train.label.to_numpy())
        scores = np.asarray(model.predict_proba(validation[features])[:, 1])
        metrics[name] = patient_capture_metrics(validation, scores, fraction=fraction)
        candidates[name] = model
        parameters[name] = estimator.get_params()
    selected = max(sorted(candidates), key=lambda name: (metrics[name]["recall"], metrics[name]["ap"]))
    return BaselineComparison(candidates, selected, tuple(features), metrics, parameters)
