"""Example model plugin: refit any fixed patient-model recipe."""

import json
from pathlib import Path

from therapy_switch.models.patient_rank import PatientEnsemble, fit_patient_model


def fit_recipe(train, validation, train_events, validation_events, features, *, settings, seed):
    recipe = json.loads(Path(settings["recipe"]).read_text())
    selected = recipe["selected"]
    models = [
        fit_patient_model(
            spec, train, validation, train_events, validation_events, features, seed=seed
        )
        for spec in selected["members"]
    ]
    estimator = PatientEnsemble(models, selected["weights"])
    estimator.evidence_status = recipe.get(
        "evidence_status", settings.get("evidence_status", "experimental")
    )
    return estimator
