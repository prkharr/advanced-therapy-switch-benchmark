import numpy as np
import pandas as pd
import pytest

from therapy_switch.evaluation.confirmation import confirm_average_precision


def frame():
    y = np.tile([0, 0, 1, 1], 20)
    return pd.DataFrame(
        {
            "cohort": np.repeat([1, 2], 40),
            "patient_id": np.tile(np.repeat(np.arange(20), 2), 2),
            "label": y,
            "candidate": 0.1 + 0.8 * y,
            "reference_a": 0.5,
            "reference_b": 0.5,
        }
    )


def test_confirmation_paired_cluster_gate_and_family_adjustment():
    data = frame()
    result = confirm_average_precision(
        data, candidate="candidate", references=["reference_a", "reference_b"], n_bootstrap=100
    )
    assert result["passes"]
    assert len(result["cohorts"]) == 2
    for row in result["comparisons"]:
        assert row["confidence_level"] == 0.975
        assert row["ci_lower"] > 0
        assert row["mean_ap_gain"] == 0.5
    data["candidate"] = data.reference_a
    result = confirm_average_precision(
        data, candidate="candidate", references=["reference_a"], n_bootstrap=100
    )
    assert not result["passes"]
    assert result["comparisons"][0]["ci_lower"] == 0
    assert result["comparisons"][0]["ci_upper"] == 0


def test_confirmation_rejects_bad_scores_and_empty_reference_family():
    data = frame()
    with pytest.raises(ValueError):
        confirm_average_precision(data, candidate="candidate", references=[], n_bootstrap=100)
    data.loc[0, "candidate"] = np.nan
    with pytest.raises(ValueError):
        confirm_average_precision(
            data, candidate="candidate", references=["reference_a"], n_bootstrap=100
        )
