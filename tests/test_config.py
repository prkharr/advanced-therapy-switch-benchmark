from copy import deepcopy

import pytest

from therapy_switch.config import ConfigError, load_config, validate_config


def test_quickstart_configuration_is_valid():
    config = load_config("configs/quickstart.yaml")
    assert config["timeline"]["observation_window_days"] == 365
    assert config["splitting"]["primary_experiment"] == "temporal"


def test_overlapping_therapy_mapping_is_rejected():
    config = load_config("configs/quickstart.yaml")
    invalid = deepcopy(config)
    invalid["therapy_mapping"]["advanced"].append(invalid["therapy_mapping"]["conventional"][0])
    with pytest.raises(ConfigError, match="overlap"):
        validate_config(invalid)


def test_invalid_split_fractions_are_rejected():
    config = load_config("configs/quickstart.yaml")
    invalid = deepcopy(config)
    invalid["splitting"]["validation_fraction"] = 0.5
    invalid["splitting"]["test_fraction"] = 0.5
    with pytest.raises(ConfigError, match="sum to < 1"):
        validate_config(invalid)
