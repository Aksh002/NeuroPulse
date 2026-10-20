"""Scientific configuration must fail closed on invalid or misspelled settings."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from neuropulse.config import DataConfig, load_data_config


def test_committed_protocol_loads():
    config_path = Path(__file__).resolve().parents[2] / "configs/data/fd001.yaml"
    config = load_data_config(config_path)
    assert (config.split_seed, config.validation_prefix_seed) == (42, 142)
    assert (config.window_length, config.rul_cap_cycles) == (30, 125)


@pytest.mark.parametrize(
    "overrides",
    [
        {"window_lenght": 30},
        {"window_length": 0},
        {"split_seed": True},
        {"split_seed": "42"},
        {"train_fraction": 1.0},
        {"sensor_variance_threshold": float("nan")},
        {"scaler_std_floor": 0.0},
        {"rul_cap_cycles": -1},
        {"subset": "FD004"},
    ],
)
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValidationError):
        DataConfig.model_validate(overrides)


def test_uncapped_target_is_explicit():
    assert DataConfig(rul_cap_cycles=None).rul_cap_cycles is None


@pytest.mark.parametrize("content", ["", "- 42\n- 142", "window_length: [30"])
def test_invalid_yaml_document_fails(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_data_config(path)
