"""Synthetic shape/numerical cases that would invalidate downstream predictions."""

import numpy as np
import pytest

from neuropulse.contracts import validate_sensor_windows


def test_valid_windows_are_not_modified():
    windows = np.random.default_rng(42).normal(size=(2, 30, 14))
    original = windows.copy()
    validate_sensor_windows(windows, window_length=30, feature_count=14)
    np.testing.assert_array_equal(windows, original)


@pytest.mark.parametrize("shape", [(0, 30, 14), (2, 14, 30), (30, 14), (2, 29, 14)])
def test_invalid_shapes_are_rejected(shape):
    with pytest.raises(ValueError):
        validate_sensor_windows(np.zeros(shape), window_length=30, feature_count=14)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_observations_are_rejected(invalid):
    windows = np.zeros((1, 30, 14))
    windows[0, 29, 13] = invalid
    with pytest.raises(ValueError):
        validate_sensor_windows(windows, window_length=30, feature_count=14)


@pytest.mark.parametrize("dtype", [bool, complex, str, object])
def test_nonreal_numerical_arrays_are_rejected(dtype):
    with pytest.raises(TypeError):
        validate_sensor_windows(
            np.zeros((1, 30, 14), dtype=dtype), window_length=30, feature_count=14
        )
