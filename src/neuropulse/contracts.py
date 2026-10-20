"""Shared numerical contracts; no neural operations are implemented in this stage."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def validate_sensor_windows(
    windows: NDArray[Any], *, window_length: int, feature_count: int
) -> None:
    """Require finite real sensor windows shaped [B, W, F] with nonempty B.

    This boundary prevents swapped time/feature axes, empty batches, and nonfinite
    observations from reaching future models. It never casts or mutates the input.
    """
    if window_length <= 0 or feature_count <= 0:
        raise ValueError("Expected window length and feature count must be positive")
    if not isinstance(windows, np.ndarray):
        raise TypeError("Sensor windows must be a NumPy array")
    if windows.ndim != 3 or windows.shape[0] == 0:
        raise ValueError("Sensor windows must have nonempty shape [B, W, F]")
    if windows.shape[1:] != (window_length, feature_count):
        raise ValueError(f"Expected [B, {window_length}, {feature_count}], got {windows.shape}")
    if windows.dtype.kind not in "fiu":
        raise TypeError("Sensor windows must contain real numeric values")
    if not np.isfinite(windows).all():
        raise ValueError("Sensor windows must contain only finite values")
