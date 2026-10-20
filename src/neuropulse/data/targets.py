"""Explicit cycle-based labels for run-to-failure training trajectories."""

import numpy as np

from neuropulse.data.readers import Array


def rul_targets(cycles: Array, cap: int | None = 125) -> Array:
    if cycles.ndim != 1 or not len(cycles) or not np.isfinite(cycles).all():
        raise ValueError("Expected a finite nonempty cycle vector")
    if np.any(np.diff(cycles) != 1) or cycles[0] != 1:
        raise ValueError("Run-to-failure cycles must start at one and be consecutive")
    raw = cycles[-1] - cycles
    if cap is not None:
        if cap <= 0:
            raise ValueError("RUL cap must be positive")
        raw = np.minimum(raw, cap)
    return np.asarray(raw, dtype=np.float64)
