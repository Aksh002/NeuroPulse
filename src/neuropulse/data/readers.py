"""Fail-closed C-MAPSS readers; no imputation, reordering, or test-label access."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Engines = dict[int, Array]
SENSOR_NAMES = tuple(f"sensor_{i:02d}" for i in range(1, 22))


def read_engines(path: Path) -> Engines:
    try:
        table = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid sensor file {path}: {exc}") from exc
    if table.shape[0] == 0 or table.shape[1] != 26 or not np.isfinite(table).all():
        raise ValueError("Expected nonempty, finite 26-column observations")
    keys = table[:, :2]
    if np.any(keys <= 0) or np.any(keys != np.floor(keys)):
        raise ValueError("Engine IDs and cycles must be positive integers")
    engines: Engines = {}
    for key in np.unique(table[:, 0]):
        rows = table[table[:, 0] == key].copy()
        if rows[0, 1] != 1 or np.any(np.diff(rows[:, 1]) != 1):
            raise ValueError(f"Engine {key}: duplicate, gapped or unordered cycles")
        engines[int(key)] = rows
    return engines


def read_test_labels(path: Path, expected_ids: list[int]) -> Array:
    """Evaluation-only: reject ambiguous official label alignment."""
    if expected_ids != list(range(1, len(expected_ids) + 1)):
        raise ValueError("Official labels require sorted contiguous test engine IDs")
    labels = np.loadtxt(path, dtype=np.float64, ndmin=1)
    if labels.shape != (len(expected_ids),) or not np.isfinite(labels).all():
        raise ValueError("Official terminal label count/shape/finite-value mismatch")
    if np.any(labels < 0) or np.any(labels != np.floor(labels)):
        raise ValueError("Official RUL labels must be nonnegative integer cycles")
    return labels
