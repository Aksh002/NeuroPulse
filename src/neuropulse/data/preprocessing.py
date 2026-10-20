"""Training-only sensor selection and population-standardization artifacts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuropulse.data.readers import Array, Engines


@dataclass(frozen=True)
class SensorScaler:
    selected: NDArray[np.int64]
    mean: Array
    std: Array
    training_ids: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        engines: Engines,
        ids: list[int],
        variance_threshold: float = 1e-8,
        std_floor: float = 1e-12,
    ) -> "SensorScaler":
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Explicit unique training engine IDs required")
        sensors = np.concatenate([engines[i][:, 5:] for i in sorted(ids)])
        selected = np.flatnonzero(sensors.var(axis=0) > variance_threshold)
        if not len(selected):
            raise ValueError("No nonconstant training sensors")
        mean = sensors[:, selected].mean(axis=0)
        std = sensors[:, selected].std(axis=0)
        std[std <= std_floor] = 1.0
        return cls(selected, mean, std, tuple(sorted(ids)))

    def transform(self, sensors: Array) -> Array:
        if sensors.ndim != 2 or sensors.shape[1] != 21 or not np.isfinite(sensors).all():
            raise ValueError("Expected finite [cycles, 21] raw sensors")
        return np.asarray((sensors[:, self.selected] - self.mean) / self.std, dtype=np.float64)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            selected=self.selected,
            mean=self.mean,
            std=self.std,
            training_ids=np.array(self.training_ids, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Path) -> "SensorScaler":
        with np.load(path, allow_pickle=False) as state:
            selected = state["selected"].copy()
            mean, std = state["mean"].copy(), state["std"].copy()
            ids = tuple(int(i) for i in state["training_ids"])
        if selected.ndim != 1 or mean.shape != selected.shape or std.shape != mean.shape:
            raise ValueError("Scaler shape mismatch")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Invalid scaler values")
        if len(set(selected.tolist())) != len(selected) or np.any(
            (selected < 0) | (selected >= 21)
        ):
            raise ValueError("Invalid selected sensor indices")
        return cls(selected, mean, std, ids)
