"""Lazy per-engine window indexing and terminal-prefix padding."""

import numpy as np
from numpy.typing import NDArray

from neuropulse.data.readers import Array, Engines
from neuropulse.data.targets import rul_targets

NDIndices = NDArray[np.int64]


def terminal_window(sequence: Array, length: int) -> tuple[Array, bool]:
    if sequence.ndim != 2 or not len(sequence) or length <= 0:
        raise ValueError("Nonempty sensor sequence and positive window length required")
    padded = len(sequence) < length
    if padded:
        sequence = np.concatenate(
            [np.repeat(sequence[:1], length - len(sequence), axis=0), sequence]
        )
    return sequence[-length:].copy(), padded


class WindowDataset:
    """Stores O(rows*features + windows) state rather than O(windows*W*features)."""

    def __init__(
        self, engines: Engines, length: int = 30, stride: int = 1, cap: int | None = 125
    ) -> None:
        if length <= 0 or stride <= 0:
            raise ValueError("Window length and stride must be positive")
        self.engines = engines
        self.length = length
        self.labels = {key: rul_targets(rows[:, 1], cap) for key, rows in engines.items()}
        self.index = [
            (key, end)
            for key, rows in sorted(engines.items())
            for end in range(length - 1, len(rows), stride)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[Array, Array]:
        key, end = self.index[index]
        return (
            self.engines[key][end - self.length + 1 : end + 1, 5:],
            np.array([self.labels[key][end]], dtype=np.float64),
        )

    def batches(self, batch_size: int, rng: np.random.Generator) -> list[NDIndices]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        indices = rng.permutation(len(self))
        return [indices[start : start + batch_size] for start in range(0, len(self), batch_size)]

    def batch(self, indices: "NDIndices") -> tuple[Array, Array]:
        samples = [self[int(i)] for i in indices]
        return np.stack([x for x, _ in samples]), np.stack([y for _, y in samples])
