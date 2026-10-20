"""Stable activations with explicit elementwise derivatives."""

import numpy as np

from neuropulse.data.readers import Array
from neuropulse.nn_numpy.parameters import CachedModule


def sigmoid(x: Array) -> Array:
    result = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    result[~positive] = e / (1 + e)
    return result


class Activation(CachedModule):
    def __init__(self, kind: str = "relu") -> None:
        super().__init__()
        if kind not in ("relu", "sigmoid", "tanh"):
            raise ValueError("Unknown activation")
        self.kind = kind

    def forward(self, x: Array, *, training: bool = False) -> Array:
        self.remember(x, training)
        if self.kind == "relu":
            return np.maximum(x, 0)
        if self.kind == "sigmoid":
            return sigmoid(x)
        return np.tanh(x)

    def backward(self, upstream: Array) -> Array:
        x = self.consume()
        if upstream.shape != x.shape:
            raise ValueError("Activation upstream shape mismatch")
        if self.kind == "relu":
            derivative = (x > 0).astype(np.float64)
        elif self.kind == "sigmoid":
            a = sigmoid(x)
            derivative = a * (1 - a)
        else:
            derivative = 1 - np.tanh(x) ** 2
        return np.asarray(upstream * derivative, dtype=np.float64)
