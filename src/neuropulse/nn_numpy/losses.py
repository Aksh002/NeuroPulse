"""Scalar mean-reduced losses; the backward pass performs reduction exactly once."""

import numpy as np

from neuropulse.data.readers import Array
from neuropulse.nn_numpy.activations import sigmoid


class Loss:
    def __init__(self, kind: str = "mse") -> None:
        if kind not in ("mse", "bce_logits"):
            raise ValueError("Unknown loss")
        self.kind = kind
        self.gradient: Array | None = None

    def forward(self, prediction: Array, target: Array) -> float:
        if prediction.shape != target.shape or not prediction.size:
            raise ValueError("Loss inputs require matching nonempty shapes")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Nonfinite loss inputs")
        if self.kind == "mse":
            self.gradient = 2 * (prediction - target) / target.size
            return float(np.mean((prediction - target) ** 2))
        if np.any((target < 0) | (target > 1)):
            raise ValueError("Binary targets must lie in [0,1]")
        self.gradient = (sigmoid(prediction) - target) / target.size
        return float(
            np.mean(
                np.maximum(prediction, 0)
                - prediction * target
                + np.log1p(np.exp(-np.abs(prediction)))
            )
        )

    def backward(self) -> Array:
        if self.gradient is None:
            raise RuntimeError("Loss backward requires forward")
        gradient, self.gradient = self.gradient, None
        return gradient
