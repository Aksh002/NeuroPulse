"""Basic Conv1D cross-correlation with explicit overlap-accumulating backward pass."""

import numpy as np

from neuropulse.data.readers import Array
from neuropulse.nn_numpy.parameters import CachedModule, Parameter


class Conv1D(CachedModule):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel: int,
        rng: np.random.Generator,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        if min(inputs, outputs, kernel, stride) <= 0 or padding < 0:
            raise ValueError("Invalid convolution dimensions")
        self.stride, self.padding = stride, padding
        self.weight = Parameter(
            rng.normal(0, np.sqrt(2 / (inputs * kernel)), (outputs, inputs, kernel))
        )
        self.bias = Parameter(np.zeros(outputs))

    def forward(self, x: Array, *, training: bool = False) -> Array:
        if x.ndim != 3 or x.shape[1] != self.weight.data.shape[1]:
            raise ValueError("Conv1D expects [B,C,W]")
        size = (x.shape[2] + 2 * self.padding - self.weight.data.shape[2]) // self.stride + 1
        if size <= 0:
            raise ValueError("Kernel exceeds padded sequence")
        self.remember(x, training)
        padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)))
        result = np.zeros((len(x), len(self.bias.data), size))
        for t in range(size):
            patch = padded[:, :, t * self.stride : t * self.stride + self.weight.data.shape[2]]
            result[:, :, t] = np.einsum("bcj,ocj->bo", patch, self.weight.data) + self.bias.data
        return result

    def backward(self, upstream: Array) -> Array:
        x = self.consume()
        kernel = self.weight.data.shape[2]
        size = (x.shape[2] + 2 * self.padding - kernel) // self.stride + 1
        if upstream.shape != (len(x), len(self.bias.data), size):
            raise ValueError("Convolution upstream shape mismatch")
        padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)))
        dx = np.zeros_like(padded)
        self.bias.grad += upstream.sum(axis=(0, 2))
        for t in range(size):
            start = t * self.stride
            self.weight.grad += np.einsum(
                "bo,bcj->ocj", upstream[:, :, t], padded[:, :, start : start + kernel]
            )
            dx[:, :, start : start + kernel] += np.einsum(
                "bo,ocj->bcj", upstream[:, :, t], self.weight.data
            )
        return dx[:, :, self.padding : self.padding + x.shape[2]]

    def named_parameters(self) -> dict[str, Parameter]:
        return {"weight": self.weight, "bias": self.bias}
