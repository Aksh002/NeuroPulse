"""Manual dense chain rule and sequential composition; no autodiff dependency."""

import numpy as np

from neuropulse.data.readers import Array
from neuropulse.nn_numpy.parameters import CachedModule, Module, Parameter


class Dense(CachedModule):
    def __init__(
        self, inputs: int, outputs: int, rng: np.random.Generator, initialization: str = "he"
    ) -> None:
        super().__init__()
        if inputs <= 0 or outputs <= 0 or initialization not in ("he", "xavier"):
            raise ValueError("Invalid dense dimensions or initialization")
        std = np.sqrt(2 / (inputs if initialization == "he" else inputs + outputs))
        self.weight = Parameter(rng.normal(0, std, (inputs, outputs)))
        self.bias = Parameter(np.zeros(outputs))

    def forward(self, x: Array, *, training: bool = False) -> Array:
        if x.ndim != 2 or x.shape[1] != self.weight.data.shape[0]:
            raise ValueError("Dense expects [B, input_features]")
        self.remember(x, training)
        return np.asarray(x @ self.weight.data + self.bias.data, dtype=np.float64)

    def backward(self, upstream: Array) -> Array:
        x = self.consume()
        if upstream.shape != (len(x), len(self.bias.data)):
            raise ValueError("Dense upstream shape mismatch")
        self.weight.grad += x.T @ upstream
        self.bias.grad += upstream.sum(axis=0)
        return np.asarray(upstream @ self.weight.data.T, dtype=np.float64)

    def named_parameters(self) -> dict[str, Parameter]:
        return {"weight": self.weight, "bias": self.bias}


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Array, *, training: bool = False) -> Array:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, upstream: Array) -> Array:
        for layer in reversed(self.layers):
            upstream = layer.backward(upstream)
        return upstream

    def named_parameters(self) -> dict[str, Parameter]:
        return {
            f"{i}.{key}": p
            for i, layer in enumerate(self.layers)
            for key, p in layer.named_parameters().items()
        }
