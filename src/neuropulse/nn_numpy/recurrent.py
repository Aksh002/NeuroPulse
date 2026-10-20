"""Tanh recurrent computation and manually accumulated BPTT over independent windows."""

import numpy as np

from neuropulse.data.readers import Array
from neuropulse.nn_numpy.parameters import CachedModule, Parameter


class RNN(CachedModule):
    def __init__(self, inputs: int, hidden: int, rng: np.random.Generator) -> None:
        super().__init__()
        if min(inputs, hidden) <= 0:
            raise ValueError("Invalid recurrent dimensions")
        self.input_weight = Parameter(
            rng.normal(0, np.sqrt(2 / (inputs + hidden)), (inputs, hidden))
        )
        self.hidden_weight = Parameter(rng.normal(0, np.sqrt(1 / hidden), (hidden, hidden)))
        self.bias = Parameter(np.zeros(hidden))
        self.states: Array | None = None

    def forward(self, x: Array, *, training: bool = False) -> Array:
        if x.ndim != 3 or not x.shape[1] or x.shape[2] != self.input_weight.data.shape[0]:
            raise ValueError("RNN expects nonempty [B,T,F]")
        self.remember(x, training)
        states = np.zeros((len(x), x.shape[1] + 1, len(self.bias.data)))
        for t in range(x.shape[1]):
            states[:, t + 1] = np.tanh(
                x[:, t] @ self.input_weight.data
                + states[:, t] @ self.hidden_weight.data
                + self.bias.data
            )
        if training:
            self.states = states
        return states[:, 1:].copy()

    def backward(self, upstream: Array) -> Array:
        x = self.consume()
        states, self.states = self.states, None
        if states is None or upstream.shape != states[:, 1:].shape:
            raise ValueError("Recurrent state/upstream mismatch")
        dx = np.zeros_like(x)
        dh: Array = np.zeros((len(x), len(self.bias.data)))
        for t in range(x.shape[1] - 1, -1, -1):
            da = (upstream[:, t] + dh) * (1 - states[:, t + 1] ** 2)
            self.input_weight.grad += x[:, t].T @ da
            self.hidden_weight.grad += states[:, t].T @ da
            self.bias.grad += da.sum(axis=0)
            dx[:, t] = da @ self.input_weight.data.T
            dh = da @ self.hidden_weight.data.T
        return dx

    def named_parameters(self) -> dict[str, Parameter]:
        return {
            "input_weight": self.input_weight,
            "hidden_weight": self.hidden_weight,
            "bias": self.bias,
        }
