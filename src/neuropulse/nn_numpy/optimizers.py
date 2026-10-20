"""SGD, momentum, Adam and global clipping with named serializable optimizer state."""

from typing import Any

import numpy as np

from neuropulse.nn_numpy.parameters import Parameter


class Optimizer:
    def __init__(
        self,
        parameters: dict[str, Parameter],
        kind: str = "adam",
        learning_rate: float = 0.001,
        momentum: float = 0.9,
    ) -> None:
        if kind not in ("sgd", "momentum", "adam") or learning_rate <= 0:
            raise ValueError("Invalid optimizer")
        self.parameters, self.kind, self.lr, self.momentum = (
            parameters,
            kind,
            learning_rate,
            momentum,
        )
        self.step_count = 0
        self.first = {k: np.zeros_like(p.data) for k, p in parameters.items()}
        self.second = {k: np.zeros_like(p.data) for k, p in parameters.items()}

    def zero_grad(self) -> None:
        for p in self.parameters.values():
            p.grad.fill(0)

    def clip_grad(self, limit: float = 5.0) -> float:
        if limit <= 0:
            raise ValueError("Clipping limit must be positive")
        norm = float(np.sqrt(sum(float(np.sum(p.grad**2)) for p in self.parameters.values())))
        if not np.isfinite(norm):
            raise ValueError("Nonfinite gradient norm")
        scale = min(1.0, limit / (norm + 1e-12))
        for p in self.parameters.values():
            p.grad *= scale
        return norm

    def step(self) -> None:
        if any(not np.isfinite(p.grad).all() for p in self.parameters.values()):
            raise ValueError("Nonfinite gradients")
        self.step_count += 1
        for name, p in self.parameters.items():
            g = p.grad
            if self.kind == "sgd":
                update = g
            elif self.kind == "momentum":
                self.first[name] = self.momentum * self.first[name] + g
                update = self.first[name]
            else:
                self.first[name] = 0.9 * self.first[name] + 0.1 * g
                self.second[name] = 0.999 * self.second[name] + 0.001 * g**2
                m = self.first[name] / (1 - 0.9**self.step_count)
                v = self.second[name] / (1 - 0.999**self.step_count)
                update = m / (np.sqrt(v) + 1e-8)
            p.data -= self.lr * update

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "lr": self.lr,
            "momentum": self.momentum,
            "step_count": self.step_count,
            "first": {k: v.tolist() for k, v in self.first.items()},
            "second": {k: v.tolist() for k, v in self.second.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["kind"] != self.kind or set(state["first"]) != set(self.parameters):
            raise ValueError("Optimizer state mismatch")
        for name, p in self.parameters.items():
            if (
                np.shape(state["first"][name]) != p.data.shape
                or np.shape(state["second"][name]) != p.data.shape
            ):
                raise ValueError("Optimizer state shape mismatch")
        self.lr, self.momentum, self.step_count = (
            state["lr"],
            state["momentum"],
            state["step_count"],
        )
        self.first = {k: np.array(v, dtype=np.float64) for k, v in state["first"].items()}
        self.second = {k: np.array(v, dtype=np.float64) for k, v in state["second"].items()}
