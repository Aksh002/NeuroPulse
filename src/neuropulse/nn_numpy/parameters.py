"""Named float64 parameters with explicit accumulated gradients."""

from dataclasses import dataclass, field

import numpy as np

from neuropulse.data.readers import Array


@dataclass
class Parameter:
    data: Array
    grad: Array = field(init=False)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64).copy()
        self.grad = np.zeros_like(self.data)


class Module:
    def forward(self, x: Array, *, training: bool = False) -> Array:
        raise NotImplementedError

    def backward(self, upstream: Array) -> Array:
        raise NotImplementedError

    def named_parameters(self) -> dict[str, Parameter]:
        return {}

    def state_dict(self) -> dict[str, Array]:
        return {name: p.data.copy() for name, p in self.named_parameters().items()}

    def load_state_dict(self, state: dict[str, Array]) -> None:
        parameters = self.named_parameters()
        if state.keys() != parameters.keys():
            raise ValueError("Parameter names mismatch")
        for name, value in state.items():
            if value.shape != parameters[name].data.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid parameter: {name}")
        for name, value in state.items():
            parameters[name].data[:] = value


class CachedModule(Module):
    def __init__(self) -> None:
        self.cache: Array | None = None

    def remember(self, x: Array, training: bool) -> None:
        if training:
            if self.cache is not None:
                raise RuntimeError(
                    "Unmatched training forward; run backward before another forward"
                )
            self.cache = x.copy()

    def consume(self) -> Array:
        if self.cache is None:
            raise RuntimeError("Backward requires an unmatched training forward")
        x, self.cache = self.cache, None
        return x
