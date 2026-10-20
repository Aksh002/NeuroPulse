"""Manual mini-batch learning with explicit history and resumable RNG/optimizer state."""

from typing import Any

import numpy as np

from neuropulse.data.readers import Array
from neuropulse.nn_numpy.layers import Sequential
from neuropulse.nn_numpy.losses import Loss
from neuropulse.nn_numpy.optimizers import Optimizer


def fit(
    model: Sequential,
    x: Array,
    y: Array,
    optimizer: Optimizer,
    rng: np.random.Generator,
    epochs: int = 100,
    batch_size: int = 64,
    loss_kind: str = "mse",
) -> list[dict[str, float]]:
    if len(x) != len(y) or len(x) == 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("Invalid training data/budget")
    history = []
    loss = Loss(loss_kind)
    for epoch in range(epochs):
        indices = rng.permutation(len(x))
        total = 0.0
        for start in range(0, len(x), batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad()
            prediction = model.forward(x[batch], training=True)
            total += loss.forward(prediction, y[batch]) * len(batch)
            model.backward(loss.backward())
            optimizer.clip_grad()
            optimizer.step()
        history.append({"epoch": float(epoch + 1), "loss": total / len(x)})
    return history


def training_state(optimizer: Optimizer, rng: np.random.Generator) -> dict[str, Any]:
    return {"optimizer": optimizer.state_dict(), "rng": rng.bit_generator.state}
