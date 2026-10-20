"""Manual BPTT must learn a sequence-dependent target, not just match gradients."""

import numpy as np

from neuropulse.nn_numpy.layers import Dense
from neuropulse.nn_numpy.losses import Loss
from neuropulse.nn_numpy.optimizers import Optimizer
from neuropulse.nn_numpy.recurrent import RNN


def test_basic_rnn_learns_recent_sequence_sum():
    rng = np.random.default_rng(41)
    inputs = rng.uniform(-0.5, 0.5, (48, 4, 1))
    targets = inputs[:, -2:, 0].sum(axis=1, keepdims=True)
    cell = RNN(1, 8, rng)
    head = Dense(8, 1, rng, "xavier")
    optimizer = Optimizer(
        {
            **{f"cell.{name}": p for name, p in cell.named_parameters().items()},
            **{f"head.{name}": p for name, p in head.named_parameters().items()},
        },
        learning_rate=0.01,
    )
    loss = Loss()

    def measure() -> float:
        states = cell.forward(inputs)
        return loss.forward(head.forward(states[:, -1]), targets)

    initial = measure()
    loss.backward()
    for _ in range(250):
        optimizer.zero_grad()
        states = cell.forward(inputs, training=True)
        prediction = head.forward(states[:, -1], training=True)
        loss.forward(prediction, targets)
        last_gradient = head.backward(loss.backward())
        upstream = np.zeros_like(states)
        upstream[:, -1] = last_gradient
        cell.backward(upstream)
        optimizer.clip_grad()
        optimizer.step()
    assert measure() < initial * 0.02
