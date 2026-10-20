"""Custom regression/autoencoder construction using verified manual dense modules."""

import numpy as np

from neuropulse.config import TrainConfig
from neuropulse.nn_numpy.activations import Activation
from neuropulse.nn_numpy.layers import Dense, Sequential
from neuropulse.nn_numpy.parameters import Module


def build_mlp(features: int, config: TrainConfig) -> Sequential:
    rng = np.random.default_rng(config.seed)
    layers: list[Module] = []
    previous = config.window_length * features
    init = (
        ("he" if config.activation == "relu" else "xavier")
        if config.initialization == "appropriate"
        else config.initialization
    )
    for depth in range(config.hidden_layers):
        width = max(8, config.dense_width // (2**depth))
        layers.extend([Dense(previous, width, rng, init), Activation(config.activation)])
        previous = width
    layers.append(Dense(previous, 1, rng, "xavier"))
    return Sequential(*layers)


def build_autoencoder(features: int, window_length: int, seed: int = 42) -> Sequential:
    """Dense reconstruction model on flattened standardized sensor windows.

    The encoder maps W*F -> 128 -> 32 and the decoder maps 32 -> 128 -> W*F.
    The caller uses MSE against the input; no RUL or fault-onset label is used.
    """
    if features <= 0 or window_length <= 0:
        raise ValueError("Positive feature count and window length required")
    width = features * window_length
    rng = np.random.default_rng(seed)
    return Sequential(
        Dense(width, 128, rng, "he"),
        Activation("relu"),
        Dense(128, 32, rng, "xavier"),
        Dense(32, 128, rng, "he"),
        Activation("relu"),
        Dense(128, width, rng, "xavier"),
    )
