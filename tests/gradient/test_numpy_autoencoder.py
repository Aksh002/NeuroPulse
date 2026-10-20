"""The manual dense autoencoder must learn reconstruction and survive export."""

import numpy as np

from neuropulse.models.numpy_mlp import build_autoencoder
from neuropulse.nn_numpy.losses import Loss
from neuropulse.nn_numpy.optimizers import Optimizer
from neuropulse.nn_numpy.serialization import load, save


def test_manual_autoencoder_reconstructs_and_round_trips(tmp_path):
    rng = np.random.default_rng(12)
    latent = rng.normal(size=(24, 2))
    basis = rng.normal(size=(2, 12))
    inputs = np.asarray(latent @ basis, dtype=np.float64)
    model = build_autoencoder(features=3, window_length=4)
    optimizer = Optimizer(model.named_parameters(), learning_rate=0.003)
    loss = Loss()
    initial = loss.forward(model.forward(inputs), inputs)
    loss.backward()
    for _ in range(160):
        optimizer.zero_grad()
        loss.forward(model.forward(inputs, training=True), inputs)
        model.backward(loss.backward())
        optimizer.clip_grad()
        optimizer.step()
    final = loss.forward(model.forward(inputs), inputs)
    assert final < initial * 0.08
    output = model.forward(inputs)
    path = tmp_path / "autoencoder"
    save(model, path)
    restored, _ = load(path)
    np.testing.assert_array_equal(restored.forward(inputs), output)
