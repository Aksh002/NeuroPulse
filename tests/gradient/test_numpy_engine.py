"""Independent finite-difference gradients, updates, learning and checkpoint checks."""

import numpy as np
import pytest

from neuropulse.nn_numpy.activations import Activation, sigmoid
from neuropulse.nn_numpy.conv import Conv1D
from neuropulse.nn_numpy.layers import Dense, Sequential
from neuropulse.nn_numpy.losses import Loss
from neuropulse.nn_numpy.optimizers import Optimizer
from neuropulse.nn_numpy.parameters import Parameter
from neuropulse.nn_numpy.recurrent import RNN
from neuropulse.nn_numpy.serialization import load, save
from neuropulse.nn_numpy.trainer import fit, training_state


def numerical(array, function):
    result = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + 1e-5
        positive = function()
        array[index] = original - 1e-5
        negative = function()
        array[index] = original
        result[index] = (positive - negative) / 2e-5
    return result


@pytest.mark.gradient
@pytest.mark.parametrize("kind", ["dense", "relu", "sigmoid", "tanh", "conv", "rnn"])
def test_input_and_all_parameter_gradients(kind):
    rng = np.random.default_rng(41)
    if kind == "dense":
        layer, x = Dense(2, 3, rng), rng.normal(size=(2, 2))
    elif kind == "conv":
        layer, x = Conv1D(2, 2, 3, rng, stride=2, padding=1), rng.normal(size=(2, 2, 4))
    elif kind == "rnn":
        layer, x = RNN(2, 2, rng), rng.normal(size=(2, 3, 2))
    else:
        layer, x = Activation(kind), np.array([[-1.4, 0.6], [0.8, -0.3]])
    upstream = rng.normal(size=layer.forward(x).shape)
    layer.forward(x, training=True)
    dx = layer.backward(upstream)

    def scalar():
        return float(np.sum(layer.forward(x) * upstream))

    np.testing.assert_allclose(dx, numerical(x, scalar), atol=1e-6, rtol=1e-4)
    for parameter in layer.named_parameters().values():
        np.testing.assert_allclose(
            parameter.grad, numerical(parameter.data, scalar), atol=1e-6, rtol=1e-4
        )


@pytest.mark.parametrize("kind", ["mse", "bce_logits"])
def test_loss_gradient_and_no_broadcast(kind):
    x = np.array([[-1.3], [2.4]])
    y = np.array([[0.0], [1.0]])
    loss = Loss(kind)
    loss.forward(x, y)
    analytical = loss.backward()
    np.testing.assert_allclose(analytical, numerical(x, lambda: loss.forward(x, y)), atol=1e-7)
    with pytest.raises(ValueError):
        loss.forward(x, y[:, 0])
    assert np.isfinite(Loss("bce_logits").forward(np.array([[-1000.0], [1000.0]]), y))
    assert np.isfinite(sigmoid(np.array([-1000.0, 1000.0]))).all()


@pytest.mark.parametrize("kind", ["sgd", "momentum", "adam"])
def test_optimizer_and_state_round_trip(kind):
    p = Parameter(np.array([1.0, -1.0]))
    optimizer = Optimizer({"p": p}, kind, learning_rate=0.1)
    p.grad[:] = [2, -2]
    optimizer.step()
    expected = [0.9, -0.9] if kind == "adam" else [0.8, -0.8]
    np.testing.assert_allclose(p.data, expected, atol=1e-8)
    other = Parameter(p.data)
    restored = Optimizer({"p": other}, kind)
    restored.load_state_dict(optimizer.state_dict())
    p.grad[:], other.grad[:] = [3, -1], [3, -1]
    optimizer.step()
    restored.step()
    np.testing.assert_array_equal(p.data, other.data)
    if kind == "momentum":
        np.testing.assert_allclose(p.data, [0.32, -0.52])


def test_global_clipping_and_forward_cache_guard():
    p, q = Parameter(np.array([0.0])), Parameter(np.array([0.0]))
    p.grad[:], q.grad[:] = 3, 4
    optimizer = Optimizer({"p": p, "q": q})
    assert optimizer.clip_grad(2) == 5
    assert np.isclose(p.grad[0] ** 2 + q.grad[0] ** 2, 4)
    layer = Dense(1, 1, np.random.default_rng(2))
    layer.forward(np.ones((1, 1)), training=True)
    with pytest.raises(RuntimeError):
        layer.forward(np.ones((1, 1)), training=True)


def test_loss_decrease_serialization_and_exact_training_resume(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 2))
    y = 2 * x[:, :1] - x[:, 1:2]
    model = Sequential(Dense(2, 8, rng), Activation("tanh"), Dense(8, 1, rng, "xavier"))
    optimizer = Optimizer(model.named_parameters(), learning_rate=0.02)
    history = fit(model, x, y, optimizer, rng, epochs=80, batch_size=16)
    assert history[-1]["loss"] < history[0]["loss"] * 0.1
    checkpoint = tmp_path / "checkpoint"
    save(model, checkpoint, training_state(optimizer, rng))
    restored, extra = load(checkpoint)
    np.testing.assert_array_equal(restored.forward(x), model.forward(x))
    resumed = Optimizer(restored.named_parameters())
    resumed.load_state_dict(extra["optimizer"])
    next_rng = np.random.default_rng()
    next_rng.bit_generator.state = extra["rng"]
    fit(model, x, y, optimizer, rng, epochs=3, batch_size=16)
    fit(restored, x, y, resumed, next_rng, epochs=3, batch_size=16)
    np.testing.assert_array_equal(restored.forward(x), model.forward(x))


def test_dense_autoencoder_reconstruction_decreases():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(64, 1))
    x = np.concatenate([latent, latent * 0.4, latent * -0.6], axis=1)
    model = Sequential(
        Dense(3, 8, rng),
        Activation("tanh"),
        Dense(8, 2, rng, "xavier"),
        Dense(2, 8, rng),
        Activation("tanh"),
        Dense(8, 3, rng, "xavier"),
    )
    history = fit(
        model,
        x,
        x,
        Optimizer(model.named_parameters(), learning_rate=0.01),
        rng,
        epochs=100,
        batch_size=16,
    )
    assert history[-1]["loss"] < history[0]["loss"] * 0.2
