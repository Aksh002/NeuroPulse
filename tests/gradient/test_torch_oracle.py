"""Equivalent float64 PyTorch oracles; educational implementation stays NumPy-only."""

import numpy as np
import pytest

from neuropulse.nn_numpy.conv import Conv1D
from neuropulse.nn_numpy.layers import Dense
from neuropulse.nn_numpy.optimizers import Optimizer
from neuropulse.nn_numpy.parameters import Parameter
from neuropulse.nn_numpy.recurrent import RNN

torch = pytest.importorskip("torch", reason="Install the research extra for independent parity")


@pytest.mark.parametrize("kind", ["dense", "conv", "rnn"])
def test_forward_input_and_parameter_parity(kind):
    rng = np.random.default_rng(42)
    if kind == "dense":
        custom = Dense(2, 3, rng)
        oracle = torch.nn.Linear(2, 3).double()
        oracle.weight.data.copy_(torch.from_numpy(custom.weight.data.T))
        oracle.bias.data.copy_(torch.from_numpy(custom.bias.data))
        x = rng.normal(size=(2, 2))
    elif kind == "conv":
        custom = Conv1D(2, 3, 3, rng, stride=2, padding=1)
        oracle = torch.nn.Conv1d(2, 3, 3, stride=2, padding=1).double()
        oracle.weight.data.copy_(torch.from_numpy(custom.weight.data))
        oracle.bias.data.copy_(torch.from_numpy(custom.bias.data))
        x = rng.normal(size=(2, 2, 6))
    else:
        custom = RNN(2, 3, rng)
        oracle = torch.nn.RNN(2, 3, batch_first=True, bias=True).double()
        oracle.weight_ih_l0.data.copy_(torch.from_numpy(custom.input_weight.data.T))
        oracle.weight_hh_l0.data.copy_(torch.from_numpy(custom.hidden_weight.data.T))
        oracle.bias_ih_l0.data.copy_(torch.from_numpy(custom.bias.data))
        oracle.bias_hh_l0.data.zero_()
        x = rng.normal(size=(2, 4, 2))
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    expected = oracle(tx)
    if kind == "rnn":
        expected = expected[0]
    actual = custom.forward(x, training=True)
    upstream = rng.normal(size=actual.shape)
    expected.backward(torch.from_numpy(upstream))
    dx = custom.backward(upstream)
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(dx, tx.grad.numpy(), atol=1e-8, rtol=1e-6)
    if kind == "rnn":
        np.testing.assert_allclose(custom.input_weight.grad, oracle.weight_ih_l0.grad.numpy().T)
        np.testing.assert_allclose(custom.hidden_weight.grad, oracle.weight_hh_l0.grad.numpy().T)
        np.testing.assert_allclose(custom.bias.grad, oracle.bias_ih_l0.grad.numpy())
    else:
        weights = oracle.weight.grad.numpy().T if kind == "dense" else oracle.weight.grad.numpy()
        np.testing.assert_allclose(custom.weight.grad, weights, atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(custom.bias.grad, oracle.bias.grad.numpy(), atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("kind", ["sgd", "momentum", "adam"])
def test_optimizer_parity(kind):
    custom = Parameter(np.array([1.0, 2.0, -3.0]))
    oracle = torch.nn.Parameter(torch.tensor(custom.data, dtype=torch.float64))
    manual = Optimizer({"parameter": custom}, kind, learning_rate=0.01)
    reference = (
        torch.optim.Adam([oracle], lr=0.01)
        if kind == "adam"
        else torch.optim.SGD([oracle], lr=0.01, momentum=0.9 if kind == "momentum" else 0)
    )
    for g in ([1.0, -2.0, 3.0], [-0.5, 2.0, 1.0]):
        custom.grad[:] = g
        oracle.grad = torch.tensor(g, dtype=torch.float64)
        manual.step()
        reference.step()
        np.testing.assert_allclose(custom.data, oracle.detach().numpy(), atol=1e-8, rtol=1e-6)
