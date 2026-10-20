import numpy as np
import pytest

from neuropulse.config import TrainConfig
from neuropulse.evaluation.metrics import (
    bootstrap_intervals,
    paired_rmse_difference,
    rul_metrics,
)


def test_hand_metrics_and_asymmetric_score():
    metrics = rul_metrics(np.array([7.0, 30.0]), np.array([20.0, 20.0]))
    assert metrics["mae_cycles"] == 11.5
    assert metrics["rmse_cycles"] == np.sqrt((169 + 100) / 2)
    assert np.isclose(metrics["nasa_score_sum"], 2 * (np.e - 1))
    assert (
        rul_metrics(np.array([30.0]), np.array([20.0]))["nasa_score_sum"]
        > rul_metrics(np.array([10.0]), np.array([20.0]))["nasa_score_sum"]
    )
    assert bootstrap_intervals(
        np.array([1.0, 2.0]), np.array([2.0, 4.0]), 20
    ) == bootstrap_intervals(np.array([1.0, 2.0]), np.array([2.0, 4.0]), 20)


def test_paired_engine_bootstrap_preserves_shared_truth_and_sign():
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    better = truth + 1
    worse = truth + 3
    result = paired_rmse_difference(better, worse, truth, draws=100)
    assert result["difference_cycles"] == pytest.approx(-2)
    assert result["engine_bootstrap_95"] == pytest.approx([-2, -2])
    with pytest.raises(ValueError):
        paired_rmse_difference(better[:-1], worse, truth)


@pytest.mark.parametrize("family", ["mlp", "cnn", "lstm", "cnn_lstm", "autoencoder"])
def test_shapes_finite_gradients_state_reset_and_checkpoint(family, tmp_path):
    torch = pytest.importorskip("torch")
    from neuropulse.models.temporal import WindowModel

    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = WindowModel(3, TrainConfig(family=family, window_length=10))
    x = torch.randn(4, 10, 3)
    y = model(x)
    assert y.shape == (x.shape if family == "autoencoder" else (4, 1))
    y.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    before = model(x).detach()
    model(torch.randn_like(x))
    torch.testing.assert_close(before, model(x))
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = WindowModel(3, TrainConfig(family=family, window_length=10))
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    torch.testing.assert_close(before, restored(x))
