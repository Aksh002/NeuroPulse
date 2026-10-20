"""Explicit uncapped cycle metrics and engine-population bootstrap uncertainty."""

import numpy as np

from neuropulse.data.readers import Array


def rul_metrics(prediction: Array, truth: Array) -> dict[str, float | int]:
    if prediction.shape != truth.shape or not truth.size:
        raise ValueError("Matching nonempty metric vectors required")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite metric inputs")
    d = prediction - truth
    with np.errstate(over="raise"):
        score = np.expm1(np.where(d < 0, -d / 13.0, d / 10.0))
    return {
        "mae_cycles": float(np.abs(d).mean()),
        "rmse_cycles": float(np.sqrt((d**2).mean())),
        "nasa_score_sum": float(score.sum()),
        "nasa_score_mean": float(score.mean()),
        "engines": int(truth.size),
    }


def bootstrap_intervals(
    prediction: Array, truth: Array, draws: int = 2000, seed: int = 2026
) -> dict[str, list[float]]:
    rul_metrics(prediction, truth)
    if prediction.ndim != 1:
        raise ValueError("Bootstrap samples one prediction per independent engine")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"mae_cycles": [], "rmse_cycles": []}
    for _ in range(draws):
        indices = rng.integers(0, len(truth), len(truth))
        result = rul_metrics(prediction[indices], truth[indices])
        for key in samples:
            samples[key].append(float(result[key]))
    return {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in samples.items()}


def paired_rmse_difference(
    left: Array, right: Array, truth: Array, draws: int = 2000, seed: int = 2026
) -> dict[str, float | list[float]]:
    """Engine-paired RMSE(left)-RMSE(right); interval resamples engines, not windows."""
    rul_metrics(left, truth)
    rul_metrics(right, truth)
    if left.ndim != 1 or draws <= 0:
        raise ValueError("One prediction per engine and positive bootstrap draws required")
    point = float(np.sqrt(np.mean((left - truth) ** 2)) - np.sqrt(np.mean((right - truth) ** 2)))
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(draws):
        index = rng.integers(0, len(truth), len(truth))
        sampled.append(
            float(
                np.sqrt(np.mean((left[index] - truth[index]) ** 2))
                - np.sqrt(np.mean((right[index] - truth[index]) ** 2))
            )
        )
    return {
        "difference_cycles": point,
        "engine_bootstrap_95": np.quantile(sampled, [0.025, 0.975]).tolist(),
    }
