"""Educational manual-AE FD001 proxy experiment with validation-only calibration."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from neuropulse.data.acquire import sha256
from neuropulse.data.prepare import transformed, verify_raw
from neuropulse.data.preprocessing import SensorScaler
from neuropulse.data.readers import read_engines
from neuropulse.data.windows import WindowDataset
from neuropulse.models.numpy_mlp import build_autoencoder
from neuropulse.nn_numpy.losses import Loss
from neuropulse.nn_numpy.optimizers import Optimizer
from neuropulse.nn_numpy.serialization import save
from neuropulse.training.train import json_write


def _healthy_proxy(dataset: WindowDataset, lengths: dict[int, int]) -> WindowDataset:
    dataset.index = [
        (engine_id, end)
        for engine_id, end in dataset.index
        if end + 1 <= 50 and lengths[engine_id] - end - 1 >= 125
    ]
    if not dataset.index:
        raise ValueError("No complete early-life proxy windows")
    return dataset


def run(root: Path = Path("."), seed: int = 42, epochs: int = 100) -> Path:
    if epochs <= 0:
        raise ValueError("Positive epoch budget required")
    raw = root / "data/raw/cmapss"
    prepared = root / "data/processed/FD001"
    hashes = verify_raw(raw)
    manifest = json.loads((prepared / "manifest.json").read_text())
    if (
        manifest["raw_hashes"] != hashes
        or sha256((prepared / "scaler.npz").read_bytes()) != manifest["scaler_sha256"]
    ):
        raise ValueError("Prepared/raw data or scaler provenance mismatch")
    scaler = SensorScaler.load(prepared / "scaler.npz")
    engines = read_engines(raw / "train_FD001.txt")
    window = manifest["config"]["window_length"]
    train = _healthy_proxy(
        WindowDataset(transformed(engines, scaler, manifest["train_ids"]), window),
        {i: len(engines[i]) for i in manifest["train_ids"]},
    )
    validation = _healthy_proxy(
        WindowDataset(transformed(engines, scaler, manifest["validation_ids"]), window),
        {i: len(engines[i]) for i in manifest["validation_ids"]},
    )
    features = len(scaler.selected)
    model = build_autoencoder(features, window, seed)
    optimizer = Optimizer(model.named_parameters(), learning_rate=0.001)
    loss = Loss()
    rng = np.random.default_rng(seed)
    validation_inputs = np.stack([validation[i][0].ravel() for i in range(len(validation))])
    best, best_state, stale = float("inf"), model.state_dict(), 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for indices in train.batches(64, rng):
            x, _ = train.batch(indices)
            flattened = x.reshape(len(x), -1)
            optimizer.zero_grad()
            total += loss.forward(model.forward(flattened, training=True), flattened) * len(x)
            model.backward(loss.backward())
            optimizer.clip_grad()
            optimizer.step()
        validation_loss = float(
            np.mean((model.forward(validation_inputs) - validation_inputs) ** 2)
        )
        history.append((epoch, total / len(train), validation_loss))
        if validation_loss < best - 1e-8:
            best, best_state, stale = validation_loss, model.state_dict(), 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    errors = np.mean((model.forward(validation_inputs) - validation_inputs) ** 2, axis=1)
    threshold = float(np.quantile(errors, 0.99, method="linear"))
    output = root / f"results/manual-ae-fd001-seed{seed}"
    if output.exists():
        raise ValueError("Manual AE run already exists; preserve its artifacts")
    output.mkdir(parents=True)
    save(model, output / "weights")
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "training_reconstruction_mse", "validation_reconstruction_mse"])
        writer.writerows(history)
    json_write(
        output / "manifest.json",
        {
            "schema_version": "1.0",
            "family": "numpy_dense_autoencoder",
            "dataset": "cmapss",
            "subset": "FD001",
            "seed": seed,
            "window_length": window,
            "feature_order": manifest["selected_sensors"],
            "train_ids": manifest["train_ids"],
            "validation_ids": manifest["validation_ids"],
            "training_windows": len(train),
            "healthy_proxy_validation_windows": len(validation),
            "validation_reconstruction_mse": float(errors.mean()),
            "threshold_q99": threshold,
            "label_basis": "early_life_proxy_not_verified_healthy",
            "elapsed_seconds": time.perf_counter() - start,
            "raw_hashes": hashes,
            "scaler_sha256": manifest["scaler_sha256"],
            "weights_sha256": sha256((output / "weights.npz").read_bytes()),
            "selection": "lowest held-out healthy-proxy reconstruction MSE",
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    print(run(args.root, args.seed, args.epochs))


if __name__ == "__main__":
    main()
