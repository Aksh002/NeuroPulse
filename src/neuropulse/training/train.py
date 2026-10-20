"""Reproducible FD001 training, validation-only checkpoint selection and immutable export."""

import argparse
import copy
import csv
import json
import platform
import random
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neuropulse.config import TrainConfig, load_train_config
from neuropulse.data.acquire import sha256
from neuropulse.data.prepare import transformed, verify_raw
from neuropulse.data.preprocessing import SensorScaler
from neuropulse.data.readers import Array, read_engines
from neuropulse.data.windows import WindowDataset, terminal_window
from neuropulse.evaluation.metrics import bootstrap_intervals, rul_metrics


def json_write(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")


def run(
    config: TrainConfig,
    raw: Path = Path("data/raw/cmapss"),
    processed: Path = Path("data/processed/FD001"),
    results: Path = Path("results"),
    artifacts: Path = Path("artifacts/models"),
) -> Path:
    # BLAS thread limits must be set before starting this process; torch is constrained below.
    random.seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    data_hashes = verify_raw(raw)
    prep = json.loads((processed / "manifest.json").read_text())
    if data_hashes != prep["raw_hashes"]:
        raise ValueError("Prepared/raw data provenance mismatch")
    if sha256((processed / "scaler.npz").read_bytes()) != prep["scaler_sha256"]:
        raise ValueError("Prepared scaler checksum mismatch")
    scaler = SensorScaler.load(processed / "scaler.npz")
    if list(scaler.training_ids) != prep["train_ids"]:
        raise ValueError("Scaler training population mismatch")
    engines = read_engines(raw / "train_FD001.txt")
    train_engines = transformed(engines, scaler, prep["train_ids"])
    dataset = WindowDataset(train_engines, config.window_length, cap=config.rul_cap_cycles)
    if not len(dataset):
        raise ValueError("No training windows")
    validation_ids = prep["validation_ids"]
    validation_endpoints = {int(k): v for k, v in prep["validation_prefixes"].items()}
    validation_windows, padded = zip(
        *[
            terminal_window(
                scaler.transform(engines[i][: validation_endpoints[i], 5:]), config.window_length
            )
            for i in validation_ids
        ],
        strict=True,
    )
    vx = np.stack(validation_windows)
    vy = np.array(
        [len(engines[i]) - validation_endpoints[i] for i in validation_ids], dtype=np.float64
    )
    validation_healthy: Array | None = None
    if config.family == "autoencoder":
        dataset.index = [
            (i, end)
            for i, end in dataset.index
            if end + 1 <= 50 and len(train_engines[i]) - end - 1 >= 125
        ]
        healthy = WindowDataset(transformed(engines, scaler, validation_ids), config.window_length)
        healthy.index = [
            (i, end)
            for i, end in healthy.index
            if end + 1 <= 50 and len(engines[i]) - end - 1 >= 125
        ]
        if not len(dataset) or not len(healthy):
            raise ValueError("Healthy proxy has no complete windows")
        validation_healthy = np.stack([healthy[j][0] for j in range(len(healthy))])
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    config_hash = sha256(json.dumps(config.model_dump(), sort_keys=True).encode())
    run_id = (
        f"cmapss_fd001_{config.family}_w{config.window_length:03d}_"
        f"seed{config.seed:03d}_{timestamp}_{config_hash[:8]}"
    )
    output = results / run_id
    output.mkdir(parents=True)
    import yaml

    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump()), encoding="utf-8"
    )
    json_write(
        output / "splits.json", {"train_ids": prep["train_ids"], "validation_ids": validation_ids}
    )
    json_write(output / "validation_prefixes.json", validation_endpoints)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        )
    except OSError:
        revision, dirty = "", True
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "source_revision": revision or "unversioned",
        "dirty": dirty,
        "threads": 2,
        "dataset_hashes": data_hashes,
        "config_hash": config_hash,
    }
    json_write(output / "environment.json", environment)
    json_write(
        output / "manifest.json",
        {
            "run_id": run_id,
            "status": "running",
            "family": config.family,
            "config": config.model_dump(),
        },
    )
    start = time.perf_counter()
    history: list[dict[str, float | int]] = []
    predictor: Any = None
    optimizer: Any
    predict: Callable[[Array], Array]
    try:
        if config.family in ("linear", "random_forest"):
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression

            predictor = (
                LinearRegression(n_jobs=2)
                if config.family == "linear"
                else RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=2,
                    random_state=config.seed,
                    n_jobs=2,
                )
            )
            x = np.stack([dataset[i][0].ravel() for i in range(len(dataset))])
            y = np.array([dataset[i][1][0] for i in range(len(dataset))])
            predictor.fit(x, y)

            def sklearn_predict(batch: Array) -> Array:
                return np.asarray(
                    predictor.predict(batch.reshape(len(batch), -1)), dtype=np.float64
                ).reshape(-1, 1)

            predict = sklearn_predict
            import joblib

            joblib.dump(predictor, output / "weights.joblib")
            backend, weights = "sklearn", "weights.joblib"
        elif config.family == "numpy_mlp":
            from neuropulse.models.numpy_mlp import build_mlp
            from neuropulse.nn_numpy.losses import Loss
            from neuropulse.nn_numpy.optimizers import Optimizer
            from neuropulse.nn_numpy.serialization import save

            predictor = build_mlp(len(scaler.selected), config)
            optimizer = Optimizer(
                predictor.named_parameters(), config.optimizer, config.learning_rate
            )
            loss, best, best_state, stale = Loss(), float("inf"), predictor.state_dict(), 0
            for epoch in range(1, config.epochs + 1):
                epoch_start, total = time.perf_counter(), 0.0
                for indices in dataset.batches(config.batch_size, rng):
                    x, y = dataset.batch(indices)
                    optimizer.zero_grad()
                    prediction = predictor.forward(x.reshape(len(x), -1), training=True)
                    total += loss.forward(prediction, y) * len(x)
                    predictor.backward(loss.backward())
                    optimizer.clip_grad()
                    optimizer.step()
                error = rul_metrics(
                    np.maximum(0, predictor.forward(vx.reshape(len(vx), -1))[:, 0]), vy
                )
                metric = float(error["rmse_cycles"])
                history.append(
                    {
                        "epoch": epoch,
                        "training_loss": total / len(dataset),
                        "validation_rmse": metric,
                        "epoch_seconds": time.perf_counter() - epoch_start,
                    }
                )
                print(f"{run_id} epoch={epoch} val_RMSE={metric:.3f}", flush=True)
                if metric < best - 1e-4:
                    best, best_state, stale = metric, predictor.state_dict(), 0
                else:
                    stale += 1
                save(
                    predictor,
                    output / "last",
                    {
                        "optimizer": optimizer.state_dict(),
                        "rng": rng.bit_generator.state,
                        "epoch": epoch,
                    },
                )
                if stale >= config.patience:
                    break
            predictor.load_state_dict(best_state)
            save(predictor, output / "weights")

            def numpy_predict(batch: Array) -> Array:
                return np.asarray(
                    predictor.forward(batch.reshape(len(batch), -1)), dtype=np.float64
                )

            predict = numpy_predict
            backend, weights = "numpy", "weights.npz"
        else:
            import torch

            from neuropulse.models.temporal import WindowModel

            torch.set_num_threads(2)
            torch.manual_seed(config.seed)
            torch.use_deterministic_algorithms(True)
            predictor = WindowModel(len(scaler.selected), config)
            if config.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    predictor.parameters(),
                    lr=config.learning_rate,
                    weight_decay=0 if config.family == "autoencoder" else config.weight_decay,
                )
            else:
                optimizer = torch.optim.SGD(
                    predictor.parameters(),
                    lr=config.learning_rate,
                    momentum=0.9 if config.optimizer == "momentum" else 0,
                    weight_decay=0 if config.family == "autoencoder" else config.weight_decay,
                )
            best, best_state, stale = float("inf"), copy.deepcopy(predictor.state_dict()), 0

            def torch_predict(batch: Array) -> Array:
                predictor.eval()
                with torch.no_grad():
                    parts = [
                        predictor(torch.from_numpy(batch[s : s + 256].astype(np.float32))).numpy()
                        for s in range(0, len(batch), 256)
                    ]
                    return np.asarray(np.concatenate(parts), dtype=np.float64)

            predict = torch_predict
            for epoch in range(1, config.epochs + 1):
                epoch_start, total = time.perf_counter(), 0.0
                predictor.train()
                for indices in dataset.batches(config.batch_size, rng):
                    x, y = dataset.batch(indices)
                    tx = torch.from_numpy(x.astype(np.float32))
                    ty = (
                        tx
                        if config.family == "autoencoder"
                        else torch.from_numpy(y.astype(np.float32))
                    )
                    optimizer.zero_grad()
                    loss_value = torch.nn.functional.mse_loss(predictor(tx), ty)
                    if not torch.isfinite(loss_value):
                        raise ValueError("Nonfinite training loss")
                    torch.autograd.backward(loss_value)
                    torch.nn.utils.clip_grad_norm_(
                        predictor.parameters(), 5, error_if_nonfinite=True
                    )
                    optimizer.step()
                    total += float(loss_value.detach()) * len(x)
                if validation_healthy is not None:
                    metric = float(((predict(validation_healthy) - validation_healthy) ** 2).mean())
                else:
                    metric = float(rul_metrics(np.maximum(0, predict(vx)[:, 0]), vy)["rmse_cycles"])
                history.append(
                    {
                        "epoch": epoch,
                        "training_loss": total / len(dataset),
                        "validation_rmse"
                        if validation_healthy is None
                        else "validation_reconstruction_mse": metric,
                        "epoch_seconds": time.perf_counter() - epoch_start,
                    }
                )
                print(f"{run_id} epoch={epoch} val_metric={metric:.4f}", flush=True)
                if metric < best - (1e-4 if validation_healthy is None else 1e-8):
                    best, best_state, stale = metric, copy.deepcopy(predictor.state_dict()), 0
                else:
                    stale += 1
                torch.save(
                    {
                        "weights": predictor.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "numpy_rng": rng.bit_generator.state,
                        "config": config.model_dump(),
                        "epoch": epoch,
                    },
                    output / "last.pt",
                )
                if stale >= config.patience:
                    break
            predictor.load_state_dict(best_state)
            torch.save(predictor.state_dict(), output / "weights.pt")
            environment["torch"] = torch.__version__
            backend, weights = "torch", "weights.pt"
        elapsed = time.perf_counter() - start
        metrics: dict[str, Any]
        if validation_healthy is not None:
            score = ((predict(validation_healthy) - validation_healthy) ** 2).mean(axis=(1, 2))
            threshold = float(np.quantile(score, 0.99, method="linear"))
            metrics = {
                "healthy_proxy_validation_mse": float(score.mean()),
                "threshold": threshold,
                "healthy_proxy_validation_windows": len(score),
                "healthy_proxy_train_windows": len(dataset),
                "label_basis": "early_life_proxy",
            }
        else:
            raw_predictions = predict(vx)[:, 0]
            postprocessed = np.maximum(0, raw_predictions)
            metrics = {
                "population": "held_out_training_engine_truncated_prefixes",
                "target_policy": "uncapped_truth",
                "validation": rul_metrics(postprocessed, vy),
                "raw_validation": rul_metrics(raw_predictions, vy),
                "bootstrap_95": bootstrap_intervals(postprocessed, vy),
                "padding_count": sum(padded),
                "negative_output_count": int((raw_predictions < 0).sum()),
            }
            columns = [
                "engine_id",
                "cycle",
                "raw_rul_cycles",
                "rul_cycles",
                "true_rul_cycles",
                "padding_applied",
            ]
            with (output / "predictions.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(
                    (
                        i,
                        validation_endpoints[i],
                        raw_predictions[j],
                        postprocessed[j],
                        vy[j],
                        padded[j],
                    )
                    for j, i in enumerate(validation_ids)
                )
            metrics["prediction_file"] = "predictions.csv"
            import importlib.util

            if importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"):
                import pandas as pd

                pd.read_csv(output / "predictions.csv").to_parquet(
                    output / "predictions.parquet", index=False
                )
        if history:
            with (output / "history.csv").open("w", newline="") as stream:
                history_writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                history_writer.writeheader()
                history_writer.writerows(history)
        json_write(output / "metrics.json", metrics)
        json_write(output / "environment.json", environment)
        json_write(
            output / "timing.json",
            {
                "training_seconds": elapsed,
                "epochs_completed": len(history),
                "training_windows": len(dataset),
            },
        )
        weights_hash = sha256((output / weights).read_bytes())
        model_hash = sha256((weights_hash + config_hash + prep["scaler_sha256"]).encode())
        model_version = f"{config.family}-fd001-v1-{model_hash[:8]}"
        bundle = artifacts / model_version
        bundle.mkdir(parents=True, exist_ok=True)
        filenames = [weights, "scaler.npz"] + (["weights.json"] if backend == "numpy" else [])
        shutil.copy2(processed / "scaler.npz", output / "scaler.npz")
        for filename in filenames:
            destination = bundle / filename
            content = (output / filename).read_bytes()
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError("Immutable model bundle collision")
            destination.write_bytes(content)
        manifest = {
            "schema_version": "1.0",
            "model_version": model_version,
            "run_id": run_id,
            "family": config.family,
            "backend": backend,
            "dataset": "cmapss",
            "subset": "FD001",
            "config": config.model_dump(),
            "features": prep["selected_sensors"],
            "training_ids": prep["train_ids"],
            "validation_ids": validation_ids,
            "validation_prefixes": validation_endpoints,
            "raw_hashes": data_hashes,
            "postprocessing": "max(0, raw); no upper cap",
            "weights": weights,
            "checksums": {f: sha256((bundle / f).read_bytes()) for f in filenames},
            "metrics": metrics,
            "environment": environment,
            "anomaly_threshold": metrics.get("threshold"),
        }
        if not (bundle / "manifest.json").exists():
            json_write(bundle / "manifest.json", manifest)
        json_write(output / "manifest.json", {**manifest, "status": "complete"})
        print(f"COMPLETE {run_id} -> {model_version}", flush=True)
        return output
    except Exception as exc:
        json_write(
            output / "manifest.json",
            {
                "run_id": run_id,
                "status": "failed",
                "config": config.model_dump(),
                "error": str(exc),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--family",
        choices=[
            "linear",
            "random_forest",
            "numpy_mlp",
            "mlp",
            "cnn",
            "lstm",
            "cnn_lstm",
            "autoencoder",
        ],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = load_train_config(args.config) if args.config else TrainConfig()
    overrides = {
        key: getattr(args, key)
        for key in ("family", "seed", "epochs")
        if getattr(args, key) is not None
    }
    run(TrainConfig.model_validate({**config.model_dump(), **overrides}))


if __name__ == "__main__":
    main()
