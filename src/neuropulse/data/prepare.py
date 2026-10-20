"""Prepare FD001 with verified provenance and no official test-label reads."""

import argparse
import json
from pathlib import Path

import numpy as np

from neuropulse.config import load_data_config
from neuropulse.data.acquire import atomic_write, sha256
from neuropulse.data.preprocessing import SensorScaler
from neuropulse.data.readers import SENSOR_NAMES, Engines, read_engines
from neuropulse.data.splits import split_engines, validation_prefixes


def verify_raw(raw: Path) -> dict[str, str]:
    manifest = json.loads((raw / "manifest.json").read_text())
    hashes: dict[str, str] = {}
    for name in ("train_FD001.txt", "test_FD001.txt"):
        actual = sha256((raw / name).read_bytes())
        if actual != manifest["files"][name]["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {name}")
        hashes[name] = actual
    return hashes


def transformed(engines: Engines, scaler: SensorScaler, ids: list[int]) -> Engines:
    return {
        i: np.concatenate([engines[i][:, :5], scaler.transform(engines[i][:, 5:])], axis=1)
        for i in ids
    }


def prepare(config_path: Path, raw: Path, output: Path) -> dict[str, object]:
    config = load_data_config(config_path)
    hashes = verify_raw(raw)
    engines = read_engines(raw / "train_FD001.txt")
    # Official test observations are structurally checked, never used to fit a transform.
    test_engines = read_engines(raw / "test_FD001.txt")
    train, validation = split_engines(engines, config.train_fraction, config.split_seed)
    endpoints = validation_prefixes(engines, validation, config.validation_prefix_seed)
    scaler = SensorScaler.fit(
        engines, train, config.sensor_variance_threshold, config.scaler_std_floor
    )
    training_rows = np.concatenate([engines[i] for i in train])
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "config": config.model_dump(),
        "raw_hashes": hashes,
        "train_ids": train,
        "validation_ids": validation,
        "validation_prefixes": endpoints,
        "test_ids": sorted(test_engines),
        "official_test_labels_read": False,
        "row_counts": {
            "train": sum(len(engines[i]) for i in train),
            "validation": sum(len(engines[i]) for i in validation),
            "test": sum(len(r) for r in test_engines.values()),
        },
        "training_sensor_statistics": {
            name: {
                "mean": float(training_rows[:, 5 + j].mean()),
                "std": float(training_rows[:, 5 + j].std()),
                "min": float(training_rows[:, 5 + j].min()),
                "max": float(training_rows[:, 5 + j].max()),
            }
            for j, name in enumerate(SENSOR_NAMES)
        },
        "selected_sensors": [SENSOR_NAMES[i] for i in scaler.selected],
        "trajectory_lengths": {str(i): len(rows) for i, rows in engines.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    scaler.save(output / "scaler.npz")
    summary["scaler_sha256"] = sha256((output / "scaler.npz").read_bytes())
    atomic_write(output / "manifest.json", json.dumps(summary, indent=2).encode())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/fd001.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/cmapss"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/FD001"))
    args = parser.parse_args()
    summary = prepare(args.config, args.raw, args.output)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("row_counts", "selected_sensors", "official_test_labels_read")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
