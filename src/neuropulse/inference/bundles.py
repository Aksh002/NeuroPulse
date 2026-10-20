"""Verified trusted-local model adapters with explicit scientific compatibility checks."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from neuropulse.config import TrainConfig
from neuropulse.contracts import validate_sensor_windows
from neuropulse.data.acquire import sha256
from neuropulse.data.preprocessing import SensorScaler
from neuropulse.data.readers import SENSOR_NAMES, Array


class ModelBundle:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        meta = self.manifest
        if (
            meta["schema_version"] != "1.0"
            or meta["dataset"] != "cmapss"
            or meta["subset"] != "FD001"
        ):
            raise ValueError("Unsupported artifact schema/dataset")
        self.config = TrainConfig.model_validate(meta["config"])
        allowed = {"weights.pt", "weights.npz", "weights.json", "weights.joblib", "scaler.npz"}
        if not set(meta["checksums"]) <= allowed or meta["weights"] not in meta["checksums"]:
            raise ValueError("Unexpected artifact filenames")
        for name, expected in meta["checksums"].items():
            if sha256((directory / name).read_bytes()) != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        self.scaler = SensorScaler.load(directory / "scaler.npz")
        if meta["features"] != [SENSOR_NAMES[i] for i in self.scaler.selected]:
            raise ValueError("Artifact feature order mismatch")
        if list(self.scaler.training_ids) != meta["training_ids"]:
            raise ValueError("Artifact scaler population mismatch")
        self.model: Any
        if meta["backend"] == "numpy" and self.config.family == "numpy_mlp":
            from neuropulse.nn_numpy.serialization import load

            self.model, _ = load(directory / "weights")
        elif meta["backend"] == "torch":
            import torch

            from neuropulse.models.temporal import WindowModel

            self.model = WindowModel(len(self.scaler.selected), self.config)
            self.model.load_state_dict(
                torch.load(directory / meta["weights"], map_location="cpu", weights_only=True)
            )
            self.model.eval()
        elif meta["backend"] == "sklearn" and self.config.family in ("linear", "random_forest"):
            import joblib

            # Only manifests generated locally and loaded from the configured trusted root.
            self.model = joblib.load(directory / meta["weights"])
        else:
            raise ValueError("Backend/family mismatch")

    def predict(self, windows: Array) -> Array:
        validate_sensor_windows(
            windows,
            window_length=self.config.window_length,
            feature_count=len(self.scaler.selected),
        )
        backend = self.manifest["backend"]
        if backend == "torch":
            import torch

            with torch.no_grad():
                result = self.model(torch.from_numpy(windows.astype(np.float32))).numpy()
        else:
            flattened = windows.reshape(len(windows), -1)
            result = (
                self.model.forward(flattened)
                if backend == "numpy"
                else self.model.predict(flattened).reshape(-1, 1)
            )
        result = np.asarray(result, dtype=np.float64)
        expected = windows.shape if self.config.family == "autoencoder" else (len(windows), 1)
        if result.shape != expected or not np.isfinite(result).all():
            raise ValueError("Nonfinite or malformed model output")
        return result

    def compatible(self, anomaly: "ModelBundle") -> bool:
        return (
            anomaly.config.family == "autoencoder"
            and self.config.window_length == anomaly.config.window_length
            and self.manifest["features"] == anomaly.manifest["features"]
            and self.manifest["checksums"]["scaler.npz"]
            == anomaly.manifest["checksums"]["scaler.npz"]
        )
