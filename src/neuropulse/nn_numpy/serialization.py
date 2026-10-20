"""NPZ/JSON checkpoints, with no pickle or dynamically executed architecture."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from neuropulse.nn_numpy.activations import Activation
from neuropulse.nn_numpy.layers import Dense, Sequential


def save(model: Sequential, path: Path, extra: dict[str, Any] | None = None) -> None:
    layers = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            layers.append(
                {
                    "kind": "dense",
                    "inputs": layer.weight.data.shape[0],
                    "outputs": layer.weight.data.shape[1],
                }
            )
        elif isinstance(layer, Activation):
            layers.append({"kind": layer.kind})
        else:
            raise ValueError("Unsupported serialized layer")
    np.savez(path.with_suffix(".npz"), allow_pickle=False, **model.state_dict())
    path.with_suffix(".json").write_text(
        json.dumps({"schema_version": "1.0", "layers": layers, "extra": extra or {}}, indent=2),
        encoding="utf-8",
    )


def load(path: Path) -> tuple[Sequential, dict[str, Any]]:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["schema_version"] != "1.0":
        raise ValueError("Unsupported checkpoint schema")
    rng = np.random.default_rng(0)
    layers = [
        Dense(s["inputs"], s["outputs"], rng) if s["kind"] == "dense" else Activation(s["kind"])
        for s in meta["layers"]
    ]
    model = Sequential(*layers)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        model.load_state_dict({k: arrays[k] for k in arrays.files})
    return model, dict(meta["extra"])
