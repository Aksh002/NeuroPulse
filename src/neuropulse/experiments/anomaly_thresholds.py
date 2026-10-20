"""Qualified AE threshold sensitivity on held-out early/late FD001 proxy windows."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neuropulse.data.acquire import sha256
from neuropulse.data.prepare import verify_raw
from neuropulse.data.readers import read_engines
from neuropulse.inference.bundles import ModelBundle
from neuropulse.training.train import json_write


def run(
    root: Path = Path("."),
    config_path: Path = Path("configs/experiments/anomaly-thresholds.yaml"),
) -> Path:
    protocol_bytes = (root / config_path).read_bytes()
    protocol = yaml.safe_load(protocol_bytes)
    if protocol["schema_version"] != "1.0" or protocol["quantiles"] != [0.95, 0.99, 0.995]:
        raise ValueError("Unsupported anomaly threshold protocol")
    selection = json.loads((root / protocol["model_selection"]).read_text())
    model = ModelBundle(root / "artifacts/models" / selection["anomaly_model_version"])
    if model.config.family != "autoencoder":
        raise ValueError("Selected anomaly bundle must be a dense AE")
    raw = root / "data/raw/cmapss"
    prepared = json.loads((root / "data/processed/FD001/manifest.json").read_text())
    if verify_raw(raw) != prepared["raw_hashes"]:
        raise ValueError("Raw/prepared hashes differ")
    engines = read_engines(raw / "train_FD001.txt")
    early, late = [], []
    width = model.config.window_length
    for engine_id in prepared["validation_ids"]:
        rows = engines[engine_id]
        transformed = model.scaler.transform(rows[:, 5:])
        windows = np.stack(
            [transformed[end - width + 1 : end + 1] for end in range(width - 1, len(rows))]
        )
        scores = np.mean((windows - model.predict(windows)) ** 2, axis=(1, 2))
        for offset, score in enumerate(scores):
            end_cycle = offset + width
            if end_cycle <= 50 and len(rows) - end_cycle >= 125:
                early.append(float(score))
            if end_cycle > len(rows) - 50:
                late.append(float(score))
    if not early or not late:
        raise ValueError("Proxy populations must both be nonempty")
    early_scores, late_scores = np.asarray(early), np.asarray(late)
    outcomes: dict[str, Any] = {}
    for quantile in protocol["quantiles"]:
        threshold = float(np.quantile(early_scores, quantile, method="linear"))
        outcomes[str(quantile)] = {
            "threshold": threshold,
            "early_proxy_exceedance_fraction": float(np.mean(early_scores > threshold)),
            "late_proxy_exceedance_fraction": float(np.mean(late_scores > threshold)),
        }
    if not np.isclose(outcomes["0.99"]["threshold"], model.manifest["anomaly_threshold"]):
        raise ValueError("Saved q99 calibration differs from reconstructed proxy scores")
    output = root / "results/studies/anomaly-thresholds-validation-v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Threshold study output already exists")
    json_write(
        output,
        {
            "schema_version": "1.0",
            "protocol_sha256": sha256(protocol_bytes),
            "model_version": model.manifest["model_version"],
            "population": "held-out FD001 training engines only; no official test data",
            "early_proxy_windows": len(early),
            "late_proxy_windows": len(late),
            "early_proxy_score_mean": float(early_scores.mean()),
            "late_proxy_score_mean": float(late_scores.mean()),
            "quantiles": outcomes,
            "limitations": [
                "Thresholds and early-proxy exceedance are measured on "
                "the same calibration windows",
                "Early/late windows are correlated within engines; fractions are descriptive",
                "Early life is not verified healthy and late life is not verified fault onset",
                "No anomaly classification accuracy or independent false-alarm rate is claimed",
            ],
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(run(args.root))


if __name__ == "__main__":
    main()
