"""Calibrate equal warning-cycle budgets on validation engines, then compare held-out peers."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from neuropulse.data.acquire import sha256
from neuropulse.data.prepare import verify_raw
from neuropulse.data.readers import Array, read_engines
from neuropulse.inference.bundles import ModelBundle
from neuropulse.training.train import json_write

BoolArray = NDArray[np.bool_]


def _sustained(flags: BoolArray, count: int) -> BoolArray:
    if count <= 0:
        raise ValueError("Positive persistence required")
    output = np.zeros(len(flags), dtype=bool)
    for index in range(count - 1, len(flags)):
        output[index] = bool(np.all(flags[index - count + 1 : index + 1]))
    return output


def _alerts(
    policy: str,
    rul: Array,
    scores: Array,
    rul_limit: float,
    anomaly_limit: float,
    persistence: int,
) -> BoolArray:
    low_rul = rul <= rul_limit
    anomaly = _sustained(scores > anomaly_limit, persistence)
    if policy == "rul_only":
        return low_rul
    if policy == "anomaly_only":
        return anomaly
    if policy == "combined":
        return np.asarray(low_rul | anomaly, dtype=np.bool_)
    raise ValueError("Unknown policy")


def run(
    root: Path = Path("."),
    config_path: Path = Path("configs/experiments/matched-alerts.yaml"),
) -> Path:
    config_bytes = (root / config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["schema_version"] != "1.0" or config["target_warning_cycle_fraction"] != 0.10:
        raise ValueError("Unsupported budget protocol")
    selected = json.loads((root / config["model_selection"]).read_text())
    models = root / "artifacts/models"
    rul_bundle = ModelBundle(models / selected["model_version"])
    ae_bundle = ModelBundle(models / selected["anomaly_model_version"])
    if not rul_bundle.compatible(ae_bundle):
        raise ValueError("Branches are incompatible")
    raw = root / "data/raw/cmapss"
    preparation = json.loads((root / "data/processed/FD001/manifest.json").read_text())
    if verify_raw(raw) != preparation["raw_hashes"]:
        raise ValueError("Raw/prepared hashes differ")
    engines = read_engines(raw / "train_FD001.txt")
    ids = sorted(preparation["validation_ids"])
    if len(ids) != 20:
        raise ValueError("Expected 20 validation engines for 10/10 comparison")
    width = rul_bundle.config.window_length
    sequences: dict[int, tuple[Array, Array, int]] = {}
    for engine_id in ids:
        rows = engines[engine_id]
        transformed = rul_bundle.scaler.transform(rows[:, 5:])
        windows = np.stack(
            [transformed[end - width + 1 : end + 1] for end in range(width - 1, len(rows))]
        )
        rul = np.maximum(0, rul_bundle.predict(windows)[:, 0])
        scores = np.mean((windows - ae_bundle.predict(windows)) ** 2, axis=(1, 2))
        sequences[engine_id] = (rul, scores, len(rows))
    calibration, evaluation = ids[:10], ids[10:]
    calib_rul = np.concatenate([sequences[i][0] for i in calibration])
    calib_scores = np.concatenate([sequences[i][1] for i in calibration])
    persistence = config["anomaly_persistence_cycles"]

    def rate(policy: str, rul_limit: float, anomaly_limit: float) -> float:
        count, total = 0, 0
        for engine_id in calibration:
            rul, scores, _ = sequences[engine_id]
            count += int(_alerts(policy, rul, scores, rul_limit, anomaly_limit, persistence).sum())
            total += len(rul)
        return count / total

    results: dict[str, Any] = {}
    grid = np.linspace(0, 1, config["quantile_grid_points"])
    for policy in ("rul_only", "anomaly_only", "combined"):
        candidates = []
        for quantile in grid:
            rul_limit = float(np.quantile(calib_rul, quantile))
            anomaly_limit = float(np.quantile(calib_scores, 1 - quantile))
            if policy == "rul_only":
                anomaly_limit = float("inf")
            if policy == "anomaly_only":
                rul_limit = -1.0
            candidate_rate = rate(policy, rul_limit, anomaly_limit)
            candidates.append(
                (
                    abs(candidate_rate - config["target_warning_cycle_fraction"]),
                    candidate_rate,
                    quantile,
                    rul_limit,
                    anomaly_limit,
                )
            )
        _, calibration_rate, quantile, rul_limit, anomaly_limit = min(candidates)
        warned, warning_cycles, first_warning_leads = 0, 0, []
        first_cycles: dict[str, int | None] = {}
        total_cycles = 0
        for engine_id in evaluation:
            rul, scores, lifetime = sequences[engine_id]
            flags = _alerts(policy, rul, scores, rul_limit, anomaly_limit, persistence)
            warning_cycles += int(flags.sum())
            total_cycles += len(flags)
            if flags.any():
                warned += 1
                cycle = int(np.flatnonzero(flags)[0]) + width
                first_cycles[str(engine_id)] = cycle
                first_warning_leads.append(lifetime - cycle)
            else:
                first_cycles[str(engine_id)] = None
        results[policy] = {
            "calibration_warning_cycle_fraction": calibration_rate,
            "evaluation_warning_cycle_fraction": warning_cycles / total_cycles,
            "quantile": float(quantile),
            "rul_limit_cycles": rul_limit if np.isfinite(rul_limit) else None,
            "anomaly_limit": anomaly_limit if np.isfinite(anomaly_limit) else None,
            "evaluation_engines_warned": warned,
            "evaluation_mean_first_warning_lead_cycles": (
                float(np.mean(first_warning_leads)) if first_warning_leads else None
            ),
            "evaluation_first_warning_cycles": first_cycles,
        }
    output = root / "results/studies/matched-alerts-validation-v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Study result already exists")
    json_write(
        output,
        {
            "schema_version": "1.0",
            "protocol_sha256": sha256(config_bytes),
            "population": "held-out FD001 training engines only; no official test data",
            "model_version": rul_bundle.manifest["model_version"],
            "anomaly_model_version": ae_bundle.manifest["model_version"],
            "calibration_ids": calibration,
            "evaluation_ids": evaluation,
            "target_warning_cycle_fraction": config["target_warning_cycle_fraction"],
            "comparison": "binary any-warning state; not severity or maintenance cost",
            "results": results,
            "limitations": [
                "Exploratory after official test; no model or primary suite re-selection",
                "Both groups participated in prior model selection; "
                "policy calibration uses only the first group",
                "Early-life proxy is not verified healthy or fault-onset ground truth",
                "Warning-cycle rates are matched on calibration trajectories, "
                "not guaranteed on evaluation",
                "Only ten evaluation engines and one selected model pair",
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
