"""Validation-only noise and demonstration-rule diagnostics; never use official test labels."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neuropulse.data.acquire import sha256
from neuropulse.data.prepare import verify_raw
from neuropulse.data.readers import read_engines
from neuropulse.data.windows import terminal_window
from neuropulse.decisions.rules import DemoRules, RuleState
from neuropulse.evaluation.metrics import rul_metrics
from neuropulse.inference.bundles import ModelBundle
from neuropulse.training.train import json_write


def _summary(
    first: list[int | None], lifetimes: list[int], transitions: list[int]
) -> dict[str, Any]:
    leads = [end - cycle for end, cycle in zip(lifetimes, first, strict=True) if cycle is not None]
    return {
        "engines": len(lifetimes),
        "engines_warned": len(leads),
        "mean_first_warning_lead_cycles": float(np.mean(leads)) if leads else None,
        "median_first_warning_lead_cycles": float(np.median(leads)) if leads else None,
        "alert_state_transitions": sum(transitions),
    }


def run(
    root: Path = Path("."),
    config_path: Path = Path("configs/experiments/exploratory.yaml"),
) -> Path:
    config = yaml.safe_load((root / config_path).read_text())
    if (
        config["schema_version"] != "1.0"
        or config["decision_comparison"] != "unmatched demonstration thresholds"
    ):
        raise ValueError("Unsupported exploratory protocol")
    selected = json.loads((root / config["model_selection"]).read_text())
    models = root / "artifacts/models"
    rul = ModelBundle(models / selected["model_version"])
    anomaly = ModelBundle(models / selected["anomaly_model_version"])
    if not rul.compatible(anomaly):
        raise ValueError("Selected model branches are incompatible")
    preparation = json.loads((root / "data/processed/FD001/manifest.json").read_text())
    if verify_raw(root / "data/raw/cmapss") != preparation["raw_hashes"]:
        raise ValueError("Raw data changed after preparation")
    engines = read_engines(root / "data/raw/cmapss/train_FD001.txt")
    ids = preparation["validation_ids"]
    endpoints = {int(k): v for k, v in preparation["validation_prefixes"].items()}
    rng = np.random.default_rng(config["noise_seed"])
    rules = DemoRules(**yaml.safe_load((root / config["decision_threshold_source"]).read_text()))
    threshold = anomaly.manifest["anomaly_threshold"]
    clean_predictions = []
    noisy: dict[str, list[float]] = {str(s): [] for s in config["standardized_noise_sigma"]}
    truth = []
    first: dict[str, list[int | None]] = {
        name: [] for name in ("rul_only", "anomaly_only", "combined")
    }
    transitions: dict[str, list[int]] = {name: [] for name in first}
    lifetimes = []
    for engine_id in ids:
        rows = engines[engine_id]
        transformed = rul.scaler.transform(rows[:, 5:])
        endpoint = endpoints[engine_id]
        terminal, _ = terminal_window(transformed[:endpoint], rul.config.window_length)
        clean_predictions.append(float(max(0, rul.predict(terminal[None, :, :])[0, 0])))
        truth.append(float(len(rows) - endpoint))
        for sigma in config["standardized_noise_sigma"]:
            noisy_window = terminal + rng.normal(0, sigma, terminal.shape)
            noisy[str(sigma)].append(float(max(0, rul.predict(noisy_window[None, :, :])[0, 0])))
        width = rul.config.window_length
        windows = np.stack(
            [transformed[end - width + 1 : end + 1] for end in range(width - 1, len(rows))]
        )
        rul_values = np.maximum(0, rul.predict(windows)[:, 0])
        reconstruction = anomaly.predict(windows)
        scores = np.mean((windows - reconstruction) ** 2, axis=(1, 2))
        states = {"rul_only": RuleState(), "combined": RuleState()}
        sustained_count = 0
        previous = {name: "normal" for name in first}
        initial: dict[str, int | None] = {name: None for name in first}
        counts = {name: 0 for name in first}
        for offset, (prediction, score) in enumerate(zip(rul_values, scores, strict=True)):
            cycle = offset + width
            flag = bool(score > threshold)
            only, _, states["rul_only"] = rules.apply(
                float(prediction), cycle, None, states["rul_only"]
            )
            combined, _, states["combined"] = rules.apply(
                float(prediction), cycle, flag, states["combined"]
            )
            sustained_count = sustained_count + 1 if flag else 0
            anomaly_only = "monitor" if sustained_count >= rules.persistence else "normal"
            for name, recommendation in (
                ("rul_only", only),
                ("anomaly_only", anomaly_only),
                ("combined", combined),
            ):
                if recommendation != "normal" and initial[name] is None:
                    initial[name] = cycle
                if recommendation != previous[name]:
                    counts[name] += 1
                previous[name] = recommendation
        lifetimes.append(len(rows))
        for name in first:
            first[name].append(initial[name])
            transitions[name].append(counts[name])
    target = np.asarray(truth)
    clean = np.asarray(clean_predictions)
    report = {
        "schema_version": "1.0",
        "protocol_sha256": sha256((root / config_path).read_bytes()),
        "population": "20 held-out FD001 training engines; validation data only",
        "model_version": rul.manifest["model_version"],
        "anomaly_model_version": anomaly.manifest["model_version"],
        "noise": {
            "clean": rul_metrics(clean, target),
            **{key: rul_metrics(np.asarray(values), target) for key, values in noisy.items()},
        },
        "decisions": {name: _summary(first[name], lifetimes, transitions[name]) for name in first},
        "limitations": [
            "Exploratory post-official-test study; no official labels or observations used here",
            "Decision policies have unmatched alert budgets and different severities",
            "AE threshold was calibrated on healthy-proxy windows from this validation population",
            "Early-life proxy is not verified fault-onset ground truth",
            "Noise is Gaussian in standardized sensor space, not validated physical noise",
        ],
    }
    output = root / "results/studies/exploratory-validation-v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Study output already exists; preserve the original result")
    json_write(output, report)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(run(args.root))


if __name__ == "__main__":
    main()
