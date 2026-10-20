"""Final official terminal evaluation requires a complete frozen validation-selected suite."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from neuropulse.data.acquire import sha256
from neuropulse.data.prepare import verify_raw
from neuropulse.data.readers import read_engines, read_test_labels
from neuropulse.data.windows import terminal_window
from neuropulse.evaluation.metrics import bootstrap_intervals, rul_metrics
from neuropulse.inference.bundles import ModelBundle
from neuropulse.training.train import json_write


def evaluate_suite(
    suite_path: Path,
    raw: Path = Path("data/raw/cmapss"),
    results: Path = Path("results"),
    artifacts: Path = Path("artifacts/models"),
) -> None:
    suite = json.loads(suite_path.read_text())
    if suite.get("frozen") is not True or not suite["runs"]:
        raise ValueError("A nonempty frozen suite is required before official test evaluation")
    verify_raw(raw)
    acquisition = json.loads((raw / "manifest.json").read_text())
    label_file = raw / "RUL_FD001.txt"
    if sha256(label_file.read_bytes()) != acquisition["files"]["RUL_FD001.txt"]["sha256"]:
        raise ValueError("Official label checksum mismatch")
    engines = read_engines(raw / "test_FD001.txt")
    ids = sorted(engines)
    truth = read_test_labels(label_file, ids)
    for entry in suite["runs"]:
        output = results / entry["run_id"]
        if sha256((output / "manifest.json").read_bytes()) != entry["manifest_sha256"]:
            raise ValueError("Frozen run manifest changed")
        bundle = ModelBundle(artifacts / entry["model_version"])
        if bundle.config.family == "autoencoder":
            continue
        if bundle.manifest["raw_hashes"] != verify_raw(raw):
            raise ValueError("Model/evaluation data hashes differ")
        pairs = [
            terminal_window(bundle.scaler.transform(engines[i][:, 5:]), bundle.config.window_length)
            for i in ids
        ]
        windows = np.stack([x for x, _ in pairs])
        raw_predictions = bundle.predict(windows)[:, 0]
        prediction = np.maximum(0, raw_predictions)
        target = output / "final_test"
        target.mkdir(exist_ok=True)
        json_write(
            target / "metrics.json",
            {
                "population": "official_test_terminal_100_engines",
                "label_policy": "uncapped official terminal RUL",
                "metrics": rul_metrics(prediction, truth),
                "raw_metrics": rul_metrics(raw_predictions, truth),
                "bootstrap_95": bootstrap_intervals(prediction, truth),
                "padding_count": sum(p for _, p in pairs),
                "frozen_suite": suite_path.name,
            },
        )
        with (target / "predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["engine_id", "raw_rul_cycles", "rul_cycles", "true_rul_cycles"])
            writer.writerows(zip(ids, raw_predictions, prediction, truth, strict=True))
        print(f"FINAL {entry['run_id']}: {rul_metrics(prediction, truth)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--final-test", action="store_true", required=True)
    args = parser.parse_args()
    evaluate_suite(args.suite)


if __name__ == "__main__":
    main()
