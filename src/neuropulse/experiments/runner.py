"""Controlled three-seed suite; reuse equivalent completed runs and freeze before final test."""

import argparse
import json
import re
from pathlib import Path

from neuropulse.config import TrainConfig
from neuropulse.data.acquire import sha256
from neuropulse.training.train import json_write, run


def run_suite(name: str, jobs: list[TrainConfig], results: Path = Path("results")) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not jobs:
        raise ValueError("A safe suite name and nonempty job list are required")
    suite_path = results / "suites" / f"{name}.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    if suite_path.exists():
        raise ValueError("Suite name already exists; use a new name to preserve frozen evidence")
    entries = []
    json_write(suite_path, {"schema_version": "1.0", "name": name, "frozen": False, "runs": []})
    for config in jobs:
        candidates = []
        for path in sorted(results.glob("*/manifest.json")):
            meta = json.loads(path.read_text())
            if meta.get("status") == "complete" and meta.get("config") == config.model_dump():
                candidates.append(path.parent)
        output = candidates[-1] if candidates else run(config, results=results)
        meta = json.loads((output / "manifest.json").read_text())
        entries.append(
            {
                "run_id": output.name,
                "model_version": meta["model_version"],
                "config": config.model_dump(),
                "manifest_sha256": sha256((output / "manifest.json").read_bytes()),
            }
        )
        json_write(
            suite_path, {"schema_version": "1.0", "name": name, "frozen": False, "runs": entries}
        )
    json_write(
        suite_path,
        {
            "schema_version": "1.0",
            "name": name,
            "frozen": True,
            "selection": "validation only; uncapped truncated-prefix RMSE",
            "runs": entries,
        },
    )
    manifests = [
        json.loads((results / item["run_id"] / "manifest.json").read_text()) for item in entries
    ]
    regressors = [m for m in manifests if m["family"] != "autoencoder"]
    # Ancillary sensitivity suites must not replace the primary demo selection.
    if regressors and (name == "architecture-v1" or not (results / "deployment.json").exists()):
        chosen = min(regressors, key=lambda m: m["metrics"]["validation"]["rmse_cycles"])
        aes = [
            m
            for m in manifests
            if m["family"] == "autoencoder"
            and m["config"]["window_length"] == chosen["config"]["window_length"]
        ]
        json_write(
            results / "deployment.json",
            {
                "model_version": chosen["model_version"],
                "anomaly_model_version": aes[0]["model_version"] if aes else None,
                "selection": "lowest validation RMSE; no official test selection",
                "suite": name,
            },
        )
    print(f"FROZEN {suite_path}", flush=True)
    return suite_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--families", nargs="+", default=["linear", "random_forest", "mlp", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.suite:
        import yaml

        suite = yaml.safe_load(args.suite.read_text())
        if not isinstance(suite, dict) or set(suite) != {"name", "jobs"}:
            raise ValueError("Suite requires exactly name and jobs")
        run_suite(suite["name"], [TrainConfig.model_validate(job) for job in suite["jobs"]])
        return
    if not args.name:
        parser.error("--name or --suite required")
    run_suite(
        args.name,
        [
            TrainConfig.model_validate({"family": family, "seed": seed, "epochs": args.epochs})
            for family in args.families
            for seed in args.seeds
        ],
    )


if __name__ == "__main__":
    main()
