"""Render the frozen FD001 experiments into a reproducible, qualified Markdown report."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from neuropulse.data.acquire import sha256
from neuropulse.evaluation.metrics import paired_rmse_difference


def _runs(suite_path: Path, results: Path) -> list[tuple[dict[str, Any], Path]]:
    suite = json.loads(suite_path.read_text())
    if suite.get("frozen") is not True:
        raise ValueError(f"Suite is not frozen: {suite_path}")
    runs = []
    for item in suite["runs"]:
        directory = results / item["run_id"]
        source = (directory / "manifest.json").read_bytes()
        if sha256(source) != item["manifest_sha256"]:
            raise ValueError(f"Run manifest changed: {item['run_id']}")
        runs.append((json.loads(source), directory))
    return runs


def render(results: Path = Path("results")) -> str:
    primary = _runs(results / "suites/architecture-v1.json", results)
    sensitivity = _runs(results / "suites/sensitivity-v1.json", results)
    extra_windows = _runs(
        results / "suites/temporal-windows-exploratory-v1.json", results
    )
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    ae = []
    for manifest, directory in primary:
        if manifest["family"] == "autoencoder":
            ae.append(manifest)
            continue
        final = json.loads((directory / "final_test/metrics.json").read_text())
        groups[manifest["family"]].append((manifest, final))
    if set(groups) != {
        "linear",
        "random_forest",
        "numpy_mlp",
        "mlp",
        "lstm",
        "cnn",
        "cnn_lstm",
    } or any(len(entries) != 3 for entries in groups.values()):
        raise ValueError("Architecture suite lacks the planned three seeds per regressor")
    selected = min(
        (m for m, _ in primary if m["family"] != "autoencoder"),
        key=lambda m: m["metrics"]["validation"]["rmse_cycles"],
    )
    deployment = json.loads((results / "deployment.json").read_text())
    if deployment["model_version"] != selected["model_version"]:
        raise ValueError("Deployment differs from primary validation-only selection")
    lines = [
        "# FD001 measured results",
        "",
        "Generated from checksum-verified frozen `architecture-v1`, `sensitivity-v1`, "
        "and `temporal-windows-exploratory-v1` suites. "
        "Run `uv run --no-sync python -m neuropulse.evaluation.report --output "
        "docs/research/results.md` to regenerate after local artifacts are available.",
        "",
        "The fixed protocol uses 80 training and 20 validation engines, a sampled truncated "
        "prefix per validation engine, a 125-cycle training target cap, and uncapped "
        "terminal truth for the 100 official test engines. Model selection used validation "
        "only. Each row below is the mean of seeds 42, 43, and 44; the range is across those "
        "three runs, not a confidence interval. Linear regression is deterministic under "
        "these settings and thus has identical seed results.",
        "",
        "| Architecture | Validation RMSE (cycles) | Official test MAE (cycles) | "
        "Official test RMSE (cycles) | Test RMSE seed range | NASA score sum | "
        "Training seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("linear", "random_forest", "numpy_mlp", "mlp", "lstm", "cnn", "cnn_lstm"):
        entries = groups[family]
        values = [entry[1]["metrics"]["rmse_cycles"] for entry in entries]
        means = {
            key: statistics.mean(entry[1]["metrics"][key] for entry in entries)
            for key in ("mae_cycles", "rmse_cycles", "nasa_score_sum")
        }
        validation = statistics.mean(
            entry[0]["metrics"]["validation"]["rmse_cycles"] for entry in entries
        )
        seconds = statistics.mean(
            json.loads((results / entry[0]["run_id"] / "timing.json").read_text())[
                "training_seconds"
            ]
            for entry in entries
        )
        lines.append(
            f"| {family} | {validation:.3f} | {means['mae_cycles']:.3f} | "
            f"{means['rmse_cycles']:.3f} | {min(values):.3f}–{max(values):.3f} | "
            f"{means['nasa_score_sum']:.1f} | {seconds:.1f} |"
        )
    lines += [
        "",
        "A paired official-test comparison uses the same 100 engines for each "
        "PyTorch MLP/LSTM seed pair. Negative MLP−LSTM RMSE differences favor "
        "the MLP. Engine bootstrap intervals condition on each fitted pair and "
        "do not include variation across training seeds.",
        "",
        "| Seed | MLP − LSTM RMSE (cycles) | Paired engine bootstrap 95% |",
        "|---:|---:|---:|",
    ]
    for seed in (42, 43, 44):
        rows = []
        for family in ("mlp", "lstm"):
            run = next(
                m for m, _ in primary if m["family"] == family and m["config"]["seed"] == seed
            )
            path = results / run["run_id"] / "final_test/predictions.csv"
            with path.open(newline="") as stream:
                rows.append(list(csv.DictReader(stream)))
        if len(rows[0]) != 100 or [r["engine_id"] for r in rows[0]] != [
            r["engine_id"] for r in rows[1]
        ]:
            raise ValueError("Paired official test engine IDs differ")
        truth = np.array([float(r["true_rul_cycles"]) for r in rows[0]])
        if not np.array_equal(truth, [float(r["true_rul_cycles"]) for r in rows[1]]):
            raise ValueError("Paired official truths differ")
        left = np.array([float(r["rul_cycles"]) for r in rows[0]])
        right = np.array([float(r["rul_cycles"]) for r in rows[1]])
        comparison = paired_rmse_difference(left, right, truth)
        interval = comparison["engine_bootstrap_95"]
        difference = comparison["difference_cycles"]
        assert isinstance(interval, list) and isinstance(difference, float)
        lower, upper = interval
        lines.append(f"| {seed} | {difference:.3f} | [{lower:.3f}, {upper:.3f}] |")
    ae_scores = ", ".join(f"{m['metrics']['healthy_proxy_validation_mse']:.4f}" for m in ae)
    manual_ae_path = results / "manual-ae-fd001-seed42"
    if not manual_ae_path.is_dir():
        raise ValueError("Manual-AE FD001 evidence is missing")
    manual_ae = json.loads((manual_ae_path / "manifest.json").read_text())
    if sha256((manual_ae_path / "weights.npz").read_bytes()) != manual_ae["weights_sha256"]:
        raise ValueError("Manual-AE weights changed after evaluation")
    lines += [
        "",
        "The single best validation run belongs to the LSTM family. The selected "
        f"seed-{selected['config']['seed']} LSTM bundle powers the demonstration; "
        "selection does not use official test scores. "
        "Across the fixed defaults, the PyTorch MLP has a lower mean test RMSE than the "
        "LSTM. Architecture rankings differ between the 20-engine validation sample and "
        "100-engine official test population. These small, fixed-budget experiments do "
        "not establish architecture superiority in general.",
        "",
        "## Planned validation sensitivity",
        "",
        "The sensitivity suite used seed 42 only. Its runs were frozen before official test "
        "labels were opened; its outcomes did not alter the primary model selection. "
        "The following are uncapped validation RMSE values in cycles:",
        "",
        "| Family | Window 10 | Window 30 primary | Window 50 | Window 100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in ("mlp", "lstm"):
        baseline = next(
            m for m, _ in primary if m["family"] == family and m["config"]["seed"] == 42
        )
        variants = {30: baseline["metrics"]["validation"]["rmse_cycles"]}
        for m, _ in sensitivity:
            c = m["config"]
            if m["family"] == family and c["window_length"] in (10, 50, 100):
                variants[c["window_length"]] = m["metrics"]["validation"]["rmse_cycles"]
        lines.append(
            f"| {family} | " + " | ".join(f"{variants[w]:.3f}" for w in (10, 30, 50, 100)) + " |"
        )
    lines += [
        "",
        "Window 50 has the lowest observed seed-42 validation RMSE for both tested "
        "families. This is a single-seed sensitivity finding. Wider MLP inputs also change "
        "its parameter count. Cap, width, depth, activation, initialization, and optimizer "
        "candidate outcomes remain in the frozen sensitivity suite for review.",
        "",
        "## Anomaly branch and uncertainty",
        "",
        "Three PyTorch dense autoencoders were trained on early-life proxy windows; their "
        "held-out healthy-proxy reconstruction MSE values were "
        f"{ae_scores}. "
        "Each threshold is the 99th percentile of that run's held-out proxy scores. "
        "Early life is a proxy, not a verified healthy label. There are no verified fault "
        "onset labels here, so anomaly recall and lead time are not claimed. The custom "
        "NumPy dense autoencoder also passed synthetic reconstruction/checkpoint tests "
        f"and a real FD001 proxy run: {manual_ae['training_windows']} training windows, "
        f"{manual_ae['healthy_proxy_validation_windows']} held-out proxy windows, "
        f"validation reconstruction MSE {manual_ae['validation_reconstruction_mse']:.4f}, "
        f"and q99 threshold {manual_ae['threshold_q99']:.4f}. These models have "
        "different implementations and run configurations, so this is not a matched "
        "architecture comparison.",
        "",
        "Each run contains a separate 2,000-resample engine-level bootstrap 95% interval "
        "for MAE and RMSE. Those intervals describe evaluation-population uncertainty "
        "conditional on one fitted model; the seed range above describes training "
        "variation. The official terminal evaluation uses one prediction per engine and "
        "never clips the truth or upper-clips predictions. The NASA score penalizes late "
        "predictions more heavily. Results are for a simulated single-condition benchmark "
        "and do not validate a deployed maintenance policy.",
        "",
    ]
    study_path = results / "studies/exploratory-validation-v1.json"
    if study_path.exists():
        study = json.loads(study_path.read_text())
        protocol = Path("configs/experiments/exploratory.yaml")
        if study["protocol_sha256"] != sha256(protocol.read_bytes()):
            raise ValueError("Exploratory protocol changed after evaluation")
        lines += [
            "## Exploratory validation diagnostics",
            "",
            "After official evaluation, the selected bundle was examined on the 20 "
            "validation engines only. Gaussian noise added to standardized sensors at "
            "σ=0.05 and σ=0.10 gave RMSE values of "
            f"{study['noise']['0.05']['rmse_cycles']:.3f} and "
            f"{study['noise']['0.1']['rmse_cycles']:.3f} cycles, versus "
            f"{study['noise']['clean']['rmse_cycles']:.3f} clean. The small apparent "
            "improvement is one fixed random draw on 20 engines, not evidence that "
            "sensor noise improves the model.",
            "",
            "With the demonstration thresholds, RUL-only, anomaly-only, and combined "
            "rules warned on all 20 validation trajectories. Their mean first-warning "
            "lead times were "
            f"{study['decisions']['rul_only']['mean_first_warning_lead_cycles']:.1f}, "
            f"{study['decisions']['anomaly_only']['mean_first_warning_lead_cycles']:.1f}, "
            f"and {study['decisions']['combined']['mean_first_warning_lead_cycles']:.1f} "
            "cycles, respectively. The anomaly-only event is a monitoring state; the "
            "policies have different severity and unmatched alert budgets. The AE "
            "threshold was calibrated using healthy-proxy windows from this same "
            "validation population, so these figures cannot establish an independent "
            "decision benefit. No verified fault-onset or physical noise labels exist.",
            "",
        ]
    anomaly_path = results / "studies/anomaly-thresholds-validation-v1.json"
    if anomaly_path.exists():
        anomaly_study = json.loads(anomaly_path.read_text())
        protocol = Path("configs/experiments/anomaly-thresholds.yaml")
        if anomaly_study["protocol_sha256"] != sha256(protocol.read_bytes()):
            raise ValueError("Anomaly threshold protocol changed after evaluation")
        lines += [
            "## Anomaly threshold sensitivity on proxy windows",
            "",
            f"The deployed AE scored {anomaly_study['early_proxy_windows']} early-life "
            f"and {anomaly_study['late_proxy_windows']} late-life validation windows. "
            "The q95/q99/q99.5 thresholds and descriptive exceedance fractions are:",
            "",
            "| Early-proxy quantile | Threshold | Early-proxy exceedance | Late-proxy exceedance |",
            "|---:|---:|---:|---:|",
        ]
        for quantile in ("0.95", "0.99", "0.995"):
            value = anomaly_study["quantiles"][quantile]
            lines.append(
                f"| {quantile} | {value['threshold']:.4f} | "
                f"{value['early_proxy_exceedance_fraction']:.3f} | "
                f"{value['late_proxy_exceedance_fraction']:.3f} |"
            )
        lines += [
            "",
            "Early and late life are RUL-derived proxy groups, not verified healthy "
            "and fault-onset labels. Windows from one engine are correlated. The "
            "early-proxy thresholds and exceedance fractions use the same calibration "
            "windows, so they are not independent false-alarm estimates. The higher "
            "late-proxy scores are descriptive and do not establish fault-detection "
            "accuracy or actionable warning lead time.",
            "",
        ]
    matched_path = results / "studies/matched-alerts-validation-v1.json"
    if matched_path.exists():
        matched = json.loads(matched_path.read_text())
        protocol = Path("configs/experiments/matched-alerts.yaml")
        if matched["protocol_sha256"] != sha256(protocol.read_bytes()):
            raise ValueError("Matched-alert protocol changed after evaluation")
        lines += [
            "## Matched warning-budget validation comparison",
            "",
            "A separate exploratory policy study used the first 10 sorted validation "
            "engines to tune each policy toward warnings on 10% of available "
            "complete-window cycles. It then evaluated the last 10 engines without "
            "changing those limits. Any warning state was counted as one binary event; "
            "warning severity and maintenance cost were not compared.",
            "",
            "| Policy | Calibration warning-cycle share | Evaluation warning-cycle share | "
            "Evaluation engines warned | Mean first-warning lead (cycles) |",
            "|---|---:|---:|---:|---:|",
        ]
        for policy in ("rul_only", "anomaly_only", "combined"):
            value = matched["results"][policy]
            lines.append(
                f"| {policy} | {value['calibration_warning_cycle_fraction']:.3f} | "
                f"{value['evaluation_warning_cycle_fraction']:.3f} | "
                f"{value['evaluation_engines_warned']}/10 | "
                f"{value['evaluation_mean_first_warning_lead_cycles']:.1f} |"
            )
        combined_lead = matched["results"]["combined"]["evaluation_mean_first_warning_lead_cycles"]
        lines += [
            "",
            f"The combined rule's {combined_lead:.1f}-cycle "
            "mean lead falls between RUL-only and "
            "anomaly-only in this tiny evaluation group; it does not demonstrate "
            "improvement. Calibration rates were close to 0.10, while evaluation "
            "rates diverged. All 20 validation engines had already participated in "
            "model selection, and early life remains a proxy rather than verified "
            "fault-onset ground truth. This study does not validate a deployment "
            "threshold or industrial maintenance cost.",
            "",
        ]
    lines += [
        "## Additional history sensitivity",
        "",
        "After official evaluation, the same four window lengths were evaluated "
        "for the remaining neural regressors at seed 42. This is explicitly "
        "exploratory and did not change the selected model or official scores.",
        "",
        "| Family | Window 10 | Window 30 primary | Window 50 | Window 100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in ("numpy_mlp", "cnn", "cnn_lstm"):
        baseline = next(
            m for m, _ in primary if m["family"] == family and m["config"]["seed"] == 42
        )
        variants = {30: baseline["metrics"]["validation"]["rmse_cycles"]}
        for manifest, _ in extra_windows:
            if manifest["family"] == family:
                variants[manifest["config"]["window_length"]] = manifest["metrics"][
                    "validation"
                ]["rmse_cycles"]
        if set(variants) != {10, 30, 50, 100}:
            raise ValueError(f"Incomplete exploratory window study: {family}")
        lines.append(
            f"| {family} | "
            + " | ".join(f"{variants[window]:.3f}" for window in (10, 30, 50, 100))
            + " |"
        )
    lines.append("")
    lines += [
        "## Reproduction and provenance",
        "",
        "The frozen suite files are under `results/suites/`; each ignored local run directory "
        "contains resolved configuration, split IDs, prefix endpoints, environment, "
        "training curves, per-engine predictions, timing, metrics, and checksums. "
        "`final_test/metrics.json` and `final_test/predictions.csv` hold the official "
        "terminal outcomes. The benchmark download URL, hashes, and unresolved usage "
        "terms are documented in [the dataset audit](../data/dataset-audit.md). "
        "See [the protocol](protocol.md) for target and split policy. The initial local "
        "runs were made from an unversioned, uncommitted working tree; exact source "
        "revision reproduction cannot yet be claimed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = render()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
