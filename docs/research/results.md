# FD001 measured results

Generated from checksum-verified frozen `architecture-v1`, `sensitivity-v1`, and `temporal-windows-exploratory-v1` suites. Run `uv run --no-sync python -m neuropulse.evaluation.report --output docs/research/results.md` to regenerate after local artifacts are available.

The fixed protocol uses 80 training and 20 validation engines, a sampled truncated prefix per validation engine, a 125-cycle training target cap, and uncapped terminal truth for the 100 official test engines. Model selection used validation only. Each row below is the mean of seeds 42, 43, and 44; the range is across those three runs, not a confidence interval. Linear regression is deterministic under these settings and thus has identical seed results.

| Architecture | Validation RMSE (cycles) | Official test MAE (cycles) | Official test RMSE (cycles) | Test RMSE seed range | NASA score sum | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| linear | 18.663 | 13.038 | 16.338 | 16.338–16.338 | 430.9 | 2.3 |
| random_forest | 22.091 | 13.445 | 17.984 | 17.930–18.030 | 858.4 | 656.7 |
| numpy_mlp | 16.577 | 11.952 | 16.596 | 16.310–17.125 | 469.6 | 10.8 |
| mlp | 17.097 | 11.778 | 15.765 | 15.200–16.186 | 383.1 | 11.1 |
| lstm | 15.449 | 12.291 | 16.097 | 15.209–16.599 | 455.4 | 38.2 |
| cnn | 18.797 | 13.988 | 18.221 | 17.656–18.704 | 562.5 | 21.1 |
| cnn_lstm | 16.059 | 12.174 | 16.403 | 15.841–17.338 | 498.1 | 44.0 |

A paired official-test comparison uses the same 100 engines for each PyTorch MLP/LSTM seed pair. Negative MLP−LSTM RMSE differences favor the MLP. Engine bootstrap intervals condition on each fitted pair and do not include variation across training seeds.

| Seed | MLP − LSTM RMSE (cycles) | Paired engine bootstrap 95% |
|---:|---:|---:|
| 42 | -0.296 | [-2.408, 1.803] |
| 43 | 0.699 | [-1.080, 2.683] |
| 44 | -1.399 | [-3.408, 0.702] |

The single best validation run belongs to the LSTM family. The selected seed-44 LSTM bundle powers the demonstration; selection does not use official test scores. Across the fixed defaults, the PyTorch MLP has a lower mean test RMSE than the LSTM. Architecture rankings differ between the 20-engine validation sample and 100-engine official test population. These small, fixed-budget experiments do not establish architecture superiority in general.

## Planned validation sensitivity

The sensitivity suite used seed 42 only. Its runs were frozen before official test labels were opened; its outcomes did not alter the primary model selection. The following are uncapped validation RMSE values in cycles:

| Family | Window 10 | Window 30 primary | Window 50 | Window 100 |
|---|---:|---:|---:|---:|
| mlp | 19.961 | 17.157 | 13.464 | 15.106 |
| lstm | 21.396 | 15.562 | 12.045 | 14.789 |

Window 50 has the lowest observed seed-42 validation RMSE for both tested families. This is a single-seed sensitivity finding. Wider MLP inputs also change its parameter count. Cap, width, depth, activation, initialization, and optimizer candidate outcomes remain in the frozen sensitivity suite for review.

## Anomaly branch and uncertainty

Three PyTorch dense autoencoders were trained on early-life proxy windows; their held-out healthy-proxy reconstruction MSE values were 0.3317, 0.3254, 0.3231. Each threshold is the 99th percentile of that run's held-out proxy scores. Early life is a proxy, not a verified healthy label. There are no verified fault onset labels here, so anomaly recall and lead time are not claimed. The custom NumPy dense autoencoder also passed synthetic reconstruction/checkpoint tests and a real FD001 proxy run: 1385 training windows, 332 held-out proxy windows, validation reconstruction MSE 0.2506, and q99 threshold 0.4689. These models have different implementations and run configurations, so this is not a matched architecture comparison.

Each run contains a separate 2,000-resample engine-level bootstrap 95% interval for MAE and RMSE. Those intervals describe evaluation-population uncertainty conditional on one fitted model; the seed range above describes training variation. The official terminal evaluation uses one prediction per engine and never clips the truth or upper-clips predictions. The NASA score penalizes late predictions more heavily. Results are for a simulated single-condition benchmark and do not validate a deployed maintenance policy.

## Exploratory validation diagnostics

After official evaluation, the selected bundle was examined on the 20 validation engines only. Gaussian noise added to standardized sensors at σ=0.05 and σ=0.10 gave RMSE values of 15.287 and 14.782 cycles, versus 15.322 clean. The small apparent improvement is one fixed random draw on 20 engines, not evidence that sensor noise improves the model.

With the demonstration thresholds, RUL-only, anomaly-only, and combined rules warned on all 20 validation trajectories. Their mean first-warning lead times were 48.1, 55.1, and 63.0 cycles, respectively. The anomaly-only event is a monitoring state; the policies have different severity and unmatched alert budgets. The AE threshold was calibrated using healthy-proxy windows from this same validation population, so these figures cannot establish an independent decision benefit. No verified fault-onset or physical noise labels exist.

## Anomaly threshold sensitivity on proxy windows

The deployed AE scored 332 early-life and 1000 late-life validation windows. The q95/q99/q99.5 thresholds and descriptive exceedance fractions are:

| Early-proxy quantile | Threshold | Early-proxy exceedance | Late-proxy exceedance |
|---:|---:|---:|---:|
| 0.95 | 0.6109 | 0.051 | 0.867 |
| 0.99 | 0.6869 | 0.012 | 0.848 |
| 0.995 | 0.7210 | 0.006 | 0.841 |

Early and late life are RUL-derived proxy groups, not verified healthy and fault-onset labels. Windows from one engine are correlated. The early-proxy thresholds and exceedance fractions use the same calibration windows, so they are not independent false-alarm estimates. The higher late-proxy scores are descriptive and do not establish fault-detection accuracy or actionable warning lead time.

## Matched warning-budget validation comparison

A separate exploratory policy study used the first 10 sorted validation engines to tune each policy toward warnings on 10% of available complete-window cycles. It then evaluated the last 10 engines without changing those limits. Any warning state was counted as one binary event; warning severity and maintenance cost were not compared.

| Policy | Calibration warning-cycle share | Evaluation warning-cycle share | Evaluation engines warned | Mean first-warning lead (cycles) |
|---|---:|---:|---:|---:|
| rul_only | 0.100 | 0.098 | 10/10 | 18.7 |
| anomaly_only | 0.099 | 0.134 | 9/10 | 28.1 |
| combined | 0.098 | 0.114 | 10/10 | 21.2 |

The combined rule's 21.2-cycle mean lead falls between RUL-only and anomaly-only in this tiny evaluation group; it does not demonstrate improvement. Calibration rates were close to 0.10, while evaluation rates diverged. All 20 validation engines had already participated in model selection, and early life remains a proxy rather than verified fault-onset ground truth. This study does not validate a deployment threshold or industrial maintenance cost.

## Additional history sensitivity

After official evaluation, the same four window lengths were evaluated for the remaining neural regressors at seed 42. This is explicitly exploratory and did not change the selected model or official scores.

| Family | Window 10 | Window 30 primary | Window 50 | Window 100 |
|---|---:|---:|---:|---:|
| numpy_mlp | 20.109 | 16.783 | 13.970 | 15.037 |
| cnn | 20.790 | 18.613 | 18.431 | 18.237 |
| cnn_lstm | 19.741 | 16.166 | 15.552 | 13.450 |

## Reproduction and provenance

The frozen suite files are under `results/suites/`; each ignored local run directory contains resolved configuration, split IDs, prefix endpoints, environment, training curves, per-engine predictions, timing, metrics, and checksums. `final_test/metrics.json` and `final_test/predictions.csv` hold the official terminal outcomes. The benchmark download URL, hashes, and unresolved usage terms are documented in [the dataset audit](../data/dataset-audit.md). See [the protocol](protocol.md) for target and split policy. The initial local runs were made from an unversioned, uncommitted working tree; exact source revision reproduction cannot yet be claimed.
