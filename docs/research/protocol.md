# FD001 starting protocol — version 1.0

Date frozen: 2026-10-13. This records the AGENTS.md defaults; it does not claim experiments
have run. Changing a scientific choice requires a decision record and a new resolved config.

| Choice | Fixed starting policy |
|---|---|
| Dataset | C-MAPSS FD001; structural provenance in dataset-audit.md |
| Identity | Dataset/subset/partition/engine; overlapping numerical IDs are distinct machines |
| Split | Sorted official training IDs; default_rng(42) permutation; floor(0.8*N) training |
| Validation | Remaining engines; one frozen prefix/engine using default_rng(142), endpoints ceil(0.5*T)..T-1 |
| RUL target | Training T-t; min(raw_RUL,125) for learning; uncapped truth for selection/evaluation |
| Features | Training-variance-selected sensors only; ID, cycle, total life and truth excluded |
| Scaling | Training-row population mean/std; feature mask variance >1e-8; std floor 1e-12 |
| Windows | [B,30,F], stride one, per engine; prefix-only; no future interpolation/normalization |
| Short histories | Replay warm-up; left-repeat earliest row only for terminal validation/test evaluation |
| Seeds | Model42/43/44 separate from split42, validation142 and bootstrap2026 |
| Selection | Validation uncapped terminal-prefix RMSE, best earlier epoch on ties; test reserved for final |
| Metrics | Terminal MAE/RMSE/NASA sum against uncapped truth; separate trajectory/capped/raw diagnostics |
| Uncertainty | Engine-level bootstrap, 2,000 draws; variability across training seeds reported separately |
| Compute | CPU-first, two threads max initially; measured runtime before broader suite |

The cap models an early-life plateau; it does not prove degradation begins at 125 cycles.
FD001 baselines are Linear Regression/Random Forest; initial neural regressors are custom
NumPy MLP and PyTorch MLP/LSTM. CNN/hybrid/AE studies follow their validation gates. No
industrial failure probability or actual fault-onset metric is assumed.

Three-seed architecture comparisons come first. History10/30/50/100, cap100/125/150/uncapped,
selected optimizer/depth/width/initialization studies, and anomaly/decision ablations follow
the bounded budgets in AGENTS.md. No full Cartesian grid. Prefer informative negative
results to undocumented tuning.

**Test-data policy:** no official test sensor/label information fits preprocessing,
selects features, picks hyperparameters/checkpoints, or changes thresholds. Inspecting
archive member names in Phase 0 did not read official label values. Freeze the suite
before final-test evaluation. Ground truth travels separately from future replay inputs.

Initial scientific defaults are encoded in `configs/data/fd001.yaml`. Actual engine IDs,
validation endpoints, feature masks and fitted scaler statistics do not exist yet and
must be generated from verified training data in Phase 2. Package versions are recorded
in uv.lock; no performance values are recorded before measured runs.
