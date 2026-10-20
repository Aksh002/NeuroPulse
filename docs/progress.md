# NeuroPulse stage progress

Updated: 2026-10-20. This file records implementation evidence, not expected outcomes.

| Phase | Status | Evidence / remaining gate |
|---|---|---|
| 0: research/scope | Prepared; qualified open verification items | Dataset audit, protocol, literature matrix and environment decision; terms/onset/full methods remain unresolved |
| 1: foundation | Complete locally; hosted CI pending | Locked CPU environment, 28 passing tests, Ruff/mypy, diagnostic and package build |
| 2: data pipeline | FD001 complete locally | Pinned acquisition, strict parser, split/endpoint manifests, training-only scaler, lazy windows and leakage checks; FD002–FD004 remain open |
| 3: NumPy engine | Dense core complete | Manual dense/activations/losses/optimizers/checkpoints; numerical and PyTorch parity, synthetic fit, real FD001 MLP |
| 4–5: RUL models | Complete locally | Linear/RF controls, NumPy/PyTorch MLP, LSTM, CNN, CNN-LSTM; three seeds each in frozen architecture suite |
| 6–7: advanced/anomaly | Mandatory/recommended gates complete locally | Custom Conv1D gradients/parity, basic RNN toy fit, real FD001 manual dense AE and three PyTorch AE proxy runs; manual LSTM optional |
| 8: evaluation | RQ1–5 complete with stated evidence limits; RQ6 bounded | 24-run architecture and 27-run sensitivity suites frozen before official terminal evaluation; nine post-test history runs, proxy AE thresholds, matched-budget policy study and bounded noise labeled exploratory |
| 9–11: inference/API/dashboard | Local browser demonstration working | Versioned bundles, rules, five frontend views; real Edge replay and disconnected-server tests, TypeScript/Vitest/production build |
| 12–13: integration/report | Local integration and measured report; container gate open | Offline/API equality, synthetic E2E, real browser replay, generated report and clean-commit reproduction check; Docker daemon unavailable |

## Current evidence and open gates

- FD001 acquisition and preparation used the pinned NASA-linked archive. The measured
  split contains 80 training engines (16,527 rows), 20 validation engines (4,104 rows),
  and 100 official test trajectories (13,096 rows). Training-only selection retained
  15 sensors. See [the updated audit](data/dataset-audit.md).
- The frozen `architecture-v1` suite contains 24 jobs: three seeds for NumPy MLP,
  PyTorch MLP/LSTM/CNN/CNN-LSTM/AE, linear regression, and Random Forest.
  `sensitivity-v1` contains 27 validation-only seed-42 jobs. Official test labels
  were opened through the final terminal evaluator after both suites were frozen.
- The [generated report](research/results.md) gives uncapped 100-engine official
  terminal metrics and seed ranges. Mean test RMSE is 15.765 cycles for the PyTorch
  MLP, 16.097 for LSTM, 16.338 for linear regression, and 16.596 for the manual
  NumPy MLP. These measured defaults do not establish general superiority.
- Post-test validation-only exploratory studies measured bounded standardized
  sensor noise, AE q95/q99/q99.5 proxy thresholds, and three decision branches.
  A separate 10/10 validation-engine policy split matched calibration
  warning-cycle budgets near 10%; evaluation rates diverged. These studies do
  not alter the frozen primary suite or validate fault onset or a maintenance policy.
- The manual NumPy dense AE trained on 1,385 training healthy-proxy windows and
  scored 332 held-out proxy windows; validation reconstruction MSE was 0.2506
  and q99 threshold was 0.4689. The checkpoint loaded successfully. This is
  separate from the deployed PyTorch AE and does not validate fault detection.
- Local commit `5da1db4` fixed a source revision for a new reproduction check.
  Two clean-revision reruns of the frozen seed-42 PyTorch MLP configuration
  produced identical checkpoint SHA-256 and validation RMSE 17.1339206883.
  The original unversioned run was close (17.1572099338) but not byte-identical;
  its exact original source cannot be reconstructed. See
  [reproducibility](research/reproducibility.md). No official test score was changed.
- The application selection is validation-best `lstm-fd001-v1-1dda9d2d`
  (seed 44) plus compatible AE `autoencoder-fd001-v1-a37904e3`. A real 30-cycle
  API replay returned finite RUL and anomaly outputs, and an automated local test
  matched its first complete prediction to offline inference.
- Microsoft Edge headless browser tests passed against live local FastAPI and
  Next servers: real replay, prediction after warm-up, pause/resume/reset,
  model/experiment navigation, and explicit disconnected-API error state.
- Python tests (71), Ruff, mypy (45 source files), frontend type checks,
  Vitest, two live Edge browser tests, and Next production build pass locally.
  Frontend npm audit reports zero vulnerabilities after the Vitest update.
  Docker CLI exists but its engine was unavailable; container behavior has not
  been checked. Hosted CI has not run because no remote is configured.
- Remaining full-scope work: complete local container validation. The original
  suite's source was unversioned; a separate committed-source rerun is exact,
  but cannot retroactively identify the original source. Independently labeled
  anomaly metrics require a suitable dataset and are not claimed.
  FD002–FD004 and custom LSTM remain outside the current core gate.

## Phase 1 historical foundation record

- Initialized local Git on `main`; no commit, remote, or publishing performed.
- Fixed Git's sandbox-owner mismatch with an exact-project `safe.directory` exception;
  normal-user Git works and no broader repository trust was added.
- Added installable src-layout package and module boundaries without fake model stubs.
- Added strict versioned FD001 configuration and finite real [B,W,F] sensor-window contract.
- Added synthetic tests for config errors, shape/numerical errors, independent imports and CLI errors.
- Added uv lock/environment setup, ignore rules, example settings and CPU CI workflow.
- Measured hardware and inspected NASA-linked archive in memory; no dataset retained or label values read.

## Phase 1 historical validation evidence

Validated on Windows/Python 3.12.3 with uv 0.12.4:

| Check | Result |
|---|---|
| `uv sync --locked --offline --python C:\Python312\python.exe` | Passed; 20 resolved packages including editable project |
| `uv run --no-sync pytest` | 28 passed; all fixtures synthetic |
| `uv run --no-sync ruff check .` | Passed |
| `uv run --no-sync mypy src/neuropulse` | Passed; 14 source files, Python 3.11 typing target |
| `uv run --no-sync python -m neuropulse --config configs/data/fd001.yaml` | Passed; valid config, models unavailable |
| `uv build --offline` | Built wheel and source distribution; local caches/data excluded |
| Package/ignore inspection | Typed marker and YAML config present; local environments/data/results excluded |

Locked runtime versions: NumPy 2.2.6, Pydantic 2.14.1, PyYAML 6.0.3. Development versions:
pytest 9.1.1, Ruff 0.17.0, mypy 1.20.2. uv.lock records exact resolved dependencies.
The first build exposed a missing README while scaffolding; the README was added before
the successful build. The first type check exposed NumPy 2.5.4 stub incompatibility with
the Python 3.11 typing target; the documented compatible bound resolved it, and tests were
rerun. No check failure is suppressed. GitHub-hosted CI execution is pending repository
publication; Python 3.11 runtime execution and Linux CI have not been claimed as locally run.

## Earlier planned Phase 2 task (completed)

This task was completed before model training. The original note is retained here
as a record of the stage transition; current open work is listed above.

## User actions

No mandatory manual action is required for the local application. Optionally select
the project's `.venv\Scripts\python.exe` in the IDE. Start Docker Desktop if you
want the container workflow validated. A deadline or GPU/cloud requirement would
change later experimental priorities; CPU-first remains confirmed.
