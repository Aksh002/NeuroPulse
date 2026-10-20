# Local demonstration walkthrough

Prepare the dataset and selected bundles using [README](../README.md), then
start the one-worker backend and frontend. Check `/api/v1/health/ready` before
opening the dashboard; a 503 response names a missing dataset or model artifact.

Open the Overview, choose a held-out validation engine, and start historical
replay. The first 29 cycles are a warm-up for the default 30-cycle model. At
cycle 30, the RUL estimate, reconstruction score, RUL-derived display index,
and recommendation become available. Pause, resume, change replay speed, and
reset; the server records one chronological observation per step and prevents
duplicate cycle advances. Change the sensor selector to inspect raw-prefix
trajectories. The optional actual-RUL overlay uses a separate evaluation route
and is never sent to inference.

Open Models and Experiments to inspect saved validation metrics, training
history, hyperparameters, and downloadable run metadata. Read the
[measured report](research/results.md) alongside these views: the model selected
for the demonstration was chosen from held-out training engines before the
official test evaluation. The anomaly threshold is calibrated on an early-life
healthy proxy, not on verified fault-onset labels. The recommendation rule is a
transparent research demonstration, not an industrial safety policy.

To repeat the real-data integration gate, run
`uv run --no-sync pytest tests/integration/test_real_artifacts.py -q`.
It verifies warm-up, a real bundle's offline/API prediction equality, and the
absence of truth in the replay observation. Without local data or bundles,
the test skips; the main synthetic suite still runs in CI.
