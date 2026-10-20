# Decision 0002 — independent computation and staged dependency installation

Date: 2026-10-14. The full research dependency download suffered DNS/transfer failures.
Resolution is restricted to the supported Windows/Linux targets; macOS is not a tested
deployment target. Keep core mathematics/data NumPy-only and split heavy reporting libraries
into a reporting extra. The neural extra permits PyTorch parity without requiring SciPy,
Parquet or Matplotlib first; research adds sklearn baselines; backend is independent.

Every regression run always writes predictions.csv with the same engine/cycle/raw prediction,
nonnegative prediction/uncapped truth/padding fields. Parquet is an additional representation
when reporting dependencies are installed, not a prerequisite for custom neural learning.
This storage choice changes no split, label, window, metric or scientific selection policy.
Record the prediction file in the manifest. Scientific figures will still use Matplotlib
when the reporting extra is available; missing figures are not fabricated.

Training/inference model hashes include weights, resolved configuration and scaler hashes.
Repeated equivalent exports reuse an existing immutable bundle without rewriting its
manifest; different seeds/configs receive distinct versions even if linear weights coincide.

The primary Phase 2 scope is FD001. Other subsets, secondary datasets, from-scratch LSTM,
uncertainty and live telemetry retain their independently gated optional scope.
