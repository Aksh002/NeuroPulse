# Source-revision reproduction check

The architecture and sensitivity suites were trained before the first local
Git commit. Their environment manifests record a dirty, unversioned source
tree; an early `git rev-parse HEAD` call could write literal `HEAD` despite no
resolvable commit. This has been corrected for future runs. The original
three-seed benchmark remains valid as a measured local study, but exact source
revision reproduction for those runs cannot be established retroactively.

The repository was committed locally as `5da1db49e78e2b3c7ee67d1a6a78007a6be340b0`.
The frozen seed-42 PyTorch MLP configuration was then run twice directly from
that clean revision, including one run with the original two-thread CPU
environment. Both reruns recorded `dirty: false`, produced validation RMSE
`17.133920688300552` cycles, and produced the identical `weights.pt` SHA-256
`86b88900d0cca7709a15a0d03d63abd06aeecde926e00a967a5c7c9f0416015d`.
Their ignored local run IDs end in timestamps `20261020T071227051682Z` and
`20261020T071333531427Z`. This establishes exact rerun reproducibility for
that committed source and environment on this machine.

The corresponding original frozen run had validation RMSE
`17.157209933802314` cycles and a different checkpoint hash. Its first-epoch
loss differed only at the floating-point level, but the cause cannot be
isolated from the original unversioned source. The original official test
tables were not changed or reselected using the committed reruns. The
[primary report](results.md) retains the original frozen population and labels
the later window, noise, anomaly and decision studies as exploratory.

To reproduce from a committed checkout, acquire and prepare the pinned FD001
archive using [README](../../README.md), then run the exact resolved
configuration stored in the frozen architecture suite. A direct invocation of
`neuropulse.training.train.run(TrainConfig.model_validate(config))` creates a
new ignored run without mutating the frozen suite. Verify the new manifest's
source revision, raw hashes, scaler checksum, metrics and weight hash. Do not
expect identical weights across different CPU libraries, thread settings or
software versions merely because the random seed matches.
