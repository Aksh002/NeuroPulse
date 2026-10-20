# Initial literature-to-experiment matrix

The verified bibliography/version distinctions are in AGENTS.md Section 25. This matrix
records the intended relationship to our experiments; it is not a completed full-text
systematic review. Do not add a paper's numerical claims without checking its exact protocol.

| Source | Task / dataset evidence | Planned relationship | Still requires verification |
|---|---|---|---|
| Serradilla et al. (2022), doi:10.1007/s10489-021-03004-y | Predictive maintenance survey | Task/architecture taxonomy for RQ1–RQ5; not a reproduction | Detailed comparative protocols and limits |
| Zhao et al. (2019), doi:10.1016/j.ymssp.2018.05.050 | Machine health survey | Neural computation context, AE/CNN/RNN distinctions | Published/preprint method differences and reviewed datasets |
| Nguyen/Medjaher (2019), doi:10.1016/j.ress.2019.03.018 | Prognosis-to-maintenance framework | Motivate RQ5; our transparent rules differ from its probability framework | Full model/calibration/cost methodology; no imported industrial savings |
| Zheng et al. (2017), doi:10.1109/ICPHM.2017.7998311 | LSTM RUL work | RQ1 LSTM reference; independently specified architecture | Full text, caps/windows/splits and original implementation |
| Li/Ding/Sun (2018), doi:10.1016/j.ress.2017.11.021 | Windowed CNN RUL reference | RQ1/2 CNN/window experiments | Exact normalization, target and benchmark comparability |
| Kong et al. (2019), doi:10.3390/app9194156 | C-MAPSS hybrid with health-indicator preprocessing | RQ1 hybrid/RQ5 motivation; our raw-window baseline is not exact reproduction | HI derivation and preprocessing causality before any adaptation |
| Liu et al. (2024), doi:10.3390/app14020774 | Industrial reconstruction/prediction anomaly model | RQ4 background; our first dense AE is deliberately simpler | Exact labels, datasets, thresholding and evaluation methodology |
| Saxena et al. (2008), NASA NTRS 20090029214 | Simulation and asymmetric prognosis scoring | Primary benchmark/scoring basis | Physical sensor mapping and available fault-onset annotations |

For full review, expand each record with DOI/version, access date/status, architecture,
loss/optimizer, dataset/subset, feature representation, target cap, partition/tuning,
metrics/evaluation population, calibration, code, verified limitations and adaptation.
Unavailable details stay explicitly unresolved. Research conclusions must be based on
our artifacts; architecture superiority is a hypothesis, not a completion condition.
