# C-MAPSS initial dataset audit

Audit date: 2026-10-13. Scope: source/provenance/README/FD001 schema inspection for Phase 0.
No dataset is retained locally, no training occurred, and no official RUL label values
were inspected. The larger data pipeline and quality analysis belong to Phase 2.

## Source and access evidence

- Publisher landing page: [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
- Catalog: [NASA C-MAPSS simulated data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
- Official repository-linked archive:
  `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`.
- HTTP 200, 12,429,152 bytes, fetched into memory via Python urllib.
- Outer SHA-256: `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.
- Nested `CMAPSSData.zip` SHA-256:
  `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`.
- Nested archive includes README, damage-propagation paper, and train/test/RUL files for
  FD001–FD004. This is the four-subset benchmark, not the separately listed PHM08 challenge.

## Verified schema evidence

The actual `train_FD001.txt` contains 20,631 nonempty rows, 100 engine IDs, and exactly 26
numeric tokens in every row. File SHA-256:
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
These are measured structural counts, not quality/performance claims.

The inspected README repeats the catalog's typo in its last sensor-column description.
With ID, cycle and three settings occupying five columns, the remaining 21 columns are
sensor_01–sensor_21. The parser will enforce 26 total columns; no sensor_26 is inferred.
Physical sensor-name mappings remain to be verified. The README was read as CP1252;
numeric training files were read as ASCII. Phase 2 must record encodings in provenance.

## Usage and unresolved facts

NASA's catalog does not specify a license; the inspected README does not provide an
explicit redistribution license. NASA requests repository/donor acknowledgement. This
audit does not assign a license to the data or authorize redistribution of raw files.
Use locally for the stated academic work, preserve source/attribution, and keep usage
restrictions unresolved until explicit evidence exists. Cite Saxena and Goebel (2008),
*Turbofan Engine Degradation Simulation Data Set*, NASA Prognostics Data Repository,
plus the benchmark-generation paper when reporting results. [Repository attribution guidance](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Still required in Phase 2: complete missing/invalid/finite-value and trajectory continuity
checks, measured sensor distributions, lifespan ranges, hashes for every acquired file,
label/engine alignment without tuning on official labels, condition analysis, and a
persisted acquisition manifest. No verified fault-onset annotations have been established.
GPU/epoch time and secondary-dataset terms are unrelated open verification items.

Phase 2 downloads to ignored `data/raw/cmapss` with checksums, explicit size limits,
safe archive paths, and rejection of schema/hash mismatches. It must handle the outer
archive's directory and nested ZIP, rather than expecting train files at its top level.

## Phase 2 local acquisition and measured FD001 pipeline

Update, 2026-10-20: the pinned archive was acquired into ignored
`data/raw/cmapss`. Its manifest records source URL, access time, archive and
per-file SHA-256 values, byte counts, attribution, and unresolved terms. The
prepared manifest under ignored `data/processed/FD001` records exact split IDs,
validation endpoints, feature order, scaler hash, per-sensor training statistics,
trajectory lengths, and raw file hashes. Parser checks require finite 26-column
numeric rows, valid IDs/cycles, unique ordered engine-cycle keys, and aligned
official test engine counts. Unit fixtures verify leakage and boundary behavior.

The deterministic seed-42 split yielded 80 training engines (16,527 rows) and
20 validation engines (4,104 rows). The official test observations contain 100
engines (13,096 rows). Training-only variance selection retained 15 of 21 sensor
channels. The preparation path checked official test observations structurally
without reading their RUL labels. Those labels were later read through the
separate final-evaluation command after the architecture and sensitivity suites
were frozen. No inferred fault-onset label or sensor physical unit is claimed.

The original archive's usage terms remain unresolved; the local raw files and
model artifacts are excluded from Git. FD002–FD004 condition adapters and
secondary CWRU data remain future work.
