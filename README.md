# NeuroPulse

NeuroPulse is a CPU-first research framework for remaining useful life prediction
and reconstruction-based anomaly monitoring on NASA's simulated C-MAPSS
turbofan benchmark. It includes a manually differentiated NumPy neural engine,
baselines and PyTorch models, reproducible experiment artifacts, and a historical
replay application. [AGENTS.md](AGENTS.md) is the implementation blueprint;
[stage progress](docs/progress.md) records completed and open gates. See the
[measured FD001 results](docs/research/results.md) for the frozen three-seed
architecture comparison and its limitations.

## Local setup

Requirements: Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and Node.js 20.9 or later for the dashboard. Local experiments run on CPU.

From the NeuroPulse directory:

```powershell
uv sync --locked --extra research --extra backend
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync mypy src/neuropulse
cd apps/frontend
npm ci
npm run typecheck
npm test
npm run build
```

If using a different supported interpreter, select it explicitly with
`uv sync --locked --python 3.11` or `uv sync --locked --python 3.12`.
uv writes its cache inside `.cache/uv`, avoiding this machine's restricted shared cache.
The environment is installed in `.venv`; IDE users can select `.venv\Scripts\python.exe`.
These checks use synthetic fixtures and do not require the benchmark. The
frontend uses `http://localhost:8000/api/v1` by default.

## Acquire and reproduce

The acquisition command downloads the official repository-linked archive,
checks pinned hashes, and writes ignored local files with a provenance manifest.
The NASA catalog does not specify a redistribution license; preserve attribution
and review [the dataset audit](docs/data/dataset-audit.md).

```powershell
uv run --no-sync python -m neuropulse.data.acquire
uv run --no-sync python -m neuropulse.data.prepare
uv run --no-sync python -m neuropulse.experiments.runner --name architecture-reproduction --families numpy_mlp mlp lstm cnn cnn_lstm autoencoder linear random_forest --seeds 42 43 44
uv run --no-sync python -m neuropulse.evaluation.evaluate --suite results/suites/architecture-reproduction.json --final-test
```

The suite runs 24 CPU jobs, including three 200-tree forests. Use a new suite
name for each repeat because frozen names are immutable. Run final evaluation
only after configurations and validation selections are frozen. The planned
single-seed studies are in `configs/experiments/sensitivity.yaml`. To regenerate
the committed research table from the local frozen artifacts:

```powershell
uv run --no-sync python -m neuropulse.evaluation.report --output docs/research/results.md
```

The custom dense AE has a separate reproducible FD001 proxy run:
`uv run --no-sync python -m neuropulse.training.autoencoder_numpy`.
The post-test exploratory protocols are in `configs/experiments/` and run with
`python -m neuropulse.experiments.exploratory`,
`python -m neuropulse.experiments.anomaly_thresholds`, and
`python -m neuropulse.experiments.matched_alerts` inside `uv run --no-sync`.
Each study writes an immutable ignored JSON result and is labeled exploratory.

## Historical replay

After acquisition, preparation, and model selection have created
`results/deployment.json`, open two terminals:

```powershell
uv run --no-sync uvicorn apps.backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

```powershell
cd apps/frontend
npm run dev
```

Open <http://localhost:3000>. Readiness is at
<http://localhost:8000/api/v1/health/ready> and interactive API documentation
is at <http://localhost:8000/docs>. Replay uses observed prefixes and emits a
warm-up state before a complete window exists. Evaluation truth has a separate
opt-in route. Rules are research demonstrations, not validated maintenance
policies. See the [demonstration walkthrough](docs/demo.md) and
[manual neural derivations](docs/research/derivations.md). Sessions live in one
backend process. Container operation has not
been validated because the Docker engine was unavailable locally.

With Microsoft Edge installed, run `npm run test:e2e` from `apps/frontend`.
The test starts and stops both local servers, then checks real replay,
pause/resume/reset, keyboard activation, result views, and
the disconnected-server error state. Set `NEUROPULSE_BROWSER_PATH` to another
Chromium executable when needed. The test needs local data and selected bundles;
it is separate from dataset-free CI checks.

The container configuration is in `compose.yaml`. With Docker Desktop running
and the local benchmark, prepared manifest, deployment selection, and model
bundles present, use `docker compose up --build`. Raw and prepared data and
model bundles are mounted read-only; replay logs use the writable `results`
mount. `docker compose config --quiet` passed locally, but an image build and
container replay still need an available Docker engine.

## Stages

| Stage | Deliverable | Current status |
|---|---|---|
| 0 | Sources, hardware, dataset audit and frozen FD001 protocol | Audit documented; unresolved facts explicitly tracked |
| 1 | Package, configuration, local environment, tests and CI | See validation evidence in progress log |
| 2 | Provenance-preserving acquisition, reader, splits, targets, scaling and windows | FD001 implemented and measured |
| 3 | Checked NumPy neural engine | Dense core, Conv1D and basic RNN validated |
| 4–8 | Baselines, temporal models and controlled experiments | Frozen three-seed architecture and single-seed sensitivity suites; broader studies open |
| 9–13 | Inference/rules, API, dashboard, integration and presentation | Local replay and dashboard implemented; container and full research gates open |

Follow dependency gates in AGENTS.md, not merely numeric order. Phase 6/7 advanced work does
not block the MVP. No dataset-derived metric may be reported before the corresponding run.

## Data and research

FD001 starts with fixed engine split seed 42, validation prefix seed 142, 30-cycle windows,
stride one, and a 125-cycle target cap. The cap is a hypothesis; evaluation uses uncapped
truth. Fit all feature selection and normalization exclusively on training engines.
See [the frozen protocol](docs/research/protocol.md) and
[the acquisition audit](docs/data/dataset-audit.md).

Data, weights, generated results, caches, and `.env` files are ignored by Git.
The official archive is locally acquired and verified. Official test labels
were opened only for the frozen terminal evaluation. Public availability does
not establish redistribution rights; usage terms remain unresolved.

## What you need to do

No mandatory manual action is needed for the local demonstration. Optionally
select `.venv\Scripts\python.exe` in your IDE. Start Docker Desktop only if
you want the unverified container workflow tested. GitHub CI will run after
the repository is connected and pushed; local Git initialization does not publish it.
