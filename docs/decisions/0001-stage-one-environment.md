# Decision 0001: stage-one environment and dependency boundaries

Date: 2026-10-13. Status: accepted for the initial scaffold.

AGENTS.md proposes Python 3.11 as the baseline and requires checking actual compatibility.
This machine has Python 3.12.3, uv 0.12.4, Git 2.42.0, Node 24.18.0, and npm 11.16.0.
Use the existing Python 3.12.3 locally; support Python >=3.11,<3.13 and run CI for both
minor versions. This is an environment choice, not a change to the scientific protocol.
No Node/Docker/backend setup occurs until those stages become unblocked.

Git initialization under the managed sandbox assigned `.git` to the sandbox Windows
account. Git under the user's account rejected it as dubious ownership. Ownership could
not be changed with the available Windows privileges; no ownership changes were made.
Added the exact verified workspace `D:/GitHub/1_Projects/NeuroPulse` to the user's Git
`safe.directory` configuration. This is a repository-specific exception, not a wildcard,
and normal-user `git status` then succeeded. No remote or commit was created.

The global uv cache is inaccessible under the managed workspace permissions. Set
`tool.uv.cache-dir` to `.cache/uv` and use `.venv` in the project. Ignore both in Git.
The initial dependencies are NumPy, PyYAML and Pydantic; pytest/Ruff/mypy are development
dependencies. [uv documents these dependency boundaries](https://docs.astral.sh/uv/concepts/projects/dependencies/).
Constrain NumPy to >=2.0,<2.3 for the initial shared Python 3.11/3.12 typing baseline.
The first resolution selected NumPy 2.5.4 locally; its Python-3.12-only stub syntax failed
the configured Python-3.11 mypy target. Keep the target and choose compatible NumPy stubs
rather than suppressing dependency typing errors. Revisit this bound with test evidence
when adding research dependencies; no trained result is affected by this environment fix.
NumPy neural modules must import without PyTorch or API libraries. Add research/CPU-PyTorch
and application dependencies only in the corresponding stages, with new lock/validation
evidence. Future CPU-PyTorch installation follows the explicit index configuration in
[uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

Hardware measured through Python/Win32 APIs: 12 logical processors, 16,870,006,784 bytes
physical memory (~15.7 GiB). The scaffold requires no GPU. CPU model work initially uses
at most two threads, one experiment at a time, and lazy windows. GPU model/name and actual
epoch timing remain unverified; they do not block foundation/synthetic tests.

Phase 0 research preparation and Phase 1 foundation may proceed while usage/annotation
uncertainties remain documented. They must not be marked as scientifically resolved by
successful installation. This is consistent with AGENTS.md's rule to continue independent
synthetic correctness work when access or verification is incomplete.
