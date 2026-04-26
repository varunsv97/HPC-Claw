# HPC Claw

> Full architecture and component reference: [docs/architecture.md](docs/architecture.md)

## Code Style

- Python 3.14, Pydantic v2 (`BaseModel`, `model_dump_json`, `model_validate_json`, `model_copy`)
- `pydantic-settings` for all runtime config (`AssistantSettings`, `HPC_ASSISTANT_*` env prefix)
- Stdlib `tomllib` for TOML parsing — no third-party TOML library
- `from __future__ import annotations` in every module
- DSPy (`>=2.6.0`) is optional: import it inside `try/except`, guard Signature class bodies with `if _DSPY_AVAILABLE:`

## Architecture

Three layers that compose into a closed optimization loop:

```
agent.toml (project config)
    └── PGOAAgent (loop.py)
            ├── DSPy structured reasoning (dspy_prompts.py)
            │       ├── BottleneckToHypothesis → hypothesis + edit_needed
            │       ├── HypothesisToEditPrompt → opencode_prompt
            │       └── MetricsDeltaEvaluation → verdict + should_rollback
            ├── EditDispatcher → opencode run <prompt> + git diff → EditRecord
            ├── Services layer (services.py)
            │       └── Adapters: SlurmAdapter · NCUAdapter · LIKWIDAdapter · HardwareAdapter
            └── ExperimentStore (store.py) → ~/.hpcassist/<workload_id>/
```

All Pydantic models live in `pgoa/schema.py`.  One change per optimization iteration (enforced by system prompt and architecture).

## Build and Test

```bash
# Install (from repo root)
python3 -m venv .venv && . .venv/bin/activate
pip install -e src             # installs hclaw (claw_backend + claw_tui) + all deps
pip install pytest             # for running the test suite

# Run tests
cd src && python -m pytest tests/ -q

# Smoke-test imports (no network required)
python -c "from claw_backend.pgoa.agent.loop import PGOAAgent; print('ok')"
```

## Conventions

- **Schema changes** always go in `pgoa/schema.py`. Never define Pydantic models elsewhere.
- **New profiling adapters** inherit from `pgoa/adapters/base.py::BaseAdapter` and return a `ProfileBundle`.
- **Service functions** (tool implementations) live in `pgoa/services.py`, not in the agent loop.  Both the agent and the OpenCode tool bridge call the same service layer.
- **Atomic writes** — use `ExperimentStore._atomic_write(path, content)` for any persistent state.  Never write directly with `path.write_text()` outside the store.
- **Path access control** — always resolve paths through `path_access.resolve_allowed_path()` or `resolve_writable_path()` before reading or writing user-supplied paths.
- **DSPy modules** are lazily instantiated in `get_modules()` — do not instantiate `ChainOfThought` / `Predict` at import time; they require a configured LM.
- **Agent.toml** is the operator-facing config surface.  Do not add new HPC-project-level settings to `AssistantSettings`; put them in `ProjectPathsConfig` or `PGOAProjectConfig` instead.
- **Test fixtures** are in `tests/pgoa/fixtures/` — add fixture files there when a new adapter needs sample output.
- **Job submission** (`submit_job`) and polling (`wait_for_job`) live in `pgoa/services.py`. The PGOA loop calls them autonomously — no user interaction is required after `PGOAAgent.run()` is invoked.

## Key Files

| File | Purpose |
|------|---------|
| [`agent.toml`](agent.toml) | Per-project config (readonly paths, edit roots, PGOA settings) |
| [`src/claw_backend/config.py`](src/claw_backend/config.py) | Runtime settings (`HPC_ASSISTANT_*` env vars) |
| [`src/claw_backend/project_config.py`](src/claw_backend/project_config.py) | agent.toml loader |
| [`src/claw_backend/pgoa/schema.py`](src/claw_backend/pgoa/schema.py) | All Pydantic models |
| [`src/claw_backend/pgoa/store.py`](src/claw_backend/pgoa/store.py) | Filesystem store (`~/.hpcassist/`) |
| [`src/claw_backend/pgoa/analysis.py`](src/claw_backend/pgoa/analysis.py) | Pure bottleneck classifier |
| [`src/claw_backend/pgoa/dspy_prompts.py`](src/claw_backend/pgoa/dspy_prompts.py) | Optional DSPy signatures |
| [`src/claw_backend/pgoa/edit_dispatcher.py`](src/claw_backend/pgoa/edit_dispatcher.py) | OpenCode subprocess + git diff |
| [`src/claw_backend/pgoa/agent/loop.py`](src/claw_backend/pgoa/agent/loop.py) | PGOAAgent orchestrator |
| [`src/claw_tui/app.py`](src/claw_tui/app.py) | Textual `ClawTUI(App)` — navigation + key bindings |
| [`src/claw_tui/screens/dashboard.py`](src/claw_tui/screens/dashboard.py) | TUI Dashboard (store stats, recent activity) |
| [`src/claw_tui/screens/workloads.py`](src/claw_tui/screens/workloads.py) | TUI Workloads browser (runs table + bottleneck detail) |
| [`src/claw_tui/screens/audit.py`](src/claw_tui/screens/audit.py) | TUI Edit Audit (edit trail + diff viewer) |
| [`src/claw_tui/screens/cluster.py`](src/claw_tui/screens/cluster.py) | TUI Cluster Profile viewer |
| [`src/claw_tui/screens/settings.py`](src/claw_tui/screens/settings.py) | TUI Settings viewer (active env vars) |
| [`src/claw_tui/screens/jobs.py`](src/claw_tui/screens/jobs.py) | TUI Jobs — live squeue table + scontrol detail pane |
| [`src/claw_tui/screens/explorer.py`](src/claw_tui/screens/explorer.py) | TUI Explorer — directory tree + file viewer |
| [`src/claw_tui/screens/chat.py`](src/claw_tui/screens/chat.py) | TUI Chat — conversational LLM interface |
| [`src/claw_tui/screens/hardware_env.py`](src/claw_tui/screens/hardware_env.py) | TUI Hardware/Env — CPU/GPU topology + software modules |
| [`src/claw_tui/app.tcss`](src/claw_tui/app.tcss) | Textual CSS for all TUI screens |
| [`.opencode/tools/_python.ts`](.opencode/tools/_python.ts) | TypeScript→Python bridge for OpenCode plugins |
