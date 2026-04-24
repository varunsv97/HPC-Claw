# HPC Assistant

> Full architecture and component reference: [docs/architecture.md](docs/architecture.md)

## Code Style

- Python 3.14, Pydantic v2 (`BaseModel`, `model_dump_json`, `model_validate_json`, `model_copy`)
- `pydantic-settings` for all runtime config (`AssistantSettings`, `HPC_ASSISTANT_*` env prefix)
- Stdlib `tomllib` for TOML parsing — no third-party TOML library
- `from __future__ import annotations` in every module
- DSPy (`>=3.0`) is optional: import it inside `try/except`, guard Signature class bodies with `if _DSPY_AVAILABLE:`

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
pip install -e backend            # installs hpc_assistant_backend + all deps
pip install pytest                # for running the test suite

# Run tests
cd backend && python -m pytest tests/ -q

# Smoke-test imports (no network required)
python -c "from hpc_assistant_backend.pgoa.agent.loop import PGOAAgent; print('ok')"
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

## Key Files

| File | Purpose |
|------|---------|
| [`agent.toml`](agent.toml) | Per-project config (readonly paths, edit roots, PGOA settings) |
| [`backend/src/hpc_assistant_backend/config.py`](backend/src/hpc_assistant_backend/config.py) | Runtime settings (env vars) |
| [`backend/src/hpc_assistant_backend/project_config.py`](backend/src/hpc_assistant_backend/project_config.py) | agent.toml loader |
| [`backend/src/hpc_assistant_backend/pgoa/schema.py`](backend/src/hpc_assistant_backend/pgoa/schema.py) | All Pydantic models |
| [`backend/src/hpc_assistant_backend/pgoa/store.py`](backend/src/hpc_assistant_backend/pgoa/store.py) | Filesystem store |
| [`backend/src/hpc_assistant_backend/pgoa/analysis.py`](backend/src/hpc_assistant_backend/pgoa/analysis.py) | Pure bottleneck classifier |
| [`backend/src/hpc_assistant_backend/pgoa/dspy_prompts.py`](backend/src/hpc_assistant_backend/pgoa/dspy_prompts.py) | DSPy signatures |
| [`backend/src/hpc_assistant_backend/pgoa/edit_dispatcher.py`](backend/src/hpc_assistant_backend/pgoa/edit_dispatcher.py) | OpenCode subprocess + git diff |
| [`backend/src/hpc_assistant_backend/pgoa/agent/loop.py`](backend/src/hpc_assistant_backend/pgoa/agent/loop.py) | PGOAAgent orchestrator |
| [`.opencode/tools/_python.ts`](.opencode/tools/_python.ts) | TypeScript→Python bridge for OpenCode plugins |
