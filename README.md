# hpc-assistant

> Status: This project is under active development and is not production-ready yet.

`hpc-assistant` is evolving into a cluster-side HPC AI assistant. The current
prototype focuses on safe discovery and testability:

- cluster discovery from the login node
- module/software environment discovery
- optional guardrailed probe jobs for real compute-node topology
- OpenCode tool shims as an add-on execution surface
- a lightweight Rust TUI for operator testing

The first prototype intentionally keeps the dependency footprint small. The
backend uses the OpenAI Python SDK where model access is needed, but it does
not pull in larger agent stacks such as LangChain, DSPy, or PydanticAI yet,
because the current discovery/testing workflow does not need them.

## Prototype Commands

After installing the backend package, these commands are the primary operator
entry points:

- `hpc-assistant-backend doctor`
- `hpc-assistant-backend discover-cluster`
- `hpc-assistant-backend discover-env`
- `hpc-assistant-backend probe-cluster --partition <name> --yes`

OpenCode support is still available:

- `hpc-assistant-backend show-opencode-tools`
- `hpc-assistant-backend sync-opencode-tools`

## Install

Python backend:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e backend
```

Rust TUI:

```bash
cargo build --manifest-path tui/Cargo.toml
```

Environment example:

```bash
cp .env.example .env
```

Edit `.env` to set at least:

- `HPC_ASSISTANT_FILESYSTEM_ROOTS`
- `HPC_ASSISTANT_PGOA_STORE_PATH` (defaults to `~/.hpcassist`)
- `HPC_ASSISTANT_OPENAI_API_KEY` if you want to exercise model-backed features

## Test Run On A Cluster

Start with the read-only checks:

```bash
make prototype-doctor
make prototype-cluster
make prototype-env
```

If those look correct and you explicitly want compute-node topology, run:

```bash
PYTHONPATH=backend/src .venv/bin/python -m hpc_assistant_backend probe-cluster --partition <partition> --yes
```

The Rust TUI is a thin wrapper over the same commands:

```bash
make tui-run
```

Hotkeys:

- `1` doctor
- `2` cluster
- `3` environment
- `r` refresh current panel
- `q` quit

## Guardrails

The prototype is conservative by default:

- filesystem writes are limited to configured roots
- cluster probe jobs are disabled unless explicitly enabled
- OpenCode remains an add-on tool surface, not the core orchestration layer
- the experimental PGOA loop does not submit or cancel jobs on its own

## Verification

```bash
make test
python3 -m py_compile backend/src/hpc_assistant_backend/*.py
python3 -m py_compile backend/src/hpc_assistant_backend/pgoa/*.py
```
