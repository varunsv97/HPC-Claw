# hpc-assistant

`hpc-assistant` is a Python 3.14 HPC assistant with two local pieces:

- a Python backend that talks to an OpenAI-compatible chat endpoint, calls
  cluster tools, and pauses mutating actions for approval
- a Textual TUI that gives you chat, approvals, a file explorer, and a code
  editor in one terminal UI

The repo now targets a real cluster-user workflow rather than a placeholder
demo service.

## What It Can Do

- inspect Slurm queue and job state
- inspect module availability and active modules
- browse allowed filesystem roots
- open and edit files from the TUI
- require approval before mutating tool calls like `scancel`, `module load`,
  or filesystem removal
- connect to any OpenAI-compatible endpoint by URL, model, and token

## Python Target

Both packages target Python `3.14`:

- `backend/pyproject.toml`
- `tui/pyproject.toml`

The repo launcher uses `.venv/bin/python` when available and exports
`PYTHON_GIL=1` by default for compatibility.

## Install

Inside the project venv:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e backend -e tui
```

## Initialize For A Cluster User

Create a user config with your OpenAI-compatible endpoint and token:

```bash
PYTHONPATH=backend/src .venv/bin/python -m hpc_assistant_backend \
  --config ~/.config/hpc-assistant/config.toml \
  init \
  --base-url https://api.example.com/v1 \
  --model gpt-4.1-mini \
  --api-key 'replace-me' \
  --filesystem-root ~ \
  --filesystem-root ~/scratch \
  --force
```

Useful flags:

- `--api-key-env OPENAI_API_KEY` keeps an env-var fallback in the config
- `--no-approval-required` disables review gates for mutating tool calls
- `--thread-store` and `--long-term-store` move backend data under a custom path

## Run

Use the repo launcher:

```bash
.venv/bin/python scripts/dev.py backend
.venv/bin/python scripts/dev.py tui
.venv/bin/python scripts/dev.py app
```

Or run the pieces directly:

```bash
PYTHONPATH=backend/src .venv/bin/python -m hpc_assistant_backend serve
PYTHONPATH=tui/src .venv/bin/python -m hpc_assistant_tui --backend-url http://127.0.0.1:8765
```

## Backend Surface

HTTP endpoints:

- `GET /healthz`
- `GET /session-info`
- `POST /api/session`
- `GET /api/session/{session_id}`
- `POST /api/session/{session_id}/prompt`
- `POST /api/session/{session_id}/approvals/{approval_id}`
- `GET /api/files?path=...`
- `GET /api/file?path=...`
- `POST /api/file`

CLI commands:

- `show-config`
- `show-tools`
- `serve`
- `invoke`
- `stdio`
- `init`

## TUI Notes

The TUI keeps backend calls off the UI thread with worker threads. It also
persists local UI state so it can restore the last open session and editor
buffer metadata on restart.

Current workspace panes:

- chat conversation
- tool activity
- approval review
- file explorer
- code editor
- backend status

## Verification

```bash
make test
PYTHON_GIL=1 .venv/bin/python scripts/dev.py --help
PYTHONPATH=backend/src .venv/bin/python -m hpc_assistant_backend init --help
```
