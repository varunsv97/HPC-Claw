"""Repo-aware assistant session service."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hpc_assistant_backend.assistant.opencode_adapter import describe_opencode_workspace
from hpc_assistant_backend.assistant.router import build_routes
from hpc_assistant_backend.assistant.schema import AssistantMode, AssistantSession, RepoContext
from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.opencode_tools import sync_opencode_project
from hpc_assistant_backend.path_access import resolve_allowed_path

_REPO_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "Cargo.toml",
    "package.json",
    "CMakeLists.txt",
    "Makefile",
    "README.md",
)


def build_assistant_session(
    settings: AssistantSettings,
    *,
    repo_root: str,
    goal: str | None = None,
    mode: AssistantMode = "coding",
    sync_opencode: bool = False,
) -> AssistantSession:
    repo_path = resolve_allowed_path(repo_root, settings)
    if not repo_path.exists():
        raise FileNotFoundError(repo_path)
    if not repo_path.is_dir():
        raise NotADirectoryError(repo_path)

    synced_files: list[str] = []
    if sync_opencode:
        synced_files = [str(path) for path in sync_opencode_project(repo_path)]

    opencode = describe_opencode_workspace(repo_path)
    created_at = datetime.now(tz=UTC)
    session_id = f"{mode}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    session_path = _session_path(settings, session_id)

    session = AssistantSession(
        session_id=session_id,
        created_at=created_at,
        mode=mode,
        goal=goal.strip() if goal else None,
        repo=RepoContext(
            root=str(repo_path),
            name=repo_path.name,
            has_git=(repo_path / ".git").exists(),
            guardrail_allowed=True,
            detected_files=[
                marker
                for marker in _REPO_MARKERS
                if (repo_path / marker).exists()
            ],
        ),
        opencode=opencode,
        store_root=str(Path(settings.pgoa_store_path).expanduser()),
        session_path=str(session_path),
        routes=build_routes(mode),
        synced_opencode_files=synced_files,
        next_steps=_build_next_steps(repo_path, goal, opencode.ready, mode, sync_opencode),
    )
    _write_json_atomic(session_path, session.model_dump(mode="json"))
    return session


def _build_next_steps(
    repo_root: Path,
    goal: str | None,
    opencode_ready: bool,
    mode: AssistantMode,
    sync_opencode: bool,
) -> list[str]:
    steps = [
        "Use OpenCode as the repo editing and command execution surface for code-centric work.",
        "Keep Slurm submission, status checks, and other cluster actions behind backend guardrails.",
    ]
    if not opencode_ready and not sync_opencode:
        steps.append(
            f"Prepare repo-local OpenCode shims with `hpc-assistant-backend sync-opencode-tools --root {repo_root}`."
        )
    if mode == "pgoa":
        steps.append(
            "Use the PGOA skill to drive optimization loops while still delegating edits to OpenCode."
        )
    if goal:
        steps.append(f"Current goal: {goal.strip()}")
    return steps


def _session_path(settings: AssistantSettings, session_id: str) -> Path:
    root = Path(settings.pgoa_store_path).expanduser() / "assistant" / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{session_id}.json"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
