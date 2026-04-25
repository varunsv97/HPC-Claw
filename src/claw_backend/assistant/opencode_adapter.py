"""OpenCode workspace inspection helpers.

This layer is intentionally thin: OpenCode remains the repo-native editing and
execution engine, while the backend only reports readiness and optional setup.
"""

from __future__ import annotations

from pathlib import Path

from claw_backend.assistant.schema import OpenCodeWorkspace


def describe_opencode_workspace(repo_root: Path) -> OpenCodeWorkspace:
    config_path = repo_root / "opencode.json"
    tools_dir = repo_root / ".opencode" / "tools"
    pgoa_tools = tools_dir / "pgoa.ts"
    notes: list[str] = []
    if not config_path.exists():
        notes.append("repo-local opencode.json is missing")
    if not tools_dir.exists():
        notes.append("repo-local .opencode/tools directory is missing")
    if tools_dir.exists() and not pgoa_tools.exists():
        notes.append("PGOA OpenCode shims are missing from .opencode/tools")

    return OpenCodeWorkspace(
        repo_root=str(repo_root),
        config_present=config_path.exists(),
        tools_dir_present=tools_dir.exists(),
        pgoa_tools_present=pgoa_tools.exists(),
        ready=config_path.exists() and tools_dir.exists() and pgoa_tools.exists(),
        sync_command=f"hpc-assistant-backend sync-opencode-tools --root {repo_root}",
        notes=notes,
    )
