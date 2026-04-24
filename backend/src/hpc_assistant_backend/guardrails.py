"""Operational guardrails for running the assistant on shared HPC systems."""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.path_access import configured_filesystem_roots
from hpc_assistant_backend.pgoa.env_discovery import detect_module_system

_DISCOVERY_COMMANDS = (
    "sinfo",
    "scontrol",
    "sacct",
    "sbatch",
    "scancel",
    "nvidia-smi",
    "hwloc-ls",
    "lstopo",
    "lua",
    "bash",
)


def require_cluster_probe_permission(
    settings: AssistantSettings,
    *,
    explicit_yes: bool,
) -> None:
    """Raise when the caller has not explicitly enabled probe-job execution."""
    if explicit_yes:
        return
    if settings.allow_cluster_probe_jobs:
        return
    raise PermissionError(
        "cluster probe jobs are disabled by default; pass --yes or set "
        "HPC_ASSISTANT_ALLOW_CLUSTER_PROBE_JOBS=true to allow probe-job submission"
    )


def build_doctor_report(settings: AssistantSettings) -> dict[str, Any]:
    """Return a lightweight environment/guardrail summary for prototype testing."""
    module_kind, module_version = detect_module_system(settings)
    store_path = Path(settings.pgoa_store_path).expanduser()
    command_status = {
        name: shutil.which(name) is not None
        for name in _DISCOVERY_COMMANDS
    }
    warnings: list[str] = []
    if not command_status["sinfo"]:
        warnings.append("sinfo not found; Slurm partition discovery will be unavailable")
    if module_kind == "none":
        warnings.append("no module system detected; environment discovery will be limited")
    if not settings.allow_cluster_probe_jobs:
        warnings.append("cluster probe jobs are disabled by default")

    return {
        "ok": True,
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "settings": {
            "command_timeout_seconds": settings.command_timeout_seconds,
            "openai_model": settings.openai_model,
            "openai_configured": bool(settings.openai_api_key),
            "pgoa_store_path": str(store_path),
            "filesystem_roots": [str(path) for path in configured_filesystem_roots(settings)],
            "guardrails_enabled": settings.guardrails_enabled,
            "allow_cluster_probe_jobs": settings.allow_cluster_probe_jobs,
        },
        "module_system": {
            "kind": module_kind,
            "version": module_version,
        },
        "commands": command_status,
        "warnings": warnings,
    }
