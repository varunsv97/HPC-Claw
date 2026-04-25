"""Session router for HPC Claw."""

from __future__ import annotations

from claw_backend.assistant.hpc_sidecar import build_hpc_sidecar_routes
from claw_backend.assistant.pgoa_skill import build_pgoa_routes
from claw_backend.assistant.schema import ActionRoute, AssistantMode


def build_repo_routes() -> list[ActionRoute]:
    return [
        ActionRoute(
            action="inspect_repo",
            capability="repo.inspect",
            target="opencode",
            summary="Use OpenCode's native file and search tools to inspect the working tree.",
        ),
        ActionRoute(
            action="edit_repo_files",
            capability="repo.edit",
            target="opencode",
            summary="Perform source edits, patch application, and diff review in OpenCode so the coding flow stays native.",
        ),
        ActionRoute(
            action="run_repo_commands",
            capability="repo.exec",
            target="opencode",
            summary="Run build, test, and helper commands through OpenCode's repo execution flow.",
        ),
        ActionRoute(
            action="generate_slurm_scripts",
            capability="repo.slurm.author",
            target="opencode",
            summary="Draft or revise Slurm scripts and launch wrappers as normal repo edits in OpenCode.",
        ),
    ]


def build_routes(mode: AssistantMode) -> list[ActionRoute]:
    return build_repo_routes() + build_hpc_sidecar_routes() + build_pgoa_routes(mode)
