"""PGOA skill routing.

PGOA is modeled as a specialized optimization skill inside the broader coding
assistant, not as a separate agent runtime.
"""

from __future__ import annotations

from hpc_assistant_backend.assistant.schema import ActionRoute, ApprovalGate, AssistantMode


def build_pgoa_routes(mode: AssistantMode) -> list[ActionRoute]:
    summary = (
        "Treat profile-guided optimization as a skill layered over OpenCode edits and "
        "backend-managed profiling, comparison, and guarded job actions."
    )
    if mode == "pgoa":
        summary = (
            "Use PGOA as the primary optimization mode while still delegating repo edits "
            "to OpenCode and cluster-facing work to the backend."
        )

    return [
        ActionRoute(
            action="optimize_workload",
            capability="pgoa.optimize",
            target="pgoa_skill",
            delegates_to=["opencode", "backend"],
            summary=summary,
            approval=ApprovalGate(
                required=True,
                reason="optimization loops may require submitting comparison jobs on the cluster",
                commands=["sbatch"],
            ),
        )
    ]
