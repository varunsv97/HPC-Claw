"""Backend-owned HPC sidecar routes.

These are the cluster-native capabilities that should stay behind backend
guardrails instead of being pushed into OpenCode's repo editing flow.
"""

from __future__ import annotations

from claw_backend.assistant.schema import ActionRoute, ApprovalGate


def build_hpc_sidecar_routes() -> list[ActionRoute]:
    return [
        ActionRoute(
            action="discover_cluster",
            capability="cluster.discovery",
            target="backend",
            summary="Collect login-node topology and cached partition metadata under backend guardrails.",
        ),
        ActionRoute(
            action="discover_environment",
            capability="environment.discovery",
            target="backend",
            summary="Inspect the active module system and classify available software stacks.",
        ),
        ActionRoute(
            action="submit_slurm_job",
            capability="jobs.submit",
            target="backend",
            summary="Submit Slurm jobs only through the backend so shared-cluster approvals remain explicit.",
            approval=ApprovalGate(
                required=True,
                reason="submitting a Slurm job affects shared HPC resources",
                commands=["sbatch"],
            ),
        ),
        ActionRoute(
            action="check_job_status",
            capability="jobs.status",
            target="backend",
            summary="Query job state, accounting, and partition placement using Slurm-native commands.",
        ),
        ActionRoute(
            action="read_job_output",
            capability="jobs.logs.read",
            target="backend",
            summary="Read scheduler output and collected logs without bypassing backend guardrails.",
        ),
        ActionRoute(
            action="cancel_job",
            capability="jobs.cancel",
            target="backend",
            summary="Cancel jobs only through the backend so destructive cluster actions always require approval.",
            approval=ApprovalGate(
                required=True,
                reason="cancelling a Slurm job is a destructive action",
                commands=["scancel"],
            ),
        ),
    ]
