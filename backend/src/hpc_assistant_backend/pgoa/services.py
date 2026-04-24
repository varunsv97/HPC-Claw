"""Shared service layer for PGOA operations.

This module centralizes the deterministic parts of PGOA so both the OpenCode
tool surface and the agent tool registry use the same implementation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.path_access import resolve_allowed_path, resolve_writable_path
from hpc_assistant_backend.pgoa.actions import (
    apply_slurm_binding,
    validate_slurm_binding_params,
)
from hpc_assistant_backend.pgoa.adapters.likwid import LIKWIDAdapter
from hpc_assistant_backend.pgoa.adapters.ncu import NCUAdapter
from hpc_assistant_backend.pgoa.adapters.slurm import SlurmAdapter
from hpc_assistant_backend.pgoa.analysis import analyze_bottlenecks
from hpc_assistant_backend.pgoa.cluster_discovery import (
    collect_login_node_info,
    detect_cluster_name,
    is_stale,
)
from hpc_assistant_backend.pgoa.schema import (
    ActionProposal,
    ClusterProfile,
    DeltaReport,
    HardwareInfo,
    KPIMetrics,
    ProfileBundle,
    RunHandle,
)
from hpc_assistant_backend.pgoa.store import ExperimentStore


def discover_cluster(
    settings: AssistantSettings,
    store: ExperimentStore,
    *,
    force_refresh: bool = False,
) -> ClusterProfile:
    """Return the cached ClusterProfile for this cluster, refreshing if stale.

    Only performs login-node discovery (no probe jobs). Probe jobs require
    user approval and must be triggered explicitly via
    ``cluster_discovery.run_probe_jobs()``.
    """
    # Check cache first — collect_login_node_info runs sinfo + scontrol + module
    # avail/spider and is expensive; skip all of that when the cache is fresh.
    if not force_refresh:
        cluster_name = detect_cluster_name(settings)
        cached = store.load_cluster_profile(cluster_name)
        if cached is not None and not is_stale(cached):
            return cached

    fresh = collect_login_node_info(settings)
    store.save_cluster_profile(fresh)
    return fresh


def build_kpi(
    metric: str,
    value: float | str,
    unit: str,
    *,
    lower_is_better: bool = True,
) -> KPIMetrics:
    return KPIMetrics(
        primary_metric=metric,
        value=float(value),
        unit=unit,
        lower_is_better=lower_is_better,
    )


def collect_profile(
    settings: AssistantSettings,
    store: ExperimentStore,
    *,
    workload_id: str,
    job_id: int,
    kpi: KPIMetrics,
    run_type: Literal["baseline", "iteration"] = "baseline",
    iteration: int | None = None,
    ncu_csv_path: str | None = None,
    likwid_output_path: str | None = None,
) -> tuple[RunHandle, ProfileBundle]:
    handle = store.create_run(workload_id, run_type, iteration=iteration)
    bundle = collect_slurm_profile(
        settings,
        store,
        workload_id=workload_id,
        run_id=handle.run_id,
        job_id=job_id,
        kpi=kpi,
    )
    if ncu_csv_path:
        bundle = collect_ncu_profile(
            store,
            workload_id=workload_id,
            run_id=handle.run_id,
            ncu_csv_path=ncu_csv_path,
            kpi=kpi,
        )
    if likwid_output_path:
        bundle = collect_likwid_profile(
            store,
            workload_id=workload_id,
            run_id=handle.run_id,
            likwid_output_path=likwid_output_path,
            kpi=kpi,
        )
    return handle, bundle


def collect_slurm_profile(
    settings: AssistantSettings,
    store: ExperimentStore,
    *,
    workload_id: str,
    run_id: str,
    job_id: int,
    kpi: KPIMetrics,
) -> ProfileBundle:
    partial = SlurmAdapter(settings).collect(job_id=job_id, kpi=kpi)
    return _merge_and_save_bundle(store, workload_id, run_id, partial, kpi)


def collect_ncu_profile(
    store: ExperimentStore,
    *,
    workload_id: str,
    run_id: str,
    ncu_csv_path: str,
    kpi: KPIMetrics,
) -> ProfileBundle:
    partial = NCUAdapter().collect(ncu_csv_path=ncu_csv_path, kpi=kpi)
    return _merge_and_save_bundle(store, workload_id, run_id, partial, kpi)


def collect_likwid_profile(
    store: ExperimentStore,
    *,
    workload_id: str,
    run_id: str,
    likwid_output_path: str,
    kpi: KPIMetrics,
) -> ProfileBundle:
    partial = LIKWIDAdapter().collect(likwid_output_path=likwid_output_path, kpi=kpi)
    return _merge_and_save_bundle(store, workload_id, run_id, partial, kpi)


def analyze_run(
    store: ExperimentStore,
    *,
    workload_id: str,
    run_id: str,
):
    bundle = store.load_bundle(workload_id, run_id)
    return analyze_bottlenecks(bundle)


def record_slurm_action(
    store: ExperimentStore,
    *,
    workload_id: str,
    run_id: str,
    action_name: str,
    parameters: dict,
    rationale: str,
    expected_improvement_pct: float | None = None,
) -> ActionProposal:
    proposal = ActionProposal(
        action_name=action_name,
        parameters=parameters,
        rationale=rationale,
        expected_improvement_pct=expected_improvement_pct,
    )
    store.save_rationale(
        workload_id,
        run_id,
        f"Action: {proposal.action_name}\n"
        f"Parameters: {proposal.parameters}\n"
        f"Rationale: {proposal.rationale}\n"
        f"Expected improvement: {proposal.expected_improvement_pct}%",
    )
    return proposal


def apply_binding_change(
    settings: AssistantSettings,
    store: ExperimentStore,
    *,
    workload_id: str,
    run_id: str,
    job_script_path: str,
    output_script_path: str,
    ntasks_per_node: int | None = None,
    ntasks_per_socket: int | None = None,
    cpus_per_task: int | None = None,
    mem_bind: str | None = None,
    cpu_bind: str | None = None,
    exclusive: bool = False,
) -> tuple[str, list[str]]:
    script_path = resolve_allowed_path(job_script_path, settings)
    script_content = script_path.read_text(encoding="utf-8")

    errors = validate_slurm_binding_params(
        ntasks_per_node=ntasks_per_node,
        ntasks_per_socket=ntasks_per_socket,
        cpus_per_task=cpus_per_task,
        mem_bind=mem_bind,
        cpu_bind=cpu_bind,
        exclusive=exclusive,
    )
    if errors:
        raise ValueError("; ".join(errors))

    modified, changes = apply_slurm_binding(
        job_script_content=script_content,
        ntasks_per_node=ntasks_per_node,
        ntasks_per_socket=ntasks_per_socket,
        cpus_per_task=cpus_per_task,
        mem_bind=mem_bind,
        cpu_bind=cpu_bind,
        exclusive=exclusive,
    )

    output_path = resolve_writable_path(output_script_path, settings)
    output_path.write_text(modified, encoding="utf-8")
    store.save_job_script(workload_id, run_id, modified)
    return str(output_path), changes


def compare_runs(
    store: ExperimentStore,
    *,
    workload_id: str,
    from_run_id: str,
    to_run_id: str,
    action_applied: str,
) -> DeltaReport:
    delta = store.compute_delta(
        workload_id=workload_id,
        from_run_id=from_run_id,
        to_run_id=to_run_id,
        action_applied=action_applied,
    )
    store.save_delta(workload_id, to_run_id, delta)
    return delta


def list_runs(
    store: ExperimentStore,
    *,
    workload_id: str,
) -> list[RunHandle]:
    return store.list_runs(workload_id)


def _merge_and_save_bundle(
    store: ExperimentStore,
    workload_id: str,
    run_id: str,
    partial: ProfileBundle,
    kpi: KPIMetrics,
) -> ProfileBundle:
    try:
        existing = store.load_bundle(workload_id, run_id)
    except FileNotFoundError:
        existing = _empty_bundle(run_id, kpi)
    merged = _merge_bundles(existing, partial, run_id=run_id)
    store.save_bundle(workload_id, run_id, merged)
    return merged


def _empty_bundle(run_id: str, kpi: KPIMetrics) -> ProfileBundle:
    return ProfileBundle(
        run_id=run_id,
        timestamp=datetime.now(tz=UTC),
        hardware=HardwareInfo(),
        kpi=kpi,
    )


def _merge_bundles(
    existing: ProfileBundle,
    partial: ProfileBundle,
    *,
    run_id: str,
) -> ProfileBundle:
    existing_hw = existing.hardware
    partial_hw = partial.hardware
    hardware = partial_hw if _has_hardware_data(partial_hw) else existing_hw

    workload_type = (
        partial.workload_type
        if partial.workload_type != "unknown"
        else existing.workload_type
    )

    return existing.model_copy(
        update={
            "run_id": run_id,
            "timestamp": partial.timestamp,
            "workload_type": workload_type,
            "hardware": hardware,
            "slurm": partial.slurm or existing.slurm,
            "compute": partial.compute or existing.compute,
            "cpu_perf": partial.cpu_perf or existing.cpu_perf,
            "kpi": partial.kpi,
            "raw_sources": {**existing.raw_sources, **partial.raw_sources},
        }
    )


def _has_hardware_data(hardware: HardwareInfo) -> bool:
    values = hardware.model_dump()
    return any(value not in (None, [], {}, "unknown") for value in values.values())
