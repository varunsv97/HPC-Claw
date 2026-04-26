"""Shared service layer for PGOA operations.

This module centralizes the deterministic parts of PGOA so both the OpenCode
tool surface and the agent tool registry use the same implementation.
"""

from __future__ import annotations

import re
import subprocess
import time
from datetime import UTC, datetime
from typing import Literal

from claw_backend.config import AssistantSettings
from claw_backend.path_access import resolve_allowed_path, resolve_writable_path
from claw_backend.pgoa.actions import (
    apply_slurm_binding,
    validate_slurm_binding_params,
)
from claw_backend.pgoa.adapters.likwid import LIKWIDAdapter
from claw_backend.pgoa.adapters.ncu import NCUAdapter
from claw_backend.pgoa.adapters.slurm import SlurmAdapter
from claw_backend.pgoa.analysis import analyze_bottlenecks
from claw_backend.pgoa.cluster_discovery import (
    collect_login_node_info,
    detect_cluster_name,
    is_stale,
)
from claw_backend.pgoa.schema import (
    ActionProposal,
    ClusterProfile,
    DeltaReport,
    HardwareInfo,
    KPIMetrics,
    ProfileBundle,
    RunHandle,
)
from claw_backend.pgoa.store import ExperimentStore


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


_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")
_SQUEUE_STATE_FORMAT = "%T"


def submit_job(
    settings: AssistantSettings,
    *,
    job_script_path: str,
) -> int:
    """Submit a Slurm job via sbatch and return the numeric job ID.

    The job script path must be within the approved filesystem roots.
    """
    script = resolve_allowed_path(job_script_path, settings)
    result = subprocess.run(
        ["sbatch", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=settings.command_timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    m = _JOB_ID_RE.search(result.stdout)
    if not m:
        raise RuntimeError(
            f"Could not parse job ID from sbatch output: {result.stdout!r}"
        )
    return int(m.group(1))


def wait_for_job(
    settings: AssistantSettings,
    *,
    job_id: int,
    poll_interval_s: float = 30.0,
    timeout_s: float = 3600.0,
) -> str:
    """Block until job *job_id* leaves the Slurm queue; return final sacct state.

    Raises :exc:`TimeoutError` if the job does not complete within *timeout_s*.
    Returns the first state token from sacct (e.g. ``"COMPLETED"``, ``"FAILED"``).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "squeue",
                "-j",
                str(job_id),
                "--noheader",
                "-o",
                _SQUEUE_STATE_FORMAT,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
        state_line = result.stdout.strip()
        if not state_line:
            # Job no longer in the queue — finished (any terminal state)
            break
        time.sleep(poll_interval_s)
    else:
        raise TimeoutError(
            f"Job {job_id} did not complete within {timeout_s:.0f}s"
        )

    # Retrieve final state from sacct (accounting DB, survives queue eviction)
    sacct = subprocess.run(
        [
            "sacct",
            "-j",
            str(job_id),
            "--noheader",
            "--parsable2",
            "--format=State",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=settings.command_timeout_seconds,
    )
    # sacct output has one line per job step; the first line is the main job
    states = [ln.strip() for ln in sacct.stdout.splitlines() if ln.strip()]
    return states[0] if states else "UNKNOWN"


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
