"""System prompt and message builders for the PGOA agent."""

from __future__ import annotations

from claw_backend.pgoa.schema import ClusterProfile, ModuleInfo


def _best_spec(m: ModuleInfo) -> str:
    """Return 'name/default_or_newest' or just 'name'."""
    ver = m.default_version or (m.versions[-1] if m.versions else "")
    return f"{m.name}/{ver}" if ver else m.name

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert HPC performance engineer running inside the PGOA optimization agent.

Your goal: minimize {primary_kpi} (unit: {kpi_unit}) for workload '{workload_id}'.
Convergence threshold: stop when improvement per iteration < {kpi_threshold_pct}%.

## Cluster topology
{cluster_section}

## Tool sequence you must follow

### Baseline run (first iteration)
1. create_run — run_type="baseline", workload_id="{workload_id}". Save returned run_id.
2. submit_job — job_script_path is the original job script. Save returned job_id.
3. wait_for_job — pass the job_id. Wait until final_state is returned.
4. If final_state is FAILED or CANCELLED: report the failure and stop.
5. collect_slurm_profile — pass job_id and the run_id from step 1.
6. If the job script ran ncu and produced a CSV: collect_ncu_profile with that path and the same run_id.
7. If the job script ran likwid-perfctr and produced output: collect_likwid_profile with that path and same run_id.
8. analyze_bottlenecks — always after profiling.

### Optimization iteration (repeat until converged)
9.  State your hypothesis explicitly before calling any action tool.
10. propose_slurm_action — one change per iteration.
11. apply_slurm_action — write the modified script; note the new_script_path.
12. create_run — run_type="iteration". Save new run_id.
13. submit_job — job_script_path = new_script_path from step 11. Save new job_id.
14. wait_for_job — wait for completion.
15. If final_state is FAILED or CANCELLED: roll back (discard the run), try a different hypothesis.
16. collect_slurm_profile — pass new job_id and new run_id.
17. collect_ncu_profile / collect_likwid_profile if instrumentation was present.
18. compare_runs — from_run_id = baseline run_id, to_run_id = current run_id.
19. Evaluate delta. If |delta_pct| < {kpi_threshold_pct}%: output summary and stop.
    Otherwise state next hypothesis and go to step 9.

## Rules (strictly enforced)
- Always call create_run BEFORE submit_job for every run (baseline and every iteration).
- Never reuse a run_id across different job submissions.
- Always call wait_for_job and confirm completion BEFORE collect_slurm_profile.
- Change ONE parameter group per iteration. Never bundle unrelated changes.
- Always state expected improvement before calling propose_slurm_action.
- If a run degrades performance, call compare_runs anyway, acknowledge the regression,
  roll back to the last good script, and try a different hypothesis.
- Never call apply_slurm_action without a preceding propose_slurm_action in this turn.
- Never write files outside the approved filesystem roots.
- Prefer conservative, targeted changes. Do not modify the source application; only
  modify the job script unless a code-level bottleneck is explicitly identified.

## Output format for final summary
When converged, output a markdown table:
| Iteration | Action | KPI Delta | Direction |
followed by a one-paragraph conclusion.
"""


def _format_cluster_section(profile: ClusterProfile | None) -> str:
    if profile is None:
        return "(cluster topology not yet discovered)"

    lines: list[str] = [f"Cluster: {profile.cluster_name}"]
    hw = profile.login_node_hardware
    if hw.cpu_model:
        lines.append(f"CPU model: {hw.cpu_model}")
    if hw.sockets_per_node and hw.cores_per_socket and hw.threads_per_core:
        lines.append(
            f"CPU topology: {hw.sockets_per_node} socket(s) × "
            f"{hw.cores_per_socket} cores × {hw.threads_per_core} threads "
            f"= {hw.cpus_per_node or '?'} logical CPUs/node"
        )
    if hw.numa_nodes:
        lines.append(f"NUMA nodes/socket: {hw.numa_nodes // (hw.sockets_per_node or 1)}")
    if hw.memory_gb_per_node:
        lines.append(f"Memory/node: {hw.memory_gb_per_node} GB")
    if hw.gpu_model:
        lines.append(
            f"GPU: {hw.gpu_model} (arch={hw.gpu_arch}, "
            f"mem={hw.gpu_memory_gb} GB, SMs={hw.gpu_sm_count})"
        )
    if hw.cache_l3_mb:
        lines.append(f"L3 cache: {hw.cache_l3_mb} MB")

    if profile.partitions:
        lines.append("")
        lines.append("Partitions:")
        for p in profile.partitions:
            probe_note = " [probe data]" if p.hardware_from_probe else " [scontrol estimate]"
            parts_hw = p.hardware
            hw_str_parts = []
            if parts_hw.cpus_per_node:
                hw_str_parts.append(f"{parts_hw.cpus_per_node} CPUs")
            if parts_hw.gpus_per_node:
                hw_str_parts.append(f"{parts_hw.gpus_per_node} GPUs")
            if parts_hw.memory_gb_per_node:
                hw_str_parts.append(f"{parts_hw.memory_gb_per_node} GB RAM")
            hw_str = (", ".join(hw_str_parts) + probe_note) if hw_str_parts else probe_note.strip()
            lines.append(
                f"  {p.name}: {p.total_nodes or '?'} nodes "
                f"({p.idle_nodes or '?'} idle), state={p.state} — {hw_str}"
            )

    se = profile.software_env
    if se and se.module_system != "none":
        lines.append("")
        ver_str = f" v{se.lmod_version}" if se.lmod_version else (f" v{se.tmod_version}" if se.tmod_version else "")
        spider_note = " (full hierarchy)" if se.spider_complete else " (Core tier only)"
        lines.append(f"Module system: {se.module_system}{ver_str}{spider_note}")

        if se.compilers:
            lines.append("Compilers:     " + ", ".join(_best_spec(m) for m in se.compilers))
        if se.mpi_libraries:
            lines.append("MPI:           " + ", ".join(_best_spec(m) for m in se.mpi_libraries))
        if se.gpu_toolkits:
            lines.append("GPU toolkits:  " + ", ".join(_best_spec(m) for m in se.gpu_toolkits))
        if se.math_libraries:
            lines.append("Math libs:     " + ", ".join(_best_spec(m) for m in se.math_libraries[:8]))

        if se.toolchains:
            lines.append("Toolchains (load sequences):")
            for tc in se.toolchains[:12]:  # cap to avoid overwhelming the prompt
                seq = "  →  ".join(tc.load_sequence)
                lines.append(f"  [{tc.name}]  {seq}")

    return "\n".join(lines)


def build_system_prompt(
    primary_kpi: str,
    kpi_unit: str,
    workload_id: str,
    kpi_threshold_pct: float,
    cluster_profile: ClusterProfile | None = None,
) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        primary_kpi=primary_kpi,
        kpi_unit=kpi_unit,
        workload_id=workload_id,
        kpi_threshold_pct=kpi_threshold_pct,
        cluster_section=_format_cluster_section(cluster_profile),
    )


def build_initial_user_message(
    workload_id: str,
    job_script_path: str,
    job_script_content: str,
    kpi_metric: str,
    kpi_unit: str,
) -> str:
    script_preview_lines = job_script_content.splitlines()[:60]
    script_preview = "\n".join(script_preview_lines)
    if len(job_script_content.splitlines()) > 60:
        script_preview += "\n... (truncated)"

    return (
        f"Workload ID: {workload_id}\n"
        f"Job script path: {job_script_path}\n"
        f"KPI goal: minimize {kpi_metric} ({kpi_unit})\n\n"
        f"Job script (first 60 lines):\n```bash\n{script_preview}\n```\n\n"
        "Begin the optimization loop now. Start with a baseline run: "
        "call create_run (run_type=baseline), then submit_job with the job script path above, "
        "then wait_for_job, then collect_slurm_profile."
    )
