"""Pure deterministic bottleneck analysis functions for PGOA."""

from __future__ import annotations

from hpc_assistant_backend.pgoa.schema import BottleneckReport, ProfileBundle


def analyze_bottlenecks(bundle: ProfileBundle) -> BottleneckReport:
    """Analyse a ProfileBundle and return the primary bottleneck report.

    Priority order: first matching rule wins.
    """
    details: dict[str, str] = {}
    slurm = bundle.slurm
    compute = bundle.compute

    # Populate reusable summary strings
    if slurm is not None:
        elapsed = f"{slurm.elapsed_s:.1f}s" if slurm.elapsed_s is not None else "?"
        cpus = str(slurm.alloc_cpus) if slurm.alloc_cpus is not None else "?"
        rss = f"{slurm.max_rss_mb:.0f}MB" if slurm.max_rss_mb is not None else "?"
        details["slurm_summary"] = (
            f"elapsed={elapsed} alloc_cpus={cpus} max_rss={rss}"
        )

    if compute is not None:
        occ = (
            f"{compute.sm_occupancy_pct:.1f}%"
            if compute.sm_occupancy_pct is not None
            else "?"
        )
        bw = (
            f"{compute.memory_bw_utilization_pct:.1f}%"
            if compute.memory_bw_utilization_pct is not None
            else "?"
        )
        details["compute_summary"] = (
            f"roofline={compute.roofline_position} sm_occ={occ} mem_bw_util={bw}"
        )

    # --- Rule 1: memory-bound GPU ---
    if (
        compute is not None
        and compute.roofline_position == "memory_bound"
        and compute.memory_bw_utilization_pct is not None
        and compute.memory_bw_utilization_pct > 70
    ):
        return BottleneckReport(
            run_id=bundle.run_id,
            primary_bottleneck="memory_bound_gpu",
            details=details,
            recommended_action_hint=(
                "Consider BF16/FP8 precision or fused kernels to reduce memory traffic"
            ),
        )

    # --- Rule 2: compute-bound GPU ---
    if compute is not None and compute.roofline_position == "compute_bound":
        return BottleneckReport(
            run_id=bundle.run_id,
            primary_bottleneck="compute_bound_gpu",
            details=details,
            recommended_action_hint=(
                "Check SM occupancy; consider increasing batch size or tile sizes"
            ),
        )

    # --- Rule 3: latency-bound GPU ---
    if compute is not None and compute.roofline_position == "latency_bound":
        return BottleneckReport(
            run_id=bundle.run_id,
            primary_bottleneck="latency_bound_gpu",
            details=details,
            recommended_action_hint=(
                "SM occupancy < 20%; likely too-small kernel launches or excessive synchronization"
            ),
        )

    # --- Rule 4: high RSS ---
    if (
        slurm is not None
        and slurm.max_rss_mb is not None
        and slurm.alloc_cpus is not None
        and slurm.max_rss_mb > slurm.alloc_cpus * 3800
    ):
        return BottleneckReport(
            run_id=bundle.run_id,
            primary_bottleneck="high_rss",
            details=details,
            recommended_action_hint=(
                "Job is memory-heavy; check NUMA binding (mem_bind=local) and hugepages"
            ),
        )

    # --- Rule 5: MPI binding (low CPU utilization with multiple tasks) ---
    if (
        slurm is not None
        and slurm.avg_cpu_pct is not None
        and slurm.avg_cpu_pct < 60
        and slurm.n_tasks is not None
        and slurm.n_tasks > 1
    ):
        return BottleneckReport(
            run_id=bundle.run_id,
            primary_bottleneck="mpi_binding",
            details=details,
            recommended_action_hint=(
                "Low CPU utilization with multiple tasks suggests MPI binding or imbalance issue"
            ),
        )

    # --- Rule 6: insufficient data ---
    if slurm is not None and compute is None and bundle.cpu_perf is None:
        return BottleneckReport(
            run_id=bundle.run_id,
            primary_bottleneck="insufficient_data",
            details=details,
            recommended_action_hint=(
                "Only Slurm telemetry available; collect NCU or LIKWID profiles for deeper analysis"
            ),
        )

    # --- Rule 7: no bottleneck detected ---
    return BottleneckReport(
        run_id=bundle.run_id,
        primary_bottleneck="none_detected",
        details=details,
        recommended_action_hint=(
            "No clear bottleneck detected; consider profiling with additional tools"
        ),
    )
