"""Pydantic v2 data models for the PGOA module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class HardwareInfo(BaseModel):
    # Node-level counts
    nodes: int | None = None
    cpus_per_node: int | None = None
    gpus_per_node: int | None = None

    # CPU topology
    cpu_arch: str | None = None          # "x86_64", "aarch64"
    cpu_model: str | None = None         # "Intel Xeon Gold 6140"
    cpu_features: list[str] = []         # HPC-relevant ISA flags, e.g. ["avx2", "avx512f"]
    sockets_per_node: int | None = None
    cores_per_socket: int | None = None
    threads_per_core: int | None = None
    numa_nodes: int | None = None

    # Memory
    memory_gb_per_node: float | None = None

    # Cache (per-core, from cpu0)
    cache_l1d_kb: int | None = None
    cache_l2_kb: int | None = None
    cache_l3_mb: float | None = None     # shared LLC in MB

    # GPU
    gpu_arch: str | None = None          # "Ampere", "Hopper", "Volta" …
    gpu_model: str | None = None         # "NVIDIA A100 80GB PCIe"
    gpu_memory_gb: float | None = None
    gpu_sm_count: int | None = None
    gpu_compute_capability: str | None = None  # "8.0", "9.0" …

    # Runtime hardware pressure/utilization (PGOA samples per run)
    visible_gpus: int | None = None
    gpu_utilization_pct: float | None = None
    gpu_memory_used_gb: float | None = None
    gpu_memory_utilization_pct: float | None = None
    loadavg_1m: float | None = None
    cpu_pressure_avg10: float | None = None
    memory_available_gb: float | None = None


class KernelStat(BaseModel):
    name: str
    duration_pct: float
    sm_occupancy_pct: float | None = None
    memory_throughput_pct: float | None = None


class SlurmMetrics(BaseModel):
    job_id: int
    job_name: str | None = None
    elapsed_s: float | None = None
    cpu_time_raw_s: float | None = None
    alloc_cpus: int | None = None
    alloc_nodes: int | None = None
    max_rss_mb: float | None = None
    avg_cpu_pct: float | None = None
    exit_code: str | None = None
    state: str | None = None
    partition: str | None = None
    n_tasks: int | None = None


class ComputeMetrics(BaseModel):
    roofline_position: Literal[
        "memory_bound", "compute_bound", "latency_bound", "unknown"
    ] = "unknown"
    sm_occupancy_pct: float | None = None
    achieved_flops_tflops: float | None = None
    peak_flops_tflops: float | None = None
    memory_bw_utilization_pct: float | None = None
    achieved_memory_bw_gbs: float | None = None
    peak_memory_bw_gbs: float | None = None
    top_kernels: list[KernelStat] = []


class CPUPerfMetrics(BaseModel):
    flops_dp_gflops: float | None = None
    flops_sp_gflops: float | None = None
    memory_bw_l1_gbs: float | None = None
    memory_bw_l2_gbs: float | None = None
    memory_bw_llc_gbs: float | None = None
    memory_bw_dram_gbs: float | None = None
    vectorization_ratio_pct: float | None = None
    energy_pkg_joules: float | None = None
    energy_dram_joules: float | None = None
    ipc: float | None = None


class KPIMetrics(BaseModel):
    primary_metric: str
    value: float
    unit: str
    lower_is_better: bool = True


class ProfileBundle(BaseModel):
    run_id: str
    timestamp: datetime
    workload_type: Literal[
        "mpi_cpu_only",
        "mpi_gpu_hybrid",
        "llm_inference",
        "llm_training",
        "cfd_openfoam",
        "unknown",
    ] = "unknown"
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    slurm: SlurmMetrics | None = None
    compute: ComputeMetrics | None = None
    cpu_perf: CPUPerfMetrics | None = None
    kpi: KPIMetrics
    raw_sources: dict[str, str] = {}


class BottleneckReport(BaseModel):
    run_id: str
    primary_bottleneck: Literal[
        "mpi_binding",
        "memory_bound_gpu",
        "compute_bound_gpu",
        "latency_bound_gpu",
        "cpu_memory_bound",
        "cpu_compute_bound",
        "high_rss",
        "none_detected",
        "insufficient_data",
    ]
    details: dict[str, str]
    recommended_action_hint: str


class ActionProposal(BaseModel):
    action_name: str
    parameters: dict
    rationale: str
    expected_improvement_pct: float | None = None


class DeltaReport(BaseModel):
    from_run_id: str
    to_run_id: str
    action_applied: str
    kpi_delta_pct: float
    kpi_direction: Literal["improved", "degraded", "neutral"]
    secondary_deltas: dict[str, float] = {}


class RunHandle(BaseModel):
    workload_id: str
    run_id: str
    path: Path
    run_type: Literal["baseline", "iteration"]
    iteration: int | None = None


class OptimizationResult(BaseModel):
    workload_id: str
    iterations_run: int
    best_run_id: str
    total_kpi_improvement_pct: float
    convergence_reason: str
    all_deltas: list[DeltaReport]


class PartitionInfo(BaseModel):
    """Slurm partition with representative node hardware."""
    name: str
    state: str | None = None
    total_nodes: int | None = None
    idle_nodes: int | None = None
    node_list: str | None = None
    # Hardware from a representative node in this partition
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    # Whether hardware was collected from a real compute node (probe job)
    hardware_from_probe: bool = False


class ModuleInfo(BaseModel):
    """One logical module (all versions) from the module system."""
    name: str
    versions: list[str] = []
    default_version: str | None = None
    # Convenience command using default (or newest) version
    load_cmd: str
    # Ordered commands needed to load this module, including any selector module
    # that exposes a gated MODULEPATH tier.
    load_sequence: list[str] = []
    # Discovery contexts where this module was visible.
    source_contexts: list[str] = []
    # MODULEPATH roots where this module appeared during discovery.
    modulepaths: list[str] = []
    # Best-effort metadata from module whatis/show output.
    metadata: str | None = None


class ModuleContext(BaseModel):
    """A module visibility context, such as the login default or a gated tier."""
    name: str
    load_sequence: list[str] = []
    modulepath: list[str] = []
    modules: list[ModuleInfo] = []
    discovered_by: Literal["active", "module_show"] = "active"


class Toolchain(BaseModel):
    """A resolved compiler + optional MPI/GPU combination with a module load sequence.

    Hierarchy: compiler → mpi → accelerator → math
    load_sequence must be executed in order before building/running.
    """
    name: str
    compiler: str | None = None
    mpi: str | None = None
    gpu: str | None = None
    cuda: str | None = None
    rocm: str | None = None
    math: str | None = None
    load_sequence: list[str] = []  # ordered module load commands


class SoftwareEnvironment(BaseModel):
    """Software environment discovered from the active module system.

    Classification hierarchy:
      compilers        — compiler and compiler-toolchain modules
      mpi_libraries    — MPI implementation modules
      gpu_toolkits     — GPU and accelerator toolkit modules
      math_libraries   — math, solver, BLAS/LAPACK/FFT-style modules
      io_libraries     — scientific and parallel I/O modules
      debuggers        — debugging tool modules
      profilers        — profiling and tracing tool modules
      applications     — application, framework, runtime, and interpreter modules
      other_modules    — everything else

    toolchains is a derived list of usable (compiler × mpi × gpu) combinations
    with ready-made load_sequence lists.
    """
    module_system: Literal["lmod", "tmod", "none"]
    lmod_version: str | None = None
    tmod_version: str | None = None
    # MODULEPATH entries at discovery time
    modulepath: list[str] = []
    # Additional module visibility contexts discovered by loading selector modules.
    module_contexts: list[ModuleContext] = []
    # True if module spider completed (full hierarchy); False = avail-only
    spider_complete: bool = False

    # Classified modules
    compilers: list[ModuleInfo] = []
    mpi_libraries: list[ModuleInfo] = []
    gpu_toolkits: list[ModuleInfo] = []
    math_libraries: list[ModuleInfo] = []
    io_libraries: list[ModuleInfo] = []
    debuggers: list[ModuleInfo] = []
    profilers: list[ModuleInfo] = []
    applications: list[ModuleInfo] = []
    other_modules: list[ModuleInfo] = []

    # Derived toolchain combinations (compiler × mpi × optional gpu)
    toolchains: list[Toolchain] = []


class ClusterProfile(BaseModel):
    """Cluster-level hardware and Slurm topology.

    Stored once per cluster (keyed by cluster_name) and reused across workloads.
    Login-node fields are always populated; partition hardware fields are filled
    progressively as probe jobs complete.
    """
    cluster_name: str
    discovered_at: datetime
    # Login-node hardware (CPU running on login node — approximation for compute)
    login_node_hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    # Per-partition detail; may be partial if probe jobs haven't run yet
    partitions: list[PartitionInfo] = []
    # Software environment (module system + classified modules + toolchains)
    software_env: SoftwareEnvironment | None = None
    # Raw sinfo output for reference
    raw_sinfo: str | None = None


# ---------------------------------------------------------------------------
# Edit tracking — code edits dispatched through OpenCode
# ---------------------------------------------------------------------------


class EditRecord(BaseModel):
    """A single code edit dispatched to OpenCode by the PGOA agent.

    Created when ``analyze_bottlenecks`` identifies a bottleneck that requires
    a source-code change (rather than a Slurm-script-only fix).  The record is
    linked to the ``ProfileBundle.run_id`` that triggered the edit
    (``linked_run_id_before``) and to the follow-up run that measured the
    effect (``linked_run_id_after``, filled in after ``compare_runs``).
    """
    edit_id: str
    timestamp: datetime
    # DSPy-generated hypothesis that motivated this edit
    hypothesis: str
    # Primary bottleneck category (from BottleneckReport)
    bottleneck_type: str
    # Full OpenCode prompt that was sent
    prompt_sent_to_opencode: str
    # OpenCode session ID extracted from CLI output (best-effort)
    opencode_session_id: str | None = None
    # Paths modified by OpenCode (from ``git diff --name-only``)
    files_modified: list[str] = []
    # Unified diff of all changes (from ``git diff <pre-commit>``)
    git_diff: str | None = None
    # Run that triggered this edit
    linked_run_id_before: str
    # Run measured after this edit (None until compare_runs completes)
    linked_run_id_after: str | None = None


class EditMapEntry(BaseModel):
    """Links one :class:`EditRecord` to its measured :class:`DeltaReport`."""
    edit_id: str
    edit: EditRecord
    # None until compare_runs has been called for the post-edit run
    delta: DeltaReport | None = None


class MetricsEditMap(BaseModel):
    """All edit-delta pairs accumulated during one PGOA optimization run."""
    workload_id: str
    entries: list[EditMapEntry] = []
