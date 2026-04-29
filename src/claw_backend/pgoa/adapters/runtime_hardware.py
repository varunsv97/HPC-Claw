"""RuntimeHardwareAdapter: collect transient hardware pressure for PGOA iterations.

Static topology and device inventory are collected once by
``cluster_probes.hardware_inventory`` and cached in the global cluster profile. This
adapter samples only transient state visible during a profiling run.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from claw_backend.config import AssistantSettings, load_settings
from claw_backend.pgoa.adapters.base import BaseAdapter
from claw_backend.pgoa.schema import HardwareInfo, KPIMetrics, ProfileBundle

log = logging.getLogger(__name__)


def _procfs_read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _run_nvidia_smi(settings: AssistantSettings) -> list[dict] | None:
    """Query nvidia-smi for runtime GPU utilization fields."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,multi_processor_count,"
                "utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    gpus: list[dict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            memory_gb = round(float(parts[1]) / 1024.0, 1)
        except ValueError:
            memory_gb = 0.0
        try:
            utilization_pct = float(parts[4])
        except ValueError:
            utilization_pct = None
        try:
            memory_used_gb = round(float(parts[5]) / 1024.0, 2)
        except ValueError:
            memory_used_gb = None
        gpus.append(
            {
                "memory_gb": memory_gb,
                "utilization_pct": utilization_pct,
                "memory_used_gb": memory_used_gb,
            }
        )
    return gpus or None


def _summarize_runtime_gpus(gpus: list[dict]) -> dict:
    utilizations = [
        gpu["utilization_pct"]
        for gpu in gpus
        if gpu.get("utilization_pct") is not None
    ]
    used_memory = [
        gpu["memory_used_gb"]
        for gpu in gpus
        if gpu.get("memory_used_gb") is not None
    ]
    memory_utilizations = [
        gpu["memory_used_gb"] / gpu["memory_gb"] * 100.0
        for gpu in gpus
        if gpu.get("memory_used_gb") is not None and gpu.get("memory_gb")
    ]
    return {
        "visible_gpus": len(gpus),
        "gpu_utilization_pct": (
            round(sum(utilizations) / len(utilizations), 1) if utilizations else None
        ),
        "gpu_memory_used_gb": round(sum(used_memory), 2) if used_memory else None,
        "gpu_memory_utilization_pct": (
            round(sum(memory_utilizations) / len(memory_utilizations), 1)
            if memory_utilizations
            else None
        ),
    }


def _collect_loadavg(proc_root: Path) -> dict:
    raw = _procfs_read(proc_root / "loadavg")
    if not raw:
        return {}
    parts = raw.split()
    try:
        return {"loadavg_1m": float(parts[0])}
    except (IndexError, ValueError):
        return {}


def _collect_memory_available(proc_root: Path) -> dict:
    raw = _procfs_read(proc_root / "meminfo")
    if not raw:
        return {}
    for line in raw.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return {}
        try:
            return {"memory_available_gb": round(float(parts[1]) / (1024 ** 2), 2)}
        except ValueError:
            return {}
    return {}


def _collect_cpu_pressure(proc_root: Path) -> dict:
    raw = _procfs_read(proc_root / "pressure" / "cpu")
    if not raw:
        return {}
    for line in raw.splitlines():
        if not line.startswith("some "):
            continue
        for token in line.split():
            key, _, value = token.partition("=")
            if key != "avg10":
                continue
            try:
                return {"cpu_pressure_avg10": float(value)}
            except ValueError:
                return {}
    return {}


def collect_runtime_hardware_metrics(
    settings: AssistantSettings | None = None,
    *,
    proc_root: Path = Path("/proc"),
) -> HardwareInfo:
    """Collect transient hardware pressure/utilization for the current run."""
    active_settings = settings or load_settings()
    info: dict = {}

    for label, fn, args in (
        ("load average", _collect_loadavg, (proc_root,)),
        ("available memory", _collect_memory_available, (proc_root,)),
        ("CPU pressure", _collect_cpu_pressure, (proc_root,)),
    ):
        try:
            info.update(fn(*args))
        except Exception:
            log.debug("%s unavailable", label, exc_info=True)

    gpus = _run_nvidia_smi(active_settings)
    if gpus:
        info.update(_summarize_runtime_gpus(gpus))

    return HardwareInfo(**{k: v for k, v in info.items() if v is not None})


class RuntimeHardwareAdapter(BaseAdapter):
    """BaseAdapter wrapper around runtime hardware collection."""

    def __init__(self, settings: AssistantSettings | None = None) -> None:
        self._settings = settings or load_settings()

    def name(self) -> str:
        return "runtime_hardware"

    def collect(  # type: ignore[override]
        self,
        kpi: KPIMetrics,
    ) -> ProfileBundle:
        hw = collect_runtime_hardware_metrics(self._settings)
        bundle = self._new_bundle(kpi)
        return bundle.model_copy(update={"hardware": hw})
