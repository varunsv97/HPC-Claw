"""HardwareAdapter: collect CPU/GPU topology from /sys, hwloc, and nvidia-smi."""

from __future__ import annotations

import logging
import platform
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from claw_backend.config import AssistantSettings, load_settings
from claw_backend.pgoa.adapters.base import BaseAdapter
from claw_backend.pgoa.schema import HardwareInfo, KPIMetrics, ProfileBundle

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _sysfs_read(path: Path) -> str | None:
    """Read a sysfs/procfs file, returning stripped text or None on error."""
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _parse_cpu_range(s: str) -> int | None:
    """Count CPUs described by a kernel cpulist string, e.g. '0-63' or '0,2-5,8'."""
    count = 0
    try:
        for part in s.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                count += int(hi) - int(lo) + 1
            elif part:
                count += 1
        return count if count > 0 else None
    except (ValueError, AttributeError):
        return None


def _parse_cache_size_kb(size_str: str) -> int | None:
    """Parse kernel cache size strings like '32K', '256K', '24576K', '24M'."""
    try:
        s = size_str.strip().upper()
        if s.endswith("K"):
            return int(s[:-1])
        if s.endswith("M"):
            return int(s[:-1]) * 1024
        if s.endswith("G"):
            return int(s[:-1]) * 1024 * 1024
        return int(s)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# /sys readers
# ---------------------------------------------------------------------------

def _collect_cpu_topology(sysfs_root: Path) -> dict:
    """Read socket/core/thread counts from /sys/devices/system/cpu/."""
    cpu_dir = sysfs_root / "devices" / "system" / "cpu"

    possible_str = _sysfs_read(cpu_dir / "possible")
    total_cpus = _parse_cpu_range(possible_str) if possible_str else None

    socket_ids: set[int] = set()
    core_per_socket: dict[int, set[int]] = {}
    thread_per_core: dict[tuple[int, int], set[int]] = {}

    for topo_dir in sorted(cpu_dir.glob("cpu[0-9]*/topology")):
        cpu_name = topo_dir.parent.name
        try:
            cpu_idx = int(cpu_name[3:])
        except ValueError:
            continue
        pkg_str = _sysfs_read(topo_dir / "physical_package_id")
        core_str = _sysfs_read(topo_dir / "core_id")
        if pkg_str is None or core_str is None:
            continue
        try:
            s, c = int(pkg_str), int(core_str)
        except ValueError:
            continue
        socket_ids.add(s)
        core_per_socket.setdefault(s, set()).add(c)
        thread_per_core.setdefault((s, c), set()).add(cpu_idx)

    n_sockets = len(socket_ids) if socket_ids else None
    cores_per_socket = (
        max(len(v) for v in core_per_socket.values()) if core_per_socket else None
    )
    threads_per_core = (
        max(len(v) for v in thread_per_core.values()) if thread_per_core else None
    )
    if total_cpus is None and n_sockets and cores_per_socket and threads_per_core:
        total_cpus = n_sockets * cores_per_socket * threads_per_core

    return {
        "cpus_per_node": total_cpus,
        "sockets_per_node": n_sockets,
        "cores_per_socket": cores_per_socket,
        "threads_per_core": threads_per_core,
    }


def _collect_cache_info(sysfs_root: Path) -> dict:
    """Read per-core cache sizes from /sys/devices/system/cpu/cpu0/cache/."""
    cache_dir = sysfs_root / "devices" / "system" / "cpu" / "cpu0" / "cache"
    l1d_kb = l2_kb = l3_mb = None

    for idx_dir in sorted(cache_dir.glob("index*")):
        level = _sysfs_read(idx_dir / "level")
        cache_type = _sysfs_read(idx_dir / "type")
        size_str = _sysfs_read(idx_dir / "size")
        if level is None or size_str is None:
            continue
        size_kb = _parse_cache_size_kb(size_str)
        if size_kb is None:
            continue
        if level == "1" and cache_type == "Data":
            l1d_kb = size_kb
        elif level == "2":
            l2_kb = size_kb
        elif level == "3":
            l3_mb = round(size_kb / 1024.0, 2)

    return {"cache_l1d_kb": l1d_kb, "cache_l2_kb": l2_kb, "cache_l3_mb": l3_mb}


def _collect_numa_info(sysfs_root: Path) -> dict:
    """Count NUMA nodes and sum total memory from /sys/devices/system/node/."""
    node_dir = sysfs_root / "devices" / "system" / "node"
    numa_dirs = sorted(node_dir.glob("node[0-9]*"))
    n_numa = len(numa_dirs) if numa_dirs else None

    total_mem_kb = 0.0
    for nd in numa_dirs:
        meminfo = _sysfs_read(nd / "meminfo")
        if not meminfo:
            continue
        for line in meminfo.splitlines():
            if "MemTotal" in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        total_mem_kb += float(parts[3])
                    except ValueError:
                        pass
                break

    mem_gb = round(total_mem_kb / (1024 ** 2), 1) if total_mem_kb else None
    return {"numa_nodes": n_numa, "memory_gb_per_node": mem_gb}


# ---------------------------------------------------------------------------
# /proc/cpuinfo reader
# ---------------------------------------------------------------------------

_HPC_FLAGS = frozenset({
    "avx", "avx2",
    "avx512f", "avx512cd", "avx512bw", "avx512vl", "avx512dq",
    "sse4_1", "sse4_2", "fma",
    "aes",
    "sve",       # AArch64 Scalable Vector Extension
})


def _collect_cpu_model(proc_root: Path) -> dict:
    """Extract CPU model name and HPC-relevant instruction-set flags from /proc/cpuinfo."""
    cpuinfo = _sysfs_read(proc_root / "cpuinfo")
    if not cpuinfo:
        return {}

    model_name: str | None = None
    features: list[str] | None = None

    for line in cpuinfo.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key == "model name" and model_name is None:
            model_name = val
        elif key in ("flags", "features") and features is None:
            features = sorted(f for f in val.split() if f in _HPC_FLAGS)
        if model_name and features is not None:
            break

    return {"cpu_model": model_name, "cpu_features": features or []}


# ---------------------------------------------------------------------------
# hwloc-ls / lstopo
# ---------------------------------------------------------------------------

def _run_hwloc(settings: AssistantSettings) -> str | None:
    """Run hwloc-ls (or lstopo) with XML output; return raw XML or None."""
    for cmd in (
        ["hwloc-ls", "--of", "xml", "--no-io"],
        ["lstopo", "--of", "xml", "--no-io"],
    ):
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=settings.command_timeout_seconds,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _parse_hwloc_xml(xml_text: str) -> dict:
    """Extract topology and cache info from hwloc --of xml output.

    Handles both hwloc v1 (root=<object type="Machine">) and
    v2 (root=<topology> with nested <object> elements).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.warning("Failed to parse hwloc XML")
        return {}

    machine = (
        root.find(".//object[@type='Machine']")
        if root.tag == "topology"
        else (root if root.get("type") == "Machine" else None)
    )
    if machine is None:
        return {}

    def _count(obj_type: str) -> int:
        return len(machine.findall(f".//object[@type='{obj_type}']"))

    n_packages = _count("Package")
    n_numanodes = _count("NUMANode")
    n_cores = _count("Core")
    n_pus = _count("PU")

    cores_per_socket = (n_cores // n_packages) if n_packages > 0 else None
    threads_per_core = (n_pus // n_cores) if n_cores > 0 else None

    def _cache_size_mb(obj_type: str) -> float | None:
        el = machine.find(f".//object[@type='{obj_type}']")
        if el is None:
            return None
        try:
            return round(int(el.get("cache_size", "0")) / (1024 ** 2), 2)
        except ValueError:
            return None

    def _cache_size_kb(obj_type: str) -> int | None:
        el = machine.find(f".//object[@type='{obj_type}']")
        if el is None:
            return None
        try:
            return int(el.get("cache_size", "0")) // 1024
        except ValueError:
            return None

    result: dict = {}
    if n_packages > 0:
        result["sockets_per_node"] = n_packages
    if n_numanodes > 0:
        result["numa_nodes"] = n_numanodes
    if n_pus > 0:
        result["cpus_per_node"] = n_pus
    if cores_per_socket:
        result["cores_per_socket"] = cores_per_socket
    if threads_per_core:
        result["threads_per_core"] = threads_per_core

    l3 = _cache_size_mb("L3Cache")
    if l3:
        result["cache_l3_mb"] = l3
    l2 = _cache_size_kb("L2Cache")
    if l2:
        result["cache_l2_kb"] = l2
    l1 = _cache_size_kb("L1Cache")
    if l1:
        result["cache_l1d_kb"] = l1

    return result


# ---------------------------------------------------------------------------
# nvidia-smi
# ---------------------------------------------------------------------------

_GPU_ARCH_BY_CC: dict[str, str] = {
    "7.0": "Volta",
    "7.2": "Volta",
    "7.5": "Turing",
    "8.0": "Ampere",
    "8.6": "Ampere",
    "8.7": "Ampere",
    "8.9": "Ada",
    "9.0": "Hopper",
    "10.0": "Blackwell",
}


def _gpu_arch_from_cc(compute_cap: str) -> str | None:
    return _GPU_ARCH_BY_CC.get(compute_cap.strip())


def _run_nvidia_smi(settings: AssistantSettings) -> list[dict] | None:
    """Query nvidia-smi for per-GPU name, memory, compute capability, SM count."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,multi_processor_count",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if r.returncode != 0 or not r.stdout.strip():
        return None

    gpus: list[dict] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            mem_mib = float(parts[1])
        except ValueError:
            mem_mib = 0.0
        sm_count: int | None = None
        try:
            sm_count = int(parts[3])
        except (ValueError, IndexError):
            pass
        gpus.append({
            "name": parts[0],
            "memory_gb": round(mem_mib / 1024.0, 1),
            "compute_capability": parts[2],
            "sm_count": sm_count,
        })
    return gpus or None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def collect_hardware_info(
    settings: AssistantSettings | None = None,
    *,
    sysfs_root: Path = Path("/sys"),
    proc_root: Path = Path("/proc"),
) -> HardwareInfo:
    """Probe CPU/GPU topology from /sys, /proc, hwloc, and nvidia-smi.

    All sources are best-effort; failures are logged at DEBUG level and the
    corresponding fields are left as None.  hwloc fills in any topology
    fields that /sys couldn't provide; /sys takes precedence where both succeed.
    """
    s = settings or load_settings()
    info: dict = {"cpu_arch": platform.machine()}

    for label, fn, args in (
        ("sysfs cpu topology", _collect_cpu_topology, (sysfs_root,)),
        ("sysfs cache info",   _collect_cache_info,   (sysfs_root,)),
        ("sysfs NUMA info",    _collect_numa_info,    (sysfs_root,)),
        ("/proc/cpuinfo",      _collect_cpu_model,    (proc_root,)),
    ):
        try:
            info.update(fn(*args))
        except Exception:
            log.debug("%s unavailable", label, exc_info=True)

    # hwloc fills gaps only — /sys values take precedence
    hwloc_xml = _run_hwloc(s)
    if hwloc_xml:
        for k, v in _parse_hwloc_xml(hwloc_xml).items():
            if info.get(k) is None:
                info[k] = v

    # GPU info from nvidia-smi
    gpus = _run_nvidia_smi(s)
    if gpus:
        first = gpus[0]
        cc = first["compute_capability"]
        info.update({
            "gpus_per_node": len(gpus),
            "gpu_model": first["name"],
            "gpu_memory_gb": first["memory_gb"],
            "gpu_sm_count": first["sm_count"],
            "gpu_compute_capability": cc,
            "gpu_arch": _gpu_arch_from_cc(cc),
        })

    # Filter None so Pydantic uses field defaults; keep empty lists (cpu_features=[])
    return HardwareInfo(**{k: v for k, v in info.items() if v is not None})


class HardwareAdapter(BaseAdapter):
    """BaseAdapter wrapper around collect_hardware_info()."""

    def __init__(self, settings: AssistantSettings | None = None) -> None:
        self._settings = settings or load_settings()

    def name(self) -> str:
        return "hardware"

    def collect(  # type: ignore[override]
        self,
        kpi: KPIMetrics,
        *,
        sysfs_root: Path = Path("/sys"),
        proc_root: Path = Path("/proc"),
    ) -> ProfileBundle:
        hw = collect_hardware_info(
            self._settings, sysfs_root=sysfs_root, proc_root=proc_root
        )
        bundle = self._new_bundle(kpi)
        return bundle.model_copy(update={"hardware": hw})
