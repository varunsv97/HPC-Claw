"""Cluster hardware discovery — login-node queries and optional probe jobs.

Two phases:
  1. Static cluster discovery (always, no jobs needed):
     - ``sinfo --Node --format=...``  → partition list + node counts
     - ``scontrol show node``         → first node in each partition (cpu/mem/gpu)
     - ``hwloc-ls`` / ``/sys`` / ``/proc`` for login-node static topology
     - ``nvidia-smi`` static device inventory if available

  2. Probe-job discovery (gated by caller-supplied approval callback):
     - Submits a tiny ``srun --pty ...`` (or batch job) on each target partition
       that runs ``hwloc-ls --of xml`` + ``nvidia-smi`` and writes results to a
       temp file.
     - Merges results into ClusterProfile.partitions[*].hardware.

Results are saved to ``<store_base>/_global/cluster_profiles/<cluster_name>.json``
and reused across workloads on the same cluster.  A max-age guard forces
re-discovery when the cached data is stale (default 7 days).
"""

from __future__ import annotations

import getpass
import logging
import os
import platform
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from claw_backend.config import AssistantSettings
from claw_backend.cluster_probes.software import discover_software_env
from claw_backend.pgoa.schema import ClusterProfile, HardwareInfo, PartitionInfo
from claw_backend.utils.hardware_parsing import gpu_arch_from_cc, parse_hwloc_xml

log = logging.getLogger(__name__)

_CACHE_MAX_AGE_DAYS = 7


def _sysfs_read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_cpu_range(value: str) -> int | None:
    count = 0
    try:
        for part in value.split(","):
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
    try:
        value = size_str.strip().upper()
        if value.endswith("K"):
            return int(value[:-1])
        if value.endswith("M"):
            return int(value[:-1]) * 1024
        if value.endswith("G"):
            return int(value[:-1]) * 1024 * 1024
        return int(value)
    except (ValueError, AttributeError):
        return None


def _collect_cpu_topology(sysfs_root: Path) -> dict:
    cpu_dir = sysfs_root / "devices" / "system" / "cpu"
    possible_str = _sysfs_read(cpu_dir / "possible")
    total_cpus = _parse_cpu_range(possible_str) if possible_str else None

    socket_ids: set[int] = set()
    core_per_socket: dict[int, set[int]] = {}
    thread_per_core: dict[tuple[int, int], set[int]] = {}

    for topo_dir in sorted(cpu_dir.glob("cpu[0-9]*/topology")):
        try:
            cpu_idx = int(topo_dir.parent.name[3:])
        except ValueError:
            continue
        pkg_str = _sysfs_read(topo_dir / "physical_package_id")
        core_str = _sysfs_read(topo_dir / "core_id")
        if pkg_str is None or core_str is None:
            continue
        try:
            socket_id, core_id = int(pkg_str), int(core_str)
        except ValueError:
            continue
        socket_ids.add(socket_id)
        core_per_socket.setdefault(socket_id, set()).add(core_id)
        thread_per_core.setdefault((socket_id, core_id), set()).add(cpu_idx)

    n_sockets = len(socket_ids) if socket_ids else None
    cores_per_socket = (
        max(len(cores) for cores in core_per_socket.values())
        if core_per_socket
        else None
    )
    threads_per_core = (
        max(len(threads) for threads in thread_per_core.values())
        if thread_per_core
        else None
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
    node_dir = sysfs_root / "devices" / "system" / "node"
    numa_dirs = sorted(node_dir.glob("node[0-9]*"))
    n_numa = len(numa_dirs) if numa_dirs else None

    total_mem_kb = 0.0
    for node_path in numa_dirs:
        meminfo = _sysfs_read(node_path / "meminfo")
        if not meminfo:
            continue
        for line in meminfo.splitlines():
            if "MemTotal" not in line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    total_mem_kb += float(parts[3])
                except ValueError:
                    pass
            break

    mem_gb = round(total_mem_kb / (1024 ** 2), 1) if total_mem_kb else None
    return {"numa_nodes": n_numa, "memory_gb_per_node": mem_gb}


_HPC_FLAGS = frozenset({
    "avx", "avx2",
    "avx512f", "avx512cd", "avx512bw", "avx512vl", "avx512dq",
    "sse4_1", "sse4_2", "fma",
    "aes",
    "sve",
})


def _collect_cpu_model(proc_root: Path) -> dict:
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
            features = sorted(flag for flag in val.split() if flag in _HPC_FLAGS)
        if model_name and features is not None:
            break

    return {"cpu_model": model_name, "cpu_features": features or []}


def _run_hwloc(settings: AssistantSettings) -> str | None:
    for cmd in (
        ["hwloc-ls", "--of", "xml", "--no-io"],
        ["lstopo", "--of", "xml", "--no-io"],
    ):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=settings.command_timeout_seconds,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    return None


def _query_static_gpus(settings: AssistantSettings) -> list[dict] | None:
    try:
        result = subprocess.run(
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

    if result.returncode != 0 or not result.stdout.strip():
        return None

    gpus: list[dict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            mem_mib = float(parts[1])
        except ValueError:
            mem_mib = 0.0
        try:
            sm_count = int(parts[3])
        except ValueError:
            sm_count = None
        gpus.append({
            "name": parts[0],
            "memory_gb": round(mem_mib / 1024.0, 1),
            "compute_capability": parts[2],
            "sm_count": sm_count,
        })
    return gpus or None


def _summarize_static_gpus(gpus: list[dict]) -> dict:
    first = gpus[0]
    compute_capability = first["compute_capability"]
    return {
        "gpus_per_node": len(gpus),
        "gpu_model": first["name"],
        "gpu_memory_gb": first["memory_gb"],
        "gpu_sm_count": first["sm_count"],
        "gpu_compute_capability": compute_capability,
        "gpu_arch": gpu_arch_from_cc(compute_capability),
    }

# ---------------------------------------------------------------------------
# sinfo helpers
# ---------------------------------------------------------------------------

_SINFO_FORMAT = "PartitionName,State,TotalNodes,IdleNodes,NodeList"


def _run_sinfo(settings: AssistantSettings) -> str:
    try:
        r = subprocess.run(
            ["sinfo", "--noheader", "--Format", _SINFO_FORMAT],
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
        return r.stdout if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.debug("sinfo not available or timed out")
        return ""


def _parse_sinfo(raw: str) -> list[PartitionInfo]:
    partitions: list[PartitionInfo] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            total = int(parts[2])
        except ValueError:
            total = None
        try:
            idle = int(parts[3])
        except ValueError:
            idle = None
        partitions.append(
            PartitionInfo(
                name=parts[0],
                state=parts[1],
                total_nodes=total,
                idle_nodes=idle,
                node_list=parts[4],
            )
        )
    return partitions


# ---------------------------------------------------------------------------
# scontrol show node helpers
# ---------------------------------------------------------------------------

def _run_scontrol_node(node: str, settings: AssistantSettings) -> str:
    try:
        r = subprocess.run(
            ["scontrol", "show", "node", node],
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
        return r.stdout if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _first_node_name(node_list_expr: str) -> str | None:
    """Extract the first node name from a Slurm hostlist expression.

    E.g. 'node[001-016]' → 'node001', 'node42' → 'node42'.
    """
    if not node_list_expr or node_list_expr in ("(null)", "N/A"):
        return None
    # Simple bracket expansion: take first range or literal
    m = re.match(r"([^\[,\s]+)(?:\[([^\]]+)\])?", node_list_expr)
    if not m:
        return None
    prefix = m.group(1)
    bracket = m.group(2)
    if bracket is None:
        return prefix
    # Take the first element (before first comma or dash)
    first = bracket.split(",")[0].split("-")[0]
    # Pad to match the length of the first element in the range
    pad = len(first)
    return f"{prefix}{first.zfill(pad)}"


def _parse_scontrol_node(raw: str) -> dict:
    """Extract CPU, memory, and GPU fields from ``scontrol show node`` output."""
    result: dict = {}
    for token in raw.split():
        if "=" not in token:
            continue
        k, _, v = token.partition("=")
        if k == "CfgTRES":
            # cpu=128,mem=512G,billing=128,gres/gpu=4
            for item in v.split(","):
                if item.startswith("cpu="):
                    try:
                        result["cpus_per_node"] = int(item[4:])
                    except ValueError:
                        pass
                elif item.startswith("mem="):
                    mem_str = item[4:]
                    try:
                        if mem_str.endswith("G"):
                            result["memory_gb_per_node"] = float(mem_str[:-1])
                        elif mem_str.endswith("M"):
                            result["memory_gb_per_node"] = round(float(mem_str[:-1]) / 1024, 1)
                        elif mem_str.endswith("T"):
                            result["memory_gb_per_node"] = float(mem_str[:-1]) * 1024
                    except ValueError:
                        pass
                elif item.startswith("gres/gpu="):
                    try:
                        result["gpus_per_node"] = int(item[9:])
                    except ValueError:
                        pass
        elif k == "Arch":
            result["cpu_arch"] = v
        elif k == "Sockets":
            try:
                result["sockets_per_node"] = int(v)
            except ValueError:
                pass
        elif k == "CoresPerSocket":
            try:
                result["cores_per_socket"] = int(v)
            except ValueError:
                pass
        elif k == "ThreadsPerCore":
            try:
                result["threads_per_core"] = int(v)
            except ValueError:
                pass
    return result


# ---------------------------------------------------------------------------
# Static cluster discovery
# ---------------------------------------------------------------------------

def collect_static_hardware_info(
    settings: AssistantSettings,
    *,
    sysfs_root: Path = Path("/sys"),
    proc_root: Path = Path("/proc"),
) -> HardwareInfo:
    """Collect static login-node hardware without runtime utilization fields."""
    info: dict = {"cpu_arch": platform.machine()}

    for label, fn, args in (
        ("sysfs cpu topology", _collect_cpu_topology, (sysfs_root,)),
        ("sysfs cache info", _collect_cache_info, (sysfs_root,)),
        ("sysfs NUMA info", _collect_numa_info, (sysfs_root,)),
        ("/proc/cpuinfo", _collect_cpu_model, (proc_root,)),
    ):
        try:
            info.update(fn(*args))
        except Exception:
            log.debug("%s unavailable", label, exc_info=True)

    hwloc_xml = _run_hwloc(settings)
    if hwloc_xml:
        for key, value in parse_hwloc_xml(hwloc_xml).items():
            if info.get(key) is None:
                info[key] = value

    gpus = _query_static_gpus(settings)
    if gpus:
        info.update(_summarize_static_gpus(gpus))

    return HardwareInfo(**{k: v for k, v in info.items() if v is not None})


def collect_login_node_info(settings: AssistantSettings) -> ClusterProfile:
    """Phase 1: collect static cluster hardware/config without workload metrics."""

    # Cluster name from scontrol ping (best effort)
    cluster_name = detect_cluster_name(settings)

    # Static local hardware from /sys + hwloc + nvidia-smi inventory.
    login_hw = collect_static_hardware_info(settings)

    # Partition list from sinfo
    raw_sinfo = _run_sinfo(settings)
    partitions = _parse_sinfo(raw_sinfo)

    # Enrich each partition's hardware from scontrol show node <first-node>
    enriched: list[PartitionInfo] = []
    for part in partitions:
        hw_dict: dict = {}
        first_node = _first_node_name(part.node_list or "")
        if first_node:
            raw_node = _run_scontrol_node(first_node, settings)
            if raw_node:
                hw_dict = _parse_scontrol_node(raw_node)
        hw = HardwareInfo(**{k: v for k, v in hw_dict.items() if v is not None})
        enriched.append(part.model_copy(update={"hardware": hw}))

    # Discover the software environment (Lmod/Tmod modules)
    log.debug("Discovering software environment via module system…")
    try:
        software_env = discover_software_env(settings)
    except Exception:
        log.debug("Software environment discovery failed", exc_info=True)
        software_env = None

    return ClusterProfile(
        cluster_name=cluster_name,
        discovered_at=datetime.now(tz=UTC),
        login_node_hardware=login_hw,
        partitions=enriched,
        software_env=software_env,
        raw_sinfo=raw_sinfo or None,
    )


def detect_cluster_name(settings: AssistantSettings) -> str:
    """Return the Slurm cluster name, or 'unknown' on failure."""
    try:
        r = subprocess.run(
            ["scontrol", "show", "config"],
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
        for line in r.stdout.splitlines():
            if line.strip().startswith("ClusterName"):
                _, _, name = line.partition("=")
                return name.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Probe-job discovery
# ---------------------------------------------------------------------------

_PROBE_SCRIPT = """\
#!/bin/bash
#SBATCH --job-name=pgoa_probe
#SBATCH --ntasks=1
#SBATCH --time=00:02:00
#SBATCH --output={output_file}

set -e
echo "=== hwloc ==="
hwloc-ls --of xml --no-io 2>/dev/null || echo "(hwloc unavailable)"
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap,multi_processor_count \
    --format=csv,noheader,nounits 2>/dev/null || echo "(no GPU)"
echo "=== lscpu ==="
lscpu 2>/dev/null || echo "(lscpu unavailable)"
"""

_HWLOC_XML_RE = re.compile(r"(<\?xml.*?</topology>)", re.DOTALL)


ApprovalCallback = Callable[[str], bool]
"""Called with a human-readable description; return True to allow."""


def _parse_account_partition_relationship(raw: str) -> dict[str, set[str] | None]:
    """Parse sacctmgr Account/Partition rows for one user.

    Returns a mapping: account -> allowed partitions. A value of ``None`` means
    the account is not restricted to explicit partitions.
    """
    relationships: dict[str, set[str] | None] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        account, _, partition_raw = line.partition("|")
        account = account.strip()
        if not account:
            continue
        partition_raw = partition_raw.strip()
        # Empty/(null)/*/ALL indicates no explicit partition restriction.
        if partition_raw in ("", "(null)", "*", "ALL"):
            relationships[account] = None
            continue
        partitions = {
            name.strip()
            for name in partition_raw.split(",")
            if name.strip() and name.strip() != "(null)"
        }
        previous = relationships.get(account)
        if previous is None and account in relationships:
            # Already unrestricted for this account.
            continue
        relationships[account] = (previous or set()) | partitions
    return relationships


def _account_partition_relationship(
    settings: AssistantSettings,
) -> dict[str, set[str] | None]:
    """Best-effort query of Slurm account-to-partition associations."""
    user = getpass.getuser()
    try:
        r = subprocess.run(
            [
                "sacctmgr",
                "-nP",
                "show",
                "assoc",
                f"user={user}",
                "format=Account,Partition",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.debug("Could not query account/partition associations: %s", exc)
        return {}
    if r.returncode != 0:
        stderr = r.stderr.strip()
        if stderr:
            log.debug("sacctmgr assoc query failed: %s", stderr)
        return {}
    return _parse_account_partition_relationship(r.stdout)


def _resolve_probe_partitions(
    *,
    requested: set[str] | None,
    available: set[str],
    relationships: dict[str, set[str] | None],
) -> set[str]:
    """Return partitions eligible for probe jobs after account filtering."""
    candidates = requested if requested is not None else set(available)
    if not relationships:
        return candidates

    active_account = (
        os.environ.get("SLURM_ACCOUNT", "").strip()
        or os.environ.get("SBATCH_ACCOUNT", "").strip()
        or None
    )

    if active_account and active_account in relationships:
        allowed_for_active = relationships[active_account]
        if allowed_for_active is None:
            return candidates
        return candidates & allowed_for_active

    if any(parts is None for parts in relationships.values()):
        return candidates

    allowed_union = set().union(*(parts for parts in relationships.values() if parts))
    return candidates & allowed_union if allowed_union else candidates


def run_probe_jobs(
    profile: ClusterProfile,
    settings: AssistantSettings,
    partitions: list[str] | None = None,
    *,
    approval_callback: ApprovalCallback | None = None,
    poll_interval_s: float = 10.0,
    max_wait_s: float = 120.0,
) -> ClusterProfile:
    """Phase 2: submit a 1-task probe job on each partition to collect real hw info.

    Only partitions listed in *partitions* are probed; if None, all partitions
    that still have hardware_from_probe=False are candidates.

    *approval_callback* is called once per partition with a description string.
    If it returns False (or is None), the partition is skipped silently.
    """
    requested = set(partitions) if partitions else None
    available = {part.name for part in profile.partitions}
    relationships = _account_partition_relationship(settings)
    target_names = _resolve_probe_partitions(
        requested=requested,
        available=available,
        relationships=relationships,
    )
    skipped_for_access = (requested or available) - target_names
    if skipped_for_access:
        log.info(
            "Skipping probe jobs on partitions not allowed for current account mapping: %s",
            ", ".join(sorted(skipped_for_access)),
        )
    updated = list(profile.partitions)

    for i, part in enumerate(updated):
        if part.name not in target_names:
            continue
        if part.hardware_from_probe:
            log.debug("Partition %s already has probe data, skipping", part.name)
            continue

        description = (
            f"Submit a 1-task, 2-minute probe job on partition '{part.name}' "
            f"to collect CPU/GPU topology from a real compute node."
        )
        if approval_callback and not approval_callback(description):
            log.info("Probe job skipped for partition %s (not approved)", part.name)
            continue

        hw = _run_single_probe(part.name, settings, poll_interval_s, max_wait_s)
        if hw is not None:
            updated[i] = part.model_copy(
                update={"hardware": hw, "hardware_from_probe": True}
            )

    return profile.model_copy(update={"partitions": updated})


def _run_single_probe(
    partition: str,
    settings: AssistantSettings,
    poll_interval_s: float,
    max_wait_s: float,
) -> HardwareInfo | None:
    """Submit a probe batch job, wait for output, parse hardware info."""
    with tempfile.TemporaryDirectory(prefix="pgoa_probe_") as tmpdir:
        tmp = Path(tmpdir)
        output_file = tmp / "probe_output.txt"
        script_file = tmp / "probe.sh"
        script_file.write_text(
            _PROBE_SCRIPT.format(output_file=output_file), encoding="utf-8"
        )

        try:
            r = subprocess.run(
                ["sbatch", f"--partition={partition}", str(script_file)],
                capture_output=True,
                text=True,
                check=False,
                timeout=settings.command_timeout_seconds,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.warning("sbatch unavailable: %s", exc)
            return None

        if r.returncode != 0:
            log.warning("sbatch failed for partition %s: %s", partition, r.stderr.strip())
            return None

        # Parse job ID from "Submitted batch job 12345"
        m = re.search(r"Submitted batch job (\d+)", r.stdout)
        if not m:
            log.warning("Could not parse job ID from sbatch output: %s", r.stdout.strip())
            return None
        job_id = m.group(1)

        # Poll for completion
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            time.sleep(poll_interval_s)
            if _job_finished(job_id, settings):
                break
        else:
            log.warning("Probe job %s timed out after %.0fs", job_id, max_wait_s)
            _cancel_job(job_id, settings)
            return None

        if not output_file.exists():
            log.warning("Probe output file missing for job %s", job_id)
            return None

        return _parse_probe_output(output_file.read_text(encoding="utf-8"))


def _job_finished(job_id: str, settings: AssistantSettings) -> bool:
    try:
        r = subprocess.run(
            ["sacct", "-j", job_id, "--noheader", "--format=State", "--parsable2"],
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
        states = {line.strip() for line in r.stdout.splitlines() if line.strip()}
        return bool(states) and all(
            s in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL")
            for s in states
        )
    except Exception:
        return False


def _cancel_job(job_id: str, settings: AssistantSettings) -> None:
    try:
        subprocess.run(
            ["scancel", job_id],
            capture_output=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
    except Exception:
        pass


def _parse_probe_output(text: str) -> HardwareInfo:
    """Extract HardwareInfo from the probe script output."""
    info: dict = {}

    # hwloc XML block
    xml_m = _HWLOC_XML_RE.search(text)
    if xml_m:
        info.update(parse_hwloc_xml(xml_m.group(1)))

    # nvidia-smi section
    smi_section = _extract_section(text, "=== nvidia-smi ===")
    if smi_section and "(no GPU)" not in smi_section:
        gpus = []
        for line in smi_section.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                gpus.append({
                    "name": parts[0],
                    "memory_gb": round(float(parts[1]) / 1024.0, 1),
                    "compute_capability": parts[2],
                    "sm_count": int(parts[3]) if parts[3].isdigit() else None,
                })
            except (ValueError, IndexError):
                continue
        if gpus:
            info.update(_summarize_static_gpus(gpus))

    # lscpu section (fallback if hwloc unavailable)
    lscpu_section = _extract_section(text, "=== lscpu ===")
    if lscpu_section and "(lscpu unavailable)" not in lscpu_section:
        for line in lscpu_section.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip().lower()
            v = v.strip()
            if k == "socket(s)" and "sockets_per_node" not in info:
                try:
                    info["sockets_per_node"] = int(v)
                except ValueError:
                    pass
            elif k == "core(s) per socket" and "cores_per_socket" not in info:
                try:
                    info["cores_per_socket"] = int(v)
                except ValueError:
                    pass
            elif k == "thread(s) per core" and "threads_per_core" not in info:
                try:
                    info["threads_per_core"] = int(v)
                except ValueError:
                    pass
            elif k == "cpu(s)" and "cpus_per_node" not in info:
                try:
                    info["cpus_per_node"] = int(v)
                except ValueError:
                    pass
            elif k == "model name" and "cpu_model" not in info:
                info["cpu_model"] = v
            elif k == "numa node(s)" and "numa_nodes" not in info:
                try:
                    info["numa_nodes"] = int(v)
                except ValueError:
                    pass

    return HardwareInfo(**{k: v for k, v in info.items() if v is not None})


def _extract_section(text: str, header: str) -> str:
    """Return text between *header* and the next '===' section."""
    start = text.find(header)
    if start == -1:
        return ""
    start = text.find("\n", start) + 1
    end = text.find("===", start)
    return text[start:end].strip() if end != -1 else text[start:].strip()


# ---------------------------------------------------------------------------
# Cache management (used by store.py)
# ---------------------------------------------------------------------------

def is_stale(profile: ClusterProfile, max_age_days: int = _CACHE_MAX_AGE_DAYS) -> bool:
    """Return True if the profile is older than *max_age_days*."""
    age = datetime.now(tz=UTC) - profile.discovered_at
    return age > timedelta(days=max_age_days)
