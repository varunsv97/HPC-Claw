"""Cluster hardware discovery — login-node queries and optional probe jobs.

Two phases:
  1. Login-node discovery (always, no jobs needed):
     - ``sinfo --Node --format=...``  → partition list + node counts
     - ``scontrol show node``         → first node in each partition (cpu/mem/gpu)
     - ``hwloc-ls`` / ``/sys`` / ``/proc`` via the existing HardwareAdapter
     - ``nvidia-smi`` if available

  2. Probe-job discovery (gated by caller-supplied approval callback):
     - Submits a tiny ``srun --pty ...`` (or batch job) on each target partition
       that runs ``hwloc-ls --of xml`` + ``nvidia-smi`` and writes results to a
       temp file.
     - Merges results into ClusterProfile.partitions[*].hardware.

Results are saved to ``<store_base>/_cluster/<cluster_name>.json`` and reused
across workloads on the same cluster.  A max-age guard forces re-discovery when
the cached data is stale (default 7 days).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.pgoa.adapters.hardware import collect_hardware_info
from hpc_assistant_backend.pgoa.env_discovery import discover_software_env
from hpc_assistant_backend.pgoa.schema import ClusterProfile, HardwareInfo, PartitionInfo

log = logging.getLogger(__name__)

_CACHE_MAX_AGE_DAYS = 7

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
# Login-node discovery
# ---------------------------------------------------------------------------

def collect_login_node_info(settings: AssistantSettings) -> ClusterProfile:
    """Phase 1: collect everything reachable from the login node without jobs."""

    # Cluster name from scontrol ping (best effort)
    cluster_name = _detect_cluster_name(settings)

    # Local hardware from /sys + hwloc + nvidia-smi
    login_hw = collect_hardware_info(settings)

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


def _detect_cluster_name(settings: AssistantSettings) -> str:
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
    target_names = set(partitions) if partitions else None
    updated = list(profile.partitions)

    for i, part in enumerate(updated):
        if target_names and part.name not in target_names:
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
    from hpc_assistant_backend.pgoa.adapters.hardware import (
        _parse_hwloc_xml,
        _gpu_arch_from_cc,
    )

    info: dict = {}

    # hwloc XML block
    xml_m = _HWLOC_XML_RE.search(text)
    if xml_m:
        info.update(_parse_hwloc_xml(xml_m.group(1)))

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
