"""Shared hwloc XML and GPU compute-capability parsing utilities.

These helpers are used by cluster hardware inventory and probe parsing. Keeping
them here avoids duplicating low-level parsers inside the inventory code.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

log = logging.getLogger(__name__)


def parse_hwloc_xml(xml_text: str) -> dict:
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


def gpu_arch_from_cc(compute_cap: str) -> str | None:
    """Return the GPU microarchitecture name for a compute capability string."""
    return _GPU_ARCH_BY_CC.get(compute_cap.strip())
