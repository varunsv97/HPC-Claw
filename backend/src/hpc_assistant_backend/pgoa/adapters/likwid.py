"""LIKWIDAdapter: parse likwid-perfctr text output into CPUPerfMetrics."""

from __future__ import annotations

import logging
import re

from hpc_assistant_backend.pgoa.adapters.base import BaseAdapter
from hpc_assistant_backend.pgoa.schema import CPUPerfMetrics, KPIMetrics, ProfileBundle

log = logging.getLogger(__name__)

# Regex to extract metric name and first numeric value from a LIKWID table row.
# Handles both plain format and table format with leading pipe:
#   "  DP [MFLOP/s]  |  12345.67  |  12350.00  "
#   "| IPC  |  0.69  |  0.69  |"
_ROW_RE = re.compile(r"^\s*\|?\s*(.+?)\s*\|\s*([\d.]+)")
_SECTION_SEP = re.compile(r"^-{5,}")


def _safe_float(val: str) -> float | None:
    try:
        return float(val.strip())
    except (ValueError, AttributeError):
        return None


def _sum_columns(line: str) -> float | None:
    """Sum all numeric pipe-separated values on a metric line (multi-socket)."""
    parts = line.split("|")
    values: list[float] = []
    for part in parts[1:]:  # skip the metric name in first column
        v = _safe_float(part)
        if v is not None:
            values.append(v)
    return sum(values) if values else None


def _parse_likwid(content: str) -> CPUPerfMetrics:
    metrics = CPUPerfMetrics()

    for line in content.splitlines():
        if _SECTION_SEP.match(line):
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue

        name_raw = m.group(1).strip().lower()
        # Sum all numeric columns to aggregate across sockets
        total = _sum_columns(line)
        if total is None:
            continue

        try:
            if ("dp" in name_raw and "mflop" in name_raw) or "dp mflop/s" in name_raw:
                metrics = metrics.model_copy(
                    update={"flops_dp_gflops": total / 1000.0}
                )
            elif ("sp" in name_raw and "mflop" in name_raw) or "sp mflop/s" in name_raw:
                metrics = metrics.model_copy(
                    update={"flops_sp_gflops": total / 1000.0}
                )
            elif "memory bandwidth" in name_raw or "memory bw" in name_raw:
                metrics = metrics.model_copy(
                    update={"memory_bw_dram_gbs": total / 1000.0}
                )
            elif "l2 bandwidth" in name_raw or "l2 bw" in name_raw:
                metrics = metrics.model_copy(
                    update={"memory_bw_l2_gbs": total / 1000.0}
                )
            elif "l3 bandwidth" in name_raw or "l3 bw" in name_raw or "llc bandwidth" in name_raw:
                metrics = metrics.model_copy(
                    update={"memory_bw_llc_gbs": total / 1000.0}
                )
            elif "energy [j]" in name_raw and "pkg" in name_raw:
                metrics = metrics.model_copy(update={"energy_pkg_joules": total})
            elif "energy [j]" in name_raw and "dram" in name_raw:
                metrics = metrics.model_copy(update={"energy_dram_joules": total})
            elif name_raw.strip() == "ipc" or name_raw.startswith("ipc ") or name_raw.startswith("ipc\t"):
                metrics = metrics.model_copy(update={"ipc": total})
            elif "vectorization ratio" in name_raw:
                metrics = metrics.model_copy(
                    update={"vectorization_ratio_pct": total}
                )
        except Exception:
            log.warning("Failed to parse LIKWID metric line: %r", line, exc_info=True)

    return metrics


class LIKWIDAdapter(BaseAdapter):
    def name(self) -> str:
        return "likwid"

    def collect(self, likwid_output_path: str, kpi: KPIMetrics) -> ProfileBundle:  # type: ignore[override]
        bundle = self._new_bundle(kpi)
        try:
            with open(likwid_output_path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            log.warning("Cannot open LIKWID output: %s", likwid_output_path, exc_info=True)
            return bundle

        try:
            cpu_perf = _parse_likwid(content)
        except Exception:
            log.warning("Failed to parse LIKWID output: %s", likwid_output_path, exc_info=True)
            cpu_perf = CPUPerfMetrics()

        bundle = bundle.model_copy(
            update={
                "cpu_perf": cpu_perf,
                "raw_sources": {**bundle.raw_sources, "likwid": content[:4000]},
            }
        )
        return bundle
