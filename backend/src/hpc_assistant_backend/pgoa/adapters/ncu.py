"""NCUAdapter: parse Nsight Compute --csv output into ComputeMetrics."""

from __future__ import annotations

import csv
import io
import logging
from collections import defaultdict

from hpc_assistant_backend.pgoa.adapters.base import BaseAdapter
from hpc_assistant_backend.pgoa.schema import (
    ComputeMetrics,
    KernelStat,
    KPIMetrics,
    ProfileBundle,
)

log = logging.getLogger(__name__)

_RAW_MAX_CHARS = 4000


def _find_col(headers: list[str], *substrings: str) -> str | None:
    """Return the first column whose lowercase name contains all given substrings."""
    for h in headers:
        lower = h.lower()
        if all(s.lower() in lower for s in substrings):
            return h
    return None


def _safe_float(val: str) -> float | None:
    try:
        return float(val.strip().replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _classify_roofline(
    mem_bw_util: float | None,
    sm_occ: float | None,
) -> str:
    if mem_bw_util is not None and mem_bw_util > 70:
        return "memory_bound"
    if sm_occ is not None and sm_occ > 70:
        return "compute_bound"
    if sm_occ is not None and sm_occ < 20:
        return "latency_bound"
    return "unknown"


class NCUAdapter(BaseAdapter):
    def name(self) -> str:
        return "ncu"

    def collect(self, ncu_csv_path: str, kpi: KPIMetrics) -> ProfileBundle:  # type: ignore[override]
        bundle = self._new_bundle(kpi)
        try:
            with open(ncu_csv_path, encoding="utf-8", errors="replace") as fh:
                raw_content = fh.read()
        except OSError:
            log.warning("Cannot open NCU CSV: %s", ncu_csv_path, exc_info=True)
            return bundle

        raw_snippet = raw_content[:_RAW_MAX_CHARS]

        # NCU CSV files often have comment lines starting with "==" before the
        # actual CSV data. Skip those to find the real header row.
        csv_lines = [ln for ln in raw_content.splitlines() if not ln.startswith("==")]
        csv_text = "\n".join(csv_lines)

        try:
            reader = csv.DictReader(io.StringIO(csv_text))
            rows = list(reader)
        except Exception:
            log.warning("Failed to parse NCU CSV: %s", ncu_csv_path, exc_info=True)
            bundle = bundle.model_copy(update={"raw_sources": {"ncu_csv": raw_snippet}})
            return bundle

        if not rows:
            bundle = bundle.model_copy(update={"raw_sources": {"ncu_csv": raw_snippet}})
            return bundle

        headers = list(rows[0].keys())

        # Column discovery via fuzzy substring matching
        col_sm_occ = _find_col(headers, "warps_active", "pct_of_peak")
        col_dram_bw = _find_col(headers, "dram__bytes.sum.per_second")
        col_mem_util = _find_col(headers, "sm__throughput", "pct_of_peak")
        col_duration = _find_col(headers, "Duration")
        col_kernel = _find_col(headers, "Kernel Name")

        # Aggregate metrics across all rows
        sm_occ_vals: list[float] = []
        mem_bw_util_vals: list[float] = []
        dram_bw_vals: list[float] = []

        # Per-kernel duration aggregation
        kernel_durations: dict[str, float] = defaultdict(float)
        kernel_sm_occ: dict[str, list[float]] = defaultdict(list)

        for row in rows:
            try:
                if col_sm_occ:
                    v = _safe_float(row.get(col_sm_occ, ""))
                    if v is not None:
                        sm_occ_vals.append(v)
                if col_mem_util:
                    v = _safe_float(row.get(col_mem_util, ""))
                    if v is not None:
                        mem_bw_util_vals.append(v)
                if col_dram_bw:
                    v = _safe_float(row.get(col_dram_bw, ""))
                    if v is not None:
                        dram_bw_vals.append(v)

                kernel_name = row.get(col_kernel, "unknown").strip() if col_kernel else "unknown"
                if col_duration:
                    dur = _safe_float(row.get(col_duration, ""))
                    if dur is not None:
                        kernel_durations[kernel_name] += dur
                        if col_sm_occ:
                            occ = _safe_float(row.get(col_sm_occ, ""))
                            if occ is not None:
                                kernel_sm_occ[kernel_name].append(occ)
            except Exception:
                log.warning("Skipping malformed NCU row", exc_info=True)

        avg_sm_occ = _mean(sm_occ_vals)
        avg_mem_util = _mean(mem_bw_util_vals)
        avg_dram_bw_gbs = (_mean(dram_bw_vals) / 1e9) if dram_bw_vals else None

        roofline = _classify_roofline(avg_mem_util, avg_sm_occ)

        # Top-5 kernels by total duration
        total_dur = sum(kernel_durations.values()) or 1.0
        sorted_kernels = sorted(kernel_durations.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_kernels = [
            KernelStat(
                name=kname,
                duration_pct=dur / total_dur * 100.0,
                sm_occupancy_pct=_mean(kernel_sm_occ.get(kname, [])),
            )
            for kname, dur in sorted_kernels
        ]

        compute = ComputeMetrics(
            roofline_position=roofline,  # type: ignore[arg-type]
            sm_occupancy_pct=avg_sm_occ,
            achieved_memory_bw_gbs=avg_dram_bw_gbs,
            memory_bw_utilization_pct=avg_mem_util,
            top_kernels=top_kernels,
        )

        bundle = bundle.model_copy(
            update={
                "compute": compute,
                "raw_sources": {**bundle.raw_sources, "ncu_csv": raw_snippet},
            }
        )
        return bundle


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
