"""SlurmAdapter: collect ProfileBundle from sacct/sstat for a completed job."""

from __future__ import annotations

import logging
import subprocess

from hpc_assistant_backend.config import AssistantSettings, load_settings
from hpc_assistant_backend.pgoa.adapters.base import BaseAdapter
from hpc_assistant_backend.pgoa.schema import KPIMetrics, ProfileBundle, SlurmMetrics

log = logging.getLogger(__name__)

_SACCT_FORMAT = (
    "JobID,JobName,Elapsed,CPUTimeRAW,AllocCPUS,NNodes,"
    "MaxRSS,AveCPU,ExitCode,State,Partition,NTasks"
)
_SSTAT_FORMAT = "JobID,MaxRSS,AveCPU"


def _elapsed_to_seconds(elapsed: str) -> float | None:
    """Convert HH:MM:SS (or D-HH:MM:SS) to float seconds."""
    try:
        elapsed = elapsed.strip()
        days = 0
        if "-" in elapsed:
            day_part, elapsed = elapsed.split("-", 1)
            days = int(day_part)
        parts = elapsed.split(":")
        if len(parts) != 3:
            return None
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return days * 86400 + h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None


def _rss_to_mb(rss_str: str) -> float | None:
    """Convert sacct MaxRSS (e.g. '1234567K') to megabytes."""
    try:
        val = rss_str.strip()
        if not val or val == "0":
            return None
        if val.endswith("K"):
            return float(val[:-1]) / 1024.0
        if val.endswith("M"):
            return float(val[:-1])
        if val.endswith("G"):
            return float(val[:-1]) * 1024.0
        return float(val) / 1024.0
    except (ValueError, AttributeError):
        return None


def _parse_avg_cpu(avg_cpu_str: str) -> float | None:
    """Parse AveCPU percentage, stripping trailing '%'."""
    try:
        return float(avg_cpu_str.strip().rstrip("%"))
    except (ValueError, AttributeError):
        return None


def _run_sacct(job_id: int, settings: AssistantSettings) -> str:
    result = subprocess.run(
        [
            "sacct",
            "-j",
            str(job_id),
            f"--format={_SACCT_FORMAT}",
            "--noheader",
            "--parsable2",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=settings.command_timeout_seconds,
    )
    return result.stdout


def _run_sstat(job_id: int, settings: AssistantSettings) -> str:
    try:
        result = subprocess.run(
            [
                "sstat",
                "-j",
                f"{job_id}.batch",
                f"--format={_SSTAT_FORMAT}",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=settings.command_timeout_seconds,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _parse_sacct(raw: str, job_id: int) -> SlurmMetrics:
    """Parse --parsable2 sacct output into a SlurmMetrics object."""
    metrics = SlurmMetrics(job_id=job_id)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) < 12:
            continue
        row_job_id = fields[0].strip()
        # Skip job steps (e.g. "12345.batch", "12345.0")
        if "." in row_job_id:
            continue
        try:
            (
                _,
                job_name,
                elapsed,
                cpu_time_raw,
                alloc_cpus,
                alloc_nodes,
                max_rss,
                ave_cpu,
                exit_code,
                state,
                partition,
                n_tasks,
            ) = fields[:12]

            metrics = SlurmMetrics(
                job_id=job_id,
                job_name=job_name.strip() or None,
                elapsed_s=_elapsed_to_seconds(elapsed),
                cpu_time_raw_s=_safe_int(cpu_time_raw),
                alloc_cpus=_safe_int(alloc_cpus),
                alloc_nodes=_safe_int(alloc_nodes),
                max_rss_mb=_rss_to_mb(max_rss),
                avg_cpu_pct=_parse_avg_cpu(ave_cpu),
                exit_code=exit_code.strip() or None,
                state=state.strip() or None,
                partition=partition.strip() or None,
                n_tasks=_safe_int(n_tasks),
            )
            break  # use only the main job line
        except Exception:
            log.warning("Failed to parse sacct line: %r", line, exc_info=True)
    return metrics


def _safe_int(val: str) -> int | None:
    try:
        stripped = val.strip()
        return int(stripped) if stripped else None
    except (ValueError, AttributeError):
        return None


class SlurmAdapter(BaseAdapter):
    def __init__(self, settings: AssistantSettings | None = None) -> None:
        self._settings = settings or load_settings()

    def name(self) -> str:
        return "slurm"

    def collect(self, job_id: int, kpi: KPIMetrics) -> ProfileBundle:  # type: ignore[override]
        bundle = self._new_bundle(kpi)

        sacct_raw = ""
        slurm: SlurmMetrics | None = None
        try:
            sacct_raw = _run_sacct(job_id, self._settings)
            if sacct_raw.strip():
                slurm = _parse_sacct(sacct_raw, job_id)
        except Exception:
            log.warning("sacct collection failed for job %d", job_id, exc_info=True)

        # Supplement MaxRSS from sstat if running
        try:
            sstat_raw = _run_sstat(job_id, self._settings)
            if sstat_raw.strip():
                for line in sstat_raw.splitlines():
                    fields = line.split("|")
                    if len(fields) >= 2:
                        rss_from_sstat = _rss_to_mb(fields[1])
                        if rss_from_sstat and slurm.max_rss_mb is None:
                            slurm = slurm.model_copy(update={"max_rss_mb": rss_from_sstat})
                        break
        except Exception:
            log.warning("sstat collection failed for job %d", job_id, exc_info=True)

        bundle = bundle.model_copy(
            update={
                "slurm": slurm,
                "raw_sources": {"slurm_sacct": sacct_raw},
            }
        )
        return bundle
