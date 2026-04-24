"""ExperimentStore: filesystem-backed, atomic-write storage for PGOA runs."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

from hpc_assistant_backend.pgoa.schema import (
    ClusterProfile,
    DeltaReport,
    ProfileBundle,
    RunHandle,
)

log = logging.getLogger(__name__)


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a .tmp file then rename (atomic on POSIX)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


class ExperimentStore:
    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir
        self._base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(
        self,
        workload_id: str,
        run_type: Literal["baseline", "iteration"],
        iteration: int | None = None,
    ) -> RunHandle:
        run_id = str(uuid4())
        if run_type == "baseline":
            dir_name = "baseline"
        else:
            if iteration is None:
                iteration = self._next_iteration(workload_id)
            dir_name = f"iteration_{iteration:03d}"

        path = self._base / workload_id / dir_name
        path.mkdir(parents=True, exist_ok=True)

        # Persist run_id so load_bundle can look it up
        _atomic_write(path / "run_id.txt", run_id)

        return RunHandle(
            workload_id=workload_id,
            run_id=run_id,
            path=path,
            run_type=run_type,
            iteration=iteration,
        )

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_bundle(self, workload_id: str, run_id: str, bundle: ProfileBundle) -> None:
        path = self._run_path(workload_id, run_id)
        _atomic_write(path / "profile.json", bundle.model_dump_json(indent=2))

    def save_job_script(self, workload_id: str, run_id: str, script: str) -> None:
        path = self._run_path(workload_id, run_id)
        _atomic_write(path / "job_script.sh", script)

    def save_config(self, workload_id: str, run_id: str, config: dict) -> None:
        path = self._run_path(workload_id, run_id)
        _atomic_write(path / "config.json", json.dumps(config, indent=2))

    def save_rationale(self, workload_id: str, run_id: str, rationale: str) -> None:
        path = self._run_path(workload_id, run_id)
        _atomic_write(path / "rationale.txt", rationale)

    def save_delta(self, workload_id: str, run_id: str, delta: DeltaReport) -> None:
        path = self._run_path(workload_id, run_id)
        _atomic_write(path / "delta.json", delta.model_dump_json(indent=2))

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def load_bundle(self, workload_id: str, run_id: str) -> ProfileBundle:
        path = self._run_path(workload_id, run_id)
        return ProfileBundle.model_validate_json(
            (path / "profile.json").read_text(encoding="utf-8")
        )

    def list_runs(self, workload_id: str) -> list[RunHandle]:
        workload_dir = self._base / workload_id
        if not workload_dir.exists():
            return []

        handles: list[RunHandle] = []
        baseline_handle: RunHandle | None = None

        for entry in sorted(workload_dir.iterdir()):
            if not entry.is_dir():
                continue
            run_id_file = entry / "run_id.txt"
            if not run_id_file.exists():
                continue
            run_id = run_id_file.read_text(encoding="utf-8").strip()
            name = entry.name
            if name == "baseline":
                baseline_handle = RunHandle(
                    workload_id=workload_id,
                    run_id=run_id,
                    path=entry,
                    run_type="baseline",
                )
            elif name.startswith("iteration_"):
                try:
                    iteration = int(name.split("_", 1)[1])
                except ValueError:
                    continue
                handles.append(
                    RunHandle(
                        workload_id=workload_id,
                        run_id=run_id,
                        path=entry,
                        run_type="iteration",
                        iteration=iteration,
                    )
                )

        result: list[RunHandle] = []
        if baseline_handle:
            result.append(baseline_handle)
        result.extend(sorted(handles, key=lambda h: h.iteration or 0))
        return result

    def get_baseline(self, workload_id: str) -> ProfileBundle | None:
        baseline_dir = self._base / workload_id / "baseline"
        run_id_file = baseline_dir / "run_id.txt"
        profile_file = baseline_dir / "profile.json"
        if not profile_file.exists():
            return None
        try:
            run_id = run_id_file.read_text(encoding="utf-8").strip()
            return self.load_bundle(workload_id, run_id)
        except Exception:
            log.warning("Failed to load baseline bundle for %s", workload_id, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    def compute_delta(
        self,
        workload_id: str,
        from_run_id: str,
        to_run_id: str,
        action_applied: str,
    ) -> DeltaReport:
        old = self.load_bundle(workload_id, from_run_id)
        new = self.load_bundle(workload_id, to_run_id)

        old_val = old.kpi.value
        new_val = new.kpi.value

        if old_val == 0:
            kpi_delta_pct = 0.0
        else:
            kpi_delta_pct = (new_val - old_val) / abs(old_val) * 100.0

        lower_is_better = old.kpi.lower_is_better
        if lower_is_better and kpi_delta_pct < -2.0:
            direction = "improved"
        elif lower_is_better and kpi_delta_pct > 2.0:
            direction = "degraded"
        elif not lower_is_better and kpi_delta_pct > 2.0:
            direction = "improved"
        elif not lower_is_better and kpi_delta_pct < -2.0:
            direction = "degraded"
        else:
            direction = "neutral"

        secondary = _compute_secondary_deltas(old, new)

        return DeltaReport(
            from_run_id=from_run_id,
            to_run_id=to_run_id,
            action_applied=action_applied,
            kpi_delta_pct=kpi_delta_pct,
            kpi_direction=direction,  # type: ignore[arg-type]
            secondary_deltas=secondary,
        )

    # ------------------------------------------------------------------
    # Cluster profile (shared across workloads, keyed by cluster name)
    # ------------------------------------------------------------------

    def load_cluster_profile(self, cluster_name: str) -> ClusterProfile | None:
        path = self._cluster_path(cluster_name)
        if not path.exists():
            return None
        try:
            return ClusterProfile.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to load cluster profile for %s", cluster_name, exc_info=True)
            return None

    def save_cluster_profile(self, profile: ClusterProfile) -> None:
        path = self._cluster_path(profile.cluster_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, profile.model_dump_json(indent=2))

    def _cluster_path(self, cluster_name: str) -> Path:
        # Sanitise cluster name to a safe filename
        safe = re.sub(r"[^\w\-.]", "_", cluster_name)
        return self._base / "_cluster" / f"{safe}.json"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_path(self, workload_id: str, run_id: str) -> Path:
        """Locate the directory for a run_id by scanning for run_id.txt."""
        workload_dir = self._base / workload_id
        if not workload_dir.exists():
            raise FileNotFoundError(f"Workload directory not found: {workload_dir}")
        for entry in workload_dir.iterdir():
            if not entry.is_dir():
                continue
            run_id_file = entry / "run_id.txt"
            if run_id_file.exists() and run_id_file.read_text(encoding="utf-8").strip() == run_id:
                return entry
        raise FileNotFoundError(f"Run {run_id!r} not found in workload {workload_id!r}")

    def _next_iteration(self, workload_id: str) -> int:
        workload_dir = self._base / workload_id
        if not workload_dir.exists():
            return 1
        max_iter = 0
        for entry in workload_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("iteration_"):
                try:
                    n = int(entry.name.split("_", 1)[1])
                    max_iter = max(max_iter, n)
                except ValueError:
                    pass
        return max_iter + 1


def _compute_secondary_deltas(
    old: ProfileBundle, new: ProfileBundle
) -> dict[str, float]:
    """Compute percentage deltas for all comparable numeric leaf fields."""
    deltas: dict[str, float] = {}

    def _extract_numeric(model, prefix: str) -> dict[str, float]:
        out: dict[str, float] = {}
        if model is None:
            return out
        for field_name, value in model.model_dump().items():
            if isinstance(value, (int, float)) and value is not None:
                out[f"{prefix}.{field_name}"] = float(value)
        return out

    old_fields: dict[str, float] = {}
    new_fields: dict[str, float] = {}
    for bundle, fields in ((old, old_fields), (new, new_fields)):
        fields.update(_extract_numeric(bundle.slurm, "slurm"))
        fields.update(_extract_numeric(bundle.cpu_perf, "cpu_perf"))

    for key in old_fields:
        if key in new_fields and old_fields[key] != 0:
            deltas[key] = (new_fields[key] - old_fields[key]) / abs(old_fields[key]) * 100.0

    return deltas
