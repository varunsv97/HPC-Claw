"""Tests for ExperimentStore."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from claw_backend.pgoa.schema import (
    KPIMetrics,
    ProfileBundle,
    SlurmMetrics,
)
from claw_backend.pgoa.store import ExperimentStore


def _kpi(value: float = 630.0, lower_is_better: bool = True) -> KPIMetrics:
    return KPIMetrics(
        primary_metric="elapsed_s",
        value=value,
        unit="seconds",
        lower_is_better=lower_is_better,
    )


def _make_bundle(run_id: str, kpi: KPIMetrics, **kwargs) -> ProfileBundle:
    return ProfileBundle(
        run_id=run_id,
        timestamp=datetime.now(tz=timezone.utc),
        kpi=kpi,
        **kwargs,
    )


class TestExperimentStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        self._store = ExperimentStore(self._base)

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_baseline_run(self):
        handle = self._store.create_run("wl1", "baseline")
        self.assertEqual(handle.run_type, "baseline")
        self.assertIsNotNone(handle.run_id)
        self.assertTrue(handle.path.exists())
        self.assertTrue((handle.path / "run_id.txt").exists())

    def test_create_iteration_run(self):
        handle = self._store.create_run("wl1", "iteration", iteration=1)
        self.assertEqual(handle.run_type, "iteration")
        self.assertEqual(handle.iteration, 1)

    def test_auto_increment_iteration(self):
        self._store.create_run("wl1", "iteration", iteration=1)
        h2 = self._store.create_run("wl1", "iteration")
        self.assertEqual(h2.iteration, 2)

    def test_save_and_load_bundle(self):
        handle = self._store.create_run("wl1", "baseline")
        bundle = _make_bundle(handle.run_id, _kpi())
        self._store.save_bundle("wl1", handle.run_id, bundle)
        loaded = self._store.load_bundle("wl1", handle.run_id)
        self.assertEqual(loaded.run_id, handle.run_id)

    def test_atomic_write_uses_tmp_then_rename(self):
        """Verify no leftover .tmp files after save."""
        handle = self._store.create_run("wl1", "baseline")
        bundle = _make_bundle(handle.run_id, _kpi())
        self._store.save_bundle("wl1", handle.run_id, bundle)
        tmp_files = list(handle.path.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0, "Leftover .tmp files found")

    def test_list_runs_order(self):
        b = self._store.create_run("wl1", "baseline")
        i1 = self._store.create_run("wl1", "iteration", iteration=1)
        i2 = self._store.create_run("wl1", "iteration", iteration=2)

        bundle_b = _make_bundle(b.run_id, _kpi(630.0))
        bundle_i1 = _make_bundle(i1.run_id, _kpi(600.0))
        bundle_i2 = _make_bundle(i2.run_id, _kpi(580.0))
        for wid, rid, bun in [
            ("wl1", b.run_id, bundle_b),
            ("wl1", i1.run_id, bundle_i1),
            ("wl1", i2.run_id, bundle_i2),
        ]:
            self._store.save_bundle(wid, rid, bun)

        runs = self._store.list_runs("wl1")
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0].run_type, "baseline")
        self.assertEqual(runs[1].iteration, 1)
        self.assertEqual(runs[2].iteration, 2)

    def test_get_baseline_none_when_no_profile(self):
        self._store.create_run("wl1", "baseline")
        # No save_bundle call → get_baseline should return None
        result = self._store.get_baseline("wl1")
        self.assertIsNone(result)

    def test_get_baseline_returns_bundle(self):
        handle = self._store.create_run("wl1", "baseline")
        bundle = _make_bundle(handle.run_id, _kpi())
        self._store.save_bundle("wl1", handle.run_id, bundle)
        result = self._store.get_baseline("wl1")
        self.assertIsNotNone(result)
        self.assertEqual(result.run_id, handle.run_id)

    def test_compute_delta_improved(self):
        b = self._store.create_run("wl1", "baseline")
        i1 = self._store.create_run("wl1", "iteration", iteration=1)
        self._store.save_bundle("wl1", b.run_id, _make_bundle(b.run_id, _kpi(100.0)))
        self._store.save_bundle("wl1", i1.run_id, _make_bundle(i1.run_id, _kpi(90.0)))
        delta = self._store.compute_delta("wl1", b.run_id, i1.run_id, "mem_bind=local")
        self.assertEqual(delta.kpi_direction, "improved")
        self.assertAlmostEqual(delta.kpi_delta_pct, -10.0, places=1)

    def test_compute_delta_degraded(self):
        b = self._store.create_run("wl1", "baseline")
        i1 = self._store.create_run("wl1", "iteration", iteration=1)
        self._store.save_bundle("wl1", b.run_id, _make_bundle(b.run_id, _kpi(100.0)))
        self._store.save_bundle("wl1", i1.run_id, _make_bundle(i1.run_id, _kpi(110.0)))
        delta = self._store.compute_delta("wl1", b.run_id, i1.run_id, "bad_action")
        self.assertEqual(delta.kpi_direction, "degraded")

    def test_compute_delta_neutral(self):
        b = self._store.create_run("wl1", "baseline")
        i1 = self._store.create_run("wl1", "iteration", iteration=1)
        self._store.save_bundle("wl1", b.run_id, _make_bundle(b.run_id, _kpi(100.0)))
        self._store.save_bundle("wl1", i1.run_id, _make_bundle(i1.run_id, _kpi(101.0)))
        delta = self._store.compute_delta("wl1", b.run_id, i1.run_id, "minor_tweak")
        self.assertEqual(delta.kpi_direction, "neutral")

    def test_save_delta(self):
        b = self._store.create_run("wl1", "baseline")
        i1 = self._store.create_run("wl1", "iteration", iteration=1)
        self._store.save_bundle("wl1", b.run_id, _make_bundle(b.run_id, _kpi(100.0)))
        self._store.save_bundle("wl1", i1.run_id, _make_bundle(i1.run_id, _kpi(90.0)))
        delta = self._store.compute_delta("wl1", b.run_id, i1.run_id, "action")
        self._store.save_delta("wl1", i1.run_id, delta)
        delta_file = self._store._run_path("wl1", i1.run_id) / "delta.json"
        self.assertTrue(delta_file.exists())
        data = json.loads(delta_file.read_text())
        self.assertEqual(data["kpi_direction"], "improved")

    def test_save_job_script(self):
        handle = self._store.create_run("wl1", "baseline")
        self._store.save_job_script("wl1", handle.run_id, "#!/bin/bash\n#SBATCH --ntasks=64\n")
        script_file = handle.path / "job_script.sh"
        self.assertTrue(script_file.exists())

    def test_list_runs_empty_workload(self):
        runs = self._store.list_runs("nonexistent_workload")
        self.assertEqual(runs, [])

    def test_load_bundle_nonexistent_run_raises(self):
        with self.assertRaises(FileNotFoundError):
            self._store.load_bundle("wl1", "nonexistent-run-id")


if __name__ == "__main__":
    unittest.main()
