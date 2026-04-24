"""Tests for PGOA Pydantic v2 schema models."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from hpc_assistant_backend.pgoa.schema import (
    ActionProposal,
    BottleneckReport,
    ComputeMetrics,
    CPUPerfMetrics,
    DeltaReport,
    HardwareInfo,
    KernelStat,
    KPIMetrics,
    OptimizationResult,
    ProfileBundle,
    RunHandle,
    SlurmMetrics,
)
from pathlib import Path


class TestKPIMetrics(unittest.TestCase):
    def test_defaults(self):
        kpi = KPIMetrics(primary_metric="elapsed_s", value=630.0, unit="seconds")
        self.assertTrue(kpi.lower_is_better)

    def test_explicit_fields(self):
        kpi = KPIMetrics(
            primary_metric="throughput",
            value=1234.5,
            unit="samples/s",
            lower_is_better=False,
        )
        self.assertEqual(kpi.value, 1234.5)
        self.assertFalse(kpi.lower_is_better)


class TestProfileBundle(unittest.TestCase):
    def _make_bundle(self, **kwargs):
        kpi = KPIMetrics(primary_metric="elapsed_s", value=100.0, unit="s")
        return ProfileBundle(
            run_id="abc-123",
            timestamp=datetime.now(tz=timezone.utc),
            kpi=kpi,
            **kwargs,
        )

    def test_minimal_bundle(self):
        b = self._make_bundle()
        self.assertIsNone(b.slurm)
        self.assertIsNone(b.compute)
        self.assertIsNone(b.cpu_perf)
        self.assertEqual(b.raw_sources, {})

    def test_bundle_with_slurm(self):
        slurm = SlurmMetrics(job_id=12345, elapsed_s=630.0)
        b = self._make_bundle(slurm=slurm)
        self.assertEqual(b.slurm.job_id, 12345)

    def test_model_copy_immutability(self):
        b = self._make_bundle()
        b2 = b.model_copy(update={"raw_sources": {"test": "data"}})
        self.assertEqual(b.raw_sources, {})
        self.assertEqual(b2.raw_sources["test"], "data")

    def test_json_roundtrip(self):
        b = self._make_bundle(
            slurm=SlurmMetrics(job_id=1, elapsed_s=60.0),
            hardware=HardwareInfo(nodes=2, cpus_per_node=64),
        )
        json_str = b.model_dump_json()
        restored = ProfileBundle.model_validate_json(json_str)
        self.assertEqual(restored.run_id, b.run_id)
        self.assertEqual(restored.slurm.job_id, 1)


class TestBottleneckReport(unittest.TestCase):
    def test_valid_bottleneck(self):
        r = BottleneckReport(
            run_id="abc",
            primary_bottleneck="mpi_binding",
            details={"slurm_summary": "elapsed=600s"},
            recommended_action_hint="hint",
        )
        self.assertEqual(r.primary_bottleneck, "mpi_binding")


class TestDeltaReport(unittest.TestCase):
    def test_delta_fields(self):
        d = DeltaReport(
            from_run_id="run_a",
            to_run_id="run_b",
            action_applied="mem_bind=local",
            kpi_delta_pct=-5.5,
            kpi_direction="improved",
        )
        self.assertEqual(d.kpi_direction, "improved")
        self.assertEqual(d.secondary_deltas, {})


class TestOptimizationResult(unittest.TestCase):
    def test_result_construction(self):
        r = OptimizationResult(
            workload_id="wl1",
            iterations_run=3,
            best_run_id="iter_002_run",
            total_kpi_improvement_pct=8.2,
            convergence_reason="below_threshold",
            all_deltas=[],
        )
        self.assertEqual(r.iterations_run, 3)


class TestRunHandle(unittest.TestCase):
    def test_path_field(self):
        handle = RunHandle(
            workload_id="wl",
            run_id="abc",
            path=Path("/tmp/experiments/wl/baseline"),
            run_type="baseline",
        )
        self.assertIsInstance(handle.path, Path)


if __name__ == "__main__":
    unittest.main()
