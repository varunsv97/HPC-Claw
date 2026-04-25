"""Tests for analyze_bottlenecks."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from claw_backend.pgoa.analysis import analyze_bottlenecks
from claw_backend.pgoa.schema import (
    ComputeMetrics,
    CPUPerfMetrics,
    KPIMetrics,
    ProfileBundle,
    SlurmMetrics,
)


def _kpi() -> KPIMetrics:
    return KPIMetrics(primary_metric="elapsed_s", value=600.0, unit="seconds")


def _bundle(**kwargs) -> ProfileBundle:
    return ProfileBundle(
        run_id="test-run",
        timestamp=datetime.now(tz=timezone.utc),
        kpi=_kpi(),
        **kwargs,
    )


class TestAnalyzeBottlenecks(unittest.TestCase):
    def test_memory_bound_gpu(self):
        compute = ComputeMetrics(
            roofline_position="memory_bound",
            sm_occupancy_pct=40.0,
            memory_bw_utilization_pct=85.0,
        )
        report = analyze_bottlenecks(_bundle(compute=compute))
        self.assertEqual(report.primary_bottleneck, "memory_bound_gpu")

    def test_memory_bound_gpu_below_threshold(self):
        # mem_bw_util < 70 → should NOT trigger memory_bound_gpu
        compute = ComputeMetrics(
            roofline_position="memory_bound",
            sm_occupancy_pct=40.0,
            memory_bw_utilization_pct=60.0,
        )
        report = analyze_bottlenecks(_bundle(compute=compute))
        self.assertNotEqual(report.primary_bottleneck, "memory_bound_gpu")

    def test_compute_bound_gpu(self):
        compute = ComputeMetrics(
            roofline_position="compute_bound",
            sm_occupancy_pct=85.0,
        )
        report = analyze_bottlenecks(_bundle(compute=compute))
        self.assertEqual(report.primary_bottleneck, "compute_bound_gpu")

    def test_latency_bound_gpu(self):
        compute = ComputeMetrics(
            roofline_position="latency_bound",
            sm_occupancy_pct=15.0,
        )
        report = analyze_bottlenecks(_bundle(compute=compute))
        self.assertEqual(report.primary_bottleneck, "latency_bound_gpu")

    def test_high_rss(self):
        # 64 CPUs × 3800 MB = 243,200 MB threshold; set RSS to 300,000 MB
        slurm = SlurmMetrics(job_id=1, alloc_cpus=64, max_rss_mb=300_000.0)
        report = analyze_bottlenecks(_bundle(slurm=slurm))
        self.assertEqual(report.primary_bottleneck, "high_rss")

    def test_high_rss_not_triggered_within_threshold(self):
        slurm = SlurmMetrics(job_id=1, alloc_cpus=64, max_rss_mb=100_000.0)
        report = analyze_bottlenecks(_bundle(slurm=slurm))
        self.assertNotEqual(report.primary_bottleneck, "high_rss")

    def test_mpi_binding(self):
        slurm = SlurmMetrics(job_id=1, avg_cpu_pct=45.0, n_tasks=64)
        report = analyze_bottlenecks(_bundle(slurm=slurm))
        self.assertEqual(report.primary_bottleneck, "mpi_binding")

    def test_mpi_binding_single_task_not_triggered(self):
        slurm = SlurmMetrics(job_id=1, avg_cpu_pct=45.0, n_tasks=1)
        report = analyze_bottlenecks(_bundle(slurm=slurm))
        self.assertNotEqual(report.primary_bottleneck, "mpi_binding")

    def test_insufficient_data(self):
        slurm = SlurmMetrics(job_id=1, elapsed_s=600.0)
        report = analyze_bottlenecks(_bundle(slurm=slurm))
        self.assertEqual(report.primary_bottleneck, "insufficient_data")

    def test_none_detected(self):
        report = analyze_bottlenecks(_bundle())
        self.assertEqual(report.primary_bottleneck, "none_detected")

    def test_details_include_summaries_when_slurm_available(self):
        slurm = SlurmMetrics(job_id=1, elapsed_s=600.0, alloc_cpus=32, max_rss_mb=50_000.0)
        report = analyze_bottlenecks(_bundle(slurm=slurm))
        self.assertIn("slurm_summary", report.details)

    def test_details_include_compute_summary(self):
        compute = ComputeMetrics(roofline_position="compute_bound", sm_occupancy_pct=80.0)
        report = analyze_bottlenecks(_bundle(compute=compute))
        self.assertIn("compute_summary", report.details)

    def test_run_id_propagated(self):
        report = analyze_bottlenecks(_bundle())
        self.assertEqual(report.run_id, "test-run")


if __name__ == "__main__":
    unittest.main()
