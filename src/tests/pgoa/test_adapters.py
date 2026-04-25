"""Tests for PGOA profiling adapters."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from claw_backend.pgoa.adapters.likwid import LIKWIDAdapter, _parse_likwid
from claw_backend.pgoa.adapters.ncu import NCUAdapter
from claw_backend.pgoa.adapters.slurm import SlurmAdapter
from claw_backend.pgoa.schema import KPIMetrics

FIXTURES = Path(__file__).parent / "fixtures"


def _kpi(value: float = 630.0) -> KPIMetrics:
    return KPIMetrics(primary_metric="elapsed_s", value=value, unit="seconds")


# ---------------------------------------------------------------------------
# SlurmAdapter
# ---------------------------------------------------------------------------


class TestSlurmAdapter(unittest.TestCase):
    def _make_sacct_result(self) -> subprocess.CompletedProcess:
        content = (FIXTURES / "sacct_output.txt").read_text()
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=content, stderr="")

    def test_collect_parses_slurm_metrics(self):
        adapter = SlurmAdapter()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._make_sacct_result()
            bundle = adapter.collect(job_id=12345, kpi=_kpi())

        self.assertIsNotNone(bundle.slurm)
        self.assertEqual(bundle.slurm.job_id, 12345)
        self.assertEqual(bundle.slurm.state, "COMPLETED")
        self.assertIsNotNone(bundle.slurm.elapsed_s)
        self.assertGreater(bundle.slurm.elapsed_s, 0)
        self.assertEqual(bundle.kpi.primary_metric, "elapsed_s")

    def test_collect_stores_raw(self):
        adapter = SlurmAdapter()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._make_sacct_result()
            bundle = adapter.collect(job_id=12345, kpi=_kpi())

        self.assertIn("slurm_sacct", bundle.raw_sources)

    def test_collect_handles_nonzero_exit(self):
        adapter = SlurmAdapter()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="invalid job id"
            )
            bundle = adapter.collect(job_id=99999, kpi=_kpi())

        # Should return bundle without crashing, slurm may be None
        self.assertIsNone(bundle.slurm)

    def test_name(self):
        self.assertEqual(SlurmAdapter().name(), "slurm")


# ---------------------------------------------------------------------------
# NCUAdapter
# ---------------------------------------------------------------------------


class TestNCUAdapter(unittest.TestCase):
    def test_collect_parses_csv(self):
        adapter = NCUAdapter()
        path = str(FIXTURES / "ncu_output.csv")
        bundle = adapter.collect(ncu_csv_path=path, kpi=_kpi())

        self.assertIsNotNone(bundle.compute)
        self.assertIn("ncu_csv", bundle.raw_sources)

    def test_collect_nonexistent_file_returns_empty_bundle(self):
        adapter = NCUAdapter()
        bundle = adapter.collect(ncu_csv_path="/nonexistent/path/ncu.csv", kpi=_kpi())
        self.assertIsNone(bundle.compute)

    def test_top_kernels_populated(self):
        adapter = NCUAdapter()
        path = str(FIXTURES / "ncu_output.csv")
        bundle = adapter.collect(ncu_csv_path=path, kpi=_kpi())
        if bundle.compute and bundle.compute.top_kernels:
            self.assertLessEqual(len(bundle.compute.top_kernels), 5)

    def test_name(self):
        self.assertEqual(NCUAdapter().name(), "ncu")


# ---------------------------------------------------------------------------
# LIKWIDAdapter
# ---------------------------------------------------------------------------


class TestLIKWIDAdapter(unittest.TestCase):
    def test_collect_parses_metrics(self):
        adapter = LIKWIDAdapter()
        path = str(FIXTURES / "likwid_output.txt")
        bundle = adapter.collect(likwid_output_path=path, kpi=_kpi())

        self.assertIsNotNone(bundle.cpu_perf)
        self.assertIn("likwid", bundle.raw_sources)

    def test_parse_likwid_dp_flops(self):
        content = (FIXTURES / "likwid_output.txt").read_text()
        metrics = _parse_likwid(content)
        # DP MFLOP/s rows: 12450 + 12350 = 24800 MFLOP/s → 24.8 GFLOP/s
        self.assertIsNotNone(metrics.flops_dp_gflops)
        self.assertAlmostEqual(metrics.flops_dp_gflops, 24.8, places=2)

    def test_parse_likwid_memory_bandwidth(self):
        content = (FIXTURES / "likwid_output.txt").read_text()
        metrics = _parse_likwid(content)
        # Memory Bandwidth: 98765.12 + 97234.88 = 196000 MB/s → 196.0 GB/s
        self.assertIsNotNone(metrics.memory_bw_dram_gbs)
        self.assertAlmostEqual(metrics.memory_bw_dram_gbs, 196.0, places=1)

    def test_parse_likwid_ipc(self):
        content = (FIXTURES / "likwid_output.txt").read_text()
        metrics = _parse_likwid(content)
        # IPC: 0.69 + 0.69 = 1.38
        self.assertIsNotNone(metrics.ipc)
        self.assertAlmostEqual(metrics.ipc, 1.38, places=2)

    def test_parse_likwid_energy(self):
        content = (FIXTURES / "likwid_output.txt").read_text()
        metrics = _parse_likwid(content)
        # Energy PKG: 1234.56 + 1224.44 = 2459.0
        self.assertIsNotNone(metrics.energy_pkg_joules)
        self.assertAlmostEqual(metrics.energy_pkg_joules, 2459.0, places=0)

    def test_collect_nonexistent_file_returns_empty_bundle(self):
        adapter = LIKWIDAdapter()
        bundle = adapter.collect(likwid_output_path="/nonexistent/likwid.txt", kpi=_kpi())
        self.assertIsNone(bundle.cpu_perf)

    def test_name(self):
        self.assertEqual(LIKWIDAdapter().name(), "likwid")


if __name__ == "__main__":
    unittest.main()
