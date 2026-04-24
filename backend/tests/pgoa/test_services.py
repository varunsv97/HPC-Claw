"""Tests for the shared PGOA service layer."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.pgoa.schema import ComputeMetrics, KPIMetrics, ProfileBundle, SlurmMetrics
from hpc_assistant_backend.pgoa.services import collect_ncu_profile, collect_slurm_profile
from hpc_assistant_backend.pgoa.store import ExperimentStore


def _bundle_with_slurm(run_id: str, kpi: KPIMetrics) -> ProfileBundle:
    return ProfileBundle(
        run_id=run_id,
        timestamp=datetime.now(tz=timezone.utc),
        kpi=kpi,
        slurm=SlurmMetrics(job_id=12345, elapsed_s=kpi.value, state="COMPLETED"),
    )


def _bundle_with_compute(run_id: str, kpi: KPIMetrics) -> ProfileBundle:
    return ProfileBundle(
        run_id=run_id,
        timestamp=datetime.now(tz=timezone.utc),
        kpi=kpi,
        compute=ComputeMetrics(
            roofline_position="memory_bound",
            sm_occupancy_pct=42.0,
            memory_bw_utilization_pct=88.0,
        ),
    )


class SharedServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._store = ExperimentStore(Path(self._tmp.name))
        self._settings = AssistantSettings(
            filesystem_roots=(self._tmp.name,),
            pgoa_store_path=str(Path(self._tmp.name) / "experiments"),
        )
        self._kpi = KPIMetrics(
            primary_metric="elapsed_s",
            value=100.0,
            unit="seconds",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_collect_ncu_profile_preserves_existing_slurm_metrics(self):
        handle = self._store.create_run("wl", "baseline")

        with mock.patch(
            "hpc_assistant_backend.pgoa.services.SlurmAdapter.collect",
            return_value=_bundle_with_slurm("slurm-partial", self._kpi),
        ):
            collect_slurm_profile(
                self._settings,
                self._store,
                workload_id="wl",
                run_id=handle.run_id,
                job_id=12345,
                kpi=self._kpi,
            )

        with mock.patch(
            "hpc_assistant_backend.pgoa.services.NCUAdapter.collect",
            return_value=_bundle_with_compute("ncu-partial", self._kpi),
        ):
            merged = collect_ncu_profile(
                self._store,
                workload_id="wl",
                run_id=handle.run_id,
                ncu_csv_path="/tmp/fake.csv",
                kpi=self._kpi,
            )

        self.assertIsNotNone(merged.slurm)
        self.assertIsNotNone(merged.compute)
        self.assertEqual(merged.slurm.job_id, 12345)
        self.assertEqual(merged.compute.roofline_position, "memory_bound")


if __name__ == "__main__":
    unittest.main()
