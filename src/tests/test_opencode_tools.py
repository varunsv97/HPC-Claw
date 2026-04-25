from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from claw_backend.config import AssistantSettings
from claw_backend.opencode_tools import execute_tool, sync_opencode_project, tool_manifest
from claw_backend.pgoa.schema import KPIMetrics, ProfileBundle, SlurmMetrics
from claw_backend.pgoa.store import ExperimentStore


def _make_bundle(run_id: str, elapsed: float) -> ProfileBundle:
    return ProfileBundle(
        run_id=run_id,
        timestamp=datetime.now(tz=timezone.utc),
        kpi=KPIMetrics(primary_metric="elapsed_s", value=elapsed, unit="seconds"),
        slurm=SlurmMetrics(job_id=12345, elapsed_s=elapsed, state="COMPLETED"),
    )


class OpenCodeToolManifestTests(unittest.TestCase):
    def test_manifest_has_five_tools(self):
        manifest = tool_manifest()
        self.assertEqual(manifest["tool_count"], 5)

    def test_manifest_lists_mutating_tool(self):
        manifest = tool_manifest()
        self.assertIn("pgoa_apply_binding", manifest["mutating_tools"])

    def test_manifest_tool_names(self):
        manifest = tool_manifest()
        names = {t["name"] for t in manifest["tools"]}
        self.assertEqual(names, {
            "pgoa_collect_profile",
            "pgoa_analyze",
            "pgoa_apply_binding",
            "pgoa_compare_runs",
            "pgoa_store_info",
        })


class OpenCodeToolExecuteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._store_path = Path(self._tmp.name) / "experiments"
        self._settings = AssistantSettings(
            command_timeout_seconds=4.0,
            filesystem_roots=(self._tmp.name,),
            pgoa_store_path=str(self._store_path),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _pre_seed_run(self, workload_id: str, elapsed: float) -> str:
        store = ExperimentStore(self._store_path)
        handle = store.create_run(workload_id, "baseline")
        store.save_bundle(workload_id, handle.run_id, _make_bundle(handle.run_id, elapsed))
        return handle.run_id

    def test_unknown_tool_returns_error(self):
        result = execute_tool("nonexistent_tool", {}, settings=self._settings)
        self.assertFalse(result["ok"])
        self.assertIn("Unknown tool", result["error"])

    def test_pgoa_store_info_empty_workload(self):
        result = execute_tool(
            "pgoa_store_info",
            {"workload_id": "empty_wl"},
            settings=self._settings,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["run_count"], 0)

    def test_pgoa_store_info_lists_seeded_run(self):
        run_id = self._pre_seed_run("wl_test", 600.0)
        result = execute_tool(
            "pgoa_store_info",
            {"workload_id": "wl_test"},
            settings=self._settings,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["run_count"], 1)
        self.assertEqual(result["runs"][0]["run_id"], run_id)

    def test_pgoa_analyze_returns_bottleneck(self):
        run_id = self._pre_seed_run("wl_analysis", 600.0)
        result = execute_tool(
            "pgoa_analyze",
            {"workload_id": "wl_analysis", "run_id": run_id},
            settings=self._settings,
        )
        self.assertTrue(result["ok"])
        self.assertIn("primary_bottleneck", result)

    def test_pgoa_compare_runs_computes_delta(self):
        store = ExperimentStore(self._store_path)
        baseline = store.create_run("wl_cmp", "baseline")
        iter1 = store.create_run("wl_cmp", "iteration", iteration=1)
        store.save_bundle("wl_cmp", baseline.run_id, _make_bundle(baseline.run_id, 100.0))
        store.save_bundle("wl_cmp", iter1.run_id, _make_bundle(iter1.run_id, 90.0))

        result = execute_tool(
            "pgoa_compare_runs",
            {
                "workload_id": "wl_cmp",
                "from_run_id": baseline.run_id,
                "to_run_id": iter1.run_id,
                "action_applied": "mem_bind=local",
            },
            settings=self._settings,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["kpi_direction"], "improved")
        self.assertAlmostEqual(result["kpi_delta_pct"], -10.0, places=1)

    def test_pgoa_apply_binding_rewrites_script(self):
        script = Path(self._tmp.name) / "job.sh"
        script.write_text("#!/bin/bash\n#SBATCH --ntasks-per-node=8\nsrun ./app\n")
        out = Path(self._tmp.name) / "job_v2.sh"
        run_id = self._pre_seed_run("wl_bind", 100.0)

        result = execute_tool(
            "pgoa_apply_binding",
            {
                "workload_id": "wl_bind",
                "run_id": run_id,
                "job_script_path": str(script),
                "output_script_path": str(out),
                "ntasks_per_node": 4,
                "mem_bind": "local",
            },
            settings=self._settings,
        )
        self.assertTrue(result["ok"])
        self.assertTrue(out.exists())
        content = out.read_text()
        self.assertIn("--ntasks-per-node=4", content)
        self.assertIn("--mem-bind=local", content)


class OpenCodeSyncTests(unittest.TestCase):
    def test_sync_writes_pgoa_ts_and_opencode_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            written = sync_opencode_project(root)
            paths = {p.relative_to(root).as_posix() for p in written}
            self.assertEqual(paths, {
                ".opencode/tools/_python.ts",
                ".opencode/tools/pgoa.ts",
                "opencode.json",
            })

    def test_pgoa_ts_contains_tool_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sync_opencode_project(Path(tmpdir))
            pgoa_ts = Path(tmpdir) / ".opencode" / "tools" / "pgoa.ts"
            content = pgoa_ts.read_text()
            self.assertIn('runPythonTool("pgoa_collect_profile"', content)
            self.assertIn('runPythonTool("pgoa_apply_binding"', content)

    def test_opencode_json_has_pgoa_binding_permission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sync_opencode_project(Path(tmpdir))
            cfg = json.loads((Path(tmpdir) / "opencode.json").read_text())
            self.assertIn("pgoa_apply_binding", cfg["permission"])


if __name__ == "__main__":
    unittest.main()
