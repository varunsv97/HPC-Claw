from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from hpc_assistant_backend.cli import main
from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.pgoa.schema import ClusterProfile, HardwareInfo, SoftwareEnvironment


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC = REPO_ROOT / "backend" / "src"


class BackendCliTests(unittest.TestCase):
    def test_show_opencode_tools_reports_python_backed_manifest(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BACKEND_SRC)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hpc_assistant_backend",
                "show-opencode-tools",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )

        payload = json.loads(result.stdout)
        self.assertIn("pgoa_apply_binding", payload["mutating_tools"])
        self.assertEqual(payload["tool_count"], 5)

    def test_sync_opencode_tools_writes_expected_files(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BACKEND_SRC)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hpc_assistant_backend",
                    "sync-opencode-tools",
                    "--root",
                    tmpdir,
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=env,
            )

            payload = json.loads(result.stdout)
            self.assertTrue(payload["written"])
            self.assertTrue((Path(tmpdir) / "opencode.json").exists())
            self.assertTrue((Path(tmpdir) / ".opencode" / "tools" / "pgoa.ts").exists())

    def test_run_pgoa_uses_openai_sdk_client_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = AssistantSettings(
                filesystem_roots=(tmpdir,),
                pgoa_store_path=str(Path(tmpdir) / "experiments"),
                openai_model="gpt-5",
            )
            result = SimpleNamespace(
                model_dump_json=lambda indent=2: json.dumps(
                    {"workload_id": "wl", "convergence_reason": "agent_no_tool_calls"},
                    indent=indent,
                )
            )
            script = Path(tmpdir) / "job.sh"
            script.write_text("#!/bin/bash\n#SBATCH --ntasks=4\n")

            with mock.patch("hpc_assistant_backend.cli.load_settings", return_value=settings), \
                 mock.patch("hpc_assistant_backend.cli.build_openai_client", return_value="client") as build_client, \
                 mock.patch("hpc_assistant_backend.cli.PGOAAgent") as agent_cls, \
                 mock.patch("sys.stdout.write") as stdout_write:
                agent = agent_cls.return_value
                agent.run.return_value = result

                exit_code = main(
                    [
                        "run-pgoa",
                        "--workload-id", "wl",
                        "--job-script-path", str(script),
                        "--primary-kpi", "elapsed_s",
                        "--kpi-unit", "seconds",
                        "--higher-is-better",
                    ]
                )

            self.assertEqual(exit_code, 0)
            build_client.assert_called_once_with(settings)
            agent_cls.assert_called_once()
            agent.run.assert_called_once_with(
                workload_id="wl",
                job_script_path=str(script),
                primary_kpi="elapsed_s",
                kpi_unit="seconds",
                lower_is_better=False,
            )
            self.assertTrue(stdout_write.called)

    def test_discover_cluster_prints_cluster_profile_json(self) -> None:
        profile = ClusterProfile(
            cluster_name="test-cluster",
            discovered_at=datetime.now(tz=timezone.utc),
            login_node_hardware=HardwareInfo(cpu_arch="x86_64"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = AssistantSettings(pgoa_store_path=str(Path(tmpdir) / "experiments"))
            with mock.patch("hpc_assistant_backend.cli.load_settings", return_value=settings), \
                 mock.patch("hpc_assistant_backend.cli.discover_cluster", return_value=profile), \
                 mock.patch("sys.stdout.write") as stdout_write:
                exit_code = main(["discover-cluster"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(stdout_write.called)

    def test_discover_env_prints_software_environment_json(self) -> None:
        env_profile = SoftwareEnvironment(module_system="none")
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = AssistantSettings(pgoa_store_path=str(Path(tmpdir) / "experiments"))
            with mock.patch("hpc_assistant_backend.cli.load_settings", return_value=settings), \
                 mock.patch("hpc_assistant_backend.cli.discover_software_env", return_value=env_profile), \
                 mock.patch("sys.stdout.write") as stdout_write:
                exit_code = main(["discover-env"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(stdout_write.called)

    def test_probe_cluster_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = AssistantSettings(
                pgoa_store_path=str(Path(tmpdir) / "experiments"),
                allow_cluster_probe_jobs=False,
            )
            with mock.patch("hpc_assistant_backend.cli.load_settings", return_value=settings), \
                 mock.patch("sys.stdout.write") as stdout_write:
                exit_code = main(["probe-cluster"])
        self.assertEqual(exit_code, 2)
        self.assertTrue(stdout_write.called)

    def test_doctor_prints_guardrail_report(self) -> None:
        report = {"ok": True, "warnings": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = AssistantSettings(pgoa_store_path=str(Path(tmpdir) / "experiments"))
            with mock.patch("hpc_assistant_backend.cli.load_settings", return_value=settings), \
                 mock.patch("hpc_assistant_backend.cli.build_doctor_report", return_value=report), \
                 mock.patch("sys.stdout.write") as stdout_write:
                exit_code = main(["doctor"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(stdout_write.called)


if __name__ == "__main__":
    unittest.main()
