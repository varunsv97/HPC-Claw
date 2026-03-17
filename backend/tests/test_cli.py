from __future__ import annotations

import json
import tempfile
from pathlib import Path
import os
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC = REPO_ROOT / "backend" / "src"
EXAMPLE_CONFIG = REPO_ROOT / "shared" / "config" / "hpc-assistant.example.toml"


class BackendCliTests(unittest.TestCase):
    def test_show_config_reports_example_values(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BACKEND_SRC)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hpc_assistant_backend",
                "--config",
                str(EXAMPLE_CONFIG),
                "show-config",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["backend"]["host"], "127.0.0.1")
        self.assertEqual(payload["model"]["provider"], "openai-compatible")

    def test_show_tools_reports_interrupt_metadata(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BACKEND_SRC)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hpc_assistant_backend",
                "--config",
                str(EXAMPLE_CONFIG),
                "show-tools",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )

        payload = json.loads(result.stdout)
        self.assertIn("filesystem_list", payload["active_tools"])
        self.assertEqual(
            payload["interrupt_on"]["filesystem_remove"]["allowed_decisions"],
            ["approve", "edit", "reject"],
        )

    def test_init_writes_user_config(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BACKEND_SRC)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hpc_assistant_backend",
                    "--config",
                    str(config_path),
                    "init",
                    "--force",
                    "--base-url",
                    "https://llm.example.invalid/v1",
                    "--model",
                    "cluster-model",
                    "--api-key",
                    "secret-token",
                    "--filesystem-root",
                    "~/scratch",
                    "--thread-store",
                    str(Path(tmpdir) / "threads"),
                    "--long-term-store",
                    str(Path(tmpdir) / "memory"),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=env,
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["config_path"], str(config_path))
            content = config_path.read_text(encoding="utf-8")
            self.assertIn('base_url = "https://llm.example.invalid/v1"', content)
            self.assertIn('api_key = "secret-token"', content)


if __name__ == "__main__":
    unittest.main()
