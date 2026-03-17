from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from hpc_assistant_tui import (
    FocusPane,
    TuiConfig,
    PersistedState,
    SessionState,
    default_state_path,
    help_text,
    load_snapshot,
    parse_args,
    save_snapshot,
)
from hpc_assistant_tui.cli import main


class TuiModelTests(unittest.TestCase):
    def test_help_text_mentions_backend_url(self) -> None:
        self.assertIn("--backend-url", help_text())

    def test_parse_args_accepts_overrides(self) -> None:
        parsed = parse_args(
            [
                "--backend-url",
                "http://127.0.0.1:9000",
                "--profile",
                "smoke",
                "--state-path",
                "/tmp/demo-state.json",
            ]
        )
        self.assertEqual(
            parsed,
            TuiConfig(
                backend_url="http://127.0.0.1:9000",
                profile="smoke",
                state_path="/tmp/demo-state.json",
            ),
        )

    def test_default_state_path_points_to_a_json_snapshot(self) -> None:
        self.assertEqual(default_state_path().suffix, ".json")

    def test_snapshot_round_trip_preserves_session_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tui-state.json"
            snapshot = PersistedState(
                config=TuiConfig(
                    backend_url="http://127.0.0.1:8765",
                    profile="smoke",
                    state_path=str(path),
                ),
                session=SessionState(
                    session_id="thread-1",
                    profile="smoke",
                ),
                focus=FocusPane.APPROVALS,
                notice="restored",
            )

            save_snapshot(path, snapshot)
            restored = load_snapshot(path)

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.session.session_id, "thread-1")
            self.assertEqual(restored.focus, FocusPane.APPROVALS)
            self.assertEqual(Path(restored.config.state_path), path)

    def test_tui_config_uses_environment_defaults(self) -> None:
        old_backend_url = os.environ.get("HPC_ASSISTANT_BACKEND_URL")
        old_profile = os.environ.get("HPC_ASSISTANT_PROFILE")
        try:
            os.environ["HPC_ASSISTANT_BACKEND_URL"] = "http://127.0.0.1:9999"
            os.environ["HPC_ASSISTANT_PROFILE"] = "env-profile"
            config = TuiConfig.with_defaults()
        finally:
            if old_backend_url is None:
                os.environ.pop("HPC_ASSISTANT_BACKEND_URL", None)
            else:
                os.environ["HPC_ASSISTANT_BACKEND_URL"] = old_backend_url
            if old_profile is None:
                os.environ.pop("HPC_ASSISTANT_PROFILE", None)
            else:
                os.environ["HPC_ASSISTANT_PROFILE"] = old_profile

        self.assertEqual(config.backend_url, "http://127.0.0.1:9999")
        self.assertEqual(config.profile, "env-profile")

    def test_project_targets_python_314(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.14,<3.15"', content)


class TuiCliTests(unittest.TestCase):
    def test_main_runs_textual_entrypoint_when_available(self) -> None:
        calls: list[TuiConfig] = []
        fake_module = types.SimpleNamespace(run_textual_app=lambda config: calls.append(config))

        with mock.patch.dict(
            "sys.modules",
            {"hpc_assistant_tui.textual_app": fake_module},
        ):
            exit_code = main(
                [
                    "--backend-url",
                    "http://127.0.0.1:8765",
                    "--profile",
                    "test-profile",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].profile, "test-profile")

    def test_main_reports_missing_textual_dependency(self) -> None:
        stdout = io.StringIO()
        with mock.patch.dict("sys.modules", {"hpc_assistant_tui.textual_app": None}):
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--backend-url", "http://127.0.0.1:8765"])

        self.assertEqual(exit_code, 1)
        self.assertIn("requires the `textual` package", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
