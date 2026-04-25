from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from claw_backend.assistant.service import build_assistant_session
from claw_backend.config import AssistantSettings


class AssistantSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._repo = self._root / "repo"
        self._repo.mkdir()
        (self._repo / ".git").mkdir()
        (self._repo / "README.md").write_text("# repo\n", encoding="utf-8")
        (self._repo / "Cargo.toml").write_text("[package]\nname = \"repo\"\n", encoding="utf-8")
        self._settings = AssistantSettings(
            filesystem_roots=(self._tmp.name,),
            pgoa_store_path=str(self._root / ".hpcassist"),
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_build_assistant_session_routes_repo_edits_to_opencode(self) -> None:
        session = build_assistant_session(
            self._settings,
            repo_root=str(self._repo),
            goal="add a slurm script",
            mode="coding",
        )

        route_map = {route.action: route for route in session.routes}
        self.assertEqual(route_map["edit_repo_files"].target, "opencode")
        self.assertEqual(route_map["submit_slurm_job"].target, "backend")
        self.assertTrue(route_map["submit_slurm_job"].approval.required)
        self.assertTrue(Path(session.session_path).exists())
        self.assertEqual(session.repo.name, "repo")
        self.assertIn("Cargo.toml", session.repo.detected_files)

    def test_build_assistant_session_can_sync_opencode_workspace(self) -> None:
        session = build_assistant_session(
            self._settings,
            repo_root=str(self._repo),
            mode="pgoa",
            sync_opencode=True,
        )

        self.assertTrue(session.opencode.ready)
        self.assertTrue((self._repo / "opencode.json").exists())
        self.assertTrue((self._repo / ".opencode" / "tools" / "pgoa.ts").exists())
        route_map = {route.action: route for route in session.routes}
        self.assertEqual(route_map["optimize_workload"].target, "pgoa_skill")
        self.assertEqual(route_map["optimize_workload"].delegates_to, ["opencode", "backend"])


if __name__ == "__main__":
    unittest.main()
