from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
import unittest
from unittest import mock

from hpc_assistant_backend.config import AssistantSettings, ExecutionPolicy
from hpc_assistant_backend.filesystem import resolve_allowed_path
from hpc_assistant_backend.tools import (
    build_default_registry,
    build_interrupt_policy,
    build_tool_manifest,
    build_tools,
)


class ToolTests(unittest.TestCase):
    def test_tool_registry_respects_execution_policy(self) -> None:
        registry = build_default_registry()

        read_only_tools = [
            tool_.name
            for tool_ in build_tools(
                AssistantSettings(execution_policy=ExecutionPolicy.READ_ONLY),
                registry,
            )
        ]
        approval_tools = [
            tool_.name
            for tool_ in build_tools(
                AssistantSettings(execution_policy=ExecutionPolicy.APPROVAL_REQUIRED),
                registry,
            )
        ]

        self.assertEqual(
            read_only_tools,
            [
                "runtime_status",
                "slurm_queue",
                "slurm_job_details",
                "module_avail",
                "module_list",
                "filesystem_list",
                "filesystem_head",
            ],
        )
        self.assertEqual(
            approval_tools,
            read_only_tools + [
                "slurm_cancel_job",
                "module_load",
                "filesystem_remove",
            ],
        )

        self.assertEqual(
            build_interrupt_policy(AssistantSettings(), registry),
            {
                "edit_file": True,
                "execute": True,
                "write_file": True,
            },
        )
        self.assertEqual(
            build_interrupt_policy(
                AssistantSettings(execution_policy=ExecutionPolicy.APPROVAL_REQUIRED),
                registry,
            )["filesystem_remove"],
            {
                "allowed_decisions": ["approve", "edit", "reject"],
                "description": "Review a filesystem mutation before it runs.",
            },
        )

    def test_tool_manifest_reports_active_tools_and_interrupts(self) -> None:
        with unittest.mock.patch("pathlib.Path.cwd", return_value=Path.cwd()):
            settings = AssistantSettings(
                execution_policy=ExecutionPolicy.APPROVAL_REQUIRED,
                filesystem_roots=(str(Path.cwd()),),
                command_timeout_seconds=9.0,
            )

            manifest = build_tool_manifest(settings)

        self.assertEqual(manifest["active_tools"][-1], "filesystem_remove")
        self.assertEqual(manifest["filesystem_roots"], [str(Path.cwd().resolve())])
        self.assertEqual(manifest["command_timeout_seconds"], 9.0)
        self.assertEqual(
            manifest["interrupt_on"]["slurm_cancel_job"]["allowed_decisions"],
            ["approve", "edit", "reject"],
        )

    def test_filesystem_list_normalizes_command_output(self) -> None:
        calls: list[tuple[list[str], float]] = []
        root = Path.cwd()

        def fake_run(command, capture_output, text, check, timeout):
            calls.append((command, timeout))
            return CompletedProcess(command, 0, stdout="alpha\nbeta\n", stderr="")

        with mock.patch("hpc_assistant_backend.tools.subprocess.run", side_effect=fake_run):
            settings = AssistantSettings(
                execution_policy=ExecutionPolicy.READ_ONLY,
                filesystem_roots=(str(root),),
                command_timeout_seconds=7.5,
            )
            tool_map = {tool_.name: tool_ for tool_ in build_tools(settings)}

            result = tool_map["filesystem_list"].invoke({"path": str(root), "long": True})

        self.assertEqual(result["tool"], "filesystem_list")
        self.assertEqual(result["policy"], "read_only")
        self.assertTrue(result["ok"])
        self.assertEqual(result["arguments"]["path"], str(root.resolve()))
        self.assertEqual(result["command"], ["ls", "-lA", "--", str(root.resolve())])
        self.assertEqual(calls, [(["ls", "-lA", "--", str(root.resolve())], 7.5)])

    def test_filesystem_paths_are_confined_to_allowed_roots(self) -> None:
        root = Path.cwd()
        allowed_root = root / "backend"
        outside_root = root.parent

        settings = AssistantSettings(filesystem_roots=(str(allowed_root),))

        self.assertEqual(
            resolve_allowed_path(str(allowed_root / "job.out"), settings),
            (allowed_root / "job.out").resolve(strict=False),
        )
        with self.assertRaisesRegex(ValueError, "outside approved filesystem roots"):
            resolve_allowed_path(str(outside_root), settings)


if __name__ == "__main__":
    unittest.main()
