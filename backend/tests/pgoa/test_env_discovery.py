"""Tests for software environment discovery helpers."""

from __future__ import annotations

import unittest
from unittest import mock

from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.pgoa.env_discovery import (
    _build_module_map,
    _classify,
    _derive_toolchains,
    _parse_module_list,
    discover_software_env,
)


class EnvDiscoveryHelperTests(unittest.TestCase):
    def test_parse_module_list_skips_headers_and_annotations(self):
        raw = """
/apps/modulefiles/Core:
gcc/13.2.0(default)
openmpi/4.1.6 (D)
python/3.11.8
--------------------------------
For detailed information use spider
"""
        parsed = _parse_module_list(raw)
        self.assertEqual(
            parsed,
            [
                ("gcc", "13.2.0", True),
                ("openmpi", "4.1.6", True),
                ("python", "3.11.8", False),
            ],
        )

    def test_build_module_map_prefers_default_then_newest(self):
        module_map = _build_module_map(
            [
                ("gcc", "12.3.0", False),
                ("gcc", "13.2.0", True),
                ("openmpi", "4.1.5", False),
                ("openmpi", "4.1.6", False),
            ]
        )
        self.assertEqual(module_map["gcc"].default_version, "13.2.0")
        self.assertEqual(module_map["gcc"].load_cmd, "module load gcc/13.2.0")
        self.assertEqual(module_map["openmpi"].default_version, "4.1.6")

    def test_classify_and_derive_toolchains(self):
        module_map = _build_module_map(
            [
                ("gcc", "13.2.0", True),
                ("openmpi", "4.1.6", True),
                ("cuda", "12.4", True),
                ("hdf5", "1.14", True),
            ]
        )
        self.assertEqual(_classify("gcc"), "compiler")
        self.assertEqual(_classify("openmpi"), "mpi")
        self.assertEqual(_classify("cuda"), "gpu_toolkit")
        self.assertEqual(_classify("hdf5"), "io_library")

        toolchains = _derive_toolchains(
            [module_map["gcc"]],
            [module_map["openmpi"]],
            [module_map["cuda"]],
        )
        names = {toolchain.name for toolchain in toolchains}
        self.assertIn("gcc/13.2.0", names)
        self.assertIn("gcc/13.2.0+openmpi/4.1.6", names)
        self.assertIn("gcc/13.2.0+cuda/12.4", names)


class EnvDiscoveryIntegrationShapeTests(unittest.TestCase):
    def test_discover_software_env_from_mocked_lmod_data(self):
        settings = AssistantSettings()
        with mock.patch(
            "hpc_assistant_backend.pgoa.env_discovery.detect_module_system",
            return_value=("lmod", "8.7.1"),
        ), mock.patch(
            "hpc_assistant_backend.pgoa.env_discovery._run_bash_module",
            side_effect=[
                "gcc/13.2.0(default)\npython/3.11.8\n",
                "gcc/13.2.0(default)\nopenmpi/4.1.6(default)\ncuda/12.4(default)\n",
            ],
        ):
            env = discover_software_env(settings)

        self.assertEqual(env.module_system, "lmod")
        self.assertTrue(env.spider_complete)
        self.assertEqual([m.name for m in env.compilers], ["gcc"])
        self.assertEqual([m.name for m in env.mpi_libraries], ["openmpi"])
        self.assertEqual([m.name for m in env.gpu_toolkits], ["cuda"])


if __name__ == "__main__":
    unittest.main()
