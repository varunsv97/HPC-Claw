"""Tests for software environment discovery helpers."""

from __future__ import annotations

import unittest
from unittest import mock

from claw_backend.config import AssistantSettings
from claw_backend.cluster_probes.software import (
    ModuleEntry,
    _build_module_map,
    _classify,
    _classify_modules_with_llm,
    _derive_toolchains,
    _parse_llm_classification,
    _parse_module_list,
    discover_software_env,
)
from claw_backend.pgoa.schema import ModuleInfo


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

    def test_build_module_map_preserves_context_load_sequence(self):
        module_map = _build_module_map(
            [
                ModuleEntry(
                    name="openmpi",
                    version="4.1.6",
                    is_default=True,
                    context="arch-tier",
                    context_loads=("arch-tier",),
                    modulepath="/modules/arch-tier",
                    discovered_by="module_show",
                ),
            ]
        )
        self.assertEqual(
            module_map["openmpi"].load_sequence,
            ["module load arch-tier", "module load openmpi/4.1.6"],
        )
        self.assertEqual(module_map["openmpi"].source_contexts, ["arch-tier"])
        self.assertEqual(module_map["openmpi"].modulepaths, ["/modules/arch-tier"])

    def test_classify_and_derive_toolchains(self):
        module_map = _build_module_map(
            [
                ("gcc", "13.2.0", True),
                ("openmpi", "4.1.6", True),
                ("cuda", "12.4", True),
                ("hdf5", "1.14", True),
                ("ncu", "2024.1", True),
            ]
        )
        self.assertEqual(_classify("anything", "Description: compiler suite"), "other")
        self.assertEqual(_classify("gcc"), "other")

        toolchains = _derive_toolchains(
            [module_map["gcc"]],
            [module_map["openmpi"]],
            [module_map["cuda"]],
        )
        names = {toolchain.name for toolchain in toolchains}
        self.assertIn("gcc/13.2.0", names)
        self.assertIn("gcc/13.2.0+openmpi/4.1.6", names)
        self.assertIn("gcc/13.2.0+cuda/12.4", names)

    def test_parse_llm_classification_accepts_only_known_categories(self):
        parsed = _parse_llm_classification(
            '{"modules":{"a":"compiler","b":"mpi","c":"unsupported"}}'
        )
        self.assertEqual(parsed, {"a": "compiler", "b": "mpi"})

    def test_llm_classifier_uses_discovered_metadata_payload(self):
        class Message:
            content = '{"modules":{"opaque":"compiler"}}'

        class Choice:
            message = Message()

        class Completions:
            calls = []

            @classmethod
            def create(cls, **kwargs):
                cls.calls.append(kwargs)
                return type("Response", (), {"choices": [Choice()]})()

        class Chat:
            completions = Completions

        class Client:
            chat = Chat()

        module_map = {
            "opaque": ModuleInfo(
                name="opaque",
                versions=["1.0"],
                default_version="1.0",
                load_cmd="module load opaque/1.0",
                modulepaths=["/modules/compiler"],
                metadata="whatis text from module command",
            )
        }

        with mock.patch(
            "claw_backend.openai_client.build_openai_client",
            return_value=Client(),
        ):
            result = _classify_modules_with_llm(
                module_map,
                AssistantSettings(provider_api_key="test", provider_model="test-model"),
            )

        self.assertEqual(result, {"opaque": "compiler"})
        payload = Completions.calls[0]["messages"][1]["content"]
        self.assertIn("whatis text from module command", payload)
        self.assertIn("/modules/compiler", payload)


class EnvDiscoveryIntegrationShapeTests(unittest.TestCase):
    def test_discover_software_env_from_mocked_lmod_data(self):
        settings = AssistantSettings()

        def fake_run(subcmd, _settings, timeout=None):
            if subcmd == "avail":
                return "gcc/13.2.0(default)\npython/3.11.8\n"
            if subcmd == "spider":
                return "gcc/13.2.0(default)\nopenmpi/4.1.6(default)\ncuda/12.4(default)\n"
            if subcmd == "whatis gcc/13.2.0":
                return "Description: compiler suite"
            if subcmd == "whatis openmpi/4.1.6":
                return "Description: message passing interface"
            if subcmd == "whatis cuda/12.4":
                return "Description: GPU accelerator toolkit"
            if subcmd.startswith("show "):
                return ""
            return ""

        with mock.patch(
            "claw_backend.cluster_probes.software.detect_module_system",
            return_value=("lmod", "8.7.1"),
        ), mock.patch(
            "claw_backend.cluster_probes.software._run_bash_module",
            side_effect=fake_run,
        ), mock.patch(
            "claw_backend.cluster_probes.software._classify_modules_with_llm",
            return_value={"gcc": "compiler", "openmpi": "mpi", "cuda": "gpu_toolkit"},
        ):
            env = discover_software_env(settings)

        self.assertEqual(env.module_system, "lmod")
        self.assertTrue(env.spider_complete)
        self.assertEqual([m.name for m in env.compilers], ["gcc"])
        self.assertEqual([m.name for m in env.mpi_libraries], ["openmpi"])
        self.assertEqual([m.name for m in env.gpu_toolkits], ["cuda"])

    def test_discover_software_env_loads_modulepath_contexts_from_module_show(self):
        settings = AssistantSettings()

        def fake_run(subcmd, _settings, timeout=None):
            if subcmd == "avail":
                return "/modules/Core:\narch-tier\npython/3.11.8\n"
            if subcmd == "spider":
                return ""
            if subcmd == "show arch-tier":
                return "prepend_path(\"MODULEPATH\", \"/modules/arch-tier\")"
            if subcmd == "whatis gcc/13.2.0":
                return "Description: compiler suite"
            if subcmd == "whatis openmpi/4.1.6":
                return "Description: message passing interface"
            if subcmd == "show python/3.11.8":
                return ""
            return ""

        def fake_context(subcmd, _settings, *, load_sequence=(), timeout=None, include_modulepath=False):
            self.assertEqual(load_sequence, ("arch-tier",))
            return (
                "/modules/arch-tier:\ngcc/13.2.0(default)\nopenmpi/4.1.6(default)\n",
                ["/modules/Core", "/modules/arch-tier"],
            )

        with mock.patch.dict("os.environ", {"MODULEPATH": "/modules/Core"}), \
             mock.patch(
                 "claw_backend.cluster_probes.software.detect_module_system",
                 return_value=("lmod", "8.7.1"),
             ), mock.patch(
                 "claw_backend.cluster_probes.software._run_bash_module",
                 side_effect=fake_run,
             ), mock.patch(
                 "claw_backend.cluster_probes.software._run_bash_module_context",
                 side_effect=fake_context,
             ), mock.patch(
                 "claw_backend.cluster_probes.software._classify_modules_with_llm",
                 return_value={"gcc": "compiler", "openmpi": "mpi"},
             ):
            env = discover_software_env(settings)

        self.assertEqual([ctx.name for ctx in env.module_contexts], ["active", "arch-tier"])
        self.assertEqual([m.name for m in env.compilers], ["gcc"])
        self.assertEqual(env.compilers[0].load_sequence[0], "module load arch-tier")
        self.assertIn("arch-tier", env.compilers[0].source_contexts)

    def test_discover_software_env_walks_nested_toolchain_hierarchy(self):
        settings = AssistantSettings()

        def fake_run(subcmd, _settings, timeout=None):
            if subcmd == "avail":
                return "/modules/Core:\ntoolchain-tier\n"
            if subcmd == "spider":
                return ""
            if subcmd == "show toolchain-tier":
                return "prepend_path(\"MODULEPATH\", \"/modules/toolchain-tier\")"
            return ""

        def fake_context(subcmd, _settings, *, load_sequence=(), timeout=None, include_modulepath=False):
            if subcmd == "avail" and load_sequence == ("toolchain-tier",):
                return (
                    "/modules/toolchain-tier:\ngcc/13.2.0(default)\npython/3.11.8\n",
                    ["/modules/Core", "/modules/toolchain-tier"],
                )
            if subcmd == "show gcc/13.2.0" and load_sequence == ("toolchain-tier",):
                return (
                    "prepend_path(\"MODULEPATH\", \"/modules/toolchain-tier/gcc/13.2.0\")",
                    [],
                )
            if subcmd == "whatis gcc/13.2.0" and load_sequence == ("toolchain-tier",):
                return "Description: compiler suite", []
            if subcmd == "avail" and load_sequence == ("toolchain-tier", "gcc/13.2.0"):
                return (
                    "/modules/toolchain-tier/gcc/13.2.0:\nopenmpi/4.1.6(default)\n",
                    ["/modules/Core", "/modules/toolchain-tier", "/modules/toolchain-tier/gcc/13.2.0"],
                )
            if subcmd == "whatis openmpi/4.1.6" and load_sequence == ("toolchain-tier", "gcc/13.2.0"):
                return "Description: message passing interface", []
            return "", []

        with mock.patch.dict("os.environ", {"MODULEPATH": "/modules/Core"}), \
             mock.patch(
                 "claw_backend.cluster_probes.software.detect_module_system",
                 return_value=("lmod", "8.7.1"),
             ), mock.patch(
                 "claw_backend.cluster_probes.software._run_bash_module",
                 side_effect=fake_run,
             ), mock.patch(
                 "claw_backend.cluster_probes.software._run_bash_module_context",
                 side_effect=fake_context,
             ), mock.patch(
                 "claw_backend.cluster_probes.software._classify_modules_with_llm",
                 return_value={"gcc": "compiler", "openmpi": "mpi"},
             ):
            env = discover_software_env(settings)

        self.assertEqual(
            [ctx.name for ctx in env.module_contexts],
            ["active", "toolchain-tier", "toolchain-tier + gcc/13.2.0"],
        )
        self.assertEqual([m.name for m in env.compilers], ["gcc"])
        self.assertEqual([m.name for m in env.mpi_libraries], ["openmpi"])
        self.assertEqual(
            env.mpi_libraries[0].load_sequence,
            [
                "module load toolchain-tier",
                "module load gcc/13.2.0",
                "module load openmpi/4.1.6",
            ],
        )
        self.assertIn("gcc/13.2.0+openmpi/4.1.6", {tc.name for tc in env.toolchains})


if __name__ == "__main__":
    unittest.main()
