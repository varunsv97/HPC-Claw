"""Tests for cluster discovery helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from claw_backend.cluster_probes.hardware_inventory import (
    _account_partition_relationship,
    _first_node_name,
    _parse_account_partition_relationship,
    _parse_probe_output,
    _parse_scontrol_node,
    _parse_sinfo,
    _resolve_probe_partitions,
)
from claw_backend.config import AssistantSettings


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ClusterDiscoveryHelperTests(unittest.TestCase):
    def test_parse_sinfo(self):
        raw = (
            "cpu up 24 18 cpu[001-024]\n"
            "gpu up 8 3 gpu[001-008]\n"
        )
        partitions = _parse_sinfo(raw)
        self.assertEqual([partition.name for partition in partitions], ["cpu", "gpu"])
        self.assertEqual(partitions[0].total_nodes, 24)
        self.assertEqual(partitions[1].idle_nodes, 3)

    def test_first_node_name_from_hostlist(self):
        self.assertEqual(_first_node_name("cpu[001-024]"), "cpu001")
        self.assertEqual(_first_node_name("gpu42"), "gpu42")
        self.assertIsNone(_first_node_name("(null)"))

    def test_parse_scontrol_node_extracts_cfg_tres(self):
        raw = (
            "NodeName=cpu001 Arch=x86_64 Sockets=2 CoresPerSocket=32 "
            "ThreadsPerCore=2 CfgTRES=cpu=128,mem=512G,billing=128,gres/gpu=4"
        )
        parsed = _parse_scontrol_node(raw)
        self.assertEqual(parsed["cpu_arch"], "x86_64")
        self.assertEqual(parsed["cpus_per_node"], 128)
        self.assertEqual(parsed["memory_gb_per_node"], 512.0)
        self.assertEqual(parsed["gpus_per_node"], 4)

    def test_parse_probe_output_merges_hwloc_smi_and_lscpu(self):
        hwloc_xml = (FIXTURES / "hwloc_output.xml").read_text(encoding="utf-8")
        text = (
            "=== hwloc ===\n"
            f"{hwloc_xml}\n"
            "=== nvidia-smi ===\n"
            "NVIDIA A100 80GB PCIe, 81920, 8.0, 108\n"
            "=== lscpu ===\n"
            "Socket(s): 2\n"
            "Core(s) per socket: 32\n"
            "Thread(s) per core: 2\n"
            "CPU(s): 128\n"
            "Model name: AMD EPYC\n"
            "NUMA node(s): 8\n"
        )
        hw = _parse_probe_output(text)
        self.assertEqual(hw.gpus_per_node, 1)
        self.assertEqual(hw.gpu_arch, "Ampere")
        self.assertEqual(hw.threads_per_core, 2)
        self.assertIsNotNone(hw.cpus_per_node)

    def test_parse_account_partition_relationship(self):
        raw = (
            "projA|cpu,gpu\n"
            "projA|debug\n"
            "projB|(null)\n"
            "projC|*\n"
        )
        relationships = _parse_account_partition_relationship(raw)
        self.assertEqual(relationships["projA"], {"cpu", "gpu", "debug"})
        self.assertIsNone(relationships["projB"])
        self.assertIsNone(relationships["projC"])

    @patch("claw_backend.cluster_probes.hardware_inventory.subprocess.run")
    def test_account_partition_relationship_query(self, mock_run):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "projA|cpu,gpu\nprojB|(null)\n"
        mock_run.return_value.stderr = ""
        settings = AssistantSettings(command_timeout_seconds=5.0)

        relationships = _account_partition_relationship(settings)

        self.assertEqual(relationships["projA"], {"cpu", "gpu"})
        self.assertIsNone(relationships["projB"])

    @patch.dict("os.environ", {"SLURM_ACCOUNT": "projA"}, clear=False)
    def test_resolve_probe_partitions_uses_active_account(self):
        result = _resolve_probe_partitions(
            requested=None,
            available={"cpu", "gpu", "debug"},
            relationships={"projA": {"cpu", "gpu"}, "projB": {"debug"}},
        )
        self.assertEqual(result, {"cpu", "gpu"})

    @patch.dict("os.environ", {}, clear=True)
    def test_resolve_probe_partitions_without_active_account_uses_union(self):
        result = _resolve_probe_partitions(
            requested={"cpu", "debug"},
            available={"cpu", "gpu", "debug"},
            relationships={"projA": {"cpu"}, "projB": {"debug"}},
        )
        self.assertEqual(result, {"cpu", "debug"})

    @patch.dict("os.environ", {"SLURM_ACCOUNT": "projA"}, clear=False)
    def test_resolve_probe_partitions_unrestricted_active_account(self):
        result = _resolve_probe_partitions(
            requested={"cpu", "gpu"},
            available={"cpu", "gpu", "debug"},
            relationships={"projA": None},
        )
        self.assertEqual(result, {"cpu", "gpu"})


if __name__ == "__main__":
    unittest.main()
