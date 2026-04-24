"""Tests for cluster discovery helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from hpc_assistant_backend.pgoa.cluster_discovery import (
    _first_node_name,
    _parse_probe_output,
    _parse_scontrol_node,
    _parse_sinfo,
)


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


if __name__ == "__main__":
    unittest.main()
