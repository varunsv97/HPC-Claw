"""Tests for the HardwareAdapter and its /sys, hwloc, and nvidia-smi collectors."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.pgoa.adapters.hardware import (
    HardwareAdapter,
    _collect_cache_info,
    _collect_cpu_model,
    _collect_cpu_topology,
    _collect_numa_info,
    _gpu_arch_from_cc,
    _parse_cache_size_kb,
    _parse_cpu_range,
    _parse_hwloc_xml,
    _run_nvidia_smi,
    collect_hardware_info,
)
from hpc_assistant_backend.pgoa.schema import KPIMetrics

FIXTURES = Path(__file__).parent / "fixtures"


def _settings(**kwargs) -> AssistantSettings:
    return AssistantSettings(command_timeout_seconds=5.0, **kwargs)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Unit: helpers
# ---------------------------------------------------------------------------


class TestParseCpuRange(unittest.TestCase):
    def test_simple_range(self):
        self.assertEqual(_parse_cpu_range("0-7"), 8)

    def test_single(self):
        self.assertEqual(_parse_cpu_range("0"), 1)

    def test_mixed(self):
        self.assertEqual(_parse_cpu_range("0,2-5,8"), 6)

    def test_empty(self):
        self.assertIsNone(_parse_cpu_range(""))


class TestParseCacheSizeKb(unittest.TestCase):
    def test_kilobytes(self):
        self.assertEqual(_parse_cache_size_kb("32K"), 32)

    def test_megabytes(self):
        self.assertEqual(_parse_cache_size_kb("24M"), 24 * 1024)

    def test_plain_int(self):
        self.assertEqual(_parse_cache_size_kb("32768"), 32768)

    def test_invalid(self):
        self.assertIsNone(_parse_cache_size_kb("bogus"))


class TestGpuArchFromCC(unittest.TestCase):
    def test_ampere(self):
        self.assertEqual(_gpu_arch_from_cc("8.0"), "Ampere")

    def test_hopper(self):
        self.assertEqual(_gpu_arch_from_cc("9.0"), "Hopper")

    def test_unknown(self):
        self.assertIsNone(_gpu_arch_from_cc("99.9"))


# ---------------------------------------------------------------------------
# Unit: /sys readers
# ---------------------------------------------------------------------------


class TestCollectCpuTopology(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _sysfs(self) -> Path:
        return self._root / "sys"

    def _write_cpu(self, cpu_idx: int, socket_id: int, core_id: int) -> None:
        topo = self._sysfs() / "devices" / "system" / "cpu" / f"cpu{cpu_idx}" / "topology"
        _write(topo / "physical_package_id", str(socket_id))
        _write(topo / "core_id", str(core_id))

    def test_two_socket_two_core_smt2(self):
        # 2 sockets × 2 cores × 2 threads = 8 logical CPUs
        _write(self._sysfs() / "devices" / "system" / "cpu" / "possible", "0-7")
        for cpu in range(8):
            socket = cpu // 4
            core = (cpu % 4) // 2
            self._write_cpu(cpu, socket, core)

        result = _collect_cpu_topology(self._sysfs())

        self.assertEqual(result["cpus_per_node"], 8)
        self.assertEqual(result["sockets_per_node"], 2)
        self.assertEqual(result["cores_per_socket"], 2)
        self.assertEqual(result["threads_per_core"], 2)

    def test_no_sysfs_returns_nones(self):
        result = _collect_cpu_topology(self._sysfs())  # empty dir
        self.assertIsNone(result["sockets_per_node"])
        self.assertIsNone(result["cpus_per_node"])

    def test_possible_file_used_for_total_cpus(self):
        _write(self._sysfs() / "devices" / "system" / "cpu" / "possible", "0-31")
        result = _collect_cpu_topology(self._sysfs())
        self.assertEqual(result["cpus_per_node"], 32)


class TestCollectCacheInfo(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _cache_dir(self) -> Path:
        return (
            self._root
            / "sys"
            / "devices"
            / "system"
            / "cpu"
            / "cpu0"
            / "cache"
        )

    def _write_index(self, idx: int, level: str, cache_type: str, size: str) -> None:
        d = self._cache_dir() / f"index{idx}"
        _write(d / "level", level)
        _write(d / "type", cache_type)
        _write(d / "size", size)

    def test_standard_three_level_cache(self):
        self._write_index(0, "1", "Data", "32K")
        self._write_index(1, "1", "Instruction", "32K")
        self._write_index(2, "2", "Unified", "256K")
        self._write_index(3, "3", "Unified", "8192K")

        result = _collect_cache_info(self._root / "sys")

        self.assertEqual(result["cache_l1d_kb"], 32)
        self.assertEqual(result["cache_l2_kb"], 256)
        self.assertAlmostEqual(result["cache_l3_mb"], 8.0, places=1)

    def test_missing_cache_dir_returns_nones(self):
        result = _collect_cache_info(self._root / "sys")
        self.assertIsNone(result["cache_l1d_kb"])
        self.assertIsNone(result["cache_l2_kb"])
        self.assertIsNone(result["cache_l3_mb"])


class TestCollectNumaInfo(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _node_dir(self) -> Path:
        return self._root / "sys" / "devices" / "system" / "node"

    def test_two_numa_nodes(self):
        for i in range(2):
            meminfo = (
                f"Node {i} MemTotal:       65536000 kB\n"
                f"Node {i} MemFree:        32000000 kB\n"
            )
            _write(self._node_dir() / f"node{i}" / "meminfo", meminfo)

        result = _collect_numa_info(self._root / "sys")

        self.assertEqual(result["numa_nodes"], 2)
        # 2 × 65536000 KiB ÷ 1024² ≈ 125 GiB
        self.assertGreater(result["memory_gb_per_node"], 100)

    def test_no_node_dir_returns_none(self):
        result = _collect_numa_info(self._root / "sys")
        self.assertIsNone(result["numa_nodes"])


class TestCollectCpuModel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_x86_cpuinfo(self):
        cpuinfo = (
            "processor\t: 0\n"
            "model name\t: Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz\n"
            "flags\t\t: fpu avx2 avx512f fma sse4_1 sse4_2 aes\n"
        )
        _write(self._root / "proc" / "cpuinfo", cpuinfo)
        result = _collect_cpu_model(self._root / "proc")

        self.assertEqual(result["cpu_model"], "Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz")
        self.assertIn("avx2", result["cpu_features"])
        self.assertIn("avx512f", result["cpu_features"])
        self.assertIn("fma", result["cpu_features"])
        self.assertNotIn("fpu", result["cpu_features"])  # not in HPC_FLAGS

    def test_aarch64_cpuinfo(self):
        cpuinfo = (
            "processor\t: 0\n"
            "CPU implementer\t: 0x41\n"
            "Features\t: fp asimd evtstrm aes sve\n"
        )
        _write(self._root / "proc" / "cpuinfo", cpuinfo)
        result = _collect_cpu_model(self._root / "proc")

        self.assertIn("sve", result["cpu_features"])

    def test_missing_proc_returns_empty(self):
        result = _collect_cpu_model(self._root / "proc")
        self.assertIsNone(result.get("cpu_model"))


# ---------------------------------------------------------------------------
# Unit: hwloc XML parser
# ---------------------------------------------------------------------------


class TestParseHwlocXml(unittest.TestCase):
    def _xml(self) -> str:
        return (FIXTURES / "hwloc_output.xml").read_text()

    def test_package_count(self):
        result = _parse_hwloc_xml(self._xml())
        self.assertEqual(result["sockets_per_node"], 2)

    def test_numa_nodes(self):
        result = _parse_hwloc_xml(self._xml())
        self.assertEqual(result["numa_nodes"], 2)

    def test_total_pus(self):
        # 2 packages × 2 cores × 2 threads = 8 PUs
        result = _parse_hwloc_xml(self._xml())
        self.assertEqual(result["cpus_per_node"], 8)

    def test_cores_per_socket(self):
        result = _parse_hwloc_xml(self._xml())
        self.assertEqual(result["cores_per_socket"], 2)

    def test_threads_per_core(self):
        result = _parse_hwloc_xml(self._xml())
        self.assertEqual(result["threads_per_core"], 2)

    def test_l3_cache_mb(self):
        # 25165824 bytes = 24 MiB
        result = _parse_hwloc_xml(self._xml())
        self.assertAlmostEqual(result["cache_l3_mb"], 24.0, places=0)

    def test_l2_cache_kb(self):
        # 1048576 bytes = 1024 KiB
        result = _parse_hwloc_xml(self._xml())
        self.assertEqual(result["cache_l2_kb"], 1024)

    def test_l1_cache_kb(self):
        # 32768 bytes = 32 KiB
        result = _parse_hwloc_xml(self._xml())
        self.assertEqual(result["cache_l1d_kb"], 32)

    def test_malformed_xml_returns_empty(self):
        result = _parse_hwloc_xml("<not valid xml<<")
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Unit: nvidia-smi parser
# ---------------------------------------------------------------------------


class TestRunNvidiaSmi(unittest.TestCase):
    def _make_result(self) -> subprocess.CompletedProcess:
        content = (FIXTURES / "nvidia_smi_output.txt").read_text()
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=content, stderr="")

    def test_parses_two_gpus(self):
        s = _settings()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._make_result()
            gpus = _run_nvidia_smi(s)

        self.assertIsNotNone(gpus)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["name"], "NVIDIA A100 80GB PCIe")
        self.assertAlmostEqual(gpus[0]["memory_gb"], 80.0, delta=1.0)
        self.assertEqual(gpus[0]["compute_capability"], "8.0")
        self.assertEqual(gpus[0]["sm_count"], 108)

    def test_nvidia_smi_not_found_returns_none(self):
        s = _settings()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _run_nvidia_smi(s)
        self.assertIsNone(result)

    def test_nonzero_exit_returns_none(self):
        s = _settings()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=6, stdout="", stderr="No devices found"
            )
            result = _run_nvidia_smi(s)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Integration: collect_hardware_info and HardwareAdapter
# ---------------------------------------------------------------------------


class TestCollectHardwareInfoIntegration(unittest.TestCase):
    """Full collect_hardware_info with fake /sys, fake hwloc, and mocked nvidia-smi."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build_sysfs(self):
        """Build a minimal fake /sys tree: 2 sockets × 2 cores × 2 threads = 8 CPUs."""
        sys = self._root / "sys"
        cpu_base = sys / "devices" / "system" / "cpu"
        _write(cpu_base / "possible", "0-7")

        for cpu_idx in range(8):
            socket = cpu_idx // 4
            core = (cpu_idx % 4) // 2
            topo = cpu_base / f"cpu{cpu_idx}" / "topology"
            _write(topo / "physical_package_id", str(socket))
            _write(topo / "core_id", str(core))

        cache_base = cpu_base / "cpu0" / "cache"
        for idx, (level, ctype, size) in enumerate([
            ("1", "Data",        "32K"),
            ("1", "Instruction", "32K"),
            ("2", "Unified",     "256K"),
            ("3", "Unified",     "8192K"),
        ]):
            d = cache_base / f"index{idx}"
            _write(d / "level", level)
            _write(d / "type", ctype)
            _write(d / "size", size)

        node_base = sys / "devices" / "system" / "node"
        for i in range(2):
            _write(
                node_base / f"node{i}" / "meminfo",
                f"Node {i} MemTotal:       65536000 kB\n",
            )

        return sys

    def _build_proc(self):
        proc = self._root / "proc"
        cpuinfo = (
            "model name\t: Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz\n"
            "flags\t\t: avx2 avx512f fma\n"
        )
        _write(proc / "cpuinfo", cpuinfo)
        return proc

    def test_full_collect_no_gpu(self):
        sysfs = self._build_sysfs()
        proc = self._build_proc()
        s = _settings()

        with patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_hwloc",
            return_value=None,
        ), patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_nvidia_smi",
            return_value=None,
        ):
            hw = collect_hardware_info(s, sysfs_root=sysfs, proc_root=proc)

        self.assertEqual(hw.sockets_per_node, 2)
        self.assertEqual(hw.cores_per_socket, 2)
        self.assertEqual(hw.threads_per_core, 2)
        self.assertEqual(hw.cpus_per_node, 8)
        self.assertEqual(hw.numa_nodes, 2)
        self.assertEqual(hw.cache_l1d_kb, 32)
        self.assertEqual(hw.cache_l2_kb, 256)
        self.assertAlmostEqual(hw.cache_l3_mb, 8.0, places=0)
        self.assertIn("Intel", hw.cpu_model)
        self.assertIn("avx2", hw.cpu_features)
        self.assertIsNone(hw.gpus_per_node)

    def test_gpu_fields_populated_from_nvidia_smi(self):
        sysfs = self._build_sysfs()
        proc = self._build_proc()
        s = _settings()

        fake_gpus = [
            {"name": "NVIDIA A100 80GB PCIe", "memory_gb": 80.0,
             "compute_capability": "8.0", "sm_count": 108},
            {"name": "NVIDIA A100 80GB PCIe", "memory_gb": 80.0,
             "compute_capability": "8.0", "sm_count": 108},
        ]
        with patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_hwloc",
            return_value=None,
        ), patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_nvidia_smi",
            return_value=fake_gpus,
        ):
            hw = collect_hardware_info(s, sysfs_root=sysfs, proc_root=proc)

        self.assertEqual(hw.gpus_per_node, 2)
        self.assertEqual(hw.gpu_model, "NVIDIA A100 80GB PCIe")
        self.assertEqual(hw.gpu_arch, "Ampere")
        self.assertEqual(hw.gpu_compute_capability, "8.0")
        self.assertEqual(hw.gpu_sm_count, 108)

    def test_hwloc_fills_gaps_when_sys_unavailable(self):
        """hwloc provides topology when no /sys tree exists."""
        empty_sys = self._root / "sys"  # non-existent
        proc = self._build_proc()
        s = _settings()
        hwloc_xml = (FIXTURES / "hwloc_output.xml").read_text()

        with patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_hwloc",
            return_value=hwloc_xml,
        ), patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_nvidia_smi",
            return_value=None,
        ):
            hw = collect_hardware_info(s, sysfs_root=empty_sys, proc_root=proc)

        self.assertEqual(hw.sockets_per_node, 2)
        self.assertEqual(hw.cpus_per_node, 8)
        self.assertAlmostEqual(hw.cache_l3_mb, 24.0, places=0)

    def test_sysfs_takes_precedence_over_hwloc(self):
        """/sys values are kept when hwloc would give different answers."""
        sysfs = self._build_sysfs()
        proc = self._build_proc()
        s = _settings()
        hwloc_xml = (FIXTURES / "hwloc_output.xml").read_text()

        with patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_hwloc",
            return_value=hwloc_xml,
        ), patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_nvidia_smi",
            return_value=None,
        ):
            hw = collect_hardware_info(s, sysfs_root=sysfs, proc_root=proc)

        # /sys says 8 CPUs (matches hwloc too here), sockets=2, cores_per_socket=2
        self.assertEqual(hw.cpus_per_node, 8)
        self.assertEqual(hw.sockets_per_node, 2)


class TestHardwareAdapter(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_adapter_returns_bundle_with_hardware(self):
        s = _settings()
        adapter = HardwareAdapter(settings=s)
        kpi = KPIMetrics(primary_metric="elapsed_s", value=100.0, unit="seconds")

        with patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_hwloc",
            return_value=None,
        ), patch(
            "hpc_assistant_backend.pgoa.adapters.hardware._run_nvidia_smi",
            return_value=None,
        ):
            bundle = adapter.collect(
                kpi,
                sysfs_root=self._root / "sys",
                proc_root=self._root / "proc",
            )

        self.assertEqual(bundle.kpi.primary_metric, "elapsed_s")
        self.assertIsNotNone(bundle.hardware)
        self.assertIsNotNone(bundle.hardware.cpu_arch)  # platform.machine() always works

    def test_adapter_name(self):
        self.assertEqual(HardwareAdapter().name(), "hardware")
