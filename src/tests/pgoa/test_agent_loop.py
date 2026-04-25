"""Tests for the PGOA agent loop (PGOAAgent) using mocked LLM client."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from unittest import mock

from claw_backend.pgoa.agent.loop import PGOAAgent
from claw_backend.pgoa.schema import KPIMetrics, ProfileBundle, SlurmMetrics
from claw_backend.pgoa.store import ExperimentStore


def _kpi(value: float = 630.0) -> KPIMetrics:
    return KPIMetrics(primary_metric="elapsed_s", value=value, unit="seconds")


def _make_bundle(run_id: str, kpi: KPIMetrics) -> ProfileBundle:
    return ProfileBundle(
        run_id=run_id,
        timestamp=datetime.now(tz=timezone.utc),
        kpi=kpi,
        slurm=SlurmMetrics(job_id=12345, elapsed_s=kpi.value, state="COMPLETED"),
    )


def _make_tool_call(call_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        ),
    )


def _make_response(tool_calls: list | None, content: str = "") -> SimpleNamespace:
    msg = SimpleNamespace(
        tool_calls=tool_calls,
        content=content,
    )
    msg.model_dump = lambda exclude_unset=False: {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tc.id,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (tool_calls or [])
        ],
    }
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class TestPGOAAgentNoToolCalls(unittest.TestCase):
    """Agent should stop immediately when LLM returns no tool calls."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._store = ExperimentStore(Path(self._tmp.name))
        self._client = MagicMock()
        self._agent = PGOAAgent(
            store=self._store,
            llm_client=self._client,
            max_iterations=5,
            kpi_threshold_pct=2.0,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_stops_on_no_tool_calls(self):
        self._client.chat.completions.create.return_value = _make_response(
            tool_calls=None, content="No changes needed."
        )
        # Need a dummy job script
        script = Path(self._tmp.name) / "job.sh"
        script.write_text("#!/bin/bash\n#SBATCH --ntasks=64\nsrun ./app\n")

        result = self._agent.run(
            workload_id="wl_test",
            job_script_path=str(script),
            primary_kpi="elapsed_s",
            kpi_unit="seconds",
        )
        self.assertEqual(result.convergence_reason, "agent_no_tool_calls")
        self.assertEqual(result.iterations_run, 1)

    def test_result_has_workload_id(self):
        self._client.chat.completions.create.return_value = _make_response(
            tool_calls=None
        )
        script = Path(self._tmp.name) / "job.sh"
        script.write_text("#!/bin/bash\n")

        result = self._agent.run(
            workload_id="my_workload",
            job_script_path=str(script),
            primary_kpi="elapsed_s",
            kpi_unit="s",
        )
        self.assertEqual(result.workload_id, "my_workload")


class TestPGOAAgentBelowThreshold(unittest.TestCase):
    """Agent should converge when compare_runs returns delta < threshold."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._base = Path(self._tmp.name)
        self._store = ExperimentStore(self._base)
        self._client = MagicMock()
        self._agent = PGOAAgent(
            store=self._store,
            llm_client=self._client,
            max_iterations=5,
            kpi_threshold_pct=2.0,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_converges_below_threshold(self):
        # Build compare_runs tool call args with fixed run IDs
        from_id = "from-run-fixed"
        to_id = "to-run-fixed"
        compare_args = {
            "workload_id": "wl",
            "from_run_id": from_id,
            "to_run_id": to_id,
            "action_applied": "mem_bind=local",
        }
        compare_tc = _make_tool_call("tc1", "compare_runs", compare_args)
        self._client.chat.completions.create.return_value = _make_response(
            tool_calls=[compare_tc]
        )

        script = self._base / "job.sh"
        script.write_text("#!/bin/bash\n#SBATCH --ntasks=64\n")

        # Patch dispatch so compare_runs returns a 0.5% delta (below 2% threshold)
        with mock.patch.object(
            PGOAAgent,
            "_dispatch",
            return_value={
                "from_run_id": from_id,
                "to_run_id": to_id,
                "kpi_delta_pct": -0.5,
                "kpi_direction": "neutral",
                "secondary_deltas": {},
            },
        ):
            result = self._agent.run(
                workload_id="wl",
                job_script_path=str(script),
                primary_kpi="elapsed_s",
                kpi_unit="seconds",
            )

        self.assertEqual(result.convergence_reason, "below_threshold")


class TestPGOAAgentUnknownTool(unittest.TestCase):
    """Agent should handle unknown tool calls gracefully."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._store = ExperimentStore(Path(self._tmp.name))
        self._client = MagicMock()
        self._agent = PGOAAgent(
            store=self._store,
            llm_client=self._client,
            max_iterations=2,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_unknown_tool_returns_error_in_message(self):
        bad_tc = _make_tool_call("tc_bad", "nonexistent_tool", {})
        self._client.chat.completions.create.side_effect = [
            _make_response(tool_calls=[bad_tc]),
            _make_response(tool_calls=None, content="Stopping."),
        ]
        script = Path(self._tmp.name) / "job.sh"
        script.write_text("#!/bin/bash\n")

        result = self._agent.run(
            workload_id="wl",
            job_script_path=str(script),
            primary_kpi="elapsed_s",
            kpi_unit="s",
        )
        # Should complete without exception
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
