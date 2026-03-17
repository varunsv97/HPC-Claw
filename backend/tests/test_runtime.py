from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, ToolMessage

from hpc_assistant_backend.config import AssistantSettings, ExecutionPolicy
from hpc_assistant_backend.runtime import AssistantRuntime, TurnStatus


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[object]] = []
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


class RuntimeTests(unittest.TestCase):
    def test_runtime_executes_read_only_tool_and_returns_assistant_text(self) -> None:
        model = FakeModel(
            [
                AIMessage(content="", tool_calls=[{"id": "tool-1", "name": "runtime_status", "args": {}}]),
                AIMessage(content="ready"),
            ]
        )
        runtime = AssistantRuntime(settings=AssistantSettings(), model=model)

        result = runtime.start_session("thread-1", user_id="alice").invoke("hello")

        self.assertIs(result.status, TurnStatus.COMPLETED)
        self.assertEqual(result.output_text, "ready")
        self.assertEqual(model.bound_tool_names[0], "runtime_status")
        self.assertEqual(model.calls[0][-1].content, "hello")
        self.assertIsInstance(model.calls[1][-1], ToolMessage)
        self.assertEqual(model.calls[1][-1].name, "runtime_status")

    def test_runtime_pauses_for_approval_and_resumes_after_review(self) -> None:
        model = FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "tool-1", "name": "slurm_cancel_job", "args": {"job_id": "12345"}}],
                ),
                AIMessage(content="Cancelled"),
            ]
        )
        runtime = AssistantRuntime(
            settings=AssistantSettings(execution_policy=ExecutionPolicy.APPROVAL_REQUIRED),
            model=model,
        )
        runtime._invoke_tool = lambda tool_name, arguments: {  # type: ignore[method-assign]
            "tool": tool_name,
            "ok": True,
            "stdout": "cancelled",
            "stderr": "",
            "exit_code": 0,
            "command": ["scancel", str(arguments["job_id"])],
        }
        session = runtime.start_session("thread-2")

        interrupted = session.invoke("cancel job 12345")

        self.assertIs(interrupted.status, TurnStatus.INTERRUPTED)
        self.assertTrue(interrupted.interrupts)
        self.assertEqual(interrupted.interrupts[0].value["command"], "scancel 12345")

        completed = session.resume(
            {
                "approval_id": interrupted.interrupts[0].id,
                "decision": "approve",
            }
        )

        self.assertIs(completed.status, TurnStatus.COMPLETED)
        self.assertEqual(completed.output_text, "Cancelled")


if __name__ == "__main__":
    unittest.main()
