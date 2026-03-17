from __future__ import annotations

import io
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from hpc_assistant_backend.runtime import AgentTurnResult, PendingInterrupt, TurnStatus
from hpc_assistant_backend.stdio import StdioServer


class FakeSession:
    def __init__(self, invoke_result: AgentTurnResult, resume_result: AgentTurnResult) -> None:
        self.user_id = "alice"
        self._invoke_result = invoke_result
        self._resume_result = resume_result
        self.resume_payloads: list[object] = []
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> AgentTurnResult:
        self.prompts.append(prompt)
        return self._invoke_result

    def resume(self, payload: object) -> AgentTurnResult:
        self.resume_payloads.append(payload)
        return self._resume_result


class FakeRuntime:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.settings = type(
            "Settings",
            (),
            {
                "execution_policy": type("Policy", (), {"value": "read_only"})(),
                "assistant_id": "hpc-assistant",
            },
        )()

    def start_session(self, thread_id: str, user_id: str | None = None) -> FakeSession:
        self.session.user_id = user_id or "alice"
        return self.session


class StdioServerTests(unittest.TestCase):
    def test_stdio_protocol_emits_tool_activity_and_approvals(self) -> None:
        invoke_result = AgentTurnResult(
            thread_id="session-1",
            status=TurnStatus.INTERRUPTED,
            output_text="",
            interrupts=(
                PendingInterrupt(
                    id="approval-1",
                    value={
                        "action_requests": [
                            {
                                "name": "write_file",
                                "args": {"file_path": "/tmp/demo.txt", "content": "demo"},
                                "description": "Tool execution requires approval",
                            }
                        ],
                        "review_configs": [
                            {
                                "action_name": "write_file",
                                "allowed_decisions": ["approve", "edit", "reject"],
                            }
                        ],
                    },
                ),
            ),
            raw_output={
                "messages": [
                    HumanMessage(content="cancel job"),
                    AIMessage(content="", tool_calls=[{"id": "tool-1", "name": "write_file", "args": {"file_path": "/tmp/demo.txt", "content": "demo"}}]),
                ]
            },
        )
        resume_result = AgentTurnResult(
            thread_id="session-1",
            status=TurnStatus.COMPLETED,
            output_text="Done",
            interrupts=(),
            raw_output={
                "messages": [
                    HumanMessage(content="cancel job"),
                    AIMessage(content="", tool_calls=[{"id": "tool-1", "name": "write_file", "args": {"file_path": "/tmp/demo.txt", "content": "demo"}}]),
                    ToolMessage(content="wrote file", tool_call_id="tool-1", name="write_file"),
                    AIMessage(content="Done"),
                ]
            },
        )

        server = StdioServer(FakeRuntime(FakeSession(invoke_result, resume_result)))
        instream = io.StringIO(
            "\n".join(
                [
                    json.dumps({"id": "req-1", "command": "start_session", "session_id": "session-1", "user_id": "alice"}),
                    json.dumps({"id": "req-2", "command": "submit_prompt", "session_id": "session-1", "prompt": "cancel job"}),
                    json.dumps(
                        {
                            "id": "req-3",
                            "command": "resolve_approval",
                            "session_id": "session-1",
                            "approval_id": "approval-1",
                            "decisions": [{"type": "approve"}],
                        }
                    ),
                    "",
                ]
            )
        )
        outstream = io.StringIO()

        server.run(instream=instream, outstream=outstream)

        events = [json.loads(line) for line in outstream.getvalue().splitlines()]
        event_types = [event["type"] for event in events]
        self.assertEqual(
            event_types,
            [
                "ready",
                "session_started",
                "user_message",
                "tool_call",
                "approval_requested",
                "turn_completed",
                "tool_result",
                "assistant_message",
                "turn_completed",
            ],
        )
        self.assertEqual(events[4]["approval"]["id"], "approval-1")
        self.assertEqual(events[5]["status"], "interrupted")
        self.assertEqual(events[8]["status"], "completed")
        self.assertEqual(events[8]["output_text"], "Done")

    def test_resolve_approval_uses_interrupt_id_mapping(self) -> None:
        invoke_result = AgentTurnResult(
            thread_id="session-1",
            status=TurnStatus.INTERRUPTED,
            output_text="",
            interrupts=(PendingInterrupt(id="approval-1", value={"action_requests": [], "review_configs": []}),),
            raw_output={"messages": [HumanMessage(content="x")]},
        )
        fake_session = FakeSession(
            invoke_result=invoke_result,
            resume_result=AgentTurnResult(
                thread_id="session-1",
                status=TurnStatus.COMPLETED,
                output_text="ok",
                interrupts=(),
                raw_output={"messages": [HumanMessage(content="x"), AIMessage(content="ok")]},
            ),
        )
        server = StdioServer(FakeRuntime(fake_session))
        instream = io.StringIO(
            "\n".join(
                [
                    json.dumps({"command": "start_session", "session_id": "session-1"}),
                    json.dumps({"command": "submit_prompt", "session_id": "session-1", "prompt": "x"}),
                    json.dumps(
                        {
                            "command": "resolve_approval",
                            "session_id": "session-1",
                            "approval_id": "approval-1",
                            "decisions": [{"type": "reject", "message": "no"}],
                        }
                    ),
                    "",
                ]
            )
        )

        server.run(instream=instream, outstream=io.StringIO())

        self.assertEqual(
            fake_session.resume_payloads,
            [{"approval-1": {"decisions": [{"type": "reject", "message": "no"}]}}],
        )
