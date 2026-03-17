from __future__ import annotations

import json
from pathlib import Path
import threading
import unittest
import urllib.request

from hpc_assistant_backend.config import BackendConfig
from hpc_assistant_backend.runtime import AgentTurnResult, PendingInterrupt, TurnStatus
from hpc_assistant_backend.service import create_server


class FakeSession:
    def __init__(self, invoke_result: AgentTurnResult, resume_result: AgentTurnResult) -> None:
        self.invoke_result = invoke_result
        self.resume_result = resume_result
        self.prompts: list[str] = []
        self.resume_payloads: list[object] = []

    def invoke(self, prompt: str) -> AgentTurnResult:
        self.prompts.append(prompt)
        return self.invoke_result

    def resume(self, payload: object) -> AgentTurnResult:
        self.resume_payloads.append(payload)
        return self.resume_result


class FakeRuntime:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def start_session(self, thread_id: str, user_id: str | None = None) -> FakeSession:
        return self.session


class BackendServiceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fake_session = FakeSession(
            invoke_result=AgentTurnResult(
                thread_id="session-1",
                status=TurnStatus.INTERRUPTED,
                output_text="",
                interrupts=(
                    PendingInterrupt(
                        id="approval-1",
                        value={
                            "id": "approval-1",
                            "tool_call_id": "tool-1",
                            "title": "Review slurm_cancel_job",
                            "command": "scancel 12345",
                            "rationale": "Needs review",
                            "action_requests": [],
                            "review_configs": [],
                        },
                    ),
                ),
                raw_output={
                    "events": [
                        {
                            "type": "tool_call",
                            "tool_call": {
                                "id": "tool-1",
                                "name": "slurm_cancel_job",
                                "args": {"job_id": "12345"},
                                "status": "waiting_approval",
                                "command": "scancel 12345",
                            },
                        },
                        {
                            "type": "approval_requested",
                            "approval": {
                                "id": "approval-1",
                                "tool_call_id": "tool-1",
                                "title": "Review slurm_cancel_job",
                                "command": "scancel 12345",
                                "rationale": "Needs review",
                            },
                        },
                    ]
                },
            ),
            resume_result=AgentTurnResult(
                thread_id="session-1",
                status=TurnStatus.COMPLETED,
                output_text="Done",
                interrupts=(),
                raw_output={
                    "events": [
                        {
                            "type": "tool_result",
                            "tool_result": {
                                "tool_call_id": "tool-1",
                                "name": "slurm_cancel_job",
                                "status": "success",
                                "content": "$ scancel 12345 --signal TERM\ncancelled",
                                "command": "scancel 12345 --signal TERM",
                                "result": {"ok": True},
                            },
                        },
                        {
                            "type": "assistant_message",
                            "message": {
                                "role": "assistant",
                                "content": "Done",
                            },
                        },
                    ]
                },
            ),
        )
        try:
            cls.server = create_server(BackendConfig(port=0), runtime=FakeRuntime(fake_session))
        except PermissionError as error:
            raise unittest.SkipTest(f"socket binding is not available in this environment: {error}")
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        with urllib.request.urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 200)
            return json.load(response)

    def test_open_session_and_resume_existing_state(self) -> None:
        opened = self.request("POST", "/api/session", {"profile": "local"})
        session = opened["session"]
        self.assertFalse(opened["restored"])
        self.assertEqual(session["profile"], "local")
        self.assertGreaterEqual(len(session["messages"]), 1)

        resumed = self.request(
            "POST",
            "/api/session",
            {"profile": "local", "session_id": session["session_id"]},
        )
        self.assertTrue(resumed["restored"])
        self.assertEqual(resumed["session"]["session_id"], session["session_id"])

    def test_session_info_exposes_tool_manifest_and_filesystem_roots(self) -> None:
        payload = self.request("GET", "/session-info")

        self.assertTrue(payload["approval_required"])
        self.assertIn("slurm_queue", payload["tools"]["active_tools"])
        self.assertEqual(payload["filesystem"]["roots"], [str(Path("~").expanduser())])

    def test_prompt_and_approval_resolution_update_session_state(self) -> None:
        opened = self.request("POST", "/api/session", {"profile": "local"})
        session_id = opened["session"]["session_id"]

        pending = self.request(
            "POST",
            f"/api/session/{session_id}/prompt",
            {"prompt": "Cancel job 12345 for me"},
        )
        approval = pending["session"]["approvals"][-1]
        tool_event = pending["session"]["tool_events"][-1]
        self.assertEqual(approval["status"], "pending")
        self.assertEqual(tool_event["status"], "waiting_approval")
        self.assertEqual(approval["command"], "scancel 12345")

        approved = self.request(
            "POST",
            f"/api/session/{session_id}/approvals/{approval['id']}",
            {"decision": "edit", "edited_command": "scancel 12345 --signal TERM"},
        )

        updated_approval = approved["session"]["approvals"][-1]
        updated_tool = approved["session"]["tool_events"][-1]
        self.assertEqual(updated_approval["status"], "approved")
        self.assertEqual(updated_approval["decision"], "edit")
        self.assertEqual(updated_tool["status"], "completed")
        self.assertIn("--signal TERM", updated_tool["command"])


if __name__ == "__main__":
    unittest.main()
