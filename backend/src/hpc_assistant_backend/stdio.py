"""JSONL stdio bridge for the Textual TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from typing import Any, TextIO
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.runtime import AssistantRuntime, AssistantSession, PendingInterrupt


PROTOCOL_VERSION = "2026-03-17"


@dataclass(slots=True)
class SessionState:
    """Per-session state needed by the stdio bridge."""

    session_id: str
    session: AssistantSession
    emitted_message_count: int = 0
    pending_approvals: dict[str, PendingInterrupt] = field(default_factory=dict)


class StdioServer:
    """Serve the backend over line-delimited JSON on stdin/stdout."""

    def __init__(self, runtime: AssistantRuntime) -> None:
        self.runtime = runtime
        self.sessions: dict[str, SessionState] = {}
        self._sequence = 0

    def run(
        self,
        instream: TextIO | None = None,
        outstream: TextIO | None = None,
    ) -> int:
        source = instream or sys.stdin
        sink = outstream or sys.stdout
        self._emit(
            sink,
            {
                "type": "ready",
                "protocol_version": PROTOCOL_VERSION,
                "transport": "jsonl-stdio",
                "commands": [
                    "start_session",
                    "submit_prompt",
                    "resolve_approval",
                ],
                "events": [
                    "ready",
                    "session_started",
                    "user_message",
                    "assistant_message",
                    "tool_call",
                    "tool_result",
                    "approval_requested",
                    "turn_completed",
                    "error",
                ],
            },
        )

        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            request_id = uuid4().hex
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("command must be a JSON object")
                request_id = str(payload.get("request_id") or payload.get("id") or request_id)
                self._handle_command(payload, request_id, sink)
            except Exception as error:  # noqa: BLE001 - protocol must remain alive on bad input
                self._emit_error(sink, request_id, None, "invalid_command", str(error))

        return 0

    def _handle_command(
        self,
        payload: dict[str, Any],
        request_id: str,
        sink: TextIO,
    ) -> None:
        command = str(payload.get("command", "")).strip()
        if command == "start_session":
            self._start_session(payload, request_id, sink)
            return
        if command == "submit_prompt":
            self._submit_prompt(payload, request_id, sink)
            return
        if command == "resolve_approval":
            self._resolve_approval(payload, request_id, sink)
            return
        self._emit_error(sink, request_id, None, "unknown_command", f"unsupported command: {command or '<missing>'}")

    def _start_session(
        self,
        payload: dict[str, Any],
        request_id: str,
        sink: TextIO,
    ) -> None:
        session_id = str(payload.get("session_id") or uuid4().hex)
        user_id = payload.get("user_id")
        if user_id is not None and not isinstance(user_id, str):
            self._emit_error(sink, request_id, session_id, "invalid_user_id", "user_id must be a string")
            return

        restored = session_id in self.sessions
        if not restored:
            session = self.runtime.start_session(session_id, user_id=user_id)
            self.sessions[session_id] = SessionState(session_id=session_id, session=session)

        self._emit(
            sink,
            {
                "type": "session_started",
                "request_id": request_id,
                "session_id": session_id,
                "restored": restored,
                "execution_policy": self.runtime.settings.execution_policy.value,
                "assistant_id": self.runtime.settings.assistant_id,
                "user_id": self.sessions[session_id].session.user_id,
            },
        )

    def _submit_prompt(
        self,
        payload: dict[str, Any],
        request_id: str,
        sink: TextIO,
    ) -> None:
        session_state = self._require_session(payload, request_id, sink)
        if session_state is None:
            return

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._emit_error(sink, request_id, session_state.session_id, "invalid_prompt", "prompt must be a non-empty string")
            return

        self._emit(
            sink,
            {
                "type": "user_message",
                "request_id": request_id,
                "session_id": session_state.session_id,
                "message": {
                    "role": "user",
                    "content": prompt,
                },
            },
        )
        result = session_state.session.invoke(prompt)
        self._emit_turn_events(session_state, request_id, sink, result.raw_output, result.output_text, result.interrupts)

    def _resolve_approval(
        self,
        payload: dict[str, Any],
        request_id: str,
        sink: TextIO,
    ) -> None:
        session_state = self._require_session(payload, request_id, sink)
        if session_state is None:
            return

        approval_id = payload.get("approval_id")
        if not isinstance(approval_id, str) or not approval_id:
            self._emit_error(sink, request_id, session_state.session_id, "invalid_approval_id", "approval_id must be a non-empty string")
            return
        if approval_id not in session_state.pending_approvals:
            self._emit_error(sink, request_id, session_state.session_id, "approval_not_found", f"unknown approval id: {approval_id}")
            return

        decisions = payload.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            self._emit_error(sink, request_id, session_state.session_id, "invalid_decisions", "decisions must be a non-empty list")
            return

        session_state.pending_approvals.pop(approval_id, None)
        result = session_state.session.resume({approval_id: {"decisions": decisions}})
        self._emit_turn_events(session_state, request_id, sink, result.raw_output, result.output_text, result.interrupts)

    def _emit_turn_events(
        self,
        session_state: SessionState,
        request_id: str,
        sink: TextIO,
        raw_output: Any,
        output_text: str,
        interrupts: tuple[PendingInterrupt, ...],
    ) -> None:
        messages = _extract_messages(raw_output)
        new_messages = messages[session_state.emitted_message_count :]
        for message in new_messages:
            if isinstance(message, HumanMessage):
                continue
            if isinstance(message, AIMessage):
                self._emit_ai_message_events(sink, request_id, session_state.session_id, message)
                continue
            if isinstance(message, ToolMessage):
                self._emit(
                    sink,
                    {
                        "type": "tool_result",
                        "request_id": request_id,
                        "session_id": session_state.session_id,
                        "tool_result": {
                            "tool_call_id": message.tool_call_id,
                            "name": message.name,
                            "status": getattr(message, "status", None) or "success",
                            "content": _message_text(message),
                        },
                    },
                )
                continue
        session_state.emitted_message_count = len(messages)

        pending_ids: list[str] = []
        for interrupt in interrupts:
            approval_id = interrupt.id or uuid4().hex
            session_state.pending_approvals[approval_id] = interrupt
            pending_ids.append(approval_id)
            approval_payload = _interrupt_to_approval_payload(interrupt, approval_id)
            self._emit(
                sink,
                {
                    "type": "approval_requested",
                    "request_id": request_id,
                    "session_id": session_state.session_id,
                    "approval": approval_payload,
                },
            )

        self._emit(
            sink,
            {
                "type": "turn_completed",
                "request_id": request_id,
                "session_id": session_state.session_id,
                "status": "interrupted" if pending_ids else "completed",
                "output_text": output_text,
                "pending_approval_ids": pending_ids,
            },
        )

    def _emit_ai_message_events(
        self,
        sink: TextIO,
        request_id: str,
        session_id: str,
        message: AIMessage,
    ) -> None:
        for tool_call in message.tool_calls:
            self._emit(
                sink,
                {
                    "type": "tool_call",
                    "request_id": request_id,
                    "session_id": session_id,
                    "tool_call": {
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args", {}),
                        "status": "requested",
                    },
                },
            )
        content = _message_text(message)
        if content:
            self._emit(
                sink,
                {
                    "type": "assistant_message",
                    "request_id": request_id,
                    "session_id": session_id,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                },
            )

    def _require_session(
        self,
        payload: dict[str, Any],
        request_id: str,
        sink: TextIO,
    ) -> SessionState | None:
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            self._emit_error(sink, request_id, None, "invalid_session_id", "session_id must be a non-empty string")
            return None
        session_state = self.sessions.get(session_id)
        if session_state is None:
            self._emit_error(sink, request_id, session_id, "session_not_found", f"unknown session id: {session_id}")
            return None
        return session_state

    def _emit(self, sink: TextIO, payload: dict[str, Any]) -> None:
        self._sequence += 1
        payload["sequence"] = self._sequence
        sink.write(json.dumps(payload))
        sink.write("\n")
        sink.flush()

    def _emit_error(
        self,
        sink: TextIO,
        request_id: str,
        session_id: str | None,
        code: str,
        message: str,
    ) -> None:
        event: dict[str, Any] = {
            "type": "error",
            "request_id": request_id,
            "code": code,
            "message": message,
        }
        if session_id is not None:
            event["session_id"] = session_id
        self._emit(sink, event)


def run_stdio(
    settings: AssistantSettings,
    instream: TextIO | None = None,
    outstream: TextIO | None = None,
) -> int:
    """Entry point used by the CLI."""

    return StdioServer(AssistantRuntime(settings=settings)).run(instream=instream, outstream=outstream)


def _extract_messages(raw_output: Any) -> list[BaseMessage]:
    if isinstance(raw_output, dict):
        messages = raw_output.get("messages", [])
        return list(messages) if isinstance(messages, list) else []
    messages = getattr(raw_output, "messages", None)
    return list(messages) if isinstance(messages, list) else []


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk)
    text = getattr(message, "text", None)
    return text if isinstance(text, str) else ""


def _interrupt_to_approval_payload(interrupt: PendingInterrupt, approval_id: str) -> dict[str, Any]:
    value = interrupt.value if isinstance(interrupt.value, dict) else {}
    action_requests = value.get("action_requests", [])
    review_configs = value.get("review_configs", [])
    return {
        "id": approval_id,
        "kind": "human_in_the_loop",
        "actions": action_requests if isinstance(action_requests, list) else [],
        "review_configs": review_configs if isinstance(review_configs, list) else [],
    }
