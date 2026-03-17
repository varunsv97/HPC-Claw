from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from json import JSONDecodeError
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from hpc_assistant_backend.config import BackendConfig
from hpc_assistant_backend.filesystem import (
    configured_filesystem_roots,
    list_directory,
    read_text_file,
    write_text_file,
)
from hpc_assistant_backend.runtime import AgentTurnResult, AssistantRuntime, AssistantSession
from hpc_assistant_backend.tools import build_tool_manifest


SESSION_PATH = r"^/api/session/(?P<session_id>[^/]+)$"
PROMPT_PATH = r"^/api/session/(?P<session_id>[^/]+)/prompt$"
APPROVAL_PATH = r"^/api/session/(?P<session_id>[^/]+)/approvals/(?P<approval_id>[^/]+)$"


@dataclass(slots=True)
class ChatMessage:
    id: str
    role: str
    content: str


@dataclass(slots=True)
class ToolEvent:
    id: str
    title: str
    command: str
    status: str
    output: str | None = None
    approval_id: str | None = None


@dataclass(slots=True)
class ApprovalRequest:
    id: str
    title: str
    command: str
    rationale: str
    status: str = "pending"
    tool_event_id: str | None = None
    decision: str | None = None


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    profile: str
    runtime_session: AssistantSession
    messages: list[ChatMessage] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    approvals: list[ApprovalRequest] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "profile": self.profile,
            "messages": [
                {"id": item.id, "role": item.role, "content": item.content}
                for item in self.messages
            ],
            "tool_events": [
                {
                    "id": item.id,
                    "title": item.title,
                    "command": item.command,
                    "status": item.status,
                    "output": item.output,
                    "approval_id": item.approval_id,
                }
                for item in self.tool_events
            ],
            "approvals": [
                {
                    "id": item.id,
                    "title": item.title,
                    "command": item.command,
                    "rationale": item.rationale,
                    "status": item.status,
                    "tool_event_id": item.tool_event_id,
                    "decision": item.decision,
                }
                for item in self.approvals
            ],
        }


class SessionStore:
    def __init__(self, runtime: AssistantRuntime) -> None:
        self.runtime = runtime
        self._lock = Lock()
        self._sessions: dict[str, SessionRecord] = {}

    def open_session(
        self,
        profile: str,
        session_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> tuple[SessionRecord, bool]:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id], True

            active_session_id = session_id or uuid4().hex
            runtime_session = self.runtime.start_session(active_session_id, user_id=user_id)
            record = SessionRecord(
                session_id=active_session_id,
                profile=profile,
                runtime_session=runtime_session,
                messages=[
                    ChatMessage(
                        id=uuid4().hex,
                        role="assistant",
                        content=(
                            "Connected to the cluster backend. I can inspect Slurm, modules, "
                            "filesystem state, and draft approval-gated commands for review."
                        ),
                    )
                ],
            )
            self._sessions[active_session_id] = record
            return record, False

    def get_session(self, session_id: str) -> SessionRecord:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise KeyError(session_id)
            return record

    def handle_prompt(self, session_id: str, prompt: str) -> SessionRecord:
        with self._lock:
            record = self._require_session(session_id)
            record.messages.append(ChatMessage(id=uuid4().hex, role="user", content=prompt))
            result = record.runtime_session.invoke(prompt)
            self._apply_turn_result(record, result)
            return record

    def review_approval(
        self,
        session_id: str,
        approval_id: str,
        decision: str,
        *,
        edited_command: str | None = None,
    ) -> SessionRecord:
        with self._lock:
            record = self._require_session(session_id)
            approval = self._find_approval(record, approval_id)
            tool_event = self._find_tool_event(record, approval.tool_event_id)

            normalized = decision.strip().lower()
            if normalized not in {"approve", "reject", "edit"}:
                raise ValueError(f"unsupported decision: {decision}")
            if normalized == "edit":
                if not edited_command:
                    raise ValueError("edited_command is required for edit decisions")
                approval.command = edited_command
                tool_event.command = edited_command

            approval.status = "rejected" if normalized == "reject" else "approved"
            approval.decision = normalized
            if normalized == "reject":
                tool_event.status = "rejected"

            result = record.runtime_session.resume(
                {
                    "approval_id": approval_id,
                    "decision": normalized,
                    "edited_command": edited_command,
                }
            )
            self._apply_turn_result(record, result)
            return record

    def _apply_turn_result(self, record: SessionRecord, result: AgentTurnResult) -> None:
        raw_output = result.raw_output if isinstance(result.raw_output, dict) else {}
        events = raw_output.get("events")
        if not isinstance(events, list):
            events = []

        saw_assistant_message = False
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", ""))
            if event_type == "assistant_message":
                payload = event.get("message")
                if isinstance(payload, dict):
                    content = str(payload.get("content", "")).strip()
                    if content:
                        record.messages.append(
                            ChatMessage(
                                id=uuid4().hex,
                                role=str(payload.get("role", "assistant")),
                                content=content,
                            )
                        )
                        saw_assistant_message = True
                continue
            if event_type == "tool_call":
                self._apply_tool_call_event(record, event.get("tool_call"))
                continue
            if event_type == "tool_result":
                self._apply_tool_result_event(record, event.get("tool_result"))
                continue
            if event_type == "approval_requested":
                self._apply_approval_event(record, event.get("approval"))

        if result.output_text and not saw_assistant_message:
            record.messages.append(
                ChatMessage(
                    id=uuid4().hex,
                    role="assistant",
                    content=result.output_text,
                )
            )

    def _apply_tool_call_event(self, record: SessionRecord, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        tool_call_id = str(payload.get("id", "")).strip()
        if not tool_call_id:
            return
        tool_event = self._find_tool_event(record, tool_call_id, required=False)
        if tool_event is None:
            tool_event = ToolEvent(
                id=tool_call_id,
                title=_titleize_tool_name(str(payload.get("name", ""))),
                command=str(payload.get("command") or ""),
                status=_normalize_tool_status(str(payload.get("status", ""))),
            )
            record.tool_events.append(tool_event)
            return

        if payload.get("command"):
            tool_event.command = str(payload["command"])
        if payload.get("status"):
            tool_event.status = _normalize_tool_status(str(payload["status"]))

    def _apply_tool_result_event(self, record: SessionRecord, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        tool_call_id = str(payload.get("tool_call_id", "")).strip()
        if not tool_call_id:
            return
        tool_event = self._find_tool_event(record, tool_call_id, required=False)
        if tool_event is None:
            tool_event = ToolEvent(
                id=tool_call_id,
                title=_titleize_tool_name(str(payload.get("name", ""))),
                command=str(payload.get("command") or ""),
                status="completed",
            )
            record.tool_events.append(tool_event)

        command = payload.get("command")
        if isinstance(command, str) and command.strip():
            tool_event.command = command
        tool_event.output = str(payload.get("content") or "")
        tool_event.status = _normalize_result_status(str(payload.get("status", "")), tool_event.status)

    def _apply_approval_event(self, record: SessionRecord, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        approval_id = str(payload.get("id", "")).strip()
        if not approval_id:
            return
        existing = next((item for item in record.approvals if item.id == approval_id), None)
        if existing is not None:
            return

        tool_call_id = str(payload.get("tool_call_id", "")).strip() or None
        approval = ApprovalRequest(
            id=approval_id,
            title=str(payload.get("title", "Review tool action")),
            command=str(payload.get("command", "")),
            rationale=str(payload.get("rationale", "")),
            tool_event_id=tool_call_id,
        )
        record.approvals.append(approval)
        if tool_call_id:
            tool_event = self._find_tool_event(record, tool_call_id, required=False)
            if tool_event is not None:
                tool_event.approval_id = approval_id
                tool_event.status = "waiting_approval"
                if approval.command:
                    tool_event.command = approval.command

    def _find_approval(self, record: SessionRecord, approval_id: str) -> ApprovalRequest:
        approval = next((item for item in record.approvals if item.id == approval_id), None)
        if approval is None:
            raise KeyError(approval_id)
        return approval

    def _find_tool_event(
        self,
        record: SessionRecord,
        tool_event_id: str | None,
        *,
        required: bool = True,
    ) -> ToolEvent | None:
        if not tool_event_id:
            if required:
                raise KeyError("missing_tool_event_id")
            return None
        tool_event = next((item for item in record.tool_events if item.id == tool_event_id), None)
        if tool_event is None and required:
            raise KeyError(tool_event_id)
        return tool_event

    def _require_session(self, session_id: str) -> SessionRecord:
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        return record


def build_status_payload(config: BackendConfig) -> dict[str, object]:
    settings = config.to_settings()
    return {
        "service": "hpc-assistant-backend",
        "status": "ok",
        "backend": {
            "host": config.host,
            "port": config.port,
        },
        "model": {
            "provider": config.provider,
            "base_url": config.base_url,
            "model": config.model,
        },
        "safety": {
            "approval_required": config.approval_required,
            "execution_policy": settings.execution_policy.value,
            "command_timeout_seconds": settings.command_timeout_seconds,
            "filesystem_roots": [str(root) for root in configured_filesystem_roots(settings)],
        },
    }


def build_session_info_payload(config: BackendConfig) -> dict[str, object]:
    settings = config.to_settings()
    return {
        "tui_hint": "Cluster workspace with chat, approvals, and editor support",
        "memory": {
            "thread_store": config.thread_store,
            "long_term_store": config.long_term_store,
        },
        "approval_required": config.approval_required,
        "tools": build_tool_manifest(settings),
        "filesystem": {
            "roots": [str(root) for root in configured_filesystem_roots(settings)],
        },
    }


class BackendRequestHandler(BaseHTTPRequestHandler):
    config: BackendConfig
    sessions: SessionStore

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/healthz":
            self._write_json(200, build_status_payload(self.config))
            return

        if path == "/session-info":
            self._write_json(200, build_session_info_payload(self.config))
            return

        if path == "/api/files":
            self._handle_list_directory(query)
            return

        if path == "/api/file":
            self._handle_read_file(query)
            return

        if path.startswith("/api/session/"):
            session_id = path.rsplit("/", 1)[-1]
            try:
                session = self.sessions.get_session(session_id)
            except KeyError:
                self._write_json(404, {"error": "session_not_found"})
                return
            self._write_json(200, {"session": session.to_dict(), "restored": True})
            return

        self._write_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        try:
            payload = self._read_json()
        except ValueError as error:
            self._write_json(400, {"error": str(error)})
            return

        if path == "/api/session":
            profile = str(payload.get("profile", "local")).strip() or "local"
            session_id = payload.get("session_id")
            user_id = payload.get("user_id")
            if session_id is not None and not isinstance(session_id, str):
                self._write_json(400, {"error": "session_id must be a string"})
                return
            if user_id is not None and not isinstance(user_id, str):
                self._write_json(400, {"error": "user_id must be a string"})
                return
            session, restored = self.sessions.open_session(
                profile=profile,
                session_id=session_id,
                user_id=user_id,
            )
            self._write_json(200, {"session": session.to_dict(), "restored": restored})
            return

        if path.endswith("/prompt") and "/api/session/" in path:
            session_id = path.split("/")[3]
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                self._write_json(400, {"error": "prompt is required"})
                return
            try:
                session = self.sessions.handle_prompt(session_id, prompt)
            except KeyError:
                self._write_json(404, {"error": "session_not_found"})
                return
            self._write_json(200, {"session": session.to_dict(), "restored": True})
            return

        if "/approvals/" in path and path.startswith("/api/session/"):
            parts = path.split("/")
            session_id = parts[3]
            approval_id = parts[5]
            decision = str(payload.get("decision", "")).strip()
            edited_command = payload.get("edited_command")
            if edited_command is not None and not isinstance(edited_command, str):
                self._write_json(400, {"error": "edited_command must be a string"})
                return
            try:
                session = self.sessions.review_approval(
                    session_id,
                    approval_id,
                    decision,
                    edited_command=edited_command.strip() if isinstance(edited_command, str) else None,
                )
            except KeyError:
                self._write_json(404, {"error": "approval_or_session_not_found"})
                return
            except ValueError as error:
                self._write_json(400, {"error": str(error)})
                return
            self._write_json(200, {"session": session.to_dict(), "restored": True})
            return

        if path == "/api/file":
            self._handle_write_file(payload)
            return

        self._write_json(404, {"error": "not_found"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_list_directory(self, query: dict[str, list[str]]) -> None:
        requested = _first_query_value(query, "path")
        include_hidden = _query_bool(query, "include_hidden", default=True)
        try:
            payload = list_directory(
                self.config.to_settings(),
                requested,
                include_hidden=include_hidden,
            )
        except FileNotFoundError:
            self._write_json(404, {"error": "path_not_found"})
            return
        except NotADirectoryError:
            self._write_json(400, {"error": "path_is_not_directory"})
            return
        except ValueError as error:
            self._write_json(400, {"error": str(error)})
            return
        self._write_json(200, payload)

    def _handle_read_file(self, query: dict[str, list[str]]) -> None:
        path = _first_query_value(query, "path")
        if not path:
            self._write_json(400, {"error": "path is required"})
            return
        try:
            payload = read_text_file(self.config.to_settings(), path)
        except FileNotFoundError:
            self._write_json(404, {"error": "path_not_found"})
            return
        except IsADirectoryError:
            self._write_json(400, {"error": "path_is_directory"})
            return
        except ValueError as error:
            self._write_json(400, {"error": str(error)})
            return
        self._write_json(200, payload)

    def _handle_write_file(self, payload: dict[str, Any]) -> None:
        path = payload.get("path")
        content = payload.get("content")
        create_parents = bool(payload.get("create_parents", True))
        if not isinstance(path, str) or not path.strip():
            self._write_json(400, {"error": "path is required"})
            return
        if not isinstance(content, str):
            self._write_json(400, {"error": "content must be a string"})
            return
        try:
            result = write_text_file(
                self.config.to_settings(),
                path.strip(),
                content,
                create_parents=create_parents,
            )
        except ValueError as error:
            self._write_json(400, {"error": str(error)})
            return
        self._write_json(200, result)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}

        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body)
        except JSONDecodeError as error:
            raise ValueError("invalid_json") from error

        if not isinstance(payload, dict):
            raise ValueError("json_body_must_be_object")
        return payload

    def _write_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(
    config: BackendConfig,
    *,
    runtime: AssistantRuntime | None = None,
) -> ThreadingHTTPServer:
    handler = type(
        "ConfiguredBackendHandler",
        (BackendRequestHandler,),
        {
            "config": config,
            "sessions": SessionStore(runtime or AssistantRuntime(settings=config.to_settings())),
        },
    )
    server = ThreadingHTTPServer((config.host, config.port), handler)
    server.daemon_threads = True
    return server


def serve(config: BackendConfig) -> None:
    server = create_server(config)
    host, port = server.server_address
    print(f"backend listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def _query_bool(query: dict[str, list[str]], key: str, *, default: bool) -> bool:
    value = _first_query_value(query, key)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


def _titleize_tool_name(name: str) -> str:
    words = [part for part in name.split("_") if part]
    return " ".join(word.capitalize() for word in words) or "Tool"


def _normalize_tool_status(status: str) -> str:
    lowered = status.lower()
    if lowered in {"waiting_approval", "waiting"}:
        return "waiting_approval"
    if lowered in {"completed", "success"}:
        return "completed"
    if lowered == "error":
        return "error"
    return lowered or "pending"


def _normalize_result_status(result_status: str, current_status: str) -> str:
    lowered = result_status.lower()
    if lowered == "success":
        return "completed"
    if lowered == "error":
        return "rejected" if current_status == "rejected" else "error"
    return current_status
