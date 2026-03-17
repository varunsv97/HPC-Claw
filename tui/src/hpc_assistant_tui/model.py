from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import os
from pathlib import Path
from typing import Any


def _env_default(key: str) -> str | None:
    value = os.environ.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def default_state_path() -> Path:
    explicit = _env_default("HPC_ASSISTANT_STATE_PATH")
    if explicit is not None:
        return Path(explicit)

    xdg_state_home = _env_default("XDG_STATE_HOME")
    if xdg_state_home is not None:
        return Path(xdg_state_home) / "hpc-assistant" / "tui-state.json"

    home = _env_default("HOME")
    if home is not None:
        return Path(home) / ".local" / "state" / "hpc-assistant" / "tui-state.json"

    return Path(".hpc-assistant-tui-state.json")


class FocusPane(StrEnum):
    CHAT = "chat"
    FILES = "files"
    EDITOR = "editor"
    TOOLS = "tools"
    APPROVALS = "approvals"
    STATUS = "status"

    def label(self) -> str:
        return {
            self.CHAT: "Chat",
            self.FILES: "Files",
            self.EDITOR: "Editor",
            self.TOOLS: "Tools",
            self.APPROVALS: "Approvals",
            self.STATUS: "Status",
        }[self]

    def next(self) -> "FocusPane":
        order = [
            self.CHAT,
            self.FILES,
            self.EDITOR,
            self.TOOLS,
            self.APPROVALS,
            self.STATUS,
        ]
        return order[(order.index(self) + 1) % len(order)]

    def previous(self) -> "FocusPane":
        order = [
            self.CHAT,
            self.FILES,
            self.EDITOR,
            self.TOOLS,
            self.APPROVALS,
            self.STATUS,
        ]
        return order[(order.index(self) - 1) % len(order)]


class InputMode(StrEnum):
    PROMPT = "prompt"
    APPROVAL_EDIT = "approval_edit"

    def label(self) -> str:
        return {
            self.PROMPT: "Prompt",
            self.APPROVAL_EDIT: "Approval Edit",
        }[self]


@dataclass(slots=True)
class TuiConfig:
    backend_url: str
    profile: str
    state_path: str

    @classmethod
    def with_defaults(cls) -> "TuiConfig":
        return cls(
            backend_url=_env_default("HPC_ASSISTANT_BACKEND_URL") or "http://127.0.0.1:8765",
            profile=_env_default("HPC_ASSISTANT_PROFILE") or "local",
            state_path=str(default_state_path()),
        )


@dataclass(slots=True)
class ChatMessage:
    id: str = ""
    role: str = ""
    content: str = ""


@dataclass(slots=True)
class ToolEvent:
    id: str = ""
    title: str = ""
    command: str = ""
    status: str = ""
    output: str | None = None
    approval_id: str | None = None


@dataclass(slots=True)
class ApprovalRequest:
    id: str = ""
    title: str = ""
    command: str = ""
    rationale: str = ""
    status: str = ""
    tool_event_id: str | None = None
    decision: str | None = None


@dataclass(slots=True)
class SessionState:
    session_id: str = ""
    profile: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    approvals: list[ApprovalRequest] = field(default_factory=list)


@dataclass(slots=True)
class SessionEnvelope:
    restored: bool = False
    session: SessionState = field(default_factory=SessionState)


@dataclass(slots=True)
class BackendAddress:
    host: str = ""
    port: int = 0


@dataclass(slots=True)
class ModelStatus:
    provider: str = ""
    base_url: str | None = None
    model: str = ""


@dataclass(slots=True)
class SafetyStatus:
    approval_required: bool = False
    execution_policy: str = ""
    command_timeout_seconds: float = 0.0
    filesystem_roots: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BackendStatus:
    service: str = ""
    status: str = ""
    backend: BackendAddress = field(default_factory=BackendAddress)
    model: ModelStatus = field(default_factory=ModelStatus)
    safety: SafetyStatus = field(default_factory=SafetyStatus)


@dataclass(slots=True)
class MemoryStatus:
    thread_store: str = ""
    long_term_store: str = ""


@dataclass(slots=True)
class ToolManifest:
    execution_policy: str = ""
    command_timeout_seconds: float = 0.0
    filesystem_roots: list[str] = field(default_factory=list)
    active_tools: list[str] = field(default_factory=list)
    catalog: list[dict[str, Any]] = field(default_factory=list)
    interrupt_on: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FilesystemStatus:
    roots: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SessionInfo:
    tui_hint: str = ""
    memory: MemoryStatus = field(default_factory=MemoryStatus)
    approval_required: bool = False
    tools: ToolManifest = field(default_factory=ToolManifest)
    filesystem: FilesystemStatus = field(default_factory=FilesystemStatus)


@dataclass(slots=True)
class FileEntry:
    name: str = ""
    path: str = ""
    is_dir: bool = False
    size: int = 0
    modified_at: str = ""


@dataclass(slots=True)
class DirectoryListing:
    path: str = ""
    entries: list[FileEntry] = field(default_factory=list)


@dataclass(slots=True)
class FileDocument:
    path: str = ""
    size: int = 0
    modified_at: str = ""
    content: str = ""
    truncated: bool = False
    encoding: str = "utf-8"


@dataclass(slots=True)
class WorkspaceState:
    current_directory: str = ""
    open_file_path: str = ""
    editor_text: str = ""
    editor_dirty: bool = False


@dataclass(slots=True)
class PersistedState:
    config: TuiConfig = field(default_factory=TuiConfig.with_defaults)
    session: SessionState = field(default_factory=SessionState)
    backend_status: BackendStatus | None = None
    session_info: SessionInfo | None = None
    draft: str = ""
    focus: FocusPane = FocusPane.CHAT
    selected_tool: int = 0
    selected_approval: int = 0
    workspace: WorkspaceState = field(default_factory=WorkspaceState)
    notice: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["focus"] = self.focus.value
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersistedState":
        return cls(
            config=_tui_config_from_dict(payload.get("config")),
            session=_session_state_from_dict(payload.get("session")),
            backend_status=_backend_status_from_dict(payload.get("backend_status")),
            session_info=_session_info_from_dict(payload.get("session_info")),
            draft=str(payload.get("draft", "")),
            focus=_focus_from_value(payload.get("focus")),
            selected_tool=max(0, int(payload.get("selected_tool", 0))),
            selected_approval=max(0, int(payload.get("selected_approval", 0))),
            workspace=_workspace_state_from_dict(payload.get("workspace")),
            notice=str(payload.get("notice", "")),
        )


def _focus_from_value(value: Any) -> FocusPane:
    try:
        return FocusPane(str(value))
    except ValueError:
        return FocusPane.CHAT


def _tui_config_from_dict(payload: Any) -> TuiConfig:
    if not isinstance(payload, dict):
        return TuiConfig.with_defaults()
    defaults = TuiConfig.with_defaults()
    return TuiConfig(
        backend_url=str(payload.get("backend_url", defaults.backend_url)),
        profile=str(payload.get("profile", defaults.profile)),
        state_path=str(payload.get("state_path", defaults.state_path)),
    )


def _chat_message_from_dict(payload: Any) -> ChatMessage:
    if not isinstance(payload, dict):
        return ChatMessage()
    return ChatMessage(
        id=str(payload.get("id", "")),
        role=str(payload.get("role", "")),
        content=str(payload.get("content", "")),
    )


def _tool_event_from_dict(payload: Any) -> ToolEvent:
    if not isinstance(payload, dict):
        return ToolEvent()
    output = payload.get("output")
    approval_id = payload.get("approval_id")
    return ToolEvent(
        id=str(payload.get("id", "")),
        title=str(payload.get("title", "")),
        command=str(payload.get("command", "")),
        status=str(payload.get("status", "")),
        output=None if output is None else str(output),
        approval_id=None if approval_id is None else str(approval_id),
    )


def _approval_request_from_dict(payload: Any) -> ApprovalRequest:
    if not isinstance(payload, dict):
        return ApprovalRequest()
    tool_event_id = payload.get("tool_event_id")
    decision = payload.get("decision")
    return ApprovalRequest(
        id=str(payload.get("id", "")),
        title=str(payload.get("title", "")),
        command=str(payload.get("command", "")),
        rationale=str(payload.get("rationale", "")),
        status=str(payload.get("status", "")),
        tool_event_id=None if tool_event_id is None else str(tool_event_id),
        decision=None if decision is None else str(decision),
    )


def _session_state_from_dict(payload: Any) -> SessionState:
    if not isinstance(payload, dict):
        return SessionState()
    return SessionState(
        session_id=str(payload.get("session_id", "")),
        profile=str(payload.get("profile", "")),
        messages=[_chat_message_from_dict(item) for item in payload.get("messages", [])],
        tool_events=[_tool_event_from_dict(item) for item in payload.get("tool_events", [])],
        approvals=[_approval_request_from_dict(item) for item in payload.get("approvals", [])],
    )


def _session_envelope_from_dict(payload: Any) -> SessionEnvelope:
    if not isinstance(payload, dict):
        return SessionEnvelope()
    return SessionEnvelope(
        restored=bool(payload.get("restored", False)),
        session=_session_state_from_dict(payload.get("session")),
    )


def _backend_address_from_dict(payload: Any) -> BackendAddress:
    if not isinstance(payload, dict):
        return BackendAddress()
    return BackendAddress(
        host=str(payload.get("host", "")),
        port=int(payload.get("port", 0) or 0),
    )


def _model_status_from_dict(payload: Any) -> ModelStatus:
    if not isinstance(payload, dict):
        return ModelStatus()
    base_url = payload.get("base_url")
    return ModelStatus(
        provider=str(payload.get("provider", "")),
        base_url=None if base_url is None else str(base_url),
        model=str(payload.get("model", "")),
    )


def _safety_status_from_dict(payload: Any) -> SafetyStatus:
    if not isinstance(payload, dict):
        return SafetyStatus()
    roots = payload.get("filesystem_roots")
    return SafetyStatus(
        approval_required=bool(payload.get("approval_required", False)),
        execution_policy=str(payload.get("execution_policy", "")),
        command_timeout_seconds=float(payload.get("command_timeout_seconds", 0.0) or 0.0),
        filesystem_roots=[str(item) for item in roots] if isinstance(roots, list) else [],
    )


def _backend_status_from_dict(payload: Any) -> BackendStatus | None:
    if not isinstance(payload, dict):
        return None
    return BackendStatus(
        service=str(payload.get("service", "")),
        status=str(payload.get("status", "")),
        backend=_backend_address_from_dict(payload.get("backend")),
        model=_model_status_from_dict(payload.get("model")),
        safety=_safety_status_from_dict(payload.get("safety")),
    )


def _memory_status_from_dict(payload: Any) -> MemoryStatus:
    if not isinstance(payload, dict):
        return MemoryStatus()
    return MemoryStatus(
        thread_store=str(payload.get("thread_store", "")),
        long_term_store=str(payload.get("long_term_store", "")),
    )


def _tool_manifest_from_dict(payload: Any) -> ToolManifest:
    if not isinstance(payload, dict):
        return ToolManifest()
    filesystem_roots = payload.get("filesystem_roots")
    active_tools = payload.get("active_tools")
    catalog = payload.get("catalog")
    interrupt_on = payload.get("interrupt_on")
    return ToolManifest(
        execution_policy=str(payload.get("execution_policy", "")),
        command_timeout_seconds=float(payload.get("command_timeout_seconds", 0.0) or 0.0),
        filesystem_roots=[str(item) for item in filesystem_roots] if isinstance(filesystem_roots, list) else [],
        active_tools=[str(item) for item in active_tools] if isinstance(active_tools, list) else [],
        catalog=[item for item in catalog if isinstance(item, dict)] if isinstance(catalog, list) else [],
        interrupt_on=dict(interrupt_on) if isinstance(interrupt_on, dict) else {},
    )


def _filesystem_status_from_dict(payload: Any) -> FilesystemStatus:
    if not isinstance(payload, dict):
        return FilesystemStatus()
    roots = payload.get("roots")
    return FilesystemStatus(
        roots=[str(item) for item in roots] if isinstance(roots, list) else [],
    )


def _session_info_from_dict(payload: Any) -> SessionInfo | None:
    if not isinstance(payload, dict):
        return None
    tui_hint = payload.get("tui_hint", payload.get("frontend_hint", ""))
    return SessionInfo(
        tui_hint=str(tui_hint),
        memory=_memory_status_from_dict(payload.get("memory")),
        approval_required=bool(payload.get("approval_required", False)),
        tools=_tool_manifest_from_dict(payload.get("tools")),
        filesystem=_filesystem_status_from_dict(payload.get("filesystem")),
    )


def _file_entry_from_dict(payload: Any) -> FileEntry:
    if not isinstance(payload, dict):
        return FileEntry()
    return FileEntry(
        name=str(payload.get("name", "")),
        path=str(payload.get("path", "")),
        is_dir=bool(payload.get("is_dir", False)),
        size=int(payload.get("size", 0) or 0),
        modified_at=str(payload.get("modified_at", "")),
    )


def _directory_listing_from_dict(payload: Any) -> DirectoryListing | None:
    if not isinstance(payload, dict):
        return None
    entries = payload.get("entries")
    return DirectoryListing(
        path=str(payload.get("path", "")),
        entries=[_file_entry_from_dict(item) for item in entries] if isinstance(entries, list) else [],
    )


def _file_document_from_dict(payload: Any) -> FileDocument | None:
    if not isinstance(payload, dict):
        return None
    return FileDocument(
        path=str(payload.get("path", "")),
        size=int(payload.get("size", 0) or 0),
        modified_at=str(payload.get("modified_at", "")),
        content=str(payload.get("content", "")),
        truncated=bool(payload.get("truncated", False)),
        encoding=str(payload.get("encoding", "utf-8")),
    )


def _workspace_state_from_dict(payload: Any) -> WorkspaceState:
    if not isinstance(payload, dict):
        return WorkspaceState()
    return WorkspaceState(
        current_directory=str(payload.get("current_directory", "")),
        open_file_path=str(payload.get("open_file_path", "")),
        editor_text=str(payload.get("editor_text", "")),
        editor_dirty=bool(payload.get("editor_dirty", False)),
    )


__all__ = [
    "ApprovalRequest",
    "BackendAddress",
    "BackendStatus",
    "ChatMessage",
    "DirectoryListing",
    "FileDocument",
    "FileEntry",
    "FilesystemStatus",
    "FocusPane",
    "InputMode",
    "MemoryStatus",
    "ModelStatus",
    "PersistedState",
    "SafetyStatus",
    "SessionEnvelope",
    "SessionInfo",
    "SessionState",
    "ToolEvent",
    "ToolManifest",
    "TuiConfig",
    "WorkspaceState",
    "default_state_path",
    "_directory_listing_from_dict",
    "_file_document_from_dict",
    "_session_envelope_from_dict",
]
