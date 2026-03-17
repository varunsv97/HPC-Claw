from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys
import sysconfig
import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, Log, Static, TextArea, Tree

from .client import BackendClient
from .model import (
    ApprovalRequest,
    BackendStatus,
    DirectoryListing,
    FileDocument,
    FocusPane,
    InputMode,
    PersistedState,
    SessionInfo,
    SessionState,
    ToolEvent,
    TuiConfig,
    WorkspaceState,
)
from .storage import load_snapshot, save_snapshot


@dataclass(slots=True)
class BrowserNode:
    path: str
    is_dir: bool
    loaded: bool = False


class HPCAssistantTextualApp(App[None]):
    TITLE = "HPC Assistant"
    SUB_TITLE = "Python Textual TUI"
    CSS = """
    Screen {
        layout: vertical;
    }

    Header {
        dock: top;
    }

    Footer {
        dock: bottom;
    }

    #summary {
        height: 3;
        border: round $accent;
        padding: 0 1;
        margin: 0 1 1 1;
    }

    #body {
        height: 1fr;
        margin: 0 1;
    }

    #left-column {
        width: 30;
        margin-right: 1;
    }

    #center-column {
        width: 3fr;
        margin-right: 1;
    }

    #right-column {
        width: 2fr;
    }

    .pane {
        border: round $panel;
        padding: 0 1;
        margin-bottom: 1;
    }

    .pane-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #files-pane {
        height: 1fr;
    }

    #status-pane {
        height: 12;
        margin-bottom: 0;
    }

    #editor-pane {
        height: 1fr;
    }

    #conversation-pane {
        height: 2fr;
    }

    #tools-pane {
        height: 1fr;
    }

    #approvals-pane {
        height: 1fr;
        margin-bottom: 0;
    }

    Log {
        height: 1fr;
    }

    Tree {
        height: 1fr;
    }

    TextArea {
        height: 1fr;
    }

    #editor-meta,
    #files-meta,
    #composer-mode {
        color: $warning;
        margin-bottom: 1;
    }

    #composer-pane {
        height: 8;
        border: round $accent;
        padding: 0 1;
        margin: 0 1 1 1;
    }

    #prompt-input {
        margin-bottom: 1;
    }

    #button-row {
        height: auto;
    }

    #button-row Button {
        margin-right: 1;
    }

    #notice {
        height: 3;
        border: round $panel;
        padding: 0 1;
        margin: 0 1 1 1;
    }
    """
    BINDINGS = [
        Binding("f5", "sync_backend", "Refresh"),
        Binding("f6", "focus_next_pane", "Next Pane"),
        Binding("f7", "focus_previous_pane", "Prev Pane"),
        Binding("f8", "select_next", "Next Item"),
        Binding("f9", "select_previous", "Prev Item"),
        Binding("f10", "approve_selected", "Approve"),
        Binding("f11", "reject_selected", "Reject"),
        Binding("f12", "edit_selected", "Edit"),
        Binding("ctrl+o", "prompt_mode", "Prompt"),
        Binding("ctrl+s", "save_file", "Save File"),
        Binding("ctrl+r", "reload_file", "Reload File"),
        Binding("ctrl+q", "save_and_quit", "Quit"),
    ]

    def __init__(self, config: TuiConfig) -> None:
        super().__init__()
        self.config = config
        self.client = BackendClient(config.backend_url)
        self.state_path = Path(config.state_path)

        self.session = SessionState(profile=config.profile)
        self.backend_status: BackendStatus | None = None
        self.session_info: SessionInfo | None = None
        self.workspace = WorkspaceState()
        self.directory_cache: dict[str, DirectoryListing] = {}
        self.open_document: FileDocument | None = None
        self.loading_editor = False
        self.draft = ""
        self.focus = FocusPane.CHAT
        self.selected_tool = 0
        self.selected_approval = 0
        self.input_mode = InputMode.PROMPT
        self.approval_edit_buffer = ""
        self.notice = ""
        self.backend_busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="summary")

        with Horizontal(id="body"):
            with Vertical(id="left-column"):
                with Vertical(id="files-pane", classes="pane"):
                    yield Label("Files", classes="pane-title")
                    yield Label("", id="files-meta")
                    yield Tree("Workspace", id="file-tree")
                with Vertical(id="status-pane", classes="pane"):
                    yield Label("Status", classes="pane-title")
                    yield Log(id="status-log", highlight=False)

            with Vertical(id="center-column"):
                with Vertical(id="editor-pane", classes="pane"):
                    yield Label("Editor", classes="pane-title")
                    yield Label("", id="editor-meta")
                    yield TextArea("", id="editor", show_line_numbers=True, soft_wrap=False)

            with Vertical(id="right-column"):
                with Vertical(id="conversation-pane", classes="pane"):
                    yield Label("Conversation", classes="pane-title")
                    yield Log(id="conversation", highlight=False, auto_scroll=True)
                with Vertical(id="tools-pane", classes="pane"):
                    yield Label("Tool Activity", classes="pane-title")
                    yield Log(id="tools-log", highlight=False)
                with Vertical(id="approvals-pane", classes="pane"):
                    yield Label("Approvals", classes="pane-title")
                    yield Log(id="approvals-log", highlight=False)

        with Vertical(id="composer-pane"):
            yield Label("", id="composer-mode")
            yield Input(
                placeholder="Type a prompt and press Enter",
                id="prompt-input",
                value="",
            )
            with Horizontal(id="button-row"):
                yield Button("Send", id="send-button")
                yield Button("Refresh", id="refresh-button")
                yield Button("Save File", id="save-file-button")
                yield Button("Reload File", id="reload-file-button")
                yield Button("Approve", id="approve-button")
                yield Button("Reject", id="reject-button")
                yield Button("Edit", id="edit-button")
                yield Button("Prompt Mode", id="prompt-mode-button")

        yield Static("", id="notice")
        yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one("#file-tree", Tree)
        tree.root.data = BrowserNode(path="", is_dir=True, loaded=True)
        tree.root.expand()

        restored_snapshot = False
        try:
            snapshot = load_snapshot(self.state_path)
        except RuntimeError as error:
            self.notice = str(error)
            snapshot = None
        if snapshot is not None:
            self.apply_snapshot(snapshot)
            restored_snapshot = True

        sync_message: str
        try:
            sync_message = self.sync_backend()
        except RuntimeError as error:
            if restored_snapshot:
                sync_message = f"Restored local snapshot, but backend sync failed: {error}"
            else:
                sync_message = f"Backend sync failed: {error}"

        if restored_snapshot and not sync_message.startswith("Restored"):
            self.notice = f"Restored local snapshot. {sync_message}"
        else:
            self.notice = sync_message

        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt-input":
            return
        if self.input_mode is InputMode.APPROVAL_EDIT:
            self.approval_edit_buffer = event.value
        else:
            self.draft = event.value

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt-input":
            return
        if self.input_mode is InputMode.APPROVAL_EDIT:
            self.submit_approval_edit()
        else:
            self.send_prompt()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "send-button":
            if self.input_mode is InputMode.APPROVAL_EDIT:
                self.submit_approval_edit()
            else:
                self.send_prompt()
            return
        if button_id == "refresh-button":
            self.action_sync_backend()
            return
        if button_id == "save-file-button":
            self.action_save_file()
            return
        if button_id == "reload-file-button":
            self.action_reload_file()
            return
        if button_id == "approve-button":
            self.action_approve_selected()
            return
        if button_id == "reject-button":
            self.action_reject_selected()
            return
        if button_id == "edit-button":
            self.action_edit_selected()
            return
        if button_id == "prompt-mode-button":
            self.action_prompt_mode()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node = event.node
        data = node.data
        if not isinstance(data, BrowserNode) or not data.path:
            return
        self.focus = FocusPane.FILES
        if data.is_dir:
            self.start_backend_task(
                f"Loading directory {data.path}...",
                lambda: self._load_directory_task(data.path),
            )
        else:
            if self.workspace.editor_dirty and data.path != self.workspace.open_file_path:
                self.notice = "Save or reload the current file before opening another one."
                self.refresh_screen()
                self.persist_state()
                return
            self.start_backend_task(
                f"Opening file {data.path}...",
                lambda: self._open_file_task(data.path),
            )
        self.refresh_screen()
        self.persist_state()

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        data = node.data
        if not isinstance(data, BrowserNode) or not data.path or not data.is_dir:
            return
        if data.loaded:
            return
        self.start_backend_task(
            f"Loading directory {data.path}...",
            lambda: self._load_directory_task(data.path),
        )
        self.refresh_screen()
        self.persist_state()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "editor":
            return
        self.workspace.editor_text = event.text_area.text
        if self.loading_editor:
            return
        current_content = self.open_document.content if self.open_document is not None else ""
        self.workspace.editor_dirty = event.text_area.text != current_content
        self.refresh_screen()
        self.persist_state()

    def action_sync_backend(self) -> None:
        self.start_backend_task("Refreshing backend in a worker thread...", self._sync_backend_task)
        self.refresh_screen()
        self.persist_state()

    def action_focus_next_pane(self) -> None:
        self.focus = self.focus.next()
        self.notice = f"Focus moved to {self.focus.label()}."
        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def action_focus_previous_pane(self) -> None:
        self.focus = self.focus.previous()
        self.notice = f"Focus moved to {self.focus.label()}."
        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def action_select_next(self) -> None:
        if self.move_selection(1):
            self.notice = f"Selection moved in {self.focus.label()}."
        else:
            self.notice = f"No selectable items in {self.focus.label()}."
        self.refresh_screen()
        self.persist_state()

    def action_select_previous(self) -> None:
        if self.move_selection(-1):
            self.notice = f"Selection moved in {self.focus.label()}."
        else:
            self.notice = f"No selectable items in {self.focus.label()}."
        self.refresh_screen()
        self.persist_state()

    def action_approve_selected(self) -> None:
        self.resolve_selected_approval("approve")

    def action_reject_selected(self) -> None:
        self.resolve_selected_approval("reject")

    def action_edit_selected(self) -> None:
        approval = self.selected_approval_request()
        if approval is None:
            self.notice = "No approval request is selected."
        elif approval.status != "pending":
            self.notice = "Selected approval request is already resolved."
        else:
            self.input_mode = InputMode.APPROVAL_EDIT
            self.focus = FocusPane.APPROVALS
            self.approval_edit_buffer = approval.command
            input_widget = self.query_one("#prompt-input", Input)
            input_widget.value = approval.command
            input_widget.focus()
            self.notice = "Editing the selected approval command. Press Enter to submit."
        self.refresh_screen()
        self.persist_state()

    def action_prompt_mode(self) -> None:
        self.input_mode = InputMode.PROMPT
        self.approval_edit_buffer = ""
        input_widget = self.query_one("#prompt-input", Input)
        input_widget.value = self.draft
        self.focus = FocusPane.CHAT
        self.notice = "Prompt mode is active."
        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def action_save_file(self) -> None:
        if not self.workspace.open_file_path:
            self.notice = "No file is open in the editor."
            self.refresh_screen()
            self.persist_state()
            return
        content = self.query_one("#editor", TextArea).text
        self.start_backend_task(
            f"Saving {self.workspace.open_file_path}...",
            lambda: self._save_file_task(self.workspace.open_file_path, content),
        )
        self.refresh_screen()
        self.persist_state()

    def action_reload_file(self) -> None:
        if not self.workspace.open_file_path:
            self.notice = "No file is open in the editor."
            self.refresh_screen()
            self.persist_state()
            return
        if self.workspace.editor_dirty:
            self.notice = "Save the file first or discard edits before reloading."
            self.refresh_screen()
            self.persist_state()
            return
        self.start_backend_task(
            f"Reloading {self.workspace.open_file_path}...",
            lambda: self._open_file_task(self.workspace.open_file_path),
        )
        self.refresh_screen()
        self.persist_state()

    def action_save_and_quit(self) -> None:
        self.notice = "Exiting the local app."
        self.persist_state()
        self.exit()

    def start_backend_task(
        self,
        notice: str,
        worker: Callable[[], None],
    ) -> None:
        if self.backend_busy:
            self.notice = "A backend request is already running."
            self.refresh_screen()
            self.persist_state()
            return

        self.backend_busy = True
        self.notice = notice
        self.refresh_screen()
        self.persist_state()

        threading.Thread(
            target=self._run_backend_task,
            args=(worker,),
            daemon=True,
            name="hpc-assistant-backend-worker",
        ).start()

    def _run_backend_task(self, worker: Callable[[], None]) -> None:
        try:
            worker()
        except Exception as error:  # noqa: BLE001 - keep the TUI responsive on backend failures
            self.call_from_thread(self._finish_backend_task_error, str(error))

    def _finish_backend_task_error(self, message: str) -> None:
        self.backend_busy = False
        self.notice = message
        self.refresh_screen()
        self.persist_state()

    def apply_snapshot(self, snapshot: PersistedState) -> None:
        self.session = snapshot.session
        if not self.session.profile:
            self.session.profile = self.config.profile
        self.backend_status = snapshot.backend_status
        self.session_info = snapshot.session_info
        self.workspace = snapshot.workspace
        self.draft = snapshot.draft
        self.focus = snapshot.focus
        self.selected_tool = snapshot.selected_tool
        self.selected_approval = snapshot.selected_approval
        self.notice = snapshot.notice
        self.normalize_selection()

    def build_snapshot(self) -> PersistedState:
        return PersistedState(
            config=self.config,
            session=self.session,
            backend_status=self.backend_status,
            session_info=self.session_info,
            draft=self.draft,
            focus=self.focus,
            selected_tool=self.selected_tool,
            selected_approval=self.selected_approval,
            workspace=self.workspace,
            notice=self.notice,
        )

    def persist_state(self) -> None:
        try:
            save_snapshot(self.state_path, self.build_snapshot())
        except RuntimeError as error:
            self.notice = str(error)
            self.query_one("#notice", Static).update(self.notice)

    def sync_backend(self) -> str:
        self.backend_status = self.client.fetch_status()
        self.session_info = self.client.fetch_session_info()
        current_session_id = self.session.session_id or None
        envelope = self.client.open_session(self.config.profile, current_session_id)
        self.session = envelope.session
        if not self.session.profile:
            self.session.profile = self.config.profile
        self.normalize_selection()
        self._initialize_workspace_from_session_info()
        return (
            "Connected to backend and restored the active session."
            if envelope.restored
            else "Connected to backend and opened a new session."
        )

    def send_prompt(self) -> None:
        prompt = self.query_one("#prompt-input", Input).value.strip()
        if not prompt:
            self.notice = "Prompt is empty."
            self.refresh_screen()
            self.persist_state()
            return
        current_session_id = self.session.session_id
        self.draft = prompt
        self.start_backend_task(
            "Sending prompt in a worker thread...",
            lambda: self._send_prompt_task(prompt, current_session_id),
        )
        self.refresh_screen()
        self.persist_state()

    def submit_approval_edit(self) -> None:
        edited_command = self.query_one("#prompt-input", Input).value.strip()
        if not edited_command:
            self.notice = "Edited command cannot be empty."
            self.refresh_screen()
            self.persist_state()
            return
        self.resolve_selected_approval("edit", edited_command)

    def resolve_selected_approval(
        self,
        decision: str,
        edited_command: str | None = None,
    ) -> None:
        approval = self.selected_approval_request()
        if approval is None:
            self.notice = "No approval request is selected."
            self.refresh_screen()
            self.persist_state()
            return
        if approval.status != "pending":
            self.notice = "Selected approval request is already resolved."
            self.refresh_screen()
            self.persist_state()
            return
        if not self.session.session_id:
            self.notice = "No active backend session is available."
            self.refresh_screen()
            self.persist_state()
            return
        session_id = self.session.session_id
        approval_id = approval.id
        self.start_backend_task(
            "Submitting approval decision in a worker thread...",
            lambda: self._resolve_approval_task(
                session_id,
                approval_id,
                decision,
                edited_command,
            ),
        )
        self.refresh_screen()
        self.persist_state()

    def move_selection(self, delta: int) -> bool:
        if self.focus is FocusPane.TOOLS:
            return _move_index(self, "selected_tool", len(self.session.tool_events), delta)
        if self.focus is FocusPane.APPROVALS:
            return _move_index(
                self,
                "selected_approval",
                len(self.session.approvals),
                delta,
            )
        return False

    def selected_approval_request(self) -> ApprovalRequest | None:
        if not self.session.approvals:
            return None
        return self.session.approvals[self.selected_approval]

    def normalize_selection(self) -> None:
        if not self.session.profile:
            self.session.profile = self.config.profile
        self.selected_tool = _clamp_index(self.selected_tool, len(self.session.tool_events))
        self.selected_approval = _clamp_index(
            self.selected_approval,
            len(self.session.approvals),
        )

    def refresh_screen(self) -> None:
        self.normalize_selection()
        self.query_one("#summary", Static).update(self.render_summary())
        self.refresh_log("#conversation", self.render_conversation_lines())
        self.refresh_log("#status-log", self.render_status_lines())
        self.refresh_log("#tools-log", self.render_tool_lines())
        self.refresh_log("#approvals-log", self.render_approval_lines())
        self.query_one("#composer-mode", Label).update(self.render_composer_mode())
        self.query_one("#files-meta", Label).update(self.render_files_meta())
        self.query_one("#editor-meta", Label).update(self.render_editor_meta())
        self.query_one("#notice", Static).update(self.notice or "Ready.")

        input_widget = self.query_one("#prompt-input", Input)
        if self.input_mode is InputMode.APPROVAL_EDIT:
            input_widget.placeholder = "Edit the selected command and press Enter"
            if input_widget.value != self.approval_edit_buffer:
                input_widget.value = self.approval_edit_buffer
        else:
            input_widget.placeholder = "Type a prompt and press Enter"
            if input_widget.value != self.draft:
                input_widget.value = self.draft

    def refresh_log(self, selector: str, lines: list[str]) -> None:
        log_widget = self.query_one(selector, Log)
        log_widget.clear()
        log_widget.write_lines(lines)

    def render_summary(self) -> str:
        session_id = self.session.session_id or "pending"
        pending_approvals = sum(
            1 for approval in self.session.approvals if approval.status == "pending"
        )
        open_file = Path(self.workspace.open_file_path).name if self.workspace.open_file_path else "none"
        return (
            "Profile: "
            f"{self.session.profile or self.config.profile} | "
            f"Session: {session_id} | "
            f"Messages: {len(self.session.messages)} | "
            f"Tools: {len(self.session.tool_events)} | "
            f"Pending approvals: {pending_approvals} | "
            f"File: {open_file} | "
            f"Focus: {self.focus.label()} | "
            f"Mode: {self.input_mode.label()} | "
            f"Runtime: {_runtime_mode_label()} | "
            f"Backend busy: {'yes' if self.backend_busy else 'no'}"
        )

    def render_conversation_lines(self) -> list[str]:
        if not self.session.messages:
            return ["No conversation yet. Type a prompt below and press Enter."]

        lines: list[str] = []
        for message in self.session.messages:
            lines.append(f"{_role_label(message.role)}")
            if message.content:
                for line in message.content.splitlines():
                    lines.append(f"  {line}")
            else:
                lines.append("  ")
            lines.append("")
        return lines[:-1] if lines else lines

    def render_status_lines(self) -> list[str]:
        backend_status = self.backend_status
        session_info = self.session_info
        active_tools = session_info.tools.active_tools if session_info is not None else []
        return [
            f"Backend URL: {self.config.backend_url}",
            f"State path: {self.config.state_path}",
            (
                "Connection: "
                f"{backend_status.status if backend_status is not None else 'offline'}"
            ),
            (
                "Model: "
                f"{backend_status.model.provider if backend_status is not None else 'unknown'}"
                "/"
                f"{backend_status.model.model if backend_status is not None else 'unknown'}"
            ),
            (
                "Execution policy: "
                f"{backend_status.safety.execution_policy if backend_status is not None else 'unknown'}"
            ),
            (
                "Approval required: "
                f"{session_info.approval_required if session_info is not None else False}"
            ),
            (
                "Workspace roots: "
                f"{', '.join(session_info.filesystem.roots) if session_info is not None else 'unknown'}"
            ),
            f"Current directory: {self.workspace.current_directory or 'n/a'}",
            f"Open file: {self.workspace.open_file_path or 'n/a'}",
            f"Active tools: {', '.join(active_tools[:4]) + (' ...' if len(active_tools) > 4 else '') if active_tools else 'n/a'}",
            (
                "TUI hint: "
                f"{session_info.tui_hint if session_info is not None else 'n/a'}"
            ),
        ]

    def render_tool_lines(self) -> list[str]:
        if not self.session.tool_events:
            return ["No tool activity yet."]

        lines: list[str] = []
        for index, event in enumerate(self.session.tool_events):
            prefix = ">" if index == self.selected_tool and self.focus is FocusPane.TOOLS else " "
            lines.append(f"{prefix} [{_normalize_status(event.status)}] {event.title}")
            lines.append(f"    cmd: {event.command}")
            if event.output:
                for line in _preview_lines(event.output, 3):
                    lines.append(f"    out: {line}")
            if event.approval_id:
                lines.append(f"    approval: {event.approval_id}")
            lines.append("")
        return lines[:-1] if lines else lines

    def render_approval_lines(self) -> list[str]:
        if not self.session.approvals:
            return ["No approval requests yet."]

        lines: list[str] = []
        for index, approval in enumerate(self.session.approvals):
            prefix = (
                ">" if index == self.selected_approval and self.focus is FocusPane.APPROVALS else " "
            )
            decision = f" ({approval.decision})" if approval.decision else ""
            lines.append(
                f"{prefix} [{_normalize_status(approval.status)}{decision}] {approval.title}"
            )
            lines.append(f"    cmd: {approval.command}")
            lines.append(f"    why: {approval.rationale}")
            lines.append("")
        return lines[:-1] if lines else lines

    def render_composer_mode(self) -> str:
        if self.input_mode is InputMode.APPROVAL_EDIT:
            approval = self.selected_approval_request()
            approval_label = approval.id if approval is not None else "n/a"
            return (
                "Composer mode: Approval Edit | "
                f"Selected approval: {approval_label} | "
                "Press Enter or Send to submit the edited command."
            )
        return "Composer mode: Prompt | Type a prompt and press Enter. Use Ctrl+S to save the editor."

    def render_files_meta(self) -> str:
        roots = self.session_info.filesystem.roots if self.session_info is not None else []
        root_text = ", ".join(Path(root).name or root for root in roots) if roots else "n/a"
        return f"Roots: {root_text} | Current: {self.workspace.current_directory or 'n/a'}"

    def render_editor_meta(self) -> str:
        if self.open_document is None:
            return "No file open. Select a file in the explorer to load it."
        dirty = "dirty" if self.workspace.editor_dirty else "saved"
        truncated = " | preview truncated" if self.open_document.truncated else ""
        return (
            f"{self.open_document.path} | {self.open_document.size} bytes | "
            f"{dirty}{truncated}"
        )

    def _sync_backend_task(self) -> None:
        backend_status = self.client.fetch_status()
        session_info = self.client.fetch_session_info()
        current_session_id = self.session.session_id or None
        envelope = self.client.open_session(self.config.profile, current_session_id)
        initial_directory = self.workspace.current_directory or _default_workspace_root(session_info)
        directory_listing = self.client.fetch_directory(initial_directory) if initial_directory else None
        document = None
        if self.workspace.open_file_path:
            document = self.client.fetch_file(self.workspace.open_file_path)
        self.call_from_thread(
            self._finish_sync_backend_task,
            backend_status,
            session_info,
            envelope.session,
            envelope.restored,
            directory_listing,
            document,
        )

    def _finish_sync_backend_task(
        self,
        backend_status: BackendStatus,
        session_info: SessionInfo,
        session: SessionState,
        restored: bool,
        directory_listing: DirectoryListing | None,
        document: FileDocument | None,
    ) -> None:
        self.backend_busy = False
        self.backend_status = backend_status
        self.session_info = session_info
        self.session = session
        if not self.session.profile:
            self.session.profile = self.config.profile
        self._initialize_workspace_from_session_info()
        if directory_listing is not None:
            self._apply_directory_listing(directory_listing)
        if document is not None:
            self._apply_open_document(document)
        self.normalize_selection()
        self.notice = (
            "Connected to backend and restored the active session."
            if restored
            else "Connected to backend and opened a new session."
        )
        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def _send_prompt_task(self, prompt: str, session_id: str) -> None:
        active_session_id = session_id
        if not active_session_id:
            envelope = self.client.open_session(self.config.profile, None)
            active_session_id = envelope.session.session_id
        envelope = self.client.send_prompt(active_session_id, prompt)
        self.call_from_thread(self._finish_send_prompt_task, envelope.session)

    def _finish_send_prompt_task(self, session: SessionState) -> None:
        self.backend_busy = False
        self.session = session
        self.draft = ""
        self.input_mode = InputMode.PROMPT
        self.query_one("#prompt-input", Input).value = ""
        self.notice = "Prompt sent to backend."
        self.normalize_selection()
        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def _resolve_approval_task(
        self,
        session_id: str,
        approval_id: str,
        decision: str,
        edited_command: str | None,
    ) -> None:
        envelope = self.client.review_approval(
            session_id,
            approval_id,
            decision,
            edited_command,
        )
        self.call_from_thread(
            self._finish_resolve_approval_task,
            envelope.session,
            decision,
        )

    def _finish_resolve_approval_task(self, session: SessionState, decision: str) -> None:
        self.backend_busy = False
        self.session = session
        self.input_mode = InputMode.PROMPT
        self.approval_edit_buffer = ""
        self.query_one("#prompt-input", Input).value = self.draft
        self.notice = {
            "approve": "Approval accepted.",
            "reject": "Approval rejected.",
            "edit": "Approval updated with the edited command.",
        }.get(decision, "Approval updated.")
        self.normalize_selection()
        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def _load_directory_task(self, path: str) -> None:
        listing = self.client.fetch_directory(path)
        self.call_from_thread(self._finish_load_directory_task, listing)

    def _finish_load_directory_task(self, listing: DirectoryListing) -> None:
        self.backend_busy = False
        self._apply_directory_listing(listing)
        self.notice = f"Loaded directory {listing.path}."
        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def _open_file_task(self, path: str) -> None:
        document = self.client.fetch_file(path)
        self.call_from_thread(self._finish_open_file_task, document)

    def _finish_open_file_task(self, document: FileDocument) -> None:
        self.backend_busy = False
        self._apply_open_document(document)
        self.notice = (
            f"Opened {document.path}."
            if not document.truncated
            else f"Opened {document.path} (preview truncated at backend limit)."
        )
        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def _save_file_task(self, path: str, content: str) -> None:
        document = self.client.save_file(path, content)
        self.call_from_thread(self._finish_save_file_task, document)

    def _finish_save_file_task(self, document: FileDocument) -> None:
        self.backend_busy = False
        self.open_document = document
        self.workspace.open_file_path = document.path
        self.workspace.editor_text = document.content
        self.workspace.editor_dirty = False
        self.notice = f"Saved {document.path}."
        self.refresh_screen()
        self._apply_focus()
        self.persist_state()

    def _initialize_workspace_from_session_info(self) -> None:
        roots = self.session_info.filesystem.roots if self.session_info is not None else []
        if not self.workspace.current_directory:
            self.workspace.current_directory = roots[0] if roots else ""
        self._rebuild_tree(roots)

    def _rebuild_tree(self, roots: list[str]) -> None:
        tree = self.query_one("#file-tree", Tree)
        tree.root.remove_children()
        tree.root.set_label("Workspace")
        tree.root.data = BrowserNode(path="", is_dir=True, loaded=True)
        for root_path in roots:
            node = tree.root.add(_tree_label(root_path, is_dir=True), data=BrowserNode(path=root_path, is_dir=True))
            node.allow_expand = True
        tree.root.expand()

    def _apply_directory_listing(self, listing: DirectoryListing) -> None:
        self.directory_cache[listing.path] = listing
        self.workspace.current_directory = listing.path
        node = self._find_tree_node(listing.path)
        if node is None:
            root = self.query_one("#file-tree", Tree).root
            node = root.add(_tree_label(listing.path, is_dir=True), data=BrowserNode(path=listing.path, is_dir=True))
        node.remove_children()
        data = node.data
        if isinstance(data, BrowserNode):
            data.loaded = True
        node.allow_expand = True
        for entry in listing.entries:
            child = node.add(
                _tree_label(entry.path, is_dir=entry.is_dir, name=entry.name),
                data=BrowserNode(path=entry.path, is_dir=entry.is_dir),
            )
            child.allow_expand = entry.is_dir
        node.expand()

    def _apply_open_document(self, document: FileDocument) -> None:
        self.open_document = document
        self.workspace.open_file_path = document.path
        self.workspace.editor_text = document.content
        self.workspace.editor_dirty = False
        editor = self.query_one("#editor", TextArea)
        self.loading_editor = True
        editor.load_text(document.content)
        editor.language = _language_for_path(document.path)
        self.loading_editor = False
        self.focus = FocusPane.EDITOR

    def _find_tree_node(self, path: str):
        tree = self.query_one("#file-tree", Tree)

        def walk(node):
            data = node.data
            if isinstance(data, BrowserNode) and data.path == path:
                return node
            for child in node.children:
                found = walk(child)
                if found is not None:
                    return found
            return None

        return walk(tree.root)

    def _apply_focus(self) -> None:
        if self.focus is FocusPane.FILES:
            self.query_one("#file-tree", Tree).focus()
            return
        if self.focus is FocusPane.EDITOR:
            self.query_one("#editor", TextArea).focus()
            return
        self.query_one("#prompt-input", Input).focus()


def run_textual_app(config: TuiConfig) -> None:
    HPCAssistantTextualApp(config).run()


def _move_index(app: HPCAssistantTextualApp, attr: str, size: int, delta: int) -> bool:
    if size == 0:
        return False
    current = getattr(app, attr)
    next_index = max(0, min(size - 1, current + delta))
    if next_index == current:
        return False
    setattr(app, attr, next_index)
    return True


def _clamp_index(index: int, size: int) -> int:
    if size <= 0:
        return 0
    return max(0, min(index, size - 1))


def _normalize_status(status: str) -> str:
    if status == "waiting_approval":
        return "waiting"
    if status == "completed":
        return "done"
    return status


def _preview_lines(text: str, limit: int) -> list[str]:
    lines = text.splitlines()
    preview = lines[:limit]
    if len(lines) > limit:
        preview.append("...")
    return preview


def _role_label(role: str) -> str:
    return {
        "assistant": "Assistant",
        "system": "System",
        "user": "You",
    }.get(role, role or "Unknown")


def _tree_label(path: str, *, is_dir: bool, name: str | None = None) -> str:
    label = name or Path(path).name or path
    return f"[D] {label}" if is_dir else f"[F] {label}"


def _default_workspace_root(session_info: SessionInfo | None) -> str | None:
    if session_info is None or not session_info.filesystem.roots:
        return None
    return session_info.filesystem.roots[0]


def _language_for_path(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".rs": "rust",
    }.get(suffix)


def _runtime_mode_label() -> str:
    py_gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    if py_gil_disabled == 1 and hasattr(sys, "_is_gil_enabled"):
        return "free-threaded (gil=off)" if not sys._is_gil_enabled() else "free-threaded build (gil=on)"
    return "standard build (gil=on)"
