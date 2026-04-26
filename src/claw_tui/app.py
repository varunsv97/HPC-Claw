"""HPC Claw TUI — Textual application."""

from __future__ import annotations

from importlib.resources import files
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Label, ListItem, ListView

from claw_tui.screens.audit import EditAuditScreen
from claw_tui.screens.chat import ChatScreen
from claw_tui.screens.cluster import ClusterScreen
from claw_tui.screens.dashboard import DashboardScreen
from claw_tui.screens.explorer import ExplorerScreen
from claw_tui.screens.hardware_env import HardwareEnvScreen
from claw_tui.screens.jobs import JobsScreen
from claw_tui.screens.settings import SettingsScreen
from claw_tui.screens.workloads import WorkloadsScreen

_NAV_ITEMS = [
    ("1", "Dashboard"),
    ("2", "Workloads"),
    ("3", "Edit Audit"),
    ("4", "Cluster"),
    ("5", "Settings"),
    ("6", "Jobs"),
    ("7", "Explorer"),
    ("8", "Chat"),
    ("9", "Hardware/Env"),
]

_SCREEN_MAP: dict[str, type] = {
    "Dashboard":    DashboardScreen,
    "Workloads":    WorkloadsScreen,
    "Edit Audit":   EditAuditScreen,
    "Cluster":      ClusterScreen,
    "Settings":     SettingsScreen,
    "Jobs":         JobsScreen,
    "Explorer":     ExplorerScreen,
    "Chat":         ChatScreen,
    "Hardware/Env": HardwareEnvScreen,
}


class ClawTUI(App):
    """HPC Claw terminal user interface."""

    CSS_PATH = "app.tcss"

    TITLE = "HPC Claw"
    SUB_TITLE = "Performance Optimization Assistant"

    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("1", "nav_dashboard", "Dashboard", show=False),
        Binding("2", "nav_workloads", "Workloads", show=False),
        Binding("3", "nav_audit", "Edit Audit", show=False),
        Binding("4", "nav_cluster", "Cluster", show=False),
        Binding("5", "nav_settings", "Settings", show=False),
        Binding("6", "nav_jobs", "Jobs", show=False),
        Binding("7", "nav_explorer", "Explorer", show=False),
        Binding("8", "nav_chat", "Chat", show=False),
        Binding("9", "nav_hwenv", "HW/Env", show=False),
        Binding("r", "refresh_screen", "Refresh", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen())

    # ── Navigation actions ──────────────────────────────────────────────

    def action_nav_dashboard(self) -> None:
        self._switch_to(DashboardScreen)

    def action_nav_workloads(self) -> None:
        self._switch_to(WorkloadsScreen)

    def action_nav_audit(self) -> None:
        self._switch_to(EditAuditScreen)

    def action_nav_cluster(self) -> None:
        self._switch_to(ClusterScreen)

    def action_nav_settings(self) -> None:
        self._switch_to(SettingsScreen)

    def action_nav_jobs(self) -> None:
        self._switch_to(JobsScreen)

    def action_nav_explorer(self) -> None:
        self._switch_to(ExplorerScreen)

    def action_nav_chat(self) -> None:
        self._switch_to(ChatScreen)

    def action_nav_hwenv(self) -> None:
        self._switch_to(HardwareEnvScreen)

    def action_refresh_screen(self) -> None:
        screen = self.screen
        if hasattr(screen, "action_refresh"):
            screen.action_refresh()  # type: ignore[union-attr]

    def _switch_to(self, screen_cls: type) -> None:
        # Replace entire stack with a fresh instance of the target screen
        while len(self.screen_stack) > 0:
            try:
                self.pop_screen()
            except Exception:
                break
        self.push_screen(screen_cls())
