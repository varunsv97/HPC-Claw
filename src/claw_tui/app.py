"""HPC Claw TUI — Textual application."""

from __future__ import annotations

from importlib.resources import files
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Label, ListItem, ListView

from claw_tui.screens.audit import EditAuditScreen
from claw_tui.screens.cluster import ClusterScreen
from claw_tui.screens.dashboard import DashboardScreen
from claw_tui.screens.settings import SettingsScreen
from claw_tui.screens.workloads import WorkloadsScreen

_NAV_ITEMS = [
    ("1", "Dashboard"),
    ("2", "Workloads"),
    ("3", "Edit Audit"),
    ("4", "Cluster"),
    ("5", "Settings"),
]

_SCREEN_MAP: dict[str, type] = {
    "Dashboard": DashboardScreen,
    "Workloads": WorkloadsScreen,
    "Edit Audit": EditAuditScreen,
    "Cluster": ClusterScreen,
    "Settings": SettingsScreen,
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
