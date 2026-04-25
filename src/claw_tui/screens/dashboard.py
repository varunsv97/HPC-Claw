"""Dashboard screen — overview of workloads, recent activity, and store health."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Label, RichLog, Static

from claw_backend.config import load_settings
from claw_backend.pgoa.store import ExperimentStore


class StatCard(Static):
    """A small metric tile."""

    def __init__(self, label: str, value: str, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="stat-label")
        yield Label(self._value, classes="stat-value")

    def update_value(self, value: str) -> None:
        self.query_one(".stat-value", Label).update(value)


class DashboardScreen(Screen):
    """Top-level dashboard showing store stats and recent run activity."""

    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Container(
            Container(
                StatCard("Workloads", "—", id="stat-workloads"),
                StatCard("Total runs", "—", id="stat-runs"),
                StatCard("Code edits", "—", id="stat-edits"),
                StatCard("Store path", "—", id="stat-store"),
                id="stat-row",
            ),
            Container(
                Label("Recent activity", classes="section-title"),
                RichLog(id="activity-log", highlight=True, markup=True, wrap=True),
                id="recent-activity",
            ),
            id="dashboard-grid",
        )

    def on_mount(self) -> None:
        self.action_refresh()

    def action_refresh(self) -> None:
        settings = load_settings()
        store_path = Path(settings.pgoa_store_path).expanduser()
        store = ExperimentStore(store_path)

        workload_ids: list[str] = []
        total_runs = 0
        total_edits = 0
        activity_lines: list[str] = []

        if store_path.exists():
            for entry in sorted(store_path.iterdir()):
                if entry.is_dir() and not entry.name.startswith("_"):
                    wid = entry.name
                    workload_ids.append(wid)
                    runs = store.list_runs(wid)
                    total_runs += len(runs)
                    edits = store.list_edit_records(wid)
                    total_edits += len(edits)
                    for r in runs[-3:]:
                        try:
                            bundle = store.load_bundle(wid, r.run_id)
                            kpi = bundle.kpi
                            activity_lines.append(
                                f"[cyan]{wid}[/] › {r.run_type} "
                                f"[bold]{kpi.value:.3g} {kpi.unit}[/] "
                                f"([dim]{r.run_id[:8]}…[/])"
                            )
                        except Exception:
                            activity_lines.append(
                                f"[cyan]{wid}[/] › {r.run_type} [dim](profile unreadable)[/]"
                            )

        self.query_one("#stat-workloads", StatCard).update_value(str(len(workload_ids)))
        self.query_one("#stat-runs", StatCard).update_value(str(total_runs))
        self.query_one("#stat-edits", StatCard).update_value(str(total_edits))
        self.query_one("#stat-store", StatCard).update_value(str(store_path))

        log = self.query_one("#activity-log", RichLog)
        log.clear()
        if activity_lines:
            for line in reversed(activity_lines[-30:]):
                log.write(line)
        else:
            log.write("[dim]No runs found in store.[/]")
