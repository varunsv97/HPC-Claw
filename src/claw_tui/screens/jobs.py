"""Jobs screen — live Slurm queue view (squeue)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Label, RichLog

from claw_backend.config import load_settings

_SQUEUE_FORMAT = "%i|%j|%T|%M|%P|%D|%C|%u|%R"
_SQUEUE_HEADERS = ("Job ID", "Name", "State", "Time", "Partition", "Nodes", "CPUs", "User", "Reason")

_STATE_STYLE = {
    "RUNNING":   "green",
    "PENDING":   "yellow",
    "FAILED":    "red",
    "CANCELLED": "dim",
    "COMPLETED": "cyan",
    "TIMEOUT":   "magenta",
}


def _squeue(timeout: float) -> list[list[str]]:
    result = subprocess.run(
        ["squeue", "--noheader", f"-o{_SQUEUE_FORMAT}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= len(_SQUEUE_HEADERS):
            rows.append(parts[: len(_SQUEUE_HEADERS)])
    return rows


class JobsScreen(Screen):
    """Shows the current Slurm queue and a detail pane for the selected job."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
    ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Active Jobs  (squeue)", classes="section-title"),
            DataTable(id="jobs-table", cursor_type="row"),
            Label("", id="jobs-status", classes="empty-hint"),
            RichLog(id="job-detail", highlight=True, markup=True, wrap=True),
        )

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        for h in _SQUEUE_HEADERS:
            table.add_column(h, key=h)
        self.action_refresh()

    def action_refresh(self) -> None:
        settings = load_settings()
        table = self.query_one("#jobs-table", DataTable)
        status = self.query_one("#jobs-status", Label)
        table.clear()

        try:
            rows = _squeue(settings.command_timeout_seconds)
        except FileNotFoundError:
            status.update("[dim]squeue not found — not running on a Slurm cluster[/dim]")
            return
        except subprocess.TimeoutExpired:
            status.update("[red]squeue timed out[/red]")
            return
        except Exception as exc:
            status.update(f"[red]squeue error: {exc}[/red]")
            return

        if not rows:
            status.update("[dim]No jobs in queue[/dim]")
            return

        status.update("")
        for parts in rows:
            state = parts[2].strip() if len(parts) > 2 else ""
            style = _STATE_STYLE.get(state, "")
            styled = [f"[{style}]{p}[/{style}]" if style else p for p in parts]
            table.add_row(*styled)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show scontrol detail for the selected job."""
        table = self.query_one("#jobs-table", DataTable)
        detail = self.query_one("#job-detail", RichLog)
        detail.clear()

        try:
            row_data = [str(table.get_cell_at((event.cursor_row, c))) for c in range(len(_SQUEUE_HEADERS))]
        except Exception:
            return

        # Strip markup to get raw job_id
        job_id = row_data[0].strip()
        for tag in ("<", ">", "[", "]"):
            job_id = job_id.replace(tag, " ")
        job_id = job_id.split()[-1] if job_id.split() else ""

        settings = load_settings()
        try:
            proc = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True,
                check=False,
                timeout=settings.command_timeout_seconds,
            )
            output = proc.stdout.strip() or proc.stderr.strip() or "(no output)"
        except Exception as exc:
            output = f"scontrol error: {exc}"

        for line in output.splitlines():
            detail.write(line)

    def action_cursor_down(self) -> None:
        self.query_one("#jobs-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#jobs-table", DataTable).action_cursor_up()
