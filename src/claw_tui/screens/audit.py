"""Edit Audit screen — browse all code edits, diffs, hypotheses, and verdicts."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Label, RichLog, Static

from claw_backend.config import load_settings
from claw_backend.pgoa.schema import EditRecord, MetricsEditMap
from claw_backend.pgoa.store import ExperimentStore


def _diff_markup(diff: str | None) -> str:
    """Convert a unified diff string to Rich markup with +/- colouring."""
    if not diff:
        return "[dim]No diff captured.[/]"
    lines = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            lines.append(f"[bold]{line}[/]")
        elif line.startswith("+"):
            lines.append(f"[green]{line}[/]")
        elif line.startswith("-"):
            lines.append(f"[red]{line}[/]")
        elif line.startswith("@@"):
            lines.append(f"[cyan]{line}[/]")
        else:
            lines.append(line)
    return "\n".join(lines)


class EditAuditScreen(Screen):
    """Table of all code edits across all workloads, with diff viewer."""

    BINDINGS = [("r", "refresh", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self._store: ExperimentStore | None = None
        # (workload_id, edit_id) → EditRecord
        self._edit_index: dict[str, tuple[str, EditRecord]] = {}

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Code Edit Audit Trail", classes="section-title"),
            DataTable(id="edits-table", cursor_type="row", zebra_stripes=True),
            Label("Hypothesis / diff", classes="section-title"),
            RichLog(id="diff-viewer", highlight=True, markup=True, wrap=False),
            id="audit-container",
        )

    def on_mount(self) -> None:
        self._init_table()
        self.action_refresh()

    def _init_table(self) -> None:
        table = self.query_one("#edits-table", DataTable)
        table.add_columns(
            "Edit ID", "Workload", "Bottleneck", "# files", "Verdict", "Timestamp",
        )

    def action_refresh(self) -> None:
        settings = load_settings()
        store_path = Path(settings.pgoa_store_path).expanduser()
        self._store = ExperimentStore(store_path)
        self._edit_index = {}

        table = self.query_one("#edits-table", DataTable)
        table.clear()

        if not store_path.exists():
            return

        for entry in sorted(store_path.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            wid = entry.name
            edit_map = self._store.load_edit_map(wid)
            verdicts: dict[str, str] = {}
            if edit_map:
                for em_entry in edit_map.entries:
                    if em_entry.delta is not None:
                        verdicts[em_entry.edit_id] = em_entry.delta.kpi_direction
            for rec in self._store.list_edit_records(wid):
                verdict = verdicts.get(rec.edit_id, "—")
                verdict_markup = {"improved": "✅ improved", "degraded": "❌ degraded", "neutral": "→ neutral"}.get(verdict, verdict)
                key = f"{wid}:{rec.edit_id}"
                self._edit_index[key] = (wid, rec)
                table.add_row(
                    rec.edit_id[:10] + "…",
                    wid,
                    rec.bottleneck_type,
                    str(len(rec.files_modified)),
                    verdict_markup,
                    rec.timestamp.strftime("%Y-%m-%d %H:%M"),
                    key=key,
                )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = event.row_key.value
        if not key or key not in self._edit_index:
            return
        wid, rec = self._edit_index[key]
        self._show_edit(wid, rec)

    def _show_edit(self, wid: str, rec: EditRecord) -> None:
        log = self.query_one("#diff-viewer", RichLog)
        log.clear()

        log.write(f"[bold]Edit[/]  {rec.edit_id}")
        log.write(f"[cyan]Workload[/]    {wid}")
        log.write(f"[cyan]Timestamp[/]   {rec.timestamp.isoformat()}")
        log.write(f"[cyan]Bottleneck[/]  {rec.bottleneck_type}")
        log.write(f"[cyan]Before run[/]  {rec.linked_run_id_before}")
        if rec.linked_run_id_after:
            log.write(f"[cyan]After run[/]   {rec.linked_run_id_after}")
        if rec.opencode_session_id:
            log.write(f"[cyan]OC session[/]  {rec.opencode_session_id}")

        log.write("")
        log.write("[bold]Hypothesis[/]")
        log.write(f"  {rec.hypothesis}")

        log.write("")
        log.write("[bold]OpenCode prompt sent[/]")
        for line in rec.prompt_sent_to_opencode.splitlines():
            log.write(f"  {line}")

        if rec.files_modified:
            log.write("")
            log.write("[bold]Files modified[/]")
            for f in rec.files_modified:
                log.write(f"  [green]{f}[/]")

        log.write("")
        log.write("[bold]Git diff[/]")
        diff_text = _diff_markup(rec.git_diff)
        for line in diff_text.splitlines():
            log.write(line)
