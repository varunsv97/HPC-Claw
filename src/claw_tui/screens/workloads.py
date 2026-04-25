"""Workloads screen — browse workloads, runs, KPI deltas, and bottleneck reports."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Label, ListItem, ListView, RichLog, Static

from claw_backend.config import load_settings
from claw_backend.pgoa.store import ExperimentStore
from claw_backend.pgoa.analysis import analyze_bottlenecks


_DIRECTION_STYLE = {
    "improved": "[green]▲ improved[/]",
    "degraded": "[red]▼ degraded[/]",
    "neutral": "[dim]→ neutral[/]",
}

_BOTTLENECK_STYLE = {
    "memory_bound_gpu": "[yellow]⚡ memory_bound_gpu[/]",
    "compute_bound_gpu": "[magenta]⚡ compute_bound_gpu[/]",
    "latency_bound_gpu": "[yellow]⏱ latency_bound_gpu[/]",
    "cpu_memory_bound": "[yellow]⚡ cpu_memory_bound[/]",
    "cpu_compute_bound": "[magenta]⚡ cpu_compute_bound[/]",
    "mpi_binding": "[cyan]🔗 mpi_binding[/]",
    "high_rss": "[red]💾 high_rss[/]",
    "none_detected": "[green]✔ none_detected[/]",
    "insufficient_data": "[dim]? insufficient_data[/]",
}


class WorkloadsScreen(Screen):
    """Two-pane view: workload list on the left, run detail table on the right."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("enter", "select_workload", "Select"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._store: ExperimentStore | None = None
        self._workload_ids: list[str] = []
        self._selected_wid: str | None = None

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Vertical(
                Label("Workloads", classes="pane-title"),
                ListView(id="workload-list"),
                id="workload-list-pane",
            ),
            Vertical(
                Label("Runs", classes="pane-title"),
                DataTable(id="runs-table", cursor_type="row", zebra_stripes=True),
                Label("Bottleneck / detail", classes="section-title"),
                RichLog(id="run-detail", highlight=True, markup=True, wrap=True),
                id="workload-detail-pane",
            ),
            id="workload-container",
        )

    def on_mount(self) -> None:
        self._init_tables()
        self.action_refresh()

    def _init_tables(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.add_columns(
            "Run", "Type", "Iter", f"KPI value", "KPI unit", "Δ%", "Direction", "Bottleneck",
        )

    def action_refresh(self) -> None:
        settings = load_settings()
        store_path = Path(settings.pgoa_store_path).expanduser()
        self._store = ExperimentStore(store_path)

        self._workload_ids = []
        if store_path.exists():
            for e in sorted(store_path.iterdir()):
                if e.is_dir() and not e.name.startswith("_"):
                    self._workload_ids.append(e.name)

        lv = self.query_one("#workload-list", ListView)
        lv.clear()
        for wid in self._workload_ids:
            lv.append(ListItem(Label(wid)))

        if self._workload_ids:
            self._load_workload(self._workload_ids[0])

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and idx < len(self._workload_ids):
            self._load_workload(self._workload_ids[idx])

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if self._store is None or self._selected_wid is None:
            return
        row_key = event.row_key.value
        if not row_key:
            return
        self._show_run_detail(self._selected_wid, row_key)

    def _load_workload(self, wid: str) -> None:
        if self._store is None:
            return
        self._selected_wid = wid
        table = self.query_one("#runs-table", DataTable)
        table.clear()

        runs = self._store.list_runs(wid)
        baseline_kpi: float | None = None

        for run in runs:
            try:
                bundle = self._store.load_bundle(wid, run.run_id)
            except Exception:
                continue

            kpi_val = bundle.kpi.value
            kpi_unit = bundle.kpi.unit
            iter_str = str(run.iteration) if run.iteration is not None else "—"

            # KPI delta vs baseline
            delta_str = "—"
            direction_str = "—"
            if run.run_type == "baseline":
                baseline_kpi = kpi_val
            elif baseline_kpi is not None:
                delta = (kpi_val - baseline_kpi) / abs(baseline_kpi) * 100.0 if baseline_kpi else 0.0
                delta_str = f"{delta:+.1f}%"
                if bundle.kpi.lower_is_better:
                    direction_str = "improved" if delta < -2 else ("degraded" if delta > 2 else "neutral")
                else:
                    direction_str = "improved" if delta > 2 else ("degraded" if delta < -2 else "neutral")

            # Quick bottleneck
            try:
                report = analyze_bottlenecks(bundle)
                bn = report.primary_bottleneck
            except Exception:
                bn = "?"

            table.add_row(
                run.run_id[:8] + "…",
                run.run_type,
                iter_str,
                f"{kpi_val:.4g}",
                kpi_unit,
                delta_str,
                direction_str,
                bn,
                key=run.run_id,
            )

        # clear detail
        self.query_one("#run-detail", RichLog).clear()
        if not runs:
            self.query_one("#run-detail", RichLog).write("[dim]No runs for this workload.[/]")

    def _show_run_detail(self, wid: str, run_id: str) -> None:
        if self._store is None:
            return
        log = self.query_one("#run-detail", RichLog)
        log.clear()
        try:
            bundle = self._store.load_bundle(wid, run_id)
        except Exception as exc:
            log.write(f"[red]Failed to load bundle: {exc}[/]")
            return

        kpi = bundle.kpi
        log.write(f"[bold]Run[/] {run_id}")
        log.write(f"[cyan]KPI[/]       {kpi.value:.6g} {kpi.unit}  (lower_is_better={kpi.lower_is_better})")
        log.write(f"[cyan]Timestamp[/] {bundle.timestamp.isoformat()}")
        log.write(f"[cyan]Workload[/]  {bundle.workload_type}")

        if bundle.slurm:
            s = bundle.slurm
            log.write("")
            log.write("[bold]Slurm[/]")
            log.write(f"  job_id    {s.job_id}   state {s.state}   exit {s.exit_code}")
            log.write(f"  elapsed   {s.elapsed_s:.1f}s   alloc_cpus {s.alloc_cpus}   nodes {s.alloc_nodes}")
            log.write(f"  max_rss   {s.max_rss_mb:.0f} MB   avg_cpu {s.avg_cpu_pct}%")

        if bundle.compute:
            c = bundle.compute
            log.write("")
            log.write("[bold]GPU compute[/]")
            log.write(f"  roofline  {c.roofline_position}")
            log.write(f"  sm_occ    {c.sm_occupancy_pct}%   mem_bw_util {c.memory_bw_utilization_pct}%")
            log.write(f"  FLOPS     {c.achieved_flops_tflops} TF  (peak {c.peak_flops_tflops} TF)")
            if c.top_kernels:
                log.write("  top kernels:")
                for k in c.top_kernels[:5]:
                    log.write(f"    {k.name[:40]:<40}  {k.duration_pct:.1f}%")

        if bundle.cpu_perf:
            p = bundle.cpu_perf
            log.write("")
            log.write("[bold]CPU perf (LIKWID)[/]")
            log.write(f"  DRAM BW   {p.memory_bw_dram_gbs} GB/s   IPC {p.ipc}")
            log.write(f"  DP FLOPS  {p.flops_dp_gflops} GF/s")

        try:
            report = analyze_bottlenecks(bundle)
            bn_pretty = _BOTTLENECK_STYLE.get(report.primary_bottleneck, report.primary_bottleneck)
            log.write("")
            log.write(f"[bold]Bottleneck[/] {bn_pretty}")
            log.write(f"  hint: {report.recommended_action_hint}")
        except Exception:
            pass

        # Edit records linked to this run
        edits = [e for e in self._store.list_edit_records(wid)
                 if e.linked_run_id_before == run_id or e.linked_run_id_after == run_id]
        if edits:
            log.write("")
            log.write("[bold]Linked code edits[/]")
            for ed in edits:
                log.write(f"  [{ed.edit_id[:8]}]  {ed.hypothesis[:80]}")
