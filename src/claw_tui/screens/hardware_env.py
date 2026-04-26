"""Hardware & Environment screen — full cluster topology and software modules viewer.

Populated after the first `hclaw discover-cluster` / cluster-discovery run.
Provides an in-TUI "Discover Now" action so the user can trigger discovery
without leaving the interface.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Label, RichLog

from claw_backend.config import load_settings
from claw_backend.pgoa.store import ExperimentStore


class HardwareEnvScreen(Screen):
    """Two-pane view: hardware topology (left) + software environment (right)."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("d", "discover", "Discover Now"),
    ]

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Vertical(
                Label("Hardware Topology", classes="pane-title"),
                RichLog(id="hw-log", highlight=True, markup=True, wrap=True),
                id="hw-pane",
            ),
            Vertical(
                Label("Software Environment", classes="pane-title"),
                RichLog(id="env-log", highlight=True, markup=True, wrap=True),
                id="env-pane",
            ),
            id="hwenv-container",
        )

    def on_mount(self) -> None:
        self.action_refresh()

    def action_refresh(self) -> None:
        self._populate()

    def action_discover(self) -> None:
        """Trigger cluster discovery (login-node only, no probe jobs)."""
        hw_log = self.query_one("#hw-log", RichLog)
        hw_log.clear()
        hw_log.write("[yellow]Running cluster discovery…[/yellow]")
        self.app.call_after_refresh(self._run_discovery)

    def _run_discovery(self) -> None:
        from claw_backend.pgoa.services import discover_cluster
        settings = load_settings()
        store_path = Path(settings.pgoa_store_path).expanduser()
        store = ExperimentStore(store_path)
        try:
            discover_cluster(settings, store, force_refresh=True)
        except Exception as exc:
            hw_log = self.query_one("#hw-log", RichLog)
            hw_log.clear()
            hw_log.write(f"[red]Discovery failed: {exc}[/red]")
            return
        self._populate()

    def _populate(self) -> None:
        hw_log = self.query_one("#hw-log", RichLog)
        env_log = self.query_one("#env-log", RichLog)
        hw_log.clear()
        env_log.clear()

        settings = load_settings()
        store_path = Path(settings.pgoa_store_path).expanduser()
        store = ExperimentStore(store_path)

        try:
            from claw_backend.pgoa.cluster_discovery import detect_cluster_name
            cluster_name = detect_cluster_name(settings)
        except Exception as exc:
            hw_log.write(f"[red]Cannot detect cluster: {exc}[/red]")
            env_log.write("[dim]No data[/dim]")
            return

        profile = store.load_cluster_profile(cluster_name)
        if profile is None:
            hw_log.write(f"[dim]No profile cached for '[cyan]{cluster_name}[/cyan]'[/dim]")
            hw_log.write("")
            hw_log.write("Press [bold]d[/bold] to run discovery now.")
            env_log.write("[dim]Run discovery first.[/dim]")
            return

        # ── Hardware pane ──────────────────────────────────────────────────
        hw = profile.login_node_hardware
        hw_log.write(f"[bold]Cluster[/bold]   [cyan]{profile.cluster_name}[/cyan]")
        hw_log.write(f"[dim]Discovered {profile.discovered_at.strftime('%Y-%m-%d %H:%M UTC')}[/dim]")
        hw_log.write("")

        hw_log.write("[bold]CPU[/bold]")
        hw_log.write(f"  Model   {hw.cpu_model or '?'}")
        hw_log.write(f"  Arch    {hw.cpu_arch or '?'}")
        hw_log.write(
            f"  Cores   {hw.cores_per_socket}c × {hw.sockets_per_node} sockets"
            f" × {hw.threads_per_core} threads = [cyan]{hw.cpus_per_node or '?'}[/cyan] logical CPUs"
        )
        hw_log.write(f"  NUMA    {hw.numa_nodes} nodes")
        if hw.cache_l3_mb:
            hw_log.write(f"  L3      {hw.cache_l3_mb} MB")
        hw_log.write(f"  RAM     {hw.memory_gb_per_node or '?'} GB / node")

        if hw.gpu_model:
            hw_log.write("")
            hw_log.write("[bold]GPU[/bold]")
            hw_log.write(f"  Model   {hw.gpu_model}")
            hw_log.write(f"  Arch    {hw.gpu_arch or '?'}")
            hw_log.write(f"  Memory  {hw.gpu_memory_gb or '?'} GB")
            hw_log.write(f"  SMs     {hw.gpu_sm_count or '?'}")
            hw_log.write(f"  CC      {hw.gpu_compute_capability or '?'}")

        if hw.network_fabric:
            hw_log.write("")
            hw_log.write(f"[bold]Network[/bold]  {hw.network_fabric}")

        hw_log.write("")
        hw_log.write(f"[bold]Partitions[/bold]  ({len(profile.partitions)})")
        for p in profile.partitions:
            probe_tag = "[green]✔ probed[/green]" if p.hardware_from_probe else "[dim]estimated[/dim]"
            hw_log.write(
                f"  [cyan]{p.name:<18}[/cyan]"
                f"  total={p.total_nodes or '?':>4}"
                f"  idle={p.idle_nodes or '?':>4}"
                f"  {probe_tag}"
            )
            phw = p.hardware
            if phw.gpu_model:
                hw_log.write(f"    GPU {phw.gpu_model}  {phw.gpu_memory_gb} GB")

        # ── Software env pane ──────────────────────────────────────────────
        se = profile.software_env
        if se is None:
            env_log.write("[dim]No software environment data — run discovery.[/dim]")
            return

        env_log.write(f"[bold]Module system[/bold]  {se.module_system}", )
        if se.lmod_version:
            env_log.write(f"  Lmod    {se.lmod_version}")
        if se.tmod_version:
            env_log.write(f"  TCL     {se.tmod_version}")
        spider_note = "full hierarchy" if se.spider_complete else "Core tier only"
        env_log.write(f"  Scope   {spider_note}")

        sections = [
            ("Compilers",   se.compilers),
            ("MPI",         se.mpi_libraries),
            ("GPU toolkits",se.gpu_toolkits),
            ("Math libs",   se.math_libraries),
            ("Debuggers",   se.debuggers),
            ("Profilers",   se.profilers),
        ]
        for title, mods in sections:
            if not mods:
                continue
            env_log.write("")
            env_log.write(f"[bold]{title}[/bold]  ({len(mods)})")
            for m in mods[:12]:
                ver = m.default_version or (m.versions[0] if m.versions else "")
                ver_str = f"  [dim]{ver}[/dim]" if ver else ""
                env_log.write(f"  [cyan]{m.name}[/cyan]{ver_str}")
            if len(mods) > 12:
                env_log.write(f"  [dim]… and {len(mods) - 12} more[/dim]")

        if se.toolchains:
            env_log.write("")
            env_log.write(f"[bold]Toolchains[/bold]  ({len(se.toolchains)})")
            for tc in se.toolchains[:10]:
                seq = " → ".join(tc.load_sequence[:4])
                if len(tc.load_sequence) > 4:
                    seq += " …"
                env_log.write(f"  [cyan]{tc.name:<20}[/cyan]  {seq}")
            if len(se.toolchains) > 10:
                env_log.write(f"  [dim]… and {len(se.toolchains) - 10} more[/dim]")
