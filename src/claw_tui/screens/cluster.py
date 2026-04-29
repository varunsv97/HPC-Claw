"""Cluster screen — show cached ClusterProfile: partitions, hardware, software env."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label, RichLog, Tree

from claw_backend.config import load_settings
from claw_backend.pgoa.store import ExperimentStore


class ClusterScreen(Screen):
    """Displays the cached ClusterProfile for the current cluster."""

    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Cluster Profile", classes="section-title"),
            RichLog(id="cluster-detail", highlight=True, markup=True, wrap=True),
        )

    def on_mount(self) -> None:
        self.action_refresh()

    def action_refresh(self) -> None:
        from claw_backend.cluster_probes.hardware_inventory import detect_cluster_name

        log = self.query_one("#cluster-detail", RichLog)
        log.clear()

        settings = load_settings()
        store_path = Path(settings.pgoa_store_path).expanduser()
        store = ExperimentStore(store_path)

        try:
            cluster_name = detect_cluster_name(settings)
        except Exception as exc:
            log.write(f"[red]Could not detect cluster name: {exc}[/]")
            return

        profile = store.load_cluster_profile(cluster_name)
        if profile is None:
            log.write(f"[dim]No cached profile for cluster '[cyan]{cluster_name}[/]'.[/]")
            log.write("[dim]Run [bold]hpc-claw discover-cluster[/] to populate it.[/]")
            return

        hw = profile.login_node_hardware
        log.write(f"[bold]Cluster[/]      {profile.cluster_name}")
        log.write(f"[cyan]Discovered[/]   {profile.discovered_at.strftime('%Y-%m-%d %H:%M UTC')}")
        log.write("")
        log.write("[bold]Login-node hardware[/]")
        log.write(f"  CPU   {hw.cpu_model or '?'} ({hw.cpu_arch or '?'})")
        log.write(f"  Cores {hw.cores_per_socket}c × {hw.sockets_per_node} sockets  "
                  f"threads/core={hw.threads_per_core}  NUMA={hw.numa_nodes}")
        log.write(f"  RAM   {hw.memory_gb_per_node} GB")
        if hw.gpu_model:
            log.write(f"  GPU   {hw.gpu_model}  {hw.gpu_memory_gb} GB  "
                      f"SMs={hw.gpu_sm_count}  CC={hw.gpu_compute_capability}")

        log.write("")
        log.write(f"[bold]Partitions[/]  ({len(profile.partitions)} total)")
        for p in profile.partitions:
            probe = "[green]✔ probed[/]" if p.hardware_from_probe else "[dim]login-node approx[/]"
            log.write(f"  [cyan]{p.name:<20}[/]  nodes={p.total_nodes}  idle={p.idle_nodes}  {probe}")
            phw = p.hardware
            if phw.gpu_model:
                log.write(f"    GPU  {phw.gpu_model}  {phw.gpu_memory_gb} GB")
            if phw.cpu_model:
                log.write(f"    CPU  {phw.cpu_model}")

        if profile.software_env:
            env = profile.software_env
            log.write("")
            log.write(f"[bold]Software environment[/]  ({env.module_system})")
            extra_contexts = [c for c in env.module_contexts if c.name != "active"]
            if extra_contexts:
                names = ", ".join(c.name for c in extra_contexts[:6])
                extra = f" +{len(extra_contexts)-6} more" if len(extra_contexts) > 6 else ""
                log.write(f"  [cyan]{'Module tiers':<14}[/] {names}{extra}")
            for label, mods in [
                ("Compilers", env.compilers),
                ("MPI", env.mpi_libraries),
                ("GPU toolkits", env.gpu_toolkits),
                ("Math libs", env.math_libraries),
                ("Debuggers", env.debuggers),
                ("Profilers", env.profilers),
            ]:
                if mods:
                    names = ", ".join(m.name for m in mods[:8])
                    extra = f" +{len(mods)-8} more" if len(mods) > 8 else ""
                    log.write(f"  [cyan]{label:<14}[/] {names}{extra}")
            if env.toolchains:
                log.write("")
                log.write(f"  [bold]Toolchains[/]  ({len(env.toolchains)})")
                for tc in env.toolchains[:8]:
                    log.write(f"    {tc.name}")
