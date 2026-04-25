"""Settings screen — display and copy active AssistantSettings."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label, RichLog

from claw_backend.config import load_settings


_SENSITIVE = {"openai_api_key"}


class SettingsScreen(Screen):
    """Read-only view of the active AssistantSettings (env vars)."""

    BINDINGS = [("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Active Settings  (HPC_ASSISTANT_* env vars)", classes="section-title"),
            RichLog(id="settings-log", highlight=True, markup=True, wrap=True),
        )

    def on_mount(self) -> None:
        self.action_refresh()

    def action_refresh(self) -> None:
        log = self.query_one("#settings-log", RichLog)
        log.clear()
        settings = load_settings()

        for field_name in settings.model_fields:
            raw = getattr(settings, field_name)
            env_var = f"HPC_ASSISTANT_{field_name.upper()}"
            if field_name in _SENSITIVE and raw:
                display = "[dim]<set>[/]"
            elif raw is None:
                display = "[dim]None[/]"
            elif isinstance(raw, tuple):
                display = ", ".join(str(v) for v in raw) if raw else "[dim](empty)[/]"
            else:
                display = str(raw)
            log.write(f"[cyan]{env_var:<45}[/]  {display}")

        log.write("")
        log.write("[dim]Edit these in your .env file or export them in your shell.[/]")
        log.write("[dim]Changes take effect on next hpc-claw-tui launch.[/]")
