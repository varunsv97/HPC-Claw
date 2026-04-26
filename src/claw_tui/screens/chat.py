"""Chat screen — conversational interface backed by the configured OpenAI model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Input, Label, RichLog

from claw_backend.config import load_settings

_SYSTEM_PROMPT = (
    "You are HPC Claw, an expert HPC performance engineering assistant. "
    "You help users understand profiling data, Slurm configurations, GPU/CPU bottlenecks, "
    "and code optimization strategies for MPI, OpenMP, and CUDA workloads. "
    "Keep responses concise and actionable."
)


@dataclass
class _ChatMessage:
    role: str   # "user" | "assistant" | "system"
    content: str


class ChatScreen(Screen):
    """Conversational chat interface with the configured LLM."""

    BINDINGS: ClassVar = [
        Binding("ctrl+c", "clear_chat", "Clear"),
        Binding("escape", "blur_input", "Defocus input"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._history: list[_ChatMessage] = [
            _ChatMessage("system", _SYSTEM_PROMPT)
        ]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Chat  [dim](Enter to send · Ctrl+C to clear)[/dim]", classes="section-title"),
            RichLog(
                id="chat-log",
                highlight=True,
                markup=True,
                wrap=True,
            ),
            Input(placeholder="Ask about your workload…", id="chat-input"),
            id="chat-container",
        )

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        log = self.query_one("#chat-log", RichLog)
        log.write("[dim]HPC Claw is ready. Ask about profiling, bottlenecks, or Slurm tuning.[/dim]")
        settings = load_settings()
        if not settings.openai_api_key:
            log.write(
                "[yellow]⚠ HPC_ASSISTANT_OPENAI_API_KEY is not set — "
                "responses will be unavailable.[/yellow]"
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()
        self._send(text)

    def _send(self, user_text: str) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(f"\n[bold cyan]You[/bold cyan]  {user_text}")
        self._history.append(_ChatMessage("user", user_text))

        settings = load_settings()
        if not settings.openai_api_key:
            log.write("[red]No API key configured.[/red]")
            return

        log.write("[dim]…thinking…[/dim]")
        self.app.call_after_refresh(self._fetch_reply, settings)

    def _fetch_reply(self, settings) -> None:  # noqa: ANN001
        log = self.query_one("#chat-log", RichLog)
        # Remove the "…thinking…" line by clearing and replaying (simple approach)
        try:
            from claw_backend.openai_client import build_openai_client
            client = build_openai_client(settings)
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": m.role, "content": m.content} for m in self._history],
                timeout=settings.openai_timeout_seconds,
            )
            reply = response.choices[0].message.content or ""
        except Exception as exc:
            reply = f"[red]Error: {exc}[/red]"

        self._history.append(_ChatMessage("assistant", reply))
        # Redraw the full history cleanly
        self._redraw_log()

    def _redraw_log(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        log.write("[dim]HPC Claw is ready. Ask about profiling, bottlenecks, or Slurm tuning.[/dim]")
        for msg in self._history:
            if msg.role == "system":
                continue
            if msg.role == "user":
                log.write(f"\n[bold cyan]You[/bold cyan]  {msg.content}")
            else:
                log.write(f"\n[bold green]Claw[/bold green]  {msg.content}")

    def action_clear_chat(self) -> None:
        self._history = [_ChatMessage("system", _SYSTEM_PROMPT)]
        log = self.query_one("#chat-log", RichLog)
        log.clear()
        log.write("[dim]Chat cleared.[/dim]")

    def action_blur_input(self) -> None:
        self.query_one("#chat-input", Input).blur()
