"""Public package surface for the HPC Claw backend."""

from claw_backend.assistant.service import build_assistant_session
from claw_backend.cli import main
from claw_backend.config import AssistantSettings, load_settings
from claw_backend.opencode_tools import execute_tool, sync_opencode_project, tool_manifest

__all__ = [
    "AssistantSettings",
    "build_assistant_session",
    "execute_tool",
    "load_settings",
    "main",
    "sync_opencode_project",
    "tool_manifest",
]
