"""Public package surface for the HPC assistant backend."""

from hpc_assistant_backend.cli import main
from hpc_assistant_backend.config import AssistantSettings, load_settings
from hpc_assistant_backend.opencode_tools import execute_tool, sync_opencode_project, tool_manifest

__all__ = [
    "AssistantSettings",
    "execute_tool",
    "load_settings",
    "main",
    "sync_opencode_project",
    "tool_manifest",
]
