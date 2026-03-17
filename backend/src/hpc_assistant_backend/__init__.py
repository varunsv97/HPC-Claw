"""Public package surface for the HPC assistant backend."""

from hpc_assistant_backend.cli import main
from hpc_assistant_backend.config import (
    AssistantSettings,
    BackendConfig,
    ExecutionPolicy,
    default_config_path,
    load_config,
    load_settings,
    render_config,
    write_config,
)
from hpc_assistant_backend.runtime import (
    AgentTurnResult,
    AssistantRuntime,
    AssistantSession,
    PendingInterrupt,
    TurnStatus,
)
from hpc_assistant_backend.tools import ToolRegistry, ToolSpec, build_default_registry

__all__ = [
    "AgentTurnResult",
    "AssistantRuntime",
    "AssistantSession",
    "AssistantSettings",
    "BackendConfig",
    "ExecutionPolicy",
    "PendingInterrupt",
    "ToolRegistry",
    "ToolSpec",
    "TurnStatus",
    "build_default_registry",
    "default_config_path",
    "load_config",
    "load_settings",
    "main",
    "render_config",
    "write_config",
]
