"""Policy-aware custom tool registration for the backend."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
import subprocess
from time import perf_counter
from typing import Any, Callable, Sequence

from langchain_core.tools import BaseTool, tool

from hpc_assistant_backend.config import AssistantSettings, ExecutionPolicy
from hpc_assistant_backend.filesystem import (
    configured_filesystem_roots,
    resolve_allowed_path,
)


ToolFactory = Callable[[AssistantSettings, "ToolRegistry"], BaseTool]
InterruptConfig = bool | dict[str, object]
DEEPAGENTS_MUTATING_TOOLS = ("edit_file", "execute", "write_file")

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._:@%+=/-]+$")
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_MODULE = re.compile(r"^[A-Za-z0-9._+:/-]+$")
_SAFE_SIGNAL = re.compile(r"^[A-Z0-9]+$")
_SAFE_USER = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Single tool registration entry."""

    name: str
    policy: ExecutionPolicy
    description: str
    factory: ToolFactory
    interrupt_config: InterruptConfig | None = None

    def manifest_entry(self) -> dict[str, object]:
        return {
            "name": self.name,
            "policy": self.policy.value,
            "description": self.description,
            "interrupt_config": self.interrupt_config,
        }


class ToolRegistry:
    """Registry used to resolve custom tools for a given policy."""

    def __init__(self, specs: Sequence[ToolSpec] | None = None) -> None:
        self._specs = list(specs or [])

    def register(self, spec: ToolSpec) -> None:
        self._specs.append(spec)

    def resolve(self, settings: AssistantSettings) -> list[BaseTool]:
        return [spec.factory(settings, self) for spec in self._allowed_specs(settings)]

    def active_names(self, settings: AssistantSettings) -> list[str]:
        return [spec.name for spec in self._allowed_specs(settings)]

    def approval_required_specs(self) -> list[ToolSpec]:
        return [spec for spec in self._specs if spec.policy is ExecutionPolicy.APPROVAL_REQUIRED]

    def manifest(self) -> list[dict[str, object]]:
        return [spec.manifest_entry() for spec in self._specs]

    def _allowed_specs(self, settings: AssistantSettings) -> list[ToolSpec]:
        allowed = {ExecutionPolicy.READ_ONLY}
        if settings.execution_policy is ExecutionPolicy.APPROVAL_REQUIRED:
            allowed.add(ExecutionPolicy.APPROVAL_REQUIRED)
        return [spec for spec in self._specs if spec.policy in allowed]


def build_default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="runtime_status",
                policy=ExecutionPolicy.READ_ONLY,
                description="Inspect runtime model, memory route, tool catalog, and safety defaults.",
                factory=_build_runtime_status_tool,
            ),
            ToolSpec(
                name="slurm_queue",
                policy=ExecutionPolicy.READ_ONLY,
                description="Inspect the Slurm queue with optional user or partition filters.",
                factory=_build_slurm_queue_tool,
            ),
            ToolSpec(
                name="slurm_job_details",
                policy=ExecutionPolicy.READ_ONLY,
                description="Inspect a single Slurm job with scontrol.",
                factory=_build_slurm_job_details_tool,
            ),
            ToolSpec(
                name="module_avail",
                policy=ExecutionPolicy.READ_ONLY,
                description="Inspect available Lmod or Environment Modules entries.",
                factory=_build_module_avail_tool,
            ),
            ToolSpec(
                name="module_list",
                policy=ExecutionPolicy.READ_ONLY,
                description="Inspect the module state visible to the session shell.",
                factory=_build_module_list_tool,
            ),
            ToolSpec(
                name="filesystem_list",
                policy=ExecutionPolicy.READ_ONLY,
                description="List directory contents inside approved filesystem roots.",
                factory=_build_filesystem_list_tool,
            ),
            ToolSpec(
                name="filesystem_head",
                policy=ExecutionPolicy.READ_ONLY,
                description="Read the first lines of a file inside approved filesystem roots.",
                factory=_build_filesystem_head_tool,
            ),
            ToolSpec(
                name="slurm_cancel_job",
                policy=ExecutionPolicy.APPROVAL_REQUIRED,
                description="Cancel a Slurm job after human review.",
                factory=_build_slurm_cancel_job_tool,
                interrupt_config={
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Review a Slurm cancellation before it runs.",
                },
            ),
            ToolSpec(
                name="module_load",
                policy=ExecutionPolicy.APPROVAL_REQUIRED,
                description="Change the active module environment after human review.",
                factory=_build_module_load_tool,
                interrupt_config={
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Review an environment-module change before it runs.",
                },
            ),
            ToolSpec(
                name="filesystem_remove",
                policy=ExecutionPolicy.APPROVAL_REQUIRED,
                description="Remove files or directories inside approved roots after human review.",
                factory=_build_filesystem_remove_tool,
                interrupt_config={
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Review a filesystem mutation before it runs.",
                },
            ),
        ]
    )


def build_tools(
    settings: AssistantSettings,
    registry: ToolRegistry | None = None,
) -> list[BaseTool]:
    active_registry = registry or build_default_registry()
    return active_registry.resolve(settings)


def build_interrupt_policy(
    settings: AssistantSettings,
    registry: ToolRegistry | None = None,
) -> dict[str, InterruptConfig]:
    active_registry = registry or build_default_registry()
    interrupted: dict[str, InterruptConfig] = {name: True for name in DEEPAGENTS_MUTATING_TOOLS}
    if settings.execution_policy is ExecutionPolicy.APPROVAL_REQUIRED:
        for spec in active_registry.approval_required_specs():
            interrupted[spec.name] = spec.interrupt_config or True
    return dict(sorted(interrupted.items()))


def build_tool_manifest(
    settings: AssistantSettings,
    registry: ToolRegistry | None = None,
) -> dict[str, object]:
    active_registry = registry or build_default_registry()
    return {
        "execution_policy": settings.execution_policy.value,
        "command_timeout_seconds": settings.command_timeout_seconds,
        "filesystem_roots": [str(root) for root in configured_filesystem_roots(settings)],
        "active_tools": active_registry.active_names(settings),
        "catalog": active_registry.manifest(),
        "interrupt_on": build_interrupt_policy(settings, active_registry),
    }


def _build_runtime_status_tool(
    settings: AssistantSettings,
    registry: ToolRegistry,
) -> BaseTool:
    @tool("runtime_status")
    def runtime_status() -> dict[str, object]:
        """Inspect runtime model, memory route, and current tool policy."""

        return {
            "assistant_id": settings.assistant_id,
            "model_name": settings.model_name,
            "execution_policy": settings.execution_policy.value,
            "memory_mount_path": settings.memory_mount_path,
            "command_timeout_seconds": settings.command_timeout_seconds,
            "filesystem_roots": [str(root) for root in configured_filesystem_roots(settings)],
            "active_tools": registry.active_names(settings),
        }

    return runtime_status


def _build_slurm_queue_tool(
    settings: AssistantSettings,
    _registry: ToolRegistry,
) -> BaseTool:
    @tool("slurm_queue")
    def slurm_queue(user: str | None = None, partition: str | None = None) -> dict[str, object]:
        """Inspect the Slurm queue with optional user and partition filters."""

        command = [
            "squeue",
            "--noheader",
            "--format=%i|%T|%u|%M|%R",
        ]
        arguments: dict[str, object] = {}
        if user is not None:
            arguments["user"] = _validated_token(user, _SAFE_USER, "user")
            command.extend(["--user", arguments["user"]])
        if partition is not None:
            arguments["partition"] = _validated_token(partition, _SAFE_TOKEN, "partition")
            command.extend(["--partition", arguments["partition"]])
        return _run_command("slurm_queue", settings, command, arguments)

    return slurm_queue


def _build_slurm_job_details_tool(
    settings: AssistantSettings,
    _registry: ToolRegistry,
) -> BaseTool:
    @tool("slurm_job_details")
    def slurm_job_details(job_id: str) -> dict[str, object]:
        """Inspect a single Slurm job with scontrol."""

        safe_job_id = _validated_token(job_id, _SAFE_JOB_ID, "job_id")
        return _run_command(
            "slurm_job_details",
            settings,
            ["scontrol", "show", "job", safe_job_id],
            {"job_id": safe_job_id},
        )

    return slurm_job_details


def _build_slurm_cancel_job_tool(
    settings: AssistantSettings,
    _registry: ToolRegistry,
) -> BaseTool:
    @tool("slurm_cancel_job")
    def slurm_cancel_job(job_id: str, signal: str | None = None) -> dict[str, object]:
        """Cancel a Slurm job. This tool is approval-gated in Deep Agents."""

        safe_job_id = _validated_token(job_id, _SAFE_JOB_ID, "job_id")
        command = ["scancel"]
        arguments: dict[str, object] = {"job_id": safe_job_id}
        if signal is not None:
            safe_signal = _validated_token(signal.upper(), _SAFE_SIGNAL, "signal")
            command.extend(["--signal", safe_signal])
            arguments["signal"] = safe_signal
        command.append(safe_job_id)
        return _run_command("slurm_cancel_job", settings, command, arguments)

    return slurm_cancel_job


def _build_module_avail_tool(
    settings: AssistantSettings,
    _registry: ToolRegistry,
) -> BaseTool:
    @tool("module_avail")
    def module_avail(query: str | None = None) -> dict[str, object]:
        """Inspect available Lmod or Environment Modules entries."""

        command = ["modulecmd", "python", "avail"]
        arguments: dict[str, object] = {}
        if query is not None:
            safe_query = _validated_token(query, _SAFE_MODULE, "query")
            command.append(safe_query)
            arguments["query"] = safe_query
        return _run_command("module_avail", settings, command, arguments)

    return module_avail


def _build_module_list_tool(
    settings: AssistantSettings,
    _registry: ToolRegistry,
) -> BaseTool:
    @tool("module_list")
    def module_list() -> dict[str, object]:
        """Inspect the module state visible to the session shell."""

        return _run_command("module_list", settings, ["modulecmd", "python", "list"], {})

    return module_list


def _build_module_load_tool(
    settings: AssistantSettings,
    _registry: ToolRegistry,
) -> BaseTool:
    @tool("module_load")
    def module_load(module_name: str) -> dict[str, object]:
        """Request a module load operation. This tool is approval-gated in Deep Agents."""

        safe_module_name = _validated_token(module_name, _SAFE_MODULE, "module_name")
        return _run_command(
            "module_load",
            settings,
            ["modulecmd", "python", "load", safe_module_name],
            {"module_name": safe_module_name},
        )

    return module_load


def _build_filesystem_list_tool(
    settings: AssistantSettings,
    _registry: ToolRegistry,
) -> BaseTool:
    @tool("filesystem_list")
    def filesystem_list(path: str = ".", long: bool = False, all_entries: bool = False) -> dict[str, object]:
        """List directory contents inside approved filesystem roots."""

        resolved_path = str(resolve_allowed_path(path, settings))
        flags = "-lA" if long else "-1A"
        if all_entries:
            flags = "-la" if long else "-1a"
        return _run_command(
            "filesystem_list",
            settings,
            ["ls", flags, "--", resolved_path],
            {
                "path": resolved_path,
                "long": bool(long),
                "all_entries": bool(all_entries),
            },
        )

    return filesystem_list


def _build_filesystem_head_tool(
    settings: AssistantSettings,
    _registry: ToolRegistry,
) -> BaseTool:
    @tool("filesystem_head")
    def filesystem_head(path: str, lines: int = 40) -> dict[str, object]:
        """Read the first lines of a file inside approved filesystem roots."""

        resolved_path = str(resolve_allowed_path(path, settings))
        safe_lines = _validated_line_count(lines)
        return _run_command(
            "filesystem_head",
            settings,
            ["head", "-n", str(safe_lines), "--", resolved_path],
            {"path": resolved_path, "lines": safe_lines},
        )

    return filesystem_head


def _build_filesystem_remove_tool(
    settings: AssistantSettings,
    _registry: ToolRegistry,
) -> BaseTool:
    @tool("filesystem_remove")
    def filesystem_remove(path: str, recursive: bool = False, force: bool = False) -> dict[str, object]:
        """Remove files or directories inside approved filesystem roots."""

        resolved_path = str(resolve_allowed_path(path, settings))
        command = ["rm"]
        if recursive:
            command.append("-r")
        if force:
            command.append("-f")
        command.extend(["--", resolved_path])
        return _run_command(
            "filesystem_remove",
            settings,
            command,
            {
                "path": resolved_path,
                "recursive": bool(recursive),
                "force": bool(force),
            },
        )

    return filesystem_remove


def _validated_token(value: str, pattern: re.Pattern[str], field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned or not pattern.fullmatch(cleaned):
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return cleaned


def _validated_line_count(value: int) -> int:
    if not isinstance(value, int):
        raise ValueError("lines must be an integer")
    if value < 1 or value > 200:
        raise ValueError("lines must be between 1 and 200")
    return value


def _run_command(
    tool_name: str,
    settings: AssistantSettings,
    command: Sequence[str],
    arguments: dict[str, object],
) -> dict[str, object]:
    started = perf_counter()
    exit_code = 0
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.command_timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except FileNotFoundError as error:
        stderr = f"missing executable: {error.filename}"
        exit_code = 127
    except subprocess.TimeoutExpired as error:
        stdout = _stream_text(error.stdout)
        stderr = _stream_text(error.stderr) or ""
        if stderr:
            stderr = f"{stderr.rstrip()}\ncommand timed out after {settings.command_timeout_seconds} seconds"
        else:
            stderr = f"command timed out after {settings.command_timeout_seconds} seconds"
        exit_code = 124

    duration_ms = int((perf_counter() - started) * 1000)
    return {
        "tool": tool_name,
        "policy": settings.execution_policy.value,
        "command": list(command),
        "arguments": arguments,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "ok": exit_code == 0,
        "duration_ms": duration_ms,
    }


def build_review_command(
    tool_name: str,
    arguments: dict[str, Any],
    settings: AssistantSettings,
) -> list[str]:
    if tool_name == "slurm_cancel_job":
        return _slurm_cancel_job_command(arguments)
    if tool_name == "module_load":
        return _module_load_command(arguments)
    if tool_name == "filesystem_remove":
        return _filesystem_remove_command(arguments, settings)
    raise KeyError(f"tool does not support review previews: {tool_name}")


def run_reviewed_command(
    command_text: str,
    settings: AssistantSettings,
) -> dict[str, object]:
    command = shlex.split(command_text)
    if not command:
        raise ValueError("reviewed command must not be empty")
    return _run_command(
        "reviewed_command",
        settings,
        command,
        {"command_text": command_text},
    )


def _slurm_cancel_job_command(arguments: dict[str, Any]) -> list[str]:
    safe_job_id = _validated_token(str(arguments.get("job_id", "")), _SAFE_JOB_ID, "job_id")
    command = ["scancel"]
    signal = arguments.get("signal")
    if signal is not None:
        safe_signal = _validated_token(str(signal).upper(), _SAFE_SIGNAL, "signal")
        command.extend(["--signal", safe_signal])
    command.append(safe_job_id)
    return command


def _module_load_command(arguments: dict[str, Any]) -> list[str]:
    safe_module_name = _validated_token(
        str(arguments.get("module_name", "")),
        _SAFE_MODULE,
        "module_name",
    )
    return ["modulecmd", "python", "load", safe_module_name]


def _filesystem_remove_command(
    arguments: dict[str, Any],
    settings: AssistantSettings,
) -> list[str]:
    resolved_path = str(resolve_allowed_path(str(arguments.get("path", "")), settings))
    command = ["rm"]
    if bool(arguments.get("recursive")):
        command.append("-r")
    if bool(arguments.get("force")):
        command.append("-f")
    command.extend(["--", resolved_path])
    return command


def _stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
