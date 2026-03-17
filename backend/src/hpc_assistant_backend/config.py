"""Configuration primitives for the backend runtime and scaffold CLI."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tomllib
from enum import Enum

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._@:+~-]+$")
_DEFAULT_CONFIG_PATH = Path("~/.config/hpc-assistant/config.toml").expanduser()


class ExecutionPolicy(str, Enum):
    """Safety policy for tool availability."""

    READ_ONLY = "read_only"
    APPROVAL_REQUIRED = "approval_required"


class AssistantSettings(BaseSettings):
    """Settings shared across model, memory, and runtime orchestration."""

    model_config = SettingsConfigDict(
        env_prefix="HPC_ASSISTANT_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    assistant_id: str = "hpc-assistant"
    default_user_id: str = "local-user"
    model_name: str = "gpt-4.1-mini"
    base_url: str | None = None
    api_key: SecretStr | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2
    temperature: float | None = None
    execution_policy: ExecutionPolicy = ExecutionPolicy.READ_ONLY
    command_timeout_seconds: float = 15.0
    filesystem_roots: tuple[str, ...] = ("~",)
    memory_mount_path: str = "/memories/"
    memory_namespace: str = "memories"
    checkpointer_backend: str = "memory"
    store_backend: str = "memory"
    debug: bool = False

    @field_validator("assistant_id", "default_user_id", "memory_namespace")
    @classmethod
    def validate_namespace_component(cls, value: str) -> str:
        if not value or not _SAFE_COMPONENT.fullmatch(value):
            msg = (
                "settings components must be non-empty and contain only letters, "
                "numbers, dots, underscores, @, +, :, hyphen, or tilde"
            )
            raise ValueError(msg)
        return value

    @field_validator("memory_mount_path")
    @classmethod
    def normalize_memory_mount_path(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("memory_mount_path must not be empty")
        if not cleaned.startswith("/"):
            cleaned = f"/{cleaned}"
        if not cleaned.endswith("/"):
            cleaned = f"{cleaned}/"
        return cleaned

    @field_validator("checkpointer_backend", "store_backend")
    @classmethod
    def validate_memory_backend(cls, value: str) -> str:
        if value != "memory":
            raise ValueError("only the 'memory' backend is implemented in this task")
        return value


def load_settings(**overrides: object) -> AssistantSettings:
    """Load settings from the environment plus explicit overrides."""

    return AssistantSettings(**overrides)


@dataclass(slots=True)
class BackendConfig:
    """File-based config used by the placeholder CLI and HTTP service."""

    host: str = "127.0.0.1"
    port: int = 8765
    provider: str = "openai-compatible"
    base_url: str | None = None
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    thread_store: str = "~/.local/share/hpc-assistant/threads"
    long_term_store: str = "~/.local/share/hpc-assistant/memory"
    approval_required: bool = True
    assistant_id: str = "hpc-assistant"
    default_user_id: str = "local-user"
    timeout_seconds: float = 60.0
    command_timeout_seconds: float = 15.0
    filesystem_roots: tuple[str, ...] = ("~",)
    debug: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": {
                "host": self.host,
                "port": self.port,
            },
            "model": {
                "provider": self.provider,
                "base_url": self.base_url,
                "model": self.model,
                "api_key_env": self.api_key_env,
                "api_key_present": self.api_key is not None,
                "api_key_source": "inline" if self.api_key is not None else "environment",
            },
            "memory": {
                "thread_store": self.thread_store,
                "long_term_store": self.long_term_store,
            },
            "safety": {
                "approval_required": self.approval_required,
                "command_timeout_seconds": self.command_timeout_seconds,
                "filesystem_roots": list(self.filesystem_roots),
            },
            "runtime": {
                "assistant_id": self.assistant_id,
                "default_user_id": self.default_user_id,
                "timeout_seconds": self.timeout_seconds,
                "debug": self.debug,
            },
        }

    def to_settings(self) -> AssistantSettings:
        api_key = self.api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        policy = (
            ExecutionPolicy.APPROVAL_REQUIRED
            if self.approval_required
            else ExecutionPolicy.READ_ONLY
        )
        return AssistantSettings(
            assistant_id=self.assistant_id,
            default_user_id=self.default_user_id,
            model_name=self.model,
            base_url=self.base_url,
            api_key=api_key,
            timeout_seconds=self.timeout_seconds,
            execution_policy=policy,
            command_timeout_seconds=self.command_timeout_seconds,
            filesystem_roots=self.filesystem_roots,
            debug=self.debug,
        )


def load_config(path: Path | None = None) -> BackendConfig:
    """Load TOML config from an explicit path, env var, or repo example."""

    config_path = _resolve_config_path(path)
    if config_path is None:
        return BackendConfig()

    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    backend = _section(payload, "backend")
    model = _section(payload, "model")
    memory = _section(payload, "memory")
    tools = _section(payload, "tools")
    runtime = _section(payload, "runtime")

    return BackendConfig(
        host=str(backend.get("host", "127.0.0.1")),
        port=int(backend.get("port", 8765)),
        provider=str(model.get("provider", "openai-compatible")),
        base_url=_optional_str(model.get("base_url")),
        model=str(model.get("model", "gpt-4.1-mini")),
        api_key_env=str(model.get("api_key_env", "OPENAI_API_KEY")),
        api_key=_optional_str(model.get("api_key")),
        thread_store=str(memory.get("thread_store", "~/.local/share/hpc-assistant/threads")),
        long_term_store=str(memory.get("long_term_store", "~/.local/share/hpc-assistant/memory")),
        approval_required=bool(tools.get("approval_required", True)),
        assistant_id=str(runtime.get("assistant_id", "hpc-assistant")),
        default_user_id=str(runtime.get("default_user_id", "local-user")),
        timeout_seconds=float(runtime.get("timeout_seconds", 60.0)),
        command_timeout_seconds=float(tools.get("command_timeout_seconds", 15.0)),
        filesystem_roots=_optional_str_tuple(tools.get("filesystem_roots"), ("~",)),
        debug=bool(runtime.get("debug", False)),
    )


def _resolve_config_path(path: Path | None) -> Path | None:
    if path is not None:
        return path.expanduser().resolve()

    env_path = os.getenv("HPC_ASSISTANT_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()

    if _DEFAULT_CONFIG_PATH.exists():
        return _DEFAULT_CONFIG_PATH.resolve()

    example_path = Path(__file__).resolve().parents[3] / "shared" / "config" / "hpc-assistant.example.toml"
    if example_path.exists():
        return example_path
    return None


def default_config_path() -> Path:
    return _DEFAULT_CONFIG_PATH


def write_config(path: Path, config: BackendConfig) -> Path:
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.write_text(render_config(config), encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return target


def render_config(config: BackendConfig) -> str:
    payload = [
        "[backend]",
        f"host = {_toml_string(config.host)}",
        f"port = {config.port}",
        "",
        "[model]",
        f"provider = {_toml_string(config.provider)}",
        f"base_url = {_toml_or_null(config.base_url)}",
        f"model = {_toml_string(config.model)}",
        f"api_key_env = {_toml_string(config.api_key_env)}",
    ]
    if config.api_key is not None:
        payload.append(f"api_key = {_toml_string(config.api_key)}")
    payload.extend(
        [
            "",
            "[memory]",
            f"thread_store = {_toml_string(config.thread_store)}",
            f"long_term_store = {_toml_string(config.long_term_store)}",
            "",
            "[tools]",
            f"approval_required = {'true' if config.approval_required else 'false'}",
            f"command_timeout_seconds = {config.command_timeout_seconds}",
            f"filesystem_roots = {_toml_array(config.filesystem_roots)}",
            "",
            "[runtime]",
            f"assistant_id = {_toml_string(config.assistant_id)}",
            f"default_user_id = {_toml_string(config.default_user_id)}",
            f"timeout_seconds = {config.timeout_seconds}",
            f"debug = {'true' if config.debug else 'false'}",
            "",
        ]
    )
    return "\n".join(payload)


def _section(payload: dict[str, object], name: str) -> dict[str, object]:
    section = payload.get(name, {})
    return section if isinstance(section, dict) else {}


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_str_tuple(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        normalized = tuple(str(item) for item in value)
        return normalized or default
    raise TypeError(f"expected a string or list of strings, received {type(value)!r}")


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_or_null(value: str | None) -> str:
    if value is None:
        return '""'
    return _toml_string(value)


def _toml_array(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"
