"""Configuration primitives for the HPC assistant OpenCode backend."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AssistantSettings(BaseSettings):
    """Settings for backend execution, filesystem access, and cluster guardrails."""

    model_config = SettingsConfigDict(
        env_prefix="HPC_ASSISTANT_",
        env_file=".env",
        extra="ignore",
    )

    command_timeout_seconds: float = 15.0
    filesystem_roots: tuple[str, ...] = ("~",)
    pgoa_store_path: str = "~/.hpcassist"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-5"
    openai_timeout_seconds: float = 60.0
    allow_cluster_probe_jobs: bool = False
    guardrails_enabled: bool = True


def load_settings(**overrides: object) -> AssistantSettings:
    """Load settings from the environment plus explicit overrides."""

    return AssistantSettings(**overrides)
