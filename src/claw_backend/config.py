"""Configuration primitives for the HPC Claw backend."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AssistantSettings(BaseSettings):
    """Settings for backend execution, filesystem access, and cluster guardrails."""

    model_config = SettingsConfigDict(
        env_prefix="HPC_CLAW_",
        env_file=".env",
        extra="ignore",
    )

    command_timeout_seconds: float = 15.0
    filesystem_roots: tuple[str, ...] = ("~",)
    # Glob patterns relative to each filesystem root, or absolute paths.
    # Files matching any pattern are write-protected — the backend will refuse
    # to write them even if they are inside an allowed filesystem root.
    # Example (env var): HPC_CLAW_READONLY_PATHS='["*.toml","src/generated/**"]'
    readonly_paths: tuple[str, ...] = ()
    pgoa_store_path: str = "~/.hpcclaw/pgoa_store"
    provider_api_key: str | None = None
    provider_base_url: str | None = None
    provider_model: str = "gpt-5"
    provider_timeout_seconds: float = 60.0
    allow_cluster_probe_jobs: bool = False
    guardrails_enabled: bool = True
    # Path to the project-level agent.toml.  When None, PGOA searches cwd upward.
    agent_toml_path: str | None = None
    # Input data directories — PGOA may read these for context, never writes them.
    # Stored separately from readonly_paths so the frontend can display them distinctly.
    # Example: HPC_CLAW_DATA_PATHS='["/scratch/data","~/datasets"]'
    data_paths: tuple[str, ...] = ()


def load_settings(**overrides: object) -> AssistantSettings:
    """Load settings from the environment plus explicit overrides."""

    return AssistantSettings(**overrides)
