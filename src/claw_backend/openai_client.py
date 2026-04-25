"""Helpers for constructing OpenAI SDK clients from repo settings."""

from __future__ import annotations

from claw_backend.config import AssistantSettings


def build_openai_client(settings: AssistantSettings):
    """Create an OpenAI SDK client using repo settings plus env defaults."""
    from openai import OpenAI

    client_kwargs: dict[str, object] = {"timeout": settings.openai_timeout_seconds}
    if settings.openai_api_key:
        client_kwargs["api_key"] = settings.openai_api_key
    if settings.openai_base_url:
        client_kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**client_kwargs)
