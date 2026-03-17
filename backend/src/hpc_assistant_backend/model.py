"""Model construction helpers."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from hpc_assistant_backend.config import AssistantSettings


def build_chat_model(settings: AssistantSettings) -> BaseChatModel:
    """Construct the OpenAI-compatible chat model used by the runtime."""

    kwargs: dict[str, object] = {
        "model": settings.model_name,
        "timeout": settings.timeout_seconds,
        "max_retries": settings.max_retries,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key is not None:
        kwargs["api_key"] = settings.api_key.get_secret_value()
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    return ChatOpenAI(**kwargs)
