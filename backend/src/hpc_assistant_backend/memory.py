"""LangGraph checkpointer, store, and Deep Agents backend wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.protocol import BackendFactory
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from hpc_assistant_backend.config import AssistantSettings


@dataclass(frozen=True, slots=True)
class MemoryBundle:
    """Concrete memory primitives used to build the agent graph."""

    checkpointer: InMemorySaver
    store: InMemoryStore
    backend_factory: BackendFactory


def build_memory_bundle(settings: AssistantSettings) -> MemoryBundle:
    """Build the checkpointer, store, and routed backend factory."""

    return MemoryBundle(
        checkpointer=build_checkpointer(settings),
        store=build_store(settings),
        backend_factory=build_backend_factory(settings),
    )


def build_checkpointer(settings: AssistantSettings) -> InMemorySaver:
    """Build the short-term thread checkpointer."""

    if settings.checkpointer_backend != "memory":
        raise ValueError(f"unsupported checkpointer backend: {settings.checkpointer_backend}")
    return InMemorySaver()


def build_store(settings: AssistantSettings) -> InMemoryStore:
    """Build the long-term memory store."""

    if settings.store_backend != "memory":
        raise ValueError(f"unsupported store backend: {settings.store_backend}")
    return InMemoryStore()


def build_backend_factory(settings: AssistantSettings) -> BackendFactory:
    """Route ephemeral files to state and `/memories/` to the LangGraph store."""

    namespace = _build_namespace_factory(settings)

    def factory(runtime: Any) -> CompositeBackend:
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                settings.memory_mount_path: StoreBackend(runtime, namespace=namespace),
            },
        )

    return factory


def _build_namespace_factory(settings: AssistantSettings):
    def namespace(context: Any) -> tuple[str, ...]:
        metadata = _extract_metadata(getattr(context.runtime, "config", None))
        assistant_id = str(metadata.get("assistant_id", settings.assistant_id))
        user_id = str(metadata.get("user_id", settings.default_user_id))
        return (assistant_id, user_id, settings.memory_namespace)

    return namespace


def _extract_metadata(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    metadata = config.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    return metadata
