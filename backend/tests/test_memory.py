from __future__ import annotations

from types import SimpleNamespace
import unittest

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.memory import build_memory_bundle


class MemoryTests(unittest.TestCase):
    def test_memory_bundle_wires_composite_backend(self) -> None:
        settings = AssistantSettings(memory_mount_path="/memories/", memory_namespace="shared-memory")
        bundle = build_memory_bundle(settings)

        runtime = SimpleNamespace(
            state={},
            store=bundle.store,
            config={"metadata": {"assistant_id": "assistant-a", "user_id": "user-b"}},
        )
        backend = bundle.backend_factory(runtime)

        self.assertIsInstance(bundle.checkpointer, InMemorySaver)
        self.assertIsInstance(bundle.store, InMemoryStore)
        self.assertIsInstance(backend, CompositeBackend)
        self.assertIsInstance(backend.default, StateBackend)
        self.assertIn(settings.memory_mount_path, backend.routes)
        self.assertIsInstance(backend.routes[settings.memory_mount_path], StoreBackend)
        self.assertEqual(
            backend.routes[settings.memory_mount_path]._get_namespace(),
            ("assistant-a", "user-b", "shared-memory"),
        )
        self.assertTrue(any(item["path"] == "/memories/" for item in backend.ls_info("/")))


if __name__ == "__main__":
    unittest.main()
