from __future__ import annotations

from pathlib import Path
import unittest

from hpc_assistant_backend.config import (
    AssistantSettings,
    BackendConfig,
    ExecutionPolicy,
    default_config_path,
    load_config,
    render_config,
)


class ConfigTests(unittest.TestCase):
    def test_settings_load_from_environment(self) -> None:
        old_values = {
            "HPC_ASSISTANT_ASSISTANT_ID": None,
            "HPC_ASSISTANT_MODEL_NAME": None,
            "HPC_ASSISTANT_BASE_URL": None,
            "HPC_ASSISTANT_API_KEY": None,
            "HPC_ASSISTANT_TIMEOUT_SECONDS": None,
            "HPC_ASSISTANT_EXECUTION_POLICY": None,
            "HPC_ASSISTANT_MEMORY_MOUNT_PATH": None,
        }
        import os

        for key in old_values:
            old_values[key] = os.environ.get(key)

        try:
            os.environ["HPC_ASSISTANT_ASSISTANT_ID"] = "cluster-agent"
            os.environ["HPC_ASSISTANT_MODEL_NAME"] = "local-compatible-model"
            os.environ["HPC_ASSISTANT_BASE_URL"] = "http://localhost:1234/v1"
            os.environ["HPC_ASSISTANT_API_KEY"] = "test-key"
            os.environ["HPC_ASSISTANT_TIMEOUT_SECONDS"] = "12.5"
            os.environ["HPC_ASSISTANT_EXECUTION_POLICY"] = "approval_required"
            os.environ["HPC_ASSISTANT_MEMORY_MOUNT_PATH"] = "long-term"

            settings = AssistantSettings()
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual(settings.assistant_id, "cluster-agent")
        self.assertEqual(settings.model_name, "local-compatible-model")
        self.assertEqual(settings.base_url, "http://localhost:1234/v1")
        self.assertIsNotNone(settings.api_key)
        self.assertEqual(settings.api_key.get_secret_value(), "test-key")
        self.assertEqual(settings.timeout_seconds, 12.5)
        self.assertIs(settings.execution_policy, ExecutionPolicy.APPROVAL_REQUIRED)
        self.assertEqual(settings.memory_mount_path, "/long-term/")

    def test_backend_config_loads_tool_defaults_from_example_config(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2] / "shared" / "config" / "hpc-assistant.example.toml"
        )

        config = load_config(config_path)
        settings = config.to_settings()

        self.assertEqual(config.command_timeout_seconds, 15.0)
        self.assertEqual(config.filesystem_roots, ("~",))
        self.assertEqual(settings.command_timeout_seconds, 15.0)
        self.assertEqual(settings.filesystem_roots, ("~",))

    def test_inline_api_key_takes_precedence_over_environment(self) -> None:
        config = BackendConfig(api_key="inline-token", api_key_env="IGNORED")

        settings = config.to_settings()

        self.assertIsNotNone(settings.api_key)
        self.assertEqual(settings.api_key.get_secret_value(), "inline-token")

    def test_render_config_includes_runtime_sections(self) -> None:
        content = render_config(BackendConfig(base_url="https://api.example.invalid/v1"))

        self.assertIn("[runtime]", content)
        self.assertIn('base_url = "https://api.example.invalid/v1"', content)
        self.assertEqual(default_config_path().name, "config.toml")


if __name__ == "__main__":
    unittest.main()
