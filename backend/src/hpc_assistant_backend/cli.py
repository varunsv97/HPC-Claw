from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import sys

from .config import BackendConfig, default_config_path, load_config, write_config
from .runtime import AssistantRuntime
from .service import serve
from .stdio import run_stdio
from .tools import build_tool_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HPC assistant backend for a real cluster user session."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Path to a TOML config file. Defaults to HPC_ASSISTANT_CONFIG, "
            "~/.config/hpc-assistant/config.toml, or the repo example config."
        ),
    )

    subcommands = parser.add_subparsers(dest="command")
    init_parser = subcommands.add_parser(
        "init",
        help="Create or update a user config for an OpenAI-compatible endpoint.",
    )
    init_parser.add_argument("--host", default="127.0.0.1", help="Backend host to bind.")
    init_parser.add_argument("--port", type=int, default=8765, help="Backend port to bind.")
    init_parser.add_argument(
        "--provider",
        default="openai-compatible",
        help="Model provider label stored in the config.",
    )
    init_parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API base URL, for example https://api.openai.com/v1.",
    )
    init_parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Default chat model name.",
    )
    init_parser.add_argument(
        "--api-key",
        help="API token to store in the config file. If omitted in a TTY, you will be prompted.",
    )
    init_parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Fallback environment variable name for the API token.",
    )
    init_parser.add_argument(
        "--assistant-id",
        default="hpc-assistant",
        help="Assistant identifier stored in runtime metadata.",
    )
    init_parser.add_argument(
        "--default-user-id",
        default="local-user",
        help="Default cluster user identifier used when none is supplied explicitly.",
    )
    init_parser.add_argument(
        "--thread-store",
        default="~/.local/share/hpc-assistant/threads",
        help="Directory used for thread/session artifacts.",
    )
    init_parser.add_argument(
        "--long-term-store",
        default="~/.local/share/hpc-assistant/memory",
        help="Directory used for durable memory artifacts.",
    )
    init_parser.add_argument(
        "--filesystem-root",
        action="append",
        dest="filesystem_roots",
        help="Approved root for file browser and file tools. Repeat to allow multiple roots.",
    )
    init_parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=30.0,
        help="Timeout applied to shell tools and reviewed commands.",
    )
    init_parser.add_argument(
        "--approval-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether mutating tool calls require review before execution.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config file without prompting.",
    )

    subcommands.add_parser("show-config", help="Print the resolved backend configuration as JSON.")
    subcommands.add_parser(
        "show-tools",
        help="Print the active HPC tool catalog and approval policy as JSON.",
    )

    serve_parser = subcommands.add_parser("serve", help="Run the localhost backend service.")
    serve_parser.add_argument("--host", help="Host to bind.")
    serve_parser.add_argument("--port", type=int, help="Port to bind.")

    invoke_parser = subcommands.add_parser("invoke", help="Send a single message through the runtime and print JSON output.")
    invoke_parser.add_argument("message", help="User message to send to the agent runtime.")
    invoke_parser.add_argument("--thread-id", default="cli-thread", help="Thread identifier for the backend session.")
    invoke_parser.add_argument("--user-id", help="User identifier stored in metadata and long-term memory namespaces.")

    subcommands.add_parser(
        "stdio",
        help="Run the JSONL stdin/stdout transport used by the Textual TUI.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "show-config"

    if command == "init":
        return _run_init(args)

    config = load_config(args.config)

    if command == "show-config":
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    if command == "show-tools":
        print(json.dumps(build_tool_manifest(config.to_settings()), indent=2))
        return 0

    if command == "invoke":
        runtime = AssistantRuntime(settings=config.to_settings())
        result = runtime.start_session(args.thread_id, user_id=args.user_id).invoke(args.message)
        print(
            json.dumps(
                {
                    "thread_id": result.thread_id,
                    "status": result.status.value,
                    "output_text": result.output_text,
                    "interrupts": [
                        {
                            "id": interrupt.id,
                            "value": interrupt.value,
                        }
                        for interrupt in result.interrupts
                    ],
                },
                indent=2,
            )
        )
        return 0

    if command == "stdio":
        return run_stdio(config.to_settings())

    if args.host:
        config.host = args.host
    if args.port is not None:
        config.port = args.port

    serve(config)
    return 0


def _run_init(args: argparse.Namespace) -> int:
    config_path = (args.config or default_config_path()).expanduser()
    if config_path.exists() and not args.force:
        print(
            f"Config already exists at {config_path}. Re-run with --force to overwrite it.",
            file=sys.stderr,
        )
        return 2

    api_key = args.api_key
    if api_key is None and sys.stdin.isatty():
        prompted = getpass.getpass("OpenAI-compatible API token (leave blank to skip storing it): ")
        api_key = prompted.strip() or None

    config = BackendConfig(
        host=args.host,
        port=args.port,
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        api_key=api_key,
        thread_store=args.thread_store,
        long_term_store=args.long_term_store,
        approval_required=bool(args.approval_required),
        assistant_id=args.assistant_id,
        default_user_id=args.default_user_id,
        command_timeout_seconds=args.command_timeout_seconds,
        filesystem_roots=tuple(args.filesystem_roots or ["~"]),
    )
    written_path = write_config(config_path, config)
    for directory in [Path(config.thread_store).expanduser(), Path(config.long_term_store).expanduser()]:
        directory.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "config_path": str(written_path),
                "backend_url": f"http://{config.host}:{config.port}",
                "model": {
                    "provider": config.provider,
                    "base_url": config.base_url,
                    "model": config.model,
                    "api_key_present": config.api_key is not None,
                    "api_key_env": config.api_key_env,
                },
                "filesystem_roots": list(config.filesystem_roots),
                "approval_required": config.approval_required,
            },
            indent=2,
        )
    )
    return 0
