from __future__ import annotations

import argparse
from typing import Sequence

from .model import TuiConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HPC Assistant Textual TUI."
    )
    parser.add_argument(
        "--backend-url",
        default=TuiConfig.with_defaults().backend_url,
        help="Backend base URL. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--profile",
        default=TuiConfig.with_defaults().profile,
        help="TUI profile name. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--state-path",
        default=TuiConfig.with_defaults().state_path,
        help="Path to the persisted TUI snapshot JSON file.",
    )
    return parser


def help_text() -> str:
    return build_parser().format_help()


def parse_args(argv: Sequence[str] | None = None) -> TuiConfig:
    namespace = build_parser().parse_args(argv)
    return TuiConfig(
        backend_url=namespace.backend_url,
        profile=namespace.profile,
        state_path=namespace.state_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        from .textual_app import run_textual_app
    except ImportError as error:
        print(
            "The Python TUI requires the `textual` package. "
            "Install it with `python -m pip install -e tui` or "
            "`python -m pip install textual`."
        )
        print(f"Import error: {error}")
        return 1

    run_textual_app(config)
    return 0
