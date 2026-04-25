"""Entry point for hpc-claw-tui CLI command."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hpc-claw-tui",
        description="HPC Claw — terminal UI for PGOA optimization results.",
    )
    parser.add_argument(
        "--store",
        metavar="PATH",
        help="Override store path (default: HPC_ASSISTANT_PGOA_STORE_PATH or ~/.hpcassist)",
    )
    parser.add_argument(
        "--env-file",
        metavar="FILE",
        default=".env",
        help="Path to .env file (default: .env in cwd)",
    )
    args = parser.parse_args()

    # Allow CLI override without touching the environment permanently
    if args.store:
        os.environ["HPC_ASSISTANT_PGOA_STORE_PATH"] = args.store

    # Load .env manually before Textual starts so settings are populated
    env_file = Path(args.env_file)
    if env_file.exists():
        _load_dotenv(env_file)

    from claw_tui.app import ClawTUI
    ClawTUI().run()


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — sets env vars for keys not already in os.environ."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


if __name__ == "__main__":
    main()
