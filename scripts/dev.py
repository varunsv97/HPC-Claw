from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = ROOT / "backend" / "src"
PROJECT_VENV = ROOT / ".venv"
PROJECT_PYTHON = PROJECT_VENV / "bin" / "python"


def runtime_python() -> str:
    if PROJECT_PYTHON.exists():
        return str(PROJECT_PYTHON)
    return sys.executable


def python_runtime_env(*extra_paths: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    path_entries = [str(path) for path in extra_paths]
    if pythonpath:
        path_entries.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    if PROJECT_PYTHON.exists():
        env["VIRTUAL_ENV"] = str(PROJECT_VENV)
        current_path = env.get("PATH", "")
        env["PATH"] = (
            f"{PROJECT_VENV / 'bin'}{os.pathsep}{current_path}"
            if current_path
            else str(PROJECT_VENV / "bin")
        )

    if PROJECT_PYTHON.exists() or sysconfig.get_config_var("Py_GIL_DISABLED") == 1:
        env.setdefault("PYTHON_GIL", "1")
    return env


def backend_env() -> dict[str, str]:
    return python_runtime_env(BACKEND_SRC)


def run_command(command: list[str], env: dict[str, str] | None = None) -> int:
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local development launcher for hpc-assistant (OpenCode path)."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    app = subcommands.add_parser(
        "app",
        help="Sync the Python-backed OpenCode custom tools and launch OpenCode in this repo.",
    )
    app.add_argument("--profile", default="local")

    subcommands.add_parser(
        "sync-opencode-tools",
        help="Regenerate the repo-local OpenCode custom tools from the Python registry.",
    )

    return parser


def run_app(profile: str) -> int:
    _ = profile
    sync_command = [
        runtime_python(),
        "-m",
        "hpc_assistant_backend",
        "sync-opencode-tools",
    ]
    sync_status = run_command(sync_command, env=backend_env())
    if sync_status != 0:
        return sync_status

    return run_command(["opencode", str(ROOT)])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "sync-opencode-tools":
            return run_command(
                [
                    runtime_python(),
                    "-m",
                    "hpc_assistant_backend",
                    "sync-opencode-tools",
                ],
                env=backend_env(),
            )
        return run_app(args.profile)
    except FileNotFoundError as error:
        parser.exit(1, f"missing executable: {error.filename}\n")


if __name__ == "__main__":
    raise SystemExit(main())
