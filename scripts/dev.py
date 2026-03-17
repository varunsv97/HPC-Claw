from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = ROOT / "backend" / "src"
TUI_SRC = ROOT / "tui" / "src"
DEFAULT_CONFIG = ROOT / "shared" / "config" / "hpc-assistant.example.toml"
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

    # Keep the GIL enabled by default, even when the project venv uses a
    # free-threaded CPython build. This is currently the most compatible mode
    # for the LangChain/LangGraph dependency stack.
    if PROJECT_PYTHON.exists() or sysconfig.get_config_var("Py_GIL_DISABLED") == 1:
        env.setdefault("PYTHON_GIL", "1")
    return env


def backend_env() -> dict[str, str]:
    env = python_runtime_env(BACKEND_SRC)
    env.setdefault("HPC_ASSISTANT_CONFIG", str(DEFAULT_CONFIG))
    return env


def tui_env() -> dict[str, str]:
    return python_runtime_env(TUI_SRC)


def run_command(command: list[str], env: dict[str, str] | None = None) -> int:
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return completed.returncode


def wait_for_backend(url: str, timeout_seconds: float = 5.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=0.5) as response:
                if response.status == 200:
                    return
        except (ConnectionError, OSError, urllib.error.URLError):
            time.sleep(0.1)

    raise TimeoutError(f"backend did not become ready at {url}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local development launcher for the Python 3.14 hpc-assistant stack, preferring .venv/bin/python with the GIL enabled."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    backend = subcommands.add_parser("backend", help="Run the Python backend scaffold.")
    backend.add_argument("--host", default="127.0.0.1")
    backend.add_argument("--port", type=int, default=8765)

    tui = subcommands.add_parser("tui", help="Run the Python Textual TUI.")
    tui.add_argument("--backend-url", default="http://127.0.0.1:8765")
    tui.add_argument("--profile", default="local")

    app = subcommands.add_parser(
        "app",
        help="Start the backend and then launch the Textual TUI against it.",
    )
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=8765)
    app.add_argument("--profile", default="local")

    return parser


def run_backend(host: str, port: int) -> int:
    command = [
        runtime_python(),
        "-m",
        "hpc_assistant_backend",
        "--config",
        str(DEFAULT_CONFIG),
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return run_command(command, env=backend_env())


def run_tui(backend_url: str, profile: str) -> int:
    command = [
        runtime_python(),
        "-m",
        "hpc_assistant_tui",
        "--backend-url",
        backend_url,
        "--profile",
        profile,
    ]
    return run_command(command, env=tui_env())


def run_app(host: str, port: int, profile: str) -> int:
    backend_url = f"http://{host}:{port}"
    backend_command = [
        runtime_python(),
        "-m",
        "hpc_assistant_backend",
        "--config",
        str(DEFAULT_CONFIG),
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]

    backend_process = subprocess.Popen(backend_command, cwd=ROOT, env=backend_env())
    try:
        wait_for_backend(backend_url)
        return run_tui(backend_url, profile)
    finally:
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
            backend_process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "backend":
            return run_backend(args.host, args.port)
        if args.command == "tui":
            return run_tui(args.backend_url, args.profile)
        return run_app(args.host, args.port, args.profile)
    except FileNotFoundError as error:
        parser.exit(1, f"missing executable: {error.filename}\n")
    except TimeoutError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
