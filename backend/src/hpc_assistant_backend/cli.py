from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assistant.service import build_assistant_session
from .config import load_settings
from .guardrails import build_doctor_report, require_cluster_probe_permission
from .openai_client import build_openai_client
from .opencode_tools import sync_opencode_project, tool_manifest  # noqa: F401
from .pgoa.agent.loop import PGOAAgent
from .pgoa.cluster_discovery import run_probe_jobs
from .pgoa.env_discovery import discover_software_env
from .pgoa.services import discover_cluster
from .pgoa.store import ExperimentStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HPC assistant backend prototype for cluster and environment discovery."
    )
    subcommands = parser.add_subparsers(dest="command")

    subcommands.add_parser(
        "show-opencode-tools",
        help="Print the Python-backed OpenCode custom-tool manifest as JSON.",
    )
    sync_parser = subcommands.add_parser(
        "sync-opencode-tools",
        help="Write repo-local OpenCode custom tools and project config from the Python registry.",
    )
    sync_parser.add_argument(
        "--root",
        type=Path,
        help="Override the project root. Defaults to the repository root.",
    )
    assist_parser = subcommands.add_parser(
        "assist",
        help="Describe and persist a repo-aware HPC assistant session without replacing OpenCode's editing flow.",
    )
    assist_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path to the working repository. Defaults to the current directory.",
    )
    assist_parser.add_argument(
        "--goal",
        help="Optional user goal for the current coding or optimization session.",
    )
    assist_parser.add_argument(
        "--mode",
        choices=("coding", "pgoa"),
        default="coding",
        help="Primary assistant mode. 'pgoa' keeps optimization as a skill inside the same session.",
    )
    assist_parser.add_argument(
        "--sync-opencode",
        action="store_true",
        help="Write repo-local OpenCode config and tool shims before persisting the session.",
    )

    discover_cluster_parser = subcommands.add_parser(
        "discover-cluster",
        help="Run login-node cluster discovery and cache the resulting profile.",
    )
    discover_cluster_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore any cached cluster profile and refresh it from the login node.",
    )

    subcommands.add_parser(
        "discover-env",
        help="Inspect the active module system and classify available software modules.",
    )

    probe_cluster_parser = subcommands.add_parser(
        "probe-cluster",
        help="Submit a short probe job on one or more partitions to collect real compute-node topology.",
    )
    probe_cluster_parser.add_argument(
        "--partition",
        action="append",
        dest="partitions",
        help="Partition to probe. Repeat to probe multiple partitions. Defaults to all unprobed partitions.",
    )
    probe_cluster_parser.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge that this command will submit short Slurm probe jobs.",
    )

    subcommands.add_parser(
        "doctor",
        help="Print a compact health and guardrail report for this installation.",
    )

    run_pgoa = subcommands.add_parser(
        "run-pgoa",
        help="Run the experimental PGOA agent via the OpenAI Python SDK.",
    )
    run_pgoa.add_argument("--workload-id", required=True, help="Unique workload identifier.")
    run_pgoa.add_argument("--job-script-path", type=Path, required=True, help="Path to the Slurm job script.")
    run_pgoa.add_argument("--primary-kpi", required=True, help="Primary KPI to optimize, e.g. elapsed_s.")
    run_pgoa.add_argument("--kpi-unit", required=True, help="Unit for the KPI, e.g. seconds.")
    run_pgoa.add_argument("--model", help="Override the configured OpenAI model.")
    run_pgoa.add_argument(
        "--max-iterations",
        type=int,
        help="Override the maximum number of agent iterations.",
    )
    run_pgoa.add_argument(
        "--kpi-threshold-pct",
        type=float,
        help="Stop when absolute KPI delta is below this percentage.",
    )
    run_pgoa.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat larger KPI values as improvements.",
    )
    run_pgoa.add_argument(
        "--allow-probe-jobs",
        action="store_true",
        help="Allow compute-node probe jobs during agent startup cluster discovery.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "show-opencode-tools"

    if command == "show-opencode-tools":
        print(json.dumps(tool_manifest(), indent=2))
        return 0

    if command == "sync-opencode-tools":
        written = sync_opencode_project(args.root)
        _print_json({"written": [str(path) for path in written]})
        return 0

    settings = load_settings()

    if command == "discover-cluster":
        store = ExperimentStore(Path(settings.pgoa_store_path).expanduser())
        profile = discover_cluster(
            settings,
            store,
            force_refresh=bool(args.force_refresh),
        )
        _print_json(profile.model_dump(mode="json"))
        return 0

    if command == "discover-env":
        env = discover_software_env(settings)
        _print_json(env.model_dump(mode="json"))
        return 0

    if command == "probe-cluster":
        store = ExperimentStore(Path(settings.pgoa_store_path).expanduser())
        try:
            require_cluster_probe_permission(
                settings,
                explicit_yes=bool(args.yes),
            )
        except PermissionError as exc:
            _print_json({"ok": False, "error": str(exc)})
            return 2
        profile = discover_cluster(settings, store)
        updated = run_probe_jobs(
            profile,
            settings,
            partitions=args.partitions,
            approval_callback=lambda _: True,
        )
        store.save_cluster_profile(updated)
        _print_json(updated.model_dump(mode="json"))
        return 0

    if command == "doctor":
        _print_json(build_doctor_report(settings))
        return 0

    if command == "assist":
        session = build_assistant_session(
            settings,
            repo_root=str(args.repo),
            goal=args.goal,
            mode=args.mode,
            sync_opencode=bool(args.sync_opencode),
        )
        _print_json(session)
        return 0

    if command == "run-pgoa":
        store = ExperimentStore(Path(settings.pgoa_store_path).expanduser())
        client = build_openai_client(settings)
        probe_callback = None
        if args.allow_probe_jobs or settings.allow_cluster_probe_jobs:
            probe_callback = lambda _: True
        agent = PGOAAgent(
            store=store,
            llm_client=client,
            model=args.model or settings.openai_model,
            max_iterations=(
                args.max_iterations if args.max_iterations is not None else 5
            ),
            kpi_threshold_pct=(
                args.kpi_threshold_pct if args.kpi_threshold_pct is not None else 2.0
            ),
            settings=settings,
            probe_approval_callback=probe_callback,
        )
        result = agent.run(
            workload_id=args.workload_id,
            job_script_path=str(args.job_script_path),
            primary_kpi=args.primary_kpi,
            kpi_unit=args.kpi_unit,
            lower_is_better=not args.higher_is_better,
        )
        _print_json(result)
        return 0

    parser.print_help()
    return 1


def _print_json(payload: object) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")  # type: ignore[assignment]
    elif hasattr(payload, "model_dump_json"):
        print(payload.model_dump_json(indent=2))  # type: ignore[attr-defined]
        return
    elif hasattr(payload, "__dict__"):
        payload = payload.__dict__  # type: ignore[assignment]
    print(json.dumps(payload, indent=2))
