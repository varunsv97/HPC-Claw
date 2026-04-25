"""OpenCode PGOA tool registry.

Five Python-backed tools that add value OpenCode's native shell/file tools cannot:

  pgoa_collect_profile  — sacct/NCU/LIKWID → typed ProfileBundle, atomic store
  pgoa_analyze          — deterministic bottleneck classification
  pgoa_apply_binding    — validated #SBATCH directive rewrite
  pgoa_compare_runs     — typed delta between two runs
  pgoa_store_info       — list experiment runs / retrieve baseline

Everything else (squeue, sinfo, module, file reads/writes) is handled by
OpenCode's native shell and file tools — no Python wrapper needed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from claw_backend.config import AssistantSettings, load_settings
from claw_backend.pgoa.services import (
    analyze_run,
    apply_binding_change,
    build_kpi,
    collect_profile,
    compare_runs,
    list_runs,
)
from claw_backend.pgoa.store import ExperimentStore

ArgKind = Literal["string", "boolean", "integer"]


@dataclass(frozen=True, slots=True)
class ArgSpec:
    name: str
    kind: ArgKind
    description: str
    required: bool = False


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args: tuple[ArgSpec, ...]
    mutating: bool = False

    def manifest_entry(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "mutating": self.mutating,
            "args": [
                {
                    "name": a.name,
                    "kind": a.kind,
                    "description": a.description,
                    "required": a.required,
                }
                for a in self.args
            ],
        }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="pgoa_collect_profile",
        description=(
            "Collect a performance profile for a completed Slurm job and store it in the "
            "experiment store. Optionally merges Nsight Compute (NCU) CSV and/or LIKWID "
            "text output. Returns the run_id needed for subsequent pgoa_* calls."
        ),
        args=(
            ArgSpec("workload_id", "string", "Unique workload identifier.", required=True),
            ArgSpec("job_id", "integer", "Slurm job ID.", required=True),
            ArgSpec("kpi_metric", "string", "Primary KPI name (e.g. elapsed_s).", required=True),
            ArgSpec("kpi_value", "string", "Current KPI value (numeric string).", required=True),
            ArgSpec("kpi_unit", "string", "Unit for the KPI (e.g. seconds).", required=True),
            ArgSpec("run_type", "string", "\"baseline\" or \"iteration\"."),
            ArgSpec("iteration", "integer", "Iteration number (if run_type=iteration)."),
            ArgSpec("ncu_csv_path", "string", "Path to Nsight Compute --csv output file."),
            ArgSpec("likwid_output_path", "string", "Path to likwid-perfctr text output file."),
            ArgSpec("lower_is_better", "boolean", "True when lower KPI is better."),
        ),
    ),
    ToolSpec(
        name="pgoa_analyze",
        description=(
            "Analyze a stored ProfileBundle and return the primary bottleneck "
            "(memory_bound_gpu, compute_bound_gpu, mpi_binding, high_rss, etc.) "
            "with a recommended action hint."
        ),
        args=(
            ArgSpec("workload_id", "string", "Workload identifier.", required=True),
            ArgSpec("run_id", "string", "Run ID returned by pgoa_collect_profile.", required=True),
        ),
    ),
    ToolSpec(
        name="pgoa_apply_binding",
        description=(
            "Validate Slurm binding parameters and rewrite the job script's #SBATCH "
            "directives in-place or to a new output path. Returns the output path and "
            "a list of changes applied."
        ),
        args=(
            ArgSpec("workload_id", "string", "Workload identifier.", required=True),
            ArgSpec("run_id", "string", "Run ID for provenance tracking.", required=True),
            ArgSpec("job_script_path", "string", "Absolute path to source job script.", required=True),
            ArgSpec("output_script_path", "string", "Absolute path for the modified script.", required=True),
            ArgSpec("ntasks_per_node", "integer", "Value for --ntasks-per-node."),
            ArgSpec("ntasks_per_socket", "integer", "Value for --ntasks-per-socket."),
            ArgSpec("cpus_per_task", "integer", "Value for --cpus-per-task."),
            ArgSpec("mem_bind", "string", "Slurm mem-bind policy (local/none/prefer/bind)."),
            ArgSpec("cpu_bind", "string", "Slurm cpu-bind policy (cores/threads/sockets/rank/none)."),
            ArgSpec("exclusive", "boolean", "Add --exclusive flag."),
        ),
        mutating=True,
    ),
    ToolSpec(
        name="pgoa_compare_runs",
        description=(
            "Compute a KPI delta between two stored runs and classify the change as "
            "improved / degraded / neutral. Returns kpi_delta_pct and secondary metric deltas."
        ),
        args=(
            ArgSpec("workload_id", "string", "Workload identifier.", required=True),
            ArgSpec("from_run_id", "string", "Baseline run ID.", required=True),
            ArgSpec("to_run_id", "string", "Iteration run ID to compare against baseline.", required=True),
            ArgSpec("action_applied", "string", "Short description of the change applied.", required=True),
        ),
    ),
    ToolSpec(
        name="pgoa_store_info",
        description=(
            "List all experiment runs for a workload (baseline first, then iterations "
            "sorted by iteration number). Returns run metadata without loading profile data."
        ),
        args=(
            ArgSpec("workload_id", "string", "Workload identifier.", required=True),
        ),
    ),
)

_SPEC_BY_NAME: dict[str, ToolSpec] = {s.name: s for s in TOOL_SPECS}


# ---------------------------------------------------------------------------
# Tool manifest
# ---------------------------------------------------------------------------


def tool_manifest() -> dict[str, object]:
    return {
        "tool_count": len(TOOL_SPECS),
        "mutating_tools": [s.name for s in TOOL_SPECS if s.mutating],
        "tools": [s.manifest_entry() for s in TOOL_SPECS],
    }


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    settings: AssistantSettings | None = None,
) -> dict[str, object]:
    if name not in _SPEC_BY_NAME:
        return {"ok": False, "error": f"Unknown tool: {name!r}"}
    cfg = settings or load_settings()
    store = _make_store(cfg)
    try:
        return _HANDLERS[name](cfg, store, arguments)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _make_store(settings: AssistantSettings) -> ExperimentStore:
    path = Path(settings.pgoa_store_path).expanduser()
    return ExperimentStore(path)


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------


def _handle_collect_profile(
    settings: AssistantSettings, store: ExperimentStore, args: dict[str, Any]
) -> dict[str, object]:
    workload_id: str = _req_str(args, "workload_id")
    job_id: int = _req_int(args, "job_id")
    kpi = build_kpi(
        _req_str(args, "kpi_metric"),
        _req_str(args, "kpi_value"),
        _req_str(args, "kpi_unit"),
        lower_is_better=bool(args.get("lower_is_better", True)),
    )
    run_type = args.get("run_type") or "baseline"
    iteration = args.get("iteration")
    handle, bundle = collect_profile(
        settings,
        store,
        workload_id=workload_id,
        job_id=job_id,
        kpi=kpi,
        run_type=run_type,
        iteration=iteration,
        ncu_csv_path=str(args["ncu_csv_path"]) if args.get("ncu_csv_path") else None,
        likwid_output_path=(
            str(args["likwid_output_path"]) if args.get("likwid_output_path") else None
        ),
    )

    return {
        "ok": True,
        "run_id": handle.run_id,
        "run_type": run_type,
        "slurm": bundle.slurm.model_dump() if bundle.slurm else None,
        "roofline": bundle.compute.roofline_position if bundle.compute else None,
    }


def _handle_analyze(
    settings: AssistantSettings, store: ExperimentStore, args: dict[str, Any]
) -> dict[str, object]:
    workload_id = _req_str(args, "workload_id")
    run_id = _req_str(args, "run_id")
    report = analyze_run(store, workload_id=workload_id, run_id=run_id)
    return {
        "ok": True,
        "run_id": run_id,
        "primary_bottleneck": report.primary_bottleneck,
        "details": report.details,
        "recommended_action_hint": report.recommended_action_hint,
    }


def _handle_apply_binding(
    settings: AssistantSettings, store: ExperimentStore, args: dict[str, Any]
) -> dict[str, object]:
    workload_id = _req_str(args, "workload_id")
    run_id = _req_str(args, "run_id")
    try:
        output_path, changes = apply_binding_change(
            settings,
            store,
            workload_id=workload_id,
            run_id=run_id,
            job_script_path=_req_str(args, "job_script_path"),
            output_script_path=_req_str(args, "output_script_path"),
            ntasks_per_node=args.get("ntasks_per_node"),
            ntasks_per_socket=args.get("ntasks_per_socket"),
            cpus_per_task=args.get("cpus_per_task"),
            mem_bind=args.get("mem_bind"),
            cpu_bind=args.get("cpu_bind"),
            exclusive=bool(args.get("exclusive", False)),
        )
    except ValueError as exc:
        return {"ok": False, "validation_errors": [part.strip() for part in str(exc).split(";")]}
    return {"ok": True, "new_script_path": output_path, "changes_applied": changes}


def _handle_compare_runs(
    settings: AssistantSettings, store: ExperimentStore, args: dict[str, Any]
) -> dict[str, object]:
    delta = compare_runs(
        store,
        workload_id=_req_str(args, "workload_id"),
        from_run_id=_req_str(args, "from_run_id"),
        to_run_id=_req_str(args, "to_run_id"),
        action_applied=_req_str(args, "action_applied"),
    )
    return {
        "ok": True,
        "from_run_id": delta.from_run_id,
        "to_run_id": delta.to_run_id,
        "kpi_delta_pct": delta.kpi_delta_pct,
        "kpi_direction": delta.kpi_direction,
        "secondary_deltas": delta.secondary_deltas,
    }


def _handle_store_info(
    settings: AssistantSettings, store: ExperimentStore, args: dict[str, Any]
) -> dict[str, object]:
    workload_id = _req_str(args, "workload_id")
    runs = list_runs(store, workload_id=workload_id)
    return {
        "ok": True,
        "workload_id": workload_id,
        "run_count": len(runs),
        "runs": [
            {
                "run_id": h.run_id,
                "run_type": h.run_type,
                "iteration": h.iteration,
                "path": str(h.path),
            }
            for h in runs
        ],
    }


_HANDLERS = {
    "pgoa_collect_profile": _handle_collect_profile,
    "pgoa_analyze": _handle_analyze,
    "pgoa_apply_binding": _handle_apply_binding,
    "pgoa_compare_runs": _handle_compare_runs,
    "pgoa_store_info": _handle_store_info,
}


# ---------------------------------------------------------------------------
# OpenCode project sync — writes pgoa.ts and opencode.json
# ---------------------------------------------------------------------------

_PYTHON_HELPER_TS = """\
import { spawn } from "child_process"
import * as path from "path"
import * as fs from "fs"

function findPython(root: string): string {
  const candidates = [
    path.join(root, ".venv", "bin", "python"),
    path.join(root, "backend", ".venv", "bin", "python"),
    "python3",
  ]
  for (const p of candidates) {
    if (p.startsWith("/") || p.startsWith(".")) {
      if (fs.existsSync(p)) return p
    } else {
      return p
    }
  }
  return "python3"
}

export async function runPythonTool(
  toolName: string,
  args: Record<string, unknown>,
  context: { root: string },
): Promise<unknown> {
  const root = context.root
  const python = findPython(root)
  const pythonPath = path.join(root, "backend", "src")

  const env = {
    ...process.env,
    PYTHONPATH: pythonPath,
    HPC_ASSISTANT_FILESYSTEM_ROOTS:
      process.env.HPC_ASSISTANT_FILESYSTEM_ROOTS || root,
  }

  const output = await new Promise<{ exitCode: number; stdout: string; stderr: string }>(
    (resolve, reject) => {
      const child = spawn(
        python,
        [
          "-m",
          "claw_backend.opencode_tools",
          "run",
          toolName,
          JSON.stringify(args ?? {}),
        ],
        { cwd: root, env },
      )
      let stdout = ""
      let stderr = ""
      child.stdout.on("data", (chunk) => { stdout += String(chunk) })
      child.stderr.on("data", (chunk) => { stderr += String(chunk) })
      child.on("error", reject)
      child.on("close", (code) => { resolve({ exitCode: code ?? 1, stdout, stderr }) })
    },
  )

  if (output.exitCode !== 0) {
    throw new Error(
      output.stderr.trim() || `python tool ${toolName} exited with code ${output.exitCode}`,
    )
  }

  try {
    return JSON.parse(output.stdout)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`python tool ${toolName} returned invalid JSON: ${message}`)
  }
}
"""


def _render_tool_ts(spec: ToolSpec) -> str:
    args_lines = []
    for a in spec.args:
        expr = {
            "string": "tool.schema.string()",
            "boolean": "tool.schema.boolean()",
            "integer": "tool.schema.number().int()",
        }[a.kind]
        expr = f"{expr}.describe({json.dumps(a.description)})"
        if not a.required:
            expr = f"{expr}.optional()"
        args_lines.append(f"    {a.name}: {expr},")
    args_body = "\n".join(args_lines)
    # Strip the "pgoa_" prefix for the TS export name
    export_name = spec.name[len("pgoa_"):]
    return (
        f"export const {export_name} = tool({{\n"
        f"  description: {json.dumps(spec.description)},\n"
        "  args: {\n"
        f"{args_body}\n"
        "  },\n"
        "  async execute(args, context) {\n"
        f"    return await runPythonTool({json.dumps(spec.name)}, args, context)\n"
        "  },\n"
        "})\n\n"
    )


def sync_opencode_project(root: Path | None = None) -> list[Path]:
    if root is None:
        root = Path(__file__).resolve().parents[3]

    tools_dir = root / ".opencode" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # _python.ts helper
    python_ts = tools_dir / "_python.ts"
    python_ts.write_text(_PYTHON_HELPER_TS, encoding="utf-8")
    written.append(python_ts)

    # pgoa.ts — all 5 PGOA tools
    pgoa_lines = [
        'import { tool } from "@opencode/tool"\n',
        'import { runPythonTool } from "./_python"\n\n',
    ]
    for spec in TOOL_SPECS:
        pgoa_lines.append(_render_tool_ts(spec))
    pgoa_ts = tools_dir / "pgoa.ts"
    pgoa_ts.write_text("".join(pgoa_lines), encoding="utf-8")
    written.append(pgoa_ts)

    # opencode.json
    config = {
        "$schema": "https://opencode.ai/config.schema.json",
        "permission": {
            "bash": "ask",
            "pgoa_apply_binding": "ask",
        },
    }
    opencode_json = root / "opencode.json"
    opencode_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    written.append(opencode_json)

    return written


# ---------------------------------------------------------------------------
# CLI — python -m claw_backend.opencode_tools show|run|sync
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, execute, and sync the Python-backed OpenCode PGOA tools."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show", help="Print the OpenCode tool manifest as JSON.")

    run_p = sub.add_parser("run", help="Execute one tool and print JSON output.")
    run_p.add_argument("tool_name")
    run_p.add_argument("arguments_json", nargs="?", default="{}")

    sync_p = sub.add_parser("sync", help="Write OpenCode tool shims and opencode.json.")
    sync_p.add_argument("--root", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "show":
        print(json.dumps(tool_manifest(), indent=2))
        return 0

    if args.command == "run":
        try:
            arguments = json.loads(args.arguments_json)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"invalid JSON: {exc.msg}"}))
            return 0
        if not isinstance(arguments, dict):
            print(json.dumps({"ok": False, "error": "arguments_json must decode to an object"}))
            return 0
        print(json.dumps(execute_tool(args.tool_name, arguments)))
        return 0

    written = sync_opencode_project(args.root)
    print(json.dumps({"written": [str(p) for p in written]}, indent=2))
    return 0


def _req_str(args: dict[str, Any], key: str) -> str:
    val = args.get(key)
    if val is None:
        raise ValueError(f"{key!r} is required")
    return str(val)


def _req_int(args: dict[str, Any], key: str) -> int:
    val = args.get(key)
    if val is None:
        raise ValueError(f"{key!r} is required")
    return int(val)


if __name__ == "__main__":
    import sys
    sys.exit(main())
