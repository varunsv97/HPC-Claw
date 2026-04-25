"""OpenAI function-call schemas for the PGOA agent loop.

These schemas are passed to the LLM API as tool definitions. All implementations
live in services.py; the loop dispatches calls there directly.
"""

from __future__ import annotations

from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "collect_slurm_profile",
            "description": (
                "Collect Slurm job telemetry (sacct + sstat) for a completed or running job."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Slurm job ID"},
                    "workload_id": {"type": "string"},
                    "run_id": {"type": "string", "description": "Run ID from create_run"},
                    "kpi_metric": {"type": "string"},
                    "kpi_value": {"type": "number"},
                    "kpi_unit": {"type": "string"},
                    "lower_is_better": {"type": "boolean"},
                },
                "required": ["job_id", "workload_id", "run_id", "kpi_metric", "kpi_value", "kpi_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "collect_ncu_profile",
            "description": "Parse an Nsight Compute --csv output file into a ProfileBundle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ncu_csv_path": {"type": "string"},
                    "workload_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "kpi_metric": {"type": "string"},
                    "kpi_value": {"type": "number"},
                    "kpi_unit": {"type": "string"},
                    "lower_is_better": {"type": "boolean"},
                },
                "required": ["ncu_csv_path", "workload_id", "run_id", "kpi_metric", "kpi_value", "kpi_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "collect_likwid_profile",
            "description": "Parse a likwid-perfctr text output file into a ProfileBundle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "likwid_output_path": {"type": "string"},
                    "workload_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "kpi_metric": {"type": "string"},
                    "kpi_value": {"type": "number"},
                    "kpi_unit": {"type": "string"},
                    "lower_is_better": {"type": "boolean"},
                },
                "required": ["likwid_output_path", "workload_id", "run_id", "kpi_metric", "kpi_value", "kpi_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_bottlenecks",
            "description": "Analyse a saved ProfileBundle and return the primary bottleneck report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workload_id": {"type": "string"},
                    "run_id": {"type": "string"},
                },
                "required": ["workload_id", "run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_slurm_action",
            "description": "Record a proposed Slurm binding change with rationale and expected improvement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workload_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "action_name": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "description": "Key-value pairs matching apply_slurm_action parameters",
                    },
                    "rationale": {"type": "string"},
                    "expected_improvement_pct": {"type": "number"},
                },
                "required": ["workload_id", "run_id", "action_name", "parameters", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_slurm_action",
            "description": (
                "Apply a Slurm binding change to a job script. "
                "Reads the script from job_script_path (must be within allowed filesystem roots), "
                "applies changes, and writes the modified script to a new path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_script_path": {"type": "string"},
                    "output_script_path": {"type": "string"},
                    "workload_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "ntasks_per_node": {"type": "integer"},
                    "ntasks_per_socket": {"type": "integer"},
                    "cpus_per_task": {"type": "integer"},
                    "mem_bind": {"type": "string"},
                    "cpu_bind": {"type": "string"},
                    "exclusive": {"type": "boolean"},
                },
                "required": ["job_script_path", "output_script_path", "workload_id", "run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_runs",
            "description": "Compute a delta report between two runs (baseline vs iteration).",
            "parameters": {
                "type": "object",
                "properties": {
                    "workload_id": {"type": "string"},
                    "from_run_id": {"type": "string"},
                    "to_run_id": {"type": "string"},
                    "action_applied": {"type": "string"},
                },
                "required": ["workload_id", "from_run_id", "to_run_id", "action_applied"],
            },
        },
    },
]
