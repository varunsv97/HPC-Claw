"""Slurm binding action space for PGOA Phase 1."""

from __future__ import annotations

import re

VALID_MEM_BIND = {"local", "none", "prefer", "bind"}
VALID_CPU_BIND = {"cores", "threads", "sockets", "rank", "none"}

# Mapping: parameter key -> (#SBATCH flag, value-format callable)
# Value format callable receives the parameter value and returns the flag value string.
_SBATCH_FLAG: dict[str, str] = {
    "ntasks_per_node": "--ntasks-per-node",
    "ntasks_per_socket": "--ntasks-per-socket",
    "cpus_per_task": "--cpus-per-task",
    "mem_bind": "--mem-bind",
    "cpu_bind": "--cpu-bind",
    "exclusive": "--exclusive",
}


def validate_slurm_binding_params(
    ntasks_per_node: int | None = None,
    ntasks_per_socket: int | None = None,
    cpus_per_task: int | None = None,
    mem_bind: str | None = None,
    cpu_bind: str | None = None,
    exclusive: bool = False,
) -> list[str]:
    """Return a list of validation errors (empty list means valid)."""
    errors: list[str] = []

    for name, value in [
        ("ntasks_per_node", ntasks_per_node),
        ("ntasks_per_socket", ntasks_per_socket),
        ("cpus_per_task", cpus_per_task),
    ]:
        if value is not None:
            if not isinstance(value, int) or value < 1:
                errors.append(f"{name} must be a positive integer, got {value!r}")

    if mem_bind is not None and mem_bind not in VALID_MEM_BIND:
        errors.append(
            f"mem_bind must be one of {sorted(VALID_MEM_BIND)!r}, got {mem_bind!r}"
        )

    if cpu_bind is not None and cpu_bind not in VALID_CPU_BIND:
        errors.append(
            f"cpu_bind must be one of {sorted(VALID_CPU_BIND)!r}, got {cpu_bind!r}"
        )

    return errors


def apply_slurm_binding(
    job_script_content: str,
    ntasks_per_node: int | None = None,
    ntasks_per_socket: int | None = None,
    cpus_per_task: int | None = None,
    mem_bind: str | None = None,
    cpu_bind: str | None = None,
    exclusive: bool = False,
) -> tuple[str, list[str]]:
    """Apply Slurm binding directives to a job script.

    Returns (modified_script, list_of_changes_applied).
    For each directive:
      - If it already exists in the script, replace it in-place.
      - If it doesn't exist, append after the last #SBATCH line.
    """
    changes: list[str] = []
    params: dict[str, str | None] = {
        "ntasks_per_node": str(ntasks_per_node) if ntasks_per_node is not None else None,
        "ntasks_per_socket": str(ntasks_per_socket) if ntasks_per_socket is not None else None,
        "cpus_per_task": str(cpus_per_task) if cpus_per_task is not None else None,
        "mem_bind": mem_bind,
        "cpu_bind": cpu_bind,
        "exclusive": "" if exclusive else None,  # flag only, no value
    }

    lines = job_script_content.splitlines(keepends=True)

    for param_key, value in params.items():
        if value is None:
            continue
        flag = _SBATCH_FLAG[param_key]
        if param_key == "exclusive":
            new_directive = f"#SBATCH {flag}"
        else:
            new_directive = f"#SBATCH {flag}={value}"

        # Try to replace an existing directive in the script
        # Match: #SBATCH --flag=<anything> or #SBATCH --flag <anything>
        flag_escaped = re.escape(flag)
        pattern = re.compile(
            rf"^(\s*#SBATCH\s+{flag_escaped})(?:=\S*|\s+\S*)?\s*$",
            re.IGNORECASE,
        )
        replaced = False
        for i, line in enumerate(lines):
            if pattern.match(line.rstrip("\n\r")):
                old_line = line.rstrip("\n\r")
                lines[i] = new_directive + "\n"
                changes.append(f"replaced: {old_line!r} -> {new_directive!r}")
                replaced = True
                break

        if not replaced:
            # Insert after the last #SBATCH line
            last_sbatch_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("#SBATCH"):
                    last_sbatch_idx = i
            insert_at = last_sbatch_idx + 1 if last_sbatch_idx >= 0 else 0
            lines.insert(insert_at, new_directive + "\n")
            changes.append(f"added: {new_directive!r}")

    return "".join(lines), changes
