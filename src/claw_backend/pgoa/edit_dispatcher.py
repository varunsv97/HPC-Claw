"""Dispatch code-level optimization edits to the OpenCode CLI.

When the PGOA agent detects a bottleneck that requires source code changes,
:func:`dispatch_code_edit` is called to:

1. Record the current git HEAD (pre-edit commit).
2. Spawn ``opencode run <prompt>`` in the project directory.
3. Capture the unified diff of all changes via ``git diff <pre-commit>``.
4. Return an :class:`~claw_backend.pgoa.schema.EditRecord` that links
   the edit to the profile run that triggered it.

The caller (``PGOAAgent``) is responsible for persisting the ``EditRecord``
via ``ExperimentStore.save_edit_record`` and later linking it to the
post-edit ``DeltaReport`` via ``ExperimentStore.save_edit_map``.
"""

from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from claw_backend.config import AssistantSettings
from claw_backend.pgoa.schema import BottleneckReport, EditRecord, ProfileBundle
from claw_backend.project_config import ProjectConfig

log = logging.getLogger(__name__)

# OpenCode code-edit sessions can be slow (LLM + file I/O); allow generous timeout.
_OPENCODE_TIMEOUT_S = 300


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dispatch_code_edit(
    hypothesis: str,
    opencode_prompt: str,
    bottleneck: BottleneckReport,
    profile: ProfileBundle,
    project_config: ProjectConfig,
    settings: AssistantSettings,
    cwd: Path | None = None,
) -> EditRecord:
    """Spawn ``opencode run`` with *opencode_prompt* and capture the resulting diff.

    Parameters
    ----------
    hypothesis:
        The DSPy-generated optimization hypothesis (stored for traceability).
    opencode_prompt:
        The full, self-contained instruction for OpenCode.
    bottleneck:
        The ``BottleneckReport`` that triggered this edit.
    profile:
        The ``ProfileBundle`` from the run that triggered this edit
        (provides ``run_id`` for linking).
    project_config:
        Loaded ``agent.toml`` — used for ``edit_roots`` (informational).
    settings:
        Backend settings.
    cwd:
        Working directory for the OpenCode process.  Defaults to ``Path.cwd()``.
    """
    edit_id = str(uuid4())
    work_dir = (cwd or Path.cwd()).resolve()

    pre_commit = _git_head(work_dir)

    log.info(
        "Dispatching OpenCode edit  edit_id=%s  bottleneck=%s  cwd=%s",
        edit_id,
        bottleneck.primary_bottleneck,
        work_dir,
    )
    log.debug("OpenCode prompt (first 200 chars): %s", opencode_prompt[:200])

    proc = _run_opencode(opencode_prompt, work_dir)

    if proc.returncode != 0:
        log.warning(
            "opencode exited with code %d: %s",
            proc.returncode,
            proc.stderr.strip()[:200],
        )

    git_diff, files_modified = _capture_diff(work_dir, pre_commit)
    session_id = _extract_session_id(proc.stdout)

    return EditRecord(
        edit_id=edit_id,
        timestamp=datetime.now(timezone.utc),
        hypothesis=hypothesis,
        bottleneck_type=bottleneck.primary_bottleneck,
        prompt_sent_to_opencode=opencode_prompt,
        opencode_session_id=session_id,
        files_modified=files_modified,
        git_diff=git_diff,
        linked_run_id_before=profile.run_id,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _run_opencode(
    prompt: str,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run ``opencode run <prompt>`` non-interactively."""
    cmd = ["opencode", "run", prompt]
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_OPENCODE_TIMEOUT_S,
        )
    except FileNotFoundError:
        log.warning("opencode CLI not found in PATH — edit was not applied")
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr="opencode not found in PATH"
        )
    except subprocess.TimeoutExpired:
        log.warning("opencode run timed out after %ds", _OPENCODE_TIMEOUT_S)
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout="", stderr=f"timed out after {_OPENCODE_TIMEOUT_S}s"
        )


def _git_head(cwd: Path) -> str | None:
    """Return the current HEAD commit SHA, or ``None`` if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _capture_diff(
    cwd: Path, pre_commit: str | None
) -> tuple[str | None, list[str]]:
    """Return ``(unified_diff, changed_file_paths)`` since *pre_commit*.

    Falls back to ``(None, [])`` if the repository has no git history or
    ``git diff`` fails for any reason.
    """
    if pre_commit is None:
        # Also try comparing against working-tree changes (unstaged)
        return _capture_unstaged_diff(cwd)

    try:
        names_result = subprocess.run(
            ["git", "diff", "--name-only", pre_commit],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        files = [f for f in names_result.stdout.splitlines() if f]

        diff_result = subprocess.run(
            ["git", "diff", pre_commit],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        diff_text = diff_result.stdout if diff_result.returncode == 0 else None
        return diff_text, files
    except Exception as exc:
        log.debug("git diff failed: %s", exc)
        return None, []


def _capture_unstaged_diff(cwd: Path) -> tuple[str | None, list[str]]:
    """Fallback: capture uncommitted working-tree changes (``git diff HEAD``)."""
    try:
        diff_result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if diff_result.returncode != 0:
            return None, []
        diff_text = diff_result.stdout
        files = [
            line[len("+++ b/"):]
            for line in diff_text.splitlines()
            if line.startswith("+++ b/")
        ]
        return diff_text or None, files
    except Exception:
        return None, []


def _extract_session_id(stdout: str) -> str | None:
    """Best-effort extraction of an OpenCode session UUID from CLI output."""
    match = re.search(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        stdout,
        re.IGNORECASE,
    )
    return match.group(0) if match else None
