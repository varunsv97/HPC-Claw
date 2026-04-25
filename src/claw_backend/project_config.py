"""Project-level configuration loaded from ``agent.toml``.

The file is searched in the current working directory first, then up the
directory tree until it is found or the filesystem root is reached.  An
explicit path can be provided to override the search.

Example ``agent.toml``::

    [project]
    name = "my-hpc-project"
    workload_script = "run.sh"

    [paths]
    readonly    = ["Makefile", "*.lock"]
    data        = ["/scratch/data/"]
    edit_roots  = ["src/"]

    [pgoa]
    primary_kpi       = "wall_time_s"
    kpi_unit          = "seconds"
    lower_is_better   = true
    max_iterations    = 5
    kpi_threshold_pct = 2.0
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ProjectPathsConfig(BaseModel):
    """Filesystem paths relevant to the project."""

    # Write-protected globs — backend never modifies files matching these.
    readonly: list[str] = []
    # Input data directories/files — PGOA may read but never write.
    data: list[str] = []
    # Source directories that OpenCode is allowed to edit.
    edit_roots: list[str] = []


class PGOAProjectConfig(BaseModel):
    """PGOA optimization loop settings."""

    primary_kpi: str = "wall_time_s"
    kpi_unit: str = "seconds"
    lower_is_better: bool = True
    max_iterations: int = 5
    kpi_threshold_pct: float = 2.0


class ProjectConfig(BaseModel):
    """Root model representing the full contents of ``agent.toml``."""

    name: str = "unnamed"
    # Path to the primary Slurm job script, relative to the config file.
    workload_script: str | None = None
    # Absolute path of the config file that was loaded (set by the loader).
    config_file: Path | None = None

    paths: ProjectPathsConfig = ProjectPathsConfig()
    pgoa: PGOAProjectConfig = PGOAProjectConfig()

    @property
    def project_root(self) -> Path | None:
        """Directory containing ``agent.toml``, or ``None`` if not loaded from disk."""
        return self.config_file.parent if self.config_file is not None else None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_FILENAME = "agent.toml"


def load_project_config(path: Path | str | None = None) -> ProjectConfig:
    """Load project config from *path*, or search the cwd and its parents.

    Returns a default :class:`ProjectConfig` when no file is found rather
    than raising — callers that need a config file should check
    ``result.config_file is not None``.
    """
    if path is not None:
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            return ProjectConfig()
        return _load_file(config_path)

    # Walk up from cwd until we find agent.toml or hit the root
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        p = candidate / _CONFIG_FILENAME
        if p.exists():
            return _load_file(p)

    return ProjectConfig()


def _load_file(p: Path) -> ProjectConfig:
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    cfg = _parse(data)
    return cfg.model_copy(update={"config_file": p})


def _parse(data: dict) -> ProjectConfig:
    project = data.get("project", {})
    paths = data.get("paths", {})
    pgoa = data.get("pgoa", {})

    return ProjectConfig(
        name=project.get("name", "unnamed"),
        workload_script=project.get("workload_script"),
        paths=ProjectPathsConfig(
            readonly=paths.get("readonly", []),
            data=paths.get("data", []),
            edit_roots=paths.get("edit_roots", []),
        ),
        pgoa=PGOAProjectConfig(
            primary_kpi=pgoa.get("primary_kpi", "wall_time_s"),
            kpi_unit=pgoa.get("kpi_unit", "seconds"),
            lower_is_better=pgoa.get("lower_is_better", True),
            max_iterations=pgoa.get("max_iterations", 5),
            kpi_threshold_pct=pgoa.get("kpi_threshold_pct", 2.0),
        ),
    )
