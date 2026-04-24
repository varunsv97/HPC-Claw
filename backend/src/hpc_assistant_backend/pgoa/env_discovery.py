"""Software environment discovery — detects Lmod and catalogs available modules.

Two sources of data, both best-effort:
  1. ``module --terse avail``   — Core-tier modules (immediately loadable, fast)
  2. ``module --terse spider``  — Full hierarchy including compiler/MPI-gated
                                  modules (may take 20–60 s on large clusters)

Classification hierarchy stored in SoftwareEnvironment:
  compilers     → gcc, intel, nvhpc, aocc, llvm, cce, …
  mpi_libraries → openmpi, intelmpi, mvapich2, mpich, hpcx, …
  gpu_toolkits  → cuda, rocm, hip, cudnn, …
  math_libraries → mkl, openblas, fftw, scalapack, armpl, …
  io_libraries  → hdf5, netcdf, adios2, pnetcdf, …
  applications  → gromacs, openfoam, python, pytorch, …
  other_modules → everything else

Toolchain derivation:
  All (compiler × mpi) pairs and (compiler × gpu) pairs are enumerated
  using the default/newest version of each module, producing Toolchain
  objects with ready-made load_sequence lists.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Literal

from hpc_assistant_backend.config import AssistantSettings
from hpc_assistant_backend.pgoa.schema import ModuleInfo, SoftwareEnvironment, Toolchain

log = logging.getLogger(__name__)

# Longer timeout for spider — it walks the full hierarchy
_SPIDER_TIMEOUT_S = 90.0

# ---------------------------------------------------------------------------
# Module system detection
# ---------------------------------------------------------------------------

ModuleSystemKind = Literal["lmod", "tmod", "none"]

# Common paths where the Lmod bash init script lives
_LMOD_INIT_CANDIDATES = [
    "/etc/profile.d/lmod.sh",
    "/usr/share/lmod/lmod/init/bash",
    "/opt/apps/lmod/lmod/init/bash",
    "/software/lmod/lmod/init/bash",
    "/apps/lmod/lmod/init/bash",
    "/usr/local/lmod/lmod/init/bash",
    "/opt/lmod/lmod/init/bash",
    "/cm/shared/apps/lmod/lmod/init/bash",  # Bright Cluster Manager
]


def detect_module_system(settings: AssistantSettings) -> tuple[ModuleSystemKind, str | None]:
    """Return (kind, version_string) for the active module system.

    Checks environment variables first (fast), then probes the command line.
    """
    env = os.environ

    # Lmod sets LMOD_VERSION when it has been sourced
    lmod_ver = env.get("LMOD_VERSION")
    if lmod_ver:
        return "lmod", lmod_ver.strip()

    # LMOD_CMD is set even when the init hasn't been sourced in this shell
    lmod_cmd = env.get("LMOD_CMD")
    if lmod_cmd and Path(lmod_cmd).exists():
        ver = _lmod_version_from_cmd(lmod_cmd, settings)
        return "lmod", ver

    # Fall back to running module --version via bash
    version_out = _run_bash_module("--version", settings, timeout=5.0)
    if version_out:
        lo = version_out.lower()
        if "lmod" in lo:
            m = re.search(r"(?:version\s+|lmod\s+)([0-9]+\.[0-9]+[\d.]*)", lo)
            return "lmod", m.group(1) if m else None
        if "modules" in lo:
            m = re.search(r"release\s+([0-9]+\.[0-9]+[\d.]*)", lo)
            return "tmod", m.group(1) if m else None

    # MODULESHOME without LMOD markers → traditional Tmod
    if env.get("MODULESHOME"):
        return "tmod", None

    return "none", None


def _lmod_version_from_cmd(lmod_cmd: str, settings: AssistantSettings) -> str | None:
    """Query Lmod version by invoking the Lua script directly."""
    try:
        r = subprocess.run(
            ["lua", lmod_cmd, "bash", "--version"],
            capture_output=True, text=True, check=False,
            timeout=settings.command_timeout_seconds,
        )
        m = re.search(r"[Vv]ersion\s*([0-9]+\.[0-9]+[\d.]*)", r.stdout + r.stderr)
        return m.group(1) if m else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lmod command runner
# ---------------------------------------------------------------------------

def _find_lmod_init() -> str | None:
    """Return path to the Lmod bash init script, derived from env or well-known paths."""
    lmod_cmd = os.environ.get("LMOD_CMD")
    if lmod_cmd:
        # LMOD_CMD is .../libexec/lmod; init/bash is at .../init/bash
        candidate = Path(lmod_cmd).parent.parent / "init" / "bash"
        if candidate.exists():
            return str(candidate)
    for path in _LMOD_INIT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _run_bash_module(subcmd: str, settings: AssistantSettings, timeout: float | None = None) -> str:
    """Execute ``module <subcmd>`` via bash and return combined stdout+stderr.

    Automatically prepends ``--terse`` unless the caller already included it.
    Sources the Lmod init script when LMOD_VERSION is not already in the env.
    """
    effective_timeout = timeout if timeout is not None else settings.command_timeout_seconds

    # Add --terse unless already present
    if "--terse" not in subcmd and subcmd not in ("--version",):
        subcmd = f"--terse {subcmd}"

    init = _find_lmod_init()
    if init:
        bash_cmd = f"source {shlex.quote(init)} 2>/dev/null && module {subcmd} 2>&1"
    elif os.environ.get("LMOD_VERSION"):
        # Module is already a shell function in the parent; bash -l may re-source it
        bash_cmd = f"module {subcmd} 2>&1"
    else:
        bash_cmd = f"module {subcmd} 2>&1"

    try:
        r = subprocess.run(
            ["bash", "-c", bash_cmd],
            capture_output=True,
            text=True,
            check=False,
            timeout=effective_timeout,
            env=os.environ,
        )
        return (r.stdout + r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.debug("module command failed (%s): %s", subcmd, exc)
        return ""


# ---------------------------------------------------------------------------
# Module list parsing
# ---------------------------------------------------------------------------

# Annotations Lmod appends after a module name
_ANNOTATION_RE = re.compile(r"\s*\([^)]*\)")


def _parse_module_list(raw: str) -> list[tuple[str, str, bool]]:
    """Parse ``module --terse avail`` or ``module --terse spider`` output.

    Returns a list of ``(name, version, is_default)`` tuples.
    Lines that are path headers, separators, or Lmod info messages are skipped.
    """
    results: list[tuple[str, str, bool]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip path/section headers (end with ':')
        if line.endswith(":"):
            continue
        # Skip separator lines and Lmod info messages
        if line.startswith("-") or line.startswith("=") or line.startswith("For detailed"):
            continue

        is_default = bool(re.search(r"\((D|default)\)", line, re.IGNORECASE))
        # Strip all annotations
        clean = _ANNOTATION_RE.sub("", line).strip()
        if not clean:
            continue

        if "/" in clean:
            name, _, version = clean.partition("/")
            results.append((name.strip(), version.strip(), is_default))
        else:
            results.append((clean.strip(), "", is_default))

    return results


def _version_key(v: str) -> tuple[int | str, ...]:
    """Sort key that orders version strings numerically where possible.

    "10.0" > "9.0", "4.1.6" > "4.1.5", non-numeric parts sort after numeric.
    """
    parts: list[int | str] = []
    for segment in re.split(r"[.\-_]", v):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(segment)
    return tuple(parts)


def _build_module_map(entries: list[tuple[str, str, bool]]) -> dict[str, ModuleInfo]:
    """Collapse (name, version, is_default) entries into a name → ModuleInfo dict.

    Key is lowercase for case-insensitive deduplication; stored name preserves
    the original casing of the first occurrence.
    """
    groups: dict[str, dict] = {}

    for name, version, is_default in entries:
        key = name.lower()
        if key not in groups:
            groups[key] = {"name": name, "versions": [], "default_version": None}
        if version and version not in groups[key]["versions"]:
            groups[key]["versions"].append(version)
        if is_default and version:
            groups[key]["default_version"] = version

    result: dict[str, ModuleInfo] = {}
    for key, info in groups.items():
        versions = sorted(info["versions"], key=_version_key)
        default = info["default_version"]
        if not default and versions:
            default = versions[-1]  # newest as fallback
        load_ver = default or (versions[0] if versions else "")
        load_cmd = (
            f"module load {info['name']}/{load_ver}"
            if load_ver
            else f"module load {info['name']}"
        )
        result[key] = ModuleInfo(
            name=info["name"],
            versions=versions,
            default_version=default,
            load_cmd=load_cmd,
        )

    return result


# ---------------------------------------------------------------------------
# Module classification
# ---------------------------------------------------------------------------

# Sets use lowercase; prefix matching is done separately below
_COMPILER_NAMES = frozenset({
    "gcc", "gcccore", "g++", "gfortran",
    "intel", "intel-compilers", "intel-compilers-classic",
    "icc", "ifort", "icx", "ifx",
    "aocc", "nvhpc", "pgi",
    "llvm", "clang", "clang-llvm",
    "cce", "craype",
    "arm", "armcompiler",
    "xl", "xlc", "xlf",
    "absoft", "fujitsu", "nag",
    # EasyBuild meta-toolchains (contain a compiler)
    "foss", "gompi", "gomkl", "iomkl", "iompi", "iccifort",
    "gfbf", "gcccuda", "fosscuda",
})

_MPI_NAMES = frozenset({
    "openmpi", "intelmpi", "impi", "intel-mpi",
    "mvapich", "mvapich2",
    "mpich", "mpich2",
    "cray-mpich", "cray-pmi",
    "hpcx", "hpcx-mpi",
    "spectrum-mpi", "platform-mpi",
    "mpi4py",
})

_GPU_NAMES = frozenset({
    "cuda", "cudatoolkit", "cudnn", "cutensor", "cufft", "cublas",
    "nccl", "nvshmem",
    "rocm", "hip", "hipblas", "hipfft", "rocsolver", "rocblas",
    "opencl", "level-zero",
    "nvhpc",  # compiler but also GPU-enabling; classified by compiler check first
})

_MATH_NAMES = frozenset({
    "mkl", "intel-mkl", "imkl", "onednn",
    "openblas",
    "blis", "amd-blis",
    "atlas",
    "fftw", "fftw3",
    "lapack", "scalapack", "blacs",
    "elpa", "slepc", "petsc",
    "armpl", "aocl", "amd-aocl",
    "flexiblas",
    "gsl",
})

_IO_NAMES = frozenset({
    "hdf5", "phdf5", "hdf4",
    "netcdf", "netcdf-c", "netcdf-cxx", "netcdf-fortran", "netcdf4",
    "parallel-netcdf", "pnetcdf",
    "adios", "adios2",
    "cgns", "sion", "sionlib",
})

_APP_NAMES = frozenset({
    "gromacs", "namd", "amber", "lammps", "vasp", "cp2k",
    "openfoam", "foam-extend",
    "nwchem", "gamess", "gaussian", "orca",
    "quantum-espresso", "qe", "abinit", "siesta", "elk", "wien2k",
    "wannier90", "wannier-tools",
    "wrf", "cesm", "icon",
    "python", "r", "julia", "matlab", "octave",
    "pytorch", "tensorflow", "jax",
    "paraview", "visit", "vmd",
    "cp2k", "nwchem",
})

# Prefix matches (cheaper than regex for common cases)
_COMPILER_PREFIXES = ("gcc", "intel", "nvhpc", "aocc", "llvm", "clang", "cce")
_MPI_PREFIXES = ("openmpi", "intelmpi", "mvapich", "mpich", "hpcx")
_GPU_PREFIXES = ("cuda", "rocm", "hip", "cudnn")
_MATH_PREFIXES = ("fftw", "openblas", "mkl", "scalapack", "petsc")
_IO_PREFIXES = ("hdf5", "netcdf", "adios")

CategoryLiteral = Literal[
    "compiler", "mpi", "gpu_toolkit", "math", "io_library", "application", "other"
]


def _classify(name_lower: str) -> CategoryLiteral:
    """Classify a module by its lowercase name."""
    if name_lower in _COMPILER_NAMES:
        return "compiler"
    if name_lower in _MPI_NAMES:
        return "mpi"
    if name_lower in _GPU_NAMES:
        return "gpu_toolkit"
    if name_lower in _MATH_NAMES:
        return "math"
    if name_lower in _IO_NAMES:
        return "io_library"
    if name_lower in _APP_NAMES:
        return "application"
    # Prefix matches (order matters: compiler before gpu for nvhpc)
    for p in _COMPILER_PREFIXES:
        if name_lower.startswith(p):
            return "compiler"
    for p in _MPI_PREFIXES:
        if name_lower.startswith(p):
            return "mpi"
    for p in _GPU_PREFIXES:
        if name_lower.startswith(p):
            return "gpu_toolkit"
    for p in _MATH_PREFIXES:
        if name_lower.startswith(p):
            return "math"
    for p in _IO_PREFIXES:
        if name_lower.startswith(p):
            return "io_library"
    return "other"


# ---------------------------------------------------------------------------
# Toolchain derivation
# ---------------------------------------------------------------------------

def _best_spec(m: ModuleInfo) -> str:
    """Return 'name/default_or_newest_version' or just 'name'."""
    ver = m.default_version or (m.versions[-1] if m.versions else "")
    return f"{m.name}/{ver}" if ver else m.name


def _derive_toolchains(
    compilers: list[ModuleInfo],
    mpi_libs: list[ModuleInfo],
    gpu_toolkits: list[ModuleInfo],
) -> list[Toolchain]:
    """Enumerate (compiler × mpi) and (compiler × gpu) toolchain combinations.

    Uses default (or newest) version for each module. Limits GPU cross-products
    to the first 3 compilers to avoid combinatorial explosion.
    """
    toolchains: list[Toolchain] = []
    seen: set[str] = set()

    def _add(tc: Toolchain) -> None:
        if tc.name not in seen:
            seen.add(tc.name)
            toolchains.append(tc)

    # Compiler-only (serial)
    for c in compilers:
        cs = _best_spec(c)
        _add(Toolchain(name=cs, compiler=cs, load_sequence=[f"module load {cs}"]))

    # Compiler + MPI
    for c in compilers:
        cs = _best_spec(c)
        for m in mpi_libs:
            ms = _best_spec(m)
            _add(Toolchain(
                name=f"{cs}+{ms}",
                compiler=cs,
                mpi=ms,
                load_sequence=[f"module load {cs}", f"module load {ms}"],
            ))

    # GPU standalone and compiler + GPU (top-3 compilers to avoid explosion)
    for g in gpu_toolkits:
        gs = _best_spec(g)
        is_cuda = "cuda" in g.name.lower()
        is_rocm = "rocm" in g.name.lower() or "hip" in g.name.lower()
        _add(Toolchain(
            name=gs,
            cuda=gs if is_cuda else None,
            rocm=gs if is_rocm else None,
            load_sequence=[f"module load {gs}"],
        ))
        for c in compilers[:3]:
            cs = _best_spec(c)
            _add(Toolchain(
                name=f"{cs}+{gs}",
                compiler=cs,
                cuda=gs if is_cuda else None,
                rocm=gs if is_rocm else None,
                load_sequence=[f"module load {cs}", f"module load {gs}"],
            ))

    return toolchains


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover_software_env(settings: AssistantSettings) -> SoftwareEnvironment:
    """Detect the module system and catalog all available modules.

    Runs ``module avail`` (fast) and attempts ``module spider`` (slower, full
    hierarchy). Falls back gracefully if either command fails or times out.
    """
    kind, version = detect_module_system(settings)

    if kind == "lmod":
        return _collect_lmod_env(settings, version)

    # tmod: detection only for now; full tmod support is a future extension
    if kind == "tmod":
        return SoftwareEnvironment(
            module_system="tmod",
            tmod_version=version,
            modulepath=os.environ.get("MODULEPATH", "").split(":"),
        )

    return SoftwareEnvironment(module_system="none")


def _collect_lmod_env(
    settings: AssistantSettings,
    lmod_version: str | None,
) -> SoftwareEnvironment:
    modulepath = [
        p for p in os.environ.get("MODULEPATH", "").split(":") if p
    ]

    # --- Phase 1: module avail (Core tier, fast) ---
    raw_avail = _run_bash_module("avail", settings)
    avail_entries = _parse_module_list(raw_avail) if raw_avail else []

    # --- Phase 2: module spider (full hierarchy, may be slow) ---
    raw_spider = _run_bash_module("spider", settings, timeout=_SPIDER_TIMEOUT_S)
    spider_complete = bool(raw_spider)
    spider_entries = _parse_module_list(raw_spider) if raw_spider else []

    # Merge: spider entries supersede avail where present
    combined_entries = avail_entries[:]
    avail_keys = {(n.lower(), v) for n, v, _ in avail_entries}
    for entry in spider_entries:
        if (entry[0].lower(), entry[1]) not in avail_keys:
            combined_entries.append(entry)

    module_map = _build_module_map(combined_entries)

    # Classify
    cats: dict[CategoryLiteral, list[ModuleInfo]] = {
        "compiler": [],
        "mpi": [],
        "gpu_toolkit": [],
        "math": [],
        "io_library": [],
        "application": [],
        "other": [],
    }
    for key, mod in sorted(module_map.items()):
        cats[_classify(key)].append(mod)

    toolchains = _derive_toolchains(
        cats["compiler"], cats["mpi"], cats["gpu_toolkit"]
    )

    return SoftwareEnvironment(
        module_system="lmod",
        lmod_version=lmod_version,
        modulepath=modulepath,
        spider_complete=spider_complete,
        compilers=cats["compiler"],
        mpi_libraries=cats["mpi"],
        gpu_toolkits=cats["gpu_toolkit"],
        math_libraries=cats["math"],
        io_libraries=cats["io_library"],
        applications=cats["application"],
        other_modules=cats["other"],
        toolchains=toolchains,
    )
