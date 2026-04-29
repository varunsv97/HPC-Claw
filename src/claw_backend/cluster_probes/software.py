"""Software environment discovery — detects Lmod and catalogs available modules.

Two sources of data, both best-effort:
  1. ``module --terse avail``   — Core-tier modules (immediately loadable, fast)
  2. ``module --terse spider``  — Full hierarchy including compiler/MPI-gated
                                  modules (may take 20–60 s on large clusters)

Classification hierarchy stored in SoftwareEnvironment:
  compilers     → compiler and compiler-toolchain modules
  mpi_libraries → MPI implementation modules
  gpu_toolkits  → GPU and accelerator toolkit modules
  math_libraries → math, solver, BLAS/LAPACK/FFT-style modules
  io_libraries  → scientific and parallel I/O modules
  debuggers     → debugging tool modules
  profilers     → profiling and tracing tool modules
  applications  → application, framework, runtime, and interpreter modules
  other_modules → everything else

Toolchain derivation:
  All (compiler × mpi) pairs and (compiler × gpu) pairs are enumerated
  using the default/newest version of each module, producing Toolchain
  objects with ready-made load_sequence lists.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Literal

from claw_backend.config import AssistantSettings
from claw_backend.pgoa.schema import ModuleContext, ModuleInfo, SoftwareEnvironment, Toolchain

log = logging.getLogger(__name__)

# Longer timeout for spider — it walks the full hierarchy
_SPIDER_TIMEOUT_S = 90.0
_CONTEXT_AVAIL_TIMEOUT_S = 20.0
_MAX_CONTEXT_MODULES = 16
_MAX_CONTEXT_SHOW_PROBES = 256
_MAX_CONTEXT_DEPTH = 4
_MAX_METADATA_PROBES = 512
_LLM_CLASSIFY_BATCH_SIZE = 80
_MAX_LLM_METADATA_CHARS = 1200
_MODULEPATH_MARKER = "__HPC_CLAW_MODULEPATH__"


@dataclass(frozen=True)
class ModuleEntry:
    name: str
    version: str
    is_default: bool
    context: str = "active"
    context_loads: tuple[str, ...] = ()
    modulepath: str | None = None
    discovered_by: str = "active"
    metadata: str | None = None

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


def _build_module_shell_command(
    subcmd: str,
    load_sequence: tuple[str, ...] = (),
    include_modulepath: bool = False,
) -> str:
    """Return a bash snippet that initializes modules, loads context, and runs a command."""
    init = _find_lmod_init()
    parts: list[str] = []
    if init:
        parts.append(f"source {shlex.quote(init)} 2>/dev/null")

    for module_spec in load_sequence:
        parts.append(f"module load {shlex.quote(module_spec)} >/dev/null 2>&1")

    parts.append(f"module {subcmd} 2>&1")
    if include_modulepath:
        parts.append(f"printf '\\n{_MODULEPATH_MARKER}%s\\n' \"$MODULEPATH\"")

    return " && ".join(parts) if parts else f"module {subcmd} 2>&1"


def _run_bash_module_context(
    subcmd: str,
    settings: AssistantSettings,
    *,
    load_sequence: tuple[str, ...] = (),
    timeout: float | None = None,
    include_modulepath: bool = False,
) -> tuple[str, list[str]]:
    """Execute ``module <subcmd>`` after optional context loads.

    Returns command output and, when requested, the MODULEPATH visible after the
    context loads. The caller is responsible for deciding whether the context is
    useful.
    """
    effective_timeout = timeout if timeout is not None else settings.command_timeout_seconds

    if "--terse" not in subcmd and subcmd not in ("--version",) and not subcmd.startswith("show "):
        subcmd = f"--terse {subcmd}"

    bash_cmd = _build_module_shell_command(
        subcmd,
        load_sequence=load_sequence,
        include_modulepath=include_modulepath,
    )

    try:
        r = subprocess.run(
            ["bash", "-c", bash_cmd],
            capture_output=True,
            text=True,
            check=False,
            timeout=effective_timeout,
            env=os.environ,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.debug("module command failed (%s): %s", subcmd, exc)
        return "", []

    output = (r.stdout + r.stderr).strip()
    modulepath: list[str] = []
    if include_modulepath and _MODULEPATH_MARKER in output:
        before, _, after = output.partition(_MODULEPATH_MARKER)
        output = before.strip()
        modulepath_line = after.splitlines()[0].strip() if after else ""
        modulepath = [p for p in modulepath_line.split(":") if p]
    return output, modulepath


def _run_bash_module(subcmd: str, settings: AssistantSettings, timeout: float | None = None) -> str:
    """Execute ``module <subcmd>`` via bash and return combined stdout+stderr.

    Automatically prepends ``--terse`` unless the caller already included it.
    Sources the Lmod init script when LMOD_VERSION is not already in the env.
    """
    output, _ = _run_bash_module_context(subcmd, settings, timeout=timeout)
    return output


# ---------------------------------------------------------------------------
# Module list parsing
# ---------------------------------------------------------------------------

# Annotations Lmod appends after a module name
_ANNOTATION_RE = re.compile(r"\s*\([^)]*\)")


def _parse_module_entries(
    raw: str,
    *,
    context: str = "active",
    context_loads: tuple[str, ...] = (),
    discovered_by: str = "active",
) -> list[ModuleEntry]:
    """Parse ``module --terse avail`` or ``module --terse spider`` output."""
    results: list[ModuleEntry] = []
    current_modulepath: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip path/section headers (end with ':')
        if line.endswith(":"):
            maybe_path = line[:-1].strip()
            current_modulepath = maybe_path if maybe_path.startswith("/") else None
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
            results.append(ModuleEntry(
                name=name.strip(),
                version=version.strip(),
                is_default=is_default,
                context=context,
                context_loads=context_loads,
                modulepath=current_modulepath,
                discovered_by=discovered_by,
            ))
        else:
            results.append(ModuleEntry(
                name=clean.strip(),
                version="",
                is_default=is_default,
                context=context,
                context_loads=context_loads,
                modulepath=current_modulepath,
                discovered_by=discovered_by,
            ))

    return results


def _parse_module_list(raw: str) -> list[tuple[str, str, bool]]:
    """Parse module command output into legacy ``(name, version, default)`` tuples."""
    return [(e.name, e.version, e.is_default) for e in _parse_module_entries(raw)]


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


def _coerce_entry(entry: ModuleEntry | tuple[str, str, bool]) -> ModuleEntry:
    if isinstance(entry, ModuleEntry):
        return entry
    name, version, is_default = entry
    return ModuleEntry(name=name, version=version, is_default=is_default)


def _module_spec(name: str, version: str | None) -> str:
    return f"{name}/{version}" if version else name


def _build_module_map(entries: list[ModuleEntry | tuple[str, str, bool]]) -> dict[str, ModuleInfo]:
    """Collapse (name, version, is_default) entries into a name → ModuleInfo dict.

    Key is lowercase for case-insensitive deduplication; stored name preserves
    the original casing of the first occurrence.
    """
    groups: dict[str, dict] = {}

    for raw_entry in entries:
        entry = _coerce_entry(raw_entry)
        key = entry.name.lower()
        if key not in groups:
            groups[key] = {
                "name": entry.name,
                "versions": [],
                "default_version": None,
                "source_contexts": set(),
                "modulepaths": set(),
                "loads_by_version": {},
                "first_loads": entry.context_loads,
                "metadata": [],
            }
        if entry.version and entry.version not in groups[key]["versions"]:
            groups[key]["versions"].append(entry.version)
        if entry.is_default and entry.version:
            groups[key]["default_version"] = entry.version
        groups[key]["source_contexts"].add(entry.context)
        if entry.modulepath:
            groups[key]["modulepaths"].add(entry.modulepath)
        if entry.metadata:
            groups[key]["metadata"].append(entry.metadata)
        load_key = entry.version or ""
        if load_key not in groups[key]["loads_by_version"]:
            groups[key]["loads_by_version"][load_key] = entry.context_loads
        elif entry.context_loads and not groups[key]["loads_by_version"][load_key]:
            groups[key]["loads_by_version"][load_key] = entry.context_loads

    result: dict[str, ModuleInfo] = {}
    for key, info in groups.items():
        versions = sorted(info["versions"], key=_version_key)
        default = info["default_version"]
        if not default and versions:
            default = versions[-1]  # newest as fallback
        load_ver = default or (versions[0] if versions else "")
        spec = _module_spec(info["name"], load_ver)
        prereq_loads = info["loads_by_version"].get(load_ver, info["first_loads"])
        metadata = "\n".join(dict.fromkeys(info["metadata"])) or None
        load_sequence = [f"module load {m}" for m in prereq_loads]
        load_sequence.append(f"module load {spec}")
        load_cmd = (
            " && ".join(load_sequence)
        )
        result[key] = ModuleInfo(
            name=info["name"],
            versions=versions,
            default_version=default,
            load_cmd=load_cmd,
            load_sequence=load_sequence,
            source_contexts=sorted(info["source_contexts"]),
            modulepaths=sorted(info["modulepaths"]),
            metadata=metadata,
        )

    return result


# ---------------------------------------------------------------------------
# Module visibility contexts
# ---------------------------------------------------------------------------

_MODULEPATH_MUTATION_RE = re.compile(
    r"(?:prepend_path|append_path|setenv|setenv\(|MODULEPATH)",
    re.IGNORECASE,
)


def _entry_spec(entry: ModuleEntry) -> str:
    return _module_spec(entry.name, entry.version)


def _module_show_changes_modulepath(
    entry: ModuleEntry,
    settings: AssistantSettings,
    *,
    context_loads: tuple[str, ...] = (),
) -> bool:
    subcmd = f"show {shlex.quote(_entry_spec(entry))}"
    if context_loads:
        raw, _ = _run_bash_module_context(
            subcmd,
            settings,
            load_sequence=context_loads,
            timeout=5.0,
        )
    else:
        raw = _run_bash_module(subcmd, settings, timeout=5.0)
    return bool(raw and _MODULEPATH_MUTATION_RE.search(raw))


def _context_candidates(
    entries_to_probe: list[ModuleEntry],
    settings: AssistantSettings,
    *,
    context_loads: tuple[str, ...] = (),
    seen_context_loads: set[tuple[str, ...]] | None = None,
) -> list[tuple[ModuleEntry, str]]:
    candidates: list[tuple[ModuleEntry, str]] = []
    seen: set[str] = set()
    entries = sorted(entries_to_probe, key=lambda e: (bool(e.version), e.name.lower()))
    for entry in entries[:_MAX_CONTEXT_SHOW_PROBES]:
        spec = _entry_spec(entry)
        if spec in seen:
            continue
        if spec in context_loads:
            continue
        candidate_loads = (*context_loads, spec)
        if seen_context_loads is not None and candidate_loads in seen_context_loads:
            continue
        if _module_show_changes_modulepath(entry, settings, context_loads=context_loads):
            seen.add(spec)
            candidates.append((entry, "module_show"))
        if len(candidates) >= _MAX_CONTEXT_MODULES:
            break
    return candidates


def _discover_module_contexts(
    settings: AssistantSettings,
    active_entries: list[ModuleEntry],
    base_modulepath: list[str],
) -> tuple[list[ModuleContext], list[ModuleEntry]]:
    contexts: list[ModuleContext] = [
        ModuleContext(
            name="active",
            load_sequence=[],
            modulepath=base_modulepath,
            modules=list(_build_module_map(active_entries).values()),
            discovered_by="active",
        )
    ]
    discovered_entries: list[ModuleEntry] = []
    seen_context_loads: set[tuple[str, ...]] = {()}
    queue: list[tuple[tuple[str, ...], list[ModuleEntry], list[str], int]] = [
        ((), active_entries, base_modulepath, 0)
    ]

    while queue and len(contexts) <= _MAX_CONTEXT_MODULES:
        parent_loads, parent_entries, parent_modulepath, depth = queue.pop(0)
        if depth >= _MAX_CONTEXT_DEPTH:
            continue

        for selector, reason in _context_candidates(
            parent_entries,
            settings,
            context_loads=parent_loads,
            seen_context_loads=seen_context_loads,
        ):
            selector_spec = _entry_spec(selector)
            context_loads = (*parent_loads, selector_spec)
            if context_loads in seen_context_loads:
                continue
            seen_context_loads.add(context_loads)

            raw, context_modulepath = _run_bash_module_context(
                "avail",
                settings,
                load_sequence=context_loads,
                timeout=_CONTEXT_AVAIL_TIMEOUT_S,
                include_modulepath=True,
            )
            if not raw:
                continue
            context_name = " + ".join(context_loads)
            entries = _parse_module_entries(
                raw,
                context=context_name,
                context_loads=context_loads,
                discovered_by=reason,
            )
            new_paths = set(context_modulepath) - set(parent_modulepath)
            new_keys = {(e.name.lower(), e.version) for e in entries} - {
                (e.name.lower(), e.version) for e in parent_entries
            }
            if not new_paths and not new_keys:
                continue

            discovered_entries.extend(entries)
            contexts.append(ModuleContext(
                name=context_name,
                load_sequence=[f"module load {spec}" for spec in context_loads],
                modulepath=context_modulepath,
                modules=list(_build_module_map(entries).values()),
                discovered_by=reason,
            ))
            queue.append((context_loads, entries, context_modulepath, depth + 1))

            if len(contexts) > _MAX_CONTEXT_MODULES:
                break

    return contexts, discovered_entries


def _module_metadata(entry: ModuleEntry, settings: AssistantSettings) -> str:
    """Return best-effort metadata for a module from module commands."""
    spec = _entry_spec(entry)
    outputs: list[str] = []
    whatis_cmd = f"whatis {shlex.quote(spec)}"
    show_cmd = f"show {shlex.quote(spec)}"
    if entry.context_loads:
        whatis, _ = _run_bash_module_context(
            whatis_cmd,
            settings,
            load_sequence=entry.context_loads,
            timeout=5.0,
        )
        show, _ = _run_bash_module_context(
            show_cmd,
            settings,
            load_sequence=entry.context_loads,
            timeout=5.0,
        )
    else:
        whatis = _run_bash_module(whatis_cmd, settings, timeout=5.0)
        show = _run_bash_module(show_cmd, settings, timeout=5.0)
    if whatis:
        outputs.append(whatis)
    if show:
        outputs.append(show)
    return "\n".join(outputs).strip()


def _enrich_entries_with_metadata(
    entries: list[ModuleEntry],
    settings: AssistantSettings,
) -> list[ModuleEntry]:
    enriched: list[ModuleEntry] = []
    metadata_by_context_spec: dict[tuple[tuple[str, ...], str], str] = {}
    probes = 0
    for entry in entries:
        key = (entry.context_loads, _entry_spec(entry))
        metadata = metadata_by_context_spec.get(key)
        if metadata is None and probes < _MAX_METADATA_PROBES:
            metadata = _module_metadata(entry, settings)
            metadata_by_context_spec[key] = metadata
            probes += 1
        enriched.append(replace(entry, metadata=metadata or entry.metadata))
    return enriched


# ---------------------------------------------------------------------------
# Module classification
# ---------------------------------------------------------------------------

CategoryLiteral = Literal[
    "compiler", "mpi", "gpu_toolkit", "math", "io_library", "debugger",
    "profiler", "application", "other"
]

_CATEGORY_KEYS: tuple[CategoryLiteral, ...] = (
    "compiler", "mpi", "gpu_toolkit", "math", "io_library", "debugger",
    "profiler", "application", "other",
)


def _classify(_name_lower: str, metadata: str | None = None) -> CategoryLiteral:
    """Deterministic fallback: do not guess categories without an LLM verdict."""
    return "other"


def _provider_configured(settings: AssistantSettings) -> bool:
    return bool(settings.provider_api_key or settings.provider_base_url)


def _llm_module_payload(key: str, module: ModuleInfo) -> dict[str, object]:
    return {
        "id": key,
        "name": module.name,
        "versions": module.versions[:12],
        "default_version": module.default_version,
        "source_contexts": module.source_contexts,
        "modulepaths": module.modulepaths,
        "metadata": (module.metadata or "")[:_MAX_LLM_METADATA_CHARS],
    }


def _parse_llm_classification(raw: str) -> dict[str, CategoryLiteral]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    items = parsed.get("modules", parsed)
    if not isinstance(items, dict):
        return {}

    result: dict[str, CategoryLiteral] = {}
    for key, category in items.items():
        if isinstance(key, str) and category in _CATEGORY_KEYS:
            result[key] = category
    return result


def _classify_modules_with_llm(
    module_map: dict[str, ModuleInfo],
    settings: AssistantSettings,
) -> dict[str, CategoryLiteral]:
    """Classify modules with an LLM using discovered module metadata only."""
    if not _provider_configured(settings) or not module_map:
        return {}

    try:
        from claw_backend.openai_client import build_openai_client

        client = build_openai_client(settings)
    except Exception:
        log.debug("LLM module classifier unavailable", exc_info=True)
        return {}

    classifications: dict[str, CategoryLiteral] = {}
    items = list(module_map.items())
    system_prompt = (
        "You classify HPC environment modules from module command output. "
        "Use only the provided module metadata, MODULEPATH paths, and visibility "
        "contexts. Do not rely on memorized module names. Return compact JSON: "
        "{\"modules\":{\"<id>\":\"<category>\"}}. Categories are: "
        + ", ".join(_CATEGORY_KEYS)
        + ". Use other when the evidence is weak."
    )

    for start in range(0, len(items), _LLM_CLASSIFY_BATCH_SIZE):
        batch = items[start:start + _LLM_CLASSIFY_BATCH_SIZE]
        payload = [_llm_module_payload(key, module) for key, module in batch]
        try:
            response = client.chat.completions.create(
                model=settings.provider_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({"modules": payload})},
                ],
                temperature=0,
                timeout=settings.provider_timeout_seconds,
            )
            content = response.choices[0].message.content or ""
        except Exception:
            log.debug("LLM module classification failed", exc_info=True)
            continue
        classifications.update(_parse_llm_classification(content))

    return classifications


# ---------------------------------------------------------------------------
# Toolchain derivation
# ---------------------------------------------------------------------------

def _best_spec(m: ModuleInfo) -> str:
    """Return 'name/default_or_newest_version' or just 'name'."""
    ver = m.default_version or (m.versions[-1] if m.versions else "")
    return f"{m.name}/{ver}" if ver else m.name


def _load_sequence_for(m: ModuleInfo) -> list[str]:
    if m.load_sequence:
        return m.load_sequence
    return [f"module load {_best_spec(m)}"]


def _merge_load_sequences(*sequences: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for sequence in sequences:
        for command in sequence:
            if command not in seen:
                seen.add(command)
                merged.append(command)
    return merged


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
        _add(Toolchain(name=cs, compiler=cs, load_sequence=_load_sequence_for(c)))

    # Compiler + MPI
    for c in compilers:
        cs = _best_spec(c)
        for m in mpi_libs:
            ms = _best_spec(m)
            _add(Toolchain(
                name=f"{cs}+{ms}",
                compiler=cs,
                mpi=ms,
                load_sequence=_merge_load_sequences(_load_sequence_for(c), _load_sequence_for(m)),
            ))

    # GPU standalone and compiler + GPU (top-3 compilers to avoid explosion)
    for g in gpu_toolkits:
        gs = _best_spec(g)
        _add(Toolchain(
            name=gs,
            gpu=gs,
            load_sequence=_load_sequence_for(g),
        ))
        for c in compilers[:3]:
            cs = _best_spec(c)
            _add(Toolchain(
                name=f"{cs}+{gs}",
                compiler=cs,
                gpu=gs,
                load_sequence=_merge_load_sequences(_load_sequence_for(c), _load_sequence_for(g)),
            ))

    return toolchains


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover_software_env(
    settings: AssistantSettings,
) -> SoftwareEnvironment:
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
    avail_entries = _parse_module_entries(raw_avail) if raw_avail else []
    module_contexts, context_entries = _discover_module_contexts(
        settings,
        avail_entries,
        modulepath,
    )

    # --- Phase 2: module spider (full hierarchy, may be slow) ---
    raw_spider = _run_bash_module("spider", settings, timeout=_SPIDER_TIMEOUT_S)
    spider_complete = bool(raw_spider)
    spider_entries = _parse_module_entries(raw_spider, context="spider") if raw_spider else []

    # Merge: spider entries supersede avail where present
    combined_entries = avail_entries[:]
    avail_keys = {(entry.name.lower(), entry.version) for entry in avail_entries}
    for entry in spider_entries:
        if (entry.name.lower(), entry.version) not in avail_keys:
            combined_entries.append(entry)
    for entry in context_entries:
        combined_entries.append(entry)

    combined_entries = _enrich_entries_with_metadata(combined_entries, settings)
    module_map = _build_module_map(combined_entries)
    llm_categories = _classify_modules_with_llm(module_map, settings)

    # Classify
    cats: dict[CategoryLiteral, list[ModuleInfo]] = {
        "compiler": [],
        "mpi": [],
        "gpu_toolkit": [],
        "math": [],
        "io_library": [],
        "debugger": [],
        "profiler": [],
        "application": [],
        "other": [],
    }
    for key, mod in sorted(module_map.items()):
        cats[llm_categories.get(key, _classify(key, mod.metadata))].append(mod)

    toolchains = _derive_toolchains(
        cats["compiler"], cats["mpi"], cats["gpu_toolkit"]
    )

    return SoftwareEnvironment(
        module_system="lmod",
        lmod_version=lmod_version,
        modulepath=modulepath,
        module_contexts=module_contexts,
        spider_complete=spider_complete,
        compilers=cats["compiler"],
        mpi_libraries=cats["mpi"],
        gpu_toolkits=cats["gpu_toolkit"],
        math_libraries=cats["math"],
        io_libraries=cats["io_library"],
        debuggers=cats["debugger"],
        profilers=cats["profiler"],
        applications=cats["application"],
        other_modules=cats["other"],
        toolchains=toolchains,
    )
