"""Path access control and resolution for approved filesystem roots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from hpc_assistant_backend.config import AssistantSettings


MAX_FILE_PREVIEW_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class FileEntry:
    name: str
    path: str
    is_dir: bool
    size: int
    modified_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "is_dir": self.is_dir,
            "size": self.size,
            "modified_at": self.modified_at,
        }


def configured_filesystem_roots(settings: AssistantSettings) -> tuple[Path, ...]:
    roots = tuple(normalize_root(root) for root in settings.filesystem_roots if str(root).strip())
    return roots or (normalize_root("~"),)


def normalize_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def resolve_allowed_path(path: str, settings: AssistantSettings) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve(strict=False)
    for root in configured_filesystem_roots(settings):
        if candidate.is_relative_to(root):
            return candidate
    allowed = ", ".join(str(root) for root in configured_filesystem_roots(settings))
    raise ValueError(f"path {candidate} is outside approved filesystem roots: {allowed}")


def list_directory(
    settings: AssistantSettings,
    path: str | None = None,
    *,
    include_hidden: bool = True,
) -> dict[str, Any]:
    target = configured_filesystem_roots(settings)[0] if path is None else resolve_allowed_path(path, settings)
    if not target.exists():
        raise FileNotFoundError(target)
    if not target.is_dir():
        raise NotADirectoryError(target)

    entries: list[FileEntry] = []
    for child in sorted(
        target.iterdir(),
        key=lambda item: (not item.is_dir(), item.name.lower()),
    ):
        if not include_hidden and child.name.startswith("."):
            continue
        stat = child.stat()
        entries.append(
            FileEntry(
                name=child.name,
                path=str(child.resolve(strict=False)),
                is_dir=child.is_dir(),
                size=stat.st_size,
                modified_at=_isoformat(stat.st_mtime),
            )
        )

    return {
        "path": str(target),
        "entries": [entry.to_dict() for entry in entries],
    }


def read_text_file(
    settings: AssistantSettings,
    path: str,
    *,
    max_bytes: int = MAX_FILE_PREVIEW_BYTES,
) -> dict[str, Any]:
    target = resolve_allowed_path(path, settings)
    if not target.exists():
        raise FileNotFoundError(target)
    if not target.is_file():
        raise IsADirectoryError(target)

    raw = target.read_bytes()
    truncated = len(raw) > max_bytes
    content = raw[:max_bytes].decode("utf-8", errors="replace")
    stat = target.stat()
    return {
        "path": str(target),
        "size": stat.st_size,
        "modified_at": _isoformat(stat.st_mtime),
        "content": content,
        "truncated": truncated,
        "encoding": "utf-8",
    }


def write_text_file(
    settings: AssistantSettings,
    path: str,
    content: str,
    *,
    create_parents: bool = True,
) -> dict[str, Any]:
    target = resolve_allowed_path(path, settings)
    if create_parents:
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    stat = target.stat()
    return {
        "path": str(target),
        "size": stat.st_size,
        "modified_at": _isoformat(stat.st_mtime),
        "created": True,
    }


def _isoformat(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
