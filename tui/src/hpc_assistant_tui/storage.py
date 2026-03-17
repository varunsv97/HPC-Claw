from __future__ import annotations

import json
from pathlib import Path

from .model import PersistedState


def load_snapshot(path: Path) -> PersistedState | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"failed to read snapshot {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"failed to decode snapshot {path}: {error}") from error

    if not isinstance(payload, dict):
        raise RuntimeError(f"snapshot {path} did not contain a JSON object")

    return PersistedState.from_dict(payload)


def save_snapshot(path: Path, snapshot: PersistedState) -> None:
    if path.parent != Path():
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path.write_text(
            json.dumps(snapshot.to_dict(), indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise RuntimeError(f"failed to write snapshot {path}: {error}") from error

