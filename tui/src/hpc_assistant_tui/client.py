from __future__ import annotations

import json
from typing import Any
from urllib import error, request
from urllib.parse import quote

from .model import (
    BackendStatus,
    DirectoryListing,
    FileDocument,
    SessionEnvelope,
    SessionInfo,
    _backend_status_from_dict,
    _directory_listing_from_dict,
    _file_document_from_dict,
    _session_envelope_from_dict,
    _session_info_from_dict,
)


class BackendClient:
    def __init__(self, base_url: str, timeout_seconds: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def fetch_status(self) -> BackendStatus:
        payload = self._request_json("GET", "/healthz")
        status = _backend_status_from_dict(payload)
        if status is None:
            raise RuntimeError("invalid JSON response from /healthz")
        return status

    def fetch_session_info(self) -> SessionInfo:
        payload = self._request_json("GET", "/session-info")
        info = _session_info_from_dict(payload)
        if info is None:
            raise RuntimeError("invalid JSON response from /session-info")
        return info

    def open_session(self, profile: str, session_id: str | None = None) -> SessionEnvelope:
        payload = self._request_json(
            "POST",
            "/api/session",
            {"profile": profile, "session_id": session_id},
        )
        return _session_envelope_from_dict(payload)

    def send_prompt(self, session_id: str, prompt: str) -> SessionEnvelope:
        payload = self._request_json(
            "POST",
            f"/api/session/{session_id}/prompt",
            {"prompt": prompt},
        )
        return _session_envelope_from_dict(payload)

    def review_approval(
        self,
        session_id: str,
        approval_id: str,
        decision: str,
        edited_command: str | None = None,
    ) -> SessionEnvelope:
        payload = self._request_json(
            "POST",
            f"/api/session/{session_id}/approvals/{approval_id}",
            {"decision": decision, "edited_command": edited_command},
        )
        return _session_envelope_from_dict(payload)

    def fetch_directory(self, path: str | None = None, *, include_hidden: bool = True) -> DirectoryListing:
        query: list[str] = []
        if path:
            query.append(f"path={quote(path)}")
        query.append(f"include_hidden={'true' if include_hidden else 'false'}")
        query_string = "&".join(query)
        suffix = f"?{query_string}" if query_string else ""
        payload = self._request_json("GET", f"/api/files{suffix}")
        listing = _directory_listing_from_dict(payload)
        if listing is None:
            raise RuntimeError("invalid JSON response from /api/files")
        return listing

    def fetch_file(self, path: str) -> FileDocument:
        payload = self._request_json("GET", f"/api/file?path={quote(path)}")
        document = _file_document_from_dict(payload)
        if document is None:
            raise RuntimeError("invalid JSON response from /api/file")
        return document

    def save_file(self, path: str, content: str) -> FileDocument:
        payload = self._request_json(
            "POST",
            "/api/file",
            {"path": path, "content": content, "create_parents": True},
        )
        document = _file_document_from_dict(payload)
        if document is None:
            raise RuntimeError("invalid JSON response from /api/file")
        document.content = content
        return document

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        http_request = request.Request(
            url=f"{self.base_url}{path}",
            data=data,
            method=method,
            headers=headers,
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if detail:
                raise RuntimeError(
                    f"{method} {path} failed with HTTP {exc.code}: {detail}"
                ) from exc
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
        except OSError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON response from {path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid JSON response from {path}")
        return payload
