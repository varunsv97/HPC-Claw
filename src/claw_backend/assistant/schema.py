"""Pydantic models for the HPC Claw control plane."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


AssistantMode = Literal["coding", "pgoa"]
RoutingTarget = Literal["opencode", "backend", "pgoa_skill"]


class ApprovalGate(BaseModel):
    required: bool = False
    reason: str | None = None
    commands: list[str] = []


class ActionRoute(BaseModel):
    action: str
    capability: str
    target: RoutingTarget
    summary: str
    delegates_to: list[RoutingTarget] = []
    approval: ApprovalGate = Field(default_factory=ApprovalGate)


class RepoContext(BaseModel):
    root: str
    name: str
    has_git: bool = False
    guardrail_allowed: bool = True
    detected_files: list[str] = []


class OpenCodeWorkspace(BaseModel):
    repo_root: str
    config_present: bool = False
    tools_dir_present: bool = False
    pgoa_tools_present: bool = False
    ready: bool = False
    sync_command: str
    notes: list[str] = []


class AssistantSession(BaseModel):
    session_id: str
    created_at: datetime
    mode: AssistantMode = "coding"
    goal: str | None = None
    repo: RepoContext
    opencode: OpenCodeWorkspace
    store_root: str
    session_path: str
    routes: list[ActionRoute] = []
    synced_opencode_files: list[str] = []
    next_steps: list[str] = []
