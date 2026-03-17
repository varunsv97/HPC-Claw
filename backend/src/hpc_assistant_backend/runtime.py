"""Session orchestration for the cluster assistant backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import shlex
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from hpc_assistant_backend.config import AssistantSettings, ExecutionPolicy, load_settings
from hpc_assistant_backend.model import build_chat_model
from hpc_assistant_backend.tools import (
    ToolRegistry,
    build_default_registry,
    build_review_command,
    build_tools,
    run_reviewed_command,
)


MAX_TOOL_STEPS = 8


class TurnStatus(str, Enum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class PendingInterrupt:
    id: str | None
    value: Any


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    thread_id: str
    status: TurnStatus
    output_text: str
    interrupts: tuple[PendingInterrupt, ...]
    raw_output: Any


@dataclass(slots=True)
class PendingApprovalState:
    approval_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    command: str
    title: str
    rationale: str

    def to_interrupt_payload(self) -> dict[str, Any]:
        return {
            "id": self.approval_id,
            "kind": "tool_review",
            "title": self.title,
            "command": self.command,
            "rationale": self.rationale,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "arguments": self.arguments,
            "action_requests": [
                {
                    "name": self.tool_name,
                    "args": self.arguments,
                    "command": self.command,
                    "description": self.rationale,
                }
            ],
            "review_configs": [
                {
                    "action_name": self.tool_name,
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "command": self.command,
                    "title": self.title,
                }
            ],
        }


@dataclass(slots=True)
class AssistantSession:
    runtime: "AssistantRuntime"
    thread_id: str
    user_id: str
    messages: list[BaseMessage] = field(default_factory=list)
    pending_approvals: dict[str, PendingApprovalState] = field(default_factory=dict)

    def invoke(self, message: str) -> AgentTurnResult:
        self.messages.append(HumanMessage(content=message))
        return self.runtime._run_turn(self)

    async def ainvoke(self, message: str) -> AgentTurnResult:
        return self.invoke(message)

    def resume(self, decision: Any) -> AgentTurnResult:
        return self.runtime._resume_turn(self, decision)

    async def aresume(self, decision: Any) -> AgentTurnResult:
        return self.resume(decision)


class AssistantRuntime:
    """High-level interface for driving tool-calling model sessions."""

    def __init__(
        self,
        settings: AssistantSettings | None = None,
        *,
        registry: ToolRegistry | None = None,
        model: Any | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.registry = registry or build_default_registry()
        self.tools = build_tools(self.settings, self.registry)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.approval_tool_names = {
            spec.name
            for spec in self.registry.approval_required_specs()
            if self.settings.execution_policy is ExecutionPolicy.APPROVAL_REQUIRED
        }
        raw_model = model or build_chat_model(self.settings)
        self.model = raw_model.bind_tools(self.tools) if hasattr(raw_model, "bind_tools") else raw_model
        self.system_prompt = _build_system_prompt(self.settings)

    def start_session(self, thread_id: str, *, user_id: str | None = None) -> AssistantSession:
        return AssistantSession(
            runtime=self,
            thread_id=thread_id,
            user_id=user_id or self.settings.default_user_id,
            messages=[SystemMessage(content=self.system_prompt)],
        )

    def invoke(self, *, thread_id: str, message: str, user_id: str | None = None) -> AgentTurnResult:
        return self.start_session(thread_id, user_id=user_id).invoke(message)

    async def ainvoke(self, *, thread_id: str, message: str, user_id: str | None = None) -> AgentTurnResult:
        return self.invoke(thread_id=thread_id, message=message, user_id=user_id)

    def resume(self, *, thread_id: str, decision: Any, user_id: str | None = None) -> AgentTurnResult:
        session = self.start_session(thread_id, user_id=user_id)
        return session.resume(decision)

    async def aresume(self, *, thread_id: str, decision: Any, user_id: str | None = None) -> AgentTurnResult:
        return self.resume(thread_id=thread_id, decision=decision, user_id=user_id)

    def _run_turn(
        self,
        session: AssistantSession,
        *,
        events: list[dict[str, Any]] | None = None,
    ) -> AgentTurnResult:
        turn_events = list(events or [])
        last_output_text = ""

        for _step in range(MAX_TOOL_STEPS):
            response = self.model.invoke(session.messages)
            if not isinstance(response, AIMessage):
                response = AIMessage(content=_content_to_text(getattr(response, "content", response)))
            session.messages.append(response)

            assistant_text = _message_to_text(response)
            if assistant_text:
                turn_events.append(
                    {
                        "type": "assistant_message",
                        "message": {
                            "role": "assistant",
                            "content": assistant_text,
                        },
                    }
                )
                last_output_text = assistant_text

            tool_calls = list(getattr(response, "tool_calls", []) or [])
            if not tool_calls:
                return self._build_turn_result(
                    session,
                    TurnStatus.COMPLETED,
                    last_output_text,
                    (),
                    turn_events,
                )

            interrupts, tool_messages, new_events = self._handle_tool_calls(session, tool_calls)
            turn_events.extend(new_events)
            session.messages.extend(tool_messages)
            if interrupts:
                return self._build_turn_result(
                    session,
                    TurnStatus.INTERRUPTED,
                    last_output_text,
                    interrupts,
                    turn_events,
                )

        exhausted_text = (
            last_output_text
            or "Stopped after reaching the tool execution limit for a single turn."
        )
        turn_events.append(
            {
                "type": "assistant_message",
                "message": {
                    "role": "assistant",
                    "content": exhausted_text,
                },
            }
        )
        return self._build_turn_result(
            session,
            TurnStatus.COMPLETED,
            exhausted_text,
            (),
            turn_events,
        )

    def _resume_turn(self, session: AssistantSession, decision: Any) -> AgentTurnResult:
        turn_events = self._apply_approval_decisions(session, decision)
        if session.pending_approvals:
            remaining = tuple(
                PendingInterrupt(id=approval_id, value=pending.to_interrupt_payload())
                for approval_id, pending in session.pending_approvals.items()
            )
            return self._build_turn_result(
                session,
                TurnStatus.INTERRUPTED,
                "",
                remaining,
                turn_events,
            )
        return self._run_turn(session, events=turn_events)

    def _handle_tool_calls(
        self,
        session: AssistantSession,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[tuple[PendingInterrupt, ...], list[ToolMessage], list[dict[str, Any]]]:
        interrupts: list[PendingInterrupt] = []
        tool_messages: list[ToolMessage] = []
        events: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_call_id = str(tool_call.get("id") or uuid4().hex)
            tool_name = str(tool_call.get("name", "")).strip()
            arguments = tool_call.get("args")
            if not isinstance(arguments, dict):
                arguments = {}

            if tool_name in self.approval_tool_names:
                approval = self._queue_approval(tool_call_id, tool_name, arguments)
                session.pending_approvals[approval.approval_id] = approval
                events.append(
                    {
                        "type": "tool_call",
                        "tool_call": {
                            "id": tool_call_id,
                            "name": tool_name,
                            "args": arguments,
                            "status": "waiting_approval",
                            "command": approval.command,
                        },
                    }
                )
                events.append(
                    {
                        "type": "approval_requested",
                        "approval": approval.to_interrupt_payload(),
                    }
                )
                interrupts.append(
                    PendingInterrupt(
                        id=approval.approval_id,
                        value=approval.to_interrupt_payload(),
                    )
                )
                continue

            tool_message, tool_event = self._execute_tool_call(tool_call_id, tool_name, arguments)
            events.append(
                {
                    "type": "tool_call",
                    "tool_call": {
                        "id": tool_call_id,
                        "name": tool_name,
                        "args": arguments,
                        "status": "completed" if tool_event["status"] == "success" else "error",
                        "command": tool_event.get("command"),
                    },
                }
            )
            events.append({"type": "tool_result", "tool_result": tool_event})
            tool_messages.append(tool_message)

        return tuple(interrupts), tool_messages, events

    def _apply_approval_decisions(
        self,
        session: AssistantSession,
        decision: Any,
    ) -> list[dict[str, Any]]:
        decision_map = _coerce_decision_map(decision)
        if not decision_map:
            raise ValueError("approval decision payload is empty")

        events: list[dict[str, Any]] = []
        for approval_id, payload in decision_map.items():
            pending = session.pending_approvals.pop(approval_id, None)
            if pending is None:
                raise KeyError(approval_id)

            decision_type, edited_command = _extract_decision(payload)
            if decision_type == "reject":
                result = {
                    "tool": pending.tool_name,
                    "ok": False,
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "Execution was rejected by the user.",
                    "command": shlex.split(pending.command),
                }
            elif decision_type == "edit":
                if not edited_command:
                    raise ValueError("edited_command is required for edit decisions")
                result = run_reviewed_command(edited_command, self.settings)
            else:
                result = self._invoke_tool(pending.tool_name, pending.arguments)

            tool_message, tool_event = _result_to_tool_payload(
                tool_call_id=pending.tool_call_id,
                tool_name=pending.tool_name,
                result=result,
            )
            session.messages.append(tool_message)
            events.append({"type": "tool_result", "tool_result": tool_event})

        return events

    def _queue_approval(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PendingApprovalState:
        command = " ".join(
            shlex.quote(part)
            for part in build_review_command(tool_name, arguments, self.settings)
        )
        title = f"Review {tool_name}"
        return PendingApprovalState(
            approval_id=uuid4().hex,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            command=command,
            title=title,
            rationale=(
                "This command changes scheduler, module, or filesystem state and must be "
                "reviewed before it runs."
            ),
        )

    def _execute_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[ToolMessage, dict[str, Any]]:
        result = self._invoke_tool(tool_name, arguments)
        return _result_to_tool_payload(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result,
        )

    def _invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            return {
                "tool": tool_name,
                "ok": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": f"unknown tool: {tool_name}",
                "command": [],
            }
        try:
            return tool.invoke(arguments)
        except Exception as error:  # noqa: BLE001 - tool errors should flow back into the model
            return {
                "tool": tool_name,
                "ok": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": str(error),
                "command": [],
            }

    def _build_turn_result(
        self,
        session: AssistantSession,
        status: TurnStatus,
        output_text: str,
        interrupts: tuple[PendingInterrupt, ...],
        events: list[dict[str, Any]],
    ) -> AgentTurnResult:
        return AgentTurnResult(
            thread_id=session.thread_id,
            status=status,
            output_text=output_text,
            interrupts=interrupts,
            raw_output={
                "messages": list(session.messages),
                "events": events,
            },
        )


def _build_system_prompt(settings: AssistantSettings) -> str:
    roots = ", ".join(settings.filesystem_roots)
    return (
        "You are an HPC cluster assistant running for a shell user on a real cluster. "
        "Use tools for live data when possible, prefer read-only inspection first, be "
        "careful with scheduler and filesystem mutations, and explain assumptions "
        "briefly. Filesystem access is restricted to these roots: "
        f"{roots}. Any mutating tool requiring approval will be paused for user review."
    )


def _coerce_decision_map(decision: Any) -> dict[str, Any]:
    if isinstance(decision, dict):
        if "approval_id" in decision:
            approval_id = str(decision.get("approval_id", "")).strip()
            if not approval_id:
                raise ValueError("approval_id must not be empty")
            return {approval_id: decision}
        return {str(key): value for key, value in decision.items()}
    raise ValueError("approval decision payload must be a mapping")


def _extract_decision(payload: Any) -> tuple[str, str | None]:
    if isinstance(payload, dict):
        direct_decision = payload.get("decision")
        if isinstance(direct_decision, str):
            decision = direct_decision.strip().lower()
            if decision == "approve":
                return "approve", None
            if decision == "reject":
                return "reject", None
            if decision == "edit":
                edited_command = payload.get("edited_command")
                return "edit", str(edited_command).strip() if isinstance(edited_command, str) else None
        decisions = payload.get("decisions")
        if isinstance(decisions, list) and decisions:
            first = decisions[0]
            if isinstance(first, dict):
                decision_type = str(first.get("type", "")).strip().lower()
                if decision_type in {"approve", "reject"}:
                    return decision_type, None
                if decision_type == "edit":
                    edited = first.get("command") or first.get("edited_command") or first.get("message")
                    return "edit", str(edited).strip() if isinstance(edited, str) else None
    raise ValueError("unsupported approval decision payload")


def _result_to_tool_payload(
    *,
    tool_call_id: str,
    tool_name: str,
    result: Any,
) -> tuple[ToolMessage, dict[str, Any]]:
    status = "success" if _result_is_ok(result) else "error"
    content = _tool_result_text(result)
    tool_message = ToolMessage(
        content=_tool_message_content(result),
        tool_call_id=tool_call_id,
        name=tool_name,
        status=status,
    )
    event = {
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "status": status,
        "content": content,
        "command": _command_text(result),
        "result": result,
    }
    return tool_message, event


def _result_is_ok(result: Any) -> bool:
    return bool(result.get("ok", False)) if isinstance(result, dict) else True


def _tool_result_text(result: Any) -> str:
    if isinstance(result, dict):
        stdout = str(result.get("stdout", "")).strip()
        stderr = str(result.get("stderr", "")).strip()
        command = _command_text(result)
        fragments = []
        if command:
            fragments.append(f"$ {command}")
        if stdout:
            fragments.append(stdout)
        if stderr:
            fragments.append(stderr)
        return "\n".join(fragment for fragment in fragments if fragment) or json.dumps(result, indent=2)
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str)


def _tool_message_content(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, sort_keys=True, default=str)


def _command_text(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    command = result.get("command")
    if isinstance(command, list):
        return " ".join(shlex.quote(str(part)) for part in command)
    if isinstance(command, str):
        return command
    command_text = result.get("command_text")
    return str(command_text) if isinstance(command_text, str) else None


def _message_to_text(message: BaseMessage | Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content) if content is not None else ""
