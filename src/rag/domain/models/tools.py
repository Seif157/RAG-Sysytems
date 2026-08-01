"""Provider-neutral function-calling value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = ["LLMMessage", "ToolCall", "ToolDefinition", "ToolResult"]


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A callable capability advertised to an LLM."""

    name: str
    description: str
    parameters_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the advertised tool contract."""
        if not self.name.strip():
            raise ValueError("tool name must be non-empty")
        if not self.description.strip():
            raise ValueError("tool description must be non-empty")
        if self.parameters_schema.get("type") != "object":
            raise ValueError("tool parameters_schema must describe an object")
        object.__setattr__(self, "parameters_schema", _frozen_mapping(self.parameters_schema))


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A provider-neutral request from the model to execute a tool."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate and defensively freeze provider arguments."""
        if not self.call_id.strip():
            raise ValueError("tool call_id must be non-empty")
        if not self.name.strip():
            raise ValueError("tool name must be non-empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool arguments must be an object mapping")
        object.__setattr__(self, "arguments", _frozen_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A controlled tool result returned to the model."""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    evidence: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Validate correlation identifiers."""
        if not self.call_id.strip() or not self.tool_name.strip():
            raise ValueError("tool result identifiers must be non-empty")


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One neutral chat message, including tool exchanges."""

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate role-specific message invariants."""
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role!r}")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
