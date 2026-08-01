"""Allowlisted, bounded execution of application tools."""

from __future__ import annotations

import asyncio

from rag.application.tools.search_documents import SearchDocumentsTool
from rag.domain.errors import ToolExecutionError, ToolValidationError
from rag.domain.models import ContextBlock, ToolCall, ToolResult

__all__ = ["ToolExecutor"]


class ToolExecutor:
    """Execute only explicitly registered read-only tools."""

    def __init__(self, search: SearchDocumentsTool, *, timeout_s: float) -> None:
        self._search = search
        self._timeout_s = timeout_s

    async def execute(
        self, call: ToolCall, *, allowed_document_ids: frozenset[str] | None = None
    ) -> tuple[ToolResult, ContextBlock | None]:
        """Validate and execute one allowlisted call within its deadline."""
        if call.name != self._search.definition.name:
            return ToolResult(call.call_id, call.name, "Unknown or unavailable tool.", True), None
        try:
            arguments = self._search.validate(call.arguments, allowed_document_ids)
            content, context = await asyncio.wait_for(
                self._search.execute(arguments), timeout=self._timeout_s
            )
            return ToolResult(call.call_id, call.name, content, evidence=context.chunks), context
        except ToolValidationError as exc:
            return ToolResult(call.call_id, call.name, exc.message, True), None
        except TimeoutError:
            return ToolResult(call.call_id, call.name, "The search timed out.", True), None
        except ToolExecutionError as exc:
            return ToolResult(call.call_id, call.name, exc.message, True), None
