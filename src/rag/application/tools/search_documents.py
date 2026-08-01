"""The single read-only tool exposed to the model."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag.application.services import ContextBuilder
from rag.domain.errors import RerankingError, ToolValidationError
from rag.domain.models import (
    ContextBlock,
    FieldFilter,
    FilterOperator,
    MetadataField,
    RetrievalRequest,
    ScoredChunk,
    ToolDefinition,
)
from rag.domain.ports import Reranker, Retriever

__all__ = ["SEARCH_DOCUMENTS_TOOL", "SearchDocumentsTool", "serialize_context"]

SEARCH_DOCUMENTS_TOOL = ToolDefinition(
    name="search_documents",
    description=(
        "Search indexed documents for evidence. Use before answering any question "
        "that may depend on document content."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 1000},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
            "document_ids": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "maxItems": 50,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True, slots=True)
class SearchArguments:
    query: str
    top_k: int
    document_ids: tuple[str, ...] | None


class SearchDocumentsTool:
    """Validate model arguments and reuse the production retrieval pipeline."""

    definition = SEARCH_DOCUMENTS_TOOL

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        *,
        max_top_k: int,
        rerank_top_k: int,
        max_query_chars: int = 1000,
        max_document_ids: int = 50,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._context_builder = context_builder
        self._max_top_k = max_top_k
        self._rerank_top_k = rerank_top_k
        self._max_query_chars = max_query_chars
        self._max_document_ids = max_document_ids

    def validate(
        self, raw: Mapping[str, Any], allowed_document_ids: frozenset[str] | None
    ) -> SearchArguments:
        """Validate types, limits, unknown fields, and document authorization."""
        unknown = set(raw) - {"query", "top_k", "document_ids"}
        if unknown:
            raise ToolValidationError(f"unknown search arguments: {sorted(unknown)}")
        query = raw.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolValidationError("query must be a non-empty string")
        query = query.strip()
        if len(query) > self._max_query_chars:
            raise ToolValidationError(f"query exceeds {self._max_query_chars} characters")
        top_k = raw.get("top_k", min(10, self._max_top_k))
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= self._max_top_k
        ):
            raise ToolValidationError(f"top_k must be an integer from 1 to {self._max_top_k}")
        requested = raw.get("document_ids")
        ids: tuple[str, ...] | None = None
        if requested is not None:
            if isinstance(requested, str) or not isinstance(requested, Sequence):
                raise ToolValidationError("document_ids must be an array or null")
            if len(requested) > self._max_document_ids or not all(
                isinstance(x, str) and x.strip() for x in requested
            ):
                raise ToolValidationError("document_ids contains invalid values or is too large")
            ids = tuple(dict.fromkeys(str(x).strip() for x in requested))
            if allowed_document_ids is not None and not set(ids) <= allowed_document_ids:
                raise ToolValidationError("one or more requested documents are not authorized")
        return SearchArguments(query, top_k, ids)

    async def execute(self, arguments: SearchArguments) -> tuple[str, ContextBlock]:
        """Retrieve and rerank evidence using the existing pipeline."""
        filters = None
        if arguments.document_ids:
            filters = FieldFilter(
                MetadataField.DOCUMENT_ID, FilterOperator.IN, arguments.document_ids
            )
        candidates = await self._retriever.retrieve(
            RetrievalRequest(query_text=arguments.query, top_k=arguments.top_k, filters=filters)
        )
        try:
            ranked = (
                await self._reranker.rerank(
                    arguments.query, candidates, min(self._rerank_top_k, len(candidates))
                )
                if candidates
                else ()
            )
        except RerankingError:
            ranked = tuple(candidates[: self._rerank_top_k])
        context = self._context_builder.build(self._deduplicate(ranked))
        return serialize_context(context, arguments.query), context

    @staticmethod
    def _deduplicate(chunks: Sequence[ScoredChunk]) -> tuple[ScoredChunk, ...]:
        return tuple({chunk.chunk_id: chunk for chunk in chunks}.values())


def serialize_context(context: ContextBlock, query: str) -> str:
    """Serialize evidence with markers matching this exact context block."""
    passages = []
    for index, scored in enumerate(context.chunks, start=1):
        metadata = scored.chunk.metadata
        passages.append(
            {
                "marker": f"[{index}]",
                "chunk_id": scored.chunk_id,
                "document_id": metadata.document_id,
                "filename": metadata.filename,
                "page": metadata.page_number,
                "section": metadata.section,
                "text": scored.text,
                "score": scored.score,
            }
        )
    return json.dumps(
        {"query": query, "passages": passages, "result_count": len(passages)}, ensure_ascii=False
    )
