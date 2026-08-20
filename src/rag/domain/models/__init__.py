"""Entities and value objects.

Models validate their own structural invariants on construction. Business rules
that span models live in :mod:`rag.domain.policies`.
"""

from rag.domain.models.answer import Answer, Citation, TokenUsage
from rag.domain.models.chunk import Chunk, ScoredChunk, ScoreSource
from rag.domain.models.collection import (
    CollectionInfo,
    CollectionSpec,
    DistanceMetric,
)
from rag.domain.models.context import ContextBlock
from rag.domain.models.conversation import Conversation, Role, Turn
from rag.domain.models.document import Document
from rag.domain.models.filters import (
    And,
    FieldFilter,
    FilterExpression,
    FilterOperator,
    Not,
    Or,
)
from rag.domain.models.generation import GenerationParams, LLMResponse, Prompt
from rag.domain.models.ingestion import (
    ChunkCandidate,
    ContentBlock,
    DocumentProperties,
    MetadataFragment,
    ParsedDocument,
    RawDocument,
)
from rag.domain.models.metadata import (
    FILTERABLE_FIELDS,
    ChunkMetadata,
    DocumentAccessPolicy,
    DocumentAccessScope,
    DocumentType,
    MetadataField,
)
from rag.domain.models.query import Query, RetrievalRequest
from rag.domain.models.tools import LLMMessage, ToolCall, ToolDefinition, ToolResult
from rag.domain.models.vectors import DenseVector, SparseVector

__all__ = [
    "FILTERABLE_FIELDS",
    "And",
    "Answer",
    "Chunk",
    "ChunkCandidate",
    "ChunkMetadata",
    "Citation",
    "CollectionInfo",
    "CollectionSpec",
    "ContentBlock",
    "ContextBlock",
    "Conversation",
    "DenseVector",
    "DistanceMetric",
    "Document",
    "DocumentAccessPolicy",
    "DocumentAccessScope",
    "DocumentProperties",
    "DocumentType",
    "FieldFilter",
    "FilterExpression",
    "FilterOperator",
    "GenerationParams",
    "LLMMessage",
    "LLMResponse",
    "MetadataField",
    "MetadataFragment",
    "Not",
    "Or",
    "ParsedDocument",
    "Prompt",
    "Query",
    "RawDocument",
    "RetrievalRequest",
    "Role",
    "ScoreSource",
    "ScoredChunk",
    "SparseVector",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "Turn",
]
