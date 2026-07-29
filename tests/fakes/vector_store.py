"""An in-memory vector store, so retrieval can be tested without Docker."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from rag.domain.errors import CollectionMismatchError, EmbeddingDimensionMismatchError
from rag.domain.models import (
    And,
    Chunk,
    CollectionInfo,
    CollectionSpec,
    DenseVector,
    FieldFilter,
    FilterExpression,
    FilterOperator,
    Not,
    Or,
    ScoredChunk,
    ScoreSource,
    SparseVector,
)
from rag.domain.ports import VectorStore

__all__ = ["InMemoryVectorStore", "matches_filter"]


def _field_value(chunk: Chunk, field: str) -> Any:
    """Read a metadata field from a chunk."""
    return getattr(chunk.metadata, field, None)


def _matches_field(chunk: Chunk, clause: FieldFilter) -> bool:
    """Evaluate a single predicate against a chunk."""
    actual = _field_value(chunk, clause.field.value)
    expected = clause.value

    match clause.operator:
        case FilterOperator.EQ:
            return bool(actual == expected)
        case FilterOperator.NE:
            return bool(actual != expected)
        case FilterOperator.IS_NULL:
            return actual is None
        case FilterOperator.IS_NOT_NULL:
            return actual is not None
        case FilterOperator.IN:
            return actual in expected
        case FilterOperator.NOT_IN:
            return actual not in expected
        case FilterOperator.CONTAINS:
            return actual is not None and expected in actual

    if actual is None:
        # A comparison against a missing value is false, not an error: a TXT
        # file genuinely has no page number, and filtering on one should exclude
        # it rather than crash the query.
        return False

    match clause.operator:
        case FilterOperator.GT:
            return bool(actual > expected)
        case FilterOperator.GTE:
            return bool(actual >= expected)
        case FilterOperator.LT:
            return bool(actual < expected)
        case FilterOperator.LTE:
            return bool(actual <= expected)
        case _:  # pragma: no cover - every operator is handled above
            raise NotImplementedError(clause.operator)


def matches_filter(chunk: Chunk, expression: FilterExpression | None) -> bool:
    """Evaluate a domain filter expression against a chunk.

    Shared with the contract suite, so the in-memory store and the real adapter
    are held to the same filter semantics rather than drifting apart.

    Args:
        chunk: The chunk to test.
        expression: The filter, or ``None`` to match everything.

    Returns:
        Whether the chunk satisfies the filter.
    """
    if expression is None:
        return True
    match expression:
        case FieldFilter():
            return _matches_field(chunk, expression)
        case And():
            return all(matches_filter(chunk, clause) for clause in expression.clauses)
        case Or():
            return any(matches_filter(chunk, clause) for clause in expression.clauses)
        case Not():
            return not matches_filter(chunk, expression.clause)
        case _:  # pragma: no cover - the hierarchy is sealed
            raise NotImplementedError(type(expression))


def _cosine(left: DenseVector, right: DenseVector) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


class InMemoryVectorStore(VectorStore):
    """Stores chunks and vectors in a dictionary.

    Honours the same domain filter language as the real adapter, so a use case
    tested against this behaves the same against Qdrant. Similarity is exact
    rather than approximate, which makes assertions deterministic.
    """

    def __init__(self, spec: CollectionSpec | None = None) -> None:
        """Initialise an empty store.

        Args:
            spec: Pre-existing collection shape, when a test needs to simulate
                a store that already holds vectors from another model.
        """
        self._spec = spec
        self._chunks: dict[str, Chunk] = {}
        self._dense: dict[str, DenseVector] = {}
        self._sparse: dict[str, SparseVector] = {}

    async def ensure_collection(self, spec: CollectionSpec) -> None:
        """Create or verify the collection."""
        if self._spec is not None and not (await self.collection_info()).is_compatible_with(spec):
            reasons = (await self.collection_info()).incompatibility_reasons(spec)
            raise CollectionMismatchError("; ".join(reasons))
        self._spec = spec

    async def collection_info(self) -> CollectionInfo:
        """Describe the collection as it exists."""
        assert self._spec is not None, "ensure_collection must be called first"
        return CollectionInfo(
            name=self._spec.name,
            dense_dimension=self._spec.dense_dimension,
            distance=self._spec.distance,
            embedding_model_id=self._spec.embedding_model_id,
            supports_sparse=self._spec.supports_sparse,
            points_count=len(self._chunks),
        )

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[DenseVector],
        sparse_vectors: Sequence[SparseVector] | None = None,
    ) -> int:
        """Insert or replace chunks, keyed on chunk id."""
        if self._spec is not None:
            for vector in dense_vectors:
                if len(vector) != self._spec.dense_dimension:
                    raise EmbeddingDimensionMismatchError(
                        f"expected {self._spec.dense_dimension}-dimensional vectors, "
                        f"got {len(vector)}"
                    )
        for index, chunk in enumerate(chunks):
            self._chunks[chunk.chunk_id] = chunk
            self._dense[chunk.chunk_id] = tuple(dense_vectors[index])
            if sparse_vectors is not None:
                self._sparse[chunk.chunk_id] = sparse_vectors[index]
        return len(chunks)

    async def search_dense(
        self,
        vector: DenseVector,
        top_k: int,
        filters: FilterExpression | None = None,
    ) -> tuple[ScoredChunk, ...]:
        """Rank stored chunks by cosine similarity."""
        scored = [
            ScoredChunk(
                chunk=chunk, score=_cosine(vector, self._dense[chunk_id]), source=ScoreSource.DENSE
            )
            for chunk_id, chunk in self._chunks.items()
            if matches_filter(chunk, filters)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return tuple(scored[:top_k])

    async def search_sparse(
        self,
        vector: SparseVector,
        top_k: int,
        filters: FilterExpression | None = None,
    ) -> tuple[ScoredChunk, ...]:
        """Rank stored chunks by sparse-vector overlap."""
        query = dict(zip(vector.indices, vector.values, strict=True))
        scored: list[ScoredChunk] = []
        for chunk_id, chunk in self._chunks.items():
            if not matches_filter(chunk, filters):
                continue
            stored = self._sparse.get(chunk_id)
            if stored is None:
                continue
            overlap = sum(
                weight * query.get(index, 0.0)
                for index, weight in zip(stored.indices, stored.values, strict=True)
            )
            if overlap > 0:
                scored.append(ScoredChunk(chunk=chunk, score=overlap, source=ScoreSource.SPARSE))
        scored.sort(key=lambda item: item.score, reverse=True)
        return tuple(scored[:top_k])

    async def delete_document(self, document_id: str, before_version: int | None = None) -> int:
        """Remove a document's chunks, optionally only superseded versions."""
        doomed = [
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if chunk.document_id == document_id
            and (before_version is None or chunk.metadata.ingest_version < before_version)
        ]
        for chunk_id in doomed:
            del self._chunks[chunk_id]
            self._dense.pop(chunk_id, None)
            self._sparse.pop(chunk_id, None)
        return len(doomed)

    # ------------------------------------------------------------- test aids #
    @property
    def chunk_count(self) -> int:
        """How many chunks are stored."""
        return len(self._chunks)

    @property
    def sparse_count(self) -> int:
        """How many chunks have a sparse vector."""
        return len(self._sparse)

    def chunk_ids(self) -> tuple[str, ...]:
        """Every stored chunk id."""
        return tuple(self._chunks)
