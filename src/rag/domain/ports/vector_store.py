"""Port: vector storage and search.

One interface rather than the three the larger design segregated. With a single
adapter and a single caller of each operation, splitting search from write buys
nothing here and costs a reader three files to understand one concept.

The reason it *was* split -- so a query-path router could not delete a vector --
does not apply: there is no router, and the query path is one use case.

Kept as a port for one concrete reason: an in-memory implementation makes the
whole retrieval pipeline testable without Docker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag.domain.models import (
    Chunk,
    CollectionInfo,
    CollectionSpec,
    DenseVector,
    FilterExpression,
    ScoredChunk,
    SparseVector,
)

__all__ = ["VectorStore"]


class VectorStore(ABC):
    """Stores chunk vectors and searches them.

    Implementations accept a *domain* filter expression, never a vendor filter
    type. Translation happens inside the adapter, which is the single place a
    vendor filter type is allowed to exist (ADR-013) -- and the reason the
    in-memory test double can honour the same filters as Qdrant.
    """

    @abstractmethod
    async def ensure_collection(self, spec: CollectionSpec) -> None:
        """Create the collection if absent, and verify it if present.

        Verification is not optional. A collection whose vectors came from a
        different embedding model answers queries plausibly and wrongly, so an
        incompatible existing collection is an error rather than something to
        adapt to (ADR-015).

        Args:
            spec: The shape the running configuration requires.

        Raises:
            CollectionMismatchError: If a collection exists but is incompatible.
            VectorStoreError: If creation or inspection fails.
        """

    @abstractmethod
    async def collection_info(self) -> CollectionInfo:
        """Describe the collection as it currently exists.

        Returns:
            The collection's actual shape, for compatibility checking.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            VectorStoreError: If inspection fails.
        """

    @abstractmethod
    async def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[DenseVector],
        sparse_vectors: Sequence[SparseVector] | None = None,
    ) -> int:
        """Insert or replace chunks and their vectors.

        Idempotent on chunk id. Because chunk ids are deterministic, a retried
        ingestion overwrites rather than duplicating (ADR-014).

        Args:
            chunks: Chunks to store.
            dense_vectors: Dense vectors, positionally aligned with ``chunks``.
            sparse_vectors: Sparse vectors, positionally aligned with ``chunks``.
                Omitted when running dense-only.

        Returns:
            The number of points written.

        Raises:
            EmbeddingDimensionMismatchError: If a vector's length disagrees with
                the collection.
            VectorStoreError: If the write fails.
        """

    @abstractmethod
    async def search_dense(
        self,
        vector: DenseVector,
        top_k: int,
        filters: FilterExpression | None = None,
    ) -> tuple[ScoredChunk, ...]:
        """Find the nearest chunks by dense vector similarity.

        Args:
            vector: Query embedding. Its length must match the collection.
            top_k: Maximum number of results.
            filters: Optional metadata filter.

        Returns:
            Up to ``top_k`` chunks, most similar first, scored with
            :attr:`~rag.domain.models.ScoreSource.DENSE`.

        Raises:
            VectorStoreError: If the search fails.
        """

    @abstractmethod
    async def search_sparse(
        self,
        vector: SparseVector,
        top_k: int,
        filters: FilterExpression | None = None,
    ) -> tuple[ScoredChunk, ...]:
        """Find the best matching chunks by lexical (sparse) similarity.

        Args:
            vector: Query sparse vector. An empty vector correctly matches
                nothing rather than everything.
            top_k: Maximum number of results.
            filters: Optional metadata filter.

        Returns:
            Up to ``top_k`` chunks, best match first, scored with
            :attr:`~rag.domain.models.ScoreSource.SPARSE`.

        Raises:
            VectorStoreError: If the search fails.
        """

    @abstractmethod
    async def delete_document(self, document_id: str, before_version: int | None = None) -> int:
        """Delete a document's chunks, optionally only superseded versions.

        Re-indexing writes the new version first and then calls this with
        ``before_version`` set. Deleting first would leave a window in which the
        document is unsearchable; writing first leaves only a window of stale
        data, which the current-version search filter already excludes
        (ADR-014).

        Args:
            document_id: The document whose chunks to remove.
            before_version: When given, delete only versions strictly below it.
                When omitted, delete every version.

        Returns:
            The number of points deleted.

        Raises:
            VectorStoreError: If the deletion fails.
        """
