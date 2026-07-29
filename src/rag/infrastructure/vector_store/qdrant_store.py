"""Qdrant vector store adapter."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from rag.domain.errors import (
    CollectionMismatchError,
    CollectionNotFoundError,
    EmbeddingDimensionMismatchError,
    VectorStoreError,
    VectorStoreUnavailableError,
)
from rag.domain.models import (
    FILTERABLE_FIELDS,
    Chunk,
    ChunkMetadata,
    CollectionInfo,
    CollectionSpec,
    DenseVector,
    DistanceMetric,
    DocumentType,
    FieldFilter,
    FilterExpression,
    FilterOperator,
    MetadataField,
    ScoredChunk,
    ScoreSource,
    SparseVector,
)
from rag.domain.ports import VectorStore
from rag.infrastructure.vector_store.filter_translator import to_qdrant_filter

__all__ = ["QdrantVectorStore"]

_DENSE_VECTOR = "dense"
_SPARSE_VECTOR = "sparse"
_TEXT_KEY = "text"
_MODEL_KEY = "embedding_model_id"

_DISTANCES: dict[DistanceMetric, qmodels.Distance] = {
    DistanceMetric.COSINE: qmodels.Distance.COSINE,
    DistanceMetric.DOT: qmodels.Distance.DOT,
    DistanceMetric.EUCLIDEAN: qmodels.Distance.EUCLID,
}


class QdrantVectorStore(VectorStore):
    """Stores chunks and their vectors in Qdrant.

    Point ids are UUIDs derived deterministically from the chunk id, because
    Qdrant accepts only integers or UUIDs as ids while chunk ids are hex
    strings. Deriving rather than generating preserves the idempotency that
    makes re-ingestion safe (ADR-014).
    """

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        """Initialise the store.

        Args:
            client: Configured Qdrant client.
            collection_name: Collection holding chunk vectors.
        """
        self._client = client
        self._collection = collection_name

    # --------------------------------------------------------- collection #
    async def ensure_collection(self, spec: CollectionSpec) -> None:
        """Create the collection and its indexes if absent, verify if present.

        Index creation lives here rather than behind a separate call because
        "make the store ready for this spec" is one concept, and splitting it
        left the indexes to be forgotten -- which they were.
        """
        try:
            exists = await self._client.collection_exists(self._collection)
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"could not reach Qdrant: {exc}", context={"collection": self._collection}
            ) from exc

        if exists:
            info = await self.collection_info()
            reasons = [
                *info.incompatibility_reasons(spec),
                *await self._sparse_scoring_reasons(spec),
            ]
            if reasons:
                raise CollectionMismatchError(
                    f"collection {self._collection!r} is incompatible: " + "; ".join(reasons),
                    context={"collection": self._collection, "reasons": list(reasons)},
                )
            # Re-run on every start-up: an upgrade that makes a new field
            # filterable needs its index, and creating one that exists is free.
            await self._create_payload_indexes()
            return

        # modifier=IDF makes Qdrant compute inverse document frequency across the
        # collection at query time. The encoder only supplies BM25's
        # term-frequency half, because IDF needs corpus-wide counts a stateless
        # encoder cannot have -- and a snapshot taken at ingest would go stale as
        # the corpus grows. Without this, sparse scoring is raw term frequency
        # and common words dominate.
        sparse_config = (
            {_SPARSE_VECTOR: qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)}
            if spec.supports_sparse
            else None
        )
        try:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    _DENSE_VECTOR: qmodels.VectorParams(
                        size=spec.dense_dimension, distance=_DISTANCES[spec.distance]
                    )
                },
                sparse_vectors_config=sparse_config,
            )
            # Record which model produced these vectors. Without it, a later run
            # cannot tell a compatible collection from one that will answer
            # plausibly and wrongly (ADR-015).
            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    qmodels.PointStruct(
                        id=_MARKER_POINT_ID,
                        vector={_DENSE_VECTOR: [0.0] * spec.dense_dimension},
                        payload={_MODEL_KEY: spec.embedding_model_id, "_marker": True},
                    )
                ],
            )
        except Exception as exc:
            raise VectorStoreError(
                f"could not create collection {self._collection!r}: {exc}",
                context={"collection": self._collection},
            ) from exc

        await self._create_payload_indexes()

    async def collection_info(self) -> CollectionInfo:
        """Describe the collection as it currently exists."""
        try:
            info = await self._client.get_collection(self._collection)
        except UnexpectedResponse as exc:
            raise CollectionNotFoundError(
                f"collection {self._collection!r} does not exist",
                context={"collection": self._collection},
            ) from exc
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"could not reach Qdrant: {exc}", context={"collection": self._collection}
            ) from exc

        vectors = info.config.params.vectors or {}
        dense = vectors.get(_DENSE_VECTOR) if isinstance(vectors, dict) else vectors
        if dense is None:
            raise CollectionMismatchError(
                f"collection {self._collection!r} has no {_DENSE_VECTOR!r} vector; "
                f"it was not created by this application",
                context={"collection": self._collection},
            )
        sparse = info.config.params.sparse_vectors or {}

        distance = next(
            (name for name, value in _DISTANCES.items() if value == dense.distance),
            DistanceMetric.COSINE,
        )
        return CollectionInfo(
            name=self._collection,
            dense_dimension=dense.size,
            distance=distance,
            embedding_model_id=await self._recorded_model(),
            supports_sparse=_SPARSE_VECTOR in sparse,
            # The marker point holds the embedding model id and is not a chunk.
            # Counting it would report "1 point" for an empty collection.
            points_count=max(0, (info.points_count or 0) - 1),
        )

    async def _sparse_scoring_reasons(self, spec: CollectionSpec) -> list[str]:
        """Report whether an existing collection scores sparse vectors correctly.

        A collection created before IDF was configured has sparse vectors that
        score on raw term frequency, so common words dominate and lexical
        retrieval quietly gets worse. Nothing in the results would reveal it,
        which is exactly why this refuses rather than warns.
        """
        if not spec.supports_sparse:
            return []

        try:
            info = await self._client.get_collection(self._collection)
        except Exception:  # pragma: no cover - already handled by the caller
            return []

        sparse = info.config.params.sparse_vectors or {}
        params = sparse.get(_SPARSE_VECTOR)
        if params is None:
            return ["collection has no sparse vectors, hybrid retrieval requires them"]
        if params.modifier != qmodels.Modifier.IDF:
            return [
                "collection's sparse vectors were created without the IDF modifier, "
                "so lexical scoring would ignore term rarity; recreate the collection"
            ]
        return []

    async def _recorded_model(self) -> str:
        """Read the embedding model recorded when the collection was created."""
        try:
            records = await self._client.retrieve(
                collection_name=self._collection,
                ids=[_MARKER_POINT_ID],
                with_payload=True,
            )
        except Exception:  # pragma: no cover - treated as "unknown"
            return ""
        if not records or not records[0].payload:
            return ""
        return str(records[0].payload.get(_MODEL_KEY, ""))

    # -------------------------------------------------------------- writes #
    async def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[DenseVector],
        sparse_vectors: Sequence[SparseVector] | None = None,
    ) -> int:
        """Insert or replace chunks and their vectors."""
        if len(chunks) != len(dense_vectors):
            raise EmbeddingDimensionMismatchError(
                f"{len(chunks)} chunks but {len(dense_vectors)} vectors"
            )

        points: list[qmodels.PointStruct] = []
        for index, chunk in enumerate(chunks):
            vector: dict[str, Any] = {_DENSE_VECTOR: list(dense_vectors[index])}
            if sparse_vectors is not None:
                sparse = sparse_vectors[index]
                vector[_SPARSE_VECTOR] = qmodels.SparseVector(
                    indices=list(sparse.indices), values=list(sparse.values)
                )
            points.append(
                qmodels.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=_to_payload(chunk),
                )
            )

        try:
            await self._client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:
            raise VectorStoreError(
                f"could not write {len(points)} points: {exc}",
                context={"collection": self._collection, "points": len(points)},
            ) from exc
        return len(points)

    async def delete_document(self, document_id: str, before_version: int | None = None) -> int:
        """Delete a document's chunks, optionally only superseded versions."""
        clause: FilterExpression = FieldFilter(
            MetadataField.DOCUMENT_ID, FilterOperator.EQ, document_id
        )
        if before_version is not None:
            clause = clause & FieldFilter(
                MetadataField.INGEST_VERSION, FilterOperator.LT, before_version
            )

        selector = to_qdrant_filter(clause)
        assert selector is not None, "a non-None expression always translates"

        try:
            result = await self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(filter=selector),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"could not delete chunks for {document_id!r}: {exc}",
                context={"document_id": document_id},
            ) from exc
        return getattr(result, "operation_id", 0) or 0

    # ------------------------------------------------------------- search #
    async def search_dense(
        self,
        vector: DenseVector,
        top_k: int,
        filters: FilterExpression | None = None,
    ) -> tuple[ScoredChunk, ...]:
        """Find the nearest chunks by dense similarity."""
        return await self._search(
            query=list(vector),
            using=_DENSE_VECTOR,
            top_k=top_k,
            filters=filters,
            source=ScoreSource.DENSE,
        )

    async def search_sparse(
        self,
        vector: SparseVector,
        top_k: int,
        filters: FilterExpression | None = None,
    ) -> tuple[ScoredChunk, ...]:
        """Find the best matching chunks by lexical similarity."""
        if vector.non_zero_count == 0:
            # No shared vocabulary: correctly matches nothing, and asking Qdrant
            # would be a round trip to learn that.
            return ()
        return await self._search(
            query=qmodels.SparseVector(indices=list(vector.indices), values=list(vector.values)),
            using=_SPARSE_VECTOR,
            top_k=top_k,
            filters=filters,
            source=ScoreSource.SPARSE,
        )

    async def _search(
        self,
        query: Any,
        using: str,
        top_k: int,
        filters: FilterExpression | None,
        source: ScoreSource,
    ) -> tuple[ScoredChunk, ...]:
        """Run a query and map the results back to domain types."""
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query,
                using=using,
                limit=top_k,
                query_filter=to_qdrant_filter(filters),
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"{using} search failed: {exc}", context={"collection": self._collection}
            ) from exc

        results: list[ScoredChunk] = []
        for point in response.points:
            payload = point.payload or {}
            if payload.get("_marker"):
                continue
            results.append(
                ScoredChunk(chunk=_from_payload(payload), score=point.score, source=source)
            )
        return tuple(results)

    async def _create_payload_indexes(self) -> None:
        """Index the filterable metadata fields.

        Without these, every metadata filter is a full scan and degrades
        linearly with the corpus -- the difference between filtering being a
        feature and being a trap.

        Only the filterable fields are indexed: indexing character offsets and
        provenance strings would inflate memory for no query benefit.

        Failures are swallowed deliberately. The common one is "index already
        exists", and none of them should stop the application from serving
        queries it can still answer, just more slowly.
        """
        for field in sorted(FILTERABLE_FIELDS, key=lambda f: f.value):
            schema = _INDEX_SCHEMA.get(field, qmodels.PayloadSchemaType.KEYWORD)
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field.value,
                    field_schema=schema,
                )
            except Exception:
                continue


_MARKER_POINT_ID = "00000000-0000-0000-0000-000000000000"

_INDEX_SCHEMA: dict[MetadataField, qmodels.PayloadSchemaType] = {
    MetadataField.PAGE_NUMBER: qmodels.PayloadSchemaType.INTEGER,
    MetadataField.CHUNK_INDEX: qmodels.PayloadSchemaType.INTEGER,
    MetadataField.INGEST_VERSION: qmodels.PayloadSchemaType.INTEGER,
    MetadataField.INGESTED_AT: qmodels.PayloadSchemaType.DATETIME,
    MetadataField.CREATED_AT: qmodels.PayloadSchemaType.DATETIME,
}


def _point_id(chunk_id: str) -> str:
    """Derive a stable UUID from a chunk id.

    Qdrant accepts only integers or UUIDs as point ids. Deriving one from the
    chunk id keeps upserts idempotent; generating one would not.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _to_payload(chunk: Chunk) -> dict[str, Any]:
    """Flatten a chunk into a Qdrant payload."""
    metadata = chunk.metadata
    payload: dict[str, Any] = {
        _TEXT_KEY: chunk.text,
        "document_id": metadata.document_id,
        "chunk_id": metadata.chunk_id,
        "ingest_version": metadata.ingest_version,
        "filename": metadata.filename,
        "document_type": metadata.document_type.value,
        "chunk_index": metadata.chunk_index,
        "char_start": metadata.char_start,
        "char_end": metadata.char_end,
        "token_count": metadata.token_count,
        "chunking_strategy": metadata.chunking_strategy,
        _MODEL_KEY: metadata.embedding_model_id,
        "ingested_at": metadata.ingested_at.isoformat(),
        "page_number": metadata.page_number,
        "section": metadata.section,
        "heading": metadata.heading,
        "heading_path": list(metadata.heading_path) if metadata.heading_path else None,
        "author": metadata.author,
        "title": metadata.title,
        "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
        "language": metadata.language,
    }
    return payload


def _from_payload(payload: dict[str, Any]) -> Chunk:
    """Rebuild a chunk from a Qdrant payload."""
    heading_path = payload.get("heading_path")
    created_at = payload.get("created_at")
    return Chunk(
        text=payload[_TEXT_KEY],
        metadata=ChunkMetadata(
            document_id=payload["document_id"],
            chunk_id=payload["chunk_id"],
            ingest_version=payload["ingest_version"],
            filename=payload["filename"],
            document_type=DocumentType(payload["document_type"]),
            chunk_index=payload["chunk_index"],
            char_start=payload["char_start"],
            char_end=payload["char_end"],
            token_count=payload["token_count"],
            chunking_strategy=payload["chunking_strategy"],
            embedding_model_id=payload[_MODEL_KEY],
            ingested_at=datetime.fromisoformat(payload["ingested_at"]),
            page_number=payload.get("page_number"),
            section=payload.get("section"),
            heading=payload.get("heading"),
            heading_path=tuple(heading_path) if heading_path else None,
            author=payload.get("author"),
            title=payload.get("title"),
            created_at=datetime.fromisoformat(created_at) if created_at else None,
            language=payload.get("language"),
        ),
    )
