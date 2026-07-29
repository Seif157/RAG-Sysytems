"""Ingesting an uploaded document into the index."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from rag.core.telemetry import stage
from rag.domain.models import (
    Chunk,
    ChunkCandidate,
    ChunkMetadata,
    Document,
    DocumentProperties,
    DocumentType,
    RawDocument,
    SparseVector,
)
from rag.domain.policies import derive_chunk_id
from rag.domain.ports import ChunkingStrategy, DocumentParser, Embedder, VectorStore

__all__ = ["IngestDocumentUseCase"]


def _utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class _Loader(Protocol):
    """Validates an upload and resolves its format."""

    def load(self, filename: str, content: bytes) -> RawDocument:
        """Validate and type an upload."""
        ...


class _Parsers(Protocol):
    """Selects the parser for a format."""

    def for_type(self, document_type: DocumentType) -> DocumentParser:
        """Return the parser for a format."""
        ...


class _Uploads(Protocol):
    """Stores original uploaded files."""

    def save(self, document_id: str, filename: str, content: bytes) -> object:
        """Persist an upload and return its path."""
        ...


class _Catalog(Protocol):
    """Records what has been ingested."""

    def get(self, document_id: str) -> Document | None:
        """Look up a document."""
        ...

    def put(self, document: Document) -> None:
        """Record a document."""
        ...


class _SparseEncoder(Protocol):
    """Produces lexical vectors for hybrid retrieval."""

    async def encode_documents(self, texts: Sequence[str]) -> tuple[SparseVector, ...]:
        """Encode passages."""
        ...


class IngestDocumentUseCase:
    """Turns an uploaded file into indexed, searchable chunks.

    The ordering at the end is deliberate: chunks for the new version are
    written *before* the previous version's are deleted. Deleting first would
    leave a window in which the document is unsearchable; writing first leaves
    only a window of duplicate versions (ADR-014).
    """

    def __init__(
        self,
        loader: _Loader,
        parsers: _Parsers,
        chunker: ChunkingStrategy,
        embedder: Embedder,
        store: VectorStore,
        uploads: _Uploads,
        catalog: _Catalog,
        clock: Callable[[], datetime] = _utc_now,
        sparse_encoder: _SparseEncoder | None = None,
    ) -> None:
        """Wire the use case.

        Args:
            loader: Validates uploads and resolves their format.
            parsers: Supplies the parser for a format.
            chunker: Chooses chunk boundaries.
            embedder: Produces dense vectors.
            store: Holds chunks and vectors.
            uploads: Keeps the original file.
            catalog: Records what has been ingested.
            clock: Supplies the ingestion timestamp. Injected so tests are
                deterministic without freezing global state.
            sparse_encoder: Produces lexical vectors for hybrid retrieval.
                ``None`` for dense-only deployments -- absent rather than
                disabled, because encoding vectors nothing will ever query is
                pure waste.
        """
        self._loader = loader
        self._parsers = parsers
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._uploads = uploads
        self._catalog = catalog
        self._clock = clock
        self._sparse_encoder = sparse_encoder

    async def execute(self, filename: str, content: bytes) -> Document:
        """Ingest an uploaded file.

        Args:
            filename: Original filename as uploaded.
            content: The uploaded bytes. These are what gets indexed -- the
                prototype's central defect was reading a hardcoded folder
                instead.

        Returns:
            A record of what was indexed.

        Raises:
            DocumentTooLargeError: If the upload exceeds the configured limit.
            UnsupportedFormatError: If the format is unsupported or empty.
            DocumentParsingError: If the file cannot be read.
            EmbeddingError: If embedding fails.
            VectorStoreError: If the write fails.
        """
        raw = self._loader.load(filename, content)
        document_id = self._document_id(filename)

        previous = self._catalog.get(document_id)
        if previous is not None and previous.content_hash == raw.content_hash:
            # Identical content: nothing to re-embed and nothing to re-index.
            return previous

        version = previous.ingest_version + 1 if previous else 1
        ingested_at = self._clock()

        with stage("document.parse", filename=filename) as parsing:
            parsed = self._parsers.for_type(raw.document_type).parse(raw)
            parsing.record(blocks=parsed.block_count)

        with stage("chunking", strategy=self._chunker.name) as chunking:
            candidates = self._chunker.chunk(parsed)
            chunking.record(chunks=len(candidates))

        chunks = self._build_chunks(
            candidates, raw, parsed.properties, document_id, version, ingested_at
        )

        if chunks:
            texts = [chunk.text for chunk in chunks]

            with stage("embedding", model=self._embedder.model_id) as embedding:
                vectors = await self._embedder.embed_documents(texts)
                embedding.record(vectors=len(vectors))

            sparse_vectors: tuple[SparseVector, ...] | None = None
            if self._sparse_encoder is not None:
                with stage("embedding.sparse") as sparse:
                    sparse_vectors = await self._sparse_encoder.encode_documents(texts)
                    sparse.record(vectors=len(sparse_vectors))

            with stage("vector.upsert") as upsert:
                written = await self._store.upsert(chunks, vectors, sparse_vectors)
                upsert.record(points=written, sparse=sparse_vectors is not None)

        # Write-then-delete: the new version is live before the old one goes.
        removed = await self._store.delete_document(document_id, before_version=version)

        path = self._uploads.save(document_id, filename, content)
        document = Document(
            document_id=document_id,
            filename=filename,
            document_type=raw.document_type,
            content_hash=raw.content_hash,
            size_bytes=raw.size_bytes,
            source_path=str(path),
            ingest_version=version,
            chunk_count=len(chunks),
            ingested_at=ingested_at,
        )
        self._catalog.put(document)

        with stage("ingestion.complete") as done:
            done.record(
                document_id=document_id,
                version=version,
                chunks=len(chunks),
                superseded_chunks_removed=removed,
            )
        return document

    @staticmethod
    def _document_id(filename: str) -> str:
        """Derive a document id from its name.

        Keyed on the name rather than the content, so uploading a corrected
        version of ``report.pdf`` replaces the old one instead of leaving two
        copies to compete in retrieval.
        """
        return hashlib.sha256(filename.encode()).hexdigest()[:32]

    def _build_chunks(
        self,
        candidates: Sequence[ChunkCandidate],
        raw: RawDocument,
        properties: DocumentProperties,
        document_id: str,
        version: int,
        ingested_at: datetime,
    ) -> list[Chunk]:
        """Stamp identity and provenance onto chunk candidates.

        The chunker chose the boundaries; this decides what each chunk *is*.
        Keeping the two apart is what lets a chunking strategy be tested with no
        document id, no clock and no configuration.

        Document-level properties -- author, title, creation date -- are copied
        onto every chunk. Denormalising them is what makes them filterable at
        retrieval time: a filter runs against the chunk payload, so an author
        held only on the document would be unusable in a search.
        """
        chunks: list[Chunk] = []
        for index, candidate in enumerate(candidates):
            chunks.append(
                Chunk(
                    text=candidate.text,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        chunk_id=derive_chunk_id(document_id, index, candidate.text),
                        ingest_version=version,
                        filename=raw.filename,
                        document_type=raw.document_type,
                        chunk_index=index,
                        char_start=candidate.char_start,
                        char_end=candidate.char_end,
                        token_count=candidate.token_count,
                        chunking_strategy=self._chunker.name,
                        embedding_model_id=self._embedder.model_id,
                        ingested_at=ingested_at,
                        page_number=candidate.page_number,
                        section=candidate.section,
                        heading=candidate.heading,
                        heading_path=candidate.heading_path,
                        author=properties.author,
                        title=properties.title,
                        created_at=properties.created_at,
                        language=properties.language,
                    ),
                )
            )
        return chunks
