"""The record of an ingested document.

Qdrant holds the chunks; this describes the document they came from. There is no
separate database: the uploads folder holds originals so re-chunking stays
possible, and everything else lives in the chunk payloads.

Ingestion is synchronous, so there is no job model and no status machine -- an
ingestion either returns a :class:`Document` or raises. What survives from the
larger design is :attr:`Document.ingest_version`, because re-indexing still
needs to write the new chunks before deleting the old ones (ADR-014).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rag.domain.models.metadata import DocumentType

__all__ = ["Document"]


@dataclass(frozen=True, slots=True)
class Document:
    """A document that has been ingested and indexed.

    Attributes:
        document_id: Stable identifier, derived from the content hash so that
            re-uploading the same file updates rather than duplicates.
        filename: Original uploaded filename.
        document_type: Detected format.
        content_hash: Hash of the original bytes. An unchanged hash means
            ingestion can be skipped entirely.
        size_bytes: Size of the original file.
        source_path: Where the original is kept, so the index can be rebuilt
            after a chunking or embedding-model change.
        ingest_version: Monotonic version, incremented on each re-ingestion.
            Search filters on the current version, which is what makes a
            re-index invisible to readers.
        chunk_count: How many chunks this version produced.
        ingested_at: When this version was indexed.
    """

    document_id: str
    filename: str
    document_type: DocumentType
    content_hash: str
    size_bytes: int
    source_path: str
    ingest_version: int
    chunk_count: int
    ingested_at: datetime

    def __post_init__(self) -> None:
        """Validate structural invariants."""
        if not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        if not self.content_hash.strip():
            raise ValueError("content_hash must be a non-empty string")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")
        if self.ingest_version < 1:
            raise ValueError("ingest_version must be >= 1")
        if self.chunk_count < 0:
            raise ValueError("chunk_count must be >= 0")
