"""Keeping original uploads and a record of what has been ingested.

There is no database. Qdrant holds the chunks; this holds the two things Qdrant
should not be asked for: the original bytes, and a small catalogue of what has
been ingested.

Retaining originals is what makes a chunking change or an embedding-model change
possible without asking the user to re-upload everything (ADR-023). The
catalogue is what lets the UI list documents and lets ingestion detect that a
file has not changed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.domain.models import Document, DocumentType

__all__ = ["DocumentCatalog", "UploadStore"]


class UploadStore:
    """Stores original uploaded files on disk."""

    def __init__(self, root: Path | str) -> None:
        """Initialise the store.

        Args:
            root: Directory to write uploads into. Created if absent.
        """
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """The directory uploads are written to."""
        return self._root

    def save(self, document_id: str, filename: str, content: bytes) -> Path:
        """Write an upload to disk.

        The stored name is derived from the document id rather than the user's
        filename, so a name containing path separators cannot escape the
        directory, and two documents with the same name cannot collide.

        Args:
            document_id: Identifier of the document.
            filename: Original filename, used only for its extension.
            content: The bytes to store.

        Returns:
            Path to the stored file.
        """
        suffix = Path(filename).suffix
        path = self._root / f"{document_id}{suffix}"
        path.write_bytes(content)
        return path


class DocumentCatalog:
    """A JSON record of the documents that have been ingested.

    Small enough to rewrite whole on every change, which avoids partial-write
    corruption entirely. If this file ever grows past the point where that is
    acceptable, the project has outgrown "no database" and should say so rather
    than getting clever here.
    """

    def __init__(self, path: Path | str) -> None:
        """Initialise the catalogue.

        Args:
            path: JSON file to persist to. Created on first write.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, document_id: str) -> Document | None:
        """Look up a document.

        Args:
            document_id: The document to find.

        Returns:
            The document, or ``None`` if it has never been ingested.
        """
        return self._load().get(document_id)

    def list_documents(self) -> tuple[Document, ...]:
        """Every ingested document, most recently ingested first."""
        documents = list(self._load().values())
        documents.sort(key=lambda d: d.ingested_at, reverse=True)
        return tuple(documents)

    def put(self, document: Document) -> None:
        """Record a document, replacing any previous entry.

        Args:
            document: The document to record.
        """
        documents = self._load()
        documents[document.document_id] = document
        self._save(documents.values())

    def remove(self, document_id: str) -> None:
        """Forget a document.

        Args:
            document_id: The document to remove.
        """
        documents = self._load()
        if documents.pop(document_id, None) is not None:
            self._save(documents.values())

    # ----------------------------------------------------------- persistence #
    def _load(self) -> dict[str, Document]:
        """Read the catalogue from disk."""
        if not self._path.exists():
            return {}
        raw: list[dict[str, Any]] = json.loads(self._path.read_text(encoding="utf-8"))
        return {str(entry["document_id"]): self._from_dict(entry) for entry in raw}

    def _save(self, documents: Iterable[Document]) -> None:
        """Write the catalogue to disk."""
        payload = [self._to_dict(document) for document in documents]
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _to_dict(document: Document) -> dict[str, Any]:
        """Render a document as JSON-safe values."""
        return {
            "document_id": document.document_id,
            "filename": document.filename,
            "document_type": document.document_type.value,
            "content_hash": document.content_hash,
            "size_bytes": document.size_bytes,
            "source_path": document.source_path,
            "ingest_version": document.ingest_version,
            "chunk_count": document.chunk_count,
            "ingested_at": document.ingested_at.isoformat(),
        }

    @staticmethod
    def _from_dict(entry: dict[str, Any]) -> Document:
        """Rebuild a document from stored values."""
        ingested_at = datetime.fromisoformat(str(entry["ingested_at"]))
        if ingested_at.tzinfo is None:
            # Timestamps written before the UTC rule was enforced. Assuming UTC
            # is right for every value this project has ever written.
            ingested_at = ingested_at.replace(tzinfo=UTC)
        return Document(
            document_id=str(entry["document_id"]),
            filename=str(entry["filename"]),
            document_type=DocumentType(entry["document_type"]),
            content_hash=str(entry["content_hash"]),
            size_bytes=int(entry["size_bytes"]),
            source_path=str(entry["source_path"]),
            ingest_version=int(entry["ingest_version"]),
            chunk_count=int(entry["chunk_count"]),
            ingested_at=ingested_at,
        )
