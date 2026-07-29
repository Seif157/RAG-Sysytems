"""Behaviour of the ingested-document record.

Ingestion is synchronous, so there is no job model and no status machine: an
ingestion either returns a :class:`Document` or raises. What survives is
``ingest_version``, because re-indexing still writes new chunks before deleting
old ones (ADR-014).
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from rag.domain.models import Document, DocumentType

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _document(**overrides: object) -> Document:
    defaults: dict[str, object] = {
        "document_id": "doc-1",
        "filename": "report.pdf",
        "document_type": DocumentType.PDF,
        "content_hash": "a" * 64,
        "size_bytes": 2048,
        "source_path": "uploads/report.pdf",
        "ingest_version": 1,
        "chunk_count": 128,
        "ingested_at": _NOW,
    }
    return Document(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestDocument:
    def test_carries_its_identity_and_provenance(self):
        document = _document()

        assert document.document_id == "doc-1"
        assert document.document_type is DocumentType.PDF
        assert document.chunk_count == 128

    def test_it_remembers_where_the_original_is_kept(self):
        # Retaining the original is what makes a chunking or embedding-model
        # change possible without asking for the file again.
        assert _document().source_path == "uploads/report.pdf"

    def test_versions_start_at_one(self):
        with pytest.raises(ValueError, match="ingest_version"):
            _document(ingest_version=0)

    def test_empty_filename_is_rejected(self):
        with pytest.raises(ValueError, match="filename"):
            _document(filename="  ")

    def test_empty_document_id_is_rejected(self):
        with pytest.raises(ValueError, match="document_id"):
            _document(document_id="  ")

    def test_content_hash_is_required(self):
        # An absent hash means a re-upload of unchanged content cannot be
        # detected, and every re-ingestion pays for embeddings again.
        with pytest.raises(ValueError, match="content_hash"):
            _document(content_hash="   ")

    def test_negative_size_is_rejected(self):
        with pytest.raises(ValueError, match="size_bytes"):
            _document(size_bytes=-1)

    def test_negative_chunk_count_is_rejected(self):
        with pytest.raises(ValueError, match="chunk_count"):
            _document(chunk_count=-1)

    def test_a_document_that_produced_no_chunks_is_representable(self):
        # An empty text file, or a scanned PDF with no extractable text. OCR is
        # out of scope, so zero chunks is a real outcome rather than a bug.
        assert _document(chunk_count=0).chunk_count == 0

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _document().chunk_count = 5  # type: ignore[misc]
