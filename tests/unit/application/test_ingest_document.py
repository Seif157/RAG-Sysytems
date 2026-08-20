"""Behaviour of the ingestion use case.

The prototype's central defect was that the uploaded file was never read -- it
loaded a hardcoded folder regardless of what you gave it. The first test here
exists so that cannot happen again.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.application.use_cases import IngestDocumentUseCase
from rag.domain.errors import DocumentTooLargeError, UnsupportedFormatError
from rag.domain.models import DocumentAccessPolicy, DocumentAccessScope, DocumentType
from rag.infrastructure.chunking import RecursiveChunker
from rag.infrastructure.documents import DocumentLoader, ParserRegistry, TxtParser
from rag.infrastructure.storage import DocumentCatalog, UploadStore
from tests.fakes import FakeEmbedder, InMemoryVectorStore

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _use_case(
    tmp_path: Path,
    *,
    max_bytes: int = 1_000_000,
    chunk_size: int = 200,
    overlap: int = 20,
) -> tuple[IngestDocumentUseCase, InMemoryVectorStore, FakeEmbedder]:
    store = InMemoryVectorStore()
    embedder = FakeEmbedder()
    use_case = IngestDocumentUseCase(
        loader=DocumentLoader(max_upload_bytes=max_bytes),
        parsers=ParserRegistry((TxtParser(),)),
        chunker=RecursiveChunker(chunk_size=chunk_size, chunk_overlap=overlap),
        embedder=embedder,
        store=store,
        uploads=UploadStore(tmp_path / "uploads"),
        catalog=DocumentCatalog(tmp_path / "uploads" / "catalog.json"),
        clock=lambda: _NOW,
    )
    return use_case, store, embedder


_SAMPLE = (
    "Machine learning is a field of study in artificial intelligence.\n\n"
    "It builds models from sample data, known as training data.\n\n"
    "Revenue grew by twelve percent across every region last year.\n"
)


class TestTheUploadIsActuallyUsed:
    async def test_the_uploaded_bytes_are_what_get_indexed(self, tmp_path):
        # The prototype ignored its argument and read a hardcoded folder, so
        # every answer was about the wrong document.
        use_case, store, _ = _use_case(tmp_path)

        await use_case.execute("notes.txt", b"The sky is green on Tuesdays.")

        stored = [store._chunks[cid].text for cid in store.chunk_ids()]
        assert any("sky is green" in text for text in stored)

    async def test_two_different_uploads_are_two_different_documents(self, tmp_path):
        use_case, store, _ = _use_case(tmp_path)

        first = await use_case.execute("a.txt", b"Alpha content here.")
        second = await use_case.execute("b.txt", b"Beta content here.")

        assert first.document_id != second.document_id
        assert store.chunk_count >= 2

    async def test_access_policy_is_stamped_on_every_chunk(self, tmp_path):
        use_case, store, _ = _use_case(tmp_path)
        policy = DocumentAccessPolicy(
            DocumentAccessScope.DEPARTMENT,
            scope_key="department-7",
            required_permission="hr.documents.confidential.read",
        )

        await use_case.execute("restricted.txt", b"Department policy.", access_policy=policy)

        metadata = next(iter(store._chunks.values())).metadata
        assert metadata.access_scope is DocumentAccessScope.DEPARTMENT
        assert metadata.access_scope_key == "department-7"
        assert metadata.required_permission == "hr.documents.confidential.read"


class TestIngestionResult:
    async def test_it_reports_what_was_indexed(self, tmp_path):
        use_case, _, _ = _use_case(tmp_path)

        document = await use_case.execute("notes.txt", _SAMPLE.encode())

        assert document.filename == "notes.txt"
        assert document.document_type is DocumentType.TXT
        assert document.chunk_count > 0

    async def test_the_first_ingestion_is_version_one(self, tmp_path):
        use_case, _, _ = _use_case(tmp_path)

        assert (await use_case.execute("notes.txt", _SAMPLE.encode())).ingest_version == 1

    async def test_the_original_file_is_kept(self, tmp_path):
        # Retaining it is what makes re-chunking possible without re-uploading.
        use_case, _, _ = _use_case(tmp_path)

        document = await use_case.execute("notes.txt", _SAMPLE.encode())

        assert Path(document.source_path).exists()
        assert Path(document.source_path).read_bytes() == _SAMPLE.encode()


class TestChunkMetadata:
    async def test_every_chunk_carries_full_provenance(self, tmp_path):
        use_case, store, _ = _use_case(tmp_path)

        document = await use_case.execute("notes.txt", _SAMPLE.encode())

        chunk = store._chunks[store.chunk_ids()[0]]
        assert chunk.metadata.document_id == document.document_id
        assert chunk.metadata.filename == "notes.txt"
        assert chunk.metadata.document_type is DocumentType.TXT
        assert chunk.metadata.ingest_version == 1
        assert chunk.metadata.chunking_strategy == "recursive"
        assert chunk.metadata.embedding_model_id == "fake-embedder"
        assert chunk.metadata.token_count > 0

    async def test_chunk_indices_are_sequential(self, tmp_path):
        use_case, store, _ = _use_case(tmp_path, chunk_size=60, overlap=0)

        await use_case.execute("notes.txt", _SAMPLE.encode())

        indices = sorted(store._chunks[cid].metadata.chunk_index for cid in store.chunk_ids())
        assert indices == list(range(len(indices)))

    async def test_a_plain_text_file_has_no_page_number(self, tmp_path):
        # TXT genuinely has no pages; None keeps that distinguishable from
        # "the parser failed to find one".
        use_case, store, _ = _use_case(tmp_path)

        await use_case.execute("notes.txt", _SAMPLE.encode())

        chunk = store._chunks[store.chunk_ids()[0]]
        assert chunk.metadata.page_number is None


class TestReIngestion:
    async def test_re_uploading_a_changed_file_bumps_the_version(self, tmp_path):
        use_case, _, _ = _use_case(tmp_path)

        await use_case.execute("notes.txt", b"First version of the notes.")
        second = await use_case.execute("notes.txt", b"Second version of the notes.")

        assert second.ingest_version == 2

    async def test_superseded_chunks_are_removed(self, tmp_path):
        # Otherwise the old text stays searchable and the model cites a version
        # of the document that no longer exists.
        use_case, store, _ = _use_case(tmp_path)

        await use_case.execute("notes.txt", b"The sky is green on Tuesdays.")
        await use_case.execute("notes.txt", b"The sky is blue on Tuesdays.")

        stored = [store._chunks[cid].text for cid in store.chunk_ids()]
        assert not any("green" in text for text in stored)
        assert any("blue" in text for text in stored)

    async def test_re_uploading_identical_content_does_not_re_embed(self, tmp_path):
        # The content hash makes an unchanged re-upload free.
        use_case, _, embedder = _use_case(tmp_path)

        await use_case.execute("notes.txt", _SAMPLE.encode())
        calls_after_first = embedder.embed_calls
        await use_case.execute("notes.txt", _SAMPLE.encode())

        assert embedder.embed_calls == calls_after_first

    async def test_an_unchanged_re_upload_keeps_the_same_version(self, tmp_path):
        use_case, _, _ = _use_case(tmp_path)

        first = await use_case.execute("notes.txt", _SAMPLE.encode())
        second = await use_case.execute("notes.txt", _SAMPLE.encode())

        assert second.ingest_version == first.ingest_version


class TestGuards:
    async def test_an_oversized_upload_is_rejected(self, tmp_path):
        use_case, _, _ = _use_case(tmp_path, max_bytes=10)

        with pytest.raises(DocumentTooLargeError):
            await use_case.execute("notes.txt", b"considerably more than ten bytes")

    async def test_an_unsupported_format_is_rejected(self, tmp_path):
        use_case, _, _ = _use_case(tmp_path)

        with pytest.raises(UnsupportedFormatError):
            await use_case.execute("photo.png", b"\x89PNG\r\n\x1a\n")

    async def test_an_empty_upload_is_rejected(self, tmp_path):
        use_case, _, _ = _use_case(tmp_path)

        with pytest.raises(UnsupportedFormatError):
            await use_case.execute("notes.txt", b"")

    async def test_a_file_with_no_extractable_text_indexes_nothing(self, tmp_path):
        # Whitespace-only file: a real outcome, not a crash.
        use_case, store, _ = _use_case(tmp_path)

        document = await use_case.execute("blank.txt", b"   \n\n   \n")

        assert document.chunk_count == 0
        assert store.chunk_count == 0


class TestCatalog:
    async def test_ingested_documents_can_be_listed(self, tmp_path):
        use_case, _, _ = _use_case(tmp_path)

        await use_case.execute("a.txt", b"Alpha content.")
        await use_case.execute("b.txt", b"Beta content.")

        catalog = DocumentCatalog(tmp_path / "uploads" / "catalog.json")
        assert {d.filename for d in catalog.list_documents()} == {"a.txt", "b.txt"}

    async def test_the_catalog_survives_a_restart(self, tmp_path):
        use_case, _, _ = _use_case(tmp_path)
        await use_case.execute("a.txt", b"Alpha content.")

        reopened = DocumentCatalog(tmp_path / "uploads" / "catalog.json")

        assert len(reopened.list_documents()) == 1
