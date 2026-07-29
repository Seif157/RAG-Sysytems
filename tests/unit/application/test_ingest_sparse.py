"""Ingestion writes sparse vectors when hybrid retrieval is in use.

Sparse retrieval is only as good as what was indexed. If ingestion writes dense
vectors alone, the lexical half of hybrid search silently finds nothing -- and
"finds nothing" is indistinguishable from "there was nothing to find".
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.application.use_cases import IngestDocumentUseCase
from rag.domain.models import RetrievalRequest
from rag.infrastructure.chunking import RecursiveChunker
from rag.infrastructure.documents import DocumentLoader, ParserRegistry, TxtParser
from rag.infrastructure.retrieval import Bm25SparseEncoder, SparseRetriever
from rag.infrastructure.storage import DocumentCatalog, UploadStore
from tests.fakes import FakeEmbedder, InMemoryVectorStore

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)

_SAMPLE = "Failure code ERR_CONN_4021 indicates a dropped connection to the gateway.\n"


def _use_case(
    tmp_path: Path,
    *,
    sparse: Bm25SparseEncoder | None,
) -> tuple[IngestDocumentUseCase, InMemoryVectorStore]:
    store = InMemoryVectorStore()
    use_case = IngestDocumentUseCase(
        loader=DocumentLoader(max_upload_bytes=1_000_000),
        parsers=ParserRegistry((TxtParser(),)),
        chunker=RecursiveChunker(chunk_size=200, chunk_overlap=20),
        embedder=FakeEmbedder(),
        store=store,
        uploads=UploadStore(tmp_path / "uploads"),
        catalog=DocumentCatalog(tmp_path / "uploads" / "catalog.json"),
        clock=lambda: _NOW,
        sparse_encoder=sparse,
    )
    return use_case, store


class TestSparseVectorsAreWritten:
    async def test_a_sparse_vector_is_stored_for_every_chunk(self, tmp_path):
        use_case, store = _use_case(tmp_path, sparse=Bm25SparseEncoder())

        document = await use_case.execute("notes.txt", _SAMPLE.encode())

        assert store.sparse_count == document.chunk_count

    async def test_the_stored_sparse_vectors_are_searchable(self, tmp_path):
        use_case, store = _use_case(tmp_path, sparse=Bm25SparseEncoder())
        await use_case.execute("notes.txt", _SAMPLE.encode())

        retriever = SparseRetriever(Bm25SparseEncoder(), store)
        results = await retriever.retrieve(RetrievalRequest(query_text="ERR_CONN_4021", top_k=5))

        assert results
        assert "ERR_CONN_4021" in results[0].text


class TestDenseOnlyIngestion:
    async def test_no_sparse_vectors_are_written_without_an_encoder(self, tmp_path):
        # ENABLE_HYBRID=false: paying to encode vectors nothing will query is
        # pure waste, so the encoder is simply absent rather than disabled.
        use_case, store = _use_case(tmp_path, sparse=None)

        await use_case.execute("notes.txt", _SAMPLE.encode())

        assert store.sparse_count == 0

    async def test_dense_indexing_is_unaffected(self, tmp_path):
        use_case, store = _use_case(tmp_path, sparse=None)

        document = await use_case.execute("notes.txt", _SAMPLE.encode())

        assert store.chunk_count == document.chunk_count

    async def test_sparse_search_finds_nothing_when_nothing_was_indexed(self, tmp_path):
        use_case, store = _use_case(tmp_path, sparse=None)
        await use_case.execute("notes.txt", _SAMPLE.encode())

        retriever = SparseRetriever(Bm25SparseEncoder(), store)
        results = await retriever.retrieve(RetrievalRequest(query_text="ERR_CONN_4021", top_k=5))

        assert results == ()


class TestReIngestion:
    async def test_superseded_sparse_vectors_are_removed_with_their_chunks(self, tmp_path):
        # A stale sparse vector would keep matching text that no longer exists.
        use_case, store = _use_case(tmp_path, sparse=Bm25SparseEncoder())
        await use_case.execute("notes.txt", b"Code ERR_OLD_1111 was logged.")

        await use_case.execute("notes.txt", b"Code ERR_NEW_2222 was logged.")

        retriever = SparseRetriever(Bm25SparseEncoder(), store)
        stale = await retriever.retrieve(RetrievalRequest(query_text="ERR_OLD_1111", top_k=5))
        fresh = await retriever.retrieve(RetrievalRequest(query_text="ERR_NEW_2222", top_k=5))

        assert stale == ()
        assert fresh != ()
