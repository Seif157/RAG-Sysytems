"""Behaviour of sparse and hybrid retrieval.

The hybrid retriever is a composite of retrievers, which is why switching between
dense-only and hybrid is a wiring decision rather than a branch in the query
flow. These tests cover the composition: that both halves run, that both receive
the filter, that fusion is applied, and that one half failing does not take the
query down with it.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from rag.domain.errors import RetrievalError
from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    DocumentType,
    FieldFilter,
    FilterOperator,
    MetadataField,
    RetrievalRequest,
    ScoredChunk,
    ScoreSource,
)
from rag.domain.ports import Retriever
from rag.infrastructure.retrieval import Bm25SparseEncoder, HybridRetriever, SparseRetriever
from tests.fakes import FakeEmbedder, InMemoryVectorStore

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _chunk(chunk_id: str, text: str, *, page: int | None = None) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            document_id="doc-1",
            chunk_id=chunk_id,
            ingest_version=1,
            filename="report.pdf",
            document_type=DocumentType.PDF,
            chunk_index=0,
            char_start=0,
            char_end=len(text),
            token_count=8,
            chunking_strategy="recursive",
            embedding_model_id="fake-embedder",
            ingested_at=_NOW,
            page_number=page,
        ),
    )


def _scored(chunk_id: str, score: float, source: ScoreSource) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(chunk_id, f"text {chunk_id}"), score=score, source=source)


class StubRetriever(Retriever):
    """Returns fixed results and records the requests it received."""

    def __init__(self, results: tuple[ScoredChunk, ...] = (), error: Exception | None = None):
        self.results = results
        self.error = error
        self.requests: list[RetrievalRequest] = []

    async def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        """Record and return."""
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.results


class SlowRetriever(Retriever):
    """Sleeps before answering, so concurrency is observable."""

    def __init__(self, results: tuple[ScoredChunk, ...], delay: float = 0.05):
        self.results = results
        self.delay = delay

    async def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        """Sleep, then return."""
        await asyncio.sleep(self.delay)
        return self.results


# --------------------------------------------------------------------------- #
# Sparse retrieval
# --------------------------------------------------------------------------- #
async def _populated_store() -> InMemoryVectorStore:
    """A store holding three chunks with both dense and sparse vectors."""
    store = InMemoryVectorStore()
    embedder = FakeEmbedder()
    encoder = Bm25SparseEncoder()

    chunks = [
        _chunk("semantic", "Employees are entitled to paid time away from work.", page=1),
        _chunk("exact", "Failure code ERR_CONN_4021 indicates a dropped connection.", page=2),
        _chunk("unrelated", "The cafeteria serves lunch between noon and two.", page=3),
    ]
    texts = [c.text for c in chunks]
    await store.upsert(
        chunks,
        await embedder.embed_documents(texts),
        await encoder.encode_documents(texts),
    )
    return store


class TestSparseRetriever:
    async def test_it_finds_a_chunk_by_an_exact_token(self):
        # The case sparse retrieval exists for.
        store = await _populated_store()
        retriever = SparseRetriever(Bm25SparseEncoder(), store)

        results = await retriever.retrieve(RetrievalRequest(query_text="ERR_CONN_4021", top_k=5))

        assert [r.chunk_id for r in results] == ["exact"]

    async def test_results_are_marked_as_sparse(self):
        store = await _populated_store()
        retriever = SparseRetriever(Bm25SparseEncoder(), store)

        results = await retriever.retrieve(RetrievalRequest(query_text="cafeteria", top_k=5))

        assert results[0].source is ScoreSource.SPARSE

    async def test_a_query_sharing_no_vocabulary_returns_nothing(self):
        store = await _populated_store()
        retriever = SparseRetriever(Bm25SparseEncoder(), store)

        results = await retriever.retrieve(RetrievalRequest(query_text="xylophone quokka", top_k=5))

        assert results == ()

    async def test_a_query_of_only_stop_words_returns_nothing(self):
        # An empty sparse vector must match nothing, not everything.
        store = await _populated_store()
        retriever = SparseRetriever(Bm25SparseEncoder(), store)

        results = await retriever.retrieve(RetrievalRequest(query_text="the and of", top_k=5))

        assert results == ()

    async def test_it_honours_a_metadata_filter(self):
        store = await _populated_store()
        retriever = SparseRetriever(Bm25SparseEncoder(), store)

        results = await retriever.retrieve(
            RetrievalRequest(
                query_text="ERR_CONN_4021",
                top_k=5,
                filters=FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.EQ, 99),
            )
        )

        assert results == ()

    async def test_it_respects_top_k(self):
        store = await _populated_store()
        retriever = SparseRetriever(Bm25SparseEncoder(), store)

        results = await retriever.retrieve(
            RetrievalRequest(query_text="connection lunch employees", top_k=1)
        )

        assert len(results) <= 1

    async def test_a_store_failure_becomes_a_retrieval_error(self):
        from rag.domain.errors import VectorStoreError

        class BrokenStore(InMemoryVectorStore):
            async def search_sparse(self, vector, top_k, filters=None):
                raise VectorStoreError("connection refused")

        retriever = SparseRetriever(Bm25SparseEncoder(), BrokenStore())

        with pytest.raises(RetrievalError):
            await retriever.retrieve(RetrievalRequest(query_text="anything", top_k=5))


# --------------------------------------------------------------------------- #
# Hybrid retrieval
# --------------------------------------------------------------------------- #
class TestHybridComposition:
    async def test_it_queries_both_halves(self):
        dense = StubRetriever((_scored("d", 0.9, ScoreSource.DENSE),))
        sparse = StubRetriever((_scored("s", 8.0, ScoreSource.SPARSE),))
        hybrid = HybridRetriever(dense, sparse)

        await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5))

        assert len(dense.requests) == 1
        assert len(sparse.requests) == 1

    async def test_both_halves_receive_the_same_filter(self):
        # Filtering must be preserved across both paths, or hybrid quietly
        # returns results the user excluded.
        clause = FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.EQ, "PDF")
        dense = StubRetriever()
        sparse = StubRetriever()
        hybrid = HybridRetriever(dense, sparse)

        await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5, filters=clause))

        assert dense.requests[0].filters is clause
        assert sparse.requests[0].filters is clause

    async def test_both_halves_receive_the_same_query_text(self):
        dense = StubRetriever()
        sparse = StubRetriever()
        hybrid = HybridRetriever(dense, sparse)

        await hybrid.retrieve(RetrievalRequest(query_text="exact phrase", top_k=5))

        assert dense.requests[0].query_text == "exact phrase"
        assert sparse.requests[0].query_text == "exact phrase"

    async def test_results_from_both_halves_are_merged(self):
        dense = StubRetriever((_scored("d", 0.9, ScoreSource.DENSE),))
        sparse = StubRetriever((_scored("s", 8.0, ScoreSource.SPARSE),))
        hybrid = HybridRetriever(dense, sparse)

        results = await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5))

        assert {r.chunk_id for r in results} == {"d", "s"}

    async def test_merged_results_are_marked_as_fused(self):
        dense = StubRetriever((_scored("d", 0.9, ScoreSource.DENSE),))
        sparse = StubRetriever((_scored("s", 8.0, ScoreSource.SPARSE),))
        hybrid = HybridRetriever(dense, sparse)

        results = await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5))

        assert all(r.source is ScoreSource.FUSED for r in results)

    async def test_a_chunk_found_by_both_ranks_first(self):
        dense = StubRetriever(
            (_scored("dense_only", 0.9, ScoreSource.DENSE), _scored("both", 0.5, ScoreSource.DENSE))
        )
        sparse = StubRetriever(
            (
                _scored("both", 9.0, ScoreSource.SPARSE),
                _scored("sparse_only", 3.0, ScoreSource.SPARSE),
            )
        )
        hybrid = HybridRetriever(dense, sparse)

        results = await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5))

        assert results[0].chunk_id == "both"

    async def test_it_returns_no_more_than_top_k(self):
        dense = StubRetriever(tuple(_scored(f"d{i}", 0.9, ScoreSource.DENSE) for i in range(10)))
        sparse = StubRetriever(tuple(_scored(f"s{i}", 5.0, ScoreSource.SPARSE) for i in range(10)))
        hybrid = HybridRetriever(dense, sparse)

        results = await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=6))

        assert len(results) == 6

    async def test_the_halves_run_concurrently(self):
        # Serial fan-out would double query latency for no benefit; this is why
        # the retrieval ports are async at all.
        results = (_scored("a", 1.0, ScoreSource.DENSE),)
        hybrid = HybridRetriever(SlowRetriever(results), SlowRetriever(results))

        started = asyncio.get_running_loop().time()
        await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5))
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < 0.09


class TestHybridWeighting:
    async def test_a_sparse_weight_of_zero_yields_dense_results_only(self):
        dense = StubRetriever((_scored("d", 0.9, ScoreSource.DENSE),))
        sparse = StubRetriever((_scored("s", 8.0, ScoreSource.SPARSE),))
        hybrid = HybridRetriever(dense, sparse, sparse_weight=0.0)

        results = await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5))

        assert [r.chunk_id for r in results] == ["d"]

    async def test_a_sparse_weight_of_one_yields_sparse_results_only(self):
        dense = StubRetriever((_scored("d", 0.9, ScoreSource.DENSE),))
        sparse = StubRetriever((_scored("s", 8.0, ScoreSource.SPARSE),))
        hybrid = HybridRetriever(dense, sparse, sparse_weight=1.0)

        results = await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5))

        assert [r.chunk_id for r in results] == ["s"]

    async def test_a_weight_outside_the_unit_interval_is_rejected(self):
        with pytest.raises(ValueError, match="sparse_weight"):
            HybridRetriever(StubRetriever(), StubRetriever(), sparse_weight=1.5)


class TestHybridDegradation:
    async def test_a_sparse_failure_leaves_dense_results_intact(self):
        # Lexical matching is an enhancement. Losing it should cost quality, not
        # the answer -- the dense half can still answer most questions.
        dense = StubRetriever((_scored("d", 0.9, ScoreSource.DENSE),))
        sparse = StubRetriever(error=RetrievalError("sparse index unavailable"))
        hybrid = HybridRetriever(dense, sparse)

        results = await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5))

        assert [r.chunk_id for r in results] == ["d"]

    async def test_a_dense_failure_propagates(self):
        # Dense retrieval is the load-bearing half. Silently answering from
        # lexical matches alone would be a large, invisible quality drop.
        dense = StubRetriever(error=RetrievalError("vector search failed"))
        sparse = StubRetriever((_scored("s", 8.0, ScoreSource.SPARSE),))
        hybrid = HybridRetriever(dense, sparse)

        with pytest.raises(RetrievalError):
            await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5))

    async def test_both_halves_empty_returns_nothing(self):
        hybrid = HybridRetriever(StubRetriever(), StubRetriever())

        assert await hybrid.retrieve(RetrievalRequest(query_text="q", top_k=5)) == ()


# --------------------------------------------------------------------------- #
# The comparison that justifies the feature
# --------------------------------------------------------------------------- #
class TestHybridVersusDenseOnly:
    """Representative queries where one strategy beats the other.

    The fake embedder hashes text, so it has no semantic behaviour -- these
    compare *retrieval mechanics*, not embedding quality. The exact-token case is
    the honest one: it fails under dense-only for a structural reason that a real
    embedding model shares, namely that a rare identifier is not recoverable from
    a summary vector.
    """

    async def _retrievers(self) -> tuple[Retriever, Retriever]:
        from rag.infrastructure.retrieval import DenseRetriever

        store = await _populated_store()
        dense = DenseRetriever(FakeEmbedder(), store)
        sparse = SparseRetriever(Bm25SparseEncoder(), store)
        return dense, HybridRetriever(dense, sparse)

    async def test_dense_only_misses_an_exact_identifier(self):
        dense, _ = await self._retrievers()

        results = await dense.retrieve(RetrievalRequest(query_text="ERR_CONN_4021", top_k=1))

        assert [r.chunk_id for r in results] != ["exact"]

    async def test_hybrid_finds_the_exact_identifier(self):
        _, hybrid = await self._retrievers()

        results = await hybrid.retrieve(RetrievalRequest(query_text="ERR_CONN_4021", top_k=3))

        assert results[0].chunk_id == "exact"

    async def test_hybrid_never_returns_fewer_candidates_than_dense_alone(self):
        # Fusion adds a second source; it cannot subtract from the first.
        dense, hybrid = await self._retrievers()
        request = RetrievalRequest(query_text="connection dropped", top_k=5)

        dense_results = await dense.retrieve(request)
        hybrid_results = await hybrid.retrieve(request)

        assert len(hybrid_results) >= len(dense_results)

    async def test_hybrid_preserves_filtering_that_dense_alone_would(self):
        dense, hybrid = await self._retrievers()
        clause = FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.EQ, 2)
        request = RetrievalRequest(query_text="ERR_CONN_4021", top_k=5, filters=clause)

        assert all(r.chunk.metadata.page_number == 2 for r in await dense.retrieve(request))
        assert all(r.chunk.metadata.page_number == 2 for r in await hybrid.retrieve(request))
