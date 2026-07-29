"""Hybrid retrieval against a real Qdrant.

The in-memory double proves the *composition* works. It cannot prove that
Qdrant stores sparse vectors the way the encoder produces them, that server-side
IDF is applied, or that filters reach the sparse query path -- all of which are
things a hand-written double gets right by construction and a real database may
not.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from rag.domain.errors import CollectionMismatchError
from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    CollectionSpec,
    DistanceMetric,
    DocumentType,
    FieldFilter,
    FilterOperator,
    MetadataField,
    RetrievalRequest,
)
from rag.infrastructure.retrieval import (
    Bm25SparseEncoder,
    DenseRetriever,
    HybridRetriever,
    SparseRetriever,
)
from rag.infrastructure.vector_store import QdrantVectorStore
from tests.fakes import FakeEmbedder

pytestmark = [pytest.mark.integration]

QDRANT_URL = "http://localhost:6333"
_NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)

#: A miniature corpus with a deliberate split: one chunk answers a conceptual
#: question, one contains an exact identifier no embedding would preserve.
CORPUS: tuple[tuple[str, str, int], ...] = (
    ("leave", "Employees are entitled to paid time away from work each year.", 1),
    ("error", "Failure code ERR_CONN_4021 indicates a dropped gateway connection.", 2),
    ("lunch", "The cafeteria serves lunch between noon and two every weekday.", 3),
    ("expenses", "Expense claims must be submitted within thirty days of travel.", 4),
)


def _spec(name: str) -> CollectionSpec:
    return CollectionSpec(
        name=name,
        dense_dimension=8,
        distance=DistanceMetric.COSINE,
        embedding_model_id="fake-embedder",
        supports_sparse=True,
    )


def _chunk(chunk_id: str, text: str, page: int) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            document_id="doc-1",
            chunk_id=chunk_id,
            ingest_version=1,
            filename="handbook.pdf",
            document_type=DocumentType.PDF,
            chunk_index=page - 1,
            char_start=0,
            char_end=len(text),
            token_count=12,
            chunking_strategy="recursive",
            embedding_model_id="fake-embedder",
            ingested_at=_NOW,
            page_number=page,
        ),
    )


@pytest.fixture(scope="session")
def qdrant_available() -> bool:
    """Probe Qdrant once per session."""
    import asyncio

    async def _probe() -> bool:
        client = AsyncQdrantClient(url=QDRANT_URL, timeout=2, check_compatibility=False)
        try:
            await client.get_collections()
            return True
        except Exception:
            return False
        finally:
            await client.close()

    return asyncio.run(_probe())


@pytest.fixture
async def retrievers(qdrant_available):
    """A populated Qdrant collection with dense, sparse and hybrid retrievers."""
    if not qdrant_available:
        pytest.skip("Qdrant is not running; start it with `docker compose up -d`")

    client = AsyncQdrantClient(url=QDRANT_URL, timeout=10, check_compatibility=False)
    name = f"hybrid_{uuid.uuid4().hex[:12]}"
    try:
        store = QdrantVectorStore(client, name)
        await store.ensure_collection(_spec(name))

        embedder = FakeEmbedder(dimension=8)
        encoder = Bm25SparseEncoder()
        chunks = [_chunk(cid, text, page) for cid, text, page in CORPUS]
        texts = [c.text for c in chunks]
        await store.upsert(
            chunks,
            await embedder.embed_documents(texts),
            await encoder.encode_documents(texts),
        )

        dense = DenseRetriever(embedder, store)
        sparse = SparseRetriever(encoder, store)
        yield store, dense, sparse, HybridRetriever(dense, sparse)
    finally:
        try:
            await client.delete_collection(name)
        finally:
            await client.close()


class TestSparseSearchAgainstQdrant:
    async def test_an_exact_identifier_is_found(self, retrievers):
        _, _, sparse, _ = retrievers

        results = await sparse.retrieve(RetrievalRequest(query_text="ERR_CONN_4021", top_k=5))

        assert [r.chunk_id for r in results] == ["error"]

    async def test_a_content_word_finds_its_chunk(self, retrievers):
        _, _, sparse, _ = retrievers

        results = await sparse.retrieve(RetrievalRequest(query_text="cafeteria", top_k=5))

        assert results[0].chunk_id == "lunch"

    async def test_a_query_sharing_no_vocabulary_returns_nothing(self, retrievers):
        _, _, sparse, _ = retrievers

        results = await sparse.retrieve(RetrievalRequest(query_text="xylophone quokka", top_k=5))

        assert results == ()

    async def test_an_empty_sparse_vector_never_reaches_the_server(self, retrievers):
        # A stop-word-only query encodes to nothing; asking Qdrant would be a
        # round trip to learn it matches nothing.
        _, _, sparse, _ = retrievers

        assert await sparse.retrieve(RetrievalRequest(query_text="the and of", top_k=5)) == ()

    async def test_metadata_filters_apply_to_the_sparse_path(self, retrievers):
        # Preserving the filter across both retrieval paths is the requirement;
        # dropping it on one would return chunks the user excluded.
        _, _, sparse, _ = retrievers

        results = await sparse.retrieve(
            RetrievalRequest(
                query_text="ERR_CONN_4021",
                top_k=5,
                filters=FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.EQ, 99),
            )
        )

        assert results == ()

    async def test_a_matching_filter_still_returns_the_chunk(self, retrievers):
        _, _, sparse, _ = retrievers

        results = await sparse.retrieve(
            RetrievalRequest(
                query_text="ERR_CONN_4021",
                top_k=5,
                filters=FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.EQ, 2),
            )
        )

        assert [r.chunk_id for r in results] == ["error"]


class TestServerSideIdf:
    async def test_the_collection_is_created_with_the_idf_modifier(self, retrievers):
        # Without it Qdrant scores raw term frequency and common words dominate.
        store, _, _, _ = retrievers

        info = await store._client.get_collection((await store.collection_info()).name)

        sparse_params = (info.config.params.sparse_vectors or {})["sparse"]
        assert sparse_params.modifier == qmodels.Modifier.IDF

    async def test_a_rare_term_outranks_a_common_one(self, retrievers):
        # The observable consequence of IDF: "connection" appears once in the
        # corpus, "submitted" once, but a query mixing a rare identifier with
        # common words should still surface the identifier's chunk.
        _, _, sparse, _ = retrievers

        results = await sparse.retrieve(
            RetrievalRequest(query_text="employees ERR_CONN_4021 lunch", top_k=4)
        )

        assert results[0].chunk_id == "error"


class TestHybridAgainstQdrant:
    async def test_it_fuses_results_from_both_paths(self, retrievers):
        _, _, _, hybrid = retrievers

        results = await hybrid.retrieve(
            RetrievalRequest(query_text="ERR_CONN_4021 cafeteria", top_k=4)
        )

        assert {r.chunk_id for r in results} >= {"error", "lunch"}

    async def test_filters_apply_across_both_paths(self, retrievers):
        _, _, _, hybrid = retrievers

        results = await hybrid.retrieve(
            RetrievalRequest(
                query_text="ERR_CONN_4021 cafeteria employees",
                top_k=5,
                filters=FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.LTE, 2),
            )
        )

        assert results
        assert all(r.chunk.metadata.page_number <= 2 for r in results)

    async def test_it_respects_top_k(self, retrievers):
        _, _, _, hybrid = retrievers

        results = await hybrid.retrieve(
            RetrievalRequest(query_text="employees cafeteria expense connection", top_k=2)
        )

        assert len(results) == 2


class TestHybridVersusDenseOnAQuery:
    async def test_dense_alone_misses_the_exact_identifier(self, retrievers):
        # The fake embedder hashes text, so it has no semantic behaviour. What
        # this demonstrates is structural and shared with real models: a rare
        # identifier is not recoverable from a summary vector.
        _, dense, _, _ = retrievers

        results = await dense.retrieve(RetrievalRequest(query_text="ERR_CONN_4021", top_k=1))

        assert [r.chunk_id for r in results] != ["error"]

    async def test_hybrid_recovers_it(self, retrievers):
        _, _, _, hybrid = retrievers

        results = await hybrid.retrieve(RetrievalRequest(query_text="ERR_CONN_4021", top_k=3))

        assert results[0].chunk_id == "error"


class TestIdfGuard:
    async def test_a_collection_without_idf_is_refused(self, qdrant_available):
        # Simulates a collection created before IDF was configured: sparse
        # scoring would silently ignore term rarity.
        if not qdrant_available:
            pytest.skip("Qdrant is not running")

        client = AsyncQdrantClient(url=QDRANT_URL, timeout=10, check_compatibility=False)
        name = f"noidf_{uuid.uuid4().hex[:12]}"
        try:
            await client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": qmodels.VectorParams(size=8, distance=qmodels.Distance.COSINE)
                },
                sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
            )
            await client.upsert(
                collection_name=name,
                points=[
                    qmodels.PointStruct(
                        id="00000000-0000-0000-0000-000000000000",
                        vector={"dense": [0.0] * 8},
                        payload={"embedding_model_id": "fake-embedder", "_marker": True},
                    )
                ],
            )

            store = QdrantVectorStore(client, name)
            with pytest.raises(CollectionMismatchError, match="IDF"):
                await store.ensure_collection(_spec(name))
        finally:
            try:
                await client.delete_collection(name)
            finally:
                await client.close()
