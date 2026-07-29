"""The Qdrant adapter against a real Qdrant.

Skipped when Qdrant is not reachable, so ``pytest`` still runs anywhere. Start it
with ``docker compose up -d``.

These tests exist because the in-memory double proves the *use cases* work, not
that the adapter does. Filter translation, payload round-tripping and version
deletion are exactly the things a hand-written double gets right by construction
and a real database does not.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient

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
)
from rag.infrastructure.vector_store import QdrantVectorStore

pytestmark = [pytest.mark.integration]

QDRANT_URL = "http://localhost:6333"
_DIMENSION = 4
_NOW = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)


def _spec(
    name: str, *, model: str = "fake-embedder", dimension: int = _DIMENSION
) -> CollectionSpec:
    return CollectionSpec(
        name=name,
        dense_dimension=dimension,
        distance=DistanceMetric.COSINE,
        embedding_model_id=model,
        supports_sparse=False,
    )


def _chunk(
    chunk_id: str,
    *,
    text: str = "Revenue grew by twelve percent.",
    document_id: str = "doc-1",
    version: int = 1,
    page: int | None = 4,
) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            chunk_id=chunk_id,
            ingest_version=version,
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
            section="3.2 Revenue",
        ),
    )


@pytest.fixture(scope="session")
def qdrant_available() -> bool:
    """Probe Qdrant once per session rather than once per test.

    Sixteen three-second connection timeouts is a minute of waiting to learn one
    fact.
    """
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
async def store(qdrant_available):
    """A store backed by a throwaway collection, dropped afterwards."""
    if not qdrant_available:
        pytest.skip("Qdrant is not running; start it with `docker compose up -d`")

    client = AsyncQdrantClient(url=QDRANT_URL, timeout=10, check_compatibility=False)
    name = f"test_{uuid.uuid4().hex[:12]}"
    try:
        adapter = QdrantVectorStore(client, name)
        await adapter.ensure_collection(_spec(name))
        yield adapter
    finally:
        try:
            await client.delete_collection(name)
        finally:
            await client.close()


class TestRoundTrip:
    async def test_a_stored_chunk_comes_back_intact(self, store):
        await store.upsert([_chunk("a")], [(1.0, 0.0, 0.0, 0.0)])

        results = await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=5)

        assert len(results) == 1
        assert results[0].text == "Revenue grew by twelve percent."

    async def test_metadata_survives_the_round_trip(self, store):
        # Payload mapping is where provenance is most likely to be quietly lost,
        # and a citation with the wrong page is worse than no citation.
        await store.upsert([_chunk("a")], [(1.0, 0.0, 0.0, 0.0)])

        metadata = (await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=5))[0].chunk.metadata

        assert metadata.chunk_id == "a"
        assert metadata.document_id == "doc-1"
        assert metadata.filename == "report.pdf"
        assert metadata.document_type is DocumentType.PDF
        assert metadata.page_number == 4
        assert metadata.section == "3.2 Revenue"
        assert metadata.ingested_at == _NOW

    async def test_an_absent_page_number_round_trips_as_none(self, store):
        await store.upsert([_chunk("a", page=None)], [(1.0, 0.0, 0.0, 0.0)])

        result = (await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=5))[0]

        assert result.chunk.metadata.page_number is None

    async def test_upsert_is_idempotent_on_chunk_id(self, store):
        await store.upsert([_chunk("a")], [(1.0, 0.0, 0.0, 0.0)])
        await store.upsert([_chunk("a", text="Updated text.")], [(1.0, 0.0, 0.0, 0.0)])

        results = await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=5)

        assert len(results) == 1
        assert results[0].text == "Updated text."


class TestRanking:
    async def test_the_nearest_vector_ranks_first(self, store):
        await store.upsert(
            [_chunk("near"), _chunk("far")],
            [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)],
        )

        results = await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=5)

        assert results[0].chunk_id == "near"

    async def test_top_k_is_respected(self, store):
        await store.upsert(
            [_chunk("a"), _chunk("b"), _chunk("c")],
            [(1.0, 0.0, 0.0, 0.0)] * 3,
        )

        assert len(await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=2)) == 2


class TestFilterTranslation:
    async def test_equality_filters(self, store):
        await store.upsert(
            [_chunk("a", document_id="doc-1"), _chunk("b", document_id="doc-2")],
            [(1.0, 0.0, 0.0, 0.0)] * 2,
        )

        results = await store.search_dense(
            (1.0, 0.0, 0.0, 0.0),
            top_k=5,
            filters=FieldFilter(MetadataField.DOCUMENT_ID, FilterOperator.EQ, "doc-2"),
        )

        assert [r.chunk_id for r in results] == ["b"]

    async def test_range_filters(self, store):
        await store.upsert(
            [_chunk("early", page=2), _chunk("late", page=40)],
            [(1.0, 0.0, 0.0, 0.0)] * 2,
        )

        results = await store.search_dense(
            (1.0, 0.0, 0.0, 0.0),
            top_k=5,
            filters=FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GT, 10),
        )

        assert [r.chunk_id for r in results] == ["late"]

    async def test_conjunctions(self, store):
        await store.upsert(
            [
                _chunk("match", document_id="doc-1", page=40),
                _chunk("wrong_page", document_id="doc-1", page=2),
                _chunk("wrong_doc", document_id="doc-2", page=40),
            ],
            [(1.0, 0.0, 0.0, 0.0)] * 3,
        )

        results = await store.search_dense(
            (1.0, 0.0, 0.0, 0.0),
            top_k=5,
            filters=(
                FieldFilter(MetadataField.DOCUMENT_ID, FilterOperator.EQ, "doc-1")
                & FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GT, 10)
            ),
        )

        assert [r.chunk_id for r in results] == ["match"]


class TestVersionedDeletion:
    async def test_superseded_versions_are_removed_and_the_current_one_kept(self, store):
        await store.upsert([_chunk("old", version=1)], [(1.0, 0.0, 0.0, 0.0)])
        await store.upsert([_chunk("new", version=2)], [(1.0, 0.0, 0.0, 0.0)])

        await store.delete_document("doc-1", before_version=2)

        results = await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=5)
        assert [r.chunk_id for r in results] == ["new"]

    async def test_deleting_without_a_version_removes_everything(self, store):
        await store.upsert(
            [_chunk("a", version=1), _chunk("b", version=2)],
            [(1.0, 0.0, 0.0, 0.0)] * 2,
        )

        await store.delete_document("doc-1")

        assert await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=5) == ()

    async def test_another_document_is_untouched(self, store):
        await store.upsert(
            [_chunk("mine", document_id="doc-1"), _chunk("theirs", document_id="doc-2")],
            [(1.0, 0.0, 0.0, 0.0)] * 2,
        )

        await store.delete_document("doc-1")

        results = await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=5)
        assert [r.chunk_id for r in results] == ["theirs"]


class TestCollectionCompatibility:
    async def test_reopening_with_the_same_configuration_is_fine(self, store):
        await store.ensure_collection(_spec((await store.collection_info()).name))

    async def test_a_different_embedding_model_is_refused(self, store):
        # The failure ADR-015 exists to prevent: same dimensions, different
        # model, answers that look right and are not.
        name = (await store.collection_info()).name

        with pytest.raises(CollectionMismatchError, match="embedding model"):
            await store.ensure_collection(_spec(name, model="text-embedding-3-small"))

    async def test_a_different_dimension_is_refused(self, store):
        name = (await store.collection_info()).name

        with pytest.raises(CollectionMismatchError, match="dimension"):
            await store.ensure_collection(_spec(name, dimension=8))

    async def test_creating_the_collection_also_indexes_the_filterable_fields(self, store):
        # Without payload indexes every metadata filter is a full scan, which
        # degrades linearly with the corpus. Qdrant reports what it has indexed.
        info = await store._client.get_collection((await store.collection_info()).name)

        indexed = set(info.payload_schema or {})
        assert {"document_id", "page_number", "author", "ingest_version"} <= indexed

    async def test_non_filterable_fields_are_not_indexed(self, store):
        # Indexing character offsets would cost memory for no query benefit.
        info = await store._client.get_collection((await store.collection_info()).name)

        indexed = set(info.payload_schema or {})
        assert not indexed & {"char_start", "char_end", "token_count", "chunking_strategy"}

    async def test_a_fresh_collection_reports_no_points(self, store):
        # The marker point holding the embedding model id is bookkeeping, not a
        # chunk, and reporting it as one is confusing in the UI.
        assert (await store.collection_info()).points_count == 0

    async def test_the_point_count_reflects_stored_chunks(self, store):
        await store.upsert([_chunk("a"), _chunk("b")], [(1.0, 0.0, 0.0, 0.0)] * 2)

        assert (await store.collection_info()).points_count == 2

    async def test_preparing_an_existing_collection_again_is_harmless(self, store):
        name = (await store.collection_info()).name
        await store.upsert([_chunk("a")], [(1.0, 0.0, 0.0, 0.0)])

        await store.ensure_collection(_spec(name))

        assert len(await store.search_dense((1.0, 0.0, 0.0, 0.0), top_k=5)) == 1

    async def test_the_marker_point_never_appears_in_results(self, store):
        # The collection records its embedding model in a sentinel point; it
        # must never surface as a chunk.
        await store.upsert([_chunk("a")], [(1.0, 0.0, 0.0, 0.0)])

        results = await store.search_dense((0.0, 0.0, 0.0, 1.0), top_k=10)

        assert all(r.chunk_id == "a" for r in results)
