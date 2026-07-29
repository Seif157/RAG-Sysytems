"""Behaviour of vector-store preparation at start-up.

The collection has to exist before the first upload, and it has to be the *right*
collection. Both were previously left to chance: nothing called
``ensure_collection``, so a fresh Qdrant failed on first use, and the
embedding-model compatibility guard -- the one that prevents confidently wrong
answers from stale vectors -- was unreachable code.
"""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.core.container import Container
from rag.domain.errors import CollectionMismatchError
from rag.domain.models import CollectionSpec, DistanceMetric
from tests.fakes import FakeEmbedder, FakeLLMClient, InMemoryVectorStore

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)


def _container(store: InMemoryVectorStore, **env: str) -> Container:
    settings = Settings.from_environment({"ENABLE_RERANK": "false", **env}, use_env_file=False)
    return Container(
        settings,
        vector_store=store,
        embedder=FakeEmbedder(dimension=384, model_id="BAAI/bge-small-en-v1.5"),
        llm=FakeLLMClient(),
    )


class TestPreparation:
    async def test_it_creates_the_collection(self):
        store = InMemoryVectorStore()

        await _container(store).prepare_store()

        assert (await store.collection_info()).name == "document_chunks"

    async def test_it_reports_the_collection_it_prepared(self):
        info = await _container(InMemoryVectorStore()).prepare_store()

        assert info.dense_dimension == 384
        assert info.embedding_model_id == "BAAI/bge-small-en-v1.5"

    async def test_it_uses_the_configured_collection_name(self):
        store = InMemoryVectorStore()

        await _container(store, COLLECTION_NAME="custom_chunks").prepare_store()

        assert (await store.collection_info()).name == "custom_chunks"

    async def test_preparing_twice_is_harmless(self):
        # Streamlit re-runs the script constantly; preparation must be idempotent.
        store = InMemoryVectorStore()
        container = _container(store)

        await container.prepare_store()
        await container.prepare_store()

        assert (await store.collection_info()).points_count == 0


class TestCompatibilityGuard:
    async def test_a_collection_from_another_embedding_model_is_refused(self):
        # The failure this guard exists for: same shape, different model,
        # answers that look right and are not (ADR-015).
        existing = InMemoryVectorStore(
            CollectionSpec(
                name="document_chunks",
                dense_dimension=384,
                distance=DistanceMetric.COSINE,
                embedding_model_id="models/embedding-001",
            )
        )

        with pytest.raises(CollectionMismatchError, match="embedding model"):
            await _container(existing).prepare_store()

    async def test_a_collection_of_the_wrong_width_is_refused(self):
        existing = InMemoryVectorStore(
            CollectionSpec(
                name="document_chunks",
                dense_dimension=768,
                distance=DistanceMetric.COSINE,
                embedding_model_id="BAAI/bge-small-en-v1.5",
            )
        )

        with pytest.raises(CollectionMismatchError, match="dimension"):
            await _container(existing).prepare_store()

    async def test_a_matching_collection_is_accepted(self):
        existing = InMemoryVectorStore(
            CollectionSpec(
                name="document_chunks",
                dense_dimension=384,
                distance=DistanceMetric.COSINE,
                embedding_model_id="BAAI/bge-small-en-v1.5",
            )
        )

        assert await _container(existing).prepare_store() is not None
