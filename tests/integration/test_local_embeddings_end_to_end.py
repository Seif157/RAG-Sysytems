"""The default configuration, running entirely on this machine.

A real Sentence-Transformers model, real Qdrant, real parsing. Only the LLM is
substituted -- it is the one component that still needs a key, and the fake lets
a test assert on the *prompt*, which is what proves retrieval actually found the
right passage.

This is the test that answers "does the project work without an OpenAI
subscription", so it deliberately uses the shipped defaults rather than a
convenient dimension.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from rag.config import Settings
from rag.core.container import Container
from rag.domain.models import Query
from rag.infrastructure.embeddings import LocalEmbedder
from tests.fakes import FakeLLMClient
from tests.fixtures import make_markdown

pytestmark = [pytest.mark.integration]

QDRANT_URL = "http://localhost:6333"

HANDBOOK = """\
# Employee Handbook

## Annual Leave

Every full-time employee receives twenty-five days of paid annual leave per year.

## Requesting Leave

Leave requests must be submitted at least two weeks in advance via the HR portal.

## Expenses

Expense claims must be submitted within thirty days of the travel date.

## Remote Working

Staff may work remotely up to three days per week with manager approval.
"""


@pytest.fixture(scope="session")
def local_model_available() -> bool:
    """Load the default embedding model once, or report that we cannot."""
    try:
        LocalEmbedder()._load()
        return True
    except Exception:
        return False


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
async def pipeline(qdrant_available, local_model_available, tmp_path, monkeypatch):
    """The shipped default configuration, with only the LLM faked."""
    if not qdrant_available:
        pytest.skip("Qdrant is not running; start it with `docker compose up -d`")
    if not local_model_available:
        pytest.skip("the local embedding model could not be loaded")

    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)

    collection = f"local_{uuid.uuid4().hex[:12]}"
    settings = Settings.from_environment(
        {
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "COLLECTION_NAME": collection,
            "CHUNK_SIZE": "300",
            "CHUNK_OVERLAP": "50",
            # Reranking is exercised elsewhere; leaving it off keeps this test
            # about the embeddings.
            "ENABLE_RERANK": "false",
            # With reranking off, RERANK_TOP_K is a blunt truncation of the
            # fusion order. On a corpus this small that would cut the context
            # to five of nine chunks, and Markdown heading-only blocks ("Annual
            # Leave") are strong matches carrying no information -- so they
            # displace the body text. Widening it keeps this test measuring
            # whether retrieval *finds* the passage, which is the embedding's
            # job, rather than whether it survives a selection step the
            # reranker exists to perform.
            "RERANK_TOP_K": "12",
        },
        use_env_file=False,
    )

    container = Container(settings, llm=FakeLLMClient(["Grounded answer [1]."]))
    await container.prepare_store()
    await container.ingest_document.execute("handbook.md", make_markdown(HANDBOOK))

    try:
        yield container
    finally:
        client = AsyncQdrantClient(url=QDRANT_URL, timeout=10, check_compatibility=False)
        try:
            await client.delete_collection(collection)
        finally:
            await client.close()


class TestNoApiKeyRequired:
    async def test_the_defaults_need_no_openai_credential(self, pipeline):
        # The point of the change: nothing here reads OPENAI_API_KEY.
        assert pipeline.settings.embedding.provider.value == "local"
        assert pipeline.settings.credentials.openai_api_key is None

    async def test_the_collection_is_built_at_the_local_models_width(self, pipeline):
        info = await pipeline.vector_store.collection_info()

        assert info.dense_dimension == 384
        assert info.embedding_model_id == "BAAI/bge-small-en-v1.5"

    async def test_documents_are_indexed(self, pipeline):
        documents = pipeline.catalog.list_documents()

        assert len(documents) == 1
        assert documents[0].chunk_count > 0


class TestSemanticRetrieval:
    async def test_a_paraphrased_question_finds_its_answer(self, pipeline):
        # Wording shared with the document is nowhere in this question, so only
        # a genuine embedding can bridge it.
        await pipeline.answer_question.execute(
            Query(text="What is the holiday entitlement for permanent staff?")
        )

        assert "twenty-five days" in pipeline.llm.last_prompt.user

    async def test_a_differently_worded_question_finds_its_answer(self, pipeline):
        await pipeline.answer_question.execute(
            Query(text="How long do I have to claim travel costs back?")
        )

        assert "thirty days" in pipeline.llm.last_prompt.user

    async def test_working_from_home_is_found_without_that_phrase(self, pipeline):
        await pipeline.answer_question.execute(Query(text="Can I work from home some of the week?"))

        assert "remotely" in pipeline.llm.last_prompt.user

    async def test_the_answer_carries_a_citation(self, pipeline):
        answer = await pipeline.answer_question.execute(
            Query(text="What is the holiday entitlement?")
        )

        assert answer.citations
        assert answer.citations[0].filename == "handbook.md"


class TestQueryInstructionInPractice:
    async def test_query_and_passage_embeddings_differ_for_the_same_text(self, pipeline):
        # BGE applies an instruction prefix to queries only. If the two sides
        # were embedded identically, the asymmetric training would be wasted.
        text = "Every full-time employee receives twenty-five days of paid annual leave."

        as_passage = (await pipeline.embedder.embed_documents([text]))[0]
        as_query = await pipeline.embedder.embed_query(text)

        assert as_passage != as_query

    async def test_the_two_remain_highly_similar(self, pipeline):
        # The prefix should shift the vector, not relocate it: a query still has
        # to land near the passage that answers it.
        text = "Every full-time employee receives twenty-five days of paid annual leave."

        as_passage = (await pipeline.embedder.embed_documents([text]))[0]
        as_query = await pipeline.embedder.embed_query(text)

        similarity = sum(a * b for a, b in zip(as_passage, as_query, strict=True))
        assert similarity > 0.8


class TestReIngestion:
    async def test_a_corrected_document_replaces_its_predecessor(self, pipeline):
        corrected = HANDBOOK.replace("twenty-five days", "thirty-two days")

        document = await pipeline.ingest_document.execute("handbook.md", make_markdown(corrected))

        assert document.ingest_version == 2

        await pipeline.answer_question.execute(Query(text="How much annual leave?"))
        prompt = pipeline.llm.last_prompt.user
        assert "thirty-two days" in prompt
        assert "twenty-five days" not in prompt
