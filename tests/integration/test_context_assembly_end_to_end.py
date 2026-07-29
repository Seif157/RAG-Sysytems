"""Context assembly over real documents in real Qdrant.

The policies are unit-tested against constructed chunks. This checks the
properties survive contact with real parsing, real chunking, and a real vector
store round trip -- which is where provenance is most likely to be quietly lost.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from rag.config import Settings
from rag.core.container import Container
from rag.domain.models import Query
from tests.fakes import FakeEmbedder, FakeLLMClient
from tests.fixtures import make_markdown, make_pdf

pytestmark = [pytest.mark.integration]

QDRANT_URL = "http://localhost:6333"

HANDBOOK = """\
# Employee Handbook

## Annual Leave

Every full-time employee receives twenty-five days of paid annual leave per year.

## Requesting Leave

Leave requests must be submitted at least two weeks in advance via the HR portal.

## Carrying Leave Over

Unused annual leave may not be carried into the following calendar year.
"""

NOTICE = "This document is confidential and must not be shared externally."


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
async def container(qdrant_available, tmp_path, monkeypatch):
    """A container backed by real Qdrant, with a throwaway collection."""
    if not qdrant_available:
        pytest.skip("Qdrant is not running; start it with `docker compose up -d`")

    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)

    collection = f"ctx_{uuid.uuid4().hex[:12]}"
    settings = Settings.from_environment(
        {
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "COLLECTION_NAME": collection,
            "EMBEDDING_DIMENSION": "8",
            "EMBEDDING_MODEL": "fake-embedder",
            "CHUNK_SIZE": "260",
            "CHUNK_OVERLAP": "40",
            "TOP_K": "20",
            "ENABLE_RERANK": "false",
            "CONTEXT_TOKEN_BUDGET": "2048",
        },
        use_env_file=False,
    )
    built = Container(
        settings,
        embedder=FakeEmbedder(dimension=8, model_id="fake-embedder"),
        llm=FakeLLMClient(["As stated [1][2][3][4][5]."]),
    )
    await built.prepare_store()
    try:
        yield built
    finally:
        client = AsyncQdrantClient(url=QDRANT_URL, timeout=10, check_compatibility=False)
        try:
            await client.delete_collection(collection)
        finally:
            await client.close()


class TestAssemblyOverRealDocuments:
    async def test_the_context_carries_provenance_for_every_passage(self, container):
        await container.ingest_document.execute("handbook.md", make_markdown(HANDBOOK))

        await container.answer_question.execute(Query(text="How much annual leave?"))

        prompt = container.llm.last_prompt.user
        assert "handbook.md" in prompt
        assert "Employee Handbook" in prompt

    async def test_passages_are_numbered_contiguously_in_the_prompt(self, container):
        await container.ingest_document.execute("handbook.md", make_markdown(HANDBOOK))

        await container.answer_question.execute(Query(text="leave policy"))

        prompt = container.llm.last_prompt.user
        markers = [f"[{n}]" for n in range(1, prompt.count("] (") + 1)]
        assert all(marker in prompt for marker in markers)

    async def test_a_document_appears_as_one_contiguous_run(self, container):
        # Two documents, so ordering has something to interleave and must not.
        import itertools

        await container.ingest_document.execute("handbook.md", make_markdown(HANDBOOK))
        await container.ingest_document.execute("notice.pdf", make_pdf([NOTICE]))

        answer = await container.answer_question.execute(Query(text="leave and confidentiality"))

        filenames = [c.filename for c in answer.citations]
        runs = [name for name, _ in itertools.groupby(filenames)]
        assert len(runs) == len(set(runs))

    async def test_repeated_boilerplate_is_included_once(self, container):
        # The same notice ingested as two documents. Both chunks are retrievable;
        # only one should reach the prompt.
        await container.ingest_document.execute("notice-a.pdf", make_pdf([NOTICE]))
        await container.ingest_document.execute("notice-b.pdf", make_pdf([NOTICE]))

        await container.answer_question.execute(Query(text="confidential sharing externally"))

        prompt = container.llm.last_prompt.user
        assert prompt.count("must not be shared externally") == 1

    async def test_every_citation_resolves_to_a_passage_in_the_prompt(self, container):
        # The end-to-end invariant: a citation must point at text the model was
        # actually shown, after parsing, chunking, storage, ordering and budgeting.
        await container.ingest_document.execute("handbook.md", make_markdown(HANDBOOK))

        answer = await container.answer_question.execute(Query(text="How much annual leave?"))

        prompt = container.llm.last_prompt.user
        assert answer.citations
        for citation in answer.citations:
            assert citation.snippet
            assert citation.snippet.rstrip("…")[:40] in " ".join(prompt.split())


class TestBudgetIsRespected:
    async def test_a_tight_budget_drops_passages_and_says_so(self, container, tmp_path):
        await container.ingest_document.execute("handbook.md", make_markdown(HANDBOOK))

        # Rebuild the answering path with a budget too small for everything.
        from rag.application.services import ContextBuilder
        from rag.application.use_cases import AnswerQuestionUseCase
        from rag.infrastructure.prompts import PromptBuilder

        use_case = AnswerQuestionUseCase(
            retriever=container.retriever,
            reranker=container.reranker,
            context_builder=ContextBuilder(budget_tokens=40, count_tokens=container.token_counter),
            prompt_builder=PromptBuilder(container.token_counter),
            llm=container.llm,
            generation_params=container.generation_params(),
            prompt_spec=container.prompt_spec(),
            top_k=20,
            rerank_top_k=5,
        )

        await use_case.execute(Query(text="leave policy"))

        prompt = container.llm.last_prompt.user
        assert prompt.count("] (") <= 2
