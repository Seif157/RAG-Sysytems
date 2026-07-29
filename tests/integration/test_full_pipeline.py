"""End-to-end validation of the whole pipeline.

Document → parser → metadata → chunking → embeddings → Qdrant → hybrid
retrieval → RRF fusion → BGE reranker → context builder → prompt builder → LLM →
answer with citations.

Real parsing, real Qdrant, real cross-encoder. Only the embedder and the LLM are
substituted, because both cost money per call and neither is what these tests
are checking: the embedder's job here is to be deterministic, and the fake LLM
lets a test assert on the *prompt*, which is the one artefact no live model
would let us inspect.
"""

from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from rag.config import Settings
from rag.core.container import Container
from rag.domain.errors import RAGError
from rag.domain.models import FieldFilter, FilterOperator, MetadataField, Query
from rag.domain.prompts import NOT_FOUND_PHRASE, is_refusal
from tests.fakes import FakeEmbedder, FakeLLMClient
from tests.fixtures import make_docx, make_markdown, make_pdf

pytestmark = [pytest.mark.integration]

QDRANT_URL = "http://localhost:6333"
RERANKER_MODEL = os.environ.get("RERANKER_TEST_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

HANDBOOK = """\
# Employee Handbook

## Annual Leave

Every full-time employee receives twenty-five days of paid annual leave per year.

## Requesting Leave

Leave requests must be submitted at least two weeks in advance via the HR portal.

## Expenses

Expense claims must be submitted within thirty days of the travel date.
"""

INCIDENT_PAGES = [
    "Incident Report: Gateway Outage\n\nPrepared by the platform team.",
    "Failure code ERR_CONN_4021 was logged at 03:12 and indicates a dropped connection.",
    "Mitigation: the gateway was restarted and connections recovered within eight minutes.",
]

POLICY_DOCX = [
    ("Heading 1", "Security Policy"),
    ("Heading 2", "Passwords"),
    ("Normal", "Passwords must be rotated every ninety days without exception."),
]


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
async def pipeline(qdrant_available, tmp_path, monkeypatch):
    """A fully wired pipeline over three ingested documents."""
    if not qdrant_available:
        pytest.skip("Qdrant is not running; start it with `docker compose up -d`")

    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)

    collection = f"e2e_{uuid.uuid4().hex[:12]}"
    settings = Settings.from_environment(
        {
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "COLLECTION_NAME": collection,
            "EMBEDDING_MODEL": "fake-embedder",
            "EMBEDDING_DIMENSION": "16",
            "CHUNK_SIZE": "300",
            "CHUNK_OVERLAP": "50",
            "TOP_K": "20",
            "RERANK_TOP_K": "5",
            "ENABLE_HYBRID": "true",
            "ENABLE_RERANK": "true",
            "RERANKER_PROVIDER": "bge",
            "RERANKER_MODEL": RERANKER_MODEL,
        },
        use_env_file=False,
    )

    container = Container(
        settings,
        embedder=FakeEmbedder(dimension=16, model_id="fake-embedder"),
        llm=FakeLLMClient(["Grounded answer [1]."]),
    )
    await container.prepare_store()

    try:
        container.reranker  # noqa: B018 - surfaces a model-load failure as a skip
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"reranker unavailable: {exc}")

    await container.ingest_document.execute("handbook.md", make_markdown(HANDBOOK))
    await container.ingest_document.execute("incident.pdf", make_pdf(INCIDENT_PAGES))
    await container.ingest_document.execute("security.docx", make_docx(POLICY_DOCX))

    try:
        yield container
    finally:
        client = AsyncQdrantClient(url=QDRANT_URL, timeout=10, check_compatibility=False)
        try:
            await client.delete_collection(collection)
        finally:
            await client.close()


class TestIngestionAcrossFormats:
    async def test_every_format_is_indexed(self, pipeline):
        documents = pipeline.catalog.list_documents()

        assert {d.filename for d in documents} == {
            "handbook.md",
            "incident.pdf",
            "security.docx",
        }
        assert all(d.chunk_count > 0 for d in documents)

    async def test_originals_are_retained_for_re_chunking(self, pipeline):
        from pathlib import Path

        assert all(Path(d.source_path).exists() for d in pipeline.catalog.list_documents())


class TestRetrievalQuality:
    async def test_an_exact_identifier_is_found(self, pipeline):
        # The case hybrid retrieval exists for: an embedding blurs this token.
        answer = await pipeline.answer_question.execute(Query(text="ERR_CONN_4021"))

        assert any("ERR_CONN_4021" in (c.snippet or "") for c in answer.citations)

    async def test_a_natural_language_question_reaches_its_answer(self, pipeline):
        await pipeline.answer_question.execute(
            Query(text="How many days of annual leave do employees get?")
        )

        prompt = pipeline.llm.last_prompt.user
        assert "twenty-five days" in prompt

    async def test_the_reranker_narrows_to_the_configured_count(self, pipeline):
        answer = await pipeline.answer_question.execute(Query(text="leave policy"))

        assert answer.retrieved_count > answer.reranked_count
        assert answer.reranked_count <= 5

    async def test_the_reranker_ran(self, pipeline):
        answer = await pipeline.answer_question.execute(Query(text="password rotation"))

        assert not answer.is_degraded
        assert answer.citations
        assert answer.citations[0].relevance_score is not None


class TestMetadataPropagation:
    async def test_a_pdf_citation_carries_its_page(self, pipeline):
        answer = await pipeline.answer_question.execute(Query(text="ERR_CONN_4021"))

        cited = next(c for c in answer.citations if "ERR_CONN_4021" in (c.snippet or ""))
        assert cited.filename == "incident.pdf"
        assert cited.page_number == 2

    async def test_a_docx_citation_carries_its_heading_path(self, pipeline):
        await pipeline.answer_question.execute(Query(text="passwords rotated ninety days"))

        prompt = pipeline.llm.last_prompt.user
        assert "Security Policy > Passwords" in prompt

    async def test_a_markdown_citation_carries_its_heading_path(self, pipeline):
        await pipeline.answer_question.execute(Query(text="annual leave entitlement"))

        prompt = pipeline.llm.last_prompt.user
        assert "Employee Handbook" in prompt

    async def test_pageless_formats_claim_no_page(self, pipeline):
        answer = await pipeline.answer_question.execute(Query(text="passwords rotated"))

        docx = [c for c in answer.citations if c.filename == "security.docx"]
        assert all(c.page_number is None for c in docx)


class TestCitationCorrectness:
    async def test_every_citation_points_at_text_the_model_was_shown(self, pipeline):
        # The invariant the whole system rests on. A citation naming a passage
        # the model never saw is worse than no citation at all.
        answer = await pipeline.answer_question.execute(Query(text="annual leave"))
        prompt = " ".join(pipeline.llm.last_prompt.user.split())

        assert answer.citations
        for citation in answer.citations:
            assert citation.snippet
            assert citation.snippet.rstrip("…")[:40] in prompt

    async def test_citations_carry_a_relevance_score_when_reranked(self, pipeline):
        answer = await pipeline.answer_question.execute(Query(text="annual leave"))

        assert all(
            c.relevance_score is not None and 0.0 <= c.relevance_score <= 1.0
            for c in answer.citations
        )

    async def test_duplicate_locations_are_merged(self, pipeline):
        pipeline._llm = FakeLLMClient(["Everything [1][2][3][4][5]."])
        answer = await pipeline.answer_question.execute(Query(text="leave and expenses"))

        locators = [(c.document_id, c.page_number, c.section) for c in answer.citations]
        assert len(locators) == len(set(locators))

    async def test_a_hallucinated_marker_produces_no_citation(self, pipeline):
        pipeline._llm = FakeLLMClient(["Confident nonsense [99]."])

        answer = await pipeline.answer_question.execute(Query(text="annual leave"))

        assert answer.citations == ()


class TestRefusal:
    async def test_the_prompt_prescribes_the_refusal_wording(self, pipeline):
        await pipeline.answer_question.execute(Query(text="anything"))

        assert NOT_FOUND_PHRASE in pipeline.llm.last_prompt.system

    async def test_a_refusal_is_recognisable(self, pipeline):
        pipeline._llm = FakeLLMClient([NOT_FOUND_PHRASE])

        answer = await pipeline.answer_question.execute(Query(text="the capital of France"))

        assert is_refusal(answer.text)
        assert answer.has_citations is False


class TestMetadataFiltering:
    async def test_a_filter_restricts_retrieval_across_both_paths(self, pipeline):
        clause = FieldFilter(MetadataField.FILENAME, FilterOperator.EQ, "incident.pdf")

        pipeline._llm = FakeLLMClient(["Everything [1][2][3][4][5]."])
        answer = await pipeline.answer_question.execute(
            Query(text="gateway connection leave passwords", filters=clause)
        )

        assert answer.citations
        assert all(c.filename == "incident.pdf" for c in answer.citations)

    async def test_a_page_range_filter_works(self, pipeline):
        clause = FieldFilter(MetadataField.PAGE_NUMBER, FilterOperator.GTE, 2)

        pipeline._llm = FakeLLMClient(["Everything [1][2][3][4][5]."])
        answer = await pipeline.answer_question.execute(
            Query(text="gateway outage connection", filters=clause)
        )

        paged = [c for c in answer.citations if c.page_number is not None]
        assert paged
        assert all(c.page_number >= 2 for c in paged)


class TestFailureHandling:
    async def test_an_unanswerable_question_still_returns_an_answer(self, pipeline):
        # Retrieval finds something for almost any query; what matters is that
        # the pipeline completes rather than erroring.
        answer = await pipeline.answer_question.execute(
            Query(text="the migratory patterns of arctic terns")
        )

        assert answer.text

    async def test_an_llm_failure_propagates_rather_than_inventing_an_answer(self, pipeline):
        from rag.domain.errors import LLMTimeoutError

        pipeline._llm = FakeLLMClient(error=LLMTimeoutError("60s elapsed"))

        with pytest.raises(LLMTimeoutError):
            await pipeline.answer_question.execute(Query(text="annual leave"))

    async def test_a_reranker_failure_degrades_instead_of_failing(self, pipeline):
        from rag.domain.errors import RerankingError
        from rag.domain.ports import Reranker

        class BrokenReranker(Reranker):
            async def rerank(self, query, candidates, top_n):
                raise RerankingError("model unavailable")

        pipeline.__dict__["reranker"] = BrokenReranker()
        pipeline.__dict__.pop("answer_question", None)

        answer = await pipeline.answer_question.execute(Query(text="annual leave"))

        assert answer.text
        assert "rerank" in answer.degraded_stages

    async def test_an_unsupported_upload_is_rejected_clearly(self, pipeline):
        with pytest.raises(RAGError):
            await pipeline.ingest_document.execute("photo.png", b"\x89PNG\r\n\x1a\n")


class TestReIngestion:
    async def test_a_corrected_document_replaces_its_predecessor(self, pipeline):
        corrected = HANDBOOK.replace("twenty-five days", "thirty days")

        document = await pipeline.ingest_document.execute("handbook.md", make_markdown(corrected))

        assert document.ingest_version == 2

        pipeline._llm = FakeLLMClient(["Everything [1][2][3][4][5]."])
        await pipeline.answer_question.execute(Query(text="How much annual leave?"))

        prompt = pipeline.llm.last_prompt.user
        assert "thirty days" in prompt
        assert "twenty-five days" not in prompt

    async def test_an_unchanged_re_upload_is_a_no_op(self, pipeline):
        before = pipeline.catalog.get(
            next(
                d.document_id
                for d in pipeline.catalog.list_documents()
                if d.filename == "handbook.md"
            )
        )

        again = await pipeline.ingest_document.execute("handbook.md", make_markdown(HANDBOOK))

        assert again.ingest_version == before.ingest_version
