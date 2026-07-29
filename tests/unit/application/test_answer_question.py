"""Behaviour of the question-answering use case.

The whole query flow, exercised against in-memory doubles: no network, no API
key, no Docker. Everything asserted here is behaviour a user would notice.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.application.services import ContextBuilder
from rag.application.use_cases import AnswerQuestionUseCase
from rag.domain.errors import LLMTimeoutError, RerankingError
from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    DocumentType,
    GenerationParams,
    Query,
    Role,
    ScoredChunk,
    ScoreSource,
    Turn,
)
from rag.domain.ports import Reranker, Retriever
from rag.domain.prompts import PromptSpec
from rag.infrastructure.prompts import PromptBuilder
from tests.fakes import FakeLLMClient

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 27, tzinfo=UTC)
_PARAMS = GenerationParams(temperature=0.0, max_output_tokens=512, timeout_s=30.0)
_SPEC = PromptSpec(version="v1", max_prompt_tokens=8192)


def _scored(chunk_id: str = "a", text: str = "Revenue grew by twelve percent.") -> ScoredChunk:
    metadata = ChunkMetadata(
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
        embedding_model_id="fake",
        ingested_at=_NOW,
        page_number=4,
    )
    return ScoredChunk(
        chunk=Chunk(text=text, metadata=metadata), score=0.9, source=ScoreSource.DENSE
    )


class StubRetriever(Retriever):
    """Returns a fixed result set and records what it was asked."""

    def __init__(self, results: tuple[ScoredChunk, ...] = (), error: Exception | None = None):
        self.results = results
        self.error = error
        self.requests: list[object] = []

    async def retrieve(self, request):
        """Record the request and return the fixed results."""
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.results


class PassthroughReranker(Reranker):
    """Truncates to ``top_n`` without changing order."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0

    async def rerank(self, query, candidates, top_n):
        """Return the first ``top_n`` candidates, marked as reranked."""
        self.calls += 1
        if self.error is not None:
            raise self.error
        return tuple(c.rescored(c.score, ScoreSource.RERANKED) for c in candidates[:top_n])


def _use_case(
    *,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    llm: FakeLLMClient | None = None,
    top_k: int = 20,
    rerank_top_k: int = 5,
    budget: int = 1000,
) -> tuple[AnswerQuestionUseCase, FakeLLMClient]:
    client = llm or FakeLLMClient(["Revenue grew by twelve percent [1]."])
    use_case = AnswerQuestionUseCase(
        retriever=retriever or StubRetriever((_scored(),)),
        reranker=reranker or PassthroughReranker(),
        context_builder=ContextBuilder(budget_tokens=budget),
        prompt_builder=PromptBuilder(),
        llm=client,
        generation_params=_PARAMS,
        prompt_spec=_SPEC,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
    )
    return use_case, client


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_it_returns_the_generated_answer(self):
        use_case, _ = _use_case()

        answer = await use_case.execute(Query(text="How did revenue change?"))

        assert answer.text == "Revenue grew by twelve percent [1]."

    @pytest.mark.asyncio
    async def test_it_returns_a_citation_pointing_at_the_source(self):
        use_case, _ = _use_case()

        answer = await use_case.execute(Query(text="How did revenue change?"))

        assert len(answer.citations) == 1
        assert answer.citations[0].filename == "report.pdf"
        assert answer.citations[0].page_number == 4

    @pytest.mark.asyncio
    async def test_it_records_the_model_and_prompt_version(self):
        use_case, _ = _use_case()

        answer = await use_case.execute(Query(text="How did revenue change?"))

        assert answer.model_id == "fake-llm"
        assert answer.prompt_version == "v1"

    @pytest.mark.asyncio
    async def test_it_reports_how_much_was_retrieved_and_kept(self):
        retriever = StubRetriever((_scored("a"), _scored("b"), _scored("c")))
        use_case, _ = _use_case(retriever=retriever, rerank_top_k=2)

        answer = await use_case.execute(Query(text="q"))

        assert answer.retrieved_count == 3
        assert answer.reranked_count == 2

    @pytest.mark.asyncio
    async def test_the_retrieved_context_reaches_the_model(self):
        use_case, llm = _use_case()

        await use_case.execute(Query(text="How did revenue change?"))

        assert "Revenue grew by twelve percent." in llm.last_prompt.user

    @pytest.mark.asyncio
    async def test_the_question_reaches_the_model(self):
        use_case, llm = _use_case()

        await use_case.execute(Query(text="How did revenue change?"))

        assert "How did revenue change?" in llm.last_prompt.user

    @pytest.mark.asyncio
    async def test_the_grounding_instruction_reaches_the_model(self):
        use_case, llm = _use_case()

        await use_case.execute(Query(text="q"))

        assert llm.last_prompt.system


class TestRetrievalBreadth:
    @pytest.mark.asyncio
    async def test_it_retrieves_the_configured_number_of_candidates(self):
        retriever = StubRetriever((_scored(),))
        use_case, _ = _use_case(retriever=retriever, top_k=17)

        await use_case.execute(Query(text="q"))

        assert retriever.requests[0].top_k == 17

    @pytest.mark.asyncio
    async def test_a_query_can_override_the_configured_breadth(self):
        retriever = StubRetriever((_scored(),))
        use_case, _ = _use_case(retriever=retriever, top_k=17)

        await use_case.execute(Query(text="q", top_k=3))

        assert retriever.requests[0].top_k == 3

    @pytest.mark.asyncio
    async def test_a_query_filter_is_passed_through_to_retrieval(self):
        from rag.domain.models import FieldFilter, FilterOperator, MetadataField

        clause = FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.EQ, "PDF")
        retriever = StubRetriever((_scored(),))
        use_case, _ = _use_case(retriever=retriever)

        await use_case.execute(Query(text="q", filters=clause))

        assert retriever.requests[0].filters is clause


class TestEmptyRetrieval:
    @pytest.mark.asyncio
    async def test_it_still_answers_when_nothing_was_retrieved(self):
        # The model is told the documents do not cover the question, rather than
        # the request failing.
        llm = FakeLLMClient(["That is not stated in the provided documents."])
        use_case, _ = _use_case(retriever=StubRetriever(()), llm=llm)

        answer = await use_case.execute(Query(text="q"))

        assert answer.text == "That is not stated in the provided documents."

    @pytest.mark.asyncio
    async def test_an_answer_with_no_context_has_no_citations(self):
        llm = FakeLLMClient(["Not stated in the documents."])
        use_case, _ = _use_case(retriever=StubRetriever(()), llm=llm)

        answer = await use_case.execute(Query(text="q"))

        assert answer.has_citations is False

    @pytest.mark.asyncio
    async def test_the_reranker_is_not_called_for_an_empty_candidate_set(self):
        reranker = PassthroughReranker()
        use_case, _ = _use_case(retriever=StubRetriever(()), reranker=reranker)

        await use_case.execute(Query(text="q"))

        assert reranker.calls == 0


class TestDegradation:
    @pytest.mark.asyncio
    async def test_a_reranker_failure_does_not_fail_the_question(self):
        # Reranking is optional: quality drops, availability does not.
        reranker = PassthroughReranker(error=RerankingError("model failed to load"))
        use_case, _ = _use_case(reranker=reranker)

        answer = await use_case.execute(Query(text="q"))

        assert answer.text

    @pytest.mark.asyncio
    async def test_a_reranker_failure_is_reported_as_degradation(self):
        reranker = PassthroughReranker(error=RerankingError("model failed to load"))
        use_case, _ = _use_case(reranker=reranker)

        answer = await use_case.execute(Query(text="q"))

        assert answer.is_degraded is True
        assert "rerank" in answer.degraded_stages

    @pytest.mark.asyncio
    async def test_falling_back_still_narrows_to_the_configured_count(self):
        reranker = PassthroughReranker(error=RerankingError("nope"))
        retriever = StubRetriever((_scored("a"), _scored("b"), _scored("c")))
        use_case, _ = _use_case(retriever=retriever, reranker=reranker, rerank_top_k=2)

        answer = await use_case.execute(Query(text="q"))

        assert answer.reranked_count == 2

    @pytest.mark.asyncio
    async def test_an_llm_failure_does_propagate(self):
        # Generation is mandatory. There is no answer to degrade to.
        llm = FakeLLMClient(error=LLMTimeoutError("60s elapsed"))
        use_case, _ = _use_case(llm=llm)

        with pytest.raises(LLMTimeoutError):
            await use_case.execute(Query(text="q"))


class TestConversationHistory:
    @pytest.mark.asyncio
    async def test_history_reaches_the_prompt(self):
        use_case, llm = _use_case()
        history = (
            Turn(role=Role.USER, content="Who wrote this?", created_at=_NOW),
            Turn(role=Role.ASSISTANT, content="Ada Lovelace.", created_at=_NOW),
        )

        await use_case.execute(Query(text="When was she born?"), history=history)

        assert "Ada Lovelace." in llm.last_prompt.user

    @pytest.mark.asyncio
    async def test_history_does_not_reach_retrieval(self):
        # Concatenating history onto the query dilutes the embedding and
        # collapses recall (ADR-018).
        retriever = StubRetriever((_scored(),))
        use_case, _ = _use_case(retriever=retriever)
        history = (Turn(role=Role.USER, content="Who wrote this?", created_at=_NOW),)

        await use_case.execute(Query(text="When was she born?"), history=history)

        assert retriever.requests[0].query_text == "When was she born?"
