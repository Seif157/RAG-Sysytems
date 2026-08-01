"""Function calling stays controlled, bounded, and citation-safe."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.application.services import ContextBuilder
from rag.application.tools import SearchDocumentsTool, ToolExecutor
from rag.application.use_cases import AgenticAnswerUseCase
from rag.domain.errors import ToolCallLimitError, ToolValidationError
from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    DocumentType,
    GenerationParams,
    LLMMessage,
    LLMResponse,
    Query,
    ScoredChunk,
    ScoreSource,
    ToolCall,
)
from rag.domain.ports import Reranker, Retriever
from tests.fakes import FakeLLMClient

pytestmark = pytest.mark.unit


def _scored() -> ScoredChunk:
    metadata = ChunkMetadata(
        document_id="handbook",
        chunk_id="chunk-1",
        ingest_version=1,
        filename="handbook.pdf",
        document_type=DocumentType.PDF,
        chunk_index=0,
        char_start=0,
        char_end=30,
        token_count=8,
        chunking_strategy="recursive",
        embedding_model_id="fake",
        ingested_at=datetime.now(UTC),
        page_number=14,
        section="Annual Leave",
    )
    return ScoredChunk(
        Chunk("Employees receive 21 days of annual leave.", metadata), 0.9, ScoreSource.RERANKED
    )


class _Retriever(Retriever):
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, request):
        self.calls += 1
        return (_scored(),)


class _Reranker(Reranker):
    async def rerank(self, query, candidates, top_n):
        return tuple(candidates[:top_n])


class _AgentLLM(FakeLLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.messages: list[tuple[LLMMessage, ...]] = []

    async def respond(self, messages, tools, params):
        self.messages.append(messages)
        return self.responses.pop(0)


def _use_case(llm: _AgentLLM, retriever: _Retriever | None = None, *, rounds: int = 3):
    retriever = retriever or _Retriever()
    context_builder = ContextBuilder(
        budget_tokens=1000, count_tokens=lambda text: len(text.split())
    )
    search = SearchDocumentsTool(
        retriever, _Reranker(), context_builder, max_top_k=20, rerank_top_k=5
    )
    return AgenticAnswerUseCase(
        llm=llm,
        executor=ToolExecutor(search, timeout_s=1),
        context_builder=context_builder,
        generation_params=GenerationParams(0, 100, 5),
        prompt_version="v1",
        max_rounds=rounds,
        max_calls_per_round=2,
        max_total_calls=5,
        max_result_tokens=1000,
    ), retriever


class TestToolModels:
    def test_tool_arguments_are_defensively_frozen(self):
        arguments = {"query": "leave"}
        call = ToolCall("call-1", "search_documents", arguments)
        arguments["query"] = "changed"
        assert call.arguments["query"] == "leave"

    def test_non_mapping_arguments_are_rejected(self):
        with pytest.raises(TypeError):
            ToolCall("call-1", "search_documents", [])  # type: ignore[arg-type]


class TestSearchValidation:
    @pytest.fixture
    def tool(self):
        return SearchDocumentsTool(
            _Retriever(), _Reranker(), ContextBuilder(1000), max_top_k=20, rerank_top_k=5
        )

    @pytest.mark.parametrize(
        "arguments",
        [
            {"query": ""},
            {"query": "x", "top_k": 0},
            {"query": "x", "top_k": 21},
            {"query": "x", "unexpected": True},
            {"query": "x" * 1001},
        ],
    )
    def test_invalid_arguments_are_rejected(self, tool, arguments):
        with pytest.raises(ToolValidationError):
            tool.validate(arguments, None)

    def test_unauthorized_documents_are_rejected(self, tool):
        with pytest.raises(ToolValidationError, match="authorized"):
            tool.validate({"query": "leave", "document_ids": ["secret"]}, frozenset({"handbook"}))


class TestAgenticAnswer:
    async def test_greeting_returns_directly_without_retrieval(self):
        llm = _AgentLLM([LLMResponse(model_id="fake", text="Hello! How can I help?")])
        use_case, retriever = _use_case(llm)
        answer = await use_case.execute(Query("Hello"))
        assert answer.citations == ()
        assert retriever.calls == 0

    async def test_document_question_searches_and_builds_trusted_citation(self):
        call = ToolCall("call-1", "search_documents", {"query": "annual leave"})
        llm = _AgentLLM(
            [
                LLMResponse(model_id="fake", tool_calls=(call,), finish_reason="tool_calls"),
                LLMResponse(model_id="fake", text="Employees receive 21 days [1]."),
            ]
        )
        use_case, retriever = _use_case(llm)
        answer = await use_case.execute(
            Query("What is the annual leave policy?"), allowed_document_ids=frozenset({"handbook"})
        )
        assert retriever.calls == 1
        assert answer.citations[0].document_id == "handbook"
        assert llm.messages[1][-1].role == "tool"

    async def test_unknown_citation_is_never_displayed(self):
        call = ToolCall("call-1", "search_documents", {"query": "leave"})
        llm = _AgentLLM(
            [
                LLMResponse(model_id="fake", tool_calls=(call,)),
                LLMResponse(model_id="fake", text="Unsupported claim [99]."),
            ]
        )
        use_case, _ = _use_case(llm)
        answer = await use_case.execute(Query("leave"))
        assert answer.citations == ()

    async def test_tool_rounds_are_bounded(self):
        call = ToolCall("call-1", "search_documents", {"query": "leave"})
        llm = _AgentLLM([LLMResponse(model_id="fake", tool_calls=(call,))] * 3)
        use_case, _ = _use_case(llm, rounds=1)
        with pytest.raises(ToolCallLimitError):
            await use_case.execute(Query("leave"))
