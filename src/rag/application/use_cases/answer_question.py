"""Answering a question from the indexed documents."""

from __future__ import annotations

from collections.abc import Sequence

from rag.application.services import ContextBuilder
from rag.core.telemetry import stage
from rag.domain.errors import RerankingError
from rag.domain.models import (
    Answer,
    GenerationParams,
    Query,
    RetrievalRequest,
    ScoredChunk,
    Turn,
)
from rag.domain.policies import assemble_citations
from rag.domain.ports import LLMClient, Reranker, Retriever
from rag.domain.prompts import PromptSpec

__all__ = ["AnswerQuestionUseCase"]

# Anything the prompt builder needs; a plain class rather than a port, so it is
# referenced structurally to keep the application free of an infrastructure
# import.
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from rag.domain.models import ContextBlock, Prompt


class _PromptAssembler(Protocol):
    """The prompt-building capability this use case depends on."""

    def build(
        self,
        question: str,
        context: ContextBlock,
        history: Sequence[Turn] = (),
        spec: PromptSpec | None = None,
    ) -> Prompt:
        """Assemble a prompt."""
        ...


class AnswerQuestionUseCase:
    """Retrieves, reranks, prompts and answers, with citations.

    The stage ordering encodes two decisions worth restating:

    *Reranking is optional and degrades.* A reranker failure falls back to
    retrieval order and records the degradation on the answer. Quality drops;
    availability does not.

    *Generation is mandatory and propagates.* There is no sensible answer to
    degrade to, so the error reaches the caller.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        prompt_builder: _PromptAssembler,
        llm: LLMClient,
        generation_params: GenerationParams,
        prompt_spec: PromptSpec,
        top_k: int,
        rerank_top_k: int,
    ) -> None:
        """Wire the use case.

        Args:
            retriever: Finds candidate chunks.
            reranker: Narrows candidates to the best few.
            context_builder: Budgets and renders the context block.
            prompt_builder: Assembles the prompt.
            llm: Generates the answer.
            generation_params: Provider-neutral generation settings.
            prompt_spec: The contract the prompt must satisfy.
            top_k: Default retrieval breadth.
            rerank_top_k: Default number of chunks reaching the model.
        """
        self._retriever = retriever
        self._reranker = reranker
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._generation_params = generation_params
        self._prompt_spec = prompt_spec
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k

    async def execute(self, query: Query, history: Sequence[Turn] = ()) -> Answer:
        """Answer a question from the indexed documents.

        Args:
            query: The question, with any overrides and filters.
            history: Recent conversation turns. Reaches the prompt so the model
                can resolve references, and deliberately never reaches retrieval
                -- concatenating history onto the query dilutes the embedding
                and collapses recall (ADR-018).

        Returns:
            The answer, its citations, and how it was produced.

        Raises:
            RetrievalError: If retrieval fails.
            LLMError: If generation fails.
            PromptTooLargeError: If the assembled prompt exceeds its ceiling.
        """
        top_k = query.top_k or self._top_k
        rerank_top_k = query.rerank_top_k or self._rerank_top_k
        degraded: list[str] = []

        with stage("retrieval") as retrieval:
            candidates = await self._retriever.retrieve(
                RetrievalRequest(query_text=query.text, top_k=top_k, filters=query.filters)
            )
            retrieval.record(candidates=len(candidates))

        ranked = await self._rerank(query.text, candidates, rerank_top_k, degraded)

        context = self._context_builder.build(ranked)
        prompt = self._prompt_builder.build(
            question=query.text,
            context=context,
            history=history,
            spec=self._prompt_spec,
        )

        with stage("llm.generate", model=self._llm.model_id) as generation:
            response = await self._llm.generate(prompt, self._generation_params)
            generation.record(finish_reason=response.finish_reason)

        citations = assemble_citations(response.text, context)

        return Answer(
            text=response.text,
            citations=citations,
            model_id=response.model_id,
            prompt_version=prompt.version,
            token_usage=response.usage,
            retrieved_count=len(candidates),
            reranked_count=len(ranked),
            degraded_stages=tuple(degraded),
        )

    async def _rerank(
        self,
        question: str,
        candidates: Sequence[ScoredChunk],
        top_n: int,
        degraded: list[str],
    ) -> tuple[ScoredChunk, ...]:
        """Narrow candidates, falling back to retrieval order on failure."""
        if not candidates:
            return ()

        with stage("rerank", candidates=len(candidates)) as reranking:
            try:
                ranked = await self._reranker.rerank(question, candidates, top_n)
            except RerankingError:
                # Optional stage: keep the answer, record the quality loss.
                degraded.append("rerank")
                reranking.record(degraded=True)
                return tuple(candidates[:top_n])
            reranking.record(kept=len(ranked))
            return tuple(ranked)
