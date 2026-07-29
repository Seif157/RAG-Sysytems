"""Behaviour of the cross-encoder reranker.

The reranker's contract is narrow and the narrowness is the point: it may only
re-order and truncate. A reranker that merges, edits or invents passages breaks
citation accuracy, and it would do so invisibly -- the answer would still cite
something, just not the thing it came from.

These tests inject a scoring function, so they exercise the contract without
downloading a model. The real model is exercised in the integration suite.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from rag.domain.errors import RerankingError
from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    DocumentType,
    ScoredChunk,
    ScoreSource,
)
from rag.infrastructure.reranker import BgeReranker

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _scored(chunk_id: str, text: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            text=text,
            metadata=ChunkMetadata(
                document_id="doc-1",
                chunk_id=chunk_id,
                ingest_version=1,
                filename="handbook.pdf",
                document_type=DocumentType.PDF,
                chunk_index=0,
                char_start=0,
                char_end=len(text),
                token_count=10,
                chunking_strategy="recursive",
                embedding_model_id="fake",
                ingested_at=_NOW,
                page_number=3,
            ),
        ),
        score=score,
        source=ScoreSource.FUSED,
    )


def _reranker(scores: list[float], **kwargs) -> BgeReranker:
    """A reranker whose model returns fixed scores, in candidate order."""
    return BgeReranker(model="stub", scorer=lambda pairs: scores[: len(pairs)], **kwargs)


CANDIDATES = (
    _scored("a", "The cafeteria serves lunch."),
    _scored("b", "Employees get twenty-five days of leave."),
    _scored("c", "Expense claims are due in thirty days."),
)


class TestReordering:
    async def test_it_reorders_by_the_models_judgement(self):
        # The whole point: retrieval's order is a starting guess, not an answer.
        reranker = _reranker([-5.0, 9.0, -2.0])

        results = await reranker.rerank("How much leave?", CANDIDATES, top_n=3)

        assert [r.chunk_id for r in results] == ["b", "c", "a"]

    async def test_it_truncates_to_top_n(self):
        reranker = _reranker([-5.0, 9.0, -2.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=2)

        assert [r.chunk_id for r in results] == ["b", "c"]

    async def test_asking_for_more_than_exists_returns_what_exists(self):
        reranker = _reranker([1.0, 2.0, 3.0])

        assert len(await reranker.rerank("q", CANDIDATES, top_n=99)) == 3

    async def test_results_are_ordered_by_descending_score(self):
        reranker = _reranker([1.0, 9.0, 5.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=3)

        assert [r.score for r in results] == sorted([r.score for r in results], reverse=True)


class TestTheContract:
    async def test_every_result_came_from_the_input(self):
        reranker = _reranker([1.0, 2.0, 3.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=3)

        assert {r.chunk_id for r in results} <= {c.chunk_id for c in CANDIDATES}

    async def test_chunk_text_is_not_modified(self):
        # A reranker that rewrote passages would make every citation a lie.
        reranker = _reranker([1.0, 2.0, 3.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=3)

        original = {c.chunk_id: c.text for c in CANDIDATES}
        assert all(r.text == original[r.chunk_id] for r in results)

    async def test_chunk_metadata_is_not_modified(self):
        reranker = _reranker([1.0, 2.0, 3.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=1)

        assert results[0].chunk.metadata.page_number == 3

    async def test_results_are_marked_as_reranked(self):
        # Only a reranked score is a comparable relevance score; this is what
        # allows a citation to show one honestly.
        reranker = _reranker([1.0, 2.0, 3.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=3)

        assert all(r.source is ScoreSource.RERANKED for r in results)

    async def test_no_duplicates_are_introduced(self):
        reranker = _reranker([1.0, 2.0, 3.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=3)

        assert len({r.chunk_id for r in results}) == len(results)


class TestScoreNormalisation:
    async def test_scores_are_normalised_into_the_unit_interval(self):
        # Cross-encoders emit raw logits; Citation.relevance_score requires
        # [0, 1], and a score outside it would fail validation at the very end
        # of the pipeline.
        reranker = _reranker([12.0, -12.0, 0.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=3)

        assert all(0.0 <= r.score <= 1.0 for r in results)

    async def test_a_strongly_relevant_passage_scores_near_one(self):
        reranker = _reranker([10.0, -10.0, -10.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=1)

        assert results[0].score > 0.99

    async def test_a_strongly_irrelevant_passage_scores_near_zero(self):
        reranker = _reranker([-10.0, -11.0, -12.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=3)

        assert results[-1].score < 0.01

    async def test_a_neutral_score_lands_mid_range(self):
        reranker = _reranker([0.0, -20.0, -20.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=1)

        assert results[0].score == pytest.approx(0.5, abs=0.01)

    async def test_normalisation_preserves_ordering(self):
        # Sigmoid is monotonic, so it must never change the ranking.
        reranker = _reranker([3.0, 1.0, 2.0])

        results = await reranker.rerank("q", CANDIDATES, top_n=3)

        assert [r.chunk_id for r in results] == ["a", "c", "b"]


class TestEdgeCases:
    async def test_no_candidates_yields_no_results(self):
        reranker = _reranker([])

        assert await reranker.rerank("q", (), top_n=5) == ()

    async def test_the_model_is_not_called_for_an_empty_candidate_set(self):
        calls: list[object] = []
        reranker = BgeReranker(model="stub", scorer=lambda pairs: calls.append(pairs) or [])

        await reranker.rerank("q", (), top_n=5)

        assert calls == []

    async def test_a_single_candidate_is_returned(self):
        reranker = _reranker([1.0])

        results = await reranker.rerank("q", CANDIDATES[:1], top_n=5)

        assert [r.chunk_id for r in results] == ["a"]

    async def test_a_non_positive_top_n_is_rejected(self):
        reranker = _reranker([1.0])

        with pytest.raises(ValueError, match="top_n"):
            await reranker.rerank("q", CANDIDATES, top_n=0)


class TestFailureHandling:
    async def test_a_model_failure_becomes_a_reranking_error(self):
        # Callers catch RerankingError and degrade to fusion order; a raw
        # RuntimeError from torch would escape that and fail the question.
        def explode(pairs):
            raise RuntimeError("CUDA out of memory")

        reranker = BgeReranker(model="stub", scorer=explode)

        with pytest.raises(RerankingError):
            await reranker.rerank("q", CANDIDATES, top_n=3)

    async def test_the_original_failure_is_preserved_as_the_cause(self):
        def explode(pairs):
            raise RuntimeError("CUDA out of memory")

        reranker = BgeReranker(model="stub", scorer=explode)

        with pytest.raises(RerankingError) as caught:
            await reranker.rerank("q", CANDIDATES, top_n=3)

        assert isinstance(caught.value.__cause__, RuntimeError)

    async def test_a_score_count_mismatch_is_caught(self):
        # Silently zipping mismatched lists would misattribute every score to
        # the wrong passage -- worse than failing.
        reranker = BgeReranker(model="stub", scorer=lambda pairs: [1.0])

        with pytest.raises(RerankingError, match="scores"):
            await reranker.rerank("q", CANDIDATES, top_n=3)


class TestPairConstruction:
    async def test_the_query_is_paired_with_every_candidate(self):
        seen: list[tuple[str, str]] = []

        def capture(pairs):
            seen.extend(pairs)
            return [0.0] * len(pairs)

        reranker = BgeReranker(model="stub", scorer=capture)
        await reranker.rerank("How much leave?", CANDIDATES, top_n=3)

        assert [q for q, _ in seen] == ["How much leave?"] * 3
        assert [p for _, p in seen] == [c.text for c in CANDIDATES]


class TestConcurrency:
    async def test_scoring_does_not_block_the_event_loop(self):
        # CrossEncoder.predict is synchronous and CPU-bound. Run inline, it
        # would stall every other coroutine for its whole duration.
        import time

        def slow(pairs):
            time.sleep(0.15)
            return [0.0] * len(pairs)

        reranker = BgeReranker(model="stub", scorer=slow)
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.01)
                ticks += 1

        await asyncio.gather(reranker.rerank("q", CANDIDATES, top_n=3), tick())

        assert ticks == 10
