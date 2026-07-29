"""Behaviour of reciprocal rank fusion.

The densest test in the retrieval path and the cheapest: a pure function over
ranked lists, no I/O, no models. Fusion decides which chunks reach the reranker,
so a bug here caps the quality of every answer regardless of how good the
retrievers are.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    DocumentType,
    ScoredChunk,
    ScoreSource,
)
from rag.domain.policies import reciprocal_rank_fusion

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _scored(chunk_id: str, score: float, source: ScoreSource) -> ScoredChunk:
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        ingest_version=1,
        filename="report.pdf",
        document_type=DocumentType.PDF,
        chunk_index=0,
        char_start=0,
        char_end=10,
        token_count=8,
        chunking_strategy="recursive",
        embedding_model_id="fake",
        ingested_at=_NOW,
    )
    return ScoredChunk(
        chunk=Chunk(text=f"text of {chunk_id}", metadata=metadata), score=score, source=source
    )


def _dense(*ids: str) -> tuple[ScoredChunk, ...]:
    """A dense result list in rank order, with plausibly-scaled scores."""
    return tuple(
        _scored(chunk_id, 0.9 - index * 0.1, ScoreSource.DENSE)
        for index, chunk_id in enumerate(ids)
    )


def _sparse(*ids: str) -> tuple[ScoredChunk, ...]:
    """A sparse result list in rank order. BM25 scores are unbounded."""
    return tuple(
        _scored(chunk_id, 12.0 - index * 2.0, ScoreSource.SPARSE)
        for index, chunk_id in enumerate(ids)
    )


class TestDegenerateInputs:
    def test_fusing_nothing_yields_nothing(self):
        assert reciprocal_rank_fusion([]) == ()

    def test_fusing_empty_lists_yields_nothing(self):
        assert reciprocal_rank_fusion([((), 1.0), ((), 1.0)]) == ()

    def test_a_single_list_keeps_its_order(self):
        fused = reciprocal_rank_fusion([(_dense("a", "b", "c"), 1.0)])

        assert [s.chunk_id for s in fused] == ["a", "b", "c"]

    def test_an_empty_list_alongside_a_populated_one_is_ignored(self):
        # A query sharing no vocabulary with the corpus produces no sparse hits;
        # that must not disturb the dense ranking.
        fused = reciprocal_rank_fusion([(_dense("a", "b"), 1.0), ((), 1.0)])

        assert [s.chunk_id for s in fused] == ["a", "b"]


class TestFusion:
    def test_a_chunk_found_by_both_retrievers_outranks_one_found_by_either(self):
        # The core claim: agreement between two independent signals is evidence.
        fused = reciprocal_rank_fusion(
            [(_dense("dense_only", "both"), 1.0), (_sparse("both", "sparse_only"), 1.0)]
        )

        assert next(s.chunk_id for s in fused) == "both"

    def test_every_chunk_from_every_list_survives(self):
        fused = reciprocal_rank_fusion([(_dense("a", "b"), 1.0), (_sparse("c", "d"), 1.0)])

        assert {s.chunk_id for s in fused} == {"a", "b", "c", "d"}

    def test_a_chunk_appears_once_however_many_lists_found_it(self):
        fused = reciprocal_rank_fusion([(_dense("a"), 1.0), (_sparse("a"), 1.0)])

        assert len(fused) == 1

    def test_results_are_ordered_by_descending_fused_score(self):
        fused = reciprocal_rank_fusion(
            [(_dense("a", "b", "c"), 1.0), (_sparse("c", "b", "a"), 1.0)]
        )

        scores = [s.score for s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_fused_results_are_marked_as_fused(self):
        # So a citation does not later present a fused score as a relevance
        # score; only a reranker produces one of those.
        fused = reciprocal_rank_fusion([(_dense("a"), 1.0), (_sparse("a"), 1.0)])

        assert fused[0].source is ScoreSource.FUSED

    def test_the_chunk_content_is_carried_through_unchanged(self):
        fused = reciprocal_rank_fusion([(_dense("a"), 1.0)])

        assert fused[0].text == "text of a"


class TestScaleIndependence:
    def test_raw_score_magnitudes_do_not_affect_the_outcome(self):
        # The reason RRF was chosen: BM25 scores are unbounded and cosine scores
        # are not, so any blend of magnitudes needs re-tuning per corpus.
        modest = reciprocal_rank_fusion([(_dense("a", "b"), 1.0), (_sparse("b", "a"), 1.0)])
        inflated = reciprocal_rank_fusion(
            [
                (
                    tuple(
                        _scored(cid, 1_000_000.0 - i, ScoreSource.DENSE)
                        for i, cid in enumerate(["a", "b"])
                    ),
                    1.0,
                ),
                (
                    tuple(
                        _scored(cid, 0.000_001 - i * 1e-9, ScoreSource.SPARSE)
                        for i, cid in enumerate(["b", "a"])
                    ),
                    1.0,
                ),
            ]
        )

        assert [s.chunk_id for s in modest] == [s.chunk_id for s in inflated]


class TestWeighting:
    def test_equal_weights_let_rank_decide(self):
        fused = reciprocal_rank_fusion(
            [(_dense("d1", "shared"), 1.0), (_sparse("s1", "shared"), 1.0)]
        )

        assert [s.chunk_id for s in fused][:2] == ["shared", "d1"]

    def test_a_zero_weight_removes_a_list_from_consideration(self):
        # How ENABLE_HYBRID=false and SPARSE_WEIGHT=0 differ from each other:
        # this is fusion declining to use a list it was given.
        fused = reciprocal_rank_fusion([(_dense("a"), 1.0), (_sparse("b"), 0.0)])

        assert [s.chunk_id for s in fused] == ["a"]

    def test_a_heavier_weight_promotes_that_lists_top_result(self):
        dense_favoured = reciprocal_rank_fusion([(_dense("d"), 0.9), (_sparse("s"), 0.1)])
        sparse_favoured = reciprocal_rank_fusion([(_dense("d"), 0.1), (_sparse("s"), 0.9)])

        assert dense_favoured[0].chunk_id == "d"
        assert sparse_favoured[0].chunk_id == "s"

    def test_a_negative_weight_is_rejected(self):
        with pytest.raises(ValueError, match="weight"):
            reciprocal_rank_fusion([(_dense("a"), -0.1)])


class TestTheConstant:
    def test_a_larger_constant_flattens_the_influence_of_rank(self):
        # k trades off "ranked top by one retriever" against "found by both".
        #
        # "solo" is rank 1 in dense and absent from sparse.
        # "agreed" is rank 4 in both.
        #   k=1:    solo = 1/2    = 0.5000  >  agreed = 2/5    = 0.4000
        #   k=1000: solo = 1/1001 = 0.00100 <  agreed = 2/1004 = 0.00199
        #
        # Asserted as relative order rather than overall winner, because every
        # list necessarily has a rank-1 entry competing for the top slot.
        lists = [
            (_dense("solo", "x", "y", "agreed"), 1.0),
            (_sparse("p", "q", "r", "agreed"), 1.0),
        ]

        def order(k: int) -> list[str]:
            fused = reciprocal_rank_fusion(lists, k=k)
            return [s.chunk_id for s in fused]

        sharp = order(1)
        flat = order(1000)

        assert sharp.index("solo") < sharp.index("agreed")
        assert flat.index("agreed") < flat.index("solo")

    def test_the_constant_must_be_positive(self):
        with pytest.raises(ValueError, match="k"):
            reciprocal_rank_fusion([(_dense("a"), 1.0)], k=0)


class TestLimit:
    def test_the_result_can_be_truncated(self):
        fused = reciprocal_rank_fusion([(_dense("a", "b", "c", "d"), 1.0)], limit=2)

        assert [s.chunk_id for s in fused] == ["a", "b"]

    def test_truncation_keeps_the_highest_scoring(self):
        fused = reciprocal_rank_fusion(
            [(_dense("d1", "shared"), 1.0), (_sparse("s1", "shared"), 1.0)], limit=1
        )

        assert [s.chunk_id for s in fused] == ["shared"]

    def test_a_limit_larger_than_the_input_returns_everything(self):
        assert len(reciprocal_rank_fusion([(_dense("a", "b"), 1.0)], limit=99)) == 2

    def test_a_non_positive_limit_is_rejected(self):
        with pytest.raises(ValueError, match="limit"):
            reciprocal_rank_fusion([(_dense("a"), 1.0)], limit=0)


class TestDeterminism:
    def test_ties_are_broken_deterministically(self):
        # Two chunks at the same rank in symmetric lists have identical fused
        # scores. Arbitrary ordering would make retrieval non-reproducible and
        # any quality measurement noisy.
        lists = [(_dense("a", "b"), 1.0), (_sparse("a", "b"), 1.0)]

        first = [s.chunk_id for s in reciprocal_rank_fusion(lists)]
        second = [s.chunk_id for s in reciprocal_rank_fusion(lists)]

        assert first == second
