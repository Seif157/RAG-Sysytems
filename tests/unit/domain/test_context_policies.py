"""Behaviour of the context assembly policies.

Three pure decisions that together determine what the model actually reads:
which passages survive deduplication, how many fit the budget, and where in the
context each one lands. All three are cheap to test exhaustively and expensive
to get wrong -- a mistake here degrades every answer without failing anything.
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
from rag.domain.policies import (
    deduplicate_chunks,
    order_for_attention,
    select_within_budget,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _scored(
    chunk_id: str,
    *,
    text: str = "Revenue grew by twelve percent across every region.",
    score: float = 0.9,
    tokens: int = 10,
    document: str = "doc-1",
    index: int = 0,
    page: int | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            text=text,
            metadata=ChunkMetadata(
                document_id=document,
                chunk_id=chunk_id,
                ingest_version=1,
                filename=f"{document}.pdf",
                document_type=DocumentType.PDF,
                chunk_index=index,
                char_start=0,
                char_end=len(text),
                token_count=tokens,
                chunking_strategy="recursive",
                embedding_model_id="fake",
                ingested_at=_NOW,
                page_number=page,
            ),
        ),
        score=score,
        source=ScoreSource.RERANKED,
    )


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #
class TestDeduplication:
    def test_nothing_to_deduplicate(self):
        assert deduplicate_chunks(()) == ()

    def test_distinct_passages_all_survive(self):
        chunks = (
            _scored("a", text="Revenue grew by twelve percent."),
            _scored("b", text="The cafeteria serves lunch at noon."),
        )

        assert len(deduplicate_chunks(chunks)) == 2

    def test_the_same_chunk_retrieved_twice_appears_once(self):
        # Hybrid retrieval returns the same chunk from both lists.
        chunks = (_scored("a"), _scored("a"))

        assert len(deduplicate_chunks(chunks)) == 1

    def test_identical_text_under_different_ids_is_deduplicated(self):
        # Boilerplate repeated across two documents. Paying for it twice wastes
        # budget a genuinely different passage needed.
        chunks = (
            _scored("a", text="This document is confidential.", document="doc-1"),
            _scored("b", text="This document is confidential.", document="doc-2"),
        )

        assert len(deduplicate_chunks(chunks)) == 1

    def test_the_higher_ranked_duplicate_is_the_one_kept(self):
        chunks = (
            _scored("best", text="Same words here entirely.", score=0.9),
            _scored("worse", text="Same words here entirely.", score=0.2),
        )

        assert [c.chunk_id for c in deduplicate_chunks(chunks)] == ["best"]

    def test_near_duplicates_are_removed(self):
        chunks = (
            _scored("a", text="Employees receive twenty-five days of paid annual leave."),
            _scored("b", text="Employees receive twenty-five days of paid annual leave!"),
        )

        assert len(deduplicate_chunks(chunks)) == 1

    def test_merely_similar_passages_both_survive(self):
        # Two passages about leave that say different things must both reach the
        # model; over-aggressive deduplication silently discards evidence.
        chunks = (
            _scored("a", text="Employees receive twenty-five days of paid annual leave."),
            _scored("b", text="Leave requests must be submitted two weeks in advance."),
        )

        assert len(deduplicate_chunks(chunks)) == 2

    def test_the_threshold_is_adjustable(self):
        # Six words each, five shared: Jaccard = 5/7 = 0.71.
        chunks = (
            _scored("a", text="alpha beta gamma delta epsilon zeta"),
            _scored("b", text="alpha beta gamma delta epsilon eta"),
        )

        assert len(deduplicate_chunks(chunks, threshold=0.95)) == 2
        assert len(deduplicate_chunks(chunks, threshold=0.5)) == 1

    def test_very_short_passages_are_compared_only_by_containment(self):
        # Jaccard is too coarse below a handful of words: two short sentences
        # sharing a couple of common words look identical to it, and dropping a
        # distinct passage loses evidence with no symptom.
        chunks = (
            _scored("a", text="revenue rose sharply"),
            _scored("b", text="revenue fell sharply"),
        )

        assert len(deduplicate_chunks(chunks, threshold=0.5)) == 2

    def test_a_passage_wholly_contained_in_another_is_removed(self):
        # Chunk overlap can surface a fragment alongside the chunk containing it.
        chunks = (
            _scored("long", text="Employees receive twenty-five days of paid annual leave."),
            _scored("short", text="twenty-five days of paid annual leave"),
        )

        assert [c.chunk_id for c in deduplicate_chunks(chunks)] == ["long"]

    def test_input_order_is_otherwise_preserved(self):
        chunks = (_scored("a", text="one"), _scored("b", text="two"), _scored("c", text="three"))

        assert [c.chunk_id for c in deduplicate_chunks(chunks)] == ["a", "b", "c"]

    def test_case_and_punctuation_do_not_defeat_it(self):
        chunks = (
            _scored("a", text="Revenue grew by twelve percent."),
            _scored("b", text="REVENUE GREW BY TWELVE PERCENT"),
        )

        assert len(deduplicate_chunks(chunks)) == 1

    def test_an_invalid_threshold_is_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            deduplicate_chunks((), threshold=1.5)


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #
class TestOrderingForAttention:
    def test_nothing_to_order(self):
        assert order_for_attention(()) == ()

    def test_a_single_passage_is_returned_unchanged(self):
        assert [c.chunk_id for c in order_for_attention((_scored("a"),))] == ["a"]

    def test_the_strongest_passage_comes_first(self):
        chunks = (
            _scored("best", score=0.9, document="d1"),
            _scored("mid", score=0.5, document="d2"),
            _scored("worst", score=0.1, document="d3"),
        )

        assert order_for_attention(chunks)[0].chunk_id == "best"

    def test_the_second_strongest_goes_last(self):
        # The "lost in the middle" mitigation: models attend reliably to the
        # start and end of a context and less so to the middle, so the two
        # strongest passages take the two most-read positions.
        chunks = (
            _scored("best", score=0.9, document="d1"),
            _scored("second", score=0.8, document="d2"),
            _scored("third", score=0.1, document="d3"),
        )

        ordered = order_for_attention(chunks)

        assert ordered[0].chunk_id == "best"
        assert ordered[-1].chunk_id == "second"

    def test_the_weakest_material_ends_up_in_the_middle(self):
        chunks = tuple(_scored(f"c{i}", score=1.0 - i * 0.1, document=f"d{i}") for i in range(5))

        ordered = order_for_attention(chunks)

        assert ordered[len(ordered) // 2].chunk_id == "c4"

    def test_every_passage_survives_reordering(self):
        chunks = tuple(_scored(f"c{i}", score=1.0 - i * 0.1, document=f"d{i}") for i in range(6))

        ordered = order_for_attention(chunks)

        assert {c.chunk_id for c in ordered} == {c.chunk_id for c in chunks}
        assert len(ordered) == len(chunks)

    def test_ordering_is_deterministic(self):
        chunks = tuple(_scored(f"c{i}", score=0.5, document=f"d{i}") for i in range(4))

        assert [c.chunk_id for c in order_for_attention(chunks)] == [
            c.chunk_id for c in order_for_attention(chunks)
        ]


class TestDocumentBoundaries:
    def test_passages_from_one_document_stay_together(self):
        # Interleaving sources fragments each one and makes the model work
        # harder to follow a single argument.
        chunks = (
            _scored("a1", score=0.9, document="alpha", index=0),
            _scored("b1", score=0.8, document="beta", index=0),
            _scored("a2", score=0.7, document="alpha", index=1),
        )

        documents = [c.chunk.metadata.document_id for c in order_for_attention(chunks)]

        assert documents in (["alpha", "alpha", "beta"], ["beta", "alpha", "alpha"])

    def test_document_runs_are_contiguous(self):
        chunks = (
            _scored("a1", score=0.9, document="alpha", index=0),
            _scored("b1", score=0.8, document="beta", index=0),
            _scored("a2", score=0.7, document="alpha", index=1),
            _scored("b2", score=0.6, document="beta", index=1),
        )

        documents = [c.chunk.metadata.document_id for c in order_for_attention(chunks)]

        # Each document appears as one unbroken run.
        import itertools

        runs = [document for document, _ in itertools.groupby(documents)]
        assert len(runs) == len(set(runs))

    def test_passages_within_a_document_are_in_reading_order(self):
        # A later section appearing before an earlier one reads as incoherent
        # even when both are relevant.
        chunks = (
            _scored("third", score=0.9, document="alpha", index=2),
            _scored("first", score=0.5, document="alpha", index=0),
            _scored("second", score=0.7, document="alpha", index=1),
        )

        ordered = order_for_attention(chunks)

        assert [c.chunk_id for c in ordered] == ["first", "second", "third"]

    def test_a_document_is_placed_by_its_strongest_passage(self):
        chunks = (
            _scored("weak_doc", score=0.4, document="beta", index=0),
            _scored("strong", score=0.95, document="alpha", index=0),
            _scored("also_alpha", score=0.1, document="alpha", index=1),
        )

        ordered = order_for_attention(chunks)

        assert ordered[0].chunk.metadata.document_id == "alpha"

    def test_pages_within_a_document_stay_in_order(self):
        chunks = (
            _scored("p9", score=0.9, document="alpha", index=8, page=9),
            _scored("p2", score=0.8, document="alpha", index=1, page=2),
        )

        ordered = order_for_attention(chunks)

        assert [c.chunk.metadata.page_number for c in ordered] == [2, 9]


# --------------------------------------------------------------------------- #
# Budgeting with a cost function
# --------------------------------------------------------------------------- #
class TestBudgetCost:
    def test_the_default_cost_is_the_chunk_token_count(self):
        chunks = (_scored("a", tokens=60), _scored("b", tokens=60))

        selected, dropped = select_within_budget(chunks, budget_tokens=100)

        assert len(selected) == 1
        assert dropped == 1

    def test_a_cost_function_can_account_for_rendering_overhead(self):
        # What reaches the model is the rendered block -- marker, filename, page
        # -- not the bare chunk text. Budgeting the text alone consistently
        # understates the real prompt size.
        chunks = (_scored("a", tokens=40), _scored("b", tokens=40))

        selected, dropped = select_within_budget(
            chunks, budget_tokens=100, cost=lambda c: c.chunk.metadata.token_count + 20
        )

        assert len(selected) == 1
        assert dropped == 1

    def test_a_negative_cost_is_rejected(self):
        with pytest.raises(ValueError, match="cost"):
            select_within_budget((_scored("a"),), budget_tokens=100, cost=lambda c: -1)
