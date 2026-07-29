"""Behaviour of the assembled context block.

The policies are tested individually; this covers what the builder does with
them together, and the properties that only exist once they are composed --
markers staying contiguous after reordering, and the marker map still resolving
to the right passage after everything has moved.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.application.services import ContextBuilder
from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    DocumentType,
    ScoredChunk,
    ScoreSource,
)
from rag.domain.policies import assemble_citations

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _scored(
    chunk_id: str,
    *,
    text: str | None = None,
    score: float = 0.9,
    tokens: int = 10,
    document: str = "doc-1",
    filename: str | None = None,
    index: int = 0,
    page: int | None = None,
    section: str | None = None,
) -> ScoredChunk:
    # Distinct by default: identical text is deduplicated, which would silently
    # invalidate any fixture relying on two passages surviving.
    text = text if text is not None else f"Distinct passage {chunk_id} about revenue and costs."
    return ScoredChunk(
        chunk=Chunk(
            text=text,
            metadata=ChunkMetadata(
                document_id=document,
                chunk_id=chunk_id,
                ingest_version=1,
                filename=filename or f"{document}.pdf",
                document_type=DocumentType.PDF,
                chunk_index=index,
                char_start=0,
                char_end=len(text),
                token_count=tokens,
                chunking_strategy="recursive",
                embedding_model_id="fake",
                ingested_at=_NOW,
                page_number=page,
                section=section,
            ),
        ),
        score=score,
        source=ScoreSource.RERANKED,
    )


class TestOrderingIsApplied:
    def test_the_strongest_passage_appears_first_in_the_rendered_block(self):
        chunks = (
            _scored("mid", score=0.5, document="b", text="Middling passage here."),
            _scored("best", score=0.9, document="a", text="Strongest passage here."),
            _scored("weak", score=0.1, document="c", text="Weakest passage here."),
        )

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        assert block.rendered.index("Strongest") < block.rendered.index("Weakest")

    def test_the_second_strongest_is_placed_last(self):
        chunks = (
            _scored("best", score=0.9, document="a", text="Strongest passage."),
            _scored("second", score=0.8, document="b", text="Second passage."),
            _scored("weak", score=0.1, document="c", text="Weakest passage."),
        )

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        assert block.chunks[0].chunk_id == "best"
        assert block.chunks[-1].chunk_id == "second"

    def test_passages_from_one_document_stay_together(self):
        chunks = (
            _scored("a1", score=0.9, document="alpha", index=0),
            _scored("b1", score=0.8, document="beta", index=0),
            _scored("a2", score=0.7, document="alpha", index=1),
        )

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        documents = [c.chunk.metadata.document_id for c in block.chunks]
        assert documents in (["alpha", "alpha", "beta"], ["beta", "alpha", "alpha"])


class TestDeduplicationIsApplied:
    def test_a_repeated_passage_appears_once(self):
        chunks = (
            _scored("a", text="This document is confidential.", document="d1"),
            _scored("b", text="This document is confidential.", document="d2"),
        )

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        assert len(block.chunks) == 1
        assert block.rendered.count("confidential") == 1

    def test_deduplication_happens_before_budgeting(self):
        # Otherwise a duplicate consumes budget a distinct passage needed, and
        # the distinct one is the thing that gets dropped.
        distinct = "Employees receive twenty-five days of paid annual leave."
        chunks = (
            _scored("dup1", text="Boilerplate confidentiality notice.", tokens=40),
            _scored("dup2", text="Boilerplate confidentiality notice.", tokens=40),
            _scored("real", text=distinct, tokens=40, document="d2"),
        )

        block = ContextBuilder(budget_tokens=100).build(chunks)

        assert any("annual leave" in c.text for c in block.chunks)

    def test_duplicates_are_not_counted_as_budget_drops(self):
        chunks = (_scored("a", text="Same text here."), _scored("b", text="Same text here."))

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        assert block.dropped_count == 0


class TestMarkersSurviveEverything:
    def test_markers_are_contiguous_from_one(self):
        chunks = tuple(
            _scored(f"c{i}", score=1.0 - i * 0.1, document=f"d{i}", text=f"Passage {i}.")
            for i in range(5)
        )

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        assert [m for m, _ in block.marker_to_chunk_id] == ["1", "2", "3", "4", "5"]

    def test_markers_follow_presentation_order_not_relevance(self):
        # The model reads top to bottom, so [1] must be the first thing it sees.
        chunks = (
            _scored("best", score=0.9, document="a", text="Strongest."),
            _scored("second", score=0.8, document="b", text="Second."),
        )

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        assert block.chunk_id_for_marker("1") == block.chunks[0].chunk_id

    def test_every_marker_resolves_to_a_passage_that_is_present(self):
        chunks = tuple(
            _scored(f"c{i}", score=1.0 - i * 0.1, document=f"d{i}", text=f"Passage {i}.")
            for i in range(4)
        )

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        present = {c.chunk_id for c in block.chunks}
        assert all(chunk_id in present for _, chunk_id in block.marker_to_chunk_id)

    def test_a_citation_resolves_to_the_passage_actually_rendered(self):
        # The end-to-end property: whatever the model cites must be the text it
        # was shown under that marker, after every reorder and drop.
        chunks = (
            _scored("best", score=0.9, document="a", text="Alpha content.", page=4),
            _scored("second", score=0.8, document="b", text="Beta content.", page=7),
        )

        block = ContextBuilder(budget_tokens=1000).build(chunks)
        citations = assemble_citations("As shown [2].", block)

        cited = next(c for c in block.chunks if c.chunk_id == citations[0].chunk_id)
        assert cited.chunk_id == block.chunks[1].chunk_id
        assert citations[0].page_number == cited.chunk.metadata.page_number


class TestBudgetAccountsForRendering:
    def test_the_rendered_header_is_charged_against_the_budget(self):
        # What reaches the model is the rendered block, not the bare text.
        # Counting text alone silently overruns the prompt ceiling.
        chunks = (
            _scored("a", tokens=40, filename="a-very-long-filename-indeed.pdf", page=14),
            _scored("b", tokens=40, filename="another-long-filename.pdf", page=15, document="d2"),
        )

        generous = ContextBuilder(budget_tokens=200, count_tokens=len).build(chunks)
        tight = ContextBuilder(budget_tokens=60, count_tokens=len).build(chunks)

        assert len(generous.chunks) == 2
        assert len(tight.chunks) < 2

    def test_the_reported_token_count_reflects_what_was_rendered(self):
        chunks = (_scored("a", tokens=10, page=4),)

        block = ContextBuilder(budget_tokens=1000, count_tokens=len).build(chunks)

        assert block.token_count == len(block.rendered)

    def test_an_empty_block_reports_no_tokens(self):
        block = ContextBuilder(budget_tokens=1000).build(())

        assert block.token_count == 0
        assert block.is_empty is True


class TestProvenanceIsPreserved:
    def test_page_and_section_reach_the_rendered_block(self):
        chunks = (_scored("a", page=14, section="3.2 Revenue", filename="annual.pdf"),)

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        assert "annual.pdf" in block.rendered
        assert "14" in block.rendered
        assert "3.2 Revenue" in block.rendered

    def test_absent_provenance_is_omitted_not_rendered_as_none(self):
        chunks = (_scored("a", page=None, section=None),)

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        assert "None" not in block.rendered

    def test_chunk_metadata_is_unchanged_by_assembly(self):
        chunks = (_scored("a", page=14, section="3.2 Revenue"),)

        block = ContextBuilder(budget_tokens=1000).build(chunks)

        assert block.chunks[0].chunk.metadata.page_number == 14
        assert block.chunks[0].chunk.metadata.section == "3.2 Revenue"
