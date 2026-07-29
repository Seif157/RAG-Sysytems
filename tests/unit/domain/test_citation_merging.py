"""Merging citations that point at the same place.

Retrieval routinely returns several chunks from one page -- chunk overlap alone
guarantees it. Rendered naively that produces "report.pdf · page 4" three times
in the sources list, which reads as three independent confirmations when it is
one passage split three ways.

Merging is by *locator*, not by document: two chunks on page 4 under different
headings are genuinely different places and stay separate.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    ContextBlock,
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
    document: str = "doc-1",
    filename: str = "report.pdf",
    page: int | None = 4,
    section: str | None = "3.2 Revenue",
    score: float = 0.8,
    source: ScoreSource = ScoreSource.RERANKED,
) -> ScoredChunk:
    text = text or f"Passage {chunk_id} discussing revenue in some detail."
    return ScoredChunk(
        chunk=Chunk(
            text=text,
            metadata=ChunkMetadata(
                document_id=document,
                chunk_id=chunk_id,
                ingest_version=1,
                filename=filename,
                document_type=DocumentType.PDF,
                chunk_index=0,
                char_start=0,
                char_end=len(text),
                token_count=10,
                chunking_strategy="recursive",
                embedding_model_id="fake",
                ingested_at=_NOW,
                page_number=page,
                section=section,
            ),
        ),
        score=score,
        source=source,
    )


def _context(*chunks: ScoredChunk) -> ContextBlock:
    markers = tuple((str(i), c.chunk_id) for i, c in enumerate(chunks, start=1))
    return ContextBlock(
        chunks=chunks,
        rendered="\n".join(f"[{m}] {c.text}" for (m, _), c in zip(markers, chunks, strict=True)),
        token_count=sum(c.chunk.metadata.token_count for c in chunks),
        marker_to_chunk_id=markers,
    )


class TestMerging:
    def test_two_chunks_from_the_same_page_become_one_citation(self):
        context = _context(_scored("a", page=4), _scored("b", page=4))

        citations = assemble_citations("Revenue grew [1][2].", context)

        assert len(citations) == 1
        assert citations[0].page_number == 4

    def test_the_merged_citation_records_every_supporting_chunk(self):
        # The user sees one source; the system still knows which passages
        # supported it, which is what makes the citation auditable.
        context = _context(_scored("a", page=4), _scored("b", page=4))

        citations = assemble_citations("Revenue grew [1][2].", context)

        assert set(citations[0].chunk_ids) == {"a", "b"}

    def test_chunks_from_different_pages_stay_separate(self):
        context = _context(_scored("a", page=4), _scored("b", page=9))

        citations = assemble_citations("Revenue grew [1][2].", context)

        assert {c.page_number for c in citations} == {4, 9}

    def test_chunks_from_different_documents_stay_separate(self):
        context = _context(
            _scored("a", document="d1", filename="alpha.pdf"),
            _scored("b", document="d2", filename="beta.pdf"),
        )

        citations = assemble_citations("Both agree [1][2].", context)

        assert {c.filename for c in citations} == {"alpha.pdf", "beta.pdf"}

    def test_the_same_page_under_different_headings_stays_separate(self):
        # A page can hold the end of one section and the start of the next.
        # Collapsing them would send a reader to the wrong half of the page.
        context = _context(
            _scored("a", page=4, section="3.2 Revenue"),
            _scored("b", page=4, section="3.3 Costs"),
        )

        citations = assemble_citations("Both [1][2].", context)

        assert len(citations) == 2

    def test_pageless_formats_merge_on_their_section(self):
        # DOCX and Markdown have no pages; the heading path is the locator.
        context = _context(
            _scored("a", page=None, section="Handbook > Leave"),
            _scored("b", page=None, section="Handbook > Leave"),
        )

        citations = assemble_citations("As stated [1][2].", context)

        assert len(citations) == 1

    def test_a_document_with_no_locator_at_all_still_merges_by_document(self):
        # Plain text: no page, no heading. Two chunks from one file are one
        # source as far as a reader is concerned.
        context = _context(
            _scored("a", page=None, section=None),
            _scored("b", page=None, section=None),
        )

        citations = assemble_citations("As stated [1][2].", context)

        assert len(citations) == 1


class TestMergedScores:
    def test_the_strongest_supporting_score_is_kept(self):
        # The merged citation is as good as its best evidence, not its average.
        context = _context(
            _scored("weak", page=4, score=0.30),
            _scored("strong", page=4, score=0.91),
        )

        citations = assemble_citations("Revenue grew [1][2].", context)

        assert citations[0].relevance_score == pytest.approx(0.91)

    def test_the_snippet_comes_from_the_strongest_chunk(self):
        context = _context(
            _scored("weak", page=4, score=0.30, text="A weaker aside about revenue."),
            _scored("strong", page=4, score=0.91, text="The decisive statement about revenue."),
        )

        citations = assemble_citations("Revenue grew [1][2].", context)

        assert "decisive" in (citations[0].snippet or "")

    def test_no_score_is_invented_when_nothing_was_reranked(self):
        context = _context(
            _scored("a", page=4, source=ScoreSource.FUSED),
            _scored("b", page=4, source=ScoreSource.FUSED),
        )

        citations = assemble_citations("Revenue grew [1][2].", context)

        assert citations[0].relevance_score is None

    def test_a_reranked_chunk_supplies_the_score_even_if_a_peer_was_not(self):
        context = _context(
            _scored("fused", page=4, source=ScoreSource.FUSED, score=0.99),
            _scored("ranked", page=4, source=ScoreSource.RERANKED, score=0.62),
        )

        citations = assemble_citations("Revenue grew [1][2].", context)

        assert citations[0].relevance_score == pytest.approx(0.62)


class TestOrdering:
    def test_citations_appear_in_the_order_first_cited(self):
        context = _context(
            _scored("a", page=4, section="A"),
            _scored("b", page=9, section="B"),
        )

        citations = assemble_citations("First [2], then [1].", context)

        assert [c.page_number for c in citations] == [9, 4]

    def test_a_merged_citation_takes_the_position_of_its_earliest_mention(self):
        context = _context(
            _scored("a", page=4, section="A"),
            _scored("b", page=9, section="B"),
            _scored("c", page=4, section="A"),
        )

        citations = assemble_citations("See [1], and [2], and also [3].", context)

        assert [c.page_number for c in citations] == [4, 9]


class TestUnchangedBehaviour:
    def test_uncited_passages_produce_no_citation(self):
        context = _context(_scored("a", page=4, section="A"), _scored("b", page=9, section="B"))

        citations = assemble_citations("Only the first [1].", context)

        assert len(citations) == 1

    def test_a_hallucinated_marker_is_still_dropped(self):
        context = _context(_scored("a"))

        assert assemble_citations("Revenue grew [7].", context) == ()

    def test_chunk_id_still_reads_as_the_primary_supporting_chunk(self):
        context = _context(_scored("weak", page=4, score=0.3), _scored("strong", page=4, score=0.9))

        citations = assemble_citations("Revenue grew [1][2].", context)

        assert citations[0].chunk_id == "strong"
