"""Behaviour of the pure domain policies.

No I/O, no clock, no configuration -- which makes these the cheapest and most
valuable tests in the codebase. Between them they decide chunk identity (and so
whether re-ingestion is idempotent), how much context reaches the model, and
whether an answer's citations point where they claim to.
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
from rag.domain.policies import (
    assemble_citations,
    derive_chunk_id,
    extract_citation_markers,
    select_within_budget,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 27, tzinfo=UTC)


def _scored(
    chunk_id: str = "chunk-1",
    *,
    text: str = "Revenue grew.",
    tokens: int = 10,
    score: float = 0.9,
    page: int | None = 4,
    filename: str = "report.pdf",
) -> ScoredChunk:
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        ingest_version=1,
        filename=filename,
        document_type=DocumentType.PDF,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        token_count=tokens,
        chunking_strategy="recursive",
        embedding_model_id="text-embedding-3-small",
        ingested_at=_NOW,
        page_number=page,
        section="3.2 Revenue",
    )
    return ScoredChunk(
        chunk=Chunk(text=text, metadata=metadata), score=score, source=ScoreSource.DENSE
    )


class TestChunkIdentity:
    def test_the_same_inputs_always_produce_the_same_id(self):
        # This is what makes re-ingestion idempotent: a retried run overwrites
        # rather than duplicating (ADR-014).
        first = derive_chunk_id("doc-1", 0, "Revenue grew.")
        second = derive_chunk_id("doc-1", 0, "Revenue grew.")

        assert first == second

    def test_different_text_produces_a_different_id(self):
        assert derive_chunk_id("doc-1", 0, "a") != derive_chunk_id("doc-1", 0, "b")

    def test_different_position_produces_a_different_id(self):
        # Two identical paragraphs in one document must not collide.
        assert derive_chunk_id("doc-1", 0, "same") != derive_chunk_id("doc-1", 1, "same")

    def test_different_documents_produce_different_ids(self):
        assert derive_chunk_id("doc-1", 0, "same") != derive_chunk_id("doc-2", 0, "same")

    def test_the_id_is_a_stable_hex_string(self):
        chunk_id = derive_chunk_id("doc-1", 0, "Revenue grew.")

        assert len(chunk_id) == 32
        assert all(character in "0123456789abcdef" for character in chunk_id)

    def test_the_separator_cannot_be_forged_by_content(self):
        # Naive concatenation lets ("doc-1|0", "text") and ("doc-1", "0|text")
        # collide. Length-prefixing or hashing the parts prevents it.
        assert derive_chunk_id("doc-1|0", 1, "x") != derive_chunk_id("doc-1", 0, "1|x")


class TestContextBudget:
    def test_nothing_selected_from_nothing(self):
        selected, dropped = select_within_budget((), budget_tokens=100)

        assert selected == ()
        assert dropped == 0

    def test_everything_fits_within_a_generous_budget(self):
        chunks = (_scored("a", tokens=10), _scored("b", tokens=20))

        selected, dropped = select_within_budget(chunks, budget_tokens=100)

        assert len(selected) == 2
        assert dropped == 0

    def test_rank_order_is_preserved(self):
        chunks = (_scored("a", score=0.9), _scored("b", score=0.5))

        selected, _ = select_within_budget(chunks, budget_tokens=100)

        assert [s.chunk_id for s in selected] == ["a", "b"]

    def test_chunks_that_do_not_fit_are_dropped_and_counted(self):
        chunks = (_scored("a", tokens=60), _scored("b", tokens=60))

        selected, dropped = select_within_budget(chunks, budget_tokens=100)

        assert [s.chunk_id for s in selected] == ["a"]
        assert dropped == 1

    def test_a_lower_ranked_chunk_can_still_fit_after_a_large_one_is_skipped(self):
        # Greedy fill rather than stop-on-first-miss: the budget is there to be
        # used, and a small highly-relevant chunk should not be lost because a
        # bulky one preceded it.
        chunks = (_scored("a", tokens=50), _scored("b", tokens=80), _scored("c", tokens=40))

        selected, dropped = select_within_budget(chunks, budget_tokens=100)

        assert [s.chunk_id for s in selected] == ["a", "c"]
        assert dropped == 1

    def test_duplicate_chunks_are_included_once(self):
        # Hybrid retrieval returns the same chunk from both lists; paying for it
        # twice wastes budget and biases the model by repetition.
        chunks = (_scored("a", tokens=10), _scored("a", tokens=10))

        selected, dropped = select_within_budget(chunks, budget_tokens=100)

        assert len(selected) == 1
        assert dropped == 0

    def test_a_zero_budget_selects_nothing(self):
        selected, dropped = select_within_budget((_scored("a", tokens=10),), budget_tokens=0)

        assert selected == ()
        assert dropped == 1

    def test_a_negative_budget_is_rejected(self):
        with pytest.raises(ValueError, match="budget_tokens"):
            select_within_budget((), budget_tokens=-1)


class TestCitationMarkerExtraction:
    def test_no_markers_in_plain_prose(self):
        assert extract_citation_markers("Revenue grew by 12 percent.") == ()

    def test_a_single_marker_is_found(self):
        assert extract_citation_markers("Revenue grew [1].") == ("1",)

    def test_several_markers_are_found_in_order(self):
        assert extract_citation_markers("A [2] and B [1].") == ("2", "1")

    def test_a_repeated_marker_appears_once(self):
        assert extract_citation_markers("A [1] and B [1].") == ("1",)

    def test_adjacent_markers_are_separated(self):
        assert extract_citation_markers("Both [1][2] agree.") == ("1", "2")

    def test_bracketed_prose_is_not_a_marker(self):
        # Otherwise "[see appendix]" would become a citation.
        assert extract_citation_markers("As noted [see appendix].") == ()


class TestCitationAssembly:
    def _context(self, *chunks: ScoredChunk) -> ContextBlock:
        markers = tuple((str(index), c.chunk_id) for index, c in enumerate(chunks, start=1))
        return ContextBlock(
            chunks=chunks,
            rendered="\n".join(
                f"[{m}] {c.text}" for (m, _), c in zip(markers, chunks, strict=True)
            ),
            token_count=sum(c.chunk.metadata.token_count for c in chunks),
            marker_to_chunk_id=markers,
        )

    def test_an_answer_with_no_markers_has_no_citations(self):
        context = self._context(_scored("a"))

        assert assemble_citations("Revenue grew.", context) == ()

    def test_a_cited_chunk_becomes_a_citation(self):
        context = self._context(_scored("a", filename="report.pdf", page=4))

        citations = assemble_citations("Revenue grew [1].", context)

        assert len(citations) == 1
        assert citations[0].chunk_id == "a"
        assert citations[0].filename == "report.pdf"
        assert citations[0].page_number == 4

    def test_an_uncited_chunk_produces_no_citation(self):
        # Citations describe what the answer used, not what retrieval found.
        context = self._context(_scored("a"), _scored("b"))

        citations = assemble_citations("Revenue grew [1].", context)

        assert [c.chunk_id for c in citations] == ["a"]

    def test_a_hallucinated_marker_is_dropped(self):
        # A model that invents [7] when one chunk was supplied must not produce
        # a citation pointing at an arbitrary chunk.
        context = self._context(_scored("a"))

        assert assemble_citations("Revenue grew [7].", context) == ()

    def test_citations_appear_in_the_order_they_were_first_cited(self):
        # Different pages, so the two stay separate rather than merging into one
        # citation for a shared location.
        context = self._context(_scored("a", page=4), _scored("b", page=9))

        citations = assemble_citations("B says [2], and A agrees [1].", context)

        assert [c.chunk_id for c in citations] == ["b", "a"]

    def test_a_repeated_citation_appears_once(self):
        context = self._context(_scored("a"))

        citations = assemble_citations("Revenue grew [1], as stated [1].", context)

        assert len(citations) == 1

    def test_the_relevance_score_is_carried_when_it_was_reranked(self):
        chunk = _scored("a", score=0.87)
        reranked = chunk.rescored(0.87, ScoreSource.RERANKED)
        context = self._context(reranked)

        citations = assemble_citations("Revenue grew [1].", context)

        assert citations[0].relevance_score == pytest.approx(0.87)

    def test_no_relevance_score_is_invented_when_nothing_reranked(self):
        # A raw cosine similarity is not a confidence, and presenting it as one
        # invites trust it has not earned.
        context = self._context(_scored("a", score=0.42))

        citations = assemble_citations("Revenue grew [1].", context)

        assert citations[0].relevance_score is None

    def test_a_snippet_of_the_source_is_included(self):
        long_text = "Revenue grew by twelve percent across every region. " * 10
        context = self._context(_scored("a", text=long_text))

        citations = assemble_citations("Revenue grew [1].", context)

        assert citations[0].snippet is not None
        assert len(citations[0].snippet) < len(long_text)
