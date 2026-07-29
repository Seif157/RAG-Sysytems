"""Behaviour of context assembly.

Turns ranked chunks into the block that goes into the prompt: budgeted, marked
up for citation, and carrying enough provenance for the model to attribute what
it says.
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

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 27, tzinfo=UTC)


def _scored(
    chunk_id: str = "chunk-1",
    *,
    text: str | None = None,
    tokens: int = 10,
    page: int | None = 4,
    filename: str = "report.pdf",
    section: str | None = "3.2 Revenue",
) -> ScoredChunk:
    # Distinct by default: passages with identical text are deduplicated, so a
    # shared default would make any multi-chunk fixture collapse to one.
    text = text if text is not None else f"Revenue for segment {chunk_id} grew by twelve percent."
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
        embedding_model_id="fake",
        ingested_at=_NOW,
        page_number=page,
        section=section,
    )
    return ScoredChunk(
        chunk=Chunk(text=text, metadata=metadata), score=0.9, source=ScoreSource.DENSE
    )


class TestMarkers:
    def test_markers_are_numbered_from_one(self):
        block = ContextBuilder(budget_tokens=100).build((_scored("a"), _scored("b")))

        assert block.marker_to_chunk_id == (("1", "a"), ("2", "b"))

    def test_every_marker_resolves_back_to_its_chunk(self):
        block = ContextBuilder(budget_tokens=100).build((_scored("a"), _scored("b")))

        assert block.chunk_id_for_marker("1") == "a"
        assert block.chunk_id_for_marker("2") == "b"

    def test_markers_appear_in_the_rendered_text(self):
        block = ContextBuilder(budget_tokens=100).build((_scored("a"),))

        assert "[1]" in block.rendered


class TestRendering:
    def test_the_chunk_text_is_included(self):
        block = ContextBuilder(budget_tokens=100).build((_scored("a", text="Revenue grew."),))

        assert "Revenue grew." in block.rendered

    def test_provenance_is_shown_so_the_model_can_attribute_accurately(self):
        block = ContextBuilder(budget_tokens=100).build(
            (_scored("a", filename="annual.pdf", page=14, section="3.2 Revenue"),)
        )

        assert "annual.pdf" in block.rendered
        assert "14" in block.rendered
        assert "3.2 Revenue" in block.rendered

    def test_absent_provenance_is_omitted_rather_than_rendered_as_none(self):
        # "page None" in a prompt is noise the model may repeat back.
        block = ContextBuilder(budget_tokens=100).build((_scored("a", page=None, section=None),))

        assert "None" not in block.rendered


class TestBudget:
    def test_chunks_beyond_the_budget_are_dropped_and_counted(self):
        chunks = (_scored("a", tokens=60), _scored("b", tokens=60))

        block = ContextBuilder(budget_tokens=100).build(chunks)

        assert len(block.chunks) == 1
        assert block.dropped_count == 1

    def test_the_reported_token_count_measures_the_rendered_block(self):
        # Not the sum of chunk token counts: what reaches the model includes a
        # citation marker and a provenance header per passage.
        block = ContextBuilder(budget_tokens=1000, count_tokens=len).build(
            (_scored("a", tokens=10), _scored("b", tokens=20))
        )

        assert block.token_count == len(block.rendered)
        assert block.token_count > 30

    def test_markers_stay_contiguous_after_a_chunk_is_dropped(self):
        # A gap in the numbering would leave the model citing [3] when only two
        # passages were supplied.
        #
        # The budget allows for the provenance header each passage carries, so
        # that what this exercises is the numbering rather than the arithmetic.
        chunks = (_scored("a", tokens=50), _scored("b", tokens=500), _scored("c", tokens=40))

        block = ContextBuilder(budget_tokens=120).build(chunks)

        assert [marker for marker, _ in block.marker_to_chunk_id] == ["1", "2"]
        assert [chunk_id for _, chunk_id in block.marker_to_chunk_id] == ["a", "c"]


class TestEmptyContext:
    def test_no_chunks_produces_an_empty_block(self):
        # Retrieval legitimately finds nothing; the prompt is still built and
        # the model is told to say so rather than guess.
        block = ContextBuilder(budget_tokens=100).build(())

        assert block.is_empty is True
        assert block.rendered == ""
        assert block.marker_to_chunk_id == ()
