"""Behaviour of the remaining types that appear in port signatures."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.domain.models import (
    Chunk,
    ChunkCandidate,
    ChunkMetadata,
    ContextBlock,
    DocumentType,
    ScoredChunk,
    ScoreSource,
)
from rag.domain.prompts import PromptSpec

pytestmark = pytest.mark.unit


def _scored(chunk_id: str = "chunk-1", score: float = 0.9) -> ScoredChunk:
    metadata = ChunkMetadata(
        document_id="doc-1",
        chunk_id=chunk_id,
        ingest_version=1,
        filename="report.pdf",
        document_type=DocumentType.PDF,
        chunk_index=0,
        char_start=0,
        char_end=10,
        token_count=5,
        chunking_strategy="recursive",
        embedding_model_id="text-embedding-3-small",
        ingested_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    return ScoredChunk(
        chunk=Chunk(text="Revenue grew.", metadata=metadata),
        score=score,
        source=ScoreSource.RERANKED,
    )


class TestChunkCandidate:
    def test_carries_text_position_and_token_count(self):
        candidate = ChunkCandidate(text="Revenue grew.", char_start=0, char_end=13, token_count=4)

        assert candidate.text == "Revenue grew."
        assert candidate.token_count == 4

    def test_a_chunker_does_not_decide_identity(self):
        # Boundaries are the chunker's job; identity and provenance are stamped
        # by the pipeline. Keeping them apart is what lets a chunking strategy
        # be unit tested with no document id and no clock.
        field_names = set(ChunkCandidate.__dataclass_fields__)

        assert not field_names & {"document_id", "chunk_id", "ingest_version"}

    def test_empty_text_is_rejected(self):
        with pytest.raises(ValueError, match="text"):
            ChunkCandidate(text="  ", char_start=0, char_end=2, token_count=1)

    def test_inverted_range_is_rejected(self):
        with pytest.raises(ValueError, match="char_start"):
            ChunkCandidate(text="x", char_start=5, char_end=1, token_count=1)

    def test_negative_token_count_is_rejected(self):
        with pytest.raises(ValueError, match="token_count"):
            ChunkCandidate(text="x", char_start=0, char_end=1, token_count=-1)


class TestContextBlock:
    def test_pairs_rendered_text_with_the_chunks_it_came_from(self):
        block = ContextBlock(
            chunks=(_scored(),),
            rendered="[1] Revenue grew.",
            token_count=6,
            marker_to_chunk_id=(("1", "chunk-1"),),
        )

        assert block.rendered == "[1] Revenue grew."
        assert block.token_count == 6

    def test_resolves_a_citation_marker_back_to_its_chunk(self):
        # This mapping is what turns "[1]" in an LLM response into a Citation.
        block = ContextBlock(
            chunks=(_scored(),),
            rendered="[1] Revenue grew.",
            token_count=6,
            marker_to_chunk_id=(("1", "chunk-1"),),
        )

        assert block.chunk_id_for_marker("1") == "chunk-1"

    def test_an_unknown_marker_resolves_to_nothing(self):
        # A hallucinated marker must be dropped, never guessed at.
        block = ContextBlock(
            chunks=(_scored(),),
            rendered="[1] Revenue grew.",
            token_count=6,
            marker_to_chunk_id=(("1", "chunk-1"),),
        )

        assert block.chunk_id_for_marker("7") is None

    def test_an_empty_context_is_representable(self):
        # Retrieval legitimately returns nothing; the prompt still gets built and
        # the model is instructed to refuse.
        block = ContextBlock(chunks=(), rendered="", token_count=0, marker_to_chunk_id=())

        assert block.is_empty is True

    def test_records_how_many_chunks_the_budget_forced_out(self):
        block = ContextBlock(
            chunks=(_scored(),),
            rendered="[1] Revenue grew.",
            token_count=6,
            marker_to_chunk_id=(("1", "chunk-1"),),
            dropped_count=3,
        )

        assert block.dropped_count == 3

    def test_markers_must_map_to_chunks_that_are_present(self):
        with pytest.raises(ValueError, match="marker"):
            ContextBlock(
                chunks=(_scored("chunk-1"),),
                rendered="[1] x",
                token_count=1,
                marker_to_chunk_id=(("1", "chunk-does-not-exist"),),
            )


class TestPromptSpec:
    def test_declares_what_a_prompt_must_contain(self):
        spec = PromptSpec(version="v1", max_prompt_tokens=8192)

        assert spec.version == "v1"
        assert spec.max_prompt_tokens == 8192

    def test_grounding_and_citations_are_required_by_default(self):
        # These are business rules, not template preferences: an answer that
        # cannot be traced to a chunk is a defect.
        spec = PromptSpec(version="v1", max_prompt_tokens=8192)

        assert spec.require_citations is True
        assert spec.require_grounding is True
        assert spec.allow_refusal is True

    def test_version_is_required(self):
        with pytest.raises(ValueError, match="version"):
            PromptSpec(version="", max_prompt_tokens=8192)

    def test_token_ceiling_must_be_positive(self):
        with pytest.raises(ValueError, match="max_prompt_tokens"):
            PromptSpec(version="v1", max_prompt_tokens=0)
