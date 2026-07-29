"""Behaviour of chunk and scored-chunk value objects."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    DocumentType,
    ScoredChunk,
    ScoreSource,
)

pytestmark = pytest.mark.unit


def _metadata(**overrides: object) -> ChunkMetadata:
    defaults: dict[str, object] = {
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "ingest_version": 1,
        "filename": "report.pdf",
        "document_type": DocumentType.PDF,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 120,
        "token_count": 30,
        "chunking_strategy": "recursive",
        "embedding_model_id": "text-embedding-3-small",
        "ingested_at": datetime(2026, 7, 27, tzinfo=UTC),
    }
    return ChunkMetadata(**{**defaults, **overrides})  # type: ignore[arg-type]


def _chunk(text: str = "Revenue grew by 12 percent.", **overrides: object) -> Chunk:
    metadata = overrides.pop("metadata", None) or _metadata()
    return Chunk(text=text, metadata=metadata)  # type: ignore[arg-type]


class TestChunk:
    def test_carries_text_and_metadata(self):
        chunk = _chunk("Revenue grew.")

        assert chunk.text == "Revenue grew."
        assert chunk.metadata.document_id == "doc-1"

    def test_chunk_id_is_read_through_from_metadata(self):
        # A chunk has exactly one identity; duplicating it invites divergence.
        assert _chunk().chunk_id == "chunk-1"

    def test_document_id_is_read_through_from_metadata(self):
        assert _chunk().document_id == "doc-1"

    def test_empty_text_is_rejected(self):
        with pytest.raises(ValueError, match="text"):
            _chunk("")

    def test_whitespace_only_text_is_rejected(self):
        with pytest.raises(ValueError, match="text"):
            _chunk("   \n  ")

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _chunk().text = "changed"  # type: ignore[misc]

    def test_does_not_hold_its_vector(self):
        # Vectors are a storage concern, not part of the entity (spec 8.1).
        assert not any(
            field.name in {"vector", "embedding", "dense", "sparse"}
            for field in dataclasses.fields(Chunk)
        )


class TestScoredChunk:
    def test_carries_chunk_score_and_source(self):
        scored = ScoredChunk(chunk=_chunk(), score=0.87, source=ScoreSource.DENSE)

        assert scored.score == pytest.approx(0.87)
        assert scored.source is ScoreSource.DENSE

    def test_score_source_is_recorded_so_retrieval_stays_debuggable(self):
        # Knowing whether a chunk arrived from dense, sparse, fusion or the
        # reranker is what makes a bad result diagnosable (spec 8.1).
        assert {s.value for s in ScoreSource} == {"dense", "sparse", "fused", "reranked"}

    def test_reads_through_to_the_underlying_chunk(self):
        scored = ScoredChunk(chunk=_chunk(), score=0.5, source=ScoreSource.FUSED)

        assert scored.chunk_id == "chunk-1"
        assert scored.text == "Revenue grew by 12 percent."

    def test_is_immutable(self):
        scored = ScoredChunk(chunk=_chunk(), score=0.5, source=ScoreSource.DENSE)

        with pytest.raises(dataclasses.FrozenInstanceError):
            scored.score = 0.9  # type: ignore[misc]

    def test_rescoring_produces_a_new_object_with_the_new_source(self):
        dense = ScoredChunk(chunk=_chunk(), score=0.5, source=ScoreSource.DENSE)

        reranked = dense.rescored(0.92, ScoreSource.RERANKED)

        assert reranked.score == pytest.approx(0.92)
        assert reranked.source is ScoreSource.RERANKED
        assert reranked.chunk is dense.chunk
        assert dense.score == pytest.approx(0.5)

    def test_sorts_by_descending_score(self):
        low = ScoredChunk(chunk=_chunk(), score=0.1, source=ScoreSource.DENSE)
        high = ScoredChunk(chunk=_chunk(), score=0.9, source=ScoreSource.DENSE)

        assert sorted([low, high], reverse=True) == [high, low]
