"""Chunks: the unit of retrieval.

A :class:`Chunk` is a passage of text plus the metadata describing where it came
from. It deliberately does **not** hold its embedding: vectors are a storage
concern, they differ per embedding model, and keeping them off the entity is
what allows a chunk to be reasoned about without a vector store present.

A :class:`ScoredChunk` pairs a chunk with a relevance score *and the stage that
produced it*. Carrying the source is what makes a disappointing result
diagnosable: it distinguishes "dense search ranked this highly" from "the
reranker did".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Self

from rag.domain.models.metadata import ChunkMetadata

__all__ = ["Chunk", "ScoreSource", "ScoredChunk"]


class ScoreSource(StrEnum):
    """The retrieval stage that assigned a score.

    Attributes:
        DENSE: Vector similarity search.
        SPARSE: Lexical (BM25) search.
        FUSED: Reciprocal rank fusion of dense and sparse results.
        RERANKED: Cross-encoder reranking.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    FUSED = "fused"
    RERANKED = "reranked"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage of a document.

    Attributes:
        text: The passage itself.
        metadata: Provenance and filterable attributes.
    """

    text: str
    metadata: ChunkMetadata

    def __post_init__(self) -> None:
        """Reject an empty chunk, which can never contribute to an answer."""
        if not self.text or not self.text.strip():
            raise ValueError("chunk text must be a non-empty string")

    @property
    def chunk_id(self) -> str:
        """The chunk's deterministic identifier, read through from metadata."""
        return self.metadata.chunk_id

    @property
    def document_id(self) -> str:
        """Identifier of the document this chunk came from."""
        return self.metadata.document_id


@dataclass(frozen=True, order=False, slots=True)
class ScoredChunk:
    """A chunk with a relevance score and the stage that produced it.

    Ordering is by descending score, so ``sorted(results, reverse=True)`` yields
    the most relevant chunk first.

    Attributes:
        chunk: The scored chunk.
        score: Relevance score. Scales differ by source and are only comparable
            within a single source -- which is precisely why fusion is
            rank-based rather than score-based (ADR-007).
        source: The retrieval stage that assigned the score.
    """

    chunk: Chunk
    score: float
    source: ScoreSource

    @property
    def chunk_id(self) -> str:
        """Identifier of the underlying chunk."""
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        """Text of the underlying chunk."""
        return self.chunk.text

    def rescored(self, score: float, source: ScoreSource) -> Self:
        """Return a copy carrying a new score from a later stage.

        Args:
            score: The new relevance score.
            source: The stage that produced it.

        Returns:
            A new :class:`ScoredChunk`; the original is unchanged.
        """
        return replace(self, score=score, source=source)

    def __lt__(self, other: ScoredChunk) -> bool:
        """Order by descending score so the best result sorts last ascending."""
        if not isinstance(other, ScoredChunk):
            return NotImplemented
        return self.score < other.score

    def __gt__(self, other: ScoredChunk) -> bool:
        """Order by descending score."""
        if not isinstance(other, ScoredChunk):
            return NotImplemented
        return self.score > other.score
