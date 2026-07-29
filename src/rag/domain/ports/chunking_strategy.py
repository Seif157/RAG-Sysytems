"""Port: choosing chunk boundaries."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.domain.models import ChunkCandidate, ParsedDocument

__all__ = ["ChunkingStrategy"]


class ChunkingStrategy(ABC):
    """Splits a parsed document into retrievable passages.

    Chunking is the highest-leverage quality lever in a RAG system, which is why
    it is a configurable strategy rather than a constant, and why the choice is
    measured by the evaluation harness rather than argued about (ADR-021).

    Implementations must satisfy the following invariants. They are not
    advisory: the shared contract test suite asserts every one of them against
    every strategy, which is what makes strategies genuinely interchangeable.

    * **Bounded size.** No candidate exceeds the configured maximum token count.
    * **Structure preserved.** Each candidate inherits the page, section and
      heading path of the block it came from, so citations stay accurate.
    * **Hard boundaries respected.** A candidate never spans a page break or a
      top-level heading, because a chunk that does cannot be cited precisely.
    * **No content lost.** Concatenating candidate texts in order reproduces the
      source, modulo overlap and whitespace.
    * **Deterministic.** The same input and configuration produce identical
      candidates, and therefore identical chunk ids -- which is what makes
      re-ingestion idempotent (ADR-014).

    Synchronous: chunking is CPU work in the ingestion worker.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier, recorded on every chunk for provenance."""

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> tuple[ChunkCandidate, ...]:
        """Choose chunk boundaries for a parsed document.

        Returns *candidates* rather than chunks: a strategy decides where text
        splits and what structure it inherits, while identity, tenancy and
        provenance are stamped by the pipeline. That separation is what allows a
        strategy to be tested with no tenant, no clock and no configuration.

        Args:
            document: The parsed document to split.

        Returns:
            Candidates in document order. Empty when the document contained no
            extractable text.

        Raises:
            ChunkingError: If the document cannot be split into valid chunks.
        """
