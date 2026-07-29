"""Port: retrieving candidate chunks for a query."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.domain.models import RetrievalRequest, ScoredChunk

__all__ = ["Retriever"]


class Retriever(ABC):
    """Finds candidate chunks for a query.

    Implemented three times -- dense, sparse, and hybrid -- and hybrid is itself
    a *Composite* of retrievers. Because the composite satisfies the same
    interface as its parts, switching between dense-only, sparse-only and hybrid
    retrieval is a wiring decision in the container rather than a branch in the
    retrieval service (ADR-006).

    A single-method interface is the point: it is the narrowest contract that
    still expresses everything a retrieval strategy needs.
    """

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        """Find the most relevant chunks for a request.

        Returning fewer than ``request.top_k`` results -- including none -- is a
        valid outcome, not an error. An empty corpus, an over-narrow filter, or
        a query with no lexical overlap all legitimately produce nothing, and
        the answer path handles that by refusing rather than inventing.

        Args:
            request: The resolved retrieval instruction.

        Returns:
            Up to ``request.top_k`` scored chunks, most relevant first.

        Raises:
            RetrievalError: If retrieval fails. This is a mandatory stage, so
                the failure propagates rather than degrading.
        """
