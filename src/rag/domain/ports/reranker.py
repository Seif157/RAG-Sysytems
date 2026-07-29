"""Port: re-ordering retrieved candidates by relevance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag.domain.models import ScoredChunk

__all__ = ["Reranker"]


class Reranker(ABC):
    """Re-scores retrieved candidates and keeps the best.

    Sits between retrieval and the LLM: retrieve wide (20), rerank narrow (5).
    Cross-encoders substantially outperform bi-encoder similarity for final
    ranking because they attend to query and passage jointly, which is exactly
    the comparison a bi-encoder cannot make (ADR-008).

    The contract is narrow on purpose. A reranker must be a **pure re-ordering
    and truncation** of its input:

    * every returned chunk was in the input;
    * at most ``top_n`` are returned;
    * they are ordered by descending relevance;
    * chunk text and metadata are unmodified.

    A reranker that merges, edits or invents chunks breaks citation accuracy,
    and the shared contract test suite asserts each of these properties.

    Reranking is an **optional** stage. Callers catch
    :class:`~rag.domain.errors.RerankingError`, fall back to fusion order, and
    record the degradation -- quality drops, availability does not.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        top_n: int,
    ) -> tuple[ScoredChunk, ...]:
        """Re-score candidates against the query and keep the best.

        Args:
            query: The query to score relevance against.
            candidates: Chunks from retrieval, in fusion order.
            top_n: Maximum number to return.

        Returns:
            At most ``top_n`` of the input chunks, most relevant first, each
            re-scored with :attr:`~rag.domain.models.ScoreSource.RERANKED`.

        Raises:
            RerankingError: If reranking fails. Callers are expected to degrade
                rather than fail the request.
        """
