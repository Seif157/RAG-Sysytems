"""The identity reranker."""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain.models import ScoredChunk
from rag.domain.ports import Reranker

__all__ = ["NoOpReranker"]


class NoOpReranker(Reranker):
    """Truncates to ``top_n`` without re-scoring.

    What gets wired when ``ENABLE_RERANK=false``. Its existence is why the query
    flow contains no reranking feature check: turning reranking off changes
    which object is built, not which branch runs (ADR-010).

    It also keeps the cross-encoder off the critical path in tests, where a
    400 MB model download would be an absurd price for asserting on ordering.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        top_n: int,
    ) -> tuple[ScoredChunk, ...]:
        """Return the first ``top_n`` candidates, unchanged.

        Scores keep their original :class:`~rag.domain.models.ScoreSource`, so a
        citation produced on this path correctly reports no reranked relevance
        score rather than presenting a raw similarity as one.

        Args:
            query: Ignored.
            candidates: Chunks in retrieval order.
            top_n: Maximum number to return.

        Returns:
            At most ``top_n`` candidates, in the order given.
        """
        return tuple(candidates[:top_n])
