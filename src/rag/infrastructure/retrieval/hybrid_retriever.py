"""Hybrid retrieval: dense and sparse, fused by rank (ADR-006, ADR-007)."""

from __future__ import annotations

import asyncio

from rag.core.logging import get_logger
from rag.domain.errors import RetrievalError
from rag.domain.models import RetrievalRequest, ScoredChunk
from rag.domain.policies import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag.domain.ports import Retriever

__all__ = ["HybridRetriever"]

_logger = get_logger(__name__)


class HybridRetriever(Retriever):
    """Runs two retrievers concurrently and fuses their rankings.

    A *composite* of retrievers: it satisfies the same interface as its parts,
    which is what makes hybrid-versus-dense a wiring decision in the container
    rather than a branch in the query flow.

    The two halves run concurrently. Serially they would double query latency to
    reach the same answer, and that is the whole reason the retrieval ports are
    asynchronous.

    **Asymmetric failure handling, deliberately.** If the sparse half fails the
    query still succeeds on dense results alone: lexical matching is an
    enhancement, and losing it should cost some quality rather than the answer.
    If the *dense* half fails the error propagates, because answering from
    lexical matches alone would be a large quality drop that nothing in the
    response would reveal.
    """

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        *,
        sparse_weight: float = 0.5,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        """Initialise the retriever.

        Args:
            dense: Vector similarity retriever.
            sparse: Lexical retriever.
            sparse_weight: Relative influence of the sparse list, from 0.0
                (ignore it) to 1.0 (ignore dense). ``0.5`` weights both equally.
            rrf_k: Fusion constant.

        Raises:
            ValueError: If ``sparse_weight`` is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= sparse_weight <= 1.0:
            raise ValueError(f"sparse_weight must lie in [0.0, 1.0], got {sparse_weight}")

        self._dense = dense
        self._sparse = sparse
        self._sparse_weight = sparse_weight
        self._rrf_k = rrf_k

    async def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        """Retrieve from both halves and fuse the results.

        Both halves receive the same query text and the same filter. Preserving
        the filter across both paths matters: dropping it on one would silently
        return chunks the user had excluded.

        Args:
            request: The resolved retrieval instruction.

        Returns:
            Up to ``request.top_k`` chunks ranked by fused score.

        Raises:
            RetrievalError: If dense retrieval fails.
        """
        dense_results, sparse_results = await asyncio.gather(
            self._dense.retrieve(request),
            self._sparse_or_empty(request),
        )

        fused = reciprocal_rank_fusion(
            [
                (dense_results, 1.0 - self._sparse_weight),
                (sparse_results, self._sparse_weight),
            ],
            k=self._rrf_k,
            limit=request.top_k,
        )

        _logger.debug(
            "retrieval.fused",
            dense=len(dense_results),
            sparse=len(sparse_results),
            fused=len(fused),
            overlap=len({c.chunk_id for c in dense_results} & {c.chunk_id for c in sparse_results}),
        )
        return fused

    async def _sparse_or_empty(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        """Retrieve sparse results, degrading to none on failure."""
        try:
            return await self._sparse.retrieve(request)
        except RetrievalError as exc:
            _logger.warning("retrieval.sparse_degraded", error_code=exc.code, error=exc.message)
            return ()
