"""Sparse (lexical) retrieval."""

from __future__ import annotations

from rag.domain.errors import RetrievalError, VectorStoreError
from rag.domain.models import RetrievalRequest, ScoredChunk
from rag.domain.ports import Retriever, VectorStore
from rag.infrastructure.retrieval.bm25 import Bm25SparseEncoder

__all__ = ["SparseRetriever"]


class SparseRetriever(Retriever):
    """Finds chunks by exact term overlap, scored with BM25.

    The complement to dense retrieval rather than a competitor to it: it has no
    semantic understanding at all, and in exchange it matches rare tokens --
    identifiers, part numbers, surnames, quoted phrases -- that an embedding
    blurs away.
    """

    def __init__(self, encoder: Bm25SparseEncoder, store: VectorStore) -> None:
        """Initialise the retriever.

        Args:
            encoder: Produces the query's sparse vector.
            store: Holds the chunk sparse vectors.
        """
        self._encoder = encoder
        self._store = store

    async def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        """Find chunks sharing terms with the query.

        Args:
            request: The resolved retrieval instruction.

        Returns:
            Up to ``request.top_k`` chunks, best match first. Empty when the
            query shares no vocabulary with the corpus, or consists only of stop
            words -- both correctly match nothing rather than everything.

        Raises:
            RetrievalError: If the search fails.
        """
        vector = await self._encoder.encode_query(request.query_text)

        try:
            return await self._store.search_sparse(
                vector=vector, top_k=request.top_k, filters=request.filters
            )
        except VectorStoreError as exc:
            raise RetrievalError(
                f"sparse search failed: {exc.message}", retryable=exc.retryable
            ) from exc
