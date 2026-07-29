"""Dense (vector similarity) retrieval."""

from __future__ import annotations

from rag.domain.errors import EmbeddingError, RetrievalError, VectorStoreError
from rag.domain.models import RetrievalRequest, ScoredChunk
from rag.domain.ports import Embedder, Retriever, VectorStore

__all__ = ["DenseRetriever"]


class DenseRetriever(Retriever):
    """Embeds the query and finds the nearest chunks.

    Sparse and hybrid retrieval join it next phase; because all three satisfy
    the same interface, switching between them is a wiring decision rather than
    a branch in the query flow.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        """Initialise the retriever.

        Args:
            embedder: Produces the query vector.
            store: Holds the chunk vectors.
        """
        self._embedder = embedder
        self._store = store

    async def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        """Find the most relevant chunks for a request.

        Args:
            request: The resolved retrieval instruction.

        Returns:
            Up to ``request.top_k`` chunks, most similar first. An empty result
            is a valid outcome, not an error.

        Raises:
            RetrievalError: If embedding or search fails. Both are wrapped here
                so that callers of the retrieval stage catch one thing.
        """
        try:
            vector = await self._embedder.embed_query(request.query_text)
        except EmbeddingError as exc:
            raise RetrievalError(
                f"could not embed the query: {exc.message}",
                context={"model": self._embedder.model_id},
                retryable=exc.retryable,
            ) from exc

        try:
            return await self._store.search_dense(
                vector=vector, top_k=request.top_k, filters=request.filters
            )
        except VectorStoreError as exc:
            raise RetrievalError(
                f"vector search failed: {exc.message}",
                retryable=exc.retryable,
            ) from exc
