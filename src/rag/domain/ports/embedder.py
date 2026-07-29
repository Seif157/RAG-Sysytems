"""Port: dense embedding generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag.domain.models import DenseVector

__all__ = ["Embedder"]


class Embedder(ABC):
    """Turns text into dense vectors.

    Document and query embedding are separate methods rather than one method
    with a flag, because several models are *asymmetric*: they require a
    different instruction prefix for a passage than for a query, and using the
    wrong one silently degrades recall without failing anything.

    :attr:`model_id` and :attr:`dimension` are part of the contract because they
    are recorded on the collection. Vectors from different models are not
    comparable, so a mismatch between the configured embedder and the stored
    vectors returns confidently wrong answers with plausible citations -- the
    worst failure mode in the system. Startup validation refuses to run on
    mismatch rather than degrading (ADR-015).
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable model identifier, persisted with the collection."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the vectors this embedder produces."""

    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Largest batch the provider accepts, used to size ingestion batches."""

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> tuple[DenseVector, ...]:
        """Embed passages for indexing.

        Implementations must preserve both order and length: the *n*-th vector
        corresponds to the *n*-th input. Callers pair them positionally, so a
        provider that silently drops a failed item would misattribute every
        subsequent vector.

        Args:
            texts: Passages to embed. May exceed :attr:`max_batch_size`;
                implementations are responsible for internal batching.

        Returns:
            One vector per input, in input order.

        Raises:
            EmbeddingRateLimitError: If the provider throttled the request.
            EmbeddingError: If embedding failed for any other reason.
        """

    @abstractmethod
    async def embed_query(self, text: str) -> DenseVector:
        """Embed a search query.

        Args:
            text: The query text.

        Returns:
            A single vector of length :attr:`dimension`.

        Raises:
            EmbeddingRateLimitError: If the provider throttled the request.
            EmbeddingError: If embedding failed for any other reason.
        """
