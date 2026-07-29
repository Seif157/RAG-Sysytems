"""OpenAI embeddings."""

from __future__ import annotations

from collections.abc import Sequence

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from rag.domain.errors import EmbeddingError, EmbeddingRateLimitError
from rag.domain.models import DenseVector
from rag.domain.ports import Embedder

__all__ = ["OpenAIEmbedder"]


class OpenAIEmbedder(Embedder):
    """Embeds text with an OpenAI embedding model.

    Translates the SDK's exceptions into the domain's, so retry policy reads
    ``retryable`` as data rather than matching on provider message strings.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 128,
        timeout_s: float = 60.0,
    ) -> None:
        """Initialise the embedder.

        Args:
            api_key: OpenAI credential.
            model: Embedding model identifier.
            dimension: Expected output width, validated against the collection.
            batch_size: Texts per request.
            timeout_s: Per-request timeout.
        """
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_s)
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:
        """Model identifier, persisted with the collection."""
        return self._model

    @property
    def dimension(self) -> int:
        """Width of the vectors produced."""
        return self._dimension

    @property
    def max_batch_size(self) -> int:
        """Largest batch sent per request."""
        return self._batch_size

    async def embed_documents(self, texts: Sequence[str]) -> tuple[DenseVector, ...]:
        """Embed passages, batching as needed and preserving order."""
        vectors: list[DenseVector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed(batch))
        return tuple(vectors)

    async def embed_query(self, text: str) -> DenseVector:
        """Embed a single query."""
        return (await self._embed([text]))[0]

    async def _embed(self, texts: list[str]) -> list[DenseVector]:
        """Call the API and translate failures."""
        try:
            response = await self._client.embeddings.create(model=self._model, input=texts)
        except RateLimitError as exc:
            raise EmbeddingRateLimitError(
                f"OpenAI rate limit while embedding {len(texts)} texts",
                context={"model": self._model, "batch": len(texts)},
            ) from exc
        except APITimeoutError as exc:
            raise EmbeddingError(
                f"OpenAI timed out while embedding {len(texts)} texts",
                context={"model": self._model},
                retryable=True,
            ) from exc
        except APIError as exc:
            raise EmbeddingError(
                f"OpenAI embedding request failed: {exc}",
                context={"model": self._model},
            ) from exc

        # The API returns results in request order, but it also returns an index
        # per item; sorting on it costs nothing and removes the assumption.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [tuple(item.embedding) for item in ordered]
