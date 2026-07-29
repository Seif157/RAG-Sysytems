"""Embeddings from a model running on this machine.

The default. It needs no API key, costs nothing per call, sends no document text
to a third party, and works offline once the model is cached -- which for a tool
you point at your own documents is the right default in every respect except
raw quality, where it is close enough that the reranker makes up the difference.

``BAAI/bge-small-en-v1.5`` is 384-dimensional and about 130 MB. Measured on the
handbook fixture, a relevant passage scores 0.78 against its question and an
unrelated one 0.43 -- ample separation for retrieval.

**The query instruction.** BGE retrieval models are trained asymmetrically: the
query carries an instruction prefix and the passage does not. That is exactly
what the port's separate ``embed_documents`` and ``embed_query`` exist for.
Getting it backwards, or applying the prefix to both sides, costs recall and
fails nothing -- the sort of defect that is only ever found by measuring.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from rag.core.logging import get_logger
from rag.domain.errors import EmbeddingError
from rag.domain.models import DenseVector
from rag.domain.ports import Embedder

__all__ = ["LocalEmbedder"]

_logger = get_logger(__name__)

#: Encodes a batch of texts into vectors.
type TextEncoder = Callable[[list[str]], Sequence[Sequence[float]]]

#: Instruction prefixes for model families that are trained with one. Matched on
#: a substring of the model name, because every published variant of a family
#: shares the same convention.
_QUERY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("bge-", "Represent this sentence for searching relevant passages: "),
    ("e5-", "query: "),
    ("gte-", ""),
)


def _default_query_prefix(model: str) -> str:
    """The instruction this model family expects on a query, if any."""
    lowered = model.lower()
    for marker, prefix in _QUERY_PREFIXES:
        if marker in lowered:
            return prefix
    return ""


class LocalEmbedder(Embedder):
    """Embeds text with a Sentence-Transformers model running locally."""

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        batch_size: int = 32,
        query_prefix: str | None = None,
        encoder: TextEncoder | None = None,
    ) -> None:
        """Initialise the embedder.

        Args:
            model: Sentence-Transformers model identifier.
            dimension: Expected output width. Checked against what the model
                actually returns, so a mismatch surfaces here rather than as a
                rejected write much later.
            batch_size: Texts encoded per forward pass.
            query_prefix: Instruction prepended to queries. ``None`` picks the
                convention for the model family; ``""`` disables it explicitly.
            encoder: Overrides the model with an encoding callable. Tests pass
                one so the contract can be exercised without a download.
        """
        self._model_name = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._query_prefix = _default_query_prefix(model) if query_prefix is None else query_prefix
        self._encoder = encoder

    @property
    def model_id(self) -> str:
        """Model identifier, persisted with the collection."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Width of the vectors produced."""
        return self._dimension

    @property
    def max_batch_size(self) -> int:
        """Texts encoded per forward pass."""
        return self._batch_size

    def _load(self) -> TextEncoder:
        """Build the encoding callable, loading the model on first use.

        Lazy for the same reason the reranker is: the model is a download on a
        first run, and paying for it when the container is built would make
        ``--check-store`` fetch an embedding model to answer a question about
        Qdrant.
        """
        if self._encoder is not None:
            return self._encoder

        import time

        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        model: Any = SentenceTransformer(self._model_name)
        _logger.info(
            "embedder.model_loaded",
            model=self._model_name,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

        def encode(texts: list[str]) -> Sequence[Sequence[float]]:
            # Normalised, so cosine distance in Qdrant is a plain dot product
            # and vector magnitudes cannot skew ranking.
            vectors: Any = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [tuple(float(value) for value in vector) for vector in vectors]

        self._encoder = encode
        return encode

    async def warm_up(self) -> None:
        """Load the model ahead of the first request."""
        await asyncio.to_thread(self._load)

    async def embed_documents(self, texts: Sequence[str]) -> tuple[DenseVector, ...]:
        """Embed passages for indexing, without the query instruction."""
        if not texts:
            return ()
        return await self._encode(list(texts))

    async def embed_query(self, text: str) -> DenseVector:
        """Embed a query, applying this model family's instruction prefix."""
        vectors = await self._encode([f"{self._query_prefix}{text}"])
        return vectors[0]

    async def _encode(self, texts: list[str]) -> tuple[DenseVector, ...]:
        """Encode texts in batches, off the event loop."""
        vectors: list[DenseVector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await asyncio.to_thread(self._encode_batch, batch))
        return tuple(vectors)

    def _encode_batch(self, batch: list[str]) -> list[DenseVector]:
        """Encode one batch on a worker thread, translating failures."""
        try:
            encoded = self._load()(batch)
        except Exception as exc:
            raise EmbeddingError(
                f"local embedding failed: {exc}",
                context={"model": self._model_name, "batch": len(batch)},
            ) from exc

        vectors = [tuple(float(value) for value in vector) for vector in encoded]
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingError(
                    f"{self._model_name} produced {len(vector)}-dimensional vectors, "
                    f"but EMBEDDING_DIMENSION is {self._dimension}",
                    context={"model": self._model_name, "actual": len(vector)},
                )
        return vectors
