"""A deterministic embedder that needs no API key."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from rag.domain.models import DenseVector
from rag.domain.ports import Embedder

__all__ = ["FakeEmbedder"]


class FakeEmbedder(Embedder):
    """Produces stable pseudo-random vectors from a hash of the text.

    Not a semantic model, but it satisfies the properties retrieval actually
    relies on: identical text gives an identical vector, different text gives a
    different one, and every vector has the declared dimension and unit length.
    That is enough to test the pipeline end to end without a network call.
    """

    def __init__(self, dimension: int = 8, model_id: str = "fake-embedder") -> None:
        """Initialise the embedder.

        Args:
            dimension: Vector width. Small by default so test failures stay
                readable.
            model_id: Identifier reported to the collection.
        """
        self._dimension = dimension
        self._model_id = model_id
        self.embed_calls = 0

    @property
    def model_id(self) -> str:
        """Identifier reported to the collection."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Width of the vectors produced."""
        return self._dimension

    @property
    def max_batch_size(self) -> int:
        """Largest batch accepted."""
        return 64

    def _vector(self, text: str) -> DenseVector:
        """Derive a unit-length vector deterministically from text."""
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[index % len(digest)] / 255.0 - 0.5 for index in range(self._dimension)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return tuple(value / norm for value in raw)

    async def embed_documents(self, texts: Sequence[str]) -> tuple[DenseVector, ...]:
        """Embed passages, preserving order and length."""
        self.embed_calls += 1
        return tuple(self._vector(text) for text in texts)

    async def embed_query(self, text: str) -> DenseVector:
        """Embed a query.

        Uses the same derivation as documents, so a query matching a chunk's
        text exactly retrieves it -- which is what makes end-to-end tests able
        to assert on retrieval without a real model.
        """
        self.embed_calls += 1
        return self._vector(text)
