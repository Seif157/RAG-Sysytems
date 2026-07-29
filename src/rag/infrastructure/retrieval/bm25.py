"""BM25 sparse encoding for lexical retrieval.

Dense embeddings are a lossy summary: they capture what a passage is *about* and
blur the rare tokens inside it. That is exactly the wrong trade for a user
searching ``ERR_CONN_4021``, a part number, or a surname -- the query embeds to
somewhere vaguely near the right neighbourhood and retrieves the wrong chunk.
BM25 has no semantic understanding whatsoever, and matches those tokens exactly.

**What this computes, and what it does not.** BM25's score has two halves: a
term-frequency component that saturates, and an inverse-document-frequency
component that weights rare terms higher. Only the first is computed here,
because IDF needs corpus-wide document counts that a stateless encoder cannot
know. Qdrant computes IDF itself from the collection when the sparse vector is
configured with ``modifier=IDF`` -- which is both more correct than a snapshot
taken at ingest time and the reason the collection must be created that way.

Written by hand rather than pulling in ``fastembed``: the term-frequency half of
BM25 is a dozen lines, and the alternative drags in ``onnxruntime`` and a model
download for arithmetic we can do exactly.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence

from rag.domain.models import SparseVector

__all__ = ["Bm25SparseEncoder"]

#: Tokens are runs of letters, digits and underscores. Keeping underscores and
#: digits together is deliberate: it is what preserves ``ERR_CONN_4021`` and
#: ``88421`` as single searchable tokens instead of shredding them.
_TOKEN = re.compile(r"[A-Za-z0-9_]+")

#: Words too common to carry signal. They appear in nearly every chunk, so they
#: cost index size and dilute the vector without ever discriminating.
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of",
        "on", "or", "she", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "to", "was", "were", "will", "with", "you",
        "your",
    }
)  # fmt: skip

#: Size of the hashed vocabulary. Sparse vectors are indexed by integer, so
#: tokens are hashed into this space. Large enough that collisions between two
#: real terms are rare; a collision merely makes two words look alike to the
#: lexical retriever, which the dense side is there to catch.
_VOCABULARY_SIZE = 2**20

#: Standard BM25 parameters. ``k1`` controls how fast term frequency saturates,
#: ``b`` how strongly length is normalised.
_K1 = 1.2
_B = 0.75

#: Assumed average document length in tokens, since the true corpus average is
#: not available at encode time. Chunk sizes here are bounded and fairly uniform,
#: so a constant is a good approximation -- and length normalisation is the least
#: sensitive part of the formula.
_AVERAGE_LENGTH = 128.0


def _token_index(token: str) -> int:
    """Hash a token into the sparse vocabulary space.

    Uses a stable digest rather than :func:`hash`, whose string hashing is
    randomised per process -- that would silently invalidate every stored vector
    on restart, and the symptom would be retrieval quietly returning nothing.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _VOCABULARY_SIZE


class Bm25SparseEncoder:
    """Turns text into BM25 term-frequency sparse vectors.

    A plain class rather than a port: there is one implementation, and wrapping
    it in an interface would be indirection with nothing on the other side.
    """

    def __init__(
        self,
        k1: float = _K1,
        b: float = _B,
        average_length: float = _AVERAGE_LENGTH,
    ) -> None:
        """Initialise the encoder.

        Args:
            k1: Term-frequency saturation. Higher means repetition counts for
                longer before flattening out.
            b: Length normalisation, from 0 (none) to 1 (full).
            average_length: Assumed average document length in tokens.

        Raises:
            ValueError: If the parameters are outside their valid ranges.
        """
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must lie in [0.0, 1.0]")
        if average_length <= 0:
            raise ValueError("average_length must be > 0")

        self._k1 = k1
        self._b = b
        self._average_length = average_length

    @property
    def model_id(self) -> str:
        """Identifier recorded for provenance."""
        return f"bm25(k1={self._k1},b={self._b})"

    async def encode_documents(self, texts: Sequence[str]) -> tuple[SparseVector, ...]:
        """Encode passages for indexing.

        Args:
            texts: Passages to encode.

        Returns:
            One sparse vector per input, in input order.
        """
        return tuple(self._encode(text) for text in texts)

    async def encode_query(self, text: str) -> SparseVector:
        """Encode a query.

        Args:
            text: The query text.

        Returns:
            A sparse vector. Empty when the query is only stop words or
            punctuation -- which correctly matches nothing rather than
            everything.
        """
        return self._encode(text)

    def _encode(self, text: str) -> SparseVector:
        """Compute the BM25 term-frequency vector for one piece of text."""
        tokens = [
            token for raw in _TOKEN.findall(text.lower()) if (token := raw) not in _STOP_WORDS
        ]
        if not tokens:
            return SparseVector(indices=(), values=())

        length_penalty = self._k1 * (1.0 - self._b + self._b * len(tokens) / self._average_length)

        # Collisions are possible in a hashed vocabulary, so weights are summed
        # per index rather than assigned -- two tokens landing on one index must
        # not produce a duplicate index, which is an invalid sparse vector.
        weights: dict[int, float] = {}
        for token, frequency in Counter(tokens).items():
            saturated = frequency * (self._k1 + 1.0) / (frequency + length_penalty)
            index = _token_index(token)
            weights[index] = weights.get(index, 0.0) + saturated

        ordered = sorted(weights.items())
        return SparseVector(
            indices=tuple(index for index, _ in ordered),
            values=tuple(weight for _, weight in ordered),
        )
