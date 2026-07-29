"""Vector representations.

Dense vectors are a plain tuple of floats -- there is no value in wrapping them,
and a tuple is hashable, immutable and cheap.

Sparse vectors get a type of their own because they have an invariant worth
enforcing: parallel index and value arrays that can silently disagree. A
mismatched pair produces a store-level error thousands of chunks later, at which
point the cause is untraceable.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DenseVector", "SparseVector"]

#: A dense embedding. Its length must match the collection's configured
#: dimension, which is validated at startup rather than per write (ADR-015).
type DenseVector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A sparse (lexical) vector in coordinate form.

    Attributes:
        indices: Vocabulary positions with a non-zero weight.
        values: Weights, positionally aligned with :attr:`indices`.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the parallel arrays agree and describe a well-formed vector."""
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"indices and values must be the same length, "
                f"got {len(self.indices)} and {len(self.values)}"
            )
        if any(index < 0 for index in self.indices):
            raise ValueError("sparse vector indices must be non-negative")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("sparse vector indices must not contain duplicates")

    @property
    def non_zero_count(self) -> int:
        """How many terms carry weight.

        Zero is valid: a query sharing no term with the vocabulary produces an
        empty sparse vector, and lexical retrieval correctly returns nothing.
        """
        return len(self.indices)
