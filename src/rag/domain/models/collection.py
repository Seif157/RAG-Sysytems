"""Vector collection description and compatibility.

The compatibility check here is the domain half of ADR-015. Vectors produced by
different embedding models are not comparable; a collection whose vectors came
from a different model than the one now configured will return results that look
plausible and are wrong. That is the worst failure mode this system has, because
nothing about the output signals it.

So the rule is expressed as data the collection carries, and checked at startup
by ``Bootstrap``, which refuses to serve traffic on mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["CollectionInfo", "CollectionSpec", "DistanceMetric"]


class DistanceMetric(StrEnum):
    """How vector similarity is measured.

    Part of a collection's identity: the same vectors under a different metric
    rank differently, so a metric change is as invalidating as a model change.
    """

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    """The collection shape the running configuration requires.

    Attributes:
        name: Collection name.
        dense_dimension: Dimensionality of the configured embedding model.
        distance: Similarity metric.
        embedding_model_id: Model that must have produced the stored vectors.
        supports_sparse: Whether a sparse vector is stored alongside the dense
            one, which hybrid retrieval requires (ADR-006).
    """

    name: str
    dense_dimension: int
    distance: DistanceMetric
    embedding_model_id: str
    supports_sparse: bool = True

    def __post_init__(self) -> None:
        """Validate the specification is coherent."""
        if not self.name.strip():
            raise ValueError("collection name must be a non-empty string")
        if self.dense_dimension < 1:
            raise ValueError("dense_dimension must be >= 1")
        if not self.embedding_model_id.strip():
            raise ValueError("embedding_model_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """The collection shape that actually exists in the store.

    Attributes:
        name: Collection name.
        dense_dimension: Dimensionality of the stored vectors.
        distance: Similarity metric the collection was created with.
        embedding_model_id: Model recorded as having produced the vectors.
        supports_sparse: Whether the collection carries sparse vectors.
        points_count: Number of stored points.
    """

    name: str
    dense_dimension: int
    distance: DistanceMetric
    embedding_model_id: str
    supports_sparse: bool
    points_count: int = 0

    def is_compatible_with(self, spec: CollectionSpec) -> bool:
        """Whether this collection can serve the given configuration.

        Args:
            spec: The shape the running configuration requires.

        Returns:
            ``True`` only if every dimension of compatibility holds.
        """
        return not self.incompatibility_reasons(spec)

    def incompatibility_reasons(self, spec: CollectionSpec) -> tuple[str, ...]:
        """Explain every way this collection fails to match the specification.

        Returning all reasons rather than the first one matters operationally:
        a migration that fixes the dimension only to fail on the metric wastes a
        full re-index cycle.

        Args:
            spec: The shape the running configuration requires.

        Returns:
            Human-readable reasons, empty when compatible.
        """
        reasons: list[str] = []
        if self.dense_dimension != spec.dense_dimension:
            reasons.append(
                f"dense vector dimension is {self.dense_dimension}, "
                f"configuration requires {spec.dense_dimension}"
            )
        if self.embedding_model_id != spec.embedding_model_id:
            reasons.append(
                f"collection was built with embedding model "
                f"{self.embedding_model_id!r}, configuration specifies "
                f"{spec.embedding_model_id!r}"
            )
        if self.distance is not spec.distance:
            reasons.append(
                f"distance metric is {self.distance.value}, "
                f"configuration requires {spec.distance.value}"
            )
        if spec.supports_sparse and not self.supports_sparse:
            reasons.append("collection has no sparse vectors, hybrid retrieval requires them")
        return tuple(reasons)
