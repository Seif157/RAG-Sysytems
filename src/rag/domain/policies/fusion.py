"""Reciprocal rank fusion of several ranked result lists (ADR-007).

Merges the dense and sparse retrieval lists into one ranking.

**Why rank and not score.** Cosine similarity lives in ``[-1, 1]``; BM25 is
unbounded and its magnitude depends on corpus statistics. Blending the two
requires normalising both, and any normalisation has to be re-tuned whenever the
embedding model, the corpus or the typical query length changes -- a maintenance
cost that shows up as retrieval quietly getting worse, not as a failure. RRF
discards magnitudes entirely and uses only position, so it needs no tuning and
cannot drift.

**What it rewards.** A chunk near the top of *both* lists scores higher than one
at the top of a single list. That is the point: dense and sparse retrieval fail
in different directions, so agreement between them is genuine evidence.

The formula, for each chunk::

    score = Σ over lists  weight / (k + rank)

with ``rank`` counted from 1.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain.models import ScoredChunk, ScoreSource

__all__ = ["DEFAULT_RRF_K", "reciprocal_rank_fusion"]

#: The conventional constant from the original TREC work. It sets how sharply
#: rank matters: small values make being ranked first dominant, large values
#: make fusion behave more like "how many retrievers found this at all".
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[Sequence[ScoredChunk], float]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> tuple[ScoredChunk, ...]:
    """Fuse weighted ranked lists into a single ranking.

    Args:
        ranked_lists: Pairs of ``(results, weight)``. Each result list must be
            in descending relevance order -- fusion reads position, so a
            mis-ordered input silently produces a mis-ordered output. A weight
            of ``0.0`` excludes that list.
        k: Fusion constant. Larger flattens the influence of rank.
        limit: Maximum results to return. ``None`` returns all.

    Returns:
        Chunks ordered by descending fused score, deduplicated by chunk id and
        marked :attr:`~rag.domain.models.ScoreSource.FUSED`. Ties are broken by
        chunk id so the ranking is reproducible -- otherwise any measurement of
        retrieval quality would be noisy for no reason.

    Raises:
        ValueError: If a weight is negative, ``k`` is not positive, or ``limit``
            is not positive.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when given")

    totals: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}

    for results, weight in ranked_lists:
        if weight < 0.0:
            raise ValueError(f"list weight must be >= 0.0, got {weight}")
        if weight == 0.0:
            continue
        for rank, scored in enumerate(results, start=1):
            chunk_id = scored.chunk_id
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weight / (k + rank)
            # Keep the first sighting: the lists carry the same chunk, and the
            # score is about to be replaced by the fused one anyway.
            chunks.setdefault(chunk_id, scored)

    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ordered = ordered[:limit]

    return tuple(chunks[chunk_id].rescored(score, ScoreSource.FUSED) for chunk_id, score in ordered)
