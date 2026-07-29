"""Deciding where in the context each passage goes.

Language models do not read a long context uniformly. Recall is highest at the
beginning and the end and measurably lower in the middle -- the "lost in the
middle" effect. Ordering passages by relevance therefore puts the second-best
one in the position least likely to be used, which is free quality thrown away:
the same tokens, arranged better, get read better.

The obvious fix -- alternate passages between the front and back -- has a cost
of its own. It interleaves passages from different documents, so the model sees
fragments of three sources shuffled together and has to reassemble each argument
itself.

So the alternation is applied to **groups**, not passages:

1. group passages by document, keeping each group in reading order;
2. rank groups by their strongest passage;
3. place group 1 at the front, group 2 at the back, group 3 next-to-front, and
   so on.

Each document stays contiguous and internally ordered, and the strongest
material still lands where it will actually be read.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain.models import ScoredChunk

__all__ = ["order_for_attention"]


def order_for_attention(chunks: Sequence[ScoredChunk]) -> tuple[ScoredChunk, ...]:
    """Arrange passages so the strongest occupy the best-read positions.

    Args:
        chunks: The passages to arrange, in any order.

    Returns:
        The same passages, reordered. Documents remain contiguous and in reading
        order within themselves.
    """
    if len(chunks) < 2:
        return tuple(chunks)

    groups: dict[str, list[ScoredChunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.chunk.metadata.document_id, []).append(chunk)

    ordered_groups = [
        # Reading order within a document: a later section appearing before an
        # earlier one reads as incoherent even when both are relevant.
        sorted(group, key=lambda c: (c.chunk.metadata.chunk_index, c.chunk_id))
        for group in groups.values()
    ]
    # Strongest group first. Ties break on document id so the arrangement is
    # reproducible -- otherwise identical inputs could produce different prompts.
    ordered_groups.sort(
        key=lambda group: (
            -max(chunk.score for chunk in group),
            group[0].chunk.metadata.document_id,
        )
    )

    front: list[list[ScoredChunk]] = []
    back: list[list[ScoredChunk]] = []
    for position, group in enumerate(ordered_groups):
        (front if position % 2 == 0 else back).append(group)

    return tuple(chunk for group in [*front, *reversed(back)] for chunk in group)
