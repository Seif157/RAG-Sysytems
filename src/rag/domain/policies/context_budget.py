"""Fitting retrieved passages into the prompt's context budget.

Retrieval returns passages ranked by relevance; the prompt has finite room. This
decides which ones actually reach the model.

Two rules that look like details and are not:

*Greedy fill, not stop-on-first-miss.* When a passage is too large, later
smaller ones are still considered. The budget exists to be used, and a short
highly relevant passage should not be lost because a bulky one happened to
precede it.

*Cost is injectable.* What reaches the model is the *rendered* block -- citation
marker, filename, page, section -- not the bare passage text. Budgeting the text
alone consistently understates the real prompt size, so the caller supplies the
true cost.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rag.domain.models import ScoredChunk

__all__ = ["select_within_budget"]


def _text_tokens(chunk: ScoredChunk) -> int:
    """Default cost: the passage's own token count."""
    return chunk.chunk.metadata.token_count


def select_within_budget(
    chunks: Sequence[ScoredChunk],
    budget_tokens: int,
    cost: Callable[[ScoredChunk], int] = _text_tokens,
) -> tuple[tuple[ScoredChunk, ...], int]:
    """Choose the passages that fit within a token budget.

    Args:
        chunks: Candidates in rank order, most relevant first.
        budget_tokens: Tokens available for context.
        cost: What each passage costs once rendered. Defaults to its token
            count, which ignores the rendered header.

    Returns:
        A pair of ``(selected, dropped_count)``. Selection preserves rank order.
        ``dropped_count`` being non-zero means the answer was produced with less
        context than retrieval found, which is worth surfacing when quality is
        investigated.

    Raises:
        ValueError: If the budget is negative, or a cost is negative.
    """
    if budget_tokens < 0:
        raise ValueError("budget_tokens must be >= 0")

    selected: list[ScoredChunk] = []
    seen: set[str] = set()
    used = 0
    dropped = 0

    for candidate in chunks:
        if candidate.chunk_id in seen:
            continue
        seen.add(candidate.chunk_id)

        candidate_cost = cost(candidate)
        if candidate_cost < 0:
            raise ValueError(f"cost must be >= 0, got {candidate_cost}")

        if used + candidate_cost <= budget_tokens:
            selected.append(candidate)
            used += candidate_cost
        else:
            dropped += 1

    return tuple(selected), dropped
