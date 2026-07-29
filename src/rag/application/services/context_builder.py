"""Assembling retrieved passages into the block that goes into the prompt.

Everything upstream decides *which* passages the model sees. This decides *how*,
and the arrangement matters more than it looks:

*Redundancy is removed first.* A passage that repeats one already selected costs
twice -- it consumes budget a distinct passage needed, and repetition reads as
corroboration when it is the same sentence counted again. Deduplicating before
budgeting is what stops a duplicate from displacing the passage that actually
answers the question.

*The budget is charged for what is rendered.* The model receives a marked-up
block with a provenance header, not bare text. Counting the text alone
understates the prompt by fifteen or twenty tokens per passage, which is how a
prompt quietly exceeds a ceiling it was supposed to respect.

*Order is chosen for attention, not for tidiness.* Models read the beginning and
end of a context more reliably than the middle, so the strongest material is
placed at both edges -- while keeping each document contiguous and in reading
order (:mod:`rag.domain.policies.context_ordering`).

Two invariants the whole citation chain depends on: markers are numbered from 1
in **presentation** order, so ``[1]`` is genuinely the first thing the model
sees, and every marker resolves to a passage that is actually present.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rag.domain.models import ContextBlock, ScoredChunk
from rag.domain.policies import (
    DEFAULT_DUPLICATE_THRESHOLD,
    deduplicate_chunks,
    order_for_attention,
    select_within_budget,
)

__all__ = ["ContextBuilder"]


def _estimate_tokens(text: str) -> int:
    """Approximate a token count without loading a tokenizer."""
    return max(1, len(text) // 4) if text else 0


class ContextBuilder:
    """Turns ranked passages into a budgeted, ordered, marked-up context block."""

    def __init__(
        self,
        budget_tokens: int,
        count_tokens: Callable[[str], int] = _estimate_tokens,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    ) -> None:
        """Initialise the builder.

        Args:
            budget_tokens: Tokens available for retrieved context, leaving room
                in the prompt for the instruction, history and question.
            count_tokens: Token counter, used to charge the budget for the
                rendered block rather than the bare passage text.
            duplicate_threshold: Word-overlap ratio above which two passages
                count as the same.
        """
        self._budget_tokens = budget_tokens
        self._count_tokens = count_tokens
        self._duplicate_threshold = duplicate_threshold

    def build(self, chunks: Sequence[ScoredChunk]) -> ContextBlock:
        """Select, order, number and render passages for the prompt.

        Args:
            chunks: Candidates in rank order, most relevant first.

        Returns:
            A context block. Empty when nothing was retrieved or nothing fit --
            a valid outcome, in which case the prompt instructs the model to say
            the documents do not answer the question.
        """
        distinct = deduplicate_chunks(chunks, self._duplicate_threshold)
        selected, dropped = select_within_budget(
            distinct, self._budget_tokens, cost=self._rendered_cost
        )
        ordered = order_for_attention(selected)

        markers: list[tuple[str, str]] = []
        rendered: list[str] = []
        for position, scored in enumerate(ordered, start=1):
            marker = str(position)
            markers.append((marker, scored.chunk_id))
            rendered.append(self._render(marker, scored))

        block = "\n\n".join(rendered)
        return ContextBlock(
            chunks=ordered,
            rendered=block,
            token_count=self._count_tokens(block),
            marker_to_chunk_id=tuple(markers),
            dropped_count=dropped,
        )

    def _rendered_cost(self, scored: ScoredChunk) -> int:
        """What a passage costs once it carries its marker and provenance.

        Reuses the token count stamped at ingest -- computed there with the real
        tokenizer -- and charges only the header on top. Re-tokenising every
        candidate's full text on every query would be the same answer at twenty
        times the cost.
        """
        header = f"[1] ({self._describe(scored)})\n"
        return scored.chunk.metadata.token_count + self._count_tokens(header)

    def _render(self, marker: str, scored: ScoredChunk) -> str:
        """Render one passage with its citation marker and provenance."""
        return f"[{marker}] ({self._describe(scored)})\n{scored.text.strip()}"

    @staticmethod
    def _describe(scored: ScoredChunk) -> str:
        """Describe where a passage came from, omitting whatever is unknown.

        A literal "page None" is noise the model may repeat back to the user, so
        absent provenance is left out rather than rendered.
        """
        metadata = scored.chunk.metadata
        parts = [metadata.filename]
        if metadata.page_number is not None:
            parts.append(f"page {metadata.page_number}")
        if metadata.section:
            parts.append(metadata.section)
        elif metadata.heading:
            parts.append(metadata.heading)
        return ", ".join(parts)
