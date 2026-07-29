"""Turning citation markers in a generated answer back into citations.

The model cites by emitting ``[1]``. Resolving that requires knowing which chunk
marker ``1`` referred to *in this particular prompt*, which is why the mapping
travels on the :class:`~rag.domain.models.ContextBlock` rather than being
reconstructed afterwards.

Two rules do most of the work here.

**Unknown markers are dropped, never guessed at.** A model that invents ``[7]``
when six passages were supplied must not produce a citation pointing at an
arbitrary one. A wrong citation is worse than a missing one, because it looks
like evidence.

**Citations identify locations, not chunks.** Several passages from one page are
merged into a single citation. Listing them separately shows a reader three
identical entries that read as three confirmations when they are one passage
split three ways -- while merging *across* locations would send them to the
wrong place. So the grouping key is the full locator: document, page, section.
"""

from __future__ import annotations

import re

from rag.domain.models import Citation, ContextBlock, ScoredChunk, ScoreSource

__all__ = ["assemble_citations", "extract_citation_markers"]

#: Matches a bracketed run of digits. Deliberately narrow: "[see appendix]" is
#: prose, not a citation, and treating it as one would fabricate a source.
_MARKER = re.compile(r"\[(\d+)\]")

#: How much of a passage to show alongside a citation. Enough to recognise it,
#: not so much that the answer is buried in quotations.
_SNIPPET_CHARS = 240

#: What identifies "the same place" for merging. Section is included because a
#: page can hold the end of one section and the start of the next, and
#: collapsing those would send a reader to the wrong half of the page.
type _Locator = tuple[str, int | None, str | None]


def extract_citation_markers(text: str) -> tuple[str, ...]:
    """Find the citation markers a generated answer used.

    Args:
        text: The generated answer.

    Returns:
        Marker strings in order of first appearance, without repeats.
    """
    seen: dict[str, None] = {}
    for match in _MARKER.finditer(text):
        seen.setdefault(match.group(1), None)
    return tuple(seen)


def _snippet(text: str) -> str:
    """Shorten passage text for display beside a citation."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_CHARS:
        return collapsed
    return collapsed[: _SNIPPET_CHARS - 1].rstrip() + "…"


def _locator(scored: ScoredChunk) -> _Locator:
    """The place a passage points to."""
    metadata = scored.chunk.metadata
    return (metadata.document_id, metadata.page_number, metadata.section)


def _relevance(scored: ScoredChunk) -> float | None:
    """The comparable relevance score, if there is one.

    Only a reranker produces one. A raw similarity or a fused rank score is not
    a confidence, and presenting it as one invites trust it has not earned.
    """
    return scored.score if scored.source is ScoreSource.RERANKED else None


def assemble_citations(answer_text: str, context: ContextBlock) -> tuple[Citation, ...]:
    """Build the citation list for a generated answer.

    Only passages the answer actually cited are included: citations describe
    what the answer used, not what retrieval happened to find.

    Args:
        answer_text: The generated answer, containing ``[n]`` markers.
        context: The context block the answer was generated from.

    Returns:
        Citations in the order their location was first cited, with passages
        from the same location merged.
    """
    by_chunk_id = {scored.chunk_id: scored for scored in context.chunks}

    # Insertion order is the order each *location* was first cited, so a merged
    # citation takes the position of its earliest mention.
    grouped: dict[_Locator, list[ScoredChunk]] = {}
    for marker in extract_citation_markers(answer_text):
        chunk_id = context.chunk_id_for_marker(marker)
        if chunk_id is None:
            continue
        scored = by_chunk_id.get(chunk_id)
        if scored is None:
            continue
        grouped.setdefault(_locator(scored), []).append(scored)

    citations: list[Citation] = []
    for supporting in grouped.values():
        # Strongest first: it supplies the snippet, and its id reads as the
        # citation's primary chunk.
        ranked = sorted(
            supporting,
            key=lambda scored: (_relevance(scored) is not None, _relevance(scored) or 0.0),
            reverse=True,
        )
        best = ranked[0]
        metadata = best.chunk.metadata
        scores = [score for scored in ranked if (score := _relevance(scored)) is not None]

        citations.append(
            Citation(
                chunk_ids=tuple(scored.chunk_id for scored in ranked),
                document_id=metadata.document_id,
                filename=metadata.filename,
                document_type=metadata.document_type,
                page_number=metadata.page_number,
                section=metadata.section,
                heading=metadata.heading,
                relevance_score=max(scores) if scores else None,
                snippet=_snippet(best.text),
            )
        )

    return tuple(citations)
