"""Removing redundant passages before they consume the context budget.

Exact duplicates are common -- hybrid retrieval returns the same chunk from both
its lists -- and near-duplicates are commoner than they look: boilerplate
repeated across documents, a clause quoted in two sections, or a fragment
surfacing alongside the chunk that contains it because of chunk overlap.

Each one costs twice. It consumes budget a genuinely different passage needed,
and it biases the model by repetition: seeing a claim twice reads as
corroboration when it is the same sentence counted again.

The comparison is a Jaccard similarity over normalised word sets, plus a
containment check. Deliberately crude: it catches text that is *the same*, not
text that *means* the same. Two passages saying different things about the same
topic must both survive, because discarding one silently discards evidence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag.domain.models import ScoredChunk

__all__ = ["DEFAULT_DUPLICATE_THRESHOLD", "deduplicate_chunks"]

#: Word-overlap ratio above which two passages count as the same. High on
#: purpose: the cost of wrongly dropping a distinct passage (lost evidence, no
#: symptom) is worse than the cost of keeping a near-duplicate (wasted tokens).
DEFAULT_DUPLICATE_THRESHOLD = 0.85

#: Below this many words, Jaccard is too coarse to trust -- two short sentences
#: sharing a few common words look identical to it. Short passages are compared
#: by containment alone.
_MINIMUM_WORDS_FOR_SIMILARITY = 6

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> frozenset[str]:
    """Normalise text to a set of words, ignoring case and punctuation."""
    return frozenset(_WORD.findall(text.lower()))


def _normalised(text: str) -> str:
    """Collapse text to a comparable form for the containment check."""
    return " ".join(_WORD.findall(text.lower()))


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


def deduplicate_chunks(
    chunks: Sequence[ScoredChunk],
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> tuple[ScoredChunk, ...]:
    """Drop passages that repeat one already kept.

    Candidates are considered in the order given -- which is rank order -- so
    the survivor of any duplicate pair is the higher-ranked one.

    Args:
        chunks: Candidates in rank order, most relevant first.
        threshold: Word-overlap ratio above which two passages are the same.

    Returns:
        The surviving chunks, in their original relative order.

    Raises:
        ValueError: If the threshold is outside ``[0.0, 1.0]``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0.0, 1.0], got {threshold}")

    kept: list[ScoredChunk] = []
    kept_words: list[frozenset[str]] = []
    kept_text: list[str] = []
    seen_ids: set[str] = set()

    for candidate in chunks:
        if candidate.chunk_id in seen_ids:
            continue

        words = _words(candidate.text)
        text = _normalised(candidate.text)

        if any(
            _is_redundant(words, text, other_words, other_text, threshold)
            for other_words, other_text in zip(kept_words, kept_text, strict=True)
        ):
            continue

        seen_ids.add(candidate.chunk_id)
        kept.append(candidate)
        kept_words.append(words)
        kept_text.append(text)

    return tuple(kept)


def _is_redundant(
    words: frozenset[str],
    text: str,
    other_words: frozenset[str],
    other_text: str,
    threshold: float,
) -> bool:
    """Whether a candidate adds nothing over a passage already kept."""
    # Containment first: a fragment of an already-kept passage contributes
    # nothing, however short it is.
    if text and text in other_text:
        return True
    if len(words) < _MINIMUM_WORDS_FOR_SIMILARITY:
        return False
    return _similarity(words, other_words) >= threshold
