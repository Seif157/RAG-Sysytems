"""Structure-aware recursive chunking.

The default strategy (ADR-021). Not naive fixed-size splitting: it works down a
ladder of separators -- paragraph, line, sentence, word -- and only ever falls
back to a hard character cut when a single word exceeds the limit.

Each candidate inherits the structural position of the block it came from, so
citations stay accurate, and a candidate never spans two blocks. That is what
keeps "page 14, section 3.2" true rather than approximately true.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence

from rag.domain.models import ChunkCandidate, ContentBlock, ParsedDocument
from rag.domain.ports import ChunkingStrategy

__all__ = ["RecursiveChunker"]

#: Split points in descending order of preference. Splitting on a paragraph
#: boundary preserves more meaning than splitting on a space.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

_WHITESPACE = re.compile(r"\s+")


def _default_token_count(text: str) -> int:
    """Approximate a token count without loading a tokenizer."""
    return max(1, len(text) // 4)


class RecursiveChunker(ChunkingStrategy):
    """Splits blocks on the largest natural boundary that fits."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        count_tokens: Callable[[str], int] = _default_token_count,
    ) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Characters of overlap between adjacent chunks, so a
                sentence split across a boundary is still retrievable whole from
                one side.
            count_tokens: Token counter used to stamp ``token_count``.

        Raises:
            ValueError: If the sizes are not usable.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._count_tokens = count_tokens

    @property
    def name(self) -> str:
        """Strategy identifier, recorded on every chunk."""
        return "recursive"

    def chunk(self, document: ParsedDocument) -> tuple[ChunkCandidate, ...]:
        """Split a parsed document into chunk candidates.

        Args:
            document: The parsed document.

        Returns:
            Candidates in document order. Empty when there was no text.
        """
        candidates: list[ChunkCandidate] = []
        for block in document.blocks:
            candidates.extend(self._chunk_block(block))
        return tuple(candidates)

    def _chunk_block(self, block: ContentBlock) -> Iterator[ChunkCandidate]:
        """Split one block, inheriting its structural position."""
        for text, offset in self._split(block.text):
            cleaned = text.strip()
            if not cleaned:
                continue
            start = block.char_start + offset
            yield ChunkCandidate(
                text=cleaned,
                char_start=start,
                char_end=start + len(text),
                token_count=self._count_tokens(cleaned),
                page_number=block.page_number,
                section=block.section,
                heading=block.heading,
                heading_path=block.heading_path,
            )

    def _split(self, text: str) -> list[tuple[str, int]]:
        """Split text into windows, returning each with its offset."""
        if len(text) <= self._chunk_size:
            return [(text, 0)]

        pieces = self._atomic_pieces(text)
        windows: list[tuple[str, int]] = []
        current: list[tuple[str, int]] = []
        length = 0

        for piece, offset in pieces:
            if current and length + len(piece) > self._chunk_size:
                windows.append(self._join(current))
                current = self._carry_over(current)
                length = sum(len(p) for p, _ in current)
            current.append((piece, offset))
            length += len(piece)

        if current:
            windows.append(self._join(current))
        return windows

    def _atomic_pieces(self, text: str) -> list[tuple[str, int]]:
        """Break text into the largest units that each fit a chunk."""
        pieces: list[tuple[str, int]] = [(text, 0)]
        for separator in _SEPARATORS:
            if all(len(piece) <= self._chunk_size for piece, _ in pieces):
                return pieces
            pieces = self._split_on(pieces, separator)
        return self._hard_split(pieces)

    def _split_on(self, pieces: Sequence[tuple[str, int]], separator: str) -> list[tuple[str, int]]:
        """Split any oversized piece on a separator, keeping the separator."""
        result: list[tuple[str, int]] = []
        for piece, offset in pieces:
            if len(piece) <= self._chunk_size or separator not in piece:
                result.append((piece, offset))
                continue
            cursor = 0
            for part in piece.split(separator):
                if part:
                    result.append((part + separator, offset + cursor))
                cursor += len(part) + len(separator)
        return result

    def _hard_split(self, pieces: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
        """Cut anything still oversized at the character limit.

        Reached only by text with no whitespace at all -- a base64 blob, a long
        URL. Splitting mid-token is ugly, but silently dropping it is worse.
        """
        result: list[tuple[str, int]] = []
        for piece, offset in pieces:
            if len(piece) <= self._chunk_size:
                result.append((piece, offset))
                continue
            for start in range(0, len(piece), self._chunk_size):
                result.append((piece[start : start + self._chunk_size], offset + start))
        return result

    @staticmethod
    def _join(pieces: Sequence[tuple[str, int]]) -> tuple[str, int]:
        """Join a window's pieces back into text with its starting offset."""
        return "".join(piece for piece, _ in pieces), pieces[0][1]

    def _carry_over(self, pieces: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
        """Take the tail of a window to overlap into the next one."""
        if self._chunk_overlap == 0:
            return []
        carried: list[tuple[str, int]] = []
        length = 0
        for piece, offset in reversed(pieces):
            if length >= self._chunk_overlap:
                break
            carried.insert(0, (piece, offset))
            length += len(piece)
        return carried
