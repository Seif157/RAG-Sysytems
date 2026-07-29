"""Parsing plain text."""

from __future__ import annotations

from rag.domain.errors import DocumentParsingError
from rag.domain.models import (
    ContentBlock,
    DocumentProperties,
    DocumentType,
    ParsedDocument,
    RawDocument,
)
from rag.domain.ports import DocumentParser

__all__ = ["TxtParser"]

#: Encodings to try, in order. UTF-8 first because it is nearly always right;
#: the fallbacks exist so a file exported from an older Windows tool still
#: opens rather than failing ingestion outright.
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


class TxtParser(DocumentParser):
    """Extracts text from a plain-text file.

    Blocks are split on blank lines, giving paragraph-level units with accurate
    character offsets. The chunker decides how those become chunks; splitting
    here would take that decision away from it.

    Plain text has no pages and no headings, so those stay ``None`` -- which is
    the honest answer, and keeps "this format has no pages" distinguishable from
    "the parser failed to find one".
    """

    @property
    def supported_type(self) -> DocumentType:
        """The format this parser handles."""
        return DocumentType.TXT

    def parse(self, document: RawDocument) -> ParsedDocument:
        """Extract paragraph blocks from the file.

        Args:
            document: The raw document to parse.

        Returns:
            Ordered blocks with character offsets. Empty when the file contains
            only whitespace -- a real outcome, not an error.

        Raises:
            DocumentParsingError: If the bytes cannot be decoded as text.
        """
        text = self._decode(document)
        return ParsedDocument(
            blocks=tuple(self._blocks(text)),
            properties=DocumentProperties(),
        )

    @staticmethod
    def _decode(document: RawDocument) -> str:
        """Decode bytes to text, trying a short list of encodings."""
        for encoding in _ENCODINGS:
            try:
                return document.content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise DocumentParsingError(
            f"could not decode {document.filename} as text",
            context={"filename": document.filename, "tried": list(_ENCODINGS)},
        )

    @staticmethod
    def _blocks(text: str) -> list[ContentBlock]:
        """Split text into paragraph blocks, preserving character offsets."""
        blocks: list[ContentBlock] = []
        offset = 0
        for paragraph in text.split("\n\n"):
            stripped = paragraph.strip()
            if stripped:
                start = text.index(stripped, offset)
                blocks.append(
                    ContentBlock(
                        text=stripped,
                        char_start=start,
                        char_end=start + len(stripped),
                    )
                )
                offset = start + len(stripped)
        return blocks
