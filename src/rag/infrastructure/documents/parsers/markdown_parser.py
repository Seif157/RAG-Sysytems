"""Parsing Markdown.

Markdown's heading structure is the most explicit of any format we support, and
the cheapest to read correctly -- with one trap. A ``#`` inside a fenced code
block is a shell comment, not a heading, and treating it as one silently
restructures the document and misattributes every chunk after it.

No Markdown library is used. Rendering to HTML and walking a tree would lose the
character offsets, and the subset that matters here -- fences and ATX headings --
is a few lines to read directly.
"""

from __future__ import annotations

import re

from rag.domain.errors import DocumentParsingError
from rag.domain.models import (
    ContentBlock,
    DocumentProperties,
    DocumentType,
    ParsedDocument,
    RawDocument,
)
from rag.domain.ports import DocumentParser
from rag.infrastructure.documents.parsers.heading_stack import HeadingStack

__all__ = ["MarkdownParser"]

#: An ATX heading: one to six hashes, a space, then the title.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")

#: A fenced code block delimiter, backticks or tildes.
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")

_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


class MarkdownParser(DocumentParser):
    """Extracts blocks and heading structure from Markdown."""

    @property
    def supported_type(self) -> DocumentType:
        """The format this parser handles."""
        return DocumentType.MARKDOWN

    def parse(self, document: RawDocument) -> ParsedDocument:
        """Extract blocks with their heading breadcrumb.

        Args:
            document: The raw Markdown.

        Returns:
            Blocks carrying heading context. Markdown has no pages, so
            ``page_number`` is always ``None``.

        Raises:
            DocumentParsingError: If the bytes cannot be decoded as text.
        """
        text = self._decode(document)

        blocks: list[ContentBlock] = []
        headings = HeadingStack()
        pending: list[str] = []
        pending_start = 0
        in_fence = False
        fence_marker = ""
        offset = 0

        def flush() -> None:
            """Emit whatever has accumulated as one block."""
            joined = "\n".join(pending).strip()
            if joined:
                blocks.append(
                    ContentBlock(
                        text=joined,
                        char_start=pending_start,
                        char_end=pending_start + len(joined),
                        section=headings.section,
                        heading=headings.nearest,
                        heading_path=headings.path,
                    )
                )
            pending.clear()

        for line in text.splitlines(keepends=True):
            stripped = line.rstrip("\n").rstrip("\r")
            fence = _FENCE.match(stripped)

            if in_fence:
                pending.append(stripped)
                if fence and stripped.strip().startswith(fence_marker):
                    in_fence = False
                offset += len(line)
                continue

            if fence:
                if not pending:
                    pending_start = offset
                in_fence = True
                fence_marker = fence.group(1)[0] * 3
                pending.append(stripped)
                offset += len(line)
                continue

            heading = _HEADING.match(stripped)
            if heading:
                flush()
                title = heading.group(2).strip()
                headings.push(len(heading.group(1)), title)
                if title:
                    blocks.append(
                        ContentBlock(
                            text=title,
                            char_start=offset,
                            char_end=offset + len(stripped),
                            section=headings.section,
                            heading=headings.nearest,
                            heading_path=headings.path,
                        )
                    )
                offset += len(line)
                pending_start = offset
                continue

            if not stripped.strip():
                flush()
                offset += len(line)
                pending_start = offset
                continue

            if not pending:
                pending_start = offset
            pending.append(stripped)
            offset += len(line)

        flush()
        return ParsedDocument(blocks=tuple(blocks), properties=DocumentProperties())

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
