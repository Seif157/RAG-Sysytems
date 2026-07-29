"""Parsing DOCX.

Word documents have no fixed pagination -- the page a paragraph lands on depends
on the renderer, the font and the paper size -- so ``page_number`` stays
``None``. Claiming one would produce a citation that points nowhere.

What DOCX does have is explicit heading structure, which is more useful anyway:
"Financials > Revenue" tells a reader where they are far better than "page 7".
"""

from __future__ import annotations

import io
import re

from docx import Document as OpenDocx
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError

from rag.domain.errors import CorruptDocumentError, DocumentParsingError
from rag.domain.models import (
    ContentBlock,
    DocumentProperties,
    DocumentType,
    ParsedDocument,
    RawDocument,
)
from rag.domain.ports import DocumentParser
from rag.infrastructure.documents.parsers.heading_stack import HeadingStack

__all__ = ["DocxParser"]

#: Word's built-in heading styles are named "Heading 1", "Heading 2", and so on.
_HEADING_STYLE = re.compile(r"^Heading\s+(\d+)$", re.IGNORECASE)


class DocxParser(DocumentParser):
    """Extracts paragraphs and heading structure from a DOCX."""

    @property
    def supported_type(self) -> DocumentType:
        """The format this parser handles."""
        return DocumentType.DOCX

    def parse(self, document: RawDocument) -> ParsedDocument:
        """Extract paragraph blocks with their heading breadcrumb.

        Args:
            document: The raw DOCX.

        Returns:
            Blocks carrying heading context, plus document properties.

        Raises:
            CorruptDocumentError: If the file is not a readable DOCX.
            DocumentParsingError: If extraction fails for another reason.
        """
        docx = self._open(document)

        blocks: list[ContentBlock] = []
        headings = HeadingStack()
        offset = 0

        for paragraph in docx.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue

            level = self._heading_level(paragraph.style.name if paragraph.style else None)
            if level is not None:
                headings.push(level, text)

            blocks.append(
                ContentBlock(
                    text=text,
                    char_start=offset,
                    char_end=offset + len(text),
                    section=headings.section,
                    heading=headings.nearest,
                    heading_path=headings.path,
                )
            )
            offset += len(text) + 1

        return ParsedDocument(blocks=tuple(blocks), properties=self._properties(docx))

    @staticmethod
    def _open(document: RawDocument) -> DocxDocument:
        """Open the DOCX, translating failures into domain errors."""
        try:
            return OpenDocx(io.BytesIO(document.content))
        except PackageNotFoundError as exc:
            raise CorruptDocumentError(
                f"{document.filename} is not a readable DOCX",
                context={"filename": document.filename},
            ) from exc
        except Exception as exc:
            raise DocumentParsingError(
                f"could not open {document.filename}: {exc}",
                context={"filename": document.filename},
            ) from exc

    @staticmethod
    def _heading_level(style_name: str | None) -> int | None:
        """Return the heading depth for a paragraph style, if it is one."""
        if not style_name:
            return None
        match = _HEADING_STYLE.match(style_name.strip())
        return int(match.group(1)) if match else None

    @staticmethod
    def _properties(docx: DocxDocument) -> DocumentProperties:
        """Read the document's core properties, tolerating their absence."""
        try:
            core = docx.core_properties
        except Exception:  # pragma: no cover - malformed package
            return DocumentProperties()

        def _clean(value: object) -> str | None:
            text = str(value).strip() if value else ""
            return text or None

        return DocumentProperties(
            author=_clean(core.author),
            title=_clean(core.title),
            created_at=core.created,
        )
