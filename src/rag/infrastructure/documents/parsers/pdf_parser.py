"""Parsing PDF.

The page number is the single most valuable thing a PDF citation can carry: it
turns "this came from annual-report.pdf" into a claim the reader can verify in
seconds. So text is extracted one page at a time and each block keeps the page
it came from, rather than extracting the whole document and losing the mapping.
"""

from __future__ import annotations

import io
from datetime import datetime

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.domain.errors import CorruptDocumentError, DocumentParsingError, EncryptedDocumentError
from rag.domain.models import (
    ContentBlock,
    DocumentProperties,
    DocumentType,
    ParsedDocument,
    RawDocument,
)
from rag.domain.ports import DocumentParser

__all__ = ["PdfParser"]

#: Values PDF producers write when they have nothing real to say. Keeping them
#: is worse than dropping them: an author of "anonymous" is filterable and
#: looks like a fact, so a search for documents by a named author silently
#: competes with hundreds of these. Compared case-insensitively.
_PLACEHOLDER_PROPERTIES = frozenset(
    {
        "anonymous",
        "unknown",
        "untitled",
        "none",
        "n/a",
        "-",
        "document1",
        "microsoft word",
    }
)


class PdfParser(DocumentParser):
    """Extracts text and page numbers from a PDF.

    A page with no extractable text is skipped rather than treated as an error.
    That is what a scanned page looks like, OCR is explicitly out of scope, and
    a document that is half scanned should still contribute the half it can.
    """

    @property
    def supported_type(self) -> DocumentType:
        """The format this parser handles."""
        return DocumentType.PDF

    def parse(self, document: RawDocument) -> ParsedDocument:
        """Extract one block per page that has text.

        Args:
            document: The raw PDF.

        Returns:
            Blocks carrying their page number, plus document properties.

        Raises:
            EncryptedDocumentError: If the PDF is password-protected.
            CorruptDocumentError: If the file cannot be read as a PDF.
            DocumentParsingError: If extraction fails for another reason.
        """
        reader = self._open(document)

        blocks: list[ContentBlock] = []
        offset = 0
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:
                raise DocumentParsingError(
                    f"could not extract text from page {number} of {document.filename}",
                    context={"filename": document.filename, "page": number},
                ) from exc

            if not text:
                # A scanned page, or a page of pure graphics. Both are normal.
                continue

            blocks.append(
                ContentBlock(
                    text=text,
                    char_start=offset,
                    char_end=offset + len(text),
                    page_number=number,
                )
            )
            offset += len(text) + 1

        return ParsedDocument(
            blocks=tuple(blocks),
            properties=self._properties(reader),
        )

    @staticmethod
    def _open(document: RawDocument) -> PdfReader:
        """Open the PDF, translating failures into domain errors."""
        try:
            reader = PdfReader(io.BytesIO(document.content))
        except PdfReadError as exc:
            raise CorruptDocumentError(
                f"{document.filename} is not a readable PDF",
                context={"filename": document.filename},
            ) from exc
        except Exception as exc:
            raise DocumentParsingError(
                f"could not open {document.filename}: {exc}",
                context={"filename": document.filename},
            ) from exc

        if reader.is_encrypted:
            # Some PDFs are "encrypted" with an empty owner password and open
            # fine; try that before giving up on the user's behalf.
            try:
                unlocked = bool(reader.decrypt(""))
            except Exception:
                unlocked = False
            if not unlocked:
                raise EncryptedDocumentError(
                    f"{document.filename} is password-protected",
                    context={"filename": document.filename},
                )
        return reader

    @staticmethod
    def _properties(reader: PdfReader) -> DocumentProperties:
        """Read the document's own metadata, tolerating its absence."""
        try:
            metadata = reader.metadata
        except Exception:  # pragma: no cover - malformed metadata dictionary
            metadata = None

        def _clean(value: object) -> str | None:
            text = str(value).strip() if value is not None else ""
            if not text or text.lower() in _PLACEHOLDER_PROPERTIES:
                return None
            return text

        created: datetime | None = None
        if metadata is not None:
            try:
                created = metadata.creation_date
            except Exception:
                # Malformed date strings are common and never worth failing an
                # ingestion over.
                created = None

        return DocumentProperties(
            author=_clean(metadata.author) if metadata else None,
            title=_clean(metadata.title) if metadata else None,
            created_at=created,
            page_count=len(reader.pages),
        )
