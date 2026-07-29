"""Builders for document fixtures in each supported format.

Built in code rather than committed as binaries so that what each file contains
is readable in the diff, and so a fixture cannot quietly drift from the
assertions that depend on it.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

__all__ = ["make_docx", "make_markdown", "make_pdf"]


def make_pdf(
    pages: Sequence[str],
    *,
    title: str | None = None,
    author: str | None = None,
) -> bytes:
    """Build a PDF with one line of text per page.

    Uses ``reportlab`` because ``pypdf`` reads PDFs but cannot write text onto
    one -- a fixture built with ``add_blank_page`` would contain nothing to
    extract and would make the parser tests vacuous.

    Args:
        pages: One line of text per page, in order.
        title: Document title metadata, when the test needs one.
        author: Document author metadata, when the test needs one.

    Returns:
        The PDF as bytes.
    """
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    if title is not None:
        pdf.setTitle(title)
    if author is not None:
        pdf.setAuthor(author)

    for text in pages:
        vertical = 720
        for line in text.splitlines() or [""]:
            pdf.drawString(72, vertical, line)
            vertical -= 16
        pdf.showPage()

    pdf.save()
    return buffer.getvalue()


def make_docx(
    blocks: Sequence[tuple[str, str]],
    *,
    title: str | None = None,
    author: str | None = None,
) -> bytes:
    """Build a DOCX from (style, text) pairs.

    Args:
        blocks: Pairs of paragraph style and text. Use ``"Heading 1"``,
            ``"Heading 2"`` and so on for headings, ``"Normal"`` for body text.
        title: Core-properties title.
        author: Core-properties author.

    Returns:
        The DOCX as bytes.
    """
    from docx import Document as DocxDocument

    document = DocxDocument()
    for style, text in blocks:
        document.add_paragraph(text, style=style)
    if title is not None:
        document.core_properties.title = title
    if author is not None:
        document.core_properties.author = author

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_markdown(text: str) -> bytes:
    """Encode Markdown source as bytes.

    Args:
        text: The Markdown source.

    Returns:
        UTF-8 encoded bytes.
    """
    return text.encode("utf-8")
