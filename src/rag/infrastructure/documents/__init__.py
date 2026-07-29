"""Document loading, format detection and parsing.

Parsers extract text *and structure* -- pages, headings, character offsets --
and nothing else. A parser that chunks has taken a decision belonging to the
chunking strategy.

What each format can honestly provide differs, and the parsers say so rather
than inventing values:

============  =============  ==================
Format        Page numbers   Heading structure
============  =============  ==================
PDF           yes            no
DOCX          no             yes
Markdown      no             yes
TXT           no             no
============  =============  ==================

DOCX has no fixed pagination and Markdown has no pages at all, so both leave
``page_number`` as ``None`` instead of guessing.
"""

from rag.infrastructure.documents.loader import DocumentLoader
from rag.infrastructure.documents.parsers.docx_parser import DocxParser
from rag.infrastructure.documents.parsers.markdown_parser import MarkdownParser
from rag.infrastructure.documents.parsers.pdf_parser import PdfParser
from rag.infrastructure.documents.parsers.txt_parser import TxtParser
from rag.infrastructure.documents.registry import ParserRegistry

__all__ = [
    "DocumentLoader",
    "DocxParser",
    "MarkdownParser",
    "ParserRegistry",
    "PdfParser",
    "TxtParser",
]
