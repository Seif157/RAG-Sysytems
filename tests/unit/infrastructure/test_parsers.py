"""Behaviour of the document parsers.

Every assertion here is ultimately about a citation. A parser that extracts text
but loses the page number produces an answer you cannot check; one that loses
heading structure produces chunks that read as if they came from nowhere.

The parsers share a contract, so the shared expectations are tested against all
of them at once -- that is what keeps a new format from quietly behaving
differently from the ones already trusted.
"""

from __future__ import annotations

import hashlib

import pytest

from rag.domain.errors import CorruptDocumentError, DocumentParsingError
from rag.domain.models import DocumentType, RawDocument
from rag.infrastructure.documents import (
    DocxParser,
    MarkdownParser,
    PdfParser,
    TxtParser,
)
from tests.fixtures import make_docx, make_markdown, make_pdf

pytestmark = pytest.mark.unit


def _raw(content: bytes, filename: str, document_type: DocumentType) -> RawDocument:
    return RawDocument(
        filename=filename,
        content=content,
        document_type=document_type,
        content_hash=hashlib.sha256(content).hexdigest(),
    )


# --------------------------------------------------------------------------- #
# Shared contract
# --------------------------------------------------------------------------- #
ALL_PARSERS = [
    (TxtParser(), lambda: b"First paragraph.\n\nSecond paragraph.", "a.txt", DocumentType.TXT),
    (
        PdfParser(),
        lambda: make_pdf(["First paragraph.", "Second paragraph."]),
        "a.pdf",
        DocumentType.PDF,
    ),
    (
        DocxParser(),
        lambda: make_docx([("Normal", "First paragraph."), ("Normal", "Second paragraph.")]),
        "a.docx",
        DocumentType.DOCX,
    ),
    (
        MarkdownParser(),
        lambda: make_markdown("First paragraph.\n\nSecond paragraph.\n"),
        "a.md",
        DocumentType.MARKDOWN,
    ),
]
PARSER_IDS = [parser.__class__.__name__ for parser, _, _, _ in ALL_PARSERS]


@pytest.mark.parametrize(("parser", "build", "filename", "doc_type"), ALL_PARSERS, ids=PARSER_IDS)
class TestEveryParser:
    def test_it_declares_the_type_it_handles(self, parser, build, filename, doc_type):
        assert parser.supported_type is doc_type

    def test_it_extracts_the_text(self, parser, build, filename, doc_type):
        parsed = parser.parse(_raw(build(), filename, doc_type))

        text = " ".join(block.text for block in parsed.blocks)
        assert "First paragraph." in text
        assert "Second paragraph." in text

    def test_blocks_are_in_document_order(self, parser, build, filename, doc_type):
        parsed = parser.parse(_raw(build(), filename, doc_type))

        text = " ".join(block.text for block in parsed.blocks)
        assert text.index("First") < text.index("Second")

    def test_character_offsets_are_non_decreasing(self, parser, build, filename, doc_type):
        # Offsets are what a future "highlight in source" feature depends on,
        # and an out-of-order offset is invisible until then.
        parsed = parser.parse(_raw(build(), filename, doc_type))

        starts = [block.char_start for block in parsed.blocks]
        assert starts == sorted(starts)

    def test_every_block_has_a_valid_range(self, parser, build, filename, doc_type):
        parsed = parser.parse(_raw(build(), filename, doc_type))

        assert all(block.char_end >= block.char_start for block in parsed.blocks)

    def test_no_block_is_blank(self, parser, build, filename, doc_type):
        # A whitespace-only block costs an embedding and can never be a useful
        # citation.
        parsed = parser.parse(_raw(build(), filename, doc_type))

        assert all(block.text.strip() for block in parsed.blocks)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
class TestPdfParser:
    def test_each_page_is_numbered_from_one(self):
        # The single most valuable thing a PDF citation can carry.
        raw = _raw(make_pdf(["Alpha content.", "Beta content."]), "a.pdf", DocumentType.PDF)

        parsed = PdfParser().parse(raw)

        pages = {block.page_number for block in parsed.blocks}
        assert pages == {1, 2}

    def test_text_is_attributed_to_the_page_it_came_from(self):
        raw = _raw(make_pdf(["Alpha content.", "Beta content."]), "a.pdf", DocumentType.PDF)

        parsed = PdfParser().parse(raw)

        by_page = {block.page_number: block.text for block in parsed.blocks}
        assert "Alpha" in by_page[1]
        assert "Beta" in by_page[2]

    def test_document_properties_are_extracted(self):
        raw = _raw(
            make_pdf(["Content."], title="Annual Report", author="Ada Lovelace"),
            "a.pdf",
            DocumentType.PDF,
        )

        parsed = PdfParser().parse(raw)

        assert parsed.properties.title == "Annual Report"
        assert parsed.properties.author == "Ada Lovelace"

    def test_the_page_count_is_recorded(self):
        raw = _raw(make_pdf(["One.", "Two.", "Three."]), "a.pdf", DocumentType.PDF)

        assert PdfParser().parse(raw).properties.page_count == 3

    def test_placeholder_properties_are_treated_as_absent(self):
        # Many PDF producers write "anonymous" / "untitled" when they have
        # nothing real. Keeping those is worse than dropping them: they are
        # filterable and look like facts.
        raw = _raw(make_pdf(["Content."]), "a.pdf", DocumentType.PDF)

        parsed = PdfParser().parse(raw)

        assert parsed.properties.author is None
        assert parsed.properties.title is None

    def test_a_real_author_is_kept(self):
        raw = _raw(make_pdf(["Content."], author="Ada Lovelace"), "a.pdf", DocumentType.PDF)

        assert PdfParser().parse(raw).properties.author == "Ada Lovelace"

    def test_a_page_with_no_extractable_text_is_skipped(self):
        # A scanned page. OCR is out of scope, so producing no block for it is
        # the correct outcome rather than an error.
        raw = _raw(make_pdf(["Real text.", "", "More text."]), "a.pdf", DocumentType.PDF)

        parsed = PdfParser().parse(raw)

        assert {block.page_number for block in parsed.blocks} == {1, 3}

    def test_a_pdf_with_no_text_at_all_parses_to_nothing(self):
        raw = _raw(make_pdf(["", ""]), "a.pdf", DocumentType.PDF)

        parsed = PdfParser().parse(raw)

        assert parsed.is_empty is True
        assert parsed.properties.page_count == 2

    def test_a_corrupt_file_is_rejected_clearly(self):
        raw = _raw(b"%PDF-1.4\nthis is not really a pdf", "a.pdf", DocumentType.PDF)

        with pytest.raises(DocumentParsingError):
            PdfParser().parse(raw)


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
class TestDocxParser:
    def test_headings_become_the_heading_path(self):
        raw = _raw(
            make_docx(
                [
                    ("Heading 1", "Financials"),
                    ("Heading 2", "Revenue"),
                    ("Normal", "Revenue grew by twelve percent."),
                ]
            ),
            "a.docx",
            DocumentType.DOCX,
        )

        parsed = DocxParser().parse(raw)

        body = next(b for b in parsed.blocks if "twelve percent" in b.text)
        assert body.heading_path == ("Financials", "Revenue")

    def test_the_nearest_heading_is_recorded_separately(self):
        raw = _raw(
            make_docx(
                [
                    ("Heading 1", "Financials"),
                    ("Heading 2", "Revenue"),
                    ("Normal", "Revenue grew."),
                ]
            ),
            "a.docx",
            DocumentType.DOCX,
        )

        parsed = DocxParser().parse(raw)

        body = next(b for b in parsed.blocks if "Revenue grew" in b.text)
        assert body.heading == "Revenue"
        assert body.section == "Financials > Revenue"

    def test_a_deeper_heading_replaces_only_its_own_level(self):
        raw = _raw(
            make_docx(
                [
                    ("Heading 1", "Financials"),
                    ("Heading 2", "Revenue"),
                    ("Normal", "First."),
                    ("Heading 2", "Costs"),
                    ("Normal", "Second."),
                ]
            ),
            "a.docx",
            DocumentType.DOCX,
        )

        parsed = DocxParser().parse(raw)

        second = next(b for b in parsed.blocks if "Second." in b.text)
        assert second.heading_path == ("Financials", "Costs")

    def test_a_shallower_heading_discards_deeper_levels(self):
        raw = _raw(
            make_docx(
                [
                    ("Heading 1", "Financials"),
                    ("Heading 2", "Revenue"),
                    ("Heading 1", "Operations"),
                    ("Normal", "Body text."),
                ]
            ),
            "a.docx",
            DocumentType.DOCX,
        )

        parsed = DocxParser().parse(raw)

        body = next(b for b in parsed.blocks if "Body text." in b.text)
        assert body.heading_path == ("Operations",)

    def test_text_before_any_heading_has_no_heading_path(self):
        raw = _raw(
            make_docx([("Normal", "Preamble text."), ("Heading 1", "Later")]),
            "a.docx",
            DocumentType.DOCX,
        )

        parsed = DocxParser().parse(raw)

        preamble = next(b for b in parsed.blocks if "Preamble" in b.text)
        assert preamble.heading_path is None

    def test_headings_are_kept_as_blocks_in_their_own_right(self):
        # "What does section 3.2 say?" should be able to match the heading.
        raw = _raw(
            make_docx([("Heading 1", "Financials"), ("Normal", "Body.")]),
            "a.docx",
            DocumentType.DOCX,
        )

        parsed = DocxParser().parse(raw)

        assert any(block.text == "Financials" for block in parsed.blocks)

    def test_document_properties_are_extracted(self):
        raw = _raw(
            make_docx([("Normal", "Body.")], title="Handbook", author="Ada Lovelace"),
            "a.docx",
            DocumentType.DOCX,
        )

        parsed = DocxParser().parse(raw)

        assert parsed.properties.title == "Handbook"
        assert parsed.properties.author == "Ada Lovelace"

    def test_docx_has_no_page_numbers(self):
        # Word documents have no fixed pagination; claiming a page would be a
        # citation that points nowhere.
        raw = _raw(make_docx([("Normal", "Body.")]), "a.docx", DocumentType.DOCX)

        parsed = DocxParser().parse(raw)

        assert all(block.page_number is None for block in parsed.blocks)

    def test_a_corrupt_file_is_rejected_clearly(self):
        raw = _raw(b"PK\x03\x04not-really-a-docx", "a.docx", DocumentType.DOCX)

        with pytest.raises((DocumentParsingError, CorruptDocumentError)):
            DocxParser().parse(raw)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
class TestMarkdownParser:
    def test_atx_headings_become_the_heading_path(self):
        source = "# Financials\n\n## Revenue\n\nRevenue grew by twelve percent.\n"
        raw = _raw(make_markdown(source), "a.md", DocumentType.MARKDOWN)

        parsed = MarkdownParser().parse(raw)

        body = next(b for b in parsed.blocks if "twelve percent" in b.text)
        assert body.heading_path == ("Financials", "Revenue")

    def test_a_shallower_heading_discards_deeper_levels(self):
        source = "# One\n\n## Two\n\n# Three\n\nBody text.\n"
        raw = _raw(make_markdown(source), "a.md", DocumentType.MARKDOWN)

        parsed = MarkdownParser().parse(raw)

        body = next(b for b in parsed.blocks if "Body text." in b.text)
        assert body.heading_path == ("Three",)

    def test_the_section_is_the_full_breadcrumb(self):
        source = "# Financials\n\n## Revenue\n\nBody.\n"
        raw = _raw(make_markdown(source), "a.md", DocumentType.MARKDOWN)

        parsed = MarkdownParser().parse(raw)

        body = next(b for b in parsed.blocks if b.text == "Body.")
        assert body.section == "Financials > Revenue"

    def test_a_hash_inside_a_fenced_code_block_is_not_a_heading(self):
        # Otherwise a shell comment silently restructures the document.
        source = "# Real Heading\n\n```bash\n# not a heading\necho hi\n```\n\nBody.\n"
        raw = _raw(make_markdown(source), "a.md", DocumentType.MARKDOWN)

        parsed = MarkdownParser().parse(raw)

        body = next(b for b in parsed.blocks if b.text == "Body.")
        assert body.heading_path == ("Real Heading",)

    def test_code_blocks_are_kept_as_content(self):
        source = "# Setup\n\n```bash\npip install rag\n```\n"
        raw = _raw(make_markdown(source), "a.md", DocumentType.MARKDOWN)

        parsed = MarkdownParser().parse(raw)

        assert any("pip install rag" in block.text for block in parsed.blocks)

    def test_heading_markers_are_stripped_from_the_heading_text(self):
        raw = _raw(make_markdown("# Financials\n\nBody.\n"), "a.md", DocumentType.MARKDOWN)

        parsed = MarkdownParser().parse(raw)

        assert any(block.text == "Financials" for block in parsed.blocks)

    def test_markdown_has_no_page_numbers(self):
        raw = _raw(make_markdown("# One\n\nBody.\n"), "a.md", DocumentType.MARKDOWN)

        parsed = MarkdownParser().parse(raw)

        assert all(block.page_number is None for block in parsed.blocks)

    def test_a_document_with_no_headings_still_parses(self):
        raw = _raw(make_markdown("Just prose.\n\nMore prose.\n"), "a.md", DocumentType.MARKDOWN)

        parsed = MarkdownParser().parse(raw)

        assert len(parsed.blocks) == 2
        assert all(block.heading_path is None for block in parsed.blocks)
