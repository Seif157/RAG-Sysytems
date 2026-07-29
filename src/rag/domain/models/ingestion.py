"""Types flowing through the ingestion pipeline.

These exist so that ingestion ports can be declared without any vendor type
reaching the domain: a parser returns a :class:`ParsedDocument`, never a
LlamaIndex node or a ``pypdf`` page object. That is what bounds the cost of
replacing a parser -- or LlamaIndex itself -- to the adapter that produces them
(architecture spec section 15.1).

The division of labour is deliberate:

* A **parser** produces :class:`ContentBlock` values -- text plus where it came
  from. It does not chunk, and it does not clean beyond whitespace.
* A **metadata extractor** produces a :class:`MetadataFragment` -- a partial set
  of field values, which the chain merges.
* A **chunker** turns blocks into chunks.

Each stage can therefore be replaced without disturbing its neighbours.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

from rag.domain.models.metadata import DocumentType, MetadataField

__all__ = [
    "ChunkCandidate",
    "ContentBlock",
    "DocumentProperties",
    "MetadataFragment",
    "ParsedDocument",
    "RawDocument",
]


@dataclass(frozen=True, slots=True)
class RawDocument:
    """An uploaded file, after format detection and before parsing.

    Attributes:
        filename: Original filename as uploaded.
        content: The raw bytes.
        document_type: Format determined by the loader from both the extension
            and the content itself.
        content_hash: Hash of :attr:`content`. An unchanged hash on re-upload
            means ingestion can be skipped entirely.
    """

    filename: str
    content: bytes
    document_type: DocumentType
    content_hash: str

    def __post_init__(self) -> None:
        """Validate the document has a name and a body."""
        if not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        if not self.content:
            raise ValueError("content must not be empty")
        if not self.content_hash.strip():
            raise ValueError("content_hash must be a non-empty string")

    @property
    def size_bytes(self) -> int:
        """Size of the original file, derived rather than stored."""
        return len(self.content)


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """A contiguous run of text with its position in the source document.

    Structural position is what makes a citation useful: "page 14, section 3.2"
    rather than "somewhere in report.pdf".

    Attributes:
        text: The extracted text.
        char_start: Offset of the block's start in the parsed document text.
        char_end: Offset of the block's end.
        page_number: One-based page, for paginated formats. ``None`` for TXT and
            Markdown, which genuinely have no pages.
        section: Nearest enclosing section title, if the format expresses one.
        heading: Nearest heading above this block.
        heading_path: Full heading breadcrumb, e.g. ``("3", "3.2")``.
    """

    text: str
    char_start: int
    char_end: int
    page_number: int | None = None
    section: str | None = None
    heading: str | None = None
    heading_path: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate offsets and page numbering."""
        if self.char_start < 0:
            raise ValueError("char_start must be >= 0")
        if self.char_end < self.char_start:
            raise ValueError("char_start must not exceed char_end")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number is one-based and must be >= 1")


@dataclass(frozen=True, slots=True)
class DocumentProperties:
    """Document-level properties read from the file's own metadata.

    Every field is optional and defaults to ``None`` rather than a blank string,
    so "this document declares no author" stays distinguishable from "the author
    was never extracted" (architecture spec section 8.3).

    Attributes:
        author: Declared author.
        title: Declared title.
        created_at: The document's own creation date.
        language: Detected ISO 639-1 language code.
        page_count: Total pages, for paginated formats.
    """

    author: str | None = None
    title: str | None = None
    created_at: datetime | None = None
    language: str | None = None
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """The output of parsing: ordered blocks plus document-level properties.

    Attributes:
        blocks: Content blocks in document order.
        properties: Document-level metadata.
    """

    blocks: tuple[ContentBlock, ...]
    properties: DocumentProperties = field(default_factory=DocumentProperties)

    @property
    def block_count(self) -> int:
        """Number of extracted blocks."""
        return len(self.blocks)

    @property
    def is_empty(self) -> bool:
        """Whether parsing yielded no text at all.

        A valid outcome, not an error: a scanned PDF contains no extractable
        text, and OCR is explicitly out of scope. The caller decides whether an
        empty parse should fail ingestion.
        """
        return not self.blocks


@dataclass(frozen=True, slots=True)
class ChunkCandidate:
    """A chunk boundary chosen by a chunking strategy, before identity is stamped.

    A chunker decides *where* text should be split and inherits the structural
    position of the block it came from. It knows nothing about tenants, document
    ids, ingest versions or embedding models -- the pipeline stamps those when it
    builds the final :class:`~rag.domain.models.chunk.Chunk`.

    That split is what allows a chunking strategy to be unit tested with no
    tenant, no clock and no configuration, and it is why replacing a strategy
    cannot break identity or provenance.

    Attributes:
        text: The chunk text.
        char_start: Offset of the chunk's start in the parsed document text.
        char_end: Offset of the chunk's end.
        token_count: Token count, used later for context budgeting.
        page_number: Inherited one-based page, where the format has pages.
        section: Inherited nearest section title.
        heading: Inherited nearest heading.
        heading_path: Inherited heading breadcrumb.
    """

    text: str
    char_start: int
    char_end: int
    token_count: int
    page_number: int | None = None
    section: str | None = None
    heading: str | None = None
    heading_path: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the candidate describes a usable chunk."""
        if not self.text or not self.text.strip():
            raise ValueError("chunk candidate text must be a non-empty string")
        if self.char_start < 0:
            raise ValueError("char_start must be >= 0")
        if self.char_end < self.char_start:
            raise ValueError("char_start must not exceed char_end")
        if self.token_count < 0:
            raise ValueError("token_count must be >= 0")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number is one-based and must be >= 1")


@dataclass(frozen=True, slots=True)
class MetadataFragment:
    """A partial set of metadata values produced by one extractor.

    Extractors are composed into a chain. Merge precedence is the chain's
    concern, not the fragment's: a later extractor must not overwrite a non-null
    value an earlier one established.

    Attributes:
        entries: Field/value pairs, held as a tuple so the fragment stays
            immutable and hashable.
    """

    entries: tuple[tuple[MetadataField, Any], ...] = ()

    @classmethod
    def empty(cls) -> Self:
        """Build the fragment an extractor returns when it found nothing.

        Extractor failures are non-fatal: the chain logs and continues with this.
        """
        return cls(entries=())

    @classmethod
    def of(cls, values: Mapping[MetadataField, Any]) -> Self:
        """Build a fragment from a mapping of field values.

        Args:
            values: Field/value pairs this extractor determined.

        Returns:
            An immutable fragment.
        """
        return cls(entries=tuple(sorted(values.items(), key=lambda item: item[0].value)))

    def as_dict(self) -> dict[MetadataField, Any]:
        """Return the fragment's values as a mutable mapping."""
        return dict(self.entries)

    def __bool__(self) -> bool:
        """Whether this fragment carries any value, so chains can skip empties."""
        return bool(self.entries)
