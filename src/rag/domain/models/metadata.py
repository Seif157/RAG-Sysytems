"""Chunk metadata: the schema every stored chunk carries.

Implements the schema in section 8.3 of the architecture specification.

Two rules shape this module:

1. **Undetermined means ``None``, never absent and never empty string.** A
   filter must be able to distinguish "this document has no author" from "the
   author was never extracted".
2. **Only some fields are filterable.** Payload indexes cost memory, so they are
   created for fields users actually query. :data:`FILTERABLE_FIELDS` is the
   single source of truth, and the filter expression tree enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

__all__ = [
    "FILTERABLE_FIELDS",
    "ChunkMetadata",
    "DocumentAccessPolicy",
    "DocumentAccessScope",
    "DocumentType",
    "MetadataField",
]


class DocumentType(StrEnum):
    """A supported document format.

    Adding a format means adding a member here, a parser, and one registry
    entry -- two edits to existing files (architecture spec section 15.6).
    """

    PDF = "PDF"
    DOCX = "DOCX"
    TXT = "TXT"
    MARKDOWN = "MARKDOWN"


class DocumentAccessScope(StrEnum):
    """Audience boundary applied before retrieval."""

    ORGANISATION = "ORGANISATION"
    BRANCH = "BRANCH"
    DEPARTMENT = "DEPARTMENT"
    EMPLOYEE = "EMPLOYEE"


@dataclass(frozen=True, slots=True)
class DocumentAccessPolicy:
    """Access metadata supplied by a trusted ingestion caller."""

    scope: DocumentAccessScope
    scope_key: str | None = None
    required_permission: str | None = None

    def __post_init__(self) -> None:
        if self.scope is DocumentAccessScope.ORGANISATION:
            if self.scope_key is not None:
                raise ValueError("organisation documents must not have a scope_key")
        elif self.scope_key is None or not self.scope_key.strip():
            raise ValueError(f"{self.scope.value} documents require a non-empty scope_key")
        if self.required_permission is not None and not self.required_permission.strip():
            raise ValueError("required_permission must be non-empty when supplied")


class MetadataField(StrEnum):
    """Names of the fields on :class:`ChunkMetadata`.

    Using an enum rather than raw strings means a filter cannot be built against
    a misspelled field name, and renaming a field is a type error rather than a
    silent no-match at query time.

    Every member's value equals the corresponding dataclass attribute name.
    """

    DOCUMENT_ID = "document_id"
    CHUNK_ID = "chunk_id"
    INGEST_VERSION = "ingest_version"
    FILENAME = "filename"
    DOCUMENT_TYPE = "document_type"
    PAGE_NUMBER = "page_number"
    SECTION = "section"
    HEADING = "heading"
    HEADING_PATH = "heading_path"
    AUTHOR = "author"
    TITLE = "title"
    CREATED_AT = "created_at"
    INGESTED_AT = "ingested_at"
    LANGUAGE = "language"
    CHUNK_INDEX = "chunk_index"
    CHAR_START = "char_start"
    CHAR_END = "char_end"
    TOKEN_COUNT = "token_count"
    CHUNKING_STRATEGY = "chunking_strategy"
    EMBEDDING_MODEL_ID = "embedding_model_id"
    ACCESS_SCOPE = "access_scope"
    ACCESS_SCOPE_KEY = "access_scope_key"
    REQUIRED_PERMISSION = "required_permission"


#: Fields that may appear in a retrieval filter and therefore receive a payload
#: index. The excluded fields are internal provenance and offsets: useful when
#: debugging or highlighting a source, never worth an index.
FILTERABLE_FIELDS: frozenset[MetadataField] = frozenset(
    {
        MetadataField.DOCUMENT_ID,
        MetadataField.CHUNK_ID,
        MetadataField.INGEST_VERSION,
        MetadataField.FILENAME,
        MetadataField.DOCUMENT_TYPE,
        MetadataField.PAGE_NUMBER,
        MetadataField.SECTION,
        MetadataField.HEADING,
        MetadataField.HEADING_PATH,
        MetadataField.AUTHOR,
        MetadataField.TITLE,
        MetadataField.CREATED_AT,
        MetadataField.INGESTED_AT,
        MetadataField.LANGUAGE,
        MetadataField.CHUNK_INDEX,
        MetadataField.ACCESS_SCOPE,
        MetadataField.ACCESS_SCOPE_KEY,
        MetadataField.REQUIRED_PERMISSION,
    }
)


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Metadata attached to every indexed chunk.

    Attributes:
        document_id: Identifier of the source document.
        chunk_id: Deterministic chunk identifier (ADR-014).
        ingest_version: Monotonic version of the document this chunk belongs to.
            Retrieval filters on the current version, which is what makes a
            re-index invisible to readers.
        filename: Original uploaded filename.
        document_type: Format of the source document.
        chunk_index: Zero-based ordinal of this chunk within the document.
        char_start: Character offset of the chunk's start in the parsed text.
        char_end: Character offset of the chunk's end in the parsed text.
        token_count: Token count of the chunk text, used for context budgeting.
        chunking_strategy: Strategy that produced this chunk, for provenance.
        embedding_model_id: Model whose vector represents this chunk.
        ingested_at: When this chunk was written to the index.
        page_number: One-based page for paginated formats; ``None`` for TXT and
            Markdown, which have no pages.
        section: Nearest enclosing section title, if the parser found one.
        heading: Nearest heading above the chunk, if any.
        heading_path: Full heading breadcrumb, e.g. ``("3", "3.2")``.
        author: Document author from its own properties.
        title: Document title from its own properties.
        created_at: The document's own creation date, distinct from
            :attr:`ingested_at`.
        language: Detected ISO 639-1 language code.
    """

    document_id: str
    chunk_id: str
    ingest_version: int
    filename: str
    document_type: DocumentType
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    chunking_strategy: str
    embedding_model_id: str
    ingested_at: datetime

    page_number: int | None = None
    section: str | None = None
    heading: str | None = None
    heading_path: tuple[str, ...] | None = None
    author: str | None = None
    title: str | None = None
    created_at: datetime | None = None
    language: str | None = None
    access_scope: DocumentAccessScope | None = None
    access_scope_key: str | None = None
    required_permission: str | None = None

    def __post_init__(self) -> None:
        """Validate structural invariants that no extractor may violate."""
        if not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")
        if self.ingest_version < 1:
            raise ValueError("ingest_version must be >= 1")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        if self.char_start < 0:
            raise ValueError("char_start must be >= 0")
        if self.char_end < self.char_start:
            raise ValueError("char_start must not exceed char_end")
        if self.token_count < 0:
            raise ValueError("token_count must be >= 0")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number is one-based and must be >= 1")
        if self.access_scope is DocumentAccessScope.ORGANISATION:
            if self.access_scope_key is not None:
                raise ValueError("organisation metadata must not have an access_scope_key")
        elif self.access_scope is not None and (
            self.access_scope_key is None or not self.access_scope_key.strip()
        ):
            raise ValueError(f"{self.access_scope.value} metadata requires access_scope_key")

        if self.heading_path is not None:
            # Coerce so the value object stays immutable and hashable even when
            # an extractor hands us a list. Adapters sit at an untyped boundary,
            # so the annotation alone is not a guarantee. Idempotent for tuples.
            object.__setattr__(self, "heading_path", tuple(self.heading_path))
