"""Behaviour of the chunk metadata schema (architecture spec section 8.3).

Every chunk carries the full field set. Fields that cannot be determined are
explicitly ``None`` -- never absent, never an empty string -- so that "unknown
author" and "author not extracted" stay distinguishable in filters.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from rag.domain.models import (
    FILTERABLE_FIELDS,
    ChunkMetadata,
    DocumentType,
    MetadataField,
)

pytestmark = pytest.mark.unit


def _metadata(**overrides: object) -> ChunkMetadata:
    """Build valid metadata, overriding individual fields."""
    defaults: dict[str, object] = {
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "ingest_version": 1,
        "filename": "report.pdf",
        "document_type": DocumentType.PDF,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 120,
        "token_count": 30,
        "chunking_strategy": "recursive",
        "embedding_model_id": "text-embedding-3-small",
        "ingested_at": datetime(2026, 7, 27, tzinfo=UTC),
    }
    return ChunkMetadata(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestOptionalFieldsDefaultToNone:
    @pytest.mark.parametrize(
        "field",
        [
            "page_number",
            "section",
            "heading",
            "heading_path",
            "author",
            "title",
            "created_at",
            "language",
        ],
    )
    def test_undetermined_fields_are_none_not_empty(self, field):
        assert getattr(_metadata(), field) is None


class TestValidation:
    def test_negative_chunk_index_is_rejected(self):
        with pytest.raises(ValueError, match="chunk_index"):
            _metadata(chunk_index=-1)

    def test_page_numbers_are_one_based(self):
        with pytest.raises(ValueError, match="page_number"):
            _metadata(page_number=0)

    def test_valid_page_number_is_accepted(self):
        assert _metadata(page_number=1).page_number == 1

    def test_ingest_version_must_be_positive(self):
        with pytest.raises(ValueError, match="ingest_version"):
            _metadata(ingest_version=0)

    def test_char_range_must_not_be_inverted(self):
        with pytest.raises(ValueError, match="char_start"):
            _metadata(char_start=200, char_end=100)

    def test_negative_token_count_is_rejected(self):
        with pytest.raises(ValueError, match="token_count"):
            _metadata(token_count=-1)

    def test_empty_document_id_is_rejected(self):
        with pytest.raises(ValueError, match="document_id"):
            _metadata(document_id="")


class TestImmutability:
    def test_metadata_cannot_be_mutated(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _metadata().chunk_index = 5  # type: ignore[misc]

    def test_heading_path_is_a_tuple_so_it_cannot_be_appended_to(self):
        metadata = _metadata(heading_path=("3", "3.2"))

        assert metadata.heading_path == ("3", "3.2")
        assert isinstance(metadata.heading_path, tuple)


class TestFilterableFields:
    @pytest.mark.parametrize(
        "field",
        [
            MetadataField.DOCUMENT_TYPE,
            MetadataField.PAGE_NUMBER,
            MetadataField.AUTHOR,
            MetadataField.SECTION,
            MetadataField.INGEST_VERSION,
            MetadataField.LANGUAGE,
        ],
    )
    def test_user_facing_filter_fields_are_filterable(self, field):
        assert field in FILTERABLE_FIELDS

    @pytest.mark.parametrize(
        "field",
        [
            MetadataField.CHAR_START,
            MetadataField.CHAR_END,
            MetadataField.TOKEN_COUNT,
            MetadataField.CHUNKING_STRATEGY,
            MetadataField.EMBEDDING_MODEL_ID,
        ],
    )
    def test_internal_provenance_fields_are_not_filterable(self, field):
        # Indexing every field would inflate memory for no query benefit.
        assert field not in FILTERABLE_FIELDS

    def test_every_metadata_field_maps_to_an_attribute_on_the_dataclass(self):
        attributes = {f.name for f in dataclasses.fields(ChunkMetadata)}

        assert {field.value for field in MetadataField} <= attributes


class TestDocumentType:
    def test_the_four_supported_formats_exist(self):
        assert {t.value for t in DocumentType} == {"PDF", "DOCX", "TXT", "MARKDOWN"}
