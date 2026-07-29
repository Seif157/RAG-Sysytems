"""Behaviour of the value objects that appear in port signatures.

These types exist so that ports can be declared without leaking a vendor type
into the domain: a parser returns a ``ParsedDocument``, not a LlamaIndex node; an
embedder returns vectors, not an OpenAI response object.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from rag.domain.models import (
    CollectionInfo,
    CollectionSpec,
    ContentBlock,
    DistanceMetric,
    DocumentProperties,
    DocumentType,
    GenerationParams,
    LLMResponse,
    MetadataField,
    MetadataFragment,
    ParsedDocument,
    Prompt,
    RawDocument,
    SparseVector,
    TokenUsage,
)

pytestmark = pytest.mark.unit


class TestSparseVector:
    def test_pairs_indices_with_values(self):
        vector = SparseVector(indices=(3, 17), values=(0.5, 0.25))

        assert vector.indices == (3, 17)
        assert vector.values == (0.5, 0.25)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            SparseVector(indices=(1, 2), values=(0.5,))

    def test_negative_indices_are_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            SparseVector(indices=(-1,), values=(0.5,))

    def test_duplicate_indices_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            SparseVector(indices=(4, 4), values=(0.5, 0.25))

    def test_an_empty_sparse_vector_is_valid(self):
        # A query sharing no term with the vocabulary produces one.
        assert SparseVector(indices=(), values=()).non_zero_count == 0

    def test_reports_how_many_terms_it_carries(self):
        assert SparseVector(indices=(1, 2, 3), values=(0.1, 0.2, 0.3)).non_zero_count == 3


class TestRawDocument:
    def test_carries_the_original_bytes_and_detected_type(self):
        raw = RawDocument(
            filename="notes.txt",
            content=b"hello",
            document_type=DocumentType.TXT,
            content_hash="a" * 64,
        )

        assert raw.content == b"hello"
        assert raw.document_type is DocumentType.TXT

    def test_size_is_derived_from_the_content(self):
        raw = RawDocument(
            filename="notes.txt",
            content=b"hello",
            document_type=DocumentType.TXT,
            content_hash="a" * 64,
        )

        assert raw.size_bytes == 5

    def test_empty_content_is_rejected(self):
        with pytest.raises(ValueError, match="content"):
            RawDocument(
                filename="notes.txt",
                content=b"",
                document_type=DocumentType.TXT,
                content_hash="a" * 64,
            )


class TestContentBlock:
    def test_carries_text_and_its_position_in_the_source(self):
        block = ContentBlock(text="Revenue grew.", char_start=0, char_end=13, page_number=4)

        assert block.text == "Revenue grew."
        assert block.char_start == 0
        assert block.page_number == 4

    def test_structural_position_is_optional_for_unpaginated_formats(self):
        block = ContentBlock(text="Revenue grew.", char_start=0, char_end=13)

        assert block.page_number is None
        assert block.heading_path is None

    def test_inverted_character_range_is_rejected(self):
        with pytest.raises(ValueError, match="char_start"):
            ContentBlock(text="x", char_start=10, char_end=2)

    def test_is_immutable(self):
        block = ContentBlock(text="x", char_start=0, char_end=1)

        with pytest.raises(dataclasses.FrozenInstanceError):
            block.text = "y"  # type: ignore[misc]


class TestParsedDocument:
    def test_holds_ordered_blocks_and_document_properties(self):
        parsed = ParsedDocument(
            blocks=(
                ContentBlock(text="One.", char_start=0, char_end=4),
                ContentBlock(text="Two.", char_start=5, char_end=9),
            ),
            properties=DocumentProperties(author="Ada"),
        )

        assert len(parsed.blocks) == 2
        assert parsed.properties.author == "Ada"

    def test_reports_block_count(self):
        parsed = ParsedDocument(
            blocks=(ContentBlock(text="One.", char_start=0, char_end=4),),
            properties=DocumentProperties(),
        )

        assert parsed.block_count == 1

    def test_a_document_with_no_extractable_text_is_representable(self):
        # An empty scanned PDF is a real outcome; parsers must not crash on it.
        # OCR is explicitly out of scope, so zero blocks is the correct result.
        parsed = ParsedDocument(blocks=(), properties=DocumentProperties())

        assert parsed.block_count == 0
        assert parsed.is_empty is True

    def test_document_properties_default_to_unknown_rather_than_blank(self):
        properties = DocumentProperties()

        assert properties.author is None
        assert properties.title is None
        assert properties.created_at is None
        assert properties.page_count is None


class TestMetadataFragment:
    def test_carries_partial_metadata_keyed_by_field(self):
        fragment = MetadataFragment.of({MetadataField.AUTHOR: "Ada"})

        assert fragment.as_dict() == {MetadataField.AUTHOR: "Ada"}

    def test_an_extractor_that_found_nothing_yields_an_empty_fragment(self):
        assert MetadataFragment.empty().as_dict() == {}

    def test_empty_fragment_is_falsy_so_chains_can_skip_it(self):
        assert not MetadataFragment.empty()
        assert MetadataFragment.of({MetadataField.AUTHOR: "Ada"})

    def test_is_hashable_so_fragments_can_be_deduplicated(self):
        one = MetadataFragment.of({MetadataField.AUTHOR: "Ada"})
        two = MetadataFragment.of({MetadataField.AUTHOR: "Ada"})

        assert len({one, two}) == 1


class TestPrompt:
    def test_carries_system_and_user_parts_with_a_version(self):
        prompt = Prompt(system="You answer from documents.", user="Q: Revenue?", version="v1")

        assert prompt.system.startswith("You answer")
        assert prompt.version == "v1"

    def test_version_is_required_so_answers_stay_reproducible(self):
        with pytest.raises(ValueError, match="version"):
            Prompt(system="s", user="u", version="")

    def test_empty_user_part_is_rejected(self):
        with pytest.raises(ValueError, match="user"):
            Prompt(system="s", user="", version="v1")


class TestGenerationParams:
    def test_carries_the_knobs_a_provider_needs(self):
        params = GenerationParams(temperature=0.0, max_output_tokens=2048, timeout_s=60.0)

        assert params.temperature == pytest.approx(0.0)
        assert params.max_output_tokens == 2048

    def test_negative_temperature_is_rejected(self):
        with pytest.raises(ValueError, match="temperature"):
            GenerationParams(temperature=-0.1, max_output_tokens=10, timeout_s=1.0)

    def test_non_positive_output_budget_is_rejected(self):
        with pytest.raises(ValueError, match="max_output_tokens"):
            GenerationParams(temperature=0.0, max_output_tokens=0, timeout_s=1.0)

    def test_is_deterministic_reports_whether_responses_are_cacheable(self):
        # The LLM response cache is only sound at temperature 0 (ADR-010).
        assert GenerationParams(0.0, 100, 1.0).is_deterministic is True
        assert GenerationParams(0.7, 100, 1.0).is_deterministic is False


class TestLLMResponse:
    def test_carries_text_model_and_usage(self):
        response = LLMResponse(
            text="Revenue grew 12%.",
            model_id="gemini-2.0-flash",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
        )

        assert response.text == "Revenue grew 12%."
        assert response.usage is not None
        assert response.usage.total_tokens == 120

    def test_usage_is_optional_because_not_every_provider_reports_it(self):
        assert LLMResponse(text="x", model_id="m").usage is None

    def test_records_why_generation_stopped(self):
        response = LLMResponse(text="x", model_id="m", finish_reason="max_tokens")

        assert response.finish_reason == "max_tokens"


class TestCollectionCompatibility:
    def _spec(self, **overrides: object) -> CollectionSpec:
        defaults: dict[str, object] = {
            "name": "document_chunks",
            "dense_dimension": 1536,
            "distance": DistanceMetric.COSINE,
            "embedding_model_id": "text-embedding-3-small",
            "supports_sparse": True,
        }
        return CollectionSpec(**{**defaults, **overrides})  # type: ignore[arg-type]

    def _info(self, **overrides: object) -> CollectionInfo:
        defaults: dict[str, object] = {
            "name": "document_chunks",
            "dense_dimension": 1536,
            "distance": DistanceMetric.COSINE,
            "embedding_model_id": "text-embedding-3-small",
            "supports_sparse": True,
            "points_count": 4096,
        }
        return CollectionInfo(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_a_matching_collection_is_compatible(self):
        assert self._info().is_compatible_with(self._spec()) is True

    def test_a_different_dimension_is_incompatible(self):
        # Silent dimension mismatch is the worst failure mode in the system:
        # confidently wrong answers with plausible citations (ADR-015).
        assert self._info(dense_dimension=768).is_compatible_with(self._spec()) is False

    def test_a_different_embedding_model_is_incompatible(self):
        info = self._info(embedding_model_id="models/embedding-001")

        assert info.is_compatible_with(self._spec()) is False

    def test_a_different_distance_metric_is_incompatible(self):
        assert self._info(distance=DistanceMetric.DOT).is_compatible_with(self._spec()) is False

    def test_losing_sparse_support_is_incompatible(self):
        assert self._info(supports_sparse=False).is_compatible_with(self._spec()) is False

    def test_incompatibility_explains_itself(self):
        reasons = self._info(dense_dimension=768).incompatibility_reasons(self._spec())

        assert any("dimension" in reason for reason in reasons)

    def test_a_compatible_collection_reports_no_reasons(self):
        assert self._info().incompatibility_reasons(self._spec()) == ()

    def test_dimension_must_be_positive(self):
        with pytest.raises(ValueError, match="dense_dimension"):
            self._spec(dense_dimension=0)


class TestDocumentPropertiesDates:
    def test_created_at_is_retained_when_the_document_declares_one(self):
        when = datetime(2024, 1, 1, tzinfo=UTC)

        assert DocumentProperties(created_at=when).created_at == when
