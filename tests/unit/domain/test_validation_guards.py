"""Coverage of the remaining structural guards on domain value objects.

Each of these rejects a state that would otherwise travel a long way before
causing a visible failure -- a blank identifier reaching the vector store, a
negative count reaching a token budget. They are cheap to assert and they are
the reason a malformed object never leaves the layer that built it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.domain.errors import DocumentParsingError
from rag.domain.models import (
    Answer,
    Chunk,
    ChunkCandidate,
    ChunkMetadata,
    Citation,
    CollectionSpec,
    ContentBlock,
    ContextBlock,
    Conversation,
    DistanceMetric,
    DocumentType,
    GenerationParams,
    LLMResponse,
    Prompt,
    Query,
    RawDocument,
    ScoredChunk,
    ScoreSource,
    TokenUsage,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 27, tzinfo=UTC)


def _metadata(**overrides: object) -> ChunkMetadata:
    defaults: dict[str, object] = {
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "ingest_version": 1,
        "filename": "report.pdf",
        "document_type": DocumentType.PDF,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 10,
        "token_count": 5,
        "chunking_strategy": "recursive",
        "embedding_model_id": "text-embedding-3-small",
        "ingested_at": _NOW,
    }
    return ChunkMetadata(**{**defaults, **overrides})  # type: ignore[arg-type]


def _citation(**overrides: object) -> Citation:
    defaults: dict[str, object] = {
        "chunk_ids": ("chunk-1",),
        "document_id": "doc-1",
        "filename": "report.pdf",
        "document_type": DocumentType.PDF,
    }
    return Citation(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestChunkIdentityGuards:
    def test_blank_chunk_id_is_rejected(self):
        with pytest.raises(ValueError, match="chunk_id"):
            _metadata(chunk_id="  ")

    def test_a_chunk_reads_its_document_through_from_metadata(self):
        assert Chunk(text="x", metadata=_metadata()).document_id == "doc-1"


class TestScoredChunkOrderingGuards:
    def test_comparison_with_a_foreign_type_is_declined(self):
        # Returning NotImplemented lets Python raise a proper TypeError rather
        # than silently producing a nonsensical ordering.
        scored = ScoredChunk(
            chunk=Chunk(text="x", metadata=_metadata()), score=0.5, source=ScoreSource.DENSE
        )

        with pytest.raises(TypeError):
            _ = scored < 1  # type: ignore[operator]

        with pytest.raises(TypeError):
            _ = scored > 1  # type: ignore[operator]


class TestCitationGuards:
    @pytest.mark.parametrize("field", ["document_id", "filename"])
    def test_blank_identifiers_are_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            _citation(**{field: "   "})

    def test_a_blank_supporting_chunk_id_is_rejected(self):
        with pytest.raises(ValueError, match="chunk_ids"):
            _citation(chunk_ids=("  ",))


class TestTokenUsageGuards:
    def test_negative_completion_count_is_rejected(self):
        with pytest.raises(ValueError, match="completion_tokens"):
            TokenUsage(prompt_tokens=0, completion_tokens=-1)


class TestAnswerGuards:
    def test_blank_model_id_is_rejected(self):
        with pytest.raises(ValueError, match="model_id"):
            Answer(text="x", citations=(), model_id="  ", prompt_version="v1")

    def test_negative_retrieved_count_is_rejected(self):
        with pytest.raises(ValueError, match="retrieved_count"):
            Answer(text="x", citations=(), model_id="m", prompt_version="v1", retrieved_count=-1)

    def test_negative_reranked_count_is_rejected(self):
        with pytest.raises(ValueError, match="reranked_count"):
            Answer(text="x", citations=(), model_id="m", prompt_version="v1", reranked_count=-1)


class TestContextBlockGuards:
    def test_negative_token_count_is_rejected(self):
        with pytest.raises(ValueError, match="token_count"):
            ContextBlock(chunks=(), rendered="", token_count=-1, marker_to_chunk_id=())

    def test_negative_dropped_count_is_rejected(self):
        with pytest.raises(ValueError, match="dropped_count"):
            ContextBlock(
                chunks=(), rendered="", token_count=0, marker_to_chunk_id=(), dropped_count=-1
            )


class TestConversationGuards:
    def test_blank_conversation_id_is_rejected(self):
        with pytest.raises(ValueError, match="conversation_id"):
            Conversation(conversation_id="   ")


class TestCollectionSpecGuards:
    def test_blank_name_is_rejected(self):
        with pytest.raises(ValueError, match="name"):
            CollectionSpec(
                name="  ",
                dense_dimension=1536,
                distance=DistanceMetric.COSINE,
                embedding_model_id="m",
            )

    def test_blank_embedding_model_is_rejected(self):
        # A collection with no recorded model cannot be checked for
        # compatibility, which defeats the guard that ADR-015 depends on.
        with pytest.raises(ValueError, match="embedding_model_id"):
            CollectionSpec(
                name="c",
                dense_dimension=1536,
                distance=DistanceMetric.COSINE,
                embedding_model_id="   ",
            )


class TestGenerationGuards:
    def test_prompt_rejects_a_negative_token_estimate(self):
        with pytest.raises(ValueError, match="estimated_tokens"):
            Prompt(system="s", user="u", version="v1", estimated_tokens=-1)

    def test_generation_params_reject_a_non_positive_timeout(self):
        with pytest.raises(ValueError, match="timeout_s"):
            GenerationParams(temperature=0.0, max_output_tokens=10, timeout_s=0.0)

    def test_llm_response_requires_an_attributable_model(self):
        # An answer whose model is unknown cannot be reproduced or compared.
        with pytest.raises(ValueError, match="model_id"):
            LLMResponse(text="x", model_id="   ")


class TestRawDocumentGuards:
    def test_blank_filename_is_rejected(self):
        with pytest.raises(ValueError, match="filename"):
            RawDocument(
                filename="  ",
                content=b"x",
                document_type=DocumentType.TXT,
                content_hash="a" * 64,
            )

    def test_blank_content_hash_is_rejected(self):
        # The hash is what makes a re-upload of unchanged content free.
        with pytest.raises(ValueError, match="content_hash"):
            RawDocument(
                filename="a.txt",
                content=b"x",
                document_type=DocumentType.TXT,
                content_hash="  ",
            )


class TestContentBlockGuards:
    def test_negative_offset_is_rejected(self):
        with pytest.raises(ValueError, match="char_start"):
            ContentBlock(text="x", char_start=-1, char_end=5)

    def test_page_numbers_are_one_based(self):
        with pytest.raises(ValueError, match="page_number"):
            ContentBlock(text="x", char_start=0, char_end=1, page_number=0)


class TestChunkCandidateGuards:
    def test_page_numbers_are_one_based(self):
        with pytest.raises(ValueError, match="page_number"):
            ChunkCandidate(text="x", char_start=0, char_end=1, token_count=1, page_number=0)

    def test_structural_position_is_inherited_from_the_source_block(self):
        candidate = ChunkCandidate(
            text="x",
            char_start=0,
            char_end=1,
            token_count=1,
            page_number=4,
            heading_path=("3", "3.2"),
        )

        assert candidate.page_number == 4
        assert candidate.heading_path == ("3", "3.2")


class TestQueryGuards:
    def test_rerank_override_must_be_positive(self):
        with pytest.raises(ValueError, match="rerank_top_k"):
            Query(text="q", rerank_top_k=0)


class TestErrorRepresentation:
    def test_repr_shows_the_fields_that_matter_when_debugging(self):
        rendered = repr(DocumentParsingError("bad page", context={"page": 12}))

        assert "DocumentParsingError" in rendered
        assert "bad page" in rendered
        assert "retryable" in rendered
