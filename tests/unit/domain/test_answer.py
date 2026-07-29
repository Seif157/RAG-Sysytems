"""Behaviour of the answer and citation value objects.

Correct attribution is the system's top-ranked quality attribute: an answer that
cannot be traced to a chunk is a defect. These tests pin down the shape that
makes that traceable, and the honesty rule about confidence scores -- a
fabricated number is worse than an omitted one (architecture spec section 12.3).
"""

from __future__ import annotations

import dataclasses

import pytest

from rag.domain.models import Answer, Citation, DocumentType, TokenUsage

pytestmark = pytest.mark.unit


def _citation(**overrides: object) -> Citation:
    defaults: dict[str, object] = {
        "chunk_ids": ("chunk-1",),
        "document_id": "doc-1",
        "filename": "report.pdf",
        "document_type": DocumentType.PDF,
    }
    return Citation(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestCitation:
    def test_identifies_the_chunk_and_its_document(self):
        citation = _citation()

        assert citation.chunk_id == "chunk-1"
        assert citation.document_id == "doc-1"
        assert citation.filename == "report.pdf"

    def test_locates_the_passage_within_the_document(self):
        citation = _citation(page_number=14, section="3.2 Revenue", heading="Regional")

        assert citation.page_number == 14
        assert citation.section == "3.2 Revenue"
        assert citation.heading == "Regional"

    def test_relevance_score_is_absent_when_no_reranker_ran(self):
        # Omitting is honest; inventing a number is not.
        assert _citation().relevance_score is None

    def test_relevance_score_is_retained_when_available(self):
        assert _citation(relevance_score=0.87).relevance_score == pytest.approx(0.87)

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_relevance_score_outside_the_unit_interval_is_rejected(self, score):
        with pytest.raises(ValueError, match="relevance_score"):
            _citation(relevance_score=score)

    def test_page_numbers_are_one_based(self):
        with pytest.raises(ValueError, match="page_number"):
            _citation(page_number=0)

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _citation().document_id = "other"  # type: ignore[misc]

    def test_it_reports_every_passage_that_supports_it(self):
        # Several passages from one page are merged into a single citation; the
        # reader sees one source, the system can still audit which passages
        # stood behind it.
        citation = _citation(chunk_ids=("best", "also"))

        assert citation.chunk_id == "best"
        assert citation.supporting_chunk_count == 2

    def test_a_citation_must_name_a_supporting_passage(self):
        with pytest.raises(ValueError, match="at least one"):
            _citation(chunk_ids=())


class TestTokenUsage:
    def test_totals_prompt_and_completion_tokens(self):
        usage = TokenUsage(prompt_tokens=1200, completion_tokens=300)

        assert usage.total_tokens == 1500

    def test_negative_counts_are_rejected(self):
        with pytest.raises(ValueError, match="prompt_tokens"):
            TokenUsage(prompt_tokens=-1, completion_tokens=0)


class TestAnswer:
    def test_carries_the_text_and_its_citations(self):
        answer = Answer(
            text="Revenue grew 12%.",
            citations=(_citation(),),
            model_id="gemini-2.0-flash",
            prompt_version="v1",
        )

        assert answer.text == "Revenue grew 12%."
        assert len(answer.citations) == 1

    def test_empty_answer_text_is_rejected(self):
        # An empty response is a failure, not an answer. "I don't know" is text.
        with pytest.raises(ValueError, match="text"):
            Answer(text="", citations=(), model_id="m", prompt_version="v1")

    def test_an_answer_may_legitimately_have_no_citations(self):
        answer = Answer(
            text="That is not stated in the provided documents.",
            citations=(),
            model_id="m",
            prompt_version="v1",
        )

        assert answer.has_citations is False

    def test_reports_when_it_has_citations(self):
        answer = Answer(
            text="Revenue grew 12%.",
            citations=(_citation(),),
            model_id="m",
            prompt_version="v1",
        )

        assert answer.has_citations is True

    def test_records_the_model_and_prompt_version_for_reproducibility(self):
        answer = Answer(
            text="Revenue grew 12%.",
            citations=(),
            model_id="gemini-2.0-flash",
            prompt_version="v3",
        )

        assert answer.model_id == "gemini-2.0-flash"
        assert answer.prompt_version == "v3"

    def test_is_not_degraded_by_default(self):
        answer = Answer(text="x", citations=(), model_id="m", prompt_version="v1")

        assert answer.is_degraded is False
        assert answer.degraded_stages == ()

    def test_reports_degradation_when_an_optional_stage_failed(self):
        # e.g. the reranker was unavailable and the fusion order was used.
        answer = Answer(
            text="x",
            citations=(),
            model_id="m",
            prompt_version="v1",
            degraded_stages=("rerank",),
        )

        assert answer.is_degraded is True

    def test_is_immutable(self):
        answer = Answer(text="x", citations=(), model_id="m", prompt_version="v1")

        with pytest.raises(dataclasses.FrozenInstanceError):
            answer.text = "changed"  # type: ignore[misc]
