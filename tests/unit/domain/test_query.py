"""Behaviour of the query value objects."""

from __future__ import annotations

import dataclasses

import pytest

from rag.domain.models import (
    FieldFilter,
    FilterOperator,
    MetadataField,
    Query,
    RetrievalRequest,
)

pytestmark = pytest.mark.unit


class TestQuery:
    def test_carries_the_question(self):
        assert Query(text="Who wrote this?").text == "Who wrote this?"

    def test_empty_question_is_rejected(self):
        with pytest.raises(ValueError, match="text"):
            Query(text="   ")

    def test_conversation_and_filters_are_optional(self):
        query = Query(text="Who wrote this?")

        assert query.conversation_id is None
        assert query.filters is None

    def test_carries_an_optional_metadata_filter(self):
        clause = FieldFilter(MetadataField.DOCUMENT_TYPE, FilterOperator.EQ, "PDF")

        assert Query(text="Revenue?", filters=clause).filters is clause

    def test_top_k_overrides_are_absent_by_default(self):
        # Absent means "use the configured value"; the query does not know it,
        # because configuration is injected into services rather than models.
        query = Query(text="Revenue?")

        assert query.top_k is None
        assert query.rerank_top_k is None

    def test_top_k_override_must_be_positive(self):
        with pytest.raises(ValueError, match="top_k"):
            Query(text="Revenue?", top_k=0)

    def test_rerank_override_must_be_positive(self):
        with pytest.raises(ValueError, match="rerank_top_k"):
            Query(text="Revenue?", rerank_top_k=0)

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            Query(text="Revenue?").text = "other"  # type: ignore[misc]


class TestRetrievalRequest:
    def test_carries_the_query_text_and_breadth(self):
        request = RetrievalRequest(query_text="Revenue?", top_k=20)

        assert request.query_text == "Revenue?"
        assert request.top_k == 20

    def test_top_k_must_be_positive(self):
        with pytest.raises(ValueError, match="top_k"):
            RetrievalRequest(query_text="Revenue?", top_k=0)

    def test_empty_query_text_is_rejected(self):
        with pytest.raises(ValueError, match="query_text"):
            RetrievalRequest(query_text="", top_k=20)

    def test_filters_are_optional(self):
        assert RetrievalRequest(query_text="Revenue?", top_k=5).filters is None

    def test_every_retriever_accepts_the_same_request(self):
        # This is what lets the hybrid retriever be a composite of retrievers
        # rather than a special case in the retrieval service.
        request = RetrievalRequest(query_text="Revenue?", top_k=5)

        assert dataclasses.asdict(request).keys() == {"query_text", "top_k", "filters"}
