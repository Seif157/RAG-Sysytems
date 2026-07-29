"""Behaviour of the domain exception hierarchy.

The prototype's ``customexception`` discarded ``super().__init__``, dropped the
original traceback and printed to stdout. These tests pin down the behaviour
that replaces it (ADR-017).
"""

from __future__ import annotations

import pytest

from rag.domain.errors import (
    ConfigurationError,
    DocumentParsingError,
    DocumentProcessingError,
    EmbeddingError,
    EmbeddingRateLimitError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    RAGError,
    RetrievalError,
    VectorStoreError,
    VectorStoreUnavailableError,
)

pytestmark = pytest.mark.unit


class TestRAGErrorBasics:
    def test_message_is_accessible_and_preserved_by_str(self):
        error = RAGError("something went wrong")

        assert error.message == "something went wrong"
        assert "something went wrong" in str(error)

    def test_is_an_exception_so_it_can_be_raised_and_caught(self):
        with pytest.raises(RAGError):
            raise RAGError("boom")

    def test_args_are_populated_so_standard_tooling_works(self):
        # The prototype's exception skipped super().__init__(), leaving args
        # empty and breaking pickling, logging and repr.
        error = RAGError("boom")

        assert error.args == ("boom",)


class TestErrorCode:
    def test_code_is_derived_from_the_class_name(self):
        assert DocumentParsingError("x").code == "DOCUMENT_PARSING_ERROR"

    def test_code_derivation_handles_acronyms(self):
        assert LLMRateLimitError("x").code == "LLM_RATE_LIMIT_ERROR"
        assert LLMError("x").code == "LLM_ERROR"

    def test_code_appears_in_the_string_representation(self):
        assert str(RetrievalError("no results")) == "[RETRIEVAL_ERROR] no results"

    def test_code_is_available_on_the_class_without_instantiation(self):
        assert VectorStoreError.code == "VECTOR_STORE_ERROR"


class TestRetryability:
    def test_errors_are_not_retryable_by_default(self):
        assert DocumentParsingError("corrupt").retryable is False

    def test_rate_limit_errors_are_retryable(self):
        assert EmbeddingRateLimitError("429").retryable is True
        assert LLMRateLimitError("429").retryable is True

    def test_timeout_errors_are_retryable(self):
        assert LLMTimeoutError("timed out").retryable is True

    def test_transient_infrastructure_errors_are_retryable(self):
        assert VectorStoreUnavailableError("connection refused").retryable is True

    def test_retryability_can_be_overridden_per_instance(self):
        # A caller with better information than the class default may say so.
        assert DocumentParsingError("x", retryable=True).retryable is True


class TestContext:
    def test_context_defaults_to_empty(self):
        assert RAGError("x").context == {}

    def test_context_carries_structured_detail_for_logging(self):
        error = DocumentParsingError("bad page", context={"page": 12, "parser": "pdf"})

        assert error.context == {"page": 12, "parser": "pdf"}

    def test_context_is_copied_so_later_mutation_cannot_alter_the_error(self):
        supplied = {"page": 12}
        error = DocumentParsingError("bad page", context=supplied)

        supplied["page"] = 99

        assert error.context == {"page": 12}

    def test_context_appears_in_the_string_representation(self):
        error = DocumentParsingError("bad page", context={"page": 12})

        assert "page=12" in str(error)


class TestCausePreservation:
    def test_raising_from_a_vendor_exception_preserves_the_original(self):
        vendor_failure = ValueError("vendor exploded")

        with pytest.raises(EmbeddingError) as caught:
            try:
                raise vendor_failure
            except ValueError as exc:
                raise EmbeddingError("embedding failed") from exc

        assert caught.value.__cause__ is vendor_failure

    def test_can_be_constructed_outside_an_except_block(self):
        # The prototype's exception called sys.exc_info() in __init__ and blew up
        # with AttributeError when raised outside an active exception context.
        error = RAGError("no active exception here")

        assert error.message == "no active exception here"


class TestHierarchy:
    @pytest.mark.parametrize(
        ("subclass", "parent"),
        [
            (DocumentProcessingError, RAGError),
            (DocumentParsingError, DocumentProcessingError),
            (EmbeddingRateLimitError, EmbeddingError),
            (VectorStoreUnavailableError, VectorStoreError),
            (LLMRateLimitError, LLMError),
            (LLMTimeoutError, LLMError),
            (ConfigurationError, RAGError),
        ],
    )
    def test_subclass_relationships(self, subclass, parent):
        assert issubclass(subclass, parent)

    def test_catching_a_family_catches_its_members(self):
        with pytest.raises(DocumentProcessingError):
            raise DocumentParsingError("page 3 unreadable")

    def test_every_domain_error_is_a_ragerror(self):
        with pytest.raises(RAGError):
            raise LLMTimeoutError("60s elapsed")
