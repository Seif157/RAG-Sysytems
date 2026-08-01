"""The domain exception hierarchy.

Every failure the system can express is one of these types. Adapters translate
vendor exceptions at their own boundary and re-raise as a domain error with
``raise ... from exc``, so the original traceback survives (ADR-017).

Two properties carry real weight:

``code``
    A stable, machine-readable identifier derived from the class name. It is what
    the HTTP layer returns to clients and what log aggregation groups on. Because
    it is derived, it cannot drift from the class it names.

``retryable``
    Whether retrying the same operation could plausibly succeed. Retry policies
    read this as *data* rather than matching on message substrings, which is what
    lets the task queue and the HTTP client share one decision rule.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, ClassVar

__all__ = [
    "ChunkingError",
    "CollectionMismatchError",
    "CollectionNotFoundError",
    "ConfigurationError",
    "CorruptDocumentError",
    "DocumentNotFoundError",
    "DocumentParsingError",
    "DocumentProcessingError",
    "DocumentTooLargeError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingRateLimitError",
    "EncryptedDocumentError",
    "LLMContentFilterError",
    "LLMError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "PromptError",
    "PromptTooLargeError",
    "RAGError",
    "RerankingError",
    "RetrievalError",
    "ToolCallLimitError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolValidationError",
    "UnsupportedFormatError",
    "VectorStoreError",
    "VectorStoreUnavailableError",
]

# Splits CamelCase into words while keeping acronyms intact:
# "LLMRateLimitError" -> "LLM", "Rate", "Limit", "Error"
_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _derive_code(class_name: str) -> str:
    """Convert a CamelCase class name to a SCREAMING_SNAKE_CASE error code."""
    return _WORD_BOUNDARY.sub("_", class_name).upper()


class RAGError(Exception):
    """Base class for every error the platform raises deliberately.

    Attributes:
        code: Stable machine-readable identifier, derived from the class name.
        default_retryable: Class-level default for :attr:`retryable`.
        message: Human-readable description of the failure.
        context: Structured detail for logging. Never contains document content,
            prompts or credentials.
        retryable: Whether retrying the operation could plausibly succeed.
    """

    code: ClassVar[str] = "RAG_ERROR"
    default_retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the failure.
            context: Structured detail attached to log events.
            retryable: Overrides :attr:`default_retryable` when the caller knows
                better than the class default.
        """
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context) if context else {}
        self.retryable: bool = self.default_retryable if retryable is None else retryable

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Derive :attr:`code` for every subclass that does not declare one."""
        super().__init_subclass__(**kwargs)
        if "code" not in cls.__dict__:
            cls.code = _derive_code(cls.__name__)

    def __str__(self) -> str:
        """Render as ``[CODE] message`` with any context appended."""
        rendered = f"[{self.code}] {self.message}"
        if self.context:
            detail = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
            rendered = f"{rendered} ({detail})"
        return rendered

    def __repr__(self) -> str:
        """Render an unambiguous representation for debugging."""
        return (
            f"{type(self).__name__}(message={self.message!r}, "
            f"context={self.context!r}, retryable={self.retryable!r})"
        )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(RAGError):
    """Configuration is invalid, incomplete or internally contradictory.

    Raised during startup validation so that a misconfigured deployment fails
    before serving traffic rather than on a user's first question.
    """


# --------------------------------------------------------------------------- #
# Document processing
# --------------------------------------------------------------------------- #
class DocumentProcessingError(RAGError):
    """A document could not be turned into indexable chunks."""


class UnsupportedFormatError(DocumentProcessingError):
    """The uploaded file is not one of the supported document types."""


class DocumentTooLargeError(DocumentProcessingError):
    """The uploaded file exceeds the configured size limit."""


class DocumentParsingError(DocumentProcessingError):
    """A parser failed to extract text or structure from the document."""


class CorruptDocumentError(DocumentParsingError):
    """The file is malformed and cannot be read by its parser."""


class EncryptedDocumentError(DocumentParsingError):
    """The document is password-protected and cannot be read."""


class ChunkingError(DocumentProcessingError):
    """A chunking strategy could not produce valid chunks."""


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class EmbeddingError(RAGError):
    """An embedding could not be produced."""


class EmbeddingRateLimitError(EmbeddingError):
    """The embedding provider rejected the request for rate-limit reasons."""

    default_retryable = True


class EmbeddingDimensionMismatchError(EmbeddingError):
    """A vector's dimension does not match the configured embedding model.

    Never recoverable at runtime: vectors from different models are not
    comparable, and proceeding would return confidently wrong answers (ADR-015).
    """


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
class VectorStoreError(RAGError):
    """A vector store operation failed."""


class CollectionNotFoundError(VectorStoreError):
    """The configured collection does not exist."""


class CollectionMismatchError(VectorStoreError):
    """The collection's embedding model or dimension disagrees with configuration."""


class VectorStoreUnavailableError(VectorStoreError):
    """The vector store could not be reached."""

    default_retryable = True


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class RetrievalError(RAGError):
    """Retrieval failed. Mandatory stage: this propagates rather than degrading."""


class RerankingError(RAGError):
    """Reranking failed. Optional stage: callers fall back to fusion order."""


# --------------------------------------------------------------------------- #
# Prompting and generation
# --------------------------------------------------------------------------- #
class PromptError(RAGError):
    """A prompt could not be constructed."""


class PromptTooLargeError(PromptError):
    """The assembled prompt exceeds the configured token ceiling.

    Raised by the prompt builder rather than allowing the provider to truncate
    silently, which would drop context without anyone noticing.
    """


class LLMError(RAGError):
    """The language model could not produce a response."""


class LLMRateLimitError(LLMError):
    """The LLM provider rejected the request for rate-limit reasons."""

    default_retryable = True


class LLMTimeoutError(LLMError):
    """The LLM did not respond within the configured timeout."""

    default_retryable = True


class LLMContentFilterError(LLMError):
    """The provider's safety filter blocked the request or the response."""


class ToolExecutionError(RAGError):
    """A controlled application tool could not be executed."""


class ToolValidationError(ToolExecutionError):
    """A model supplied invalid or unauthorized tool arguments."""


class ToolTimeoutError(ToolExecutionError):
    """A tool exceeded its configured execution deadline."""

    default_retryable = True


class ToolCallLimitError(ToolExecutionError):
    """The bounded function-calling loop exhausted its call allowance."""


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
class DocumentNotFoundError(RAGError):
    """The requested document is not in the index."""
