"""Translation from domain errors to HTTP responses (ADR-017).

One mapping table, in one place. Scattering status codes across routers is how a
``DocumentTooLargeError`` ends up returning 500 in one endpoint and 400 in
another, and how a client's retry logic starts guessing.

Two rules are enforced here rather than trusted to reviewers:

* **Server errors do not explain themselves.** A 5xx detail may name an internal
  host, a query or a stack frame; the client receives a generic message. What it
  does receive is the correlation id, which is the one thing a user can quote
  that makes a support request actionable.
* **Retryability is advertised.** It comes from the error type, not from a
  guess, so a client can back off correctly without parsing prose.
"""

from __future__ import annotations

from typing import Any

from rag.domain.errors import (
    CollectionMismatchError,
    ConfigurationError,
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentTooLargeError,
    EmbeddingRateLimitError,
    LLMContentFilterError,
    LLMRateLimitError,
    LLMTimeoutError,
    PromptTooLargeError,
    RAGError,
    UnsupportedFormatError,
    VectorStoreUnavailableError,
)

__all__ = ["INTERNAL_ERROR_DETAIL", "http_status_for", "problem_detail"]

#: What a client is told when something went wrong on our side.
INTERNAL_ERROR_DETAIL = (
    "The request could not be completed. Quote the correlation id when reporting this."
)

#: Most specific first: the first matching entry wins, so subclasses must precede
#: their parents.
_STATUS_BY_TYPE: tuple[tuple[type[RAGError], int], ...] = (
    (UnsupportedFormatError, 415),
    (DocumentTooLargeError, 413),
    (PromptTooLargeError, 422),
    (LLMContentFilterError, 422),
    (DocumentProcessingError, 422),
    (DocumentNotFoundError, 404),
    (EmbeddingRateLimitError, 429),
    (LLMRateLimitError, 429),
    (LLMTimeoutError, 504),
    (VectorStoreUnavailableError, 503),
    (CollectionMismatchError, 500),
    (ConfigurationError, 500),
)

_DEFAULT_STATUS = 500


def http_status_for(error: RAGError) -> int:
    """Map a domain error to an HTTP status code.

    Anything unrecognised is a 500. That is deliberate: an error nobody has
    classified is, by definition, one we did not anticipate.

    Args:
        error: The error to classify.

    Returns:
        The HTTP status code to return.
    """
    for error_type, status in _STATUS_BY_TYPE:
        if isinstance(error, error_type):
            return status
    return _DEFAULT_STATUS


def problem_detail(error: RAGError, correlation_id: str | None = None) -> dict[str, Any]:
    """Render a domain error as an RFC 7807 problem document.

    Args:
        error: The error to render.
        correlation_id: Correlation of the failing request, returned so a user
            can quote it.

    Returns:
        A JSON-serialisable problem document. For 5xx responses the detail is
        generic -- the real message stays in the logs, joined to the same
        correlation id.
    """
    status = http_status_for(error)
    return {
        "type": f"about:blank#{error.code.lower()}",
        "title": error.code.replace("_", " ").title(),
        "status": status,
        "code": error.code,
        "detail": INTERNAL_ERROR_DETAIL if status >= 500 else error.message,
        "retryable": error.retryable,
        "correlation_id": correlation_id,
    }
