"""Query value objects.

Two types rather than one, because they belong to different stages and
conflating them is how retrieval bugs hide:

``Query``
    What the user asked, plus any filter they chose.
``RetrievalRequest``
    What the retrievers actually execute: resolved text, resolved breadth,
    resolved filter. By this point every "use the configured default" decision
    has already been made, so a retriever never consults configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.domain.models.filters import FilterExpression

__all__ = ["Query", "RetrievalRequest"]


@dataclass(frozen=True, slots=True)
class Query:
    """A question as asked, before any processing.

    The ``top_k`` fields are *overrides*. Leaving them ``None`` means "use the
    configured value" -- the query deliberately does not know what that value
    is, because configuration is injected into services rather than read by
    models.

    Attributes:
        text: The user's question.
        conversation_id: Conversation this question belongs to, if any. Used to
            supply history to the prompt, never to retrieval.
        filters: Optional metadata filter chosen by the user.
        top_k: Override for retrieval breadth.
        rerank_top_k: Override for how many chunks reach the LLM.
    """

    text: str
    conversation_id: str | None = None
    filters: FilterExpression | None = None
    top_k: int | None = None
    rerank_top_k: int | None = None

    def __post_init__(self) -> None:
        """Validate the question and any breadth overrides."""
        if not self.text or not self.text.strip():
            raise ValueError("query text must be a non-empty string")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k override must be >= 1")
        if self.rerank_top_k is not None and self.rerank_top_k < 1:
            raise ValueError("rerank_top_k override must be >= 1")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """A fully resolved retrieval instruction.

    Every retriever -- dense, sparse, hybrid -- accepts this same request, which
    is what allows the hybrid retriever to be a composite of retrievers rather
    than a special case.

    Attributes:
        query_text: The text to search with.
        top_k: How many candidates to return.
        filters: Optional metadata filter. The current-version predicate is not
            here: the store adds it, so a caller cannot accidentally search
            superseded chunks mid-reindex.
    """

    query_text: str
    top_k: int
    filters: FilterExpression | None = None

    def __post_init__(self) -> None:
        """Validate the request is executable."""
        if not self.query_text or not self.query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
