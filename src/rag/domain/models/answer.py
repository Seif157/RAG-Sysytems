"""Answers and their citations.

Correct attribution is this system's highest-ranked quality attribute: an answer
that cannot be traced back to a chunk is a defect, not a lesser answer.

One rule is encoded structurally rather than left to convention:
``relevance_score`` is optional. It is populated only when a reranker actually
scored the chunk. Fabricating a confidence number when none is available is
worse than omitting the field, because a plausible number invites trust it has
not earned (architecture spec section 12.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.domain.models.metadata import DocumentType

__all__ = ["Answer", "Citation", "TokenUsage"]


@dataclass(frozen=True, slots=True)
class Citation:
    """A pointer from an answer back to the place that supports it.

    A citation identifies a *location*, not a chunk. Retrieval routinely returns
    several passages from one page -- chunk overlap alone guarantees it -- and
    listing each separately would show a reader three identical entries that
    look like three independent confirmations. So supporting passages from the
    same location are merged, and :attr:`chunk_ids` records all of them: the
    reader sees one source, and the system can still say exactly which passages
    stood behind it.

    Attributes:
        chunk_ids: Every supporting chunk, strongest first.
        document_id: Document the passages came from.
        filename: Original filename, for display.
        document_type: Format of the source document.
        page_number: One-based page, where the format has pages.
        section: Nearest enclosing section title.
        heading: Nearest heading above the passage.
        relevance_score: Normalised reranker score in ``[0, 1]``, or ``None``
            when no reranker ran. This is a *relative* score, not a calibrated
            probability. On a merged citation it is the *strongest* supporting
            score: the citation is as good as its best evidence, not its average.
        snippet: Short excerpt for display, taken from the strongest supporting
            passage. Never the full chunk.
    """

    chunk_ids: tuple[str, ...]
    document_id: str
    filename: str
    document_type: DocumentType
    page_number: int | None = None
    section: str | None = None
    heading: str | None = None
    relevance_score: float | None = None
    snippet: str | None = None

    def __post_init__(self) -> None:
        """Validate identifiers, page numbering and score range."""
        if not self.chunk_ids:
            raise ValueError("a citation must name at least one supporting chunk")
        if not all(chunk_id.strip() for chunk_id in self.chunk_ids):
            raise ValueError("chunk_ids must all be non-empty strings")
        if not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number is one-based and must be >= 1")
        if self.relevance_score is not None and not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError("relevance_score must lie in [0.0, 1.0]")

    @property
    def chunk_id(self) -> str:
        """The strongest supporting chunk."""
        return self.chunk_ids[0]

    @property
    def supporting_chunk_count(self) -> int:
        """How many passages were merged into this citation."""
        return len(self.chunk_ids)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for one LLM call.

    Attributes:
        prompt_tokens: Tokens consumed by the prompt.
        completion_tokens: Tokens produced in the response.
    """

    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        """Reject negative counts."""
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens must be >= 0")

    @property
    def total_tokens(self) -> int:
        """Total tokens billed for this call."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer together with everything needed to audit it.

    Attributes:
        text: The answer. Never empty -- an empty response is a failure, whereas
            "that is not stated in the provided documents" is a valid answer.
        citations: Supporting passages. May legitimately be empty when the
            documents do not contain the answer.
        model_id: Model that generated the text.
        prompt_version: Version of the prompt template used, so a change in
            answer quality can be attributed to a prompt change.
        token_usage: Token accounting, when the provider reports it.
        rewritten_query: The query actually retrieved with, when rewriting
            changed it.
        retrieved_count: Candidates retrieved before reranking.
        reranked_count: Chunks that reached the LLM.
        degraded_stages: Optional stages that failed and were bypassed. A
            non-empty value means the answer was produced on a reduced-quality
            path, and the client should be able to see that.
        timings_ms: Per-stage durations, as ``(stage, milliseconds)`` pairs.
    """

    text: str
    citations: tuple[Citation, ...]
    model_id: str
    prompt_version: str
    token_usage: TokenUsage | None = None
    rewritten_query: str | None = None
    retrieved_count: int = 0
    reranked_count: int = 0
    degraded_stages: tuple[str, ...] = ()
    timings_ms: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate that the answer is a real response."""
        if not self.text or not self.text.strip():
            raise ValueError("answer text must be a non-empty string")
        if not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.retrieved_count < 0:
            raise ValueError("retrieved_count must be >= 0")
        if self.reranked_count < 0:
            raise ValueError("reranked_count must be >= 0")

    @property
    def has_citations(self) -> bool:
        """Whether the answer cites any supporting passage."""
        return len(self.citations) > 0

    @property
    def is_degraded(self) -> bool:
        """Whether any optional stage was bypassed while producing this answer."""
        return len(self.degraded_stages) > 0
