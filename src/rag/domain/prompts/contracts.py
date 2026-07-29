"""What a prompt must contain, independent of how it is rendered.

This module holds a *business rule*, not a template. "Every answer must be
grounded in retrieved context and must cite its sources" is a property of the
product; whether that is expressed in Jinja, an f-string or a provider's
structured-prompt API is an infrastructure detail
(:mod:`rag.infrastructure.prompts`).

Keeping the requirement here means swapping the template engine, or the LLM
provider, cannot quietly drop the grounding instruction.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NOT_FOUND_PHRASE", "PromptSpec", "is_refusal"]

#: The exact wording the model is told to use when the retrieved passages do not
#: answer the question.
#:
#: A product decision, not a template detail, which is why it lives here rather
#: than beside the Jinja-free template that renders it. Prescribing a phrase
#: rather than saying "say so plainly" is what makes an ungrounded answer
#: *detectable*: left to itself a model refuses a dozen different ways --
#: "I'm unable to find", "the documents don't appear to", "based on the provided
#: context it is unclear" -- and nothing downstream can then tell a refusal from
#: a real answer. With a fixed phrase, the interface can, and so can a test.
NOT_FOUND_PHRASE = "The provided documents do not contain this information."


def is_refusal(answer_text: str) -> bool:
    """Whether an answer is the canonical "not in the documents" response.

    Deliberately a containment check rather than equality: a model will often
    append a sentence of its own, and treating that as a real answer would
    present a refusal as though it were grounded.

    Args:
        answer_text: The generated answer.

    Returns:
        Whether the answer declines for lack of supporting passages.
    """
    return NOT_FOUND_PHRASE.rstrip(".").lower() in answer_text.lower()


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """The contract a prompt builder must satisfy.

    Attributes:
        version: Template version, recorded on every answer so a quality change
            can be attributed to a prompt change.
        max_prompt_tokens: Hard ceiling. The builder raises
            :class:`~rag.domain.errors.PromptTooLargeError` rather than letting
            a provider truncate silently, which would drop context with no
            signal.
        require_grounding: The system instruction must tell the model to answer
            only from the supplied context.
        require_citations: The instruction must require a citation marker for
            every claim, and the context must be rendered with stable markers.
        allow_refusal: The model must be permitted -- and instructed -- to say
            the documents do not contain the answer. Without this, a model with
            weak context invents one.
        treat_context_as_data: The instruction must state that retrieved content
            is data, not instruction. Mitigation for prompt injection via
            document content; mitigation, not a guarantee.
    """

    version: str
    max_prompt_tokens: int
    require_grounding: bool = True
    require_citations: bool = True
    allow_refusal: bool = True
    treat_context_as_data: bool = True

    def __post_init__(self) -> None:
        """Validate the specification is usable."""
        if not self.version or not self.version.strip():
            raise ValueError("prompt version must be a non-empty string")
        if self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be >= 1")
