"""Prompt and generation types.

The prompt is modelled as a value object rather than a formatted string so that
the system instruction, the user content and the template version stay separable.
The version matters: it is recorded on every answer, which is what allows a
change in answer quality to be attributed to a prompt change rather than guessed
at (architecture spec section 12.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.domain.models.answer import TokenUsage

__all__ = ["GenerationParams", "LLMResponse", "Prompt"]


@dataclass(frozen=True, slots=True)
class Prompt:
    """An assembled prompt, ready for a provider.

    Attributes:
        system: The system instruction: grounding rules, citation format, and
            the statement that retrieved content is data rather than
            instruction.
        user: The user turn: context block, conversation history and question.
        version: Template version, recorded on the resulting answer.
        estimated_tokens: Token estimate used for budgeting, when known.
    """

    system: str
    user: str
    version: str
    estimated_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate the prompt is complete and attributable."""
        if not self.user or not self.user.strip():
            raise ValueError("prompt user part must be a non-empty string")
        if not self.version or not self.version.strip():
            raise ValueError("prompt version must be a non-empty string")
        if self.estimated_tokens is not None and self.estimated_tokens < 0:
            raise ValueError("estimated_tokens must be >= 0")


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Provider-neutral generation settings.

    Attributes:
        temperature: Sampling temperature.
        max_output_tokens: Ceiling on generated tokens.
        timeout_s: How long to wait before raising ``LLMTimeoutError``.
    """

    temperature: float
    max_output_tokens: int
    timeout_s: float

    def __post_init__(self) -> None:
        """Validate the parameters are usable."""
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0.0")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be > 0.0")

    @property
    def is_deterministic(self) -> bool:
        """Whether the same prompt yields the same completion.

        The LLM response cache is only wired when this holds. Caching sampled
        output would serve one arbitrary sample forever (ADR-010).
        """
        return self.temperature == 0.0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A completion returned by a provider.

    Attributes:
        text: The generated text.
        model_id: Model that produced it, recorded for reproducibility.
        usage: Token accounting, when the provider reports it.
        finish_reason: Why generation stopped, when the provider reports it.
            A value of ``"max_tokens"`` means the answer was truncated, which
            callers may wish to surface.
    """

    text: str
    model_id: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the response is attributable to a model."""
        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
