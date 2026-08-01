"""Port: language model generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from rag.domain.errors import LLMError
from rag.domain.models import GenerationParams, LLMMessage, LLMResponse, Prompt, ToolDefinition

__all__ = ["LLMClient"]


class LLMClient(ABC):
    """Generates an answer from an assembled prompt.

    The whole point of this interface is that Gemini, OpenAI, Anthropic and a
    local model are interchangeable without any change above
    :mod:`rag.infrastructure`. Swapping providers requires no re-indexing --
    unlike the embedder, the LLM is not part of the index.

    Implementations translate their SDK's exceptions into the ``LLMError``
    family. That translation is what lets retry policy read
    :attr:`~rag.domain.errors.RAGError.retryable` as data rather than matching
    on provider-specific message strings, which is how retry logic rots.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable model identifier, recorded on every answer."""

    @abstractmethod
    async def generate(self, prompt: Prompt, params: GenerationParams) -> LLMResponse:
        """Generate a complete response.

        Args:
            prompt: The assembled prompt.
            params: Provider-neutral generation settings.

        Returns:
            The generated text with usage and finish reason where the provider
            reports them.

        Raises:
            LLMRateLimitError: If the provider throttled the request.
            LLMTimeoutError: If generation exceeded ``params.timeout_s``.
            LLMContentFilterError: If a safety filter blocked the exchange.
            LLMError: On any other generation failure.
        """

    async def respond(
        self,
        messages: tuple[LLMMessage, ...],
        tools: tuple[ToolDefinition, ...],
        params: GenerationParams,
    ) -> LLMResponse:
        """Respond to a chat exchange that may contain controlled tools."""
        raise LLMError(
            "the configured LLM adapter does not support function calling",
            context={"model": self.model_id},
        )

    @abstractmethod
    def stream(self, prompt: Prompt, params: GenerationParams) -> AsyncIterator[str]:
        """Generate a response incrementally.

        Declared as a normal method returning an async iterator, rather than as
        an async generator, so that implementations are free to return any
        iterator -- including one that wraps a non-streaming provider by
        yielding a single chunk.

        Args:
            prompt: The assembled prompt.
            params: Provider-neutral generation settings.

        Returns:
            An async iterator of text fragments in order.

        Raises:
            LLMRateLimitError: If the provider throttled the request.
            LLMTimeoutError: If generation exceeded ``params.timeout_s``.
            LLMContentFilterError: If a safety filter blocked the exchange.
            LLMError: On any other generation failure.
        """
