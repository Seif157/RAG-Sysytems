"""A scripted language model, so answering can be tested without a paid key."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from rag.domain.models import GenerationParams, LLMResponse, Prompt, TokenUsage
from rag.domain.ports import LLMClient

__all__ = ["FakeLLMClient"]


class FakeLLMClient(LLMClient):
    """Returns pre-scripted responses and records the prompts it was given.

    Recording the prompts is the point: it lets a test assert that retrieved
    context, conversation history and the grounding instruction actually reached
    the model, which is the part of the pipeline most likely to break silently.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        model_id: str = "fake-llm",
        error: Exception | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            responses: Texts to return, one per call. The last is repeated once
                exhausted, so a test that makes an unexpected extra call gets a
                sensible answer rather than an IndexError.
            model_id: Identifier reported on the response.
            error: When given, every call raises it instead of answering.
        """
        self._responses = list(responses) if responses else ["A grounded answer [1]."]
        self._model_id = model_id
        self._error = error
        self.prompts: list[Prompt] = []
        self.params: list[GenerationParams] = []

    @property
    def model_id(self) -> str:
        """Identifier reported on every response."""
        return self._model_id

    @property
    def last_prompt(self) -> Prompt:
        """The most recent prompt, for assertions."""
        return self.prompts[-1]

    def _next_text(self) -> str:
        """Return the next scripted response."""
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    async def generate(self, prompt: Prompt, params: GenerationParams) -> LLMResponse:
        """Return the next scripted response."""
        self.prompts.append(prompt)
        self.params.append(params)
        if self._error is not None:
            raise self._error
        text = self._next_text()
        return LLMResponse(
            text=text,
            model_id=self._model_id,
            usage=TokenUsage(
                prompt_tokens=len(prompt.user.split()), completion_tokens=len(text.split())
            ),
            finish_reason="stop",
        )

    def stream(self, prompt: Prompt, params: GenerationParams) -> AsyncIterator[str]:
        """Yield the next scripted response one word at a time."""
        self.prompts.append(prompt)
        self.params.append(params)
        error = self._error
        text = self._next_text()

        async def _iterator() -> AsyncIterator[str]:
            if error is not None:
                raise error
            for word in text.split(" "):
                yield word + " "

        return _iterator()
