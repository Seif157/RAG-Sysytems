"""Google Gemini generation."""

from __future__ import annotations

from collections.abc import AsyncIterator

from google import genai
from google.genai import types as gtypes

from rag.domain.errors import (
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from rag.domain.models import GenerationParams, LLMResponse, Prompt, TokenUsage
from rag.domain.ports import LLMClient

__all__ = ["GeminiLLMClient"]

_RATE_LIMIT_HINTS = ("rate limit", "resource_exhausted", "quota", "429")
_TIMEOUT_HINTS = ("deadline", "timeout", "504")
_SAFETY_HINTS = ("safety", "blocked", "prohibited_content")


class GeminiLLMClient(LLMClient):
    """Generates answers with a Gemini model.

    Translates the SDK's exceptions into the ``LLMError`` family. The SDK
    signals most failures through one generic error type, so classification
    inspects the message -- which is exactly the fragility this translation
    exists to contain. Doing it once here is better than every caller doing it.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        """Initialise the client.

        Args:
            api_key: Google API credential.
            model: Gemini model identifier.
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def model_id(self) -> str:
        """Model identifier, recorded on every answer."""
        return self._model

    def _config(self, prompt: Prompt, params: GenerationParams) -> gtypes.GenerateContentConfig:
        """Build the provider request configuration."""
        return gtypes.GenerateContentConfig(
            system_instruction=prompt.system,
            temperature=params.temperature,
            max_output_tokens=params.max_output_tokens,
        )

    async def generate(self, prompt: Prompt, params: GenerationParams) -> LLMResponse:
        """Generate a complete response."""
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt.user,
                config=self._config(prompt, params),
            )
        except Exception as exc:
            raise self._translate(exc) from exc

        text = response.text or ""
        if not text.strip():
            # An empty completion usually means a safety filter fired; the SDK
            # reports it as a normal response with no content.
            raise LLMContentFilterError(
                "the model returned no text, which usually means a safety filter blocked it",
                context={"model": self._model},
            )

        return LLMResponse(
            text=text,
            model_id=self._model,
            usage=self._usage(response),
            finish_reason=self._finish_reason(response),
        )

    def stream(self, prompt: Prompt, params: GenerationParams) -> AsyncIterator[str]:
        """Generate a response incrementally."""

        async def _iterator() -> AsyncIterator[str]:
            try:
                stream = await self._client.aio.models.generate_content_stream(
                    model=self._model,
                    contents=prompt.user,
                    config=self._config(prompt, params),
                )
                async for part in stream:
                    if part.text:
                        yield part.text
            except Exception as exc:
                raise self._translate(exc) from exc

        return _iterator()

    @staticmethod
    def _usage(response: object) -> TokenUsage | None:
        """Read token usage, when the provider reported it."""
        metadata = getattr(response, "usage_metadata", None)
        if metadata is None:
            return None
        return TokenUsage(
            prompt_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
        )

    @staticmethod
    def _finish_reason(response: object) -> str | None:
        """Read why generation stopped, when the provider reported it."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        return str(reason) if reason is not None else None

    def _translate(self, exc: Exception) -> LLMError:
        """Classify a provider exception into the domain's error family."""
        message = str(exc).lower()
        context = {"model": self._model}

        if any(hint in message for hint in _RATE_LIMIT_HINTS):
            return LLMRateLimitError(f"Gemini rate limit: {exc}", context=context)
        if any(hint in message for hint in _TIMEOUT_HINTS):
            return LLMTimeoutError(f"Gemini timed out: {exc}", context=context)
        if any(hint in message for hint in _SAFETY_HINTS):
            return LLMContentFilterError(f"Gemini blocked the request: {exc}", context=context)
        return LLMError(f"Gemini generation failed: {exc}", context=context)
