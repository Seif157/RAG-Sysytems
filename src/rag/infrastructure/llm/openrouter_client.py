"""OpenRouter generation, over the OpenAI-compatible protocol.

OpenRouter fronts many vendors behind one credential and one chat-completions
endpoint, which is why it is the default: choosing an answer model becomes a
value in the environment rather than an adapter in the source. Nothing here
names a model.

The OpenAI SDK is the client, pointed at OpenRouter's base URL. That is not a
shortcut -- the protocol *is* OpenAI's, and re-implementing it over ``httpx``
would buy nothing but retries, backoff and streaming parsing to maintain.

Like every adapter behind :class:`~rag.domain.ports.LLMClient`, this one
translates the SDK's exceptions into the ``LLMError`` family, so retry policy
reads :attr:`~rag.domain.errors.RAGError.retryable` as data rather than matching
on provider-specific message strings.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from rag.domain.errors import (
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from rag.domain.models import GenerationParams, LLMResponse, Prompt, TokenUsage
from rag.domain.ports import LLMClient

__all__ = ["OpenRouterLLMClient"]

#: OpenRouter's OpenAI-compatible endpoint.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: What a provider reports when a safety filter stopped generation.
_CONTENT_FILTER_REASONS = frozenset({"content_filter", "safety"})


class OpenRouterLLMClient(LLMClient):
    """Generates answers with any model OpenRouter offers.

    The model is whatever ``LLM_MODEL`` says -- ``qwen/qwen3-8b``,
    ``anthropic/claude-sonnet-4.5``, ``google/gemini-2.0-flash``. Switching
    between them requires no code change, which is the entire reason this
    adapter exists in preference to one client per vendor.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "qwen/qwen3-8b",
        client: Any | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            api_key: OpenRouter credential.
            base_url: OpenRouter's OpenAI-compatible endpoint. Overridable so a
                proxy or gateway stays a deployment concern.
            model: OpenRouter model identifier, in ``vendor/model`` form.
            client: Overrides the SDK client. Tests pass a stub so the
                translation can be exercised without a key or a network.
        """
        self._client = (
            client if client is not None else AsyncOpenAI(api_key=api_key, base_url=base_url)
        )
        self._model = model

    @property
    def model_id(self) -> str:
        """Model identifier, recorded on every answer."""
        return self._model

    def _messages(self, prompt: Prompt) -> list[dict[str, str]]:
        """Express the prompt as chat messages.

        The system instruction stays a system message rather than being folded
        into the user turn: the grounding and citation rules are instructions to
        the model, and retrieved context is data. Collapsing the two is what
        makes a document able to talk the model out of its own rules.
        """
        messages: list[dict[str, str]] = []
        if prompt.system and prompt.system.strip():
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        return messages

    def _request(self, prompt: Prompt, params: GenerationParams) -> dict[str, Any]:
        """Build the provider request."""
        return {
            "model": self._model,
            "messages": self._messages(prompt),
            "temperature": params.temperature,
            "max_tokens": params.max_output_tokens,
            "timeout": params.timeout_s,
        }

    async def generate(self, prompt: Prompt, params: GenerationParams) -> LLMResponse:
        """Generate a complete response."""
        try:
            response = await self._client.chat.completions.create(**self._request(prompt, params))
        except Exception as exc:
            raise self._translate(exc) from exc

        choice = self._first_choice(response)
        finish_reason = getattr(choice, "finish_reason", None)
        text = getattr(getattr(choice, "message", None), "content", None) or ""

        if not text.strip():
            raise self._empty_completion(finish_reason)

        return LLMResponse(
            text=text,
            model_id=self._model,
            usage=self._usage(response),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )

    def stream(self, prompt: Prompt, params: GenerationParams) -> AsyncIterator[str]:
        """Generate a response incrementally."""

        async def _iterator() -> AsyncIterator[str]:
            try:
                stream = await self._client.chat.completions.create(
                    **self._request(prompt, params), stream=True
                )
                async for chunk in stream:
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    # The opening chunk of an OpenAI-protocol stream carries the
                    # role and no content; forwarding it emits an empty token.
                    fragment = getattr(getattr(choices[0], "delta", None), "content", None)
                    if fragment:
                        yield fragment
            except Exception as exc:
                raise self._translate(exc) from exc

        return _iterator()

    def _first_choice(self, response: object) -> object:
        """The single completion asked for, or a failure explaining its absence."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError(
                "OpenRouter returned no completion, which usually means the "
                "upstream provider rejected the request",
                context={"model": self._model},
            )
        return choices[0]

    def _empty_completion(self, finish_reason: object) -> LLMError:
        """Classify a completion that came back with no text.

        Answering with an empty string would present a failure as a
        (very short) grounded answer, so this always raises.
        """
        if str(finish_reason) in _CONTENT_FILTER_REASONS:
            return LLMContentFilterError(
                "the model returned no text because a safety filter blocked it",
                context={"model": self._model, "finish_reason": str(finish_reason)},
            )
        return LLMError(
            "the model returned an empty completion",
            context={"model": self._model, "finish_reason": str(finish_reason)},
            retryable=True,
        )

    @staticmethod
    def _usage(response: object) -> TokenUsage | None:
        """Read token usage, when the provider reported it.

        OpenRouter normalises usage across vendors, but not every upstream
        reports it, so its absence is a fact rather than an error.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def _translate(self, exc: Exception) -> LLMError:
        """Classify a provider exception into the domain's error family.

        The SDK's own types carry the classification, so unlike the Gemini
        adapter this needs no message-string matching -- except for the
        HTTP status on a generic status error, which is where OpenRouter
        reports an upstream vendor's rate limit.
        """
        context = {"model": self._model}

        if isinstance(exc, RateLimitError):
            return LLMRateLimitError(f"OpenRouter rate limit: {exc}", context=context)
        if isinstance(exc, APITimeoutError):
            return LLMTimeoutError(f"OpenRouter timed out: {exc}", context=context)
        if isinstance(exc, APIConnectionError):
            return LLMError(
                f"OpenRouter could not be reached: {exc}", context=context, retryable=True
            )

        status = getattr(exc, "status_code", None)
        if status == 429:
            return LLMRateLimitError(f"OpenRouter rate limit: {exc}", context=context)
        if status == 504:
            return LLMTimeoutError(f"OpenRouter timed out: {exc}", context=context)

        return LLMError(
            f"OpenRouter generation failed: {exc}",
            context=context,
            # 5xx is the upstream vendor having a bad minute; 4xx is our request.
            retryable=isinstance(status, int) and status >= 500,
        )
