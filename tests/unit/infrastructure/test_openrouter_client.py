"""Behaviour of the OpenRouter answer-generation adapter.

OpenRouter speaks the OpenAI chat-completions protocol, so the adapter is thin:
two messages in, one completion out. What it owes the rest of the system is the
same as every other ``LLMClient`` -- a stable ``model_id`` on every answer, and
provider failures translated into the ``LLMError`` family so retry policy reads
``retryable`` as data rather than matching on message strings.

The provider is stubbed rather than called. These tests are about the
translation, and a test that needs a paid key is a test nobody runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIStatusError, APITimeoutError, RateLimitError

from rag.domain.errors import (
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from rag.domain.models import GenerationParams, LLMMessage, Prompt, ToolDefinition
from rag.infrastructure.llm import OpenRouterLLMClient

pytestmark = pytest.mark.unit

_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

_PROMPT = Prompt(
    system="Answer only from the context. Cite every claim.",
    user="[1] Kepler died in 1630.\n\nQuestion: when did Kepler die?",
    version="v1",
)
_PARAMS = GenerationParams(temperature=0.0, max_output_tokens=256, timeout_s=30.0)


def _completion(
    text: str | None = "Kepler died in 1630 [1].",
    finish_reason: str | None = "stop",
    usage: tuple[int, int] | None = (42, 9),
    tool_calls: list[Any] | None = None,
) -> SimpleNamespace:
    """A chat completion shaped like the one the SDK returns."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=(
            None
            if usage is None
            else SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1])
        ),
    )


def _fragment(text: str | None) -> SimpleNamespace:
    """A streaming chunk shaped like the one the SDK yields."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class _Completions:
    """Stands in for ``client.chat.completions``, recording what it was asked."""

    def __init__(
        self,
        result: Any = None,
        error: Exception | None = None,
        fragments: list[Any] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result
        self._error = error
        self._fragments = fragments

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if kwargs.get("stream"):
            return self._stream()
        return self._result

    def _stream(self) -> AsyncIterator[Any]:
        fragments = list(self._fragments or [])

        async def _iterator() -> AsyncIterator[Any]:
            for fragment in fragments:
                yield fragment

        return _iterator()


def _client(
    result: Any = None,
    error: Exception | None = None,
    fragments: list[Any] | None = None,
    model: str = "qwen/qwen3-8b",
) -> tuple[OpenRouterLLMClient, _Completions]:
    """An adapter wired to a stubbed provider."""
    completions = _Completions(result=result, error=error, fragments=fragments)
    provider = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenRouterLLMClient(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model=model,
        client=provider,
    )
    return client, completions


class TestConfiguration:
    def test_it_reports_the_model_it_was_configured_with(self):
        # Changing the model must be a configuration change, not a code change.
        client, _ = _client(model="meta-llama/llama-3.3-70b-instruct")

        assert client.model_id == "meta-llama/llama-3.3-70b-instruct"

    def test_it_talks_to_the_configured_base_url(self):
        # Without this the OpenAI SDK quietly calls api.openai.com, which fails
        # with an authentication error that names the wrong provider.
        client = OpenRouterLLMClient(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="qwen/qwen3-8b",
        )

        assert str(client._client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


class TestGeneration:
    async def test_it_returns_the_generated_text(self):
        client, _ = _client(result=_completion(text="Kepler died in 1630 [1]."))

        response = await client.generate(_PROMPT, _PARAMS)

        assert response.text == "Kepler died in 1630 [1]."

    async def test_the_answer_is_attributed_to_the_model_that_produced_it(self):
        client, _ = _client(result=_completion())

        response = await client.generate(_PROMPT, _PARAMS)

        assert response.model_id == "qwen/qwen3-8b"

    async def test_the_system_instruction_and_the_question_stay_separate_messages(self):
        # Collapsing them into one user turn loses the grounding rules' standing.
        client, completions = _client(result=_completion())

        await client.generate(_PROMPT, _PARAMS)

        assert completions.calls[0]["messages"] == [
            {"role": "system", "content": _PROMPT.system},
            {"role": "user", "content": _PROMPT.user},
        ]

    async def test_a_prompt_without_a_system_instruction_sends_only_the_user_turn(self):
        client, completions = _client(result=_completion())

        await client.generate(Prompt(system="", user="Hello?", version="v1"), _PARAMS)

        assert completions.calls[0]["messages"] == [{"role": "user", "content": "Hello?"}]

    async def test_generation_parameters_reach_the_provider(self):
        client, completions = _client(result=_completion())

        await client.generate(
            _PROMPT, GenerationParams(temperature=0.4, max_output_tokens=128, timeout_s=12.0)
        )

        call = completions.calls[0]
        assert call["model"] == "qwen/qwen3-8b"
        assert call["temperature"] == pytest.approx(0.4)
        assert call["max_tokens"] == 128
        assert call["timeout"] == pytest.approx(12.0)

    async def test_token_usage_is_recorded_when_the_provider_reports_it(self):
        client, _ = _client(result=_completion(usage=(42, 9)))

        usage = (await client.generate(_PROMPT, _PARAMS)).usage

        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (42, 9)

    async def test_usage_is_absent_when_the_provider_omits_it(self):
        client, _ = _client(result=_completion(usage=None))

        assert (await client.generate(_PROMPT, _PARAMS)).usage is None

    async def test_why_generation_stopped_is_recorded(self):
        # `length` means the answer was truncated, which callers may surface.
        client, _ = _client(result=_completion(finish_reason="length"))

        assert (await client.generate(_PROMPT, _PARAMS)).finish_reason == "length"


class TestEmptyCompletions:
    async def test_an_empty_completion_fails_rather_than_answering_with_nothing(self):
        client, _ = _client(result=_completion(text=""))

        with pytest.raises(LLMError):
            await client.generate(_PROMPT, _PARAMS)

    async def test_a_filtered_completion_is_a_content_filter_error(self):
        client, _ = _client(result=_completion(text="", finish_reason="content_filter"))

        with pytest.raises(LLMContentFilterError):
            await client.generate(_PROMPT, _PARAMS)

    async def test_a_response_with_no_choices_fails(self):
        client, _ = _client(result=SimpleNamespace(choices=[], usage=None))

        with pytest.raises(LLMError):
            await client.generate(_PROMPT, _PARAMS)


class TestFailureTranslation:
    async def test_a_rate_limit_becomes_the_domain_rate_limit_error(self):
        error = RateLimitError(
            "rate limited", response=httpx.Response(429, request=_REQUEST), body=None
        )
        client, _ = _client(error=error)

        with pytest.raises(LLMRateLimitError):
            await client.generate(_PROMPT, _PARAMS)

    async def test_a_rate_limit_is_retryable(self):
        # Retry policy reads this as data rather than matching on the message.
        error = RateLimitError(
            "rate limited", response=httpx.Response(429, request=_REQUEST), body=None
        )
        client, _ = _client(error=error)

        with pytest.raises(LLMRateLimitError) as caught:
            await client.generate(_PROMPT, _PARAMS)
        assert caught.value.retryable is True

    async def test_a_timeout_becomes_the_domain_timeout_error(self):
        client, _ = _client(error=APITimeoutError(_REQUEST))

        with pytest.raises(LLMTimeoutError):
            await client.generate(_PROMPT, _PARAMS)

    async def test_any_other_provider_failure_becomes_an_llm_error(self):
        error = APIStatusError(
            "upstream is down", response=httpx.Response(502, request=_REQUEST), body=None
        )
        client, _ = _client(error=error)

        with pytest.raises(LLMError):
            await client.generate(_PROMPT, _PARAMS)

    async def test_a_failure_that_is_not_an_sdk_error_is_still_translated(self):
        client, _ = _client(error=RuntimeError("socket closed"))

        with pytest.raises(LLMError):
            await client.generate(_PROMPT, _PARAMS)

    async def test_the_error_names_the_model_that_failed(self):
        # An operator running several models needs to know which one broke.
        client, _ = _client(error=RuntimeError("socket closed"), model="qwen/qwen3-8b")

        with pytest.raises(LLMError) as caught:
            await client.generate(_PROMPT, _PARAMS)
        assert caught.value.context["model"] == "qwen/qwen3-8b"


class TestStreaming:
    async def test_it_yields_fragments_in_order(self):
        client, _ = _client(
            fragments=[_fragment("Kepler "), _fragment("died "), _fragment("1630.")]
        )

        received = [fragment async for fragment in client.stream(_PROMPT, _PARAMS)]

        assert received == ["Kepler ", "died ", "1630."]

    async def test_empty_fragments_are_skipped(self):
        # The first chunk of an OpenAI-protocol stream carries the role and no
        # content; forwarding it would emit a spurious empty token.
        client, _ = _client(fragments=[_fragment(None), _fragment("Kepler"), _fragment("")])

        received = [fragment async for fragment in client.stream(_PROMPT, _PARAMS)]

        assert received == ["Kepler"]

    async def test_it_asks_the_provider_to_stream(self):
        client, completions = _client(fragments=[_fragment("hi")])

        [fragment async for fragment in client.stream(_PROMPT, _PARAMS)]

        assert completions.calls[0]["stream"] is True

    async def test_a_streaming_failure_is_translated_too(self):
        error = RateLimitError(
            "rate limited", response=httpx.Response(429, request=_REQUEST), body=None
        )
        client, _ = _client(error=error)

        with pytest.raises(LLMRateLimitError):
            [fragment async for fragment in client.stream(_PROMPT, _PARAMS)]


class TestFunctionCalling:
    @staticmethod
    def _tool() -> ToolDefinition:
        return ToolDefinition(
            "search_documents",
            "Search documents",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    @staticmethod
    def _call(
        arguments: str = '{"query":"leave"}',
        *,
        call_id: str = "call-1",
        name: str = "search_documents",
    ) -> SimpleNamespace:
        return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))

    async def test_neutral_schema_and_tool_choice_reach_provider(self):
        client, completions = _client(result=_completion(text="Hello", tool_calls=[]))
        await client.respond((LLMMessage("user", "hello"),), (self._tool(),), _PARAMS)
        assert completions.calls[0]["tool_choice"] == "auto"
        assert completions.calls[0]["tools"][0]["function"]["name"] == "search_documents"

    async def test_single_and_multiple_calls_are_parsed(self):
        calls = [self._call(), self._call('{"query":"policy"}', call_id="call-2")]
        client, _ = _client(
            result=_completion(text=None, finish_reason="tool_calls", tool_calls=calls)
        )
        response = await client.respond((LLMMessage("user", "leave?"),), (self._tool(),), _PARAMS)
        assert [call.call_id for call in response.tool_calls] == ["call-1", "call-2"]

    @pytest.mark.parametrize("arguments", ["{bad", "[]", '"text"'])
    async def test_malformed_or_non_object_arguments_fail_safely(self, arguments):
        client, _ = _client(result=_completion(text=None, tool_calls=[self._call(arguments)]))
        with pytest.raises(LLMError):
            await client.respond((LLMMessage("user", "leave?"),), (self._tool(),), _PARAMS)

    async def test_tool_results_use_the_provider_tool_role(self):
        client, completions = _client(result=_completion(text="Done", tool_calls=[]))
        message = LLMMessage(
            "tool", '{"result_count":0}', tool_call_id="call-1", name="search_documents"
        )
        await client.respond((message,), (), _PARAMS)
        assert completions.calls[0]["messages"][0]["tool_call_id"] == "call-1"
