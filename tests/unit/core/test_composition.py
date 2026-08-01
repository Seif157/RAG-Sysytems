"""Behaviour of the composition root: error mapping, translation, wiring, startup.

``core`` is the only package allowed to know both configuration and concrete
classes. These tests cover the three jobs that follow from that: turning a
domain error into an HTTP response without leaking internals, translating
configuration into domain types, and assembling the object graph.
"""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.core.bootstrap import load_settings, run_startup_checks
from rag.core.container import Container
from rag.core.errors import http_status_for, problem_detail
from rag.core.mapping import (
    collection_spec_from,
    distance_metric_from,
    generation_params_from,
    prompt_spec_from,
)
from rag.core.providers import ProviderRegistry
from rag.domain.errors import (
    ConfigurationError,
    DocumentNotFoundError,
    DocumentParsingError,
    DocumentTooLargeError,
    EmbeddingRateLimitError,
    LLMRateLimitError,
    LLMTimeoutError,
    PromptTooLargeError,
    RAGError,
    RetrievalError,
    UnsupportedFormatError,
    VectorStoreUnavailableError,
)
from rag.domain.models import DistanceMetric
from rag.infrastructure.llm import GeminiLLMClient, OpenRouterLLMClient
from tests.fakes import FakeLLMClient

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)


def _settings(**env: str) -> Settings:
    return Settings.from_environment(env, use_env_file=False)


class TestErrorToHttpStatus:
    @pytest.mark.parametrize(
        ("error", "status"),
        [
            (UnsupportedFormatError("nope"), 415),
            (DocumentTooLargeError("too big"), 413),
            (DocumentParsingError("bad page"), 422),
            (PromptTooLargeError("too many tokens"), 422),
            (DocumentNotFoundError("gone"), 404),
            (EmbeddingRateLimitError("429"), 429),
            (LLMRateLimitError("429"), 429),
            (LLMTimeoutError("slow"), 504),
            (VectorStoreUnavailableError("refused"), 503),
            (RetrievalError("boom"), 500),
            (ConfigurationError("bad config"), 500),
            (RAGError("unknown"), 500),
        ],
    )
    def test_each_error_family_maps_to_a_sensible_status(self, error, status):
        assert http_status_for(error) == status


class TestProblemDetail:
    def test_client_errors_explain_themselves(self):
        detail = problem_detail(DocumentTooLargeError("file exceeds 50 MiB"), "corr-1")

        assert detail["status"] == 413
        assert detail["code"] == "DOCUMENT_TOO_LARGE_ERROR"
        assert detail["detail"] == "file exceeds 50 MiB"

    def test_server_errors_do_not_leak_internal_detail(self):
        detail = problem_detail(RetrievalError("qdrant said connection refused at 10.0.0.4"), "c-1")

        assert detail["status"] == 500
        assert "10.0.0.4" not in detail["detail"]

    def test_every_response_carries_the_correlation_id(self):
        # It is the one thing a user can quote that makes a report actionable.
        detail = problem_detail(RetrievalError("boom"), "corr-1")

        assert detail["correlation_id"] == "corr-1"

    def test_the_stable_code_is_always_present(self):
        detail = problem_detail(RetrievalError("boom"), "corr-1")

        assert detail["code"] == "RETRIEVAL_ERROR"

    def test_retryability_is_advertised_to_the_client(self):
        assert problem_detail(LLMRateLimitError("429"), "c")["retryable"] is True
        assert problem_detail(DocumentParsingError("bad"), "c")["retryable"] is False


class TestSettingsToDomainTranslation:
    def test_builds_the_collection_specification_the_store_must_satisfy(self):
        spec = collection_spec_from(_settings())

        assert spec.name == "document_chunks"
        assert spec.dense_dimension == 384
        assert spec.embedding_model_id == "BAAI/bge-small-en-v1.5"
        assert spec.distance is DistanceMetric.COSINE
        assert spec.supports_sparse is True

    def test_a_dense_only_deployment_asks_for_no_sparse_vectors(self):
        # Turning hybrid off changes the shape the collection must have, not
        # just which code path runs.
        assert collection_spec_from(_settings(ENABLE_HYBRID="false")).supports_sparse is False

    def test_translates_every_distance_metric(self):
        from rag.config import DistanceMetricName

        for name in DistanceMetricName:
            assert distance_metric_from(name).value == name.value

    def test_builds_generation_parameters_for_the_llm_port(self):
        params = generation_params_from(_settings(TEMPERATURE="0.0", MAX_OUTPUT_TOKENS="256"))

        assert params.max_output_tokens == 256
        assert params.is_deterministic is True

    def test_builds_the_prompt_contract(self):
        spec = prompt_spec_from(_settings(PROMPT_VERSION="v2"))

        assert spec.version == "v2"
        assert spec.max_prompt_tokens == 8192
        assert spec.require_citations is True
        assert spec.require_grounding is True


class TestProviderRegistry:
    def test_resolves_a_registered_factory(self):
        registry: ProviderRegistry[str, str] = ProviderRegistry("embedder")
        registry.register("openai", lambda: "openai-embedder")

        assert registry.create("openai") == "openai-embedder"

    def test_passes_arguments_through_to_the_factory(self):
        registry: ProviderRegistry[str, str] = ProviderRegistry("llm")
        registry.register("gemini", lambda model: f"gemini:{model}")

        assert registry.create("gemini", "flash") == "gemini:flash"

    def test_an_unknown_provider_is_a_configuration_error(self):
        registry: ProviderRegistry[str, str] = ProviderRegistry("llm")
        registry.register("gemini", lambda: "g")

        with pytest.raises(ConfigurationError, match="llm"):
            registry.create("mystery")

    def test_the_error_lists_what_is_available(self):
        # Otherwise the operator's next step is reading source code.
        registry: ProviderRegistry[str, str] = ProviderRegistry("llm")
        registry.register("gemini", lambda: "g")
        registry.register("openai", lambda: "o")

        with pytest.raises(ConfigurationError, match="gemini"):
            registry.create("mystery")

    def test_registering_the_same_key_twice_is_refused(self):
        # Silent replacement makes wiring depend on import order.
        registry: ProviderRegistry[str, str] = ProviderRegistry("llm")
        registry.register("gemini", lambda: "g")

        with pytest.raises(ConfigurationError, match="already registered"):
            registry.register("gemini", lambda: "other")

    def test_reports_what_it_knows(self):
        registry: ProviderRegistry[str, str] = ProviderRegistry("llm")
        registry.register("gemini", lambda: "g")

        assert registry.registered() == ("gemini",)


class TestContainer:
    def test_exposes_the_settings_it_was_built_from(self):
        settings = _settings()

        assert Container(settings).settings is settings

    def test_derives_the_collection_shape_from_configuration(self):
        assert Container(_settings()).collection_spec().dense_dimension == 384

    def test_derives_generation_parameters_from_configuration(self):
        container = Container(_settings(MAX_OUTPUT_TOKENS="256"))

        assert container.generation_params().max_output_tokens == 256

    def test_derives_the_prompt_contract_from_configuration(self):
        assert Container(_settings(PROMPT_VERSION="v7")).prompt_spec().version == "v7"

    def test_two_containers_are_independent(self):
        # Keeps tests from leaking state into one another.
        assert Container(_settings()).settings is not Container(_settings()).settings


class TestLLMWiring:
    """Which adapter a configured provider resolves to (ADR-009, ADR-011)."""

    def test_openrouter_is_wired_when_selected(self):
        container = Container(
            _settings(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="sk-or-v1-key")
        )

        assert isinstance(container.llm, OpenRouterLLMClient)

    def test_the_openrouter_model_comes_from_configuration(self):
        # Swapping the model must never require touching the source.
        container = Container(
            _settings(
                LLM_PROVIDER="openrouter",
                OPENROUTER_API_KEY="sk-or-v1-key",
                LLM_MODEL="qwen/qwen3-8b",
            )
        )

        assert container.llm.model_id == "qwen/qwen3-8b"

    def test_openrouter_without_its_credential_names_the_variable(self):
        container = Container(_settings(LLM_PROVIDER="openrouter"))

        with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
            _ = container.llm

    def test_gemini_remains_available_as_an_alternative(self):
        container = Container(_settings(LLM_PROVIDER="gemini", GOOGLE_API_KEY="g-key"))

        assert isinstance(container.llm, GeminiLLMClient)

    def test_gemini_without_its_credential_names_the_variable(self):
        container = Container(_settings(LLM_PROVIDER="gemini"))

        with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
            _ = container.llm

    def test_an_injected_client_overrides_the_configured_provider(self):
        # The seam the test suite hangs from: no key needed to answer a question.
        fake = FakeLLMClient()

        assert Container(_settings(LLM_PROVIDER="openrouter"), llm=fake).llm is fake


class TestLoadSettings:
    def test_valid_configuration_loads(self):
        assert load_settings({"TOP_K": "10"}, use_env_file=False).retrieval.top_k == 10

    def test_invalid_configuration_becomes_a_domain_error(self):
        # config raises pydantic's ValidationError because it imports no other
        # layer; the composition root is where it becomes a domain concept.
        with pytest.raises(ConfigurationError):
            load_settings({"TOP_K": "5", "RERANK_TOP_K": "10"}, use_env_file=False)

    def test_the_domain_error_still_explains_what_is_wrong(self):
        with pytest.raises(ConfigurationError, match="RERANK_TOP_K"):
            load_settings({"TOP_K": "5", "RERANK_TOP_K": "10"}, use_env_file=False)

    def test_the_original_validation_error_is_preserved_as_the_cause(self):
        with pytest.raises(ConfigurationError) as caught:
            load_settings({"TOP_K": "5", "RERANK_TOP_K": "10"}, use_env_file=False)

        assert caught.value.__cause__ is not None


class TestStartupChecks:
    def test_a_fully_configured_local_deployment_passes(self):
        settings = _settings(
            LLM_PROVIDER="gemini",
            GOOGLE_API_KEY="g-key",
            EMBEDDING_PROVIDER="openai",
            OPENAI_API_KEY="o-key",
        )

        assert "credentials" in run_startup_checks(settings)

    def test_a_missing_llm_credential_fails_startup(self):
        # Better here than on a user's first question.
        settings = _settings(LLM_PROVIDER="gemini", EMBEDDING_PROVIDER="local")

        with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
            run_startup_checks(settings)

    def test_a_missing_embedding_credential_fails_startup(self):
        settings = _settings(
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-key",
            EMBEDDING_PROVIDER="openai",
        )

        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            run_startup_checks(settings)

    def test_an_openrouter_deployment_passes_with_its_credential(self):
        settings = _settings(
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="sk-or-v1-key",
            EMBEDDING_PROVIDER="local",
        )

        assert "credentials" in run_startup_checks(settings)

    def test_openrouter_requires_its_own_credential(self):
        # Caught at start-up rather than on a user's first question.
        settings = _settings(LLM_PROVIDER="openrouter", EMBEDDING_PROVIDER="local")

        with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
            run_startup_checks(settings)

    def test_production_refuses_human_readable_logs(self):
        # Console output cannot be queried by log aggregation, so in production
        # it is equivalent to having no observability at all.
        settings = _settings(
            APP_ENV="production",
            LOG_FORMAT="console",
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-key",
            EMBEDDING_PROVIDER="local",
        )

        with pytest.raises(ConfigurationError, match="LOG_FORMAT"):
            run_startup_checks(settings)

    def test_production_with_json_logs_passes(self):
        settings = _settings(
            APP_ENV="production",
            LOG_FORMAT="json",
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-key",
            EMBEDDING_PROVIDER="local",
        )

        assert "logging" in run_startup_checks(settings)

    def test_the_checks_that_ran_are_reported(self):
        settings = _settings(
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-key",
            EMBEDDING_PROVIDER="local",
        )

        assert set(run_startup_checks(settings)) == {"credentials", "logging"}
