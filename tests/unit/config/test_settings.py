"""Behaviour of the typed configuration system (ADR-012).

Two things are being pinned down:

1. **The environment variable names are exactly the documented ones.** A rename
   here silently falls back to a default, which is the kind of bug that only
   shows up as "why is retrieval returning 20 chunks in production".
2. **Invalid combinations fail at construction.** ``RERANK_TOP_K > TOP_K`` is a
   silent quality bug; a mismatched embedding dimension returns confidently
   wrong answers. Both must fail loudly, at start-up.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.config import (
    AppEnv,
    ChunkingStrategyName,
    DistanceMetricName,
    EmbeddingProvider,
    LLMProvider,
    LogFormat,
    Settings,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real .env or shell exports leaking into these tests."""
    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)


def _settings(**env: str) -> Settings:
    """Build settings from an explicit environment, ignoring any .env file."""
    return Settings.from_environment(env, use_env_file=False)


class TestDefaults:
    def test_defaults_match_the_documented_configuration(self):
        settings = _settings()

        assert settings.app.env is AppEnv.LOCAL
        assert settings.llm.provider is LLMProvider.OPENROUTER
        assert settings.embedding.provider is EmbeddingProvider.LOCAL
        assert settings.embedding.model == "BAAI/bge-small-en-v1.5"
        assert settings.embedding.dimension == 384
        assert settings.chunking.strategy is ChunkingStrategyName.RECURSIVE
        assert settings.retrieval.top_k == 20
        assert settings.retrieval.rerank_top_k == 5

    def test_temperature_defaults_to_zero_for_deterministic_answers(self):
        assert _settings().llm.temperature == pytest.approx(0.0)

    def test_hybrid_retrieval_and_reranking_are_on_by_default(self):
        settings = _settings()

        assert settings.retrieval.enable_hybrid is True
        assert settings.reranker.enable_rerank is True


class TestEnvironmentVariableNames:
    @pytest.mark.parametrize(
        ("variable", "value", "accessor"),
        [
            ("APP_ENV", "production", lambda s: s.app.env.value),
            ("LOG_LEVEL", "DEBUG", lambda s: s.app.log_level),
            ("LOG_FORMAT", "json", lambda s: s.app.log_format.value),
            ("LLM_PROVIDER", "gemini", lambda s: s.llm.provider.value),
            ("LLM_MODEL", "gpt-4o", lambda s: s.llm.model),
            ("TEMPERATURE", "0.7", lambda s: s.llm.temperature),
            ("MAX_OUTPUT_TOKENS", "512", lambda s: s.llm.max_output_tokens),
            ("LLM_TIMEOUT_S", "30.0", lambda s: s.llm.timeout_s),
            ("EMBEDDING_PROVIDER", "huggingface", lambda s: s.embedding.provider.value),
            ("EMBEDDING_BATCH_SIZE", "64", lambda s: s.embedding.batch_size),
            ("CHUNKING_STRATEGY", "recursive", lambda s: s.chunking.strategy.value),
            ("CHUNK_SIZE", "512", lambda s: s.chunking.chunk_size),
            ("CHUNK_OVERLAP", "64", lambda s: s.chunking.chunk_overlap),
            ("MAX_CHUNK_TOKENS", "2048", lambda s: s.chunking.max_chunk_tokens),
            ("QDRANT_URL", "http://q:6333", lambda s: s.vector_store.qdrant_url),
            ("COLLECTION_NAME", "custom", lambda s: s.vector_store.collection_name),
            ("DISTANCE_METRIC", "dot", lambda s: s.vector_store.distance.value),
            ("TOP_K", "40", lambda s: s.retrieval.top_k),
            ("RERANK_TOP_K", "8", lambda s: s.retrieval.rerank_top_k),
            ("RRF_K", "10", lambda s: s.retrieval.rrf_k),
            ("ENABLE_RERANK", "false", lambda s: s.reranker.enable_rerank),
            ("RERANKER_MODEL", "BAAI/other", lambda s: s.reranker.model),
            ("MEMORY_WINDOW_TURNS", "12", lambda s: s.memory.window_turns),
            ("MAX_PROMPT_TOKENS", "16384", lambda s: s.context.max_prompt_tokens),
            ("CONTEXT_TOKEN_BUDGET", "1024", lambda s: s.context.token_budget),
            ("PROMPT_VERSION", "v9", lambda s: s.context.prompt_version),
            ("MAX_UPLOAD_BYTES", "1024", lambda s: s.ingestion.max_upload_bytes),
            ("UPLOAD_DIR", "./data", lambda s: s.ingestion.upload_dir),
        ],
    )
    def test_variable_is_read_under_its_specified_name(self, variable, value, accessor):
        settings = _settings(**{variable: value})

        assert str(accessor(settings)).lower() == value.lower()

    def test_enable_hybrid_is_read_under_its_specified_name(self):
        assert _settings(ENABLE_HYBRID="false").retrieval.enable_hybrid is False

    def test_allowed_mime_types_parses_a_comma_separated_list(self):
        settings = _settings(ALLOWED_MIME_TYPES="text/plain,application/pdf")

        assert settings.ingestion.allowed_mime_types == ("text/plain", "application/pdf")

    def test_no_variable_exists_for_a_capability_that_was_removed(self):
        # Guards against a setting outliving the feature it configured, which is
        # how a config file starts lying to whoever reads it.
        names = set(Settings.environment_variable_names())

        assert not names & {
            "DATABASE_URL",
            "REDIS_URL",
            "BLOB_BACKEND",
            "S3_BUCKET",
            "VECTOR_DATABASE",
            "ENABLE_CACHE",
            "CACHE_BACKEND",
            "ENABLE_QUERY_REWRITE",
            "MEMORY_BACKEND",
            "INGEST_CONCURRENCY",
        }


class TestOpenRouterConfiguration:
    """Selecting OpenRouter must be configuration only -- no code change."""

    def test_the_provider_is_selectable_by_name(self):
        assert _settings(LLM_PROVIDER="openrouter").llm.provider is LLMProvider.OPENROUTER

    def test_the_model_is_configurable(self):
        # The point of routing through OpenRouter: any model it offers is a
        # one-line change, and none of them are named in the source.
        assert _settings(LLM_MODEL="qwen/qwen3-8b").llm.model == "qwen/qwen3-8b"

    def test_its_credential_is_read_under_its_specified_name(self):
        settings = _settings(OPENROUTER_API_KEY="sk-or-v1-key")

        assert settings.credentials.openrouter_api_key is not None
        assert settings.credentials.openrouter_api_key.get_secret_value() == "sk-or-v1-key"

    def test_its_credential_is_held_as_a_secret(self):
        assert "sk-or-v1-key" not in repr(_settings(OPENROUTER_API_KEY="sk-or-v1-key"))

    def test_a_blank_credential_is_absent_rather_than_an_empty_secret(self):
        assert _settings(OPENROUTER_API_KEY="  ").credentials.openrouter_api_key is None

    def test_the_base_url_defaults_to_openrouter(self):
        assert _settings().llm.openrouter_base_url == "https://openrouter.ai/api/v1"

    def test_the_base_url_is_configurable(self):
        # A proxy or a gateway in front of OpenRouter is a deployment concern.
        settings = _settings(OPENROUTER_BASE_URL="http://gateway.internal/v1")

        assert settings.llm.openrouter_base_url == "http://gateway.internal/v1"

    def test_gemini_remains_selectable(self):
        assert _settings(LLM_PROVIDER="gemini").llm.provider is LLMProvider.GEMINI


class TestSecrets:
    def test_api_keys_are_not_exposed_by_repr(self):
        settings = _settings(OPENAI_API_KEY="sk-super-secret-value")

        assert "sk-super-secret-value" not in repr(settings)
        assert "sk-super-secret-value" not in str(settings)

    def test_the_secret_can_still_be_read_deliberately(self):
        settings = _settings(OPENAI_API_KEY="sk-super-secret-value")

        assert settings.credentials.openai_api_key is not None
        assert settings.credentials.openai_api_key.get_secret_value() == "sk-super-secret-value"

    def test_missing_credentials_are_none_rather_than_empty(self):
        assert _settings().credentials.google_api_key is None


class TestCrossFieldValidation:
    def test_tool_calling_is_disabled_by_default(self):
        assert _settings().agent.enable_tool_calling is False

    def test_gemini_tool_calling_is_rejected_at_startup(self):
        with pytest.raises(ValidationError, match="openrouter"):
            _settings(ENABLE_TOOL_CALLING="true", LLM_PROVIDER="gemini")

    def test_tool_limits_must_be_positive_and_bounded(self):
        with pytest.raises(ValidationError):
            _settings(MAX_TOOL_ROUNDS="0")
        with pytest.raises(ValidationError):
            _settings(MAX_TOOL_CALLS_PER_ROUND="6")

    def test_rerank_top_k_may_not_exceed_top_k(self):
        # Asking the reranker for more chunks than retrieval produced is a
        # silent quality bug: it never errors, it just under-fills the context.
        with pytest.raises(ValidationError, match="RERANK_TOP_K"):
            _settings(TOP_K="5", RERANK_TOP_K="10")

    def test_equal_top_k_and_rerank_top_k_is_allowed(self):
        assert _settings(TOP_K="5", RERANK_TOP_K="5").retrieval.rerank_top_k == 5

    def test_the_pairing_is_unconstrained_when_reranking_is_off(self):
        settings = _settings(TOP_K="5", RERANK_TOP_K="10", ENABLE_RERANK="false")

        assert settings.retrieval.rerank_top_k == 10

    def test_chunk_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            _settings(CHUNK_SIZE="500", CHUNK_OVERLAP="500")

    def test_context_budget_must_fit_inside_the_prompt_ceiling(self):
        with pytest.raises(ValidationError, match="CONTEXT_TOKEN_BUDGET"):
            _settings(MAX_PROMPT_TOKENS="1000", CONTEXT_TOKEN_BUDGET="1000")

    def test_embedding_dimension_must_match_a_known_model(self):
        # A silent mismatch produces confidently wrong answers (ADR-015).
        with pytest.raises(ValidationError, match="EMBEDDING_DIMENSION"):
            _settings(EMBEDDING_MODEL="BAAI/bge-small-en-v1.5", EMBEDDING_DIMENSION="1536")

    def test_a_known_model_with_the_right_dimension_is_accepted(self):
        settings = _settings(
            EMBEDDING_PROVIDER="openai",
            EMBEDDING_MODEL="text-embedding-3-small",
            EMBEDDING_DIMENSION="1536",
        )

        assert settings.embedding.dimension == 1536

    def test_switching_to_a_hosted_provider_is_a_configuration_change(self):
        # The whole point of the abstraction: no code changes to move to OpenAI.
        settings = _settings(
            EMBEDDING_PROVIDER="openai",
            EMBEDDING_MODEL="text-embedding-3-large",
            EMBEDDING_DIMENSION="3072",
        )

        assert settings.embedding.provider.value == "openai"

    def test_an_unknown_model_is_trusted_with_the_declared_dimension(self):
        # We cannot know every model. Refusing unknown ones would block any new
        # release; the collection compatibility check still catches a genuine
        # mismatch before a single query is served.
        settings = _settings(EMBEDDING_MODEL="some/new-model", EMBEDDING_DIMENSION="999")

        assert settings.embedding.dimension == 999

    def test_enabling_rerank_with_the_noop_provider_is_contradictory(self):
        with pytest.raises(ValidationError, match="RERANKER_PROVIDER"):
            _settings(ENABLE_RERANK="true", RERANKER_PROVIDER="noop")


class TestDerivedValues:
    def test_production_is_distinguishable_from_local(self):
        assert _settings(APP_ENV="production").app.is_production is True
        assert _settings(APP_ENV="local").app.is_production is False

    def test_distance_metric_is_expressed_as_a_config_level_name(self):
        # Deliberately a config enum, not the domain DistanceMetric: the
        # dependency rule forbids config importing domain.
        assert _settings(DISTANCE_METRIC="cosine").vector_store.distance is (
            DistanceMetricName.COSINE
        )


class TestRangeValidation:
    @pytest.mark.parametrize(
        ("variable", "value"),
        [
            ("TOP_K", "0"),
            ("RERANK_TOP_K", "0"),
            ("CHUNK_SIZE", "0"),
            ("CHUNK_OVERLAP", "-1"),
            ("MAX_OUTPUT_TOKENS", "0"),
            ("EMBEDDING_BATCH_SIZE", "0"),
            ("MAX_UPLOAD_BYTES", "0"),
            ("TEMPERATURE", "-0.5"),
            ("MEMORY_WINDOW_TURNS", "-1"),
        ],
    )
    def test_out_of_range_values_are_rejected(self, variable, value):
        with pytest.raises(ValidationError):
            _settings(**{variable: value})

    def test_an_unrecognised_log_level_is_rejected(self):
        with pytest.raises(ValidationError, match="LOG_LEVEL"):
            _settings(LOG_LEVEL="CHATTY")


class TestEnums:
    def test_an_unknown_provider_is_rejected_rather_than_silently_defaulted(self):
        with pytest.raises(ValidationError):
            _settings(LLM_PROVIDER="not-a-provider")

    def test_only_providers_with_complete_adapters_are_selectable(self):
        assert {p.value for p in LLMProvider} == {"openrouter", "gemini"}
        assert {p.value for p in EmbeddingProvider} == {
            "openai",
            "huggingface",
            "local",
        }
        assert {f.value for f in LogFormat} == {"console", "json"}
        assert {c.value for c in ChunkingStrategyName} == {"recursive"}

    @pytest.mark.parametrize(
        ("variable", "value"),
        [
            ("LLM_PROVIDER", "openai"),
            ("LLM_PROVIDER", "anthropic"),
            ("LLM_PROVIDER", "local"),
            ("EMBEDDING_PROVIDER", "gemini"),
            ("CHUNKING_STRATEGY", "semantic"),
            ("CHUNKING_STRATEGY", "sentence"),
        ],
    )
    def test_unimplemented_options_are_rejected_during_settings_loading(self, variable, value):
        with pytest.raises(ValidationError):
            _settings(**{variable: value})
