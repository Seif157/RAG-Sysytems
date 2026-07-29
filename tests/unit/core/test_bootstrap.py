"""Behaviour of full application start-up.

Every entry point -- the API, the worker, the diagnostics command -- calls
:func:`bootstrap` and nothing else, so they cannot drift apart in how they
start. These tests cover the path those entry points actually take.
"""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.core.bootstrap import bootstrap
from rag.core.container import Container
from rag.core.providers import ProviderRegistry
from rag.domain.errors import ConfigurationError
from rag.domain.models import DistanceMetric

pytestmark = pytest.mark.unit

_LOCAL_ONLY = {"LLM_PROVIDER": "local", "EMBEDDING_PROVIDER": "local"}


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)


class TestBootstrap:
    def test_a_valid_configuration_produces_a_container(self):
        container = bootstrap(_LOCAL_ONLY, use_env_file=False)

        assert isinstance(container, Container)

    def test_the_container_carries_the_loaded_settings(self):
        container = bootstrap({**_LOCAL_ONLY, "TOP_K": "13"}, use_env_file=False)

        assert container.settings.retrieval.top_k == 13

    def test_invalid_configuration_stops_start_up(self):
        with pytest.raises(ConfigurationError, match="RERANK_TOP_K"):
            bootstrap({**_LOCAL_ONLY, "TOP_K": "5", "RERANK_TOP_K": "10"}, use_env_file=False)

    def test_a_failed_check_stops_start_up(self):
        # Fails here rather than on a user's first question.
        with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
            bootstrap({"LLM_PROVIDER": "gemini", "EMBEDDING_PROVIDER": "local"}, use_env_file=False)

    def test_logging_is_configured_as_part_of_start_up(self):
        import structlog

        bootstrap(_LOCAL_ONLY, use_env_file=False)

        assert structlog.is_configured()


class TestContainerDerivedValues:
    def test_it_builds_the_collection_specification(self):
        container = bootstrap(_LOCAL_ONLY, use_env_file=False)

        spec = container.collection_spec()

        assert spec.dense_dimension == 384
        assert spec.distance is DistanceMetric.COSINE

    def test_it_builds_generation_parameters(self):
        container = bootstrap({**_LOCAL_ONLY, "MAX_OUTPUT_TOKENS": "128"}, use_env_file=False)

        assert container.generation_params().max_output_tokens == 128

    def test_it_builds_the_prompt_contract(self):
        container = bootstrap({**_LOCAL_ONLY, "PROMPT_VERSION": "v7"}, use_env_file=False)

        assert container.prompt_spec().version == "v7"


class TestRegistryMembership:
    def test_membership_can_be_checked_before_resolving(self):
        registry: ProviderRegistry[str, str] = ProviderRegistry("llm")
        registry.register("gemini", lambda: "g")

        assert "gemini" in registry
        assert "mystery" not in registry
