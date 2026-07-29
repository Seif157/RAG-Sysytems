"""The documented configuration matches the code that reads it.

``.env.example`` is the only place most people will look to learn what this
system can be configured with. A variable that exists in code but not in the
file is undiscoverable; one in the file but not in code is a lie that someone
will eventually set and wonder why nothing happened.

Neither failure produces an error at runtime, which is exactly why it needs a
test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rag.config import Settings

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ENV_EXAMPLE = ROOT / ".env.example"

_ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=", re.MULTILINE)


def _documented_variables() -> set[str]:
    """Every variable assigned in the example environment file."""
    return set(_ASSIGNMENT.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))


class TestDocumentationMatchesCode:
    def test_the_example_file_exists(self):
        assert ENV_EXAMPLE.exists()

    def test_every_setting_the_code_reads_is_documented(self):
        undocumented = set(Settings.environment_variable_names()) - _documented_variables()

        assert undocumented == set(), (
            f"these settings are read by the code but absent from .env.example, "
            f"so nobody will discover them: {sorted(undocumented)}"
        )

    def test_nothing_documented_is_ignored_by_the_code(self):
        stale = _documented_variables() - set(Settings.environment_variable_names())

        assert stale == set(), (
            f"these variables are documented but read by nothing, so setting "
            f"them does nothing: {sorted(stale)}"
        )


class TestDefaultsAreCoherent:
    def test_the_defaults_load(self):
        # A default set that does not satisfy its own cross-field validation
        # would mean the system cannot start without configuration.
        assert Settings.from_environment({}, use_env_file=False) is not None

    def test_retrieval_is_wider_than_reranking(self):
        settings = Settings.from_environment({}, use_env_file=False)

        assert settings.retrieval.top_k > settings.retrieval.rerank_top_k

    def test_the_context_budget_leaves_room_for_the_rest_of_the_prompt(self):
        settings = Settings.from_environment({}, use_env_file=False)

        assert settings.context.token_budget < settings.context.max_prompt_tokens

    def test_chunks_fit_within_the_reranker_window(self):
        # A cross-encoder truncates at its own max length. Chunks larger than
        # that get silently cut, and the tail never influences the score.
        settings = Settings.from_environment({}, use_env_file=False)

        assert settings.chunking.max_chunk_tokens <= 1024

    def test_generation_is_deterministic_by_default(self):
        # Reproducible answers make quality changes attributable.
        settings = Settings.from_environment({}, use_env_file=False)

        assert settings.llm.temperature == 0.0

    def test_the_documented_embedding_dimension_matches_its_model(self):
        # The mismatch this guards against returns confidently wrong answers.
        settings = Settings.from_environment({}, use_env_file=False)

        from rag.config import expected_dimension_for

        expected = expected_dimension_for(settings.embedding.model)
        assert expected is None or expected == settings.embedding.dimension


class TestTheExampleFileIsUsable:
    def test_it_parses_as_configuration(self):
        # Every documented default must actually be valid; a placeholder that
        # fails validation makes `cp .env.example .env` a broken first step.
        values = dict(
            line.split("=", 1)
            for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
            if _ASSIGNMENT.match(line)
        )
        cleaned = {key: value.split("#")[0].strip() for key, value in values.items()}
        populated = {key: value for key, value in cleaned.items() if value}

        assert Settings.from_environment(populated, use_env_file=False) is not None
