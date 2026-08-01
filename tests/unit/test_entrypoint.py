"""Behaviour of the ``python -m rag`` diagnostics entry point.

Phase 1 has no server to start, but it does have a system that either boots or
does not. This command is that check: it loads configuration, runs the start-up
validation and reports what was resolved. It is what makes "the project is
runnable after every phase" a verifiable claim rather than an assertion.
"""

from __future__ import annotations

import io

import pytest

from rag.__main__ import main
from rag.config import Settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)


_LOCAL_ONLY = {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "test-key",
    "EMBEDDING_PROVIDER": "local",
}


class TestSuccessfulBoot:
    def test_a_valid_configuration_exits_zero(self):
        assert main([], env=_LOCAL_ONLY, stream=io.StringIO()) == 0

    def test_it_reports_the_resolved_providers(self):
        out = io.StringIO()

        main([], env=_LOCAL_ONLY, stream=out)

        report = out.getvalue()
        assert "local" in report
        assert "qdrant" in report

    def test_it_reports_which_checks_passed(self):
        out = io.StringIO()

        main([], env=_LOCAL_ONLY, stream=out)

        assert "credentials" in out.getvalue()

    def test_it_never_prints_a_secret(self):
        out = io.StringIO()

        main(
            [],
            env={
                "LLM_PROVIDER": "gemini",
                "GOOGLE_API_KEY": "sk-do-not-print",
                **{"EMBEDDING_PROVIDER": "local"},
            },
            stream=out,
        )

        assert "sk-do-not-print" not in out.getvalue()


class TestFailedBoot:
    def test_an_invalid_configuration_exits_non_zero(self):
        env = {"TOP_K": "5", "RERANK_TOP_K": "10", **_LOCAL_ONLY}

        assert main([], env=env, stream=io.StringIO()) == 1

    def test_it_explains_what_is_wrong(self):
        out = io.StringIO()
        env = {"TOP_K": "5", "RERANK_TOP_K": "10", **_LOCAL_ONLY}

        main([], env=env, stream=out)

        assert "RERANK_TOP_K" in out.getvalue()

    def test_a_missing_credential_exits_non_zero_and_names_the_variable(self):
        out = io.StringIO()

        exit_code = main(
            [], env={"LLM_PROVIDER": "gemini", "EMBEDDING_PROVIDER": "local"}, stream=out
        )

        assert exit_code == 1
        assert "GOOGLE_API_KEY" in out.getvalue()


class TestVariableListing:
    def test_it_can_list_every_variable_the_system_reads(self):
        out = io.StringIO()

        exit_code = main(["--list-settings"], env=_LOCAL_ONLY, stream=out)

        assert exit_code == 0
        assert "TOP_K" in out.getvalue()
        assert "CHUNKING_STRATEGY" in out.getvalue()

    def test_listing_does_not_require_a_valid_configuration(self):
        # Useful precisely when configuration is broken.
        out = io.StringIO()
        env = {"TOP_K": "5", "RERANK_TOP_K": "10"}

        assert main(["--list-settings"], env=env, stream=out) == 0
