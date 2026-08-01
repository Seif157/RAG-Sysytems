"""The ``--check-store`` diagnostics flag.

Verifying that the vector store is reachable and compatible should not require
launching Streamlit and uploading a file to find out.
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


_LOCAL = {
    "LLM_PROVIDER": "openrouter",
    "OPENROUTER_API_KEY": "test-key",
    "EMBEDDING_PROVIDER": "local",
    "ENABLE_RERANK": "false",
}


class TestCheckStore:
    def test_an_unreachable_store_exits_non_zero(self):
        out = io.StringIO()
        env = {**_LOCAL, "QDRANT_URL": "http://127.0.0.1:1"}

        exit_code = main(["--check-store"], env=env, stream=out)

        assert exit_code == 1

    def test_it_explains_that_the_store_could_not_be_reached(self):
        out = io.StringIO()
        env = {**_LOCAL, "QDRANT_URL": "http://127.0.0.1:1"}

        main(["--check-store"], env=env, stream=out)

        report = out.getvalue().lower()
        assert "qdrant" in report or "store" in report

    def test_a_bad_configuration_still_fails_before_touching_the_store(self):
        out = io.StringIO()
        env = {**_LOCAL, "TOP_K": "5", "RERANK_TOP_K": "10", "ENABLE_RERANK": "true"}

        assert main(["--check-store"], env=env, stream=out) == 1
        assert "RERANK_TOP_K" in out.getvalue()
