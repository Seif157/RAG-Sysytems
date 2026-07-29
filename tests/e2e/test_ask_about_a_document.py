"""The whole slice, end to end: upload a document, ask about it, get a citation.

Wired through the real :class:`~rag.core.container.Container`, so this exercises
the actual composition -- not a hand-assembled arrangement that only exists in
the test. Only the three things that would cost money or need Docker are
substituted.

This is the test that would have caught the prototype's central defect, where
the uploaded file was never read.
"""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.core.container import Container
from rag.domain.models import Query
from tests.fakes import FakeEmbedder, FakeLLMClient, InMemoryVectorStore

pytestmark = pytest.mark.e2e


LEAVE_POLICY = """\
Annual Leave Policy

Every full-time employee receives twenty-five days of paid annual leave per year.

Leave requests must be submitted at least two weeks in advance through the HR portal.

Unused leave may be carried over, but no more than five days may be carried into
the following year.
"""


@pytest.fixture
def container(tmp_path, monkeypatch) -> Container:
    """A fully wired application with no network dependencies."""
    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_environment(
        {
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "CHUNK_SIZE": "220",
            "CHUNK_OVERLAP": "40",
            "TOP_K": "10",
            "RERANK_TOP_K": "3",
            "ENABLE_RERANK": "false",
        },
        use_env_file=False,
    )
    return Container(
        settings,
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
        llm=FakeLLMClient(
            ["Full-time employees receive twenty-five days of paid annual leave [1]."]
        ),
    )


class TestUploadThenAsk:
    async def test_a_document_can_be_ingested_and_then_questioned(self, container):
        document = await container.ingest_document.execute("leave.txt", LEAVE_POLICY.encode())
        assert document.chunk_count > 0

        answer = await container.answer_question.execute(Query(text="How much annual leave?"))

        assert answer.text
        assert answer.has_citations

    async def test_the_citation_points_at_the_uploaded_file(self, container):
        await container.ingest_document.execute("leave.txt", LEAVE_POLICY.encode())

        answer = await container.answer_question.execute(Query(text="How much annual leave?"))

        assert answer.citations[0].filename == "leave.txt"
        assert answer.citations[0].document_type.value == "TXT"

    async def test_the_cited_chunk_really_came_from_the_document(self, container):
        # The citation must resolve to text that is actually in the upload --
        # this is the property the prototype silently violated.
        await container.ingest_document.execute("leave.txt", LEAVE_POLICY.encode())

        answer = await container.answer_question.execute(Query(text="How much annual leave?"))

        snippet = answer.citations[0].snippet or ""
        assert snippet
        normalised = " ".join(LEAVE_POLICY.split())
        assert snippet.rstrip("…") in normalised

    async def test_the_documents_own_words_reach_the_model(self, container):
        await container.ingest_document.execute("leave.txt", LEAVE_POLICY.encode())

        await container.answer_question.execute(Query(text="How much annual leave?"))

        prompt = container.llm.last_prompt
        assert "annual leave" in prompt.user.lower()
        assert "twenty-five days" in prompt.user

    async def test_asking_before_uploading_yields_an_ungrounded_answer(self, container):
        # Nothing indexed: the pipeline still runs and the model is told the
        # documents do not cover the question.
        answer = await container.answer_question.execute(Query(text="How much annual leave?"))

        assert answer.retrieved_count == 0
        assert answer.has_citations is False


class TestReIngestion:
    async def test_a_corrected_document_replaces_the_original(self, container):
        await container.ingest_document.execute("leave.txt", LEAVE_POLICY.encode())
        corrected = LEAVE_POLICY.replace("twenty-five days", "thirty days")

        document = await container.ingest_document.execute("leave.txt", corrected.encode())

        assert document.ingest_version == 2
        await container.answer_question.execute(Query(text="How much annual leave?"))
        assert "thirty days" in container.llm.last_prompt.user
        assert "twenty-five days" not in container.llm.last_prompt.user


class TestConversation:
    async def test_history_is_carried_into_a_follow_up(self, container):
        from datetime import UTC, datetime

        from rag.domain.models import Role, Turn

        await container.ingest_document.execute("leave.txt", LEAVE_POLICY.encode())
        now = datetime.now(UTC)
        history = (
            Turn(role=Role.USER, content="How much annual leave?", created_at=now),
            Turn(role=Role.ASSISTANT, content="Twenty-five days.", created_at=now),
        )

        await container.answer_question.execute(
            Query(text="How far in advance must I request it?"), history=history
        )

        assert "Twenty-five days." in container.llm.last_prompt.user
