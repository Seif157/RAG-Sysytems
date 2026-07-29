"""Structural metadata survives the whole pipeline into a citation.

This is the point of parsing PDFs and DOCX at all. Extracting a page number is
worth nothing if it is lost between the parser and the answer, and every stage
in between -- chunker, metadata stamping, vector payload, context rendering,
citation assembly -- is somewhere it could be dropped silently.

So the assertion is made where a user would notice it: on the citation.
"""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.core.container import Container
from rag.domain.models import Query
from tests.fakes import FakeEmbedder, FakeLLMClient, InMemoryVectorStore
from tests.fixtures import make_docx, make_markdown, make_pdf

pytestmark = pytest.mark.e2e


#: Cites several markers so the assertions do not depend on which passage lands
#: first. Context ordering places the strongest material at the edges and keeps
#: each document in reading order, so rank does not determine marker number.
_CITE_EVERYTHING = "As stated [1][2][3][4][5]."


def _citation_containing(answer, needle: str):
    """The citation whose snippet contains a phrase, or None."""
    return next((c for c in answer.citations if needle.lower() in (c.snippet or "").lower()), None)


def _container(tmp_path, monkeypatch, answer: str = _CITE_EVERYTHING) -> Container:
    for name in Settings.environment_variable_names():
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_environment(
        {
            "UPLOAD_DIR": str(tmp_path / "uploads"),
            "CHUNK_SIZE": "400",
            "CHUNK_OVERLAP": "40",
            "ENABLE_RERANK": "false",
        },
        use_env_file=False,
    )
    return Container(
        settings,
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
        llm=FakeLLMClient([answer]),
    )


class TestPdfCitations:
    async def test_a_citation_carries_the_page_the_text_came_from(self, tmp_path, monkeypatch):
        container = _container(tmp_path, monkeypatch)
        pdf = make_pdf(
            [
                "Introduction to the annual report.",
                "Revenue grew by twelve percent across every region.",
            ]
        )
        await container.ingest_document.execute("annual.pdf", pdf)

        answer = await container.answer_question.execute(Query(text="Revenue grew by twelve"))

        assert answer.citations
        assert all(c.page_number in (1, 2) for c in answer.citations)

    async def test_the_page_number_matches_the_page_holding_the_text(self, tmp_path, monkeypatch):
        container = _container(tmp_path, monkeypatch)
        pdf = make_pdf(["Nothing relevant here.", "The magic phrase is sarsaparilla."])
        await container.ingest_document.execute("annual.pdf", pdf)

        answer = await container.answer_question.execute(
            Query(text="The magic phrase is sarsaparilla.")
        )

        cited = _citation_containing(answer, "sarsaparilla")
        assert cited is not None
        assert cited.page_number == 2

    async def test_the_document_author_reaches_the_chunk_metadata(self, tmp_path, monkeypatch):
        # Needed for the `author = "..."` filter to be more than a language
        # feature with nothing behind it.
        container = _container(tmp_path, monkeypatch)
        pdf = make_pdf(["Body text here."], author="Ada Lovelace", title="Annual Report")

        await container.ingest_document.execute("annual.pdf", pdf)

        store = container.vector_store
        chunk = store._chunks[store.chunk_ids()[0]]
        assert chunk.metadata.author == "Ada Lovelace"
        assert chunk.metadata.title == "Annual Report"


class TestDocxCitations:
    async def test_a_citation_carries_the_section_heading(self, tmp_path, monkeypatch):
        container = _container(tmp_path, monkeypatch)
        docx = make_docx(
            [
                ("Heading 1", "Financials"),
                ("Heading 2", "Revenue"),
                ("Normal", "Revenue grew by twelve percent across every region."),
            ]
        )
        await container.ingest_document.execute("report.docx", docx)

        answer = await container.answer_question.execute(
            Query(text="Revenue grew by twelve percent across every region.")
        )

        cited = _citation_containing(answer, "Revenue grew by twelve percent")
        assert cited is not None
        assert cited.section == "Financials > Revenue"
        assert cited.heading == "Revenue"

    async def test_docx_citations_claim_no_page_number(self, tmp_path, monkeypatch):
        # Word has no fixed pagination; a page number would point nowhere.
        container = _container(tmp_path, monkeypatch)
        docx = make_docx([("Normal", "The magic phrase is sarsaparilla.")])
        await container.ingest_document.execute("report.docx", docx)

        answer = await container.answer_question.execute(
            Query(text="The magic phrase is sarsaparilla.")
        )

        assert answer.citations[0].page_number is None


class TestMarkdownCitations:
    async def test_a_citation_carries_the_heading_breadcrumb(self, tmp_path, monkeypatch):
        container = _container(tmp_path, monkeypatch)
        source = "# Handbook\n\n## Leave Policy\n\nEmployees receive twenty-five days.\n"
        await container.ingest_document.execute("handbook.md", make_markdown(source))

        answer = await container.answer_question.execute(
            Query(text="Employees receive twenty-five days.")
        )

        # The heading block and the body beneath it share a location, so they
        # merge into one citation for that section.
        assert any(c.section == "Handbook > Leave Policy" for c in answer.citations)


class TestContextTellsTheModelWhereTextCameFrom:
    async def test_the_prompt_names_the_page(self, tmp_path, monkeypatch):
        # The model can only attribute accurately if it is told the provenance.
        container = _container(tmp_path, monkeypatch)
        pdf = make_pdf(["Filler.", "The magic phrase is sarsaparilla."])
        await container.ingest_document.execute("annual.pdf", pdf)

        await container.answer_question.execute(Query(text="The magic phrase is sarsaparilla."))

        assert "page 2" in container.llm.last_prompt.user

    async def test_the_prompt_names_the_section(self, tmp_path, monkeypatch):
        container = _container(tmp_path, monkeypatch)
        source = "# Handbook\n\n## Leave Policy\n\nEmployees receive twenty-five days.\n"
        await container.ingest_document.execute("handbook.md", make_markdown(source))

        await container.answer_question.execute(Query(text="Employees receive twenty-five days."))

        assert "Handbook > Leave Policy" in container.llm.last_prompt.user


class TestEveryFormatIngests:
    @pytest.mark.parametrize(
        ("filename", "build"),
        [
            ("a.pdf", lambda: make_pdf(["Some content here."])),
            ("a.docx", lambda: make_docx([("Normal", "Some content here.")])),
            ("a.md", lambda: make_markdown("Some content here.\n")),
            ("a.txt", lambda: b"Some content here."),
        ],
    )
    async def test_it_ingests_and_indexes(self, tmp_path, monkeypatch, filename, build):
        container = _container(tmp_path, monkeypatch)

        document = await container.ingest_document.execute(filename, build())

        assert document.chunk_count > 0
