"""The Streamlit front end.

A thin client: it renders, collects input, and calls use cases. No parsing, no
retrieval, no prompting happens here. That confinement is what keeps the front
end replaceable, and it is the opposite of the prototype, where the UI callback
*was* the pipeline.

Run with::

    streamlit run src/rag/presentation/ui/app.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from rag.core.bootstrap import bootstrap
from rag.core.container import Container
from rag.domain.errors import RAGError
from rag.domain.models import Answer, Query, Role, Turn
from rag.domain.prompts import is_refusal

_HISTORY_KEY = "conversation"
_INGESTED_KEY = "ingested_hashes"


@st.cache_resource(show_spinner="Starting up — loading models and checking the index…")
def _container() -> Container:
    """Build the application once and reuse it across reruns.

    Streamlit re-executes this script on every interaction. Without caching, a
    fresh Qdrant client and tokenizer would be constructed on each keystroke.

    Preparing the vector store happens here for the same reason: it must run
    before the first upload, and it must run *once*.
    """
    container = bootstrap()
    _run(container.prepare_store())
    # Loading the cross-encoder takes tens of seconds -- minutes on a first run,
    # which downloads it. Paying that here, under a visible spinner, is far
    # better than having it land silently on the first question.
    _run(container.warm_up())
    return container


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from Streamlit's synchronous execution model.

    Streamlit has no event loop of its own, so each call gets a fresh one. That
    is fine here: ingestion and answering are each a single top-level await, and
    the alternative -- managing a long-lived loop across script reruns -- buys
    nothing for a single-user tool.
    """
    return asyncio.run(coroutine)


def _render_answer(answer: Answer) -> None:
    """Render an answer and its citations."""
    if is_refusal(answer.text):
        # A refusal is a correct outcome, not a failure -- but presenting it in
        # the same style as a grounded answer invites it to be misread as one.
        st.info(answer.text)
        st.caption("Nothing in the indexed documents supports an answer to this question.")
        return

    st.markdown(answer.text)

    if answer.citations:
        st.caption("Sources")
        for index, citation in enumerate(answer.citations, start=1):
            location = [citation.filename]
            if citation.page_number is not None:
                location.append(f"page {citation.page_number}")
            if citation.section:
                location.append(citation.section)
            elif citation.heading:
                location.append(citation.heading)

            score = (
                f" · relevance {citation.relevance_score:.2f}"
                if citation.relevance_score is not None
                else ""
            )
            passages = (
                f" · {citation.supporting_chunk_count} passages"
                if citation.supporting_chunk_count > 1
                else ""
            )
            with st.expander(f"[{index}] {' · '.join(location)}{score}{passages}"):
                st.write(citation.snippet or "")
    else:
        st.warning("This answer cites no sources, so it may not be grounded in your documents.")

    if answer.is_degraded:
        st.warning(
            f"Answered on a reduced-quality path: {', '.join(answer.degraded_stages)} "
            f"was unavailable."
        )


def _sidebar(container: Container) -> None:
    """Render document upload and the list of indexed documents."""
    st.sidebar.header("Documents")

    uploaded = st.sidebar.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt", "md", "markdown"],
        help="PDF citations carry page numbers; DOCX and Markdown carry section headings.",
    )

    if uploaded is not None:
        content = uploaded.getvalue()
        seen = st.session_state.setdefault(_INGESTED_KEY, set())
        marker = (uploaded.name, hash(content))
        # Streamlit re-runs the script on every interaction and keeps the
        # uploaded file in the widget, so without this guard a single upload
        # would be re-ingested on every keystroke in the question box.
        if marker not in seen:
            with st.sidebar.status(f"Ingesting {uploaded.name}…", expanded=False) as status:
                try:
                    document = _run(container.ingest_document.execute(uploaded.name, content))
                except RAGError as error:
                    status.update(label=f"Could not ingest {uploaded.name}", state="error")
                    st.sidebar.error(error.message)
                else:
                    seen.add(marker)
                    status.update(
                        label=f"{document.filename} — {document.chunk_count} chunks",
                        state="complete",
                    )

    documents = container.catalog.list_documents()
    if documents:
        st.sidebar.caption(f"{len(documents)} indexed")
        for document in documents:
            st.sidebar.write(
                f"**{document.filename}** — {document.chunk_count} chunks, "
                f"v{document.ingest_version}"
            )
    else:
        st.sidebar.info("No documents yet. Upload one to get started.")

    with st.sidebar.expander("Configuration"):
        settings = container.settings
        st.write(
            {
                "llm": f"{settings.llm.provider.value} / {settings.llm.model}",
                "embedding": settings.embedding.model,
                "chunking": settings.chunking.strategy.value,
                "top_k": settings.retrieval.top_k,
                "rerank_top_k": settings.retrieval.rerank_top_k,
            }
        )


def main() -> None:
    """Render the application."""
    st.set_page_config(page_title="Document RAG", page_icon="📄", layout="centered")
    st.title("Document RAG")
    st.caption("Ask questions about your documents. Every answer cites its sources.")

    try:
        container = _container()
    except RAGError as error:
        st.error(f"Configuration problem: {error.message}")
        st.stop()

    _sidebar(container)

    history: list[Turn] = st.session_state.setdefault(_HISTORY_KEY, [])

    for turn in history:
        with st.chat_message("user" if turn.role is Role.USER else "assistant"):
            st.markdown(turn.content)

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    if not container.catalog.list_documents():
        st.warning("Upload a document first — there is nothing to search yet.")
        return

    with st.chat_message("user"):
        st.markdown(question)

    window = container.settings.memory.window_turns
    with st.chat_message("assistant"), st.spinner("Searching your documents…"):
        try:
            answer = _run(
                container.answer_question.execute(
                    Query(text=question),
                    history=tuple(history[-window:]) if window else (),
                )
            )
        except RAGError as error:
            st.error(error.message)
            return
        _render_answer(answer)

    now = datetime.now(UTC)
    history.append(Turn(role=Role.USER, content=question, created_at=now))
    history.append(Turn(role=Role.ASSISTANT, content=answer.text, created_at=now))


main()
