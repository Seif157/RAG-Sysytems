"""Behaviour of prompt assembly.

The system prompt is the most load-bearing text in the project. Everything
upstream decides what the model *can* see; this decides what it is told to do
with it, and the single worst output this system can produce is a confident
answer that no retrieved passage supports.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rag.domain.errors import PromptTooLargeError
from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    ContextBlock,
    DocumentType,
    Role,
    ScoredChunk,
    ScoreSource,
    Turn,
)
from rag.domain.prompts import NOT_FOUND_PHRASE, PromptSpec, is_refusal
from rag.infrastructure.prompts import PromptBuilder

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 28, tzinfo=UTC)
_SPEC = PromptSpec(version="v1", max_prompt_tokens=8192)


def _block(*texts: str) -> ContextBlock:
    chunks = []
    for index, text in enumerate(texts, start=1):
        chunks.append(
            ScoredChunk(
                chunk=Chunk(
                    text=text,
                    metadata=ChunkMetadata(
                        document_id="doc-1",
                        chunk_id=f"c{index}",
                        ingest_version=1,
                        filename="handbook.pdf",
                        document_type=DocumentType.PDF,
                        chunk_index=index,
                        char_start=0,
                        char_end=len(text),
                        token_count=10,
                        chunking_strategy="recursive",
                        embedding_model_id="fake",
                        ingested_at=_NOW,
                        page_number=index,
                    ),
                ),
                score=0.9,
                source=ScoreSource.RERANKED,
            )
        )
    markers = tuple((str(i), c.chunk_id) for i, c in enumerate(chunks, start=1))
    return ContextBlock(
        chunks=tuple(chunks),
        rendered="\n\n".join(f"[{m}] {c.text}" for (m, _), c in zip(markers, chunks, strict=True)),
        token_count=10 * len(chunks),
        marker_to_chunk_id=markers,
    )


class TestSeparation:
    def test_the_system_prompt_is_built_separately_from_the_user_prompt(self):
        prompt = PromptBuilder().build("How much leave?", _block("Body."), spec=_SPEC)

        assert prompt.system
        assert prompt.user
        assert prompt.system != prompt.user

    def test_the_question_goes_in_the_user_prompt_not_the_system_prompt(self):
        prompt = PromptBuilder().build("How much leave?", _block("Body."), spec=_SPEC)

        assert "How much leave?" in prompt.user
        assert "How much leave?" not in prompt.system

    def test_the_context_goes_in_the_user_prompt(self):
        prompt = PromptBuilder().build("q", _block("Distinctive body text."), spec=_SPEC)

        assert "Distinctive body text." in prompt.user
        assert "Distinctive body text." not in prompt.system

    def test_the_prompt_is_tagged_with_the_template_version(self):
        spec = PromptSpec(version="v7", max_prompt_tokens=8192)

        assert PromptBuilder().build("q", _block("Body."), spec=spec).version == "v7"


class TestGroundingInstructions:
    def test_it_tells_the_model_to_answer_only_from_the_context(self):
        system = PromptBuilder().build("q", _block("Body."), spec=_SPEC).system.lower()

        assert "only" in system
        assert "context" in system

    def test_it_forbids_outside_knowledge(self):
        system = PromptBuilder().build("q", _block("Body."), spec=_SPEC).system.lower()

        assert "outside knowledge" in system or "general knowledge" in system

    def test_it_requires_citation_markers(self):
        system = PromptBuilder().build("q", _block("Body."), spec=_SPEC).system

        assert "[1]" in system or "square bracket" in system.lower()

    def test_it_frames_retrieved_text_as_data_not_instruction(self):
        # Mitigation for prompt injection via document content. Mitigation, not
        # a guarantee -- worth stating plainly rather than claiming otherwise.
        system = PromptBuilder().build("q", _block("Body."), spec=_SPEC).system.lower()

        assert "never follow" in system or "not follow" in system


class TestRefusal:
    def test_it_supplies_an_exact_phrase_for_a_missing_answer(self):
        # A canonical phrase is what makes an ungrounded answer *detectable*.
        # "Say so plainly" leaves the model free to phrase it a dozen ways, and
        # neither the UI nor a test can then tell refusal from a real answer.
        system = PromptBuilder().build("q", _block("Body."), spec=_SPEC).system

        assert NOT_FOUND_PHRASE in system

    def test_a_refusal_is_recognisable_downstream(self):
        assert is_refusal(NOT_FOUND_PHRASE) is True

    def test_a_refusal_with_an_appended_sentence_is_still_a_refusal(self):
        # Models append. Treating that as a real answer would present a refusal
        # as grounded.
        assert is_refusal(f"{NOT_FOUND_PHRASE} You may wish to check the intranet.") is True

    def test_a_real_answer_is_not_mistaken_for_a_refusal(self):
        assert is_refusal("Employees receive twenty-five days of leave [1].") is False

    def test_it_forbids_guessing(self):
        system = PromptBuilder().build("q", _block("Body."), spec=_SPEC).system.lower()

        assert "guess" in system

    def test_it_covers_the_partially_answerable_case(self):
        # Most real questions are half-covered. Without instruction the model
        # either refuses everything or invents the missing half.
        system = PromptBuilder().build("q", _block("Body."), spec=_SPEC).system.lower()

        assert "part" in system

    def test_an_empty_context_is_stated_explicitly(self):
        prompt = PromptBuilder().build("q", _block(), spec=_SPEC)

        assert NOT_FOUND_PHRASE in prompt.user

    def test_an_empty_context_does_not_pretend_to_have_passages(self):
        prompt = PromptBuilder().build("q", _block(), spec=_SPEC)

        assert "[1]" not in prompt.user


class TestConversationHistory:
    def test_history_is_included_when_present(self):
        history = (
            Turn(role=Role.USER, content="Who wrote this?", created_at=_NOW),
            Turn(role=Role.ASSISTANT, content="Ada Lovelace.", created_at=_NOW),
        )

        prompt = PromptBuilder().build("When?", _block("Body."), history, _SPEC)

        assert "Ada Lovelace." in prompt.user

    def test_speakers_are_distinguishable(self):
        history = (Turn(role=Role.USER, content="Who wrote this?", created_at=_NOW),)

        prompt = PromptBuilder().build("When?", _block("Body."), history, _SPEC)

        assert "User:" in prompt.user

    def test_no_history_section_appears_on_a_first_turn(self):
        prompt = PromptBuilder().build("q", _block("Body."), (), _SPEC)

        assert "CONVERSATION" not in prompt.user

    def test_history_is_clearly_separated_from_the_context(self):
        # Otherwise the model may cite a previous answer as though it were a
        # retrieved passage.
        history = (Turn(role=Role.ASSISTANT, content="Earlier answer.", created_at=_NOW),)

        prompt = PromptBuilder().build("q", _block("Retrieved body."), history, _SPEC)

        assert prompt.user.index("Earlier answer.") < prompt.user.index("Retrieved body.")


class TestContextOrderIsPreserved:
    def test_passages_appear_in_the_order_the_context_builder_chose(self):
        # Ordering is an attention decision made upstream; re-sorting here would
        # silently undo it.
        block = _block("First passage.", "Second passage.", "Third passage.")

        prompt = PromptBuilder().build("q", block, spec=_SPEC)

        assert (
            prompt.user.index("First passage.")
            < prompt.user.index("Second passage.")
            < prompt.user.index("Third passage.")
        )

    def test_citation_markers_survive_into_the_prompt(self):
        block = _block("First passage.", "Second passage.")

        prompt = PromptBuilder().build("q", block, spec=_SPEC)

        assert "[1]" in prompt.user
        assert "[2]" in prompt.user


class TestTokenCeiling:
    def test_an_oversized_prompt_is_refused_rather_than_truncated(self):
        # Letting the provider truncate drops context with no signal at all.
        spec = PromptSpec(version="v1", max_prompt_tokens=10)

        with pytest.raises(PromptTooLargeError):
            PromptBuilder().build("q", _block("Body text " * 200), spec=spec)

    def test_the_error_reports_the_ceiling_it_exceeded(self):
        spec = PromptSpec(version="v1", max_prompt_tokens=10)

        with pytest.raises(PromptTooLargeError) as caught:
            PromptBuilder().build("q", _block("Body text " * 200), spec=spec)

        assert caught.value.context["limit"] == 10

    def test_a_prompt_within_the_ceiling_reports_its_size(self):
        prompt = PromptBuilder().build("q", _block("Body."), spec=_SPEC)

        assert prompt.estimated_tokens is not None
        assert prompt.estimated_tokens > 0
