"""Behaviour of the recursive chunker.

Chunking is the highest-leverage quality lever in a RAG system: it decides what
a retriever can possibly find and what a citation can possibly point at. These
tests pin down the invariants every strategy must satisfy, so the semantic and
sentence strategies can be held to the same standard when they arrive.
"""

from __future__ import annotations

import pytest

from rag.domain.models import ContentBlock, DocumentProperties, ParsedDocument
from rag.infrastructure.chunking import RecursiveChunker

pytestmark = pytest.mark.unit


def _document(*blocks: ContentBlock) -> ParsedDocument:
    return ParsedDocument(blocks=blocks, properties=DocumentProperties())


def _block(text: str, *, start: int = 0, page: int | None = None, section: str | None = None):
    return ContentBlock(
        text=text,
        char_start=start,
        char_end=start + len(text),
        page_number=page,
        section=section,
    )


class TestConstruction:
    def test_overlap_must_be_smaller_than_the_chunk(self):
        # Equal overlap means every chunk repeats its predecessor and the
        # chunker never advances.
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveChunker(chunk_size=0, chunk_overlap=0)

    def test_negative_overlap_is_rejected(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveChunker(chunk_size=100, chunk_overlap=-1)

    def test_it_reports_its_name_for_provenance(self):
        assert RecursiveChunker(chunk_size=100, chunk_overlap=0).name == "recursive"


class TestBasicSplitting:
    def test_an_empty_document_yields_nothing(self):
        assert RecursiveChunker(chunk_size=100, chunk_overlap=0).chunk(_document()) == ()

    def test_text_shorter_than_the_limit_stays_whole(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)

        candidates = chunker.chunk(_document(_block("Revenue grew by twelve percent.")))

        assert len(candidates) == 1
        assert candidates[0].text == "Revenue grew by twelve percent."

    def test_long_text_is_split(self):
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=0)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth one."

        candidates = chunker.chunk(_document(_block(text)))

        assert len(candidates) > 1

    def test_no_chunk_greatly_exceeds_the_configured_size(self):
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=0)
        text = " ".join(f"word{index}" for index in range(200))

        candidates = chunker.chunk(_document(_block(text)))

        # One separator's worth of slack: a piece is added before the limit is
        # re-checked, so a chunk may end slightly over.
        assert all(len(c.text) <= 60 + 20 for c in candidates)

    def test_whitespace_only_text_produces_no_chunks(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)

        assert chunker.chunk(_document(_block("   \n  \t "))) == ()


class TestSplitPreference:
    def test_it_prefers_sentence_boundaries_over_arbitrary_cuts(self):
        # A chunk ending mid-word is both unreadable as a citation and worse to
        # embed than one ending at a sentence.
        chunker = RecursiveChunker(chunk_size=45, chunk_overlap=0)
        text = "The first sentence. The second sentence. The third sentence."

        candidates = chunker.chunk(_document(_block(text)))

        assert all(c.text.endswith(".") for c in candidates)

    def test_text_with_no_whitespace_is_still_split(self):
        # A base64 blob or a very long URL. Cutting mid-token is ugly; silently
        # dropping the content is worse.
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        text = "x" * 100

        candidates = chunker.chunk(_document(_block(text)))

        assert len(candidates) == 5
        assert "".join(c.text for c in candidates) == text


class TestNoContentIsLost:
    def test_every_word_survives_chunking(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = " ".join(f"word{index}" for index in range(60))

        candidates = chunker.chunk(_document(_block(text)))

        recovered = " ".join(c.text for c in candidates).split()
        assert recovered == text.split()


class TestStructureIsInherited:
    def test_a_chunk_inherits_its_blocks_page(self):
        # This is what keeps "page 14" true rather than approximately true.
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)

        candidates = chunker.chunk(_document(_block("Revenue grew.", page=14)))

        assert candidates[0].page_number == 14

    def test_a_chunk_inherits_its_blocks_section(self):
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)

        candidates = chunker.chunk(_document(_block("Revenue grew.", section="3.2 Revenue")))

        assert candidates[0].section == "3.2 Revenue"

    def test_a_chunk_never_spans_two_blocks(self):
        # Blocks are the parser's structural units. A chunk crossing one would
        # carry the page number of only half its own text.
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)

        candidates = chunker.chunk(
            _document(
                _block("Page one content.", start=0, page=1),
                _block("Page two content.", start=20, page=2),
            )
        )

        assert [c.page_number for c in candidates] == [1, 2]

    def test_offsets_are_relative_to_the_document_not_the_block(self):
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)

        candidates = chunker.chunk(_document(_block("Second block.", start=100)))

        assert candidates[0].char_start >= 100


class TestOverlap:
    def test_overlap_repeats_text_between_adjacent_chunks(self):
        # So a sentence split across a boundary is still retrievable whole from
        # one side of it.
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=20)
        text = " ".join(f"word{index}" for index in range(40))

        candidates = chunker.chunk(_document(_block(text)))

        first_words = set(candidates[0].text.split())
        second_words = set(candidates[1].text.split())
        assert first_words & second_words

    def test_zero_overlap_repeats_nothing(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = " ".join(f"word{index}" for index in range(40))

        candidates = chunker.chunk(_document(_block(text)))

        assert not set(candidates[0].text.split()) & set(candidates[1].text.split())


class TestDeterminism:
    def test_the_same_input_produces_identical_chunks(self):
        # Chunk ids are derived from text and position, so a non-deterministic
        # chunker would break re-ingestion idempotency (ADR-014).
        text = " ".join(f"word{index}" for index in range(80))
        document = _document(_block(text))

        first = RecursiveChunker(chunk_size=60, chunk_overlap=10).chunk(document)
        second = RecursiveChunker(chunk_size=60, chunk_overlap=10).chunk(document)

        assert [c.text for c in first] == [c.text for c in second]
        assert [c.char_start for c in first] == [c.char_start for c in second]


class TestTokenCounting:
    def test_the_injected_counter_is_used(self):
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0, count_tokens=lambda t: 42)

        candidates = chunker.chunk(_document(_block("Revenue grew.")))

        assert candidates[0].token_count == 42

    def test_every_chunk_carries_a_positive_token_count(self):
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=0)
        text = " ".join(f"word{index}" for index in range(40))

        candidates = chunker.chunk(_document(_block(text)))

        assert all(c.token_count > 0 for c in candidates)
