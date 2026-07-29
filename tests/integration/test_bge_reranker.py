"""The reranker against a real cross-encoder model.

The unit tests prove the *contract* with an injected scorer. Only a real model
proves the thing the feature exists for: that a cross-encoder actually
distinguishes a passage which answers the question from one that merely shares
its vocabulary.

Uses a small cross-encoder by default so the suite stays runnable. The class
holds no BGE-specific logic -- it wraps any Sentence-Transformers cross-encoder --
so this exercises the same code path the configured BGE model takes. Point it at
the real one with ``RERANKER_TEST_MODEL``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from rag.domain.models import (
    Chunk,
    ChunkMetadata,
    DocumentType,
    ScoredChunk,
    ScoreSource,
)
from rag.infrastructure.reranker import BgeReranker

pytestmark = [pytest.mark.integration]

MODEL = os.environ.get("RERANKER_TEST_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _scored(chunk_id: str, text: str, rank: int) -> ScoredChunk:
    """A candidate as retrieval would hand it over, in fusion order."""
    return ScoredChunk(
        chunk=Chunk(
            text=text,
            metadata=ChunkMetadata(
                document_id="handbook",
                chunk_id=chunk_id,
                ingest_version=1,
                filename="handbook.pdf",
                document_type=DocumentType.PDF,
                chunk_index=rank,
                char_start=0,
                char_end=len(text),
                token_count=16,
                chunking_strategy="recursive",
                embedding_model_id="fake",
                ingested_at=_NOW,
                page_number=rank + 1,
            ),
        ),
        score=1.0 / (rank + 1),
        source=ScoreSource.FUSED,
    )


@pytest.fixture(scope="session")
def reranker() -> BgeReranker:
    """A reranker backed by a real cross-encoder, skipped if unavailable."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("sentence-transformers is not installed")

    instance = BgeReranker(model=MODEL, batch_size=8)
    try:
        instance._load()
    except Exception as exc:  # pragma: no cover - no network, or no such model
        pytest.skip(f"could not load {MODEL}: {exc}")
    return instance


#: Deliberately adversarial ordering: the passage that answers the question is
#: placed last, and the distractors share its vocabulary. Lexical overlap alone
#: cannot separate these; joint scoring can.
LEAVE_CANDIDATES = (
    _scored("policy_ref", "Leave policy questions should be directed to the HR portal.", 0),
    _scored("sick", "Sick leave is recorded separately from annual leave entitlement.", 1),
    _scored("carry", "Unused annual leave may not be carried into the following year.", 2),
    _scored("answer", "Every full-time employee receives twenty-five days of annual leave.", 3),
)


class TestRealModelReordering:
    async def test_it_promotes_the_passage_that_answers_the_question(self, reranker):
        # The passage arrived ranked last. A bi-encoder cannot tell it apart
        # from its neighbours; they all discuss annual leave.
        results = await reranker.rerank(
            "How many days of annual leave do full-time employees get?",
            LEAVE_CANDIDATES,
            top_n=4,
        )

        assert results[0].chunk_id == "answer"

    async def test_it_narrows_twenty_candidates_to_five(self, reranker):
        # The configured pipeline shape: retrieve wide, rerank narrow.
        candidates = (
            *(
                _scored(f"filler_{i}", f"Unrelated administrative note number {i}.", i)
                for i in range(19)
            ),
            _scored("answer", "Employees receive twenty-five days of annual leave.", 19),
        )

        results = await reranker.rerank("How much annual leave?", candidates, top_n=5)

        assert len(results) == 5
        assert results[0].chunk_id == "answer"

    async def test_it_separates_relevant_from_irrelevant(self, reranker):
        results = await reranker.rerank("How many days of annual leave?", LEAVE_CANDIDATES, top_n=4)

        by_id = {r.chunk_id: r.score for r in results}
        assert by_id["answer"] > by_id["policy_ref"]


class TestRealModelScores:
    async def test_scores_lie_in_the_unit_interval(self, reranker):
        results = await reranker.rerank("How much leave?", LEAVE_CANDIDATES, top_n=4)

        assert all(0.0 <= r.score <= 1.0 for r in results)

    async def test_scores_are_usable_as_citation_relevance(self, reranker):
        # The end of the pipeline: a score outside [0, 1] would fail Citation
        # validation after all the work was done.
        from rag.domain.models import Citation

        results = await reranker.rerank("How much leave?", LEAVE_CANDIDATES, top_n=1)

        citation = Citation(
            chunk_ids=(results[0].chunk_id,),
            document_id="handbook",
            filename="handbook.pdf",
            document_type=DocumentType.PDF,
            relevance_score=results[0].score,
        )
        assert citation.relevance_score is not None


class TestRealModelContract:
    async def test_passages_are_returned_unmodified(self, reranker):
        results = await reranker.rerank("How much leave?", LEAVE_CANDIDATES, top_n=4)

        original = {c.chunk_id: c.text for c in LEAVE_CANDIDATES}
        assert all(r.text == original[r.chunk_id] for r in results)

    async def test_metadata_survives_reranking(self, reranker):
        results = await reranker.rerank("How much leave?", LEAVE_CANDIDATES, top_n=1)

        assert results[0].chunk.metadata.page_number is not None

    async def test_every_result_is_marked_reranked(self, reranker):
        results = await reranker.rerank("How much leave?", LEAVE_CANDIDATES, top_n=4)

        assert all(r.source is ScoreSource.RERANKED for r in results)

    async def test_reranking_is_deterministic(self, reranker):
        # Non-deterministic ranking would make any quality measurement noise.
        first = await reranker.rerank("How much leave?", LEAVE_CANDIDATES, top_n=4)
        second = await reranker.rerank("How much leave?", LEAVE_CANDIDATES, top_n=4)

        assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
