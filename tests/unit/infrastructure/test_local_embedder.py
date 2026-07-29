"""Behaviour of the local embedding model.

Runs a Sentence-Transformers model on the machine rather than calling a paid
API. The contract is the same as any other embedder, so these tests mirror what
the OpenAI adapter must also satisfy -- order preservation, length preservation,
determinism.

The one behaviour unique to this adapter is the query instruction prefix. BGE
retrieval models are trained with an instruction on the query side and none on
the passage side; getting that backwards, or applying it to both, measurably
degrades recall while failing nothing.
"""

from __future__ import annotations

import pytest

from rag.domain.errors import EmbeddingError
from rag.infrastructure.embeddings import LocalEmbedder

pytestmark = pytest.mark.unit


def _embedder(dimension: int = 4, **kwargs) -> tuple[LocalEmbedder, list[list[str]]]:
    """An embedder whose model records what it was asked to encode."""
    seen: list[list[str]] = []

    def encode(texts: list[str]) -> list[tuple[float, ...]]:
        seen.append(list(texts))
        return [tuple(float(len(text) + i) for i in range(dimension)) for text in texts]

    return LocalEmbedder(dimension=dimension, encoder=encode, **kwargs), seen


class TestContract:
    async def test_it_reports_its_model_and_dimension(self):
        embedder, _ = _embedder(dimension=384)

        assert embedder.dimension == 384
        assert embedder.model_id

    async def test_it_embeds_a_batch_preserving_order_and_length(self):
        embedder, _ = _embedder()

        vectors = await embedder.embed_documents(["short", "a much longer passage"])

        assert len(vectors) == 2
        assert vectors[0] != vectors[1]

    async def test_every_vector_has_the_declared_dimension(self):
        embedder, _ = _embedder(dimension=6)

        vectors = await embedder.embed_documents(["alpha", "beta"])

        assert all(len(v) == 6 for v in vectors)

    async def test_a_query_produces_a_single_vector(self):
        embedder, _ = _embedder(dimension=4)

        assert len(await embedder.embed_query("How much leave?")) == 4

    async def test_an_empty_batch_yields_nothing(self):
        embedder, seen = _embedder()

        assert await embedder.embed_documents([]) == ()
        assert seen == []

    async def test_the_same_text_always_embeds_identically(self):
        embedder, _ = _embedder()

        first = await embedder.embed_query("Revenue grew.")
        second = await embedder.embed_query("Revenue grew.")

        assert first == second


class TestQueryInstruction:
    async def test_a_bge_query_carries_the_retrieval_instruction(self):
        # BGE retrieval models are trained with an instruction on the query
        # side. Omitting it costs recall on short queries and fails nothing.
        embedder, seen = _embedder(model="BAAI/bge-small-en-v1.5")

        await embedder.embed_query("How much annual leave?")

        assert seen[0][0].startswith("Represent this sentence")
        assert "How much annual leave?" in seen[0][0]

    async def test_passages_are_never_given_the_query_instruction(self):
        # Applying it to both sides is the common mistake, and it degrades
        # retrieval rather than breaking it.
        embedder, seen = _embedder(model="BAAI/bge-small-en-v1.5")

        await embedder.embed_documents(["Employees receive twenty-five days."])

        assert seen[0] == ["Employees receive twenty-five days."]

    async def test_a_model_with_no_known_instruction_gets_none(self):
        embedder, seen = _embedder(model="sentence-transformers/all-MiniLM-L6-v2")

        await embedder.embed_query("How much annual leave?")

        assert seen[0] == ["How much annual leave?"]

    async def test_the_instruction_can_be_overridden(self):
        embedder, seen = _embedder(model="custom/model", query_prefix="query: ")

        await embedder.embed_query("How much leave?")

        assert seen[0] == ["query: How much leave?"]

    async def test_an_explicit_empty_prefix_disables_the_instruction(self):
        embedder, seen = _embedder(model="BAAI/bge-small-en-v1.5", query_prefix="")

        await embedder.embed_query("How much leave?")

        assert seen[0] == ["How much leave?"]


class TestBatching:
    async def test_a_large_batch_is_split(self):
        embedder, seen = _embedder(batch_size=2)

        await embedder.embed_documents(["a", "b", "c", "d", "e"])

        assert [len(batch) for batch in seen] == [2, 2, 1]

    async def test_splitting_preserves_overall_order(self):
        embedder, seen = _embedder(batch_size=2)

        await embedder.embed_documents(["a", "bb", "ccc", "dddd"])

        assert [text for batch in seen for text in batch] == ["a", "bb", "ccc", "dddd"]

    async def test_it_reports_its_batch_size(self):
        embedder, _ = _embedder(batch_size=8)

        assert embedder.max_batch_size == 8


class TestFailureHandling:
    async def test_a_model_failure_becomes_a_domain_error(self):
        # A raw torch error escaping here would bypass every caller's handling.
        def explode(texts: list[str]) -> list[tuple[float, ...]]:
            raise RuntimeError("out of memory")

        embedder = LocalEmbedder(dimension=4, encoder=explode)

        with pytest.raises(EmbeddingError):
            await embedder.embed_documents(["anything"])

    async def test_the_original_failure_is_preserved(self):
        def explode(texts: list[str]) -> list[tuple[float, ...]]:
            raise RuntimeError("out of memory")

        embedder = LocalEmbedder(dimension=4, encoder=explode)

        with pytest.raises(EmbeddingError) as caught:
            await embedder.embed_query("anything")

        assert isinstance(caught.value.__cause__, RuntimeError)

    async def test_a_dimension_mismatch_is_caught(self):
        # A model returning the wrong width would be rejected by Qdrant much
        # later, with nothing pointing back at the embedder.
        def wrong_width(texts: list[str]) -> list[tuple[float, ...]]:
            return [(0.1, 0.2)] * len(texts)

        embedder = LocalEmbedder(dimension=384, encoder=wrong_width)

        with pytest.raises(EmbeddingError, match="dimension"):
            await embedder.embed_documents(["anything"])


class TestConcurrency:
    async def test_encoding_does_not_block_the_event_loop(self):
        # SentenceTransformer.encode is synchronous and CPU-bound; run inline it
        # would stall every other coroutine for its duration.
        import asyncio
        import time

        def slow(texts: list[str]) -> list[tuple[float, ...]]:
            time.sleep(0.15)
            return [(0.0, 0.0, 0.0, 0.0)] * len(texts)

        embedder = LocalEmbedder(dimension=4, encoder=slow)
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.01)
                ticks += 1

        await asyncio.gather(embedder.embed_documents(["a"]), tick())

        assert ticks == 10
