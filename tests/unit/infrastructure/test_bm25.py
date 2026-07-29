"""Behaviour of BM25 sparse encoding.

Sparse retrieval exists to catch what embeddings blur: error codes, part
numbers, surnames, exact phrases. So the properties worth testing are about rare
tokens surviving intact and common ones not dominating.

Note what this encoder deliberately does *not* do: compute IDF. Inverse document
frequency needs corpus-wide statistics that a stateless encoder cannot have.
Qdrant computes it server-side from the collection, which is both more correct
and the reason the collection is created with ``modifier=IDF``.
"""

from __future__ import annotations

import pytest

from rag.infrastructure.retrieval import Bm25SparseEncoder

pytestmark = pytest.mark.unit


@pytest.fixture
def encoder() -> Bm25SparseEncoder:
    return Bm25SparseEncoder()


class TestWellFormedOutput:
    async def test_it_produces_a_valid_sparse_vector(self, encoder):
        vector = await encoder.encode_query("Revenue grew by twelve percent")

        assert len(vector.indices) == len(vector.values)
        assert vector.non_zero_count > 0

    async def test_indices_are_non_negative(self, encoder):
        vector = await encoder.encode_query("Revenue grew")

        assert all(index >= 0 for index in vector.indices)

    async def test_weights_are_positive(self, encoder):
        vector = await encoder.encode_query("Revenue grew")

        assert all(value > 0 for value in vector.values)

    async def test_no_index_repeats(self, encoder):
        # A repeated index is an invalid sparse vector and Qdrant rejects it.
        vector = await encoder.encode_query("revenue revenue revenue")

        assert len(set(vector.indices)) == len(vector.indices)

    async def test_it_preserves_order_and_length_for_a_batch(self, encoder):
        vectors = await encoder.encode_documents(["alpha text", "beta text", "gamma text"])

        assert len(vectors) == 3
        first_alone = await encoder.encode_documents(["alpha text"])
        assert vectors[0] == first_alone[0]


class TestDeterminism:
    async def test_the_same_text_always_encodes_identically(self, encoder):
        # Chunk vectors must be reproducible or re-ingestion changes retrieval.
        first = await encoder.encode_query("Revenue grew by twelve percent")
        second = await encoder.encode_query("Revenue grew by twelve percent")

        assert first == second

    async def test_two_encoders_agree(self, encoder):
        # Indices come from hashing, so they must not depend on process state --
        # Python's string hash is randomised per process, which would silently
        # break every stored vector on restart.
        other = Bm25SparseEncoder()

        assert await encoder.encode_query("revenue") == await other.encode_query("revenue")


class TestTokenMatching:
    async def test_a_query_term_matches_the_same_term_in_a_document(self, encoder):
        document = (await encoder.encode_documents(["The revenue figures are here"]))[0]
        query = await encoder.encode_query("revenue")

        assert set(query.indices) & set(document.indices)

    async def test_unrelated_text_shares_no_terms(self, encoder):
        document = (await encoder.encode_documents(["zebra giraffe elephant"]))[0]
        query = await encoder.encode_query("quarterly revenue forecast")

        assert not set(query.indices) & set(document.indices)

    async def test_matching_is_case_insensitive(self, encoder):
        lower = await encoder.encode_query("revenue")
        upper = await encoder.encode_query("REVENUE")

        assert lower.indices == upper.indices

    async def test_punctuation_does_not_prevent_a_match(self, encoder):
        document = (await encoder.encode_documents(["Revenue, grew; sharply."]))[0]
        query = await encoder.encode_query("revenue grew sharply")

        assert set(query.indices) <= set(document.indices)

    async def test_an_exact_identifier_survives_tokenisation(self, encoder):
        # The case sparse retrieval exists for. An embedding blurs this; BM25
        # must not.
        document = (await encoder.encode_documents(["Failure code ERR_CONN_4021 was logged"]))[0]
        query = await encoder.encode_query("ERR_CONN_4021")

        assert set(query.indices) <= set(document.indices)

    async def test_a_numeric_token_is_kept(self, encoder):
        document = (await encoder.encode_documents(["Part number 88421 shipped"]))[0]
        query = await encoder.encode_query("88421")

        assert set(query.indices) <= set(document.indices)


class TestStopWords:
    async def test_common_function_words_are_dropped(self, encoder):
        # "the" appears in nearly every chunk, so it contributes nothing but
        # noise and index size.
        vector = await encoder.encode_query("the")

        assert vector.non_zero_count == 0

    async def test_a_query_of_only_stop_words_encodes_to_nothing(self, encoder):
        # Valid, and correctly matches nothing rather than everything.
        vector = await encoder.encode_query("the and of to a")

        assert vector.non_zero_count == 0

    async def test_content_words_survive_alongside_stop_words(self, encoder):
        with_stops = await encoder.encode_query("the revenue of the company")
        without = await encoder.encode_query("revenue company")

        assert set(with_stops.indices) == set(without.indices)


class TestTermFrequencySaturation:
    async def test_repetition_increases_weight(self, encoder):
        once = (await encoder.encode_documents(["revenue alpha beta gamma delta"]))[0]
        thrice = (await encoder.encode_documents(["revenue revenue revenue alpha"]))[0]

        index = next(iter(set(once.indices) & set(thrice.indices)))
        assert (
            dict(zip(thrice.indices, thrice.values, strict=True))[index]
            > dict(zip(once.indices, once.values, strict=True))[index]
        )

    async def test_repetition_saturates_rather_than_scaling_linearly(self, encoder):
        # BM25's defining property: the tenth occurrence of a word adds far less
        # than the second. Without it, a chunk that spams a term outranks one
        # that actually answers the question.
        def weight_of(vector, index: int) -> float:
            return dict(zip(vector.indices, vector.values, strict=True))[index]

        two = (await encoder.encode_documents(["revenue revenue"]))[0]
        twenty = (await encoder.encode_documents(["revenue " * 20]))[0]

        index = two.indices[0]
        ratio = weight_of(twenty, index) / weight_of(two, index)
        assert 1.0 < ratio < 4.0


class TestEdgeCases:
    async def test_empty_text_encodes_to_an_empty_vector(self, encoder):
        assert (await encoder.encode_query("")).non_zero_count == 0

    async def test_whitespace_only_text_encodes_to_an_empty_vector(self, encoder):
        assert (await encoder.encode_query("   \n\t ")).non_zero_count == 0

    async def test_an_empty_batch_yields_an_empty_result(self, encoder):
        assert await encoder.encode_documents([]) == ()

    async def test_text_of_only_punctuation_encodes_to_nothing(self, encoder):
        assert (await encoder.encode_query("!!! ??? ...")).non_zero_count == 0

    async def test_it_reports_a_model_id_for_provenance(self, encoder):
        assert encoder.model_id
