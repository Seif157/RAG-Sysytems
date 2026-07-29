"""In-memory test doubles for the domain ports.

Maintained as first-class code. Unit tests run against these rather than against
mocks, which is what keeps them fast, hermetic and honest -- a mock asserts that
a call was made, a fake asserts that the behaviour was right.

These doubles are the main practical reason the seven ports exist: the whole
test suite runs with no API key, no network and no Docker.
"""

from tests.fakes.embedder import FakeEmbedder
from tests.fakes.llm import FakeLLMClient
from tests.fakes.vector_store import InMemoryVectorStore, matches_filter

__all__ = ["FakeEmbedder", "FakeLLMClient", "InMemoryVectorStore", "matches_filter"]
