"""The interfaces the application depends on.

Seven ports, not twenty-one. Each one earns its place by having a second
implementation that exists *today* -- a different format, a different strategy,
or an in-memory double that keeps the test suite free of network calls and
Docker. Everything else in this system is a plain class, because an interface
with one implementation is indirection, not abstraction.

Ports are declared here -- in the layer that consumes them -- rather than beside
their implementations. That inversion is what makes ``infrastructure`` depend on
``domain`` instead of the reverse.

**Async where there is I/O, synchronous where there is not.** Anything that
talks to a network or a disk is a coroutine, because retrieval fans out to dense
and sparse search concurrently and a synchronous port would serialise it.
Parsing and chunking are CPU-bound and stay synchronous, where they are
considerably easier to write and test.
"""

from rag.domain.ports.chunking_strategy import ChunkingStrategy
from rag.domain.ports.document_parser import DocumentParser
from rag.domain.ports.embedder import Embedder
from rag.domain.ports.llm_client import LLMClient
from rag.domain.ports.reranker import Reranker
from rag.domain.ports.retriever import Retriever
from rag.domain.ports.vector_store import VectorStore

__all__ = [
    "ChunkingStrategy",
    "DocumentParser",
    "Embedder",
    "LLMClient",
    "Reranker",
    "Retriever",
    "VectorStore",
]
