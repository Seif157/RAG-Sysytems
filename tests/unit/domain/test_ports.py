"""Structural contract of the domain ports.

Seven ports. Each is here because a second implementation exists today -- a
different document format, a different chunking strategy, or an in-memory double
that keeps the test suite free of API keys and Docker. Anything with one
implementation is a plain class, not a port.

Ports carry no behaviour, so what is worth testing is their *shape*: that they
are abstract, declare exactly the members intended, and make I/O asynchronous.
That last one matters -- a synchronous port would serialise the concurrent
dense/sparse fan-out that hybrid retrieval depends on.
"""

from __future__ import annotations

import inspect

import pytest

from rag.domain import ports

pytestmark = pytest.mark.unit


# (port class, expected abstract members, members that must be awaitable)
PORT_SURFACE: list[tuple[type, set[str], set[str]]] = [
    (ports.DocumentParser, {"parse", "supported_type"}, set()),
    (ports.ChunkingStrategy, {"chunk", "name"}, set()),
    (
        ports.Embedder,
        {"embed_documents", "embed_query", "model_id", "dimension", "max_batch_size"},
        {"embed_documents", "embed_query"},
    ),
    (
        ports.VectorStore,
        {
            "ensure_collection",
            "collection_info",
            "upsert",
            "search_dense",
            "search_sparse",
            "delete_document",
        },
        {
            "ensure_collection",
            "collection_info",
            "upsert",
            "search_dense",
            "search_sparse",
            "delete_document",
        },
    ),
    (ports.Retriever, {"retrieve"}, {"retrieve"}),
    (ports.Reranker, {"rerank"}, {"rerank"}),
    (ports.LLMClient, {"generate", "stream", "model_id"}, {"generate"}),
]

PORT_IDS = [port.__name__ for port, _, _ in PORT_SURFACE]


@pytest.mark.parametrize(("port", "expected", "_awaitable"), PORT_SURFACE, ids=PORT_IDS)
def test_port_is_abstract(port, expected, _awaitable):
    assert inspect.isabstract(port), f"{port.__name__} must not be instantiable"


@pytest.mark.parametrize(("port", "expected", "_awaitable"), PORT_SURFACE, ids=PORT_IDS)
def test_port_cannot_be_instantiated(port, expected, _awaitable):
    with pytest.raises(TypeError):
        port()  # type: ignore[abstract]


@pytest.mark.parametrize(("port", "expected", "_awaitable"), PORT_SURFACE, ids=PORT_IDS)
def test_port_declares_exactly_the_expected_members(port, expected, _awaitable):
    assert set(port.__abstractmethods__) == expected


@pytest.mark.parametrize(("port", "_expected", "awaitable"), PORT_SURFACE, ids=PORT_IDS)
def test_io_bound_members_are_asynchronous(port, _expected, awaitable):
    for name in awaitable:
        assert inspect.iscoroutinefunction(getattr(port, name)), (
            f"{port.__name__}.{name} performs I/O and must be async, "
            f"otherwise concurrent fan-out becomes serial"
        )


@pytest.mark.parametrize(("port", "expected", "_awaitable"), PORT_SURFACE, ids=PORT_IDS)
def test_cpu_bound_ports_stay_synchronous(port, expected, _awaitable):
    # Parsing and chunking run in-process over data already in memory. Making
    # them async would buy nothing and cost every caller an await.
    if port in (ports.DocumentParser, ports.ChunkingStrategy):
        for name in expected:
            member = getattr(port, name)
            target = member.fget if isinstance(member, property) else member
            assert not inspect.iscoroutinefunction(target)


@pytest.mark.parametrize(("port", "expected", "_awaitable"), PORT_SURFACE, ids=PORT_IDS)
def test_every_member_is_documented(port, expected, _awaitable):
    for name in expected:
        member = getattr(port, name)
        target = member.fget if isinstance(member, property) else member
        assert target.__doc__, f"{port.__name__}.{name} has no docstring"


def test_the_package_exports_exactly_these_seven_ports():
    # A guard against abstraction creep: adding a port should be a deliberate
    # decision that updates this list, not something that happens quietly.
    assert set(ports.__all__) == {port.__name__ for port, _, _ in PORT_SURFACE}
    assert len(ports.__all__) == 7


class TestPortsAreImplementable:
    def test_a_minimal_retriever_can_be_written(self):
        # Proves the contract is complete: nothing abstract is left over.
        class NullRetriever(ports.Retriever):
            async def retrieve(self, request):
                """Return nothing."""
                return ()

        assert NullRetriever() is not None

    def test_a_partial_implementation_is_still_rejected(self):
        class HalfBaked(ports.Embedder):
            async def embed_query(self, text):
                """Embed one query."""
                return (0.0,)

        with pytest.raises(TypeError):
            HalfBaked()  # type: ignore[abstract]
