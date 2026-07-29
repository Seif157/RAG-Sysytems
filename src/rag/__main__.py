"""Diagnostics entry point: ``python -m rag``.

Boots the system and reports what it resolved. Until the API and the worker
exist, this is what "the project runs" means -- and it stays useful afterwards
as the fastest way to answer "is this deployment configured correctly?" without
starting a server.

Deliberately prints no secret. The whole point of holding credentials as
``SecretStr`` is defeated by one diagnostics command that renders them.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

from rag.config import Settings
from rag.core.bootstrap import load_settings, run_startup_checks
from rag.domain.errors import CollectionMismatchError, ConfigurationError, RAGError

__all__ = ["main"]

_USAGE = """usage: python -m rag [--list-settings] [--check-store]

Boots the platform and reports the resolved configuration.

options:
  --list-settings   List every environment variable the system reads and exit.
                    Works even when the current configuration is invalid.
  --check-store     Also connect to Qdrant, create or verify the collection,
                    and report it. Fails if the store is unreachable or holds
                    vectors from a different embedding model.
"""


def _report(settings: Settings, checks: Sequence[str], stream: TextIO) -> None:
    """Print a human-readable summary of a successful boot."""
    lines = [
        "Document RAG Platform -- configuration resolved",
        "",
        f"  environment        {settings.app.env.value}",
        f"  log format         {settings.app.log_format.value}",
        f"  llm                {settings.llm.provider.value} / {settings.llm.model}",
        f"  embedding          {settings.embedding.provider.value} / {settings.embedding.model} "
        f"({settings.embedding.dimension}d)",
        f"  chunking           {settings.chunking.strategy.value} "
        f"(size {settings.chunking.chunk_size}, overlap {settings.chunking.chunk_overlap})",
        f"  qdrant             {settings.vector_store.qdrant_url} "
        f"/ {settings.vector_store.collection_name}",
        f"  retrieval          top_k {settings.retrieval.top_k} "
        f"-> rerank {settings.retrieval.rerank_top_k} "
        f"(hybrid {'on' if settings.retrieval.enable_hybrid else 'off'})",
        f"  reranker           {'on' if settings.reranker.enable_rerank else 'off'} "
        f"/ {settings.reranker.provider.value}",
        f"  memory window      {settings.memory.window_turns} turns",
        f"  uploads            {settings.ingestion.upload_dir}",
        "",
        f"  startup checks     {', '.join(checks)} -- passed",
        "",
    ]
    stream.write("\n".join(lines))


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    stream: TextIO | None = None,
) -> int:
    """Run the diagnostics command.

    Args:
        argv: Command-line arguments, excluding the program name.
        env: Explicit configuration values, overriding the process environment.
            Tests pass this instead of mutating the environment.
        stream: Where to write the report. Defaults to standard output.

    Returns:
        ``0`` when the platform boots, ``1`` when configuration is invalid.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    out = stream if stream is not None else sys.stdout

    if "--help" in arguments or "-h" in arguments:
        out.write(_USAGE)
        return 0

    if "--list-settings" in arguments:
        # Deliberately before any validation: this is most useful precisely when
        # the configuration is broken.
        out.write("Environment variables read by this system:\n\n")
        for name in Settings.environment_variable_names():
            out.write(f"  {name}\n")
        return 0

    check_store = "--check-store" in arguments
    if check_store:
        arguments.remove("--check-store")

    if arguments:
        out.write(f"unrecognised arguments: {' '.join(arguments)}\n\n{_USAGE}")
        return 1

    try:
        settings = load_settings(env, use_env_file=env is None)
        checks = run_startup_checks(settings)
    except ConfigurationError as exc:
        out.write(f"configuration error\n\n  {exc.message}\n\n")
        return 1

    _report(settings, checks, out)

    if check_store:
        return _check_store(settings, out)
    return 0


def _check_store(settings: Settings, out: TextIO) -> int:
    """Connect to the vector store and report its collection.

    Kept out of the default path because it is the only thing here that touches
    the network, and a configuration check should stay usable when the
    infrastructure is down.
    """
    from rag.core.container import Container

    try:
        info = asyncio.run(Container(settings).prepare_store())
    except CollectionMismatchError as exc:
        # Reachable, but holding vectors this configuration cannot use. A
        # different problem from being down, and a different fix.
        out.write(f"  vector store       INCOMPATIBLE\n\n  {exc.message}\n\n")
        return 1
    except RAGError as exc:
        out.write(f"  vector store       UNAVAILABLE\n\n  {exc.message}\n\n")
        return 1

    out.write(
        f"  vector store       {info.name} -- {info.points_count} points, "
        f"{info.dense_dimension}d, {info.embedding_model_id}\n\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
