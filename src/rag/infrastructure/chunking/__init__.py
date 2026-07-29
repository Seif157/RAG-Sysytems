"""Chunking strategy implementations.

Recursive now; semantic and sentence in the next phase, selected by
``CHUNKING_STRATEGY`` (ADR-021). Chunking is the highest-leverage quality lever
in a RAG system, which is why it is a configurable strategy rather than a
constant.
"""

from rag.infrastructure.chunking.recursive import RecursiveChunker

__all__ = ["RecursiveChunker"]
