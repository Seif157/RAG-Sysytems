"""Chunking strategy implementations.

Recursive chunking is the implemented strategy. Chunking is the highest-leverage quality lever
in a RAG system, which is why it is a configurable strategy rather than a
constant.
"""

from rag.infrastructure.chunking.recursive import RecursiveChunker

__all__ = ["RecursiveChunker"]
