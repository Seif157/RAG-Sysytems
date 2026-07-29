"""Reranker implementations.

The no-op is what gets wired when reranking is disabled; the BGE cross-encoder
arrives next phase.
"""

from rag.infrastructure.reranker.bge_reranker import BgeReranker
from rag.infrastructure.reranker.noop_reranker import NoOpReranker

__all__ = ["BgeReranker", "NoOpReranker"]
