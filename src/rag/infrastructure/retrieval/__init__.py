"""Retriever implementations.

Dense now; sparse and hybrid next phase. Because all three satisfy the same
interface, switching between them is a wiring decision, not a branch.
"""

from rag.infrastructure.retrieval.bm25 import Bm25SparseEncoder
from rag.infrastructure.retrieval.dense_retriever import DenseRetriever
from rag.infrastructure.retrieval.hybrid_retriever import HybridRetriever
from rag.infrastructure.retrieval.sparse_retriever import SparseRetriever

__all__ = ["Bm25SparseEncoder", "DenseRetriever", "HybridRetriever", "SparseRetriever"]
