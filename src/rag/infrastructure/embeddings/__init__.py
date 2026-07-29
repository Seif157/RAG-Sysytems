"""Embedding providers.

``LocalEmbedder`` is the default: no API key, no per-call cost, and no document
text leaving the machine. ``OpenAIEmbedder`` remains available for deployments
that prefer hosted quality and have a key.
"""

from rag.infrastructure.embeddings.local_embedder import LocalEmbedder
from rag.infrastructure.embeddings.openai_embedder import OpenAIEmbedder

__all__ = ["LocalEmbedder", "OpenAIEmbedder"]
