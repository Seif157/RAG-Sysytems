"""Reference data used by configuration validation.

Kept apart from :mod:`rag.config.settings` so the rules can be read and extended
without wading through field declarations.
"""

from __future__ import annotations

from rag.config.enums import EmbeddingProvider

__all__ = [
    "DEFAULT_EMBEDDING_MODELS",
    "KNOWN_EMBEDDING_DIMENSIONS",
    "expected_dimension_for",
]


#: Output dimensionality of embedding models we know about.
#:
#: Catches the single most damaging configuration mistake: a dimension that
#: disagrees with the model. A wrong *width* is rejected by the store, but a
#: correct width paired with the wrong model is not -- that produces confidently
#: wrong answers with plausible citations (ADR-015).
#:
#: Unknown models are trusted with whatever dimension is declared. Refusing them
#: would block every new model release, and the collection compatibility check
#: still catches a genuine mismatch before a single query is served.
KNOWN_EMBEDDING_DIMENSIONS: dict[str, int] = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Google
    "models/embedding-001": 768,
    "models/text-embedding-004": 768,
    "text-embedding-004": 768,
    # BAAI (local / HuggingFace)
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    # Sentence-Transformers
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


#: Default embedding model per provider, so the example configuration stays
#: honest when the provider is changed.
DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.GEMINI: "models/embedding-001",
    EmbeddingProvider.HUGGINGFACE: "BAAI/bge-small-en-v1.5",
    EmbeddingProvider.LOCAL: "BAAI/bge-small-en-v1.5",
}


def expected_dimension_for(model: str) -> int | None:
    """Return the known output dimension of an embedding model.

    Args:
        model: The model identifier as configured.

    Returns:
        The expected dimension, or ``None`` when the model is not one we know
        about -- in which case the declared dimension is trusted.
    """
    return KNOWN_EMBEDDING_DIMENSIONS.get(model)
