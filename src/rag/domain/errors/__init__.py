"""The domain-owned exception hierarchy.

Every layer can catch these by type without importing infrastructure. Adapters
translate vendor exceptions into these types at their own boundary, preserving
the original traceback via ``__cause__`` (ADR-017).
"""

from rag.domain.errors.exceptions import (
    ChunkingError,
    CollectionMismatchError,
    CollectionNotFoundError,
    ConfigurationError,
    CorruptDocumentError,
    DocumentNotFoundError,
    DocumentParsingError,
    DocumentProcessingError,
    DocumentTooLargeError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingRateLimitError,
    EncryptedDocumentError,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    PromptError,
    PromptTooLargeError,
    RAGError,
    RerankingError,
    RetrievalError,
    UnsupportedFormatError,
    VectorStoreError,
    VectorStoreUnavailableError,
)

__all__ = [
    "ChunkingError",
    "CollectionMismatchError",
    "CollectionNotFoundError",
    "ConfigurationError",
    "CorruptDocumentError",
    "DocumentNotFoundError",
    "DocumentParsingError",
    "DocumentProcessingError",
    "DocumentTooLargeError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingRateLimitError",
    "EncryptedDocumentError",
    "LLMContentFilterError",
    "LLMError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "PromptError",
    "PromptTooLargeError",
    "RAGError",
    "RerankingError",
    "RetrievalError",
    "UnsupportedFormatError",
    "VectorStoreError",
    "VectorStoreUnavailableError",
]
