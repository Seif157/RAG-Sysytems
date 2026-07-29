"""Typed application configuration.

A leaf package: it imports no other layer of the system. Configuration is
*injected* into the objects that need it, never read by them, so a component can
be tested under two different configurations in one test session.

Invalid values and invalid combinations raise at construction, which means a
misconfigured deployment fails before it serves anything (ADR-012).
"""

from rag.config.enums import (
    AppEnv,
    ChunkingStrategyName,
    DistanceMetricName,
    EmbeddingProvider,
    LLMProvider,
    LogFormat,
    RerankerProvider,
)
from rag.config.settings import (
    AppSettings,
    ChunkingSettings,
    ContextSettings,
    CredentialsSettings,
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    MemorySettings,
    RerankerSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from rag.config.validation import KNOWN_EMBEDDING_DIMENSIONS, expected_dimension_for

__all__ = [
    "KNOWN_EMBEDDING_DIMENSIONS",
    "AppEnv",
    "AppSettings",
    "ChunkingSettings",
    "ChunkingStrategyName",
    "ContextSettings",
    "CredentialsSettings",
    "DistanceMetricName",
    "EmbeddingProvider",
    "EmbeddingSettings",
    "IngestionSettings",
    "LLMProvider",
    "LLMSettings",
    "LogFormat",
    "MemorySettings",
    "RerankerProvider",
    "RerankerSettings",
    "RetrievalSettings",
    "Settings",
    "VectorStoreSettings",
    "expected_dimension_for",
]
