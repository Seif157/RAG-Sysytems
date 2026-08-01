"""Closed sets of configuration choices.

Every provider selection is an enum rather than a free string. A typo in
``LLM_PROVIDER`` then fails at startup with a list of valid values, instead of
falling through to a default and being discovered when someone asks why the
answers changed.

These are *config-level* names, deliberately distinct from any domain enum:
:mod:`rag.config` imports no other layer (architecture spec section 4.2), so
where a concept exists on both sides -- the distance metric, for instance --
the composition root translates between them.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "AppEnv",
    "ChunkingStrategyName",
    "DistanceMetricName",
    "EmbeddingProvider",
    "LLMProvider",
    "LogFormat",
    "RerankerProvider",
]


class AppEnv(StrEnum):
    """Deployment environment."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    """How log events are rendered.

    ``console`` is human-readable for local work; ``json`` is what log
    aggregation can actually query, and is required anywhere real.
    """

    CONSOLE = "console"
    JSON = "json"


class LLMProvider(StrEnum):
    """Selectable answer-generation providers (ADR-009).

    ``openrouter`` is the default: one credential reaches every model it fronts,
    so choosing a model becomes ``LLM_MODEL`` rather than a code change. Gemini
    remains selectable for a deployment that would rather talk to Google
    directly.
    """

    OPENROUTER = "openrouter"
    GEMINI = "gemini"


class EmbeddingProvider(StrEnum):
    """Selectable embedding providers.

    Changing this invalidates every stored vector and requires a re-index
    (ADR-015).
    """

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class DistanceMetricName(StrEnum):
    """How vector similarity is measured."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class ChunkingStrategyName(StrEnum):
    """Selectable chunking strategies (ADR-021 -- recursive is default)."""

    RECURSIVE = "recursive"


class RerankerProvider(StrEnum):
    """Selectable rerankers. ``noop`` is the identity used when reranking is off."""

    BGE = "bge"
    NOOP = "noop"
