"""Translation from configuration into domain value objects.

The dependency rule forbids :mod:`rag.config` from importing :mod:`rag.domain`,
so a concept that exists on both sides -- the distance metric, for instance --
has two representations and something must translate. That something is the
composition root, which is allowed to know both.
"""

from __future__ import annotations

from rag.config import DistanceMetricName, Settings
from rag.domain.models import CollectionSpec, DistanceMetric, GenerationParams
from rag.domain.prompts import PromptSpec

__all__ = [
    "collection_spec_from",
    "distance_metric_from",
    "generation_params_from",
    "prompt_spec_from",
]

_DISTANCE_METRICS: dict[DistanceMetricName, DistanceMetric] = {
    DistanceMetricName.COSINE: DistanceMetric.COSINE,
    DistanceMetricName.DOT: DistanceMetric.DOT,
    DistanceMetricName.EUCLIDEAN: DistanceMetric.EUCLIDEAN,
}


def distance_metric_from(name: DistanceMetricName) -> DistanceMetric:
    """Translate the configured metric name into the domain metric.

    Args:
        name: The configured value.

    Returns:
        The corresponding domain metric.
    """
    return _DISTANCE_METRICS[name]


def collection_spec_from(settings: Settings) -> CollectionSpec:
    """Describe the collection shape the running configuration requires.

    Start-up compares this against the collection that actually exists and
    refuses to run on a mismatch, because vectors from different models are not
    comparable and the resulting answers look right (ADR-015).

    Args:
        settings: The running configuration.

    Returns:
        The required collection shape.
    """
    return CollectionSpec(
        name=settings.vector_store.collection_name,
        dense_dimension=settings.embedding.dimension,
        distance=distance_metric_from(settings.vector_store.distance),
        embedding_model_id=settings.embedding.model,
        supports_sparse=settings.retrieval.enable_hybrid,
    )


def generation_params_from(settings: Settings) -> GenerationParams:
    """Build the provider-neutral generation settings.

    Args:
        settings: The running configuration.

    Returns:
        Parameters to pass to any :class:`~rag.domain.ports.LLMClient`.
    """
    return GenerationParams(
        temperature=settings.llm.temperature,
        max_output_tokens=settings.llm.max_output_tokens,
        timeout_s=settings.llm.timeout_s,
    )


def prompt_spec_from(settings: Settings) -> PromptSpec:
    """Build the contract every prompt must satisfy.

    Grounding, citations and permission to refuse are not configurable: they are
    the product's correctness requirements. Only the version and the token
    ceiling come from configuration.

    Args:
        settings: The running configuration.

    Returns:
        The prompt contract.
    """
    return PromptSpec(
        version=settings.context.prompt_version,
        max_prompt_tokens=settings.context.max_prompt_tokens,
    )
