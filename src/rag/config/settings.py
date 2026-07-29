"""Typed application configuration (ADR-012).

Every value the system reads is declared here, with a type, a range and a
documented meaning. Nothing calls ``os.getenv`` outside this package, and
settings are **injected** into the objects that need them rather than read by
them -- a service that reads configuration cannot be tested with two different
configurations in one test session.

Two design points worth stating:

*Section objects, flat variable names.* Access is grouped
(``settings.retrieval.top_k``) because that reads well and keeps related values
together, but each variable keeps the exact flat name from the specification
(``TOP_K``). Grouping is for humans reading code; the names are a deployment
contract.

*Validation is cross-field and it fails at startup.* ``RERANK_TOP_K > TOP_K``
never raises at runtime -- it just quietly under-fills the context. An embedding
dimension that disagrees with its model returns confidently wrong answers. Both
are caught here, before the process serves anything.

This package imports no other layer (architecture spec section 4.2). Where a
concept also exists in the domain -- the distance metric, for instance -- the
composition root translates between the two.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Self

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.config.enums import (
    AppEnv,
    ChunkingStrategyName,
    DistanceMetricName,
    EmbeddingProvider,
    LLMProvider,
    LogFormat,
    RerankerProvider,
)
from rag.config.validation import expected_dimension_for

__all__ = [
    "AppSettings",
    "ChunkingSettings",
    "ContextSettings",
    "CredentialsSettings",
    "EmbeddingSettings",
    "IngestionSettings",
    "LLMSettings",
    "MemorySettings",
    "RerankerSettings",
    "RetrievalSettings",
    "Settings",
    "VectorStoreSettings",
]

_ENV_FILE = ".env"


class _Section(BaseSettings):
    """Base for a configuration section.

    ``case_sensitive`` is on and every field declares an explicit uppercase
    alias, so the environment variable names are exactly those in the
    specification -- not whatever a field name happens to produce.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True,
        protected_namespaces=(),
    )


def _optional_secret(value: Any) -> Any:
    """Treat a blank environment variable as absent rather than as an empty secret."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _split_csv(value: str) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into a tuple."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
class AppSettings(_Section):
    """Process-level settings.

    Attributes:
        env: Deployment environment.
        log_level: Standard logging level name.
        log_format: ``console`` for local work, ``json`` anywhere real.
    """

    env: AppEnv = Field(default=AppEnv.LOCAL, validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: LogFormat = Field(default=LogFormat.CONSOLE, validation_alias="LOG_FORMAT")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Reject a log level name the logging module would not understand."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}, got {value!r}")
        return upper

    @property
    def is_production(self) -> bool:
        """Whether relaxed, developer-only behaviour must be refused."""
        return self.env is AppEnv.PRODUCTION


class CredentialsSettings(_Section):
    """Provider API keys.

    Held as :class:`~pydantic.SecretStr` so a key cannot be leaked by an
    accidental ``repr`` in a log line or a traceback.

    Attributes:
        openrouter_api_key: Credential for OpenRouter. One key reaches every
            model it fronts, which is what makes ``LLM_MODEL`` the only thing
            that changes when the answer model changes.
        google_api_key: Credential for Gemini.
        openai_api_key: Credential for OpenAI.
        anthropic_api_key: Credential for Anthropic.
    """

    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    google_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    _blank_is_absent = field_validator(
        "openrouter_api_key",
        "google_api_key",
        "openai_api_key",
        "anthropic_api_key",
        mode="before",
    )(_optional_secret)


class LLMSettings(_Section):
    """Answer generation settings.

    Attributes:
        provider: Which LLM adapter to wire.
        model: Provider-specific model identifier. Held as a free string rather
            than an enum precisely because a provider's catalogue changes
            weekly; the deployment names the model, the code never does.
        openrouter_base_url: OpenRouter's OpenAI-compatible endpoint.
            Configurable so a gateway or proxy in front of it stays a
            deployment concern. Ignored by other providers.
        temperature: Sampling temperature. Must be ``0.0`` for response caching
            to be sound.
        max_output_tokens: Ceiling on generated tokens.
        timeout_s: How long to wait before raising a timeout error.
    """

    # OpenRouter by default: one credential fronts every model it offers, so
    # changing the answer model is a change to LLM_MODEL and nothing else.
    # Gemini remains a one-line switch for talking to Google directly.
    provider: LLMProvider = Field(default=LLMProvider.OPENROUTER, validation_alias="LLM_PROVIDER")
    model: str = Field(default="qwen/qwen3-8b", validation_alias="LLM_MODEL")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL"
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, validation_alias="TEMPERATURE")
    max_output_tokens: int = Field(default=2048, ge=1, validation_alias="MAX_OUTPUT_TOKENS")
    timeout_s: float = Field(default=60.0, gt=0.0, validation_alias="LLM_TIMEOUT_S")


class EmbeddingSettings(_Section):
    """Embedding settings.

    Changing :attr:`model` invalidates every stored vector. The collection
    records which model produced it and startup refuses to run on a mismatch,
    because a silent mismatch is the worst failure mode in the system: plausible
    citations attached to a wrong answer (ADR-015).

    Attributes:
        provider: Which embedding adapter to wire.
        model: Provider-specific model identifier.
        dimension: Output dimensionality. Checked against the model.
        batch_size: How many texts to embed per provider call.
    """

    # Local by default: no API key, no per-call cost, and no document text
    # leaving the machine. `text-embedding-3-small` remains a one-line switch
    # for deployments that prefer hosted quality and have a key.
    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.LOCAL, validation_alias="EMBEDDING_PROVIDER"
    )
    model: str = Field(default="BAAI/bge-small-en-v1.5", validation_alias="EMBEDDING_MODEL")
    dimension: int = Field(default=384, ge=1, validation_alias="EMBEDDING_DIMENSION")
    # Smaller than the OpenAI default: batches here are bounded by local memory
    # and a forward pass, not by a request size limit.
    batch_size: int = Field(default=32, ge=1, validation_alias="EMBEDDING_BATCH_SIZE")

    @model_validator(mode="after")
    def _dimension_matches_model(self) -> Self:
        """Reject a dimension that disagrees with a model we know."""
        expected = expected_dimension_for(self.model)
        if expected is not None and expected != self.dimension:
            raise ValueError(
                f"EMBEDDING_DIMENSION is {self.dimension} but EMBEDDING_MODEL "
                f"{self.model!r} produces {expected}-dimensional vectors"
            )
        return self


class ChunkingSettings(_Section):
    """Chunking settings (ADR-021).

    Attributes:
        strategy: Which chunking strategy to wire.
        chunk_size: Target chunk size, in the strategy's own units.
        chunk_overlap: Overlap between adjacent chunks.
        max_chunk_tokens: Hard ceiling every strategy must respect.
        semantic_breakpoint_percentile: Similarity percentile at which the
            semantic strategy splits. Ignored by other strategies.
    """

    strategy: ChunkingStrategyName = Field(
        default=ChunkingStrategyName.RECURSIVE, validation_alias="CHUNKING_STRATEGY"
    )
    chunk_size: int = Field(default=800, ge=1, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, ge=0, validation_alias="CHUNK_OVERLAP")
    max_chunk_tokens: int = Field(default=1024, ge=1, validation_alias="MAX_CHUNK_TOKENS")
    semantic_breakpoint_percentile: int = Field(
        default=95, ge=1, le=99, validation_alias="SEMANTIC_BREAKPOINT_PERCENTILE"
    )

    @model_validator(mode="after")
    def _overlap_fits_inside_chunk(self) -> Self:
        """Reject an overlap that meets or exceeds the chunk size.

        An overlap equal to the chunk size makes every chunk identical to its
        neighbour and the chunker never advances.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self


class VectorStoreSettings(_Section):
    """Vector database settings.

    Attributes:
        qdrant_url: Qdrant endpoint.
        qdrant_api_key: Qdrant credential, when the deployment requires one.
        collection_name: Collection holding chunk vectors.
        distance: Similarity metric. Part of the collection's identity.
    """

    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str = Field(default="document_chunks", validation_alias="COLLECTION_NAME")
    distance: DistanceMetricName = Field(
        default=DistanceMetricName.COSINE, validation_alias="DISTANCE_METRIC"
    )

    _blank_is_absent = field_validator("qdrant_api_key", mode="before")(_optional_secret)


class RetrievalSettings(_Section):
    """Retrieval settings (ADR-006, ADR-007).

    Attributes:
        top_k: Candidates retrieved before reranking.
        rerank_top_k: Chunks passed to the LLM. Never more than :attr:`top_k`.
        enable_hybrid: Whether to fan out to sparse search alongside dense.
        rrf_k: Reciprocal rank fusion constant. Larger values flatten the
            contribution of rank position.
        sparse_weight: Relative weight of the sparse list during fusion.
    """

    top_k: int = Field(default=20, ge=1, validation_alias="TOP_K")
    rerank_top_k: int = Field(default=5, ge=1, validation_alias="RERANK_TOP_K")
    enable_hybrid: bool = Field(default=True, validation_alias="ENABLE_HYBRID")
    rrf_k: int = Field(default=60, ge=1, validation_alias="RRF_K")
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0, validation_alias="SPARSE_WEIGHT")

    # The relationship between top_k and rerank_top_k depends on whether
    # reranking is enabled at all, so it is validated at the root rather than
    # here -- this section cannot see the reranker's toggle.


class RerankerSettings(_Section):
    """Reranker settings (ADR-008).

    Attributes:
        enable_rerank: Whether to rerank at all.
        provider: Which reranker adapter to wire.
        model: Cross-encoder model identifier.
        batch_size: Query/passage pairs scored per batch.
    """

    enable_rerank: bool = Field(default=True, validation_alias="ENABLE_RERANK")
    provider: RerankerProvider = Field(
        default=RerankerProvider.BGE, validation_alias="RERANKER_PROVIDER"
    )
    # Defaults to the base model rather than v2-m3. Measured on CPU with 20
    # candidates, v2-m3 costs ~2100 ms per query against ~400 ms for base, and
    # reranking is otherwise 99% of the non-generation query time. v2-m3 remains
    # the better choice on a GPU, or wherever quality outranks latency.
    model: str = Field(default="BAAI/bge-reranker-base", validation_alias="RERANKER_MODEL")
    batch_size: int = Field(default=16, ge=1, validation_alias="RERANKER_BATCH_SIZE")

    @model_validator(mode="after")
    def _enabled_reranker_must_actually_rerank(self) -> Self:
        """Reject the contradictory "reranking on, reranker is a no-op" pairing."""
        if self.enable_rerank and self.provider is RerankerProvider.NOOP:
            raise ValueError(
                "ENABLE_RERANK is true but RERANKER_PROVIDER is 'noop'; "
                "set ENABLE_RERANK=false to disable reranking"
            )
        return self


class MemorySettings(_Section):
    """Conversation memory settings (ADR-018).

    History is held in the Streamlit session, so there is no backend to choose.

    Attributes:
        window_turns: How many recent turns reach the prompt. Applied at read
            time, so changing it takes effect immediately and does not discard
            history a longer window would still want.
    """

    window_turns: int = Field(default=6, ge=0, validation_alias="MEMORY_WINDOW_TURNS")


class ContextSettings(_Section):
    """Prompt and context construction settings.

    Attributes:
        max_prompt_tokens: Hard ceiling on the assembled prompt.
        token_budget: Share of the prompt reserved for retrieved context. Must
            leave room for the instruction, the history and the question.
        prompt_version: Template version, recorded on every answer.
    """

    max_prompt_tokens: int = Field(default=8192, ge=1, validation_alias="MAX_PROMPT_TOKENS")
    token_budget: int = Field(default=4096, ge=1, validation_alias="CONTEXT_TOKEN_BUDGET")
    prompt_version: str = Field(default="v1", validation_alias="PROMPT_VERSION")

    @model_validator(mode="after")
    def _context_leaves_room_for_the_rest_of_the_prompt(self) -> Self:
        """Reject a context budget that consumes the entire prompt ceiling."""
        if self.token_budget >= self.max_prompt_tokens:
            raise ValueError(
                f"CONTEXT_TOKEN_BUDGET ({self.token_budget}) must be smaller than "
                f"MAX_PROMPT_TOKENS ({self.max_prompt_tokens}) to leave room for "
                f"the system instruction, conversation history and question"
            )
        return self


class IngestionSettings(_Section):
    """Upload and ingestion settings.

    Attributes:
        max_upload_bytes: Largest file accepted.
        allowed_mime_types_raw: Comma-separated MIME allow-list, as configured.
        upload_dir: Where original uploads are kept. Retaining them is what
            makes re-chunking and an embedding-model change possible without
            asking for the files again.
    """

    max_upload_bytes: int = Field(default=52_428_800, ge=1, validation_alias="MAX_UPLOAD_BYTES")
    allowed_mime_types_raw: str = Field(
        default=(
            "application/pdf,"
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document,"
            "text/plain,text/markdown"
        ),
        validation_alias="ALLOWED_MIME_TYPES",
    )
    upload_dir: str = Field(default="./uploads", validation_alias="UPLOAD_DIR")

    @property
    def allowed_mime_types(self) -> tuple[str, ...]:
        """The MIME allow-list, parsed."""
        return _split_csv(self.allowed_mime_types_raw)


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
_SECTION_TYPES: dict[str, type[_Section]] = {
    "app": AppSettings,
    "credentials": CredentialsSettings,
    "llm": LLMSettings,
    "embedding": EmbeddingSettings,
    "chunking": ChunkingSettings,
    "vector_store": VectorStoreSettings,
    "retrieval": RetrievalSettings,
    "reranker": RerankerSettings,
    "memory": MemorySettings,
    "context": ContextSettings,
    "ingestion": IngestionSettings,
}


def _aliases_of(section: type[_Section]) -> Iterator[tuple[str, str]]:
    """Yield ``(environment variable, field name)`` pairs for a section."""
    for field_name, field_info in section.model_fields.items():
        alias = field_info.validation_alias
        if isinstance(alias, str):
            yield alias, field_name


class Settings(BaseModel):
    """The complete, validated configuration of a running process.

    Build it with :meth:`load` in production or :meth:`from_environment` in
    tests. Constructing it is the validation: an invalid combination raises
    immediately, which is the whole point of failing at startup rather than at
    first use.

    Attributes:
        app: Process-level settings.
        credentials: Provider API keys.
        llm: Answer generation settings.
        embedding: Embedding settings.
        chunking: Chunking settings.
        vector_store: Vector database settings.
        retrieval: Retrieval settings.
        reranker: Reranker settings.
        memory: Conversation memory settings.
        context: Prompt and context construction settings.
        ingestion: Upload and ingestion settings.
    """

    model_config = {"frozen": True}

    app: AppSettings
    credentials: CredentialsSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    reranker: RerankerSettings
    memory: MemorySettings
    context: ContextSettings
    ingestion: IngestionSettings

    # ----------------------------------------------------------------- build #
    @classmethod
    def load(cls) -> Self:
        """Build settings from the process environment and ``.env``.

        Returns:
            Validated settings.

        Raises:
            pydantic.ValidationError: If any value or combination is invalid.
                The composition root translates this into a
                ``ConfigurationError`` -- config itself raises no domain type,
                because it imports no other layer.
        """
        return cls.from_environment({}, use_env_file=True)

    @classmethod
    def from_environment(
        cls,
        env: Mapping[str, str],
        *,
        use_env_file: bool = True,
    ) -> Self:
        """Build settings, overriding the process environment with ``env``.

        Args:
            env: Explicit variable values, keyed by environment variable name.
            use_env_file: Whether to also read the ``.env`` file. Tests pass
                ``False`` so a developer's local file cannot change a result.

        Returns:
            Validated settings.

        Raises:
            pydantic.ValidationError: If any value or combination is invalid.
        """
        env_file = _ENV_FILE if use_env_file else None
        sections: dict[str, Any] = {}
        for attribute, section_type in _SECTION_TYPES.items():
            overrides = {
                field_name: env[alias]
                for alias, field_name in _aliases_of(section_type)
                if alias in env
            }
            # `_env_file` is a pydantic-settings init hook, not a declared field,
            # so it is invisible to the type checker.
            sections[attribute] = section_type(_env_file=env_file, **overrides)  # type: ignore[call-arg]
        return cls(**sections)

    @classmethod
    def environment_variable_names(cls) -> tuple[str, ...]:
        """Every environment variable this configuration reads.

        Useful for documentation, for startup diagnostics, and for tests that
        need to isolate themselves from a developer's shell.

        Returns:
            Variable names, sorted.
        """
        names = {alias for section in _SECTION_TYPES.values() for alias, _ in _aliases_of(section)}
        return tuple(sorted(names))

    # ------------------------------------------------------ cross-section rules #
    @model_validator(mode="after")
    def _reranking_narrows_what_retrieval_produced(self) -> Self:
        """Refuse a reranker asked for more chunks than retrieval can supply.

        Lives at the root rather than on ``RetrievalSettings`` because the
        reranker's own toggle is part of the judgement: with reranking off, the
        two values are independent.
        """
        if self.reranker.enable_rerank and self.retrieval.rerank_top_k > self.retrieval.top_k:
            raise ValueError(
                f"RERANK_TOP_K ({self.retrieval.rerank_top_k}) must not exceed "
                f"TOP_K ({self.retrieval.top_k})"
            )
        return self
