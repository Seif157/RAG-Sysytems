"""The composition root: the one place that knows every concrete class.

Everything above :mod:`rag.infrastructure` depends on interfaces. Here is where
interfaces become objects, and where optional behaviour is expressed by *which
object gets built* rather than by a flag inside a call site -- reranking off
wires a no-op reranker, so the query flow contains no feature checks (ADR-010).

Written by hand rather than with a DI framework. The object graph is legible as
ordinary code, "go to definition" reaches the real class, and wiring mistakes
are type errors rather than runtime surprises (ADR-011).
"""

from __future__ import annotations

from functools import cached_property

from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from rag.application.services import ContextBuilder
from rag.application.tools import SearchDocumentsTool, ToolExecutor
from rag.application.use_cases import (
    AgenticAnswerUseCase,
    AnswerQuestionUseCase,
    IngestDocumentUseCase,
)
from rag.config import (
    ChunkingStrategyName,
    EmbeddingProvider,
    LLMProvider,
    RerankerProvider,
    Settings,
)
from rag.core.logging import get_logger
from rag.core.mapping import collection_spec_from, generation_params_from, prompt_spec_from
from rag.domain.errors import ConfigurationError
from rag.domain.models import CollectionInfo, CollectionSpec, GenerationParams
from rag.domain.ports import ChunkingStrategy, Embedder, LLMClient, Reranker, Retriever, VectorStore
from rag.domain.prompts import PromptSpec
from rag.infrastructure.chunking import RecursiveChunker
from rag.infrastructure.documents import (
    DocumentLoader,
    DocxParser,
    MarkdownParser,
    ParserRegistry,
    PdfParser,
    TxtParser,
)
from rag.infrastructure.embeddings import LocalEmbedder, OpenAIEmbedder
from rag.infrastructure.llm import GeminiLLMClient, OpenRouterLLMClient
from rag.infrastructure.prompts import PromptBuilder
from rag.infrastructure.reranker import BgeReranker, NoOpReranker
from rag.infrastructure.retrieval import (
    Bm25SparseEncoder,
    DenseRetriever,
    HybridRetriever,
    SparseRetriever,
)
from rag.infrastructure.storage import DocumentCatalog, UploadStore
from rag.infrastructure.tokenization import TokenCounter
from rag.infrastructure.vector_store import QdrantVectorStore

__all__ = ["Container"]

_CATALOG_FILENAME = "catalog.json"

_logger = get_logger(__name__)


class Container:
    """Builds and holds the application's object graph.

    Dependencies are built once and cached. Two containers share nothing, which
    keeps tests independent of one another.

    Configuration enums contain only providers with complete adapters, so an
    unsupported choice is rejected while settings are parsed.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        """Build a container.

        Args:
            settings: Validated configuration driving every wiring decision.
            vector_store: Overrides the Qdrant store. Tests pass an in-memory
                double; this is the seam the whole test suite hangs from.
            embedder: Overrides the embedding provider.
            llm: Overrides the language model.
        """
        self._settings = settings
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm

    @property
    def settings(self) -> Settings:
        """The configuration this container was built from."""
        return self._settings

    # ------------------------------------------------- configuration-derived #
    def collection_spec(self) -> CollectionSpec:
        """The collection shape the configured embedding model requires."""
        return collection_spec_from(self._settings)

    def generation_params(self) -> GenerationParams:
        """Generation settings to pass to the LLM port."""
        return generation_params_from(self._settings)

    def prompt_spec(self) -> PromptSpec:
        """The contract every prompt must satisfy."""
        return prompt_spec_from(self._settings)

    # ------------------------------------------------------------- adapters #
    @cached_property
    def token_counter(self) -> TokenCounter:
        """Counts tokens for chunking and budgeting."""
        return TokenCounter(self._settings.embedding.model)

    @cached_property
    def embedder(self) -> Embedder:
        """The configured embedding provider."""
        if self._embedder is not None:
            return self._embedder

        provider = self._settings.embedding.provider

        # Local and HuggingFace are one adapter: both run a Sentence-Transformers
        # model on this machine, and the distinction is only where the weights
        # were published.
        if provider in (EmbeddingProvider.LOCAL, EmbeddingProvider.HUGGINGFACE):
            return LocalEmbedder(
                model=self._settings.embedding.model,
                dimension=self._settings.embedding.dimension,
                batch_size=self._settings.embedding.batch_size,
            )

        if provider is EmbeddingProvider.OPENAI:
            return OpenAIEmbedder(
                api_key=self._require_secret("OPENAI_API_KEY", "openai_api_key"),
                model=self._settings.embedding.model,
                dimension=self._settings.embedding.dimension,
                batch_size=self._settings.embedding.batch_size,
            )

        raise AssertionError(f"unhandled embedding provider: {provider!r}")

    @cached_property
    def llm(self) -> LLMClient:
        """The configured language model."""
        if self._llm is not None:
            return self._llm

        provider = self._settings.llm.provider

        if provider is LLMProvider.OPENROUTER:
            return OpenRouterLLMClient(
                api_key=self._require_secret("OPENROUTER_API_KEY", "openrouter_api_key"),
                base_url=self._settings.llm.openrouter_base_url,
                model=self._settings.llm.model,
            )

        if provider is LLMProvider.GEMINI:
            return GeminiLLMClient(
                api_key=self._require_secret("GOOGLE_API_KEY", "google_api_key"),
                model=self._settings.llm.model,
            )

        raise AssertionError(f"unhandled LLM provider: {provider!r}")

    @cached_property
    def vector_store(self) -> VectorStore:
        """The Qdrant store."""
        if self._vector_store is not None:
            return self._vector_store

        api_key = self._settings.vector_store.qdrant_api_key
        client = AsyncQdrantClient(
            url=self._settings.vector_store.qdrant_url,
            api_key=api_key.get_secret_value() if api_key else None,
            # The client's version check runs on a background thread and raises
            # there when the server is unreachable, which surfaces as an
            # unrelated failure somewhere else entirely. The compose file pins
            # the server version, so the check buys nothing.
            check_compatibility=False,
        )
        return QdrantVectorStore(client, self._settings.vector_store.collection_name)

    @cached_property
    def chunker(self) -> ChunkingStrategy:
        """The configured chunking strategy."""
        strategy = self._settings.chunking.strategy
        if strategy is not ChunkingStrategyName.RECURSIVE:
            raise AssertionError(f"unhandled chunking strategy: {strategy!r}")
        return RecursiveChunker(
            chunk_size=self._settings.chunking.chunk_size,
            chunk_overlap=self._settings.chunking.chunk_overlap,
            count_tokens=self.token_counter,
        )

    @cached_property
    def reranker(self) -> Reranker:
        """The configured reranker.

        Disabling reranking wires the no-op rather than setting a flag, so no
        call site needs to know reranking is optional.
        """
        if not self._settings.reranker.enable_rerank:
            return NoOpReranker()

        provider = self._settings.reranker.provider
        if provider is RerankerProvider.NOOP:
            return NoOpReranker()
        if provider is not RerankerProvider.BGE:
            raise ConfigurationError(
                f"RERANKER_PROVIDER={provider.value!r} has no adapter; available: bge, noop",
                context={"variable": "RERANKER_PROVIDER", "value": provider.value},
            )

        return BgeReranker(
            model=self._settings.reranker.model,
            batch_size=self._settings.reranker.batch_size,
        )

    @cached_property
    def sparse_encoder(self) -> Bm25SparseEncoder | None:
        """The lexical encoder, or ``None`` for a dense-only deployment.

        Absent rather than disabled: encoding vectors that nothing will query
        would cost ingestion time for no benefit.
        """
        if not self._settings.retrieval.enable_hybrid:
            return None
        return Bm25SparseEncoder()

    @cached_property
    def retriever(self) -> Retriever:
        """The configured retriever.

        Hybrid retrieval is a *composite* of retrievers satisfying the same
        interface, so choosing between it and dense-only happens here rather
        than as a branch in the query flow.
        """
        dense = DenseRetriever(embedder=self.embedder, store=self.vector_store)

        encoder = self.sparse_encoder
        if encoder is None:
            return dense

        return HybridRetriever(
            dense=dense,
            sparse=SparseRetriever(encoder=encoder, store=self.vector_store),
            sparse_weight=self._settings.retrieval.sparse_weight,
            rrf_k=self._settings.retrieval.rrf_k,
        )

    @cached_property
    def parsers(self) -> ParserRegistry:
        """The parsers for every supported format.

        The one place a new format is registered. Adding one is a new file plus
        this line.
        """
        return ParserRegistry((PdfParser(), DocxParser(), MarkdownParser(), TxtParser()))

    @cached_property
    def uploads(self) -> UploadStore:
        """Where original uploaded files are kept."""
        return UploadStore(self._settings.ingestion.upload_dir)

    @cached_property
    def catalog(self) -> DocumentCatalog:
        """The record of what has been ingested."""
        return DocumentCatalog(self.uploads.root / _CATALOG_FILENAME)

    # ---------------------------------------------------------- preparation #
    async def prepare_store(self) -> CollectionInfo:
        """Make the vector store ready to serve, or refuse to.

        Creates the collection and its payload indexes if absent, and verifies
        it if present. Must run before the first upload -- and *should* run at
        start-up rather than lazily, because the check it performs is the one
        that prevents the system's worst failure: a collection whose vectors
        came from a different embedding model answers plausibly and wrongly
        (ADR-015).

        Idempotent, so the Streamlit script re-running is harmless.

        Returns:
            The prepared collection's actual shape.

        Raises:
            CollectionMismatchError: If an existing collection is incompatible
                with the running configuration.
            VectorStoreUnavailableError: If the store cannot be reached.
        """
        await self.vector_store.ensure_collection(self.collection_spec())
        return await self.vector_store.collection_info()

    async def warm_up(self) -> None:
        """Load anything expensive before the first question rather than during it.

        The cross-encoder is hundreds of megabytes to gigabytes and takes tens of
        seconds to load from disk -- minutes on a first run, which downloads it.
        Left lazy, that cost lands on whoever asks the first question, and looks
        exactly like the system having hung.

        Best-effort: a reranker that cannot load should degrade the *answer*, not
        prevent start-up, and the query path already handles that.
        """
        for name, component in (("embedder", self.embedder), ("reranker", self.reranker)):
            warm = getattr(component, "warm_up", None)
            if warm is None:
                continue
            try:
                await warm()
            except Exception as exc:  # pragma: no cover - degradation is the point
                _logger.warning("warmup.failed", component=name, error=str(exc))

    # ------------------------------------------------------------ use cases #
    @cached_property
    def ingest_document(self) -> IngestDocumentUseCase:
        """The ingestion use case."""
        return IngestDocumentUseCase(
            loader=DocumentLoader(self._settings.ingestion.max_upload_bytes),
            parsers=self.parsers,
            chunker=self.chunker,
            embedder=self.embedder,
            store=self.vector_store,
            uploads=self.uploads,
            catalog=self.catalog,
            sparse_encoder=self.sparse_encoder,
        )

    @cached_property
    def answer_question(self) -> AnswerQuestionUseCase | AgenticAnswerUseCase:
        """The question-answering use case."""
        context_builder = ContextBuilder(
            budget_tokens=self._settings.context.token_budget,
            count_tokens=self.token_counter,
        )
        if self._settings.agent.enable_tool_calling:
            search = SearchDocumentsTool(
                self.retriever,
                self.reranker,
                context_builder,
                max_top_k=self._settings.agent.search_max_top_k,
                rerank_top_k=self._settings.retrieval.rerank_top_k,
            )
            return AgenticAnswerUseCase(
                llm=self.llm,
                executor=ToolExecutor(search, timeout_s=self._settings.agent.tool_timeout_s),
                context_builder=context_builder,
                generation_params=self.generation_params(),
                prompt_version=self._settings.context.prompt_version,
                max_rounds=self._settings.agent.max_tool_rounds,
                max_calls_per_round=self._settings.agent.max_calls_per_round,
                max_total_calls=self._settings.agent.max_total_calls,
                max_result_tokens=self._settings.agent.max_result_tokens,
            )
        return AnswerQuestionUseCase(
            retriever=self.retriever,
            reranker=self.reranker,
            context_builder=context_builder,
            prompt_builder=PromptBuilder(self.token_counter),
            llm=self.llm,
            generation_params=self.generation_params(),
            prompt_spec=self.prompt_spec(),
            top_k=self._settings.retrieval.top_k,
            rerank_top_k=self._settings.retrieval.rerank_top_k,
        )

    # ---------------------------------------------------------------- helpers #
    def _require_secret(self, variable: str, attribute: str) -> str:
        """Read a credential, failing with the variable's name if absent."""
        secret: SecretStr | None = getattr(self._settings.credentials, attribute, None)
        if secret is None or not secret.get_secret_value().strip():
            raise ConfigurationError(
                f"{variable} is required for the configured provider",
                context={"variable": variable},
            )
        return secret.get_secret_value()
