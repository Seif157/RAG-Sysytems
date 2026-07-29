"""Start-up: load configuration, validate it, and build the object graph.

Fail fast, and fail with an explanation. A misconfigured deployment should stop
here rather than serve a user's first question with a missing credential or a
collection built by a different embedding model.

Checks grow with the system. Phase 1 validates what exists -- credentials for
the selected providers, and log format -- and later phases add connectivity and
collection-compatibility checks as their adapters arrive.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from rag.config import EmbeddingProvider, LLMProvider, LogFormat, Settings
from rag.core.container import Container
from rag.core.logging import configure_logging, get_logger
from rag.domain.errors import ConfigurationError

__all__ = ["bootstrap", "load_settings", "run_startup_checks"]

_logger = get_logger(__name__)

#: Which credential each answer-generation provider needs. A local model needs
#: none, which is what makes it the convenient choice for CI.
_LLM_CREDENTIALS: dict[LLMProvider, str] = {
    LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
    LLMProvider.GEMINI: "GOOGLE_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
}

#: Which credential each embedding provider needs.
_EMBEDDING_CREDENTIALS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.OPENAI: "OPENAI_API_KEY",
    EmbeddingProvider.GEMINI: "GOOGLE_API_KEY",
}


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    use_env_file: bool = True,
) -> Settings:
    """Load and validate configuration.

    :mod:`rag.config` raises pydantic's :class:`~pydantic.ValidationError`
    because it imports no other layer. Translating it into a domain error is the
    composition root's job, and it happens here so that every caller upstream --
    the API, the worker, the CLI -- can catch one thing.

    Args:
        env: Explicit variable values, overriding the process environment.
        use_env_file: Whether to also read ``.env``.

    Returns:
        Validated settings.

    Raises:
        ConfigurationError: If any value or combination is invalid. The original
            validation error is preserved as ``__cause__``.
    """
    try:
        return Settings.from_environment(env or {}, use_env_file=use_env_file)
    except ValidationError as exc:
        messages = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'settings'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"invalid configuration: {messages}") from exc


def run_startup_checks(settings: Settings) -> tuple[str, ...]:
    """Verify the deployment can actually do its job.

    Args:
        settings: The loaded configuration.

    Returns:
        The names of the checks that passed, for logging.

    Raises:
        ConfigurationError: On the first failed check, naming the environment
            variable that needs attention.
    """
    _check_credentials(settings)
    _check_logging(settings)
    return ("credentials", "logging")


def _check_credentials(settings: Settings) -> None:
    """Verify a credential exists for each selected provider."""
    required: list[tuple[str, str]] = []

    if (llm_variable := _LLM_CREDENTIALS.get(settings.llm.provider)) is not None:
        required.append((llm_variable, f"LLM_PROVIDER={settings.llm.provider.value}"))

    embedding_variable = _EMBEDDING_CREDENTIALS.get(settings.embedding.provider)
    if embedding_variable is not None:
        required.append(
            (embedding_variable, f"EMBEDDING_PROVIDER={settings.embedding.provider.value}")
        )

    for variable, because in required:
        secret = getattr(settings.credentials, variable.lower(), None)
        if secret is None or not secret.get_secret_value().strip():
            raise ConfigurationError(
                f"{variable} is required because {because}",
                context={"variable": variable, "reason": because},
            )


def _check_logging(settings: Settings) -> None:
    """Verify production emits logs that can actually be queried."""
    if settings.app.is_production and settings.app.log_format is not LogFormat.JSON:
        raise ConfigurationError(
            "LOG_FORMAT must be 'json' when APP_ENV=production; console output "
            "cannot be queried by log aggregation",
            context={"variable": "LOG_FORMAT", "value": settings.app.log_format.value},
        )


def bootstrap(
    env: Mapping[str, str] | None = None,
    *,
    use_env_file: bool = True,
) -> Container:
    """Perform full application start-up.

    Loads configuration, configures logging, runs the start-up checks and builds
    the container. Every entry point -- the API, the worker, any CLI -- calls
    this and nothing else, so they cannot drift apart in how they start.

    Args:
        env: Explicit variable values, overriding the process environment.
        use_env_file: Whether to also read ``.env``.

    Returns:
        A container ready to serve.

    Raises:
        ConfigurationError: If configuration is invalid or a check fails.
    """
    settings = load_settings(env, use_env_file=use_env_file)
    configure_logging(settings)

    checks = run_startup_checks(settings)
    _logger.info(
        "startup.checks_passed",
        checks=list(checks),
        app_env=settings.app.env.value,
        llm_provider=settings.llm.provider.value,
        embedding_provider=settings.embedding.provider.value,
        embedding_model=settings.embedding.model,
        collection=settings.vector_store.collection_name,
    )
    return Container(settings)
