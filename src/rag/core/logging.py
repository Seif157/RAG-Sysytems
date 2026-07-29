"""Structured logging configuration (ADR-016).

Log events are structured data, not formatted strings. The requirement is
per-stage latency across two processes, and no amount of grepping recovers
``duration_ms`` from a sentence. JSON everywhere real; a readable console
renderer locally, because developers read logs with their eyes.

Every event automatically carries whatever is bound in
:mod:`rag.core.context`, so no function has to accept and forward a logger.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any, TextIO

import structlog

from rag.config import LogFormat, Settings
from rag.core.context import current_request_context

__all__ = ["add_request_context", "configure_logging", "get_logger"]


def add_request_context(
    _logger: Any,
    _method: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Merge the ambient request context into an event.

    A processor rather than a call-site responsibility: correlation that depends
    on being remembered is correlation that goes missing exactly when it matters.

    Args:
        _logger: The wrapped logger. Unused.
        _method: The log method name. Unused.
        event_dict: The event being built.

    Returns:
        The event, with any bound correlation, tenant and request ids merged in.
    """
    event_dict.update(current_request_context().as_payload())
    return event_dict


def configure_logging(settings: Settings, stream: TextIO | None = None) -> None:
    """Configure structlog and the standard library logging module.

    Safe to call more than once; the last call wins. The API, the worker and the
    test suite each call it during their own start-up.

    Args:
        settings: Supplies the level and the renderer to use.
        stream: Where rendered events are written. Defaults to standard output;
            tests pass a buffer so they can assert on real rendered output
            rather than on an intercepted event dictionary.
    """
    level = getattr(logging, settings.app.log_level, logging.INFO)
    destination = stream if stream is not None else sys.stdout

    logging.basicConfig(
        format="%(message)s",
        stream=destination,
        level=level,
        force=True,
    )

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.app.log_format is LogFormat.JSON
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_request_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=destination),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to a component name.

    Args:
        name: Component identifier, conventionally the module's ``__name__``.

    Returns:
        A structlog logger. Events are structured: pass fields as keyword
        arguments rather than formatting them into the message.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
