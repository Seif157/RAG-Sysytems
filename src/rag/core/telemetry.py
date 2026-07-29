"""Per-stage timing and outcome instrumentation (architecture spec section 16.2).

Every pipeline stage -- parse, chunk, embed, index, retrieve, rerank, generate --
runs inside :func:`stage`, which emits exactly one event carrying the stage
name, its duration, its outcome and any fields the stage chose to record.

Uniformity is the point. When every stage reports the same shape, "which stage
is slow?" is a query rather than an investigation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from rag.core.logging import get_logger

__all__ = ["StageRecorder", "stage"]

_logger = get_logger(__name__)


class StageRecorder:
    """Collects fields and elapsed time for one stage.

    Attributes:
        name: The stage's name, used as the ``stage`` field on the event.
    """

    def __init__(self, name: str, fields: dict[str, Any]) -> None:
        """Initialise a recorder for a named stage.

        Args:
            name: The stage's name.
            fields: Fields known before the stage begins.
        """
        self.name = name
        self._fields = dict(fields)
        self._duration_ms: float | None = None

    def record(self, **fields: Any) -> None:
        """Attach fields discovered while the stage runs.

        A stage rarely knows its own counts up front -- how many chunks it
        produced, how many cache hits it saw -- so they are recorded as they
        become known and emitted once at the end.

        Args:
            **fields: Values to attach to this stage's event.
        """
        self._fields.update(fields)

    @property
    def duration_ms(self) -> float | None:
        """Elapsed time in milliseconds, available once the stage has finished."""
        return self._duration_ms

    @property
    def fields(self) -> dict[str, Any]:
        """The fields recorded so far."""
        return dict(self._fields)


@contextmanager
def stage(
    name: str,
    *,
    timer: Callable[[], float] = time.perf_counter,
    **fields: Any,
) -> Iterator[StageRecorder]:
    """Time a pipeline stage and emit one structured event for it.

    On success the event is logged at info level with ``outcome="success"``. On
    failure it is logged at error level with ``outcome="error"`` and the
    exception's code, and the exception is re-raised unchanged -- this is
    instrumentation, not error handling. Deciding whether a stage failure is
    fatal or degradable belongs to the caller that knows whether the stage is
    optional.

    The duration is emitted either way. Knowing a stage failed after thirty
    seconds rather than thirty milliseconds is the difference between diagnosing
    a timeout and diagnosing a validation error.

    Args:
        name: Stage name, e.g. ``"retrieval.dense"``.
        timer: Monotonic clock, injectable so tests need not really wait.
        **fields: Fields known before the stage begins.

    Yields:
        A :class:`StageRecorder` for attaching fields as they become known.
    """
    recorder = StageRecorder(name, fields)
    started = timer()
    try:
        yield recorder
    except BaseException as exc:
        recorder._duration_ms = (timer() - started) * 1000.0
        _logger.error(
            "stage.completed",
            stage=name,
            duration_ms=recorder._duration_ms,
            outcome="error",
            error_code=getattr(exc, "code", type(exc).__name__),
            retryable=getattr(exc, "retryable", False),
            **recorder.fields,
        )
        raise
    else:
        recorder._duration_ms = (timer() - started) * 1000.0
        _logger.info(
            "stage.completed",
            stage=name,
            duration_ms=recorder._duration_ms,
            outcome="success",
            **recorder.fields,
        )
