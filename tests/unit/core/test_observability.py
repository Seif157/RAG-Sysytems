"""Behaviour of request context, structured logging and stage telemetry.

The requirement is per-stage latency measurement across two processes
(architecture spec section 16.2). That needs three things working together: a
correlation id that survives async boundaries and reaches the worker, structured
events rather than formatted strings, and one timing event per pipeline stage.

The prototype had none of them -- its log filename was fixed at import time, it
produced one file per process, and several of its calls logged an empty string.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest
import structlog

from rag.config import Settings
from rag.core.context import (
    RequestContext,
    bind_request_context,
    clear_request_context,
    current_request_context,
    new_correlation_id,
)
from rag.core.logging import add_request_context, configure_logging, get_logger
from rag.core.telemetry import stage

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_context():
    clear_request_context()
    yield
    clear_request_context()


class _FakeTimer:
    """A monotonic timer that advances only when told to."""

    def __init__(self, *readings: float) -> None:
        self._readings = list(readings)

    def __call__(self) -> float:
        return self._readings.pop(0) if len(self._readings) > 1 else self._readings[0]


class TestRequestContext:
    def test_nothing_is_bound_initially(self):
        assert current_request_context() == RequestContext()

    def test_binding_makes_values_readable(self):
        bind_request_context(correlation_id="corr-1", tenant_id="acme")

        context = current_request_context()

        assert context.correlation_id == "corr-1"
        assert context.tenant_id == "acme"

    def test_binding_one_field_leaves_the_others_intact(self):
        bind_request_context(correlation_id="corr-1")
        bind_request_context(tenant_id="acme")

        assert current_request_context().correlation_id == "corr-1"

    def test_clearing_removes_everything(self):
        bind_request_context(correlation_id="corr-1", tenant_id="acme")

        clear_request_context()

        assert current_request_context() == RequestContext()

    def test_generated_correlation_ids_are_unique(self):
        assert new_correlation_id() != new_correlation_id()

    def test_context_survives_an_await(self):
        # Correlation must cross async boundaries or a traced request falls
        # apart the first time it hits I/O.
        async def scenario() -> str | None:
            bind_request_context(correlation_id="corr-1")
            await asyncio.sleep(0)
            return current_request_context().correlation_id

        assert asyncio.run(scenario()) == "corr-1"

    def test_context_serialises_for_a_worker_job_payload(self):
        # An ingestion trace must join to the upload request that caused it.
        bind_request_context(correlation_id="corr-1", tenant_id="acme")

        assert current_request_context().as_payload() == {
            "correlation_id": "corr-1",
            "tenant_id": "acme",
        }

    def test_an_empty_context_serialises_to_nothing(self):
        assert current_request_context().as_payload() == {}


class TestStructuredLogging:
    """Asserts on real rendered output.

    ``structlog.testing.capture_logs`` replaces the entire processor chain, so it
    can never observe the context processor -- which is precisely the behaviour
    worth testing. These tests render through the configured chain into a buffer
    instead.
    """

    @staticmethod
    def _emit(buffer: io.StringIO, **env: str) -> dict[str, object]:
        settings = Settings.from_environment({"LOG_FORMAT": "json", **env}, use_env_file=False)
        configure_logging(settings, stream=buffer)
        get_logger("test").info("document.ingested", chunks=12)
        return json.loads(buffer.getvalue().strip().splitlines()[-1])

    def test_events_carry_the_bound_correlation_id(self):
        bind_request_context(correlation_id="corr-1", tenant_id="acme")

        event = self._emit(io.StringIO())

        assert event["correlation_id"] == "corr-1"
        assert event["tenant_id"] == "acme"

    def test_events_carry_their_own_structured_fields(self):
        event = self._emit(io.StringIO())

        assert event["event"] == "document.ingested"
        assert event["chunks"] == 12

    def test_events_are_timestamped_and_levelled(self):
        event = self._emit(io.StringIO())

        assert event["level"] == "info"
        assert "timestamp" in event

    def test_unbound_context_does_not_add_empty_fields(self):
        # A null correlation id in every event is noise that hides real ones.
        event = self._emit(io.StringIO())

        assert "correlation_id" not in event

    def test_the_context_processor_merges_bound_values(self):
        bind_request_context(correlation_id="corr-1")

        merged = add_request_context(None, "info", {"event": "x"})

        assert merged == {"event": "x", "correlation_id": "corr-1"}

    def test_the_context_processor_adds_nothing_when_unbound(self):
        assert add_request_context(None, "info", {"event": "x"}) == {"event": "x"}

    def test_console_rendering_is_selected_outside_production(self):
        buffer = io.StringIO()
        settings = Settings.from_environment({"LOG_FORMAT": "console"}, use_env_file=False)

        configure_logging(settings, stream=buffer)
        get_logger("test").info("startup.complete")

        # Human-readable, therefore not JSON.
        with pytest.raises(json.JSONDecodeError):
            json.loads(buffer.getvalue().strip())

    def test_log_level_filters_quieter_events(self):
        buffer = io.StringIO()
        settings = Settings.from_environment(
            {"LOG_LEVEL": "WARNING", "LOG_FORMAT": "json"}, use_env_file=False
        )

        configure_logging(settings, stream=buffer)
        get_logger("test").debug("should.not.appear")

        assert buffer.getvalue() == ""


class TestStageTelemetry:
    def test_a_successful_stage_reports_its_duration_and_outcome(self):
        configure_logging(Settings.from_environment({}, use_env_file=False))
        timer = _FakeTimer(0.0, 0.25)

        with structlog.testing.capture_logs() as events, stage("retrieval.dense", timer=timer):
            pass

        assert events[0]["stage"] == "retrieval.dense"
        assert events[0]["duration_ms"] == pytest.approx(250.0)
        assert events[0]["outcome"] == "success"

    def test_stage_specific_fields_are_included(self):
        configure_logging(Settings.from_environment({}, use_env_file=False))

        with structlog.testing.capture_logs() as events, stage("chunking", strategy="recursive"):
            pass

        assert events[0]["strategy"] == "recursive"

    def test_fields_can_be_recorded_while_the_stage_runs(self):
        # A stage rarely knows its own counts until it has done the work.
        configure_logging(Settings.from_environment({}, use_env_file=False))

        with structlog.testing.capture_logs() as events, stage("chunking") as recorder:
            recorder.record(chunks=42)

        assert events[0]["chunks"] == 42

    def test_a_failing_stage_reports_the_error_and_re_raises(self):
        from rag.domain.errors import RetrievalError

        configure_logging(Settings.from_environment({}, use_env_file=False))

        with (
            structlog.testing.capture_logs() as events,
            pytest.raises(RetrievalError),
            stage("retrieval.dense"),
        ):
            raise RetrievalError("store unreachable")

        assert events[0]["outcome"] == "error"
        assert events[0]["error_code"] == "RETRIEVAL_ERROR"
        assert events[0]["retryable"] is False

    def test_a_non_domain_failure_is_still_reported(self):
        configure_logging(Settings.from_environment({}, use_env_file=False))

        with (
            structlog.testing.capture_logs() as events,
            pytest.raises(ZeroDivisionError),
            stage("chunking"),
        ):
            _ = 1 / 0

        assert events[0]["outcome"] == "error"
        assert events[0]["error_code"] == "ZeroDivisionError"

    def test_duration_is_reported_even_when_the_stage_fails(self):
        # Knowing a stage failed after 30 seconds versus 30 milliseconds is the
        # difference between a timeout and a validation error.
        configure_logging(Settings.from_environment({}, use_env_file=False))
        timer = _FakeTimer(0.0, 1.5)

        with (
            structlog.testing.capture_logs() as events,
            pytest.raises(ValueError),
            stage("embedding", timer=timer),
        ):
            raise ValueError("nope")

        assert events[0]["duration_ms"] == pytest.approx(1500.0)

    def test_the_stage_reports_its_own_elapsed_time_to_the_caller(self):
        configure_logging(Settings.from_environment({}, use_env_file=False))
        timer = _FakeTimer(0.0, 0.5)

        with stage("embedding", timer=timer) as recorder:
            pass

        assert recorder.duration_ms == pytest.approx(500.0)
