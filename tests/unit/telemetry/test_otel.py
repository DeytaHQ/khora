"""Tests for ``khora.telemetry._otel``.

Covers the trace_span context manager, the cached tracer/meter, the
bounded_text_hash helper, and the neo4j log bridge.
"""

from __future__ import annotations

import logging

import pytest
from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from khora.telemetry import _otel as _otel_module
from khora.telemetry import bounded_text_hash, trace_span
from khora.telemetry._otel import (
    _NEO4J_LOG_BRIDGE_MARK,
    get_meter,
    get_tracer,
    install_neo4j_log_bridge,
)


def _reset_otel_trace_globals() -> None:
    import opentelemetry.trace as _t

    _t._TRACER_PROVIDER_SET_ONCE = _t.Once()
    _t._TRACER_PROVIDER = None
    from opentelemetry.trace import ProxyTracerProvider

    _t._PROXY_TRACER_PROVIDER = ProxyTracerProvider()


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_otel_trace_globals()
    _otel_module._TRACER = _otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION)
    monkeypatch.delenv("KHORA_NEO4J_LOG_LEVEL", raising=False)


def test_bounded_text_hash_is_short_and_stable() -> None:
    h = bounded_text_hash("hello world")
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)
    assert bounded_text_hash("hello world") == h  # stable


def test_bounded_text_hash_differs_for_distinct_inputs() -> None:
    assert bounded_text_hash("a") != bounded_text_hash("b")


def test_trace_span_yields_non_recording_when_no_provider() -> None:
    with trace_span("khora.test") as span:
        # OTel proxy returns NonRecordingSpan; is_recording() == False.
        assert span.is_recording() is False
        # set_attribute is still callable; it's a no-op.
        span.set_attribute("foo", "bar")
        span.set_attributes({"x": 1, "y": 2})


def test_trace_span_yields_recording_when_provider_installed() -> None:
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    _otel_trace.set_tracer_provider(tp)
    # Rebind cached tracer to pick up the new provider.
    _otel_module._TRACER = _otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION)

    with trace_span("khora.test_recording", foo="bar", count=3) as span:
        assert span.is_recording() is True
        span.set_attribute("late", "added")

    exporter.shutdown()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "khora.test_recording"
    assert s.attributes["foo"] == "bar"
    assert s.attributes["count"] == 3
    assert s.attributes["late"] == "added"
    # khora's instrumentation scope is set to "khora".
    assert s.instrumentation_scope.name == "khora"


def test_get_tracer_returns_khora_scope() -> None:
    tracer = get_tracer()
    # OTel's Tracer doesn't expose scope directly, but the cached object
    # is the result of get_tracer("khora", version) so any span it
    # creates carries that scope (covered by the recording test above).
    assert tracer is not None
    # The meter likewise.
    assert get_meter() is not None


def test_install_neo4j_log_bridge_noop_when_env_unset() -> None:
    assert install_neo4j_log_bridge() is False
    marked = [h for h in logging.getLogger("neo4j").handlers if getattr(h, _NEO4J_LOG_BRIDGE_MARK, False)]
    assert marked == []


def test_install_neo4j_log_bridge_noop_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "")
    assert install_neo4j_log_bridge() is False


def test_install_neo4j_log_bridge_attaches_marked_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: with env set, bridge attaches exactly one marked handler."""
    monkeypatch.setenv("KHORA_NEO4J_LOG_LEVEL", "DEBUG")
    neo4j_logger = logging.getLogger("neo4j")
    # Clean slate.
    for h in list(neo4j_logger.handlers):
        if getattr(h, _NEO4J_LOG_BRIDGE_MARK, False):
            neo4j_logger.removeHandler(h)

    assert install_neo4j_log_bridge() is True
    marked = [h for h in neo4j_logger.handlers if getattr(h, _NEO4J_LOG_BRIDGE_MARK, False)]
    assert len(marked) == 1

    # Idempotent — second call removes the stale and reattaches one.
    install_neo4j_log_bridge()
    marked = [h for h in neo4j_logger.handlers if getattr(h, _NEO4J_LOG_BRIDGE_MARK, False)]
    assert len(marked) == 1

    # Cleanup.
    for h in list(marked):
        neo4j_logger.removeHandler(h)
