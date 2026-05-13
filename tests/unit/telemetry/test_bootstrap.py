"""Tests for ``khora.telemetry.bootstrap``.

Covers precedence rules, idempotency, the ``service.*`` guard,
``diagnostics()`` shape, and the deprecation alias.
"""

from __future__ import annotations

import os
import warnings

import pytest
from opentelemetry import metrics as _otel_metrics
from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from khora.telemetry import bootstrap


def _reset_otel_globals() -> None:
    """Reset OTel's process-global Tracer/MeterProvider state.

    ``trace.set_tracer_provider()`` and ``metrics.set_meter_provider()``
    are do-once per process by design — they use ``_Once`` flags that
    drop subsequent calls with a warning. Tests need to clear those
    flags between cases to exercise the bootstrap precedence rules.
    """
    import opentelemetry.metrics._internal as _m
    import opentelemetry.trace as _t
    from opentelemetry.metrics._internal import _ProxyMeterProvider
    from opentelemetry.trace import ProxyTracerProvider

    _t._TRACER_PROVIDER_SET_ONCE = _t.Once()
    _t._TRACER_PROVIDER = None
    _t._PROXY_TRACER_PROVIDER = ProxyTracerProvider()

    _m._METER_PROVIDER_SET_ONCE = _m.Once()
    _m._METER_PROVIDER = None
    _m._PROXY_METER_PROVIDER = _ProxyMeterProvider()


@pytest.fixture(autouse=True)
def _reset_bootstrap_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts with a fresh handle + clean OTEL globals + env.

    bootstrap._handle and OTel's process-global providers are both
    process-wide singletons that need resetting between tests.
    """
    monkeypatch.setattr(bootstrap, "_handle", None)
    _reset_otel_globals()
    # Rebind khora's cached tracer/meter after the reset so they pick
    # up the proxy state for the new test.
    from khora.telemetry import _otel as _otel_module

    _otel_module._TRACER = _otel_module._otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION)
    _otel_module._METER = _otel_module._otel_metrics.get_meter("khora", _otel_module._KHORA_VERSION)
    for key in list(os.environ):
        if key.startswith(("OTEL_", "LOGFIRE_")):
            monkeypatch.delenv(key, raising=False)


def test_backend_none_is_explicit_noop() -> None:
    handle = bootstrap.configure_telemetry(backend="none")
    assert handle.backend == "none"
    assert handle.khora_installed_tracer_provider is False
    assert handle.khora_installed_meter_provider is False


def test_sdk_disabled_env_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    handle = bootstrap.configure_telemetry()
    assert handle.backend == "none"


def test_idempotent_returns_same_handle() -> None:
    a = bootstrap.configure_telemetry(backend="none")
    b = bootstrap.configure_telemetry(backend="otel")
    assert a is b
    assert b.backend == "none"  # first call's decision sticks


def test_caller_supplied_tracer_provider_wins_when_proxy() -> None:
    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    tp.add_span_processor(SimpleSpanProcessor(exporter))

    handle = bootstrap.configure_telemetry(tracer_provider=tp)
    assert handle.backend == "otel"
    assert handle.khora_installed_tracer_provider is True
    assert _otel_trace.get_tracer_provider() is tp


def test_existing_real_tracer_provider_is_respected() -> None:
    # Pre-install a real provider (simulating host app or logfire).
    pre = TracerProvider()
    pre.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    _otel_trace.set_tracer_provider(pre)

    handle = bootstrap.configure_telemetry(backend="auto")
    assert handle.backend == "otel"
    assert handle.khora_installed_tracer_provider is False
    # Provider must not have been replaced.
    assert _otel_trace.get_tracer_provider() is pre


def test_resource_attributes_drop_service_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Library contract: khora must never set ``service.*`` on the Resource."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://unreachable.invalid:1")
    handle = bootstrap.configure_telemetry(
        backend="otel",
        resource_attributes={"service.name": "evil", "team": "platform"},
    )
    assert handle.backend == "otel"
    assert "service.name" not in handle.resource_attributes
    assert handle.resource_attributes.get("team") == "platform"
    assert handle.resource_attributes.get("khora.telemetry.contract.version") == bootstrap._CONTRACT_VERSION
    handle.shutdown()


def test_diagnostics_reports_proxy_before_configure() -> None:
    d = bootstrap.diagnostics()
    assert d["tracer_provider_is_proxy"] is True
    assert d["handle"] is None
    assert "khora_version" in d
    assert "contract_version" in d
    assert d["otel_env"] == {}  # autouse fixture wiped OTEL_*


def test_diagnostics_reports_handle_after_configure() -> None:
    handle = bootstrap.configure_telemetry(backend="none")
    d = bootstrap.diagnostics()
    assert d["handle"] is not None
    assert d["handle"]["backend"] == handle.backend


def test_explicit_logfire_backend_without_logfire_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend logfire isn't installed even if it is in the dev env.
    import sys

    monkeypatch.setitem(sys.modules, "logfire", None)
    monkeypatch.setattr(bootstrap, "_logfire_importable", lambda: False)
    with pytest.raises(RuntimeError, match="logfire"):
        bootstrap.configure_telemetry(backend="logfire")


def test_meter_provider_installed_when_otel_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://unreachable.invalid:1")
    handle = bootstrap.configure_telemetry(backend="otel")
    assert handle.backend == "otel"
    assert handle.khora_installed_meter_provider is True
    handle.shutdown()


def test_caller_supplied_meter_provider_wins_when_proxy() -> None:
    reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[reader])
    handle = bootstrap.configure_telemetry(meter_provider=mp)
    assert handle.khora_installed_meter_provider is True
    assert _otel_metrics.get_meter_provider() is mp


def test_deprecation_alias_for_install_neo4j_logfire_handler() -> None:
    import khora.telemetry as telemetry

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fn = telemetry.install_neo4j_logfire_handler
    assert any(issubclass(w.category, DeprecationWarning) for w in recorded), (
        f"expected DeprecationWarning, got categories={[w.category for w in recorded]}"
    )
    assert fn is telemetry.install_neo4j_log_bridge


def test_no_attr_error_for_unknown_name() -> None:
    import khora.telemetry as telemetry

    with pytest.raises(AttributeError, match="no attribute 'definitely_not_real'"):
        _ = telemetry.definitely_not_real
