"""Reset OTel globals between every test in tests/unit/telemetry/.

Several tests in this subdir install real ``TracerProvider`` /
``MeterProvider`` instances to exercise the OTel-first wiring. Those
providers are process-wide singletons under OTel's do-once semantics —
without a teardown they leak into the next test (or worse, into other
test files that come later in the run, where the unexpected
``record_exception`` machinery can mask the real exception being
tested).

This conftest resets the OTel globals AFTER each test, ensuring every
test starts and ends with the proxy providers in place.
"""

from __future__ import annotations

import pytest


def _reset_otel_globals() -> None:
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
def _reset_otel_globals_teardown():
    """Reset OTel globals after each test (in addition to per-file setup)."""
    yield
    _reset_otel_globals()
    from khora.telemetry import _otel as _otel_module
    from khora.telemetry import bootstrap

    _otel_module._TRACER = _otel_module._otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION)
    _otel_module._METER = _otel_module._otel_metrics.get_meter("khora", _otel_module._KHORA_VERSION)
    bootstrap._handle = None
