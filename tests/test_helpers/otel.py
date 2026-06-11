"""Reset OpenTelemetry provider globals between tests.

Several tests install a real ``TracerProvider`` / ``MeterProvider`` to
exercise khora's OTel-first wiring. Those providers are process-wide
singletons under OTel's do-once semantics, so without a teardown they
leak into the next test — or into other test files later in the same
worker, where the unexpected ``record_exception`` machinery on span exit
can mask the real exception a downstream test is asserting on.

This module is the single home for that reset logic. It previously lived
verbatim in both ``tests/unit/telemetry/conftest.py`` and
``tests/integration/test_otel_smoke.py``; both now import from here.
"""

from __future__ import annotations


def reset_otel_globals() -> None:
    """Restore OTel's proxy ``TracerProvider`` / ``MeterProvider`` globals.

    Re-arms the do-once latches and drops any real provider so the next
    ``set_tracer_provider`` / ``set_meter_provider`` call takes effect.
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


def reset_khora_telemetry() -> None:
    """Full reset: OTel globals + khora's cached tracer/meter + bootstrap handle.

    Use this in a teardown so a test that called
    ``configure_telemetry(backend="otel")`` leaves the process with the
    proxy providers in place and khora's module-level caches rebound to
    them — otherwise the real provider and a stale ``bootstrap._handle``
    survive into later tests.
    """
    reset_otel_globals()

    from khora.telemetry import _otel as _otel_module
    from khora.telemetry import bootstrap

    _otel_module._TRACER = _otel_module._otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION)
    _otel_module._METER = _otel_module._otel_metrics.get_meter("khora", _otel_module._KHORA_VERSION)
    bootstrap._handle = None


__all__ = ["reset_otel_globals", "reset_khora_telemetry"]
