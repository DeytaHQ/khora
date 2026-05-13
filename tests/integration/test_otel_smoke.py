"""End-to-end smoke: spans actually leave the process via OTLP/HTTP.

Spins a tiny ``http.server`` on a free port that swallows OTLP POSTs
and records them. Sets ``OTEL_EXPORTER_OTLP_ENDPOINT`` to that port,
runs ``configure_telemetry()``, emits a few public spans, and asserts
the server received at least one OTLP traces request.

This is the regression net for "the wiring ships spans through the
real HTTP exporter" — distinct from the in-memory parity test
(``tests/unit/telemetry/test_otel_parity.py``), which doesn't exercise
the actual exporter path.

Marked ``integration`` so it doesn't run in the unit-test default cut;
the SRE deliverable in CLAUDE.md calls for this in the release smoke.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from khora.telemetry import bootstrap, trace_span

pytestmark = pytest.mark.integration


class _CountingHandler(BaseHTTPRequestHandler):
    """OTLP/HTTP server that records request paths and returns 200."""

    requests: list[str] = []  # class attr — shared across handler instances

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler convention)
        length = int(self.headers.get("Content-Length", "0"))
        _body = self.rfile.read(length)
        type(self).requests.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args, **kwargs) -> None:  # noqa: ARG002
        # Silence stderr noise during tests.
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def test_spans_reach_otlp_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: trace_span() with OTLP/HTTP ships to the endpoint."""
    # Reset state.
    monkeypatch.setattr(bootstrap, "_handle", None)
    _reset_otel_globals()
    _CountingHandler.requests = []

    # Stand up a counting OTLP server on a free port.
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _CountingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        # Force fast flush.
        monkeypatch.setenv("OTEL_BSP_SCHEDULE_DELAY", "100")
        monkeypatch.setenv("OTEL_BSP_EXPORT_TIMEOUT", "2000")

        handle = bootstrap.configure_telemetry(backend="otel")
        assert handle.backend == "otel"
        assert handle.khora_installed_tracer_provider is True

        # Rebind cached tracer — bootstrap does this for env-driven path,
        # but the test fixture's reset may need it again.
        from opentelemetry import trace as _t

        from khora.telemetry import _otel as _otel_module

        _otel_module._TRACER = _t.get_tracer("khora", _otel_module._KHORA_VERSION)

        # Emit a few public spans.
        for name in ("khora.recall", "khora.remember", "khora.forget"):
            with trace_span(name) as span:
                span.set_attribute("smoke_test", "true")

        # Flush + shutdown to push spans through the exporter.
        handle.shutdown()

        # The exporter's HTTP POSTs land on `/v1/traces`.
        traces_requests = [p for p in _CountingHandler.requests if "/v1/traces" in p]
        assert traces_requests, f"expected at least one POST to /v1/traces; saw {_CountingHandler.requests!r}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
