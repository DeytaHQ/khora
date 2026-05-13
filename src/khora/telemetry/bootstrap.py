"""Bootstrap and configure khora's OpenTelemetry export path.

khora's :mod:`khora.telemetry` module emits spans and metrics through
the OTel API unconditionally. The *export* of those signals — to a
collector, to logfire's SaaS, to an in-memory test exporter, or
nowhere — is governed by which ``TracerProvider`` / ``MeterProvider``
is installed globally.

Khora deliberately does **not** install a provider at import time.
Hosts that have already bootstrapped OTel get to keep their own
provider; hosts that haven't get a no-op tracer (the OTel API ships
its own ``NonRecordingSpan``, so the cost is near zero).

To opt in to span/metric export, call :func:`configure_telemetry` once
at process start. The function honors standard ``OTEL_*`` env vars and
the optional ``logfire`` extra.

Precedence (first match wins):

1. ``backend="none"`` — explicit opt-out, do nothing.
2. ``OTEL_SDK_DISABLED=true`` (env) — do nothing.
3. Caller-supplied ``tracer_provider`` / ``meter_provider`` — set them
   as globals (only if a global isn't already non-default).
4. A non-default global ``TracerProvider`` is already installed (host
   app, ``logfire.configure()``) — leave it alone; just bind khora's
   tracer through it.
5. ``backend="logfire"`` or (``backend="auto"`` and ``LOGFIRE_TOKEN``
   or ``LOGFIRE_SEND_TO_LOGFIRE`` env is set and ``logfire`` is
   importable) — call ``logfire.configure()``.
6. ``backend="otel"`` or (``backend="auto"`` and any ``OTEL_*`` env var
   is set) — bootstrap a vanilla OTel ``TracerProvider`` +
   ``MeterProvider`` with OTLP exporters. Honors
   ``OTEL_EXPORTER_OTLP_ENDPOINT``, ``_HEADERS``, ``_PROTOCOL``,
   ``OTEL_SERVICE_NAME``, ``OTEL_RESOURCE_ATTRIBUTES``,
   ``OTEL_TRACES_SAMPLER`` (transparently via SDK).
7. Otherwise — no-op; the proxy tracer stays in place.

Library contract: khora **never** sets ``service.name`` on a Resource.
Service identity belongs to the host application
(via ``OTEL_SERVICE_NAME`` or its own SDK init). Khora identifies
itself via the instrumentation scope (``scope.name = "khora"``,
``scope.version = importlib.metadata.version("khora")``).
"""

from __future__ import annotations

import atexit
import importlib
import importlib.util
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from opentelemetry import metrics as _otel_metrics
from opentelemetry import trace as _otel_trace

from . import _otel as _otel_module
from ._attrs import bounded_text_hash  # re-export for backwards-compat

__all__ = [
    "Backend",
    "TelemetryHandle",
    "configure_telemetry",
    "shutdown_telemetry_providers",
    "diagnostics",
]

_logger = logging.getLogger(__name__)

Backend = Literal["auto", "otel", "logfire", "none"]

# Contract version surfaced as a Resource attribute. Bump when
# docs/telemetry-contract.json's "version" changes. Independent from
# khora's package version so dashboards filtering by contract can
# remain stable across patch releases.
_CONTRACT_VERSION = "1.1"


@dataclass
class TelemetryHandle:
    """Result of :func:`configure_telemetry`.

    Carries enough state for the caller to introspect what khora did
    (or didn't do) and to drive an explicit shutdown.
    """

    backend: Literal["otel", "logfire", "none"]
    """Which path was taken. ``"none"`` means no provider was bootstrapped."""

    khora_installed_tracer_provider: bool = False
    """True iff khora called ``trace.set_tracer_provider()``."""

    khora_installed_meter_provider: bool = False
    """True iff khora called ``metrics.set_meter_provider()``."""

    endpoint: str | None = None
    """Resolved OTLP endpoint, if any (``OTEL_EXPORTER_OTLP_ENDPOINT`` or override)."""

    protocol: str | None = None
    """``"grpc"`` or ``"http/protobuf"`` when ``backend == "otel"``."""

    resource_attributes: dict[str, str] = field(default_factory=dict)
    """khora-contributed resource attrs (e.g. ``khora.telemetry.contract.version``)."""

    def shutdown(self) -> None:
        """Force-flush + shutdown only the provider(s) khora installed.

        Safe to call multiple times. Does nothing for providers owned
        by the host app.
        """
        if self.khora_installed_tracer_provider:
            provider = _otel_trace.get_tracer_provider()
            force_flush = getattr(provider, "force_flush", None)
            if callable(force_flush):
                force_flush()
            shutdown = getattr(provider, "shutdown", None)
            if callable(shutdown):
                shutdown()
        if self.khora_installed_meter_provider:
            provider = _otel_metrics.get_meter_provider()
            force_flush = getattr(provider, "force_flush", None)
            if callable(force_flush):
                force_flush()
            shutdown = getattr(provider, "shutdown", None)
            if callable(shutdown):
                shutdown()


# Module-level handle so configure_telemetry() is idempotent.
_handle: TelemetryHandle | None = None


def _tracer_provider_is_proxy() -> bool:
    """Return True iff no real ``TracerProvider`` has been installed.

    The OTel API ships ``ProxyTracerProvider`` as the default; once
    ``trace.set_tracer_provider()`` is called with a real provider,
    ``get_tracer_provider()`` returns that one instead.
    """
    try:
        from opentelemetry.trace import ProxyTracerProvider
    except ImportError:
        return False
    return isinstance(_otel_trace.get_tracer_provider(), ProxyTracerProvider)


def _meter_provider_is_proxy() -> bool:
    try:
        from opentelemetry.metrics._internal import _ProxyMeterProvider
    except ImportError:
        return False
    return isinstance(_otel_metrics.get_meter_provider(), _ProxyMeterProvider)


def _any_otel_env_set() -> bool:
    return any(k.startswith("OTEL_") and os.environ[k] for k in os.environ)


def _logfire_env_active() -> bool:
    return any(os.environ.get(k) for k in ("LOGFIRE_TOKEN", "LOGFIRE_SEND_TO_LOGFIRE"))


def _logfire_importable() -> bool:
    return importlib.util.find_spec("logfire") is not None


def configure_telemetry(
    *,
    backend: Backend = "auto",
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    protocol: Literal["grpc", "http/protobuf"] | None = None,
    sampler: object | None = None,
    resource_attributes: Mapping[str, str] | None = None,
    tracer_provider: object | None = None,
    meter_provider: object | None = None,
    install_neo4j_log_bridge: bool = True,
) -> TelemetryHandle:
    """Bootstrap khora's span/metric export.

    Idempotent — calling twice returns the same handle. See module
    docstring for the precedence order.

    Notes for library users:
        - khora never sets ``service.name``; set ``OTEL_SERVICE_NAME``
          or include ``service.name`` in your own Resource.
        - When the host app has already configured OTel
          (``trace.set_tracer_provider()`` called or ``logfire.configure()``
          ran), khora leaves the provider alone and just emits through it.
        - ``backend="none"`` is an explicit no-op for tests or hosts
          that want to suppress all khora-driven setup.

    Args:
        backend: ``"auto"`` (default), ``"otel"``, ``"logfire"``, or ``"none"``.
        endpoint: OTLP endpoint. Overrides ``OTEL_EXPORTER_OTLP_ENDPOINT``.
        headers: Additional OTLP headers. Merged onto ``OTEL_EXPORTER_OTLP_HEADERS``.
        protocol: ``"grpc"`` or ``"http/protobuf"`` (default: env or http/protobuf).
        sampler: An OTel ``Sampler`` instance. Bypasses ``OTEL_TRACES_SAMPLER`` env.
        resource_attributes: Extra resource attrs. Merged over
            ``OTEL_RESOURCE_ATTRIBUTES``. Cannot override ``service.*``.
        tracer_provider: Pre-built ``TracerProvider``. If passed and no
            non-default global is set, becomes the global.
        meter_provider: Pre-built ``MeterProvider``. Same semantics.
        install_neo4j_log_bridge: Forward neo4j stdlib log records through
            the active backend's log handler when ``KHORA_NEO4J_LOG_LEVEL``
            is set.
    """
    global _handle
    if _handle is not None:
        return _handle

    if backend == "none":
        _handle = TelemetryHandle(backend="none")
        return _handle

    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        _handle = TelemetryHandle(backend="none")
        return _handle

    # 3. Caller-supplied providers (only override if nothing real is set).
    installed_tp = False
    installed_mp = False
    if tracer_provider is not None and _tracer_provider_is_proxy():
        _otel_trace.set_tracer_provider(tracer_provider)
        installed_tp = True
    if meter_provider is not None and _meter_provider_is_proxy():
        _otel_metrics.set_meter_provider(meter_provider)
        installed_mp = True
    if installed_tp or installed_mp:
        _handle = TelemetryHandle(
            backend="otel",
            khora_installed_tracer_provider=installed_tp,
            khora_installed_meter_provider=installed_mp,
        )
        _maybe_install_neo4j_log_bridge(install_neo4j_log_bridge)
        _register_atexit(_handle)
        return _handle

    # 4. Host already configured OTel — defer to it.
    if not _tracer_provider_is_proxy():
        _logger.debug("khora.telemetry: a real TracerProvider is already installed; deferring to host")
        _handle = TelemetryHandle(backend="otel")
        _maybe_install_neo4j_log_bridge(install_neo4j_log_bridge)
        return _handle

    # 5. Logfire path.
    want_logfire = backend == "logfire" or (backend == "auto" and _logfire_env_active() and _logfire_importable())
    if want_logfire:
        if not _logfire_importable():
            raise RuntimeError(
                "khora.telemetry: backend='logfire' requested but `logfire` is not installed. "
                "Install it via `pip install khora[logfire]`."
            )
        import logfire  # type: ignore[import-not-found]

        logfire.configure()
        _handle = TelemetryHandle(backend="logfire")
        _maybe_install_neo4j_log_bridge(install_neo4j_log_bridge)
        return _handle

    # 6. Vanilla OTel SDK bootstrap.
    want_otel = backend == "otel" or (backend == "auto" and _any_otel_env_set())
    if want_otel:
        _handle = _bootstrap_otel(
            endpoint=endpoint,
            headers=headers,
            protocol=protocol,
            sampler=sampler,
            resource_attributes=resource_attributes,
        )
        _maybe_install_neo4j_log_bridge(install_neo4j_log_bridge)
        _register_atexit(_handle)
        return _handle

    # 7. No-op.
    _handle = TelemetryHandle(backend="none")
    return _handle


def _bootstrap_otel(
    *,
    endpoint: str | None,
    headers: Mapping[str, str] | None,
    protocol: Literal["grpc", "http/protobuf"] | None,
    sampler: object | None,
    resource_attributes: Mapping[str, str] | None,
) -> TelemetryHandle:
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError(
            "khora.telemetry: OTel SDK is not installed. Install via `pip install khora[otel]`."
        ) from exc

    resolved_protocol = protocol or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf"
    span_exporter, metric_exporter = _build_otlp_exporters(
        protocol=resolved_protocol,
        endpoint=endpoint,
        headers=headers,
    )

    extra_attrs: dict[str, str] = {"khora.telemetry.contract.version": _CONTRACT_VERSION}
    if resource_attributes:
        for k, v in resource_attributes.items():
            if k.startswith("service."):
                _logger.warning(
                    "khora.telemetry: dropping resource_attributes[%r]; khora does not set service.* "
                    "(use OTEL_SERVICE_NAME or your host's SDK)",
                    k,
                )
                continue
            extra_attrs[k] = v

    # Resource.create({}) reads OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES
    # from env. We layer our attrs on top of that.
    resource = Resource.create(extra_attrs)

    tp_kwargs: dict[str, Any] = {"resource": resource}
    if sampler is not None:
        tp_kwargs["sampler"] = sampler
    tp = TracerProvider(**tp_kwargs)
    tp.add_span_processor(BatchSpanProcessor(span_exporter))
    _otel_trace.set_tracer_provider(tp)

    reader = PeriodicExportingMetricReader(metric_exporter)
    mp = MeterProvider(resource=resource, metric_readers=[reader])
    _otel_metrics.set_meter_provider(mp)

    # Rebind khora's cached tracer/meter so subsequent calls land on
    # the new providers. ``get_tracer`` is cheap; rebinding is just to
    # avoid the proxy hop for the lifetime of this process.
    _otel_module._TRACER = _otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION)
    _otel_module._METER = _otel_metrics.get_meter("khora", _otel_module._KHORA_VERSION)

    return TelemetryHandle(
        backend="otel",
        khora_installed_tracer_provider=True,
        khora_installed_meter_provider=True,
        endpoint=endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        protocol=resolved_protocol,
        resource_attributes=extra_attrs,
    )


def _build_otlp_exporters(
    *,
    protocol: str,
    endpoint: str | None,
    headers: Mapping[str, str] | None,
) -> tuple[Any, Any]:
    if protocol.startswith("grpc"):
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as exc:
            raise RuntimeError(
                "khora.telemetry: gRPC OTLP exporter not installed. Install via `pip install khora[otel-grpc]`."
            ) from exc
    else:
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as exc:
            raise RuntimeError(
                "khora.telemetry: HTTP OTLP exporter not installed. Install via `pip install khora[otel]`."
            ) from exc

    span_kwargs: dict[str, Any] = {}
    metric_kwargs: dict[str, Any] = {}
    if endpoint:
        span_kwargs["endpoint"] = endpoint
        metric_kwargs["endpoint"] = endpoint
    if headers:
        span_kwargs["headers"] = dict(headers)
        metric_kwargs["headers"] = dict(headers)
    return OTLPSpanExporter(**span_kwargs), OTLPMetricExporter(**metric_kwargs)


def _maybe_install_neo4j_log_bridge(enabled: bool) -> None:
    if not enabled:
        return
    try:
        _otel_module.install_neo4j_log_bridge()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("khora.telemetry: neo4j log bridge install failed: %s", exc)


def _register_atexit(handle: TelemetryHandle) -> None:
    if handle.khora_installed_tracer_provider or handle.khora_installed_meter_provider:
        atexit.register(handle.shutdown)


def shutdown_telemetry_providers() -> None:
    """Force-flush + shutdown providers khora installed.

    No-op when no handle exists or the host app owns its provider.
    """
    if _handle is not None:
        _handle.shutdown()


def diagnostics() -> dict[str, Any]:
    """Snapshot of the active telemetry configuration.

    Useful for debugging "I see no spans" cases — prints which provider
    is installed, what khora bootstrapped (or didn't), what endpoint is
    in play. Safe to call before or after :func:`configure_telemetry`.
    """
    tp = _otel_trace.get_tracer_provider()
    mp = _otel_metrics.get_meter_provider()
    return {
        "khora_version": _otel_module._KHORA_VERSION,
        "contract_version": _CONTRACT_VERSION,
        "handle": (
            {
                "backend": _handle.backend,
                "khora_installed_tracer_provider": _handle.khora_installed_tracer_provider,
                "khora_installed_meter_provider": _handle.khora_installed_meter_provider,
                "endpoint": _handle.endpoint,
                "protocol": _handle.protocol,
                "resource_attributes": dict(_handle.resource_attributes),
            }
            if _handle is not None
            else None
        ),
        "tracer_provider_class": f"{type(tp).__module__}.{type(tp).__name__}",
        "tracer_provider_is_proxy": _tracer_provider_is_proxy(),
        "meter_provider_class": f"{type(mp).__module__}.{type(mp).__name__}",
        "meter_provider_is_proxy": _meter_provider_is_proxy(),
        "logfire_loaded": "logfire" in sys.modules,
        "otel_env": {k: v for k, v in os.environ.items() if k.startswith("OTEL_")},
    }


# Re-export to keep import surfaces tight for callers.
__all__ += ["bounded_text_hash"]
