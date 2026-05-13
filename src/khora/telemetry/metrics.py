"""OTel metric instrument helpers.

Thin wrappers over ``opentelemetry.metrics.Meter.create_*`` that use
khora's instrumentation scope (``khora`` + package version). When no
real ``MeterProvider`` is installed, the OTel API returns no-op
instruments — these helpers stay safe to call from cold paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import _otel as _otel_module


def metric_counter(name: str, *, unit: str = "", description: str = "") -> Any:
    return _otel_module._METER.create_counter(name, unit=unit, description=description)


def metric_histogram(name: str, *, unit: str = "", description: str = "") -> Any:
    return _otel_module._METER.create_histogram(name, unit=unit, description=description)


def metric_gauge_callback(
    name: str,
    callbacks: Sequence[Any],
    *,
    unit: str = "",
    description: str = "",
) -> None:
    _otel_module._METER.create_observable_gauge(
        name,
        callbacks=callbacks,
        unit=unit,
        description=description,
    )
