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

from tests.test_helpers.otel import reset_khora_telemetry


@pytest.fixture(autouse=True)
def _reset_otel_globals_teardown():
    """Reset OTel globals after each test (in addition to per-file setup)."""
    yield
    reset_khora_telemetry()
