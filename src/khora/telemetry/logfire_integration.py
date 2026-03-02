"""Optional Logfire integration for Khora telemetry.

When ``logfire`` is installed, ``logfire_span`` emits real OpenTelemetry spans
via the Logfire SDK.  Otherwise it yields ``None`` (zero-cost no-op).

Install with::

    pip install khora[logfire]
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

try:
    import logfire as _logfire

    _HAS_LOGFIRE = True
except ImportError:
    _logfire = None
    _HAS_LOGFIRE = False


@contextmanager
def logfire_span(name: str, /, **attributes: Any):
    """Emit a Logfire span if available, otherwise no-op."""
    if _HAS_LOGFIRE:
        with _logfire.span(name, **attributes) as span:
            yield span
    else:
        yield None
