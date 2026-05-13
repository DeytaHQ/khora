"""Provider-agnostic helpers for telemetry attributes.

These helpers are independent of any specific OpenTelemetry SDK or
exporter implementation. They're safe to call from any code path,
regardless of whether telemetry is configured.
"""

from __future__ import annotations

import hashlib


def bounded_text_hash(text: str) -> str:
    """Return a short, bounded hash for use as a span attribute.

    Span/metric backends bill per distinct attribute value. Using raw
    user text as a span attribute causes unbounded cardinality. Hash to
    the first 8 hex chars — collisions are irrelevant for grouping
    observability data, but cardinality stays bounded by actual unique
    inputs rather than every keystroke variant.
    """
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
