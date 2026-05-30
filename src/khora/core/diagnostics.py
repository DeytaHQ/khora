"""Failure-observability TypedDicts shared across engines.

See ``docs/architecture/failure-observability-contract.md`` for the
convention. Three optional/additive metadata keys let callers detect
silently-degraded results without changing call signatures:

- ``degradations`` - a fallback path was taken or a component returned
  partial data, but the operation still produced a result.
- ``errors`` - an exception was caught and swallowed; the operation
  still returned a result.
- ``skipped`` - the operation deliberately did not run for a declared
  reason (matches the pre-existing ``DreamResult.metadata['skip_reasons']``
  shape from #880).

All three are ``TypedDict``s so they round-trip cleanly through JSON
without requiring a dataclass. Callers SHOULD treat a missing key as
an empty list.
"""

from __future__ import annotations

from typing import TypedDict


class Degradation(TypedDict, total=False):
    """One channel / component took a fallback path or returned partial data.

    The operation still produced a result. Log at ``WARNING``; emit a
    ``khora.{engine}.{component}.degraded_total{reason}`` counter.
    """

    component: str
    reason: str
    detail: str | None
    exception: str | None


class ErrorRecord(TypedDict, total=False):
    """An exception was caught and swallowed; the operation still returned.

    Distinct from ``Degradation`` in intent: an ``ErrorRecord`` says
    "something we expected to work raised"; a ``Degradation`` says "we
    knowingly took a slower / less complete path". Log at ``ERROR``.
    """

    component: str
    reason: str
    exception: str
    detail: str | None


class SkipReason(TypedDict, total=False):
    """An op kind was deliberately not run for a declared reason.

    Shape matches ``DreamResult.metadata['skip_reasons']`` (#880) so
    the dream module's existing entries are valid ``SkipReason`` values
    without translation. Log at ``INFO``.
    """

    op_kind: str
    reason: str
    detail: str | None


__all__ = ["Degradation", "ErrorRecord", "SkipReason"]
