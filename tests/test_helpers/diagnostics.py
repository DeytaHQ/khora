"""Assertions for the ADR-001 failure-observability convention.

Tests of "should this have degraded silently?" want a single helper
that pulls the conventional metadata keys (``degradations``,
``errors``, ``skipped``) off a result regardless of whether the result
uses a top-level ``metadata`` dict (``DreamResult``) or an
``engine_info`` dict (``RecallResult``).
"""

from __future__ import annotations

from typing import Any


def _diagnostics_bag(result: Any) -> dict[str, Any]:
    """Pull the diagnostics container off a result.

    Looks at ``result.metadata`` first, then ``result.engine_info``.
    Returns ``{}`` when neither is present so the helper stays usable
    on plain dicts or other lightweight result shapes.
    """
    bag = getattr(result, "metadata", None)
    if isinstance(bag, dict):
        return bag
    bag = getattr(result, "engine_info", None)
    if isinstance(bag, dict):
        return bag
    if isinstance(result, dict):
        return result
    return {}


def assert_no_silent_degradation(result: Any) -> None:
    """Fail if ``result`` carries any ADR-001 degradation / error entries.

    A pass means: no ``degradations`` and no ``errors`` were recorded.
    ``skipped`` is intentionally tolerated - skipping is a declared
    choice (e.g. an op kind the engine does not implement), not a
    silent failure.

    Use in tests that want to assert "this code path produced a result
    on the happy path", before tightening assertions about counts.
    """
    bag = _diagnostics_bag(result)
    degradations = bag.get("degradations", [])
    errors = bag.get("errors", [])

    problems: list[str] = []
    if degradations:
        problems.append(f"degradations={degradations}")
    if errors:
        problems.append(f"errors={errors}")
    if problems:
        raise AssertionError("silent degradation detected: " + "; ".join(problems))


__all__ = ["assert_no_silent_degradation"]
