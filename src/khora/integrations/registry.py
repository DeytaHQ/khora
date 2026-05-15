"""Adapter registry — entry-point discovery + explicit registration.

Two paths get an adapter into the registry:

1. **Entry points** (the production path): a distribution publishes
   ``[project.entry-points."khora.integrations"]`` in its ``pyproject.toml``.
   :func:`discover` lazily walks the group on first call and caches the
   result. ``pip install khora-some-framework`` works with no edits to
   khora.

2. **Explicit registration** (tests + notebooks): :func:`register` adds
   or overrides a name. :func:`clear` resets the cache for test
   isolation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

# Module-level state. The lock guards both _explicit and the discovery
# cache so concurrent first-call discoveries don't double-walk
# entry_points or race on the cache.
_lock = threading.Lock()
_explicit: dict[str, Any] = {}
_discovered: dict[str, Any] | None = None  # None == not yet walked


def discover() -> dict[str, Any]:
    """Return the registered adapter factories, keyed by name.

    Lazy: walks ``importlib.metadata.entry_points(group="khora.integrations")``
    on first call, caches the result. Subsequent calls are O(1).

    Explicit registrations (:func:`register`) take precedence over
    entry-point registrations with the same name — that's intentional, it
    is how tests override a real adapter with a fake.

    Returns:
        A fresh dict mapping name to factory. Modifying the returned
        dict does NOT mutate the registry (it's a copy).
    """
    global _discovered
    with _lock:
        if _discovered is None:
            _discovered = {}
            for ep in entry_points(group="khora.integrations"):
                # Load lazily-ish: importlib.metadata returns EntryPoint
                # objects; we keep them un-loaded so importing khora
                # doesn't import every framework adapter. Callers that
                # want the actual class call .load() themselves, or use
                # register() to put a pre-loaded factory in.
                _discovered[ep.name] = ep
        merged = dict(_discovered)
    # Apply explicit registrations on top (outside the lock — _explicit
    # is only mutated under the lock too, but reading a dict snapshot is
    # cheap and we already have our discovered copy).
    merged.update(_explicit)
    return merged


def register(name: str, factory: Callable[..., Any]) -> None:
    """Register an adapter factory under ``name``.

    Overrides any entry-point registration of the same name. Useful in
    tests (inject a fake) and notebooks (don't ship a distribution).

    Args:
        name: Adapter name (e.g. ``"crewai"``). Lowercase, no dots.
        factory: A zero-arg callable that returns an adapter instance,
            or a class. Anything :func:`callable` works — the consumer
            decides how to invoke it.
    """
    if not name:
        raise ValueError("Adapter name must be non-empty")
    with _lock:
        _explicit[name] = factory


def clear() -> None:
    """Clear all explicit registrations and the discovery cache.

    Test-only escape hatch. Production code should never need this.
    Call it in test ``tearDown`` / fixture finalisation to keep tests
    isolated from one another.
    """
    global _discovered
    with _lock:
        _explicit.clear()
        _discovered = None
