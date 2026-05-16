"""Dream backend capability Protocol.

Placeholder for Phase 0.1 — the real Protocol body (the methods a graph
or vector backend must implement to participate in dream-phase ops)
lands in #656. Kept here so downstream tickets can already import the
symbol without churning the import path later.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DreamCapable(Protocol):
    """Marker Protocol for backends that participate in dream-phase ops.

    Full method surface is defined in #656. In Phase 0.1, the only
    requirement is the marker attribute below so ``isinstance(x,
    DreamCapable)`` works at runtime against backends that have opted in.
    """

    dream_capable: bool
