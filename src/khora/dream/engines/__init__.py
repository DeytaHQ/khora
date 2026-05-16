"""Engine-specific dream operations.

Each engine (vectorcypher, chronicle, ...) ships its own subpackage of
read-only / mutation dream ops. Subpackages are imported on demand by
the orchestrator (#661); nothing here is part of ``khora.__all__``.

Stability: **internal** — the layout under this package may evolve
through Phase 1 without a major-version bump.
"""

from __future__ import annotations
