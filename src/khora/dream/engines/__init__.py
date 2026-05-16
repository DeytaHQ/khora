"""Engine-specific dream operations.

Each engine (vectorcypher, chronicle, ...) ships its own subpackage of
read-only / mutation dream ops. Subpackages are imported on demand by
the orchestrator (#661); nothing here is part of ``khora.__all__``.

Stability: **internal** — the dream-op surface evolves through the
Phase 1-3 rollout. See the umbrella ticket #649 for the stability split.
"""

from __future__ import annotations
