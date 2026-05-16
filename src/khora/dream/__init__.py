"""Dream phase — periodic consolidation, dedupe, and pruning pass.

Phase 0.1 lands the typed surface only. The orchestrator that turns a
plan into observed changes ships in #661; the report sinks ship in #666.

Stability:

- **Public** (in ``__all__``): :class:`DreamConfig`, :class:`DreamMode`,
  :class:`DreamScope`, :class:`DreamResult`, :class:`DreamRunInfo`,
  :class:`OpKind`, :func:`acquire_namespace_dream_lock`,
  :class:`DreamLockUnavailable`.
- **Internal** (importable but may evolve without a major-version bump):
  :class:`DreamOp`, :class:`DreamPlan`, :class:`Checkpoint`,
  :class:`DreamProgress`, :class:`DreamDiff`, :class:`OpSummary`,
  :class:`DreamCapable`, :class:`DreamOrchestrator`.
"""

from __future__ import annotations

from khora.dream.config import DreamConfig, DreamOpsConfig
from khora.dream.locks import DreamLockUnavailable, acquire_namespace_dream_lock
from khora.dream.orchestrator import DreamOrchestrator
from khora.dream.plan import Checkpoint, DreamOp, DreamPlan, DreamScope, OpKind
from khora.dream.protocol import DreamCapable
from khora.dream.result import (
    DreamDiff,
    DreamMode,
    DreamProgress,
    DreamResult,
    DreamRunInfo,
    OpSummary,
)

__all__ = [
    "DreamConfig",
    "DreamLockUnavailable",
    "DreamMode",
    "DreamResult",
    "DreamRunInfo",
    "DreamScope",
    "OpKind",
    "acquire_namespace_dream_lock",
]

# Re-bind internal symbols at module level so static-analysis sees them as
# used; not in __all__ — see module docstring for the stability split.
_INTERNAL = (
    DreamOp,
    DreamPlan,
    Checkpoint,
    DreamDiff,
    DreamProgress,
    OpSummary,
    DreamOpsConfig,
    DreamCapable,
    DreamOrchestrator,
)
del _INTERNAL
