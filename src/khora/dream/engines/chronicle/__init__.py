"""Chronicle dream-phase op implementations.

Pure plan-builders producing a :class:`khora.dream.DreamOp`. Each op
records the *intent* of one consolidation action; the orchestrator
(#661) is what eventually decides to apply it.

Phase 1.1 (#652) lands the read-only abstention-drift report.
Phase 1.2 (#654) lands the read-only tombstone audit. Phase 2.4 (#664)
lands the dry-run-only fact-compaction planner that consumes the
audit's ``recommended_retention_days`` output; the matching apply
handler lands in v0.15 / #669 (:func:`apply_chronicle_fact_compaction`).

Stability: **internal** (Phase 1 / Phase 2 / Phase 4).
"""

from __future__ import annotations

from khora.dream.engines.chronicle.abstention_drift import (
    plan_chronicle_abstention_drift,
    record_abstention_sample,
    reset_abstention_samples,
)
from khora.dream.engines.chronicle.event_clustering import (
    apply_event_clustering,
    plan_chronicle_event_clustering,
)
from khora.dream.engines.chronicle.fact_compaction import (
    apply_chronicle_fact_compaction,
    plan_chronicle_fact_compaction,
)
from khora.dream.engines.chronicle.tombstone_audit import (
    plan_chronicle_tombstone_audit,
)

__all__ = [
    "apply_chronicle_fact_compaction",
    "apply_event_clustering",
    "plan_chronicle_abstention_drift",
    "plan_chronicle_event_clustering",
    "plan_chronicle_fact_compaction",
    "plan_chronicle_tombstone_audit",
    "record_abstention_sample",
    "reset_abstention_samples",
]
