"""Vectorcypher-engine dream operations.

Phase 1 audits — read-only ops returning a :class:`khora.dream.DreamOp`.
Mutation ops land in Phase 2 with ``mode="dry-run"`` enforced.
"""

from __future__ import annotations

from khora.dream.engines.vectorcypher.centroid_recompute import (
    plan_vectorcypher_centroid_recompute,
)
from khora.dream.engines.vectorcypher.community_summary import (
    plan_vectorcypher_community_summary,
)
from khora.dream.engines.vectorcypher.dedupe_entities import (
    plan_vectorcypher_dedupe_entities,
)
from khora.dream.engines.vectorcypher.orphan_report import (
    plan_vectorcypher_orphan_report,
)
from khora.dream.engines.vectorcypher.schema_drift import (
    plan_vectorcypher_schema_drift,
)
from khora.dream.engines.vectorcypher.source_chunk_ids_audit import (
    plan_vectorcypher_source_chunk_ids_audit,
)
from khora.dream.engines.vectorcypher.source_chunk_ids_gc import (
    plan_vectorcypher_source_chunk_ids_gc,
)

__all__ = [
    "plan_vectorcypher_centroid_recompute",
    "plan_vectorcypher_community_summary",
    "plan_vectorcypher_dedupe_entities",
    "plan_vectorcypher_orphan_report",
    "plan_vectorcypher_schema_drift",
    "plan_vectorcypher_source_chunk_ids_audit",
    "plan_vectorcypher_source_chunk_ids_gc",
]
