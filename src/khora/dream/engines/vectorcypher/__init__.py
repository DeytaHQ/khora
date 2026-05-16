"""Vectorcypher-engine dream operations.

Phase 1 / 1.3 — read-only schema-drift report (#655). More vectorcypher
dream ops (cross-batch entity resolution, centroid recompute, etc.)
land in Phase 2.
"""

from khora.dream.engines.vectorcypher.schema_drift import (
    plan_vectorcypher_schema_drift,
)

__all__ = ["plan_vectorcypher_schema_drift"]
