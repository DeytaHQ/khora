"""VectorCypher engine - hybrid vector+graph retrieval.

The VectorCypher engine combines:
- Vector search (pgvector) for entry entity discovery
- Cypher traversal (Neo4j) for relationship expansion
- Smart query routing for optimal performance
- Skeleton-based indexing for cost efficiency
- Bi-temporal edges for temporal reasoning

Usage:
    from khora.engines.vectorcypher import VectorCypherEngine

    engine = VectorCypherEngine(config)
    await engine.connect()

    # Store with temporal context
    await engine.remember("Meeting notes...", namespace_id, occurred_at=datetime(...), entity_types=[...], relationship_types=[...])

    # Retrieve with hybrid search
    result = await engine.recall("What did we discuss?", namespace_id)
"""

from .dual_nodes import ChunkNode, DualNodeManager, EntityChunkLink
from .engine import VectorCypherConfig, VectorCypherEngine
from .fusion import (
    FusedResult,
    apply_recency_boost,
    normalize_scores,
    reciprocal_rank_fusion,
    weighted_rrf,
    weighted_rrf_normalized,
)
from .retriever import RetrieverConfig, VectorCypherResult, VectorCypherRetriever
from .router import QueryComplexity, QueryComplexityRouter, RouterConfig, RoutingDecision
from .temporal_detection import TemporalCategory, TemporalDetector, TemporalSignal

__all__ = [
    # Engine
    "VectorCypherConfig",
    "VectorCypherEngine",
    # Retriever
    "RetrieverConfig",
    "VectorCypherResult",
    "VectorCypherRetriever",
    # Router
    "QueryComplexity",
    "QueryComplexityRouter",
    "RouterConfig",
    "RoutingDecision",
    # Dual nodes
    "ChunkNode",
    "DualNodeManager",
    "EntityChunkLink",
    # Temporal detection
    "TemporalCategory",
    "TemporalDetector",
    "TemporalSignal",
    # Fusion
    "FusedResult",
    "apply_recency_boost",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "weighted_rrf",
    "weighted_rrf_normalized",
]
