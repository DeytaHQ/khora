"""Per-engine tuning profiles for query pipeline configuration.

Each engine type (graphrag, vectorcypher, skeleton, chronicle) has different
strengths and workloads. These profiles provide recommended ``QueryConfig``
overrides that tune the retrieval pipeline for each engine's sweet spot.

Usage::

    from khora.query.engine_profiles import get_engine_profile, apply_engine_profile

    # Get overrides dict
    overrides = get_engine_profile("graphrag")

    # Apply to an existing QueryConfig
    config = apply_engine_profile(config, "graphrag")

    # Create from QuerySettings with engine profile applied
    config = QueryConfig.from_settings(settings)
    config = apply_engine_profile(config, engine_name)
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

_PROFILES: dict[str, dict[str, Any]] = {
    "graphrag": {
        # Entities are first-class citizens — equalize vector and graph weight
        "vector_weight": 0.40,
        "graph_weight": 0.40,
        "keyword_weight": 0.20,
        # Wider recall for graph neighborhood expansion
        "stage1_recall_limit": 250,
        "stage1_vector_ratio": 0.40,
        "stage1_graph_ratio": 0.40,
        "stage1_keyword_ratio": 0.20,
        # CombMNZ rewards consensus across vector + graph sources
        "fusion_strategy": "combmnz",
        # Stronger entity linking for knowledge bases
        "entity_linking_fuzzy_threshold": 0.45,
        "entity_linking_max_candidates": 15,
        "linked_entity_boost": 2.0,
        # Less reranker trust — CombMNZ fusion is already strong signal
        "reranking_blend_weight": 0.65,
        # Equal weight for query similarity vs entity score in graph chunks
        "graph_chunk_query_sim_weight": 0.5,
    },
    "vectorcypher": {
        # Graph traversal is the star — highest graph weight
        "vector_weight": 0.35,
        "graph_weight": 0.50,
        "keyword_weight": 0.15,
        # Larger recall for multi-hop traversal
        "stage1_recall_limit": 300,
        "stage1_vector_ratio": 0.35,
        "stage1_graph_ratio": 0.50,
        "stage1_keyword_ratio": 0.15,
        # RRF — rank-based fusion avoids vector/graph score distribution mismatch
        "fusion_strategy": "rrf",
        # Entity linking critical for Cypher query generation
        "entity_linking_fuzzy_threshold": 0.40,
        "entity_linking_max_candidates": 15,
        "linked_entity_boost": 2.5,
        # Higher reranker trust — multi-hop results can be noisy
        "reranking_blend_weight": 0.75,
        # Favor entity/relationship score over raw query similarity
        "graph_chunk_query_sim_weight": 0.4,
        # Slight relevance bias for multi-hop answer chains
        "diversity_lambda": 0.6,
        # Less discount on HyDE — helps find better graph entry points
        "expanded_query_discount": 0.8,
    },
    "skeleton": {
        # No graph backend — vector + keyword only
        "vector_weight": 0.70,
        "graph_weight": 0.0,
        "keyword_weight": 0.30,
        # Small recall — chat history is shallow
        "stage1_recall_limit": 100,
        "stage1_vector_ratio": 0.70,
        "stage1_graph_ratio": 0.0,
        "stage1_keyword_ratio": 0.30,
        # Simple min_max normalization — only 2 sources
        "fusion_normalization": "min_max",
        # Disable LLM-dependent features for cost
        "enable_entity_linking": False,
        "linked_entity_boost": 1.0,
        "enable_reranking": False,
        "enable_hyde": "never",
        "enable_multi_stage": False,
        "enable_query_understanding": False,
        "adaptive_diversity": False,
        # Recency-first for chat workloads
        "apply_recency_bias": True,
        "recency_weight": 0.3,
        "recency_decay_days": 7.0,
        "temporal_hard_cutoff_days": 14.0,
        "temporal_half_life_hours": 12.0,
        # More diversity to avoid chat message redundancy
        "diversity_lambda": 0.4,
    },
    "chronicle": {
        # No graph backend — PostgreSQL + pgvector only
        "vector_weight": 0.40,
        "graph_weight": 0.0,
        "keyword_weight": 0.25,
        # BM25 parity with vector for exact-match temporal queries
        "stage1_recall_limit": 250,
        "stage1_vector_ratio": 0.50,
        "stage1_graph_ratio": 0.0,
        "stage1_keyword_ratio": 0.50,
        # Strong entity linking for person/event tracking
        "enable_entity_linking": True,
        "entity_linking_fuzzy_threshold": 0.40,
        "entity_linking_max_candidates": 15,
        "linked_entity_boost": 2.0,
        # Lower reranker trust — 4-channel RRF fusion is already strong
        "reranking_blend_weight": 0.60,
        # Temporal is king — Ebbinghaus-aligned half-life
        "apply_recency_bias": True,
        "recency_weight": 0.15,
        "recency_decay_days": 14.0,
        "temporal_hard_cutoff_days": 60.0,
        "temporal_half_life_hours": 168.0,
        # Slight relevance bias for benchmark factoid queries
        "diversity_lambda": 0.55,
        # Less discount on expanded queries
        "expanded_query_discount": 0.75,
    },
}


def list_engine_profiles() -> list[str]:
    """Return available engine profile names."""
    return list(_PROFILES)


def get_engine_profile(engine_name: str) -> dict[str, Any]:
    """Return the tuning overrides for an engine.

    Args:
        engine_name: Engine name (graphrag, vectorcypher, skeleton, chronicle).

    Returns:
        Dict of QueryConfig field overrides. Empty dict if no profile exists.
    """
    return dict(_PROFILES.get(engine_name, {}))


def apply_engine_profile(config: Any, engine_name: str) -> Any:
    """Apply engine-specific tuning overrides to a QueryConfig.

    Only overrides fields that exist on the config object. Returns the
    same config instance (mutated in place) for convenience.

    Args:
        config: A QueryConfig dataclass instance.
        engine_name: Engine name to apply profile for.

    Returns:
        The same config instance with profile values applied.
    """
    overrides = _PROFILES.get(engine_name, {})
    for field_name, value in overrides.items():
        if hasattr(config, field_name):
            setattr(config, field_name, value)
    return config
