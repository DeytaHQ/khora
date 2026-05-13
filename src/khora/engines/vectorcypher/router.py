"""Back-compat shim: the router moved to ``khora.query.router`` (Chronicle #6).

Existing imports from ``khora.engines.vectorcypher.router`` continue to work
unchanged. New code should import from ``khora.query.router`` directly.
"""

from __future__ import annotations

from khora.query.router import (
    TYPED_ENTITY_NOUN_MAP,
    TYPED_ENTITY_RECENCY_PATTERN,
    QueryComplexity,
    QueryComplexityRouter,
    RouterConfig,
    RoutingDecision,
    match_typed_entity_recent,
)

__all__ = [
    "TYPED_ENTITY_NOUN_MAP",
    "TYPED_ENTITY_RECENCY_PATTERN",
    "QueryComplexity",
    "QueryComplexityRouter",
    "RouterConfig",
    "RoutingDecision",
    "match_typed_entity_recent",
]
