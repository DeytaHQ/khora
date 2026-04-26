"""Back-compat shim: the router moved to ``khora.query.router`` (Chronicle #6).

Existing imports from ``khora.engines.vectorcypher.router`` continue to work
unchanged. New code should import from ``khora.query.router`` directly.
"""

from __future__ import annotations

from khora.query.router import (
    QueryComplexity,
    QueryComplexityRouter,
    RouterConfig,
    RoutingDecision,
)

__all__ = [
    "QueryComplexity",
    "QueryComplexityRouter",
    "RouterConfig",
    "RoutingDecision",
]
