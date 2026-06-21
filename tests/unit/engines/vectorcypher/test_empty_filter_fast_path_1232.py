"""#1232 — an empty filter must not disable the typed-entity-recent fast path.

The fast-path gate must test whether the filter CONSTRAINS anything, not
whether a filter object is present. ``filter={}`` / ``RecallFilter()`` parse to
a non-null AST with zero children (a constraint-free match-everything ``AND``),
so they have nothing to enforce and must keep the fast path. A genuinely
constraining filter still falls through to the full retrieve path (the fast
Cypher cannot enforce caller filters).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.filter import RecallFilter, parse_to_ast


def _typed_entity_recent_retriever() -> VectorCypherRetriever:
    """A retriever whose router always classifies TYPED_ENTITY_RECENT, with
    both fast and slow sub-paths stubbed so we can see which one fires."""
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig()
    retriever._storage = AsyncMock()
    retriever._embedder = AsyncMock()
    retriever._embedder.model_name = "mock"
    retriever._embedder.dimension = 8
    retriever._embedder.cache_stats = {"hits": 0}
    retriever._embedder.embed = AsyncMock(return_value=[0.1] * 8)

    routing = RoutingDecision(
        complexity=QueryComplexity.TYPED_ENTITY_RECENT,
        use_graph=True,
        graph_depth=1,
        confidence=0.9,
        reasoning="typed-entity-recent",
    )
    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(return_value=routing)

    empty_result = VectorCypherResult(chunks=[], entities=[], relationships=[], routing_decision=routing, metadata={})
    retriever._typed_entity_recent_retrieve = AsyncMock(return_value=empty_result)
    retriever._vectorcypher_retrieve = AsyncMock(return_value=empty_result)
    retriever._simple_retrieve = AsyncMock(return_value=empty_result)
    return retriever


@pytest.mark.parametrize(
    "filter_ast",
    [
        None,
        parse_to_ast(RecallFilter()),
        parse_to_ast(RecallFilter.model_validate({})),
    ],
    ids=["none", "RecallFilter()", "filter={}"],
)
async def test_empty_filter_keeps_fast_path(filter_ast) -> None:
    """None / RecallFilter() / {} all take the typed-entity-recent fast path."""
    retriever = _typed_entity_recent_retriever()
    await retriever.retrieve("show me recent decisions", uuid4(), filter_ast=filter_ast)
    retriever._typed_entity_recent_retrieve.assert_awaited_once()


async def test_constraining_filter_skips_fast_path() -> None:
    """A filter that actually constrains must NOT take the fast path (the fast
    Cypher cannot enforce caller filters)."""
    retriever = _typed_entity_recent_retriever()
    constraining = parse_to_ast(RecallFilter.model_validate({"source_name": "alpha"}))
    assert constraining.children  # control: this filter constrains
    await retriever.retrieve("show me recent decisions", uuid4(), filter_ast=constraining)
    retriever._typed_entity_recent_retrieve.assert_not_awaited()
    retriever._vectorcypher_retrieve.assert_awaited_once()
