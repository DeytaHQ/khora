"""Filter-pushdown spy — VectorCypher partial-failure path (path 9).

White-box spy proving the recall filter's behavior at the graph-channel
*partial-failure* boundary: when the Postgres (vector-channel) compile
succeeds but the Cypher (graph-channel) compile cannot express a predicate,
the resulting ``RecallFilterUnsupportedError`` must SURFACE — it must NOT be
swallowed by ``_vector_only_fallback``. Letting it fall back to a vector-only
search would silently drop the filter on the graph channel and leak
filter-violating chunks into the result, which is exactly the failure this
guards against.

Why this is a *differential* test (three arms):

  A "the fallback was not called" assertion is vacuously true if the spy
  could never observe the call in the first place. So this module asserts
  the arms against the SAME spied method:

  * Arm A (transient, positive control): ``compile_cypher`` raises
    ``ServiceUnavailable`` (a ``_NEO4J_TRANSIENT_ERRORS`` member) → the outer
    ``except`` in ``retrieve()`` catches it and ``_vector_only_fallback`` IS
    called. This proves the spy can actually observe the fallback firing, so
    Arm B's ``count == 0`` is a real signal, not a dead assertion.

  * Arm B (filter-unsupported): ``compile_cypher`` raises
    ``RecallFilterUnsupportedError`` (a ``KhoraError``, NOT a member of the
    retriever's ``_NEO4J_TRANSIENT_ERRORS`` tuple) → the error PROPAGATES out
    of ``retrieve()`` and ``_vector_only_fallback`` is NEVER called. This is
    the filter-pushdown contract working correctly: a filter the graph channel
    cannot honor fails loud rather than degrading to an unfiltered search.

  * Arm C (KNOWN GAP, documented not fixed): on the Arm-A transient path, the
    ``_vector_only_fallback`` re-run does NOT thread ``filter_ast`` — its
    signature has no such parameter and the ``_simple_retrieve`` it calls is
    not handed one — so the transient-graph-error fallback returns UNFILTERED
    results. This is a potential filter leak. The test below pins the CURRENT
    behavior (no filter reaches the fallback) so the gap is visible and a later
    fix has a failing-then-passing anchor; it is intentionally NOT fixed here
    (engine change is out of scope for this test ticket).

The spy point is the ``compile_cypher`` over-fetch probe at the top of
``_vectorcypher_retrieve`` (it is the first filter-touching code on that path
and is gated only on ``filter_ast is not None``), reached from ``retrieve()``
via ``mode=GRAPH`` (forces the VectorCypher branch deterministically, no
routing heuristics). Storage/embedder/router are mocked, so this runs with no
Postgres or Neo4j — it is pure control-flow over the retriever, not a
data-path test, and carries no service skip.

This module asserts ONLY filter error-propagation / fallback control-flow —
never result sets or ranking (that is a separate pillar).
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from neo4j.exceptions import ServiceUnavailable

from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.filter import FilterNode, RecallFilter, RecallFilterUnsupportedError, parse_to_ast
from khora.query import SearchMode

pytestmark = pytest.mark.filter_enforcement


def _make_retriever() -> VectorCypherRetriever:
    """A retriever with mocked I/O — no Postgres / Neo4j, no real driver."""
    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=AsyncMock(),
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=AsyncMock(),
    )


def _graph_routing() -> RoutingDecision:
    """A routing decision that sends ``retrieve()`` down the VectorCypher branch.

    ``mode=GRAPH`` already forces ``force_graph`` in ``retrieve()``; this only
    supplies the fields the branch reads (entry limit / depth) so the over-fetch
    probe is reached deterministically without depending on the real router's
    complexity heuristics.
    """
    return RoutingDecision(
        complexity=QueryComplexity.COMPLEX,
        use_graph=True,
        graph_depth=2,
        confidence=1.0,
        reasoning="forced graph branch for the partial-failure spy",
    )


def _wire(retriever: VectorCypherRetriever) -> list[object]:
    """Stub router + embedder; spy ``_vector_only_fallback`` via a call log.

    What stays REAL: the test calls the genuine ``retrieve()`` and the genuine
    ``_vectorcypher_retrieve``, so the actual outer ``try/except`` and the actual
    "fall back to vector-only?" DECISION execute unchanged. Only three leaves are
    stubbed, none of which is the partial-failure logic itself:

    * ``_router.route`` / ``_embedder.embed`` — so the call reaches the graph
      branch without a real model / router (``mode=GRAPH`` forces the branch;
      this just satisfies its inputs).
    * ``_vector_only_fallback`` — replaced with a passthrough spy. Its REAL body
      runs ``_simple_retrieve`` over the full storage stack, which is irrelevant
      to the contract under test (does the real dispatch CALL it, and with what
      filter?). The spy captures the EXACT args the real dispatch passes, so
      Arm C's "no filter_ast reached the fallback" reads the true boundary.

    Returns the live call-log list (one entry per real-dispatch invocation).
    """
    retriever._router.route = AsyncMock(return_value=_graph_routing())  # type: ignore[method-assign]
    retriever._embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])  # type: ignore[attr-defined]

    calls: list[object] = []

    async def _spied_fallback(*args: object, **kwargs: object) -> VectorCypherResult:
        calls.append((args, kwargs))
        return VectorCypherResult(chunks=[], entities=[], routing_decision=_graph_routing(), metadata={})

    retriever._vector_only_fallback = _spied_fallback  # type: ignore[method-assign]
    return calls


# A residual-metadata filter: ``metadata.*`` is unpushable to Cypher, so this is
# the shape that drives the graph channel's compile in real use. The value is
# irrelevant here — the compile is monkeypatched to raise — but using a
# realistic filter keeps the test honest about what reaches the boundary.
_FILTER = RecallFilter.model_validate({"metadata": {"channel": {"$eq": "eng"}}})


def _has_filter_node(call: tuple[tuple[object, ...], dict[str, object]]) -> bool:
    """Whether a captured ``_vector_only_fallback`` call carried a filter AST.

    Checks the ``filter_ast`` kwarg and every positional arg for a
    :class:`FilterNode`. ``_vector_only_fallback`` has no ``filter_ast``
    parameter today, so this is expected to be ``False`` — that is the gap
    Arm C documents.
    """
    args, kwargs = call
    if isinstance(kwargs.get("filter_ast"), FilterNode):
        return True
    return any(isinstance(a, FilterNode) for a in args)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transient_neo4j_error_does_trigger_vector_only_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arm A (positive control): a transient Neo4j error DOES hit the fallback.

    Proves the spy can observe ``_vector_only_fallback`` firing — so Arm B's
    ``count == 0`` is a meaningful signal, not a dead assertion. Raising the
    transient error from the same ``compile_cypher`` probe keeps the arms
    differing in exactly one variable: the exception type.
    """
    retriever = _make_retriever()
    fallback_calls = _wire(retriever)

    def _raise_transient(ast: object, ctx: object) -> object:
        raise ServiceUnavailable("neo4j unreachable")

    # Patch the SOURCE module, not the retriever module: ``compile_cypher`` is
    # imported FUNCTION-LOCALLY inside ``_vectorcypher_retrieve``, so it is not a
    # ``retriever`` module attribute. A ``setattr`` on the retriever module would
    # silently no-op and this test would pass vacuously. Do NOT "simplify" the
    # target back to ``khora.engines.vectorcypher.retriever.compile_cypher``.
    monkeypatch.setattr("khora.filter.compilers.cypher.compile_cypher", _raise_transient)

    # The transient error is caught by the outer ``except _NEO4J_TRANSIENT_ERRORS``
    # in ``retrieve()`` and routed to the fallback — so ``retrieve()`` returns.
    result = await retriever.retrieve(
        query="anything",
        namespace_id=uuid4(),
        mode=SearchMode.GRAPH,
        filter_ast=parse_to_ast(_FILTER),
    )

    # Proof we went through the REAL retrieve() dispatch (not a shortcut into the
    # fallback): the genuine routing + embedding steps ran before the branch.
    retriever._router.route.assert_awaited()  # type: ignore[attr-defined]
    retriever._embedder.embed.assert_awaited()  # type: ignore[attr-defined]
    assert len(fallback_calls) >= 1, "a transient Neo4j error MUST route to _vector_only_fallback"
    assert result is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_filter_unsupported_surfaces_and_skips_vector_only_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arm B: a graph-channel filter-unsupported error propagates; no fallback.

    ``RecallFilterUnsupportedError`` is not in ``_NEO4J_TRANSIENT_ERRORS``, so
    the outer ``except`` in ``retrieve()`` does not catch it and
    ``_vector_only_fallback`` is never reached — the filter fails loud instead
    of degrading to an unfiltered search.
    """
    retriever = _make_retriever()
    fallback_calls = _wire(retriever)

    def _raise_unsupported(ast: object, ctx: object) -> object:
        raise RecallFilterUnsupportedError("metadata.channel", "metadata predicates are not pushed down to Cypher")

    # Same source-module target as Arm A (see that test for the rationale).
    monkeypatch.setattr("khora.filter.compilers.cypher.compile_cypher", _raise_unsupported)

    with pytest.raises(RecallFilterUnsupportedError):
        await retriever.retrieve(
            query="anything",
            namespace_id=uuid4(),
            mode=SearchMode.GRAPH,
            filter_ast=parse_to_ast(_FILTER),
        )

    assert fallback_calls == [], "_vector_only_fallback must NOT be called on a filter-unsupported error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transient_fallback_drops_filter_KNOWN_GAP(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arm C: DOCUMENTS a known filter leak on the transient-error fallback path.

    KNOWN GAP: transient-graph-error vector-only fallback does not thread
    ``filter_ast`` → unfiltered results; tracked separately for follow-up.

    On Arm A's transient path, ``retrieve()`` routes to ``_vector_only_fallback``,
    which has NO ``filter_ast`` parameter and calls ``_simple_retrieve`` without
    one. So the caller's recall filter is silently dropped on this degradation
    path and the fallback returns unfiltered results — a potential filter leak.

    This test pins the CURRENT behavior (no filter AST reaches the fallback) so
    the gap is visible and a future fix has a failing-then-passing anchor. It is
    intentionally NOT fixed here: changing ``_vector_only_fallback`` to thread the
    filter is an engine change, out of scope for this test ticket. The audit gate
    allowlists ``_vector_only_fallback`` for the SAME reason.
    """
    retriever = _make_retriever()
    fallback_calls = _wire(retriever)

    def _raise_transient(ast: object, ctx: object) -> object:
        raise ServiceUnavailable("neo4j unreachable")

    # Same source-module target as Arm A (see that test for the rationale).
    monkeypatch.setattr("khora.filter.compilers.cypher.compile_cypher", _raise_transient)

    await retriever.retrieve(
        query="anything",
        namespace_id=uuid4(),
        mode=SearchMode.GRAPH,
        filter_ast=parse_to_ast(_FILTER),
    )

    assert len(fallback_calls) >= 1, "precondition: the transient error must reach the fallback"
    # The gap: NOT ONE captured fallback call carries the filter AST. When the
    # engine is fixed to thread filter_ast into _vector_only_fallback, this
    # assertion flips and this test should be updated to assert the filter IS
    # carried (canonical_hash equality) instead.
    assert not any(_has_filter_node(c) for c in fallback_calls), (
        "KNOWN GAP changed: _vector_only_fallback now receives a filter AST — "
        "the transient-fallback filter leak may be fixed; update this test to "
        "assert the filter is threaded (canonical_hash equality) and remove the "
        "audit-gate allowlist entry for _vector_only_fallback."
    )
