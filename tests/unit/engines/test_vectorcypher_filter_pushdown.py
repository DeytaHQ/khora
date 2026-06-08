"""VectorCypher pushes the recall-filter WHERE into BOTH chunk channels.

`compile_postgres` now lowers the public ``filter=`` document to a single
``khora_chunks`` WHERE predicate that VectorCypher pushes down into BOTH of its
chunk channels:

* the VECTOR channel (``_vector_search_chunks`` -> ``vector_store.search``), and
* the BM25 channel (``_bm25_search_chunks`` -> the temporal store's
  ``search_fulltext`` / the coordinator's ``search_fulltext_chunks``).

Both channels must apply the SAME WHERE so that no filter-violating chunk
reaches RRF fusion from either side.

Why mock/spy level (no live Postgres): this file runs in the main ``test`` job,
which provisions no database. It pins the engine->channel WIRING contract: that
the validated ``filter_ast`` (built exactly as the facade builds it —
``RecallFilter.model_validate`` then ``parse_to_ast``, see khora.py:2045-2046)
is threaded into both channel calls. The end-to-end row-set proof (that the
emitted SQL actually narrows live rows) is the job of the Postgres matrix /
filter-conformance suite — deliberately NOT duplicated here. We DO compile the
AST to a SQL string via ``compile_postgres`` and assert byte-equality between the
two channels, so "same WHERE" is proven, not merely "a WHERE on each side".

Scope guard: this exercises ONLY the two chunk channels touched by the primary
filtered-recall path. It is NOT the broader multi-channel pushdown-spy harness.

The representative filter under test (3 predicates, one per emission kind —
typed system column, a date range, and a JSONB array-membership):

    {"source_name": "linear",
     "occurred_at": {"$gte": "2026-04-05"},
     "metadata.tag": {"$in": ["urgent", "release"]}}
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from loguru import logger as loguru_logger
from sqlalchemy import ColumnElement
from sqlalchemy.dialects import postgresql

from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.filter import CompileContext, FilterNode, RecallFilter, parse_to_ast
from khora.filter.compilers.postgres import compile_postgres
from khora.query.temporal_detection import TemporalCategory, TemporalSignal
from khora.storage.backends.pgvector import PgVectorBackend

pytestmark = pytest.mark.unit


# Bridge loguru into pytest caplog. The relational backend's Layer-B refusal
# WARNING is emitted via loguru (``from loguru import logger``), but pytest's
# ``caplog`` only captures stdlib ``logging`` records by default — without this
# bridge a caplog assertion would silently pass-by-absence (vacuous). Mirrors the
# established fixture in tests/unit/engines/chronicle/test_channel_degradation.py.
@pytest.fixture
def loguru_caplog() -> Iterator[str]:
    bridge_name = "khora.test.filter_pushdown"

    def _sink(message: Any) -> None:
        record = message.record
        logging.getLogger(bridge_name).log(record["level"].no, record["message"])

    sink_id = loguru_logger.add(_sink, level="WARNING", format="{message}")
    try:
        yield bridge_name
    finally:
        loguru_logger.remove(sink_id)


# The representative three-predicate filter (one predicate per emission kind).
_RECALL_FILTER: dict[str, Any] = {
    "source_name": "linear",
    "occurred_at": {"$gte": "2026-04-05"},
    "metadata.tag": {"$in": ["urgent", "release"]},
}


def _build_filter_ast() -> FilterNode:
    """Build the canonical AST exactly as the public facade does (khora.py:2045)."""
    return parse_to_ast(RecallFilter.model_validate(_RECALL_FILTER))


def _compiled_where_sql(ast: FilterNode) -> str:
    """Compile an AST to the literal-bound ``khora_chunks`` WHERE SQL string.

    Mirrors the backend's own compile call (pgvector.py:463-466): same
    ``backend_target`` / ``on_unsupported``. ``literal_binds=True`` inlines the
    operands so two predicates that differ only by bind-parameter numbering still
    compare equal — what we assert is the SQL shape, not asyncpg's ``$N`` slots.
    """
    compiled = compile_postgres(
        ast,
        CompileContext(backend_target="khora_chunks", on_unsupported="raise"),
    )
    return str(
        compiled.predicate.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def _predicate_to_sql(predicate: ColumnElement[Any]) -> str:
    """Render a captured ``ColumnElement`` predicate to literal-bound SQL."""
    return str(
        predicate.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def _make_retriever(
    ns_id: UUID,
    *,
    complexity: QueryComplexity = QueryComplexity.MODERATE,
) -> VectorCypherRetriever:
    """A retriever wired so BOTH the vector and BM25 chunk channels fire.

    Mirrors ``TestGracefulDegradation.retriever`` /
    ``TestExplicitTemporalSignalSkipsFallback`` in
    ``test_vectorcypher_retriever.py``.

    ``complexity`` selects which PRIMARY filtered-recall path a plain
    ``recall(query, filter=...)`` HYBRID routes through (this ticket's scope is
    the primary path only — carrying the filter across the adaptive sub-searches,
    session fan-out / restrictive-fallback / CHANGE-recency / vector-only
    fallback, is deferred follow-up work and is deliberately NOT exercised here):

    * ``MODERATE`` (``use_graph=True``) -> ``_vectorcypher_retrieve`` (vector +
      BM25 channels launch in parallel). Graph helpers are stubbed so the path
      completes without Neo4j.
    * ``SIMPLE`` -> ``_simple_retrieve`` (vector + BM25 channels, no graph).

    Common wiring:

    * ``storage.search_similar_entities`` returns one entry entity, so the
      moderate path stays on ``_vectorcypher_retrieve`` instead of dropping to
      the no-entry-entities fallback.
    * ``enable_bm25_channel=True`` so ``_bm25_search_chunks`` actually launches.
    * ``enable_session_aware_search=False`` so the per-session fan-out (a
      deferred-scope sub-search) doesn't issue extra channel calls and muddy the spy.
    """
    vector_store = AsyncMock()
    neo4j_driver = AsyncMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.model_name = "test-model"
    embedder.dimension = 1536

    # Vector channel result (consumed by _vector_search_chunks).
    vec_result = MagicMock()
    vec_result.chunk = MagicMock()
    vec_result.chunk.id = uuid4()
    vec_result.chunk.namespace_id = ns_id
    vec_result.chunk.document_id = uuid4()
    vec_result.chunk.content = "vector channel chunk"
    vec_result.chunk.occurred_at = None
    vec_result.chunk.created_at = None
    vec_result.chunk.source_timestamp = None
    vec_result.chunk.metadata = {}
    vec_result.chunk.chunker_info = {}
    vec_result.combined_score = 0.85
    vec_result.similarity = 0.85
    vector_store.search = AsyncMock(return_value=[vec_result])

    # BM25 channel result (consumed by _bm25_search_chunks via the temporal
    # store's search_fulltext). Returns list[(Chunk, score)].
    from khora.core.models import Chunk

    bm25_chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="bm25 channel chunk")
    vector_store.search_fulltext = AsyncMock(return_value=[(bm25_chunk, 1.0)])

    storage = AsyncMock()
    storage.search_similar_entities = AsyncMock(return_value=[(uuid4(), 0.9)])
    storage.get_entities_batch = AsyncMock(return_value={})
    storage.search_fulltext_chunks = AsyncMock(return_value=[(bm25_chunk, 1.0)])

    config = RetrieverConfig(
        enable_bm25_channel=True,
        enable_session_aware_search=False,
    )
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=neo4j_driver,
        embedder=embedder,
        config=config,
        storage=storage,
    )

    # Route per ``complexity``: SIMPLE -> _simple_retrieve (use_graph=False);
    # MODERATE -> _vectorcypher_retrieve (use_graph=True). HYBRID mode (the
    # default) honours the router's complexity for both.
    use_graph = complexity is not QueryComplexity.SIMPLE
    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=complexity,
            use_graph=use_graph,
            graph_depth=2 if use_graph else 0,
            confidence=0.8,
            reasoning=complexity.value,
        )
    )
    retriever._router.compute_adaptive_depth = MagicMock(return_value=2)

    # Stub graph helpers so the cypher-expansion path completes without Neo4j.
    retriever._cypher_expand = AsyncMock(return_value=({}, {}))
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])
    retriever._version_filter_entities = AsyncMock(return_value=[])

    return retriever


def _capture_bm25_predicate(retriever: VectorCypherRetriever, sink: dict[str, Any]) -> None:
    """Wrap the BM25 fulltext seams so their call kwargs are captured.

    The BM25 channel prefers the temporal store's ``search_fulltext`` and falls
    back to the coordinator's ``search_fulltext_chunks`` (retriever.py:3158-3177).
    Whichever fires, we record its kwargs; the test then asserts the filter
    predicate reached it.
    """
    store = retriever._vector_store
    coord = retriever._storage

    real_store_ft = store.search_fulltext
    real_coord_ft = coord.search_fulltext_chunks

    async def _store_spy(*args: Any, **kwargs: Any) -> Any:
        sink["search_fulltext"] = kwargs
        return await real_store_ft(*args, **kwargs)

    async def _coord_spy(*args: Any, **kwargs: Any) -> Any:
        sink["search_fulltext_chunks"] = kwargs
        return await real_coord_ft(*args, **kwargs)

    store.search_fulltext = _store_spy  # type: ignore[method-assign]
    coord.search_fulltext_chunks = _coord_spy  # type: ignore[method-assign]


# The backend implementation threads ``filter_ast`` from ``retrieve()`` all the
# way down through ``_vectorcypher_retrieve`` into both channel methods. The
# gate below stays SKIPPED only while that chain is STRUCTURALLY incomplete
# (including any transient half-edit where ``retrieve`` has the param but the
# channels don't — which would otherwise raise a confusing ``TypeError``
# mid-recall). Once the full chain is present the gate activates and
# the tests run for real: a behaviourally-broken impl (filter dropped, or a
# different WHERE on the BM25 side) then FAILS LOUDLY. The gate gates on STRUCTURE
# (signatures) only — it never inspects behaviour, so it cannot mask a real bug.
_THREADING_CHAIN = (
    "retrieve",
    "_vectorcypher_retrieve",
    "_vector_search_chunks",
    "_bm25_search_chunks",
)


def _pushdown_chain_ready() -> bool:
    return all(
        "filter_ast" in inspect.signature(getattr(VectorCypherRetriever, m)).parameters for m in _THREADING_CHAIN
    )


_PUSHDOWN_READY = _pushdown_chain_ready()
_pushdown_gate = pytest.mark.skipif(
    not _PUSHDOWN_READY,
    reason=(
        "filter_ast is not yet threaded through the "
        "full VectorCypher channel chain "
        "(retrieve -> _vectorcypher_retrieve -> _vector_search_chunks / _bm25_search_chunks)"
    ),
)


# The two PRIMARY filtered-recall paths a plain ``recall(query, filter=...)``
# HYBRID routes through (this ticket's scope). SIMPLE -> ``_simple_retrieve``;
# MODERATE -> ``_vectorcypher_retrieve``. Both run the vector + BM25 channels and
# forward the filter; the core (a)/(b)/(c) assertions hold on BOTH, so they are
# parametrized over the pair.
@_pushdown_gate
@pytest.mark.parametrize(
    "complexity",
    [QueryComplexity.SIMPLE, QueryComplexity.MODERATE],
    ids=["simple_path", "moderate_path"],
)
class TestVectorCypherFilterPushdownBothChannels:
    """The compiled WHERE reaches BOTH the vector and BM25 channels."""

    async def test_vector_channel_applies_where(self, complexity: QueryComplexity) -> None:
        """(a) The VECTOR channel forwards the filter into ``vector_store.search``."""
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=complexity)
        ast = _build_filter_ast()

        await retriever.retrieve(
            "alpha bravo charlie",
            ns_id,
            limit=10,
            filter_ast=ast,
        )

        # vector_store.search must have been called carrying the filter.
        assert retriever._vector_store.search.await_count >= 1, "vector channel did not run"
        kwargs = retriever._vector_store.search.await_args.kwargs
        forwarded = _extract_channel_filter(kwargs)
        assert forwarded is not None, f"VECTOR channel dropped the filter; vector_store.search kwargs={sorted(kwargs)}"
        # The forwarded filter resolves to the canonical khora_chunks WHERE,
        # whether the impl threads the raw FilterNode or a precompiled ColumnElement.
        assert _channel_where_sql(forwarded) == _compiled_where_sql(ast)

    async def test_bm25_channel_applies_same_where(self, complexity: QueryComplexity) -> None:
        """(b) The BM25 channel applies the SAME WHERE as the vector channel.

        Proven by byte-equality of the literal-bound SQL, so no filter-violating
        chunk can reach RRF from the keyword side.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=complexity)
        ast = _build_filter_ast()

        bm25_calls: dict[str, Any] = {}
        _capture_bm25_predicate(retriever, bm25_calls)

        await retriever.retrieve(
            "alpha bravo charlie",
            ns_id,
            limit=10,
            filter_ast=ast,
        )

        assert bm25_calls, "BM25 channel issued no fulltext call (search_fulltext / search_fulltext_chunks)"
        # Whichever fulltext seam fired must carry the filter as a predicate that
        # compiles to the SAME WHERE the vector channel applied.
        forwarded = None
        for kwargs in bm25_calls.values():
            forwarded = _extract_channel_filter(kwargs)
            if forwarded is not None:
                break
        assert forwarded is not None, f"BM25 channel dropped the filter; captured kwargs={bm25_calls}"
        assert _channel_where_sql(forwarded) == _compiled_where_sql(ast), (
            "BM25 channel applied a DIFFERENT WHERE than the vector channel"
        )

    async def test_no_filter_adds_no_where_on_either_channel(self, complexity: QueryComplexity) -> None:
        """(c) Regression guard: ``filter_ast=None`` -> NO WHERE on either channel.

        Behaviour must be unchanged for the no-filter path: the vector channel
        passes ``filter_ast=None`` (or omits it) and the BM25 channel adds no
        filter predicate.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=complexity)

        bm25_calls: dict[str, Any] = {}
        _capture_bm25_predicate(retriever, bm25_calls)

        await retriever.retrieve(
            "alpha bravo charlie",
            ns_id,
            limit=10,
            filter_ast=None,
        )

        # Vector channel: no filter forwarded.
        assert retriever._vector_store.search.await_count >= 1
        vec_kwargs = retriever._vector_store.search.await_args.kwargs
        assert vec_kwargs.get("filter_ast") is None, (
            f"no-filter recall leaked a filter into the vector channel: {vec_kwargs.get('filter_ast')!r}"
        )

        # BM25 channel: no filter predicate forwarded.
        for kwargs in bm25_calls.values():
            forwarded = _extract_channel_filter(kwargs)
            assert forwarded is None, f"no-filter recall leaked a filter into the BM25 channel: {forwarded!r}"

    async def test_bm25_filter_disables_unfilterable_coordinator_fallback(self, complexity: QueryComplexity) -> None:
        """(b, smuggling guard): under a filter, BM25 must NOT fall back to the
        coordinator's legacy ``chunks`` table.

        The coordinator fallback (``search_fulltext_chunks``) reads the legacy
        ``chunks`` table, whose schema cannot carry the ``khora_chunks``-compiled
        WHERE. If BM25 fell back to it under a filter, it would return UNFILTERED
        rows that smuggle filter-violating chunks into RRF — exactly the PG
        post-filter backstop the filtered path forbids. So when the temporal fulltext path
        yields nothing AND a filter is present, BM25 must return ``[]`` instead of
        taking the fallback (retriever.py:3229 guard).
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=complexity)
        ast = _build_filter_ast()

        # Temporal fulltext path returns EMPTY so the fallback decision is reached.
        retriever._vector_store.search_fulltext = AsyncMock(return_value=[])

        await retriever.retrieve(
            "alpha bravo charlie",
            ns_id,
            limit=10,
            filter_ast=ast,
        )

        # The temporal path WAS consulted (and carried the filter)...
        assert retriever._vector_store.search_fulltext.await_count >= 1
        ft_kwargs = retriever._vector_store.search_fulltext.await_args.kwargs
        assert _extract_channel_filter(ft_kwargs) is not None, "temporal fulltext path did not carry the filter"
        # ...but the unfilterable coordinator fallback must NOT have been used.
        retriever._storage.search_fulltext_chunks.assert_not_awaited()

    async def test_bm25_no_filter_still_uses_coordinator_fallback(self, complexity: QueryComplexity) -> None:
        """(c, control): with NO filter, the coordinator fallback path is unchanged.

        The smuggling guard is filter-gated, so the legacy
        ``search_fulltext_chunks`` fallback still fires when the temporal path is
        empty and no filter is supplied — proving the guard adds no regression to
        the no-filter behaviour.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=complexity)

        # Temporal path empty -> fallback decision reached; no filter -> allowed.
        retriever._vector_store.search_fulltext = AsyncMock(return_value=[])

        await retriever.retrieve(
            "alpha bravo charlie",
            ns_id,
            limit=10,
            filter_ast=None,
        )

        retriever._storage.search_fulltext_chunks.assert_awaited()


# A channel may forward the filter under any of these kwarg names. The
# implementation contract threads ``filter_ast``; we also accept a precompiled-
# predicate spelling so the assertion targets the VALUE, not the name.
_FILTER_KWARG_NAMES = ("filter_ast", "filter_predicate", "where_predicate", "compiled_filter")


def _extract_channel_filter(kwargs: dict[str, Any]) -> Any:
    """Pull the forwarded filter argument out of a captured channel call.

    The implementation may thread either the raw ``filter_ast`` (``FilterNode``)
    or a precompiled ``ColumnElement`` predicate. Accept either kwarg name; return
    ``None`` when no filter was forwarded (the no-filter regression path).
    """
    for key in _FILTER_KWARG_NAMES:
        if kwargs.get(key) is not None:
            return kwargs[key]
    return None


def _channel_where_sql(forwarded: Any) -> str:
    """Render a forwarded filter to a comparable WHERE SQL string.

    Handles both implementation shapes: a raw ``FilterNode`` (compiled here,
    exactly as the backend would compile it) or an already-compiled
    ``ColumnElement`` predicate.
    """
    if isinstance(forwarded, FilterNode):
        return _compiled_where_sql(forwarded)
    return _predicate_to_sql(forwarded)


# Layer B is independent of the retriever threading chain — it lives on the
# relational backend and is always present once the backend change lands. Gate it
# on that method signature so a stale build skips loudly rather than erroring.
_LAYER_B_READY = "filter_ast" in inspect.signature(PgVectorBackend.search_fulltext).parameters
_layer_b_gate = pytest.mark.skipif(
    not _LAYER_B_READY,
    reason="relational backend refusal not landed: PgVectorBackend.search_fulltext has no filter_ast param yet",
)


@_layer_b_gate
class TestRelationalBackendRefusesFilter:
    """Layer B: the relational ``chunks`` backend REFUSES a recall filter.

    The relational ``chunks`` table lacks the denormalized filter columns the
    ``khora_chunks`` compiler targets, so it cannot honor a recall filter. Rather
    than compile (SQL error) or return unfiltered rows (smuggling — forbidden,
    no PG post-filter backstop), ``PgVectorBackend.search_fulltext`` returns
    ``[]`` and logs an ADR-001 ``Degradation`` WARNING when a filter is present.

    This is the backstop for any caller that reaches the relational backend
    directly with a filter (the VectorCypher BM25 channel's Layer-A guard already
    keeps the normal path off this method under a filter — see
    ``test_bm25_filter_disables_unfilterable_coordinator_fallback``).

    No live PG: the ``filter_ast is not None`` branch returns before
    ``_get_session()``, so the backend is built via ``__new__`` (no ``__init__``,
    no engine) and called directly.
    """

    async def test_search_fulltext_refuses_and_warns_under_filter(
        self, caplog: pytest.LogCaptureFixture, loguru_caplog: str
    ) -> None:
        """With a filter present, the relational backend returns [] + logs the
        ADR-001 refusal WARNING (reason=recall_filter_unsupported_on_relational_chunks)."""
        backend = PgVectorBackend.__new__(PgVectorBackend)  # no __init__ -> no DB engine
        ast = _build_filter_ast()

        with caplog.at_level(logging.WARNING, logger=loguru_caplog):
            result = await backend.search_fulltext(uuid4(), "alpha bravo charlie", filter_ast=ast)

        # Refusal: empty result, no smuggled rows.
        assert result == [], "relational backend must return [] (no unfiltered rows) under a filter"

        # ADR-001 surfaced signal: a WARNING carrying the documented reason.
        warns = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "recall_filter_unsupported_on_relational_chunks" in r.getMessage()
        ]
        assert warns, (
            "expected an ADR-001 refusal WARNING with reason "
            "'recall_filter_unsupported_on_relational_chunks'; "
            f"records={[r.getMessage() for r in caplog.records]}"
        )


def _change_recency_retriever(ns_id: UUID) -> VectorCypherRetriever:
    """A MODERATE-path retriever wired so BOTH adaptive sub-searches CAN fire.

    Builds on ``_make_retriever`` but enables the recency channel and stubs the
    two adaptive sub-search entry points so we can assert whether they run:

    * ``_decompose_change_query`` — the CHANGE-decomposition entry (gated on the
      query-string-derived CHANGE signal + truthy ``version_history``).
    * ``_recency_channel_chunks`` — the recency-channel entry (gated on the
      query-string-derived ``is_temporal`` signal + a category ``default_window``).

    ``temporal_recency_floor_enabled=False`` keeps ``synthesis_vetoed`` False so
    the recency gate is reachable; ``_fetch_version_history`` returns a truthy
    list so the CHANGE gate is reachable. With those satisfied, ``filter_ast`` is
    the only remaining deciding condition on each gate.
    """
    retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)
    retriever._config = RetrieverConfig(
        enable_bm25_channel=True,
        enable_session_aware_search=False,
        temporal_recency_channel_enabled=True,
        temporal_recency_floor_enabled=False,
    )
    retriever._fetch_version_history = AsyncMock(return_value=[{"entity": "x"}])
    retriever._decompose_change_query = MagicMock(return_value="current state of the config")
    retriever._recency_channel_chunks = AsyncMock(return_value=[])
    return retriever


_CHANGE_SIGNAL = TemporalSignal(
    is_temporal=True,
    category=TemporalCategory.CHANGE,
    confidence=0.9,
    source="dictionary",
)


@_pushdown_gate
class TestFilterGatesAdaptiveSubSearches:
    """Adaptive sub-searches must NOT contaminate RRF under a ``filter_ast`` recall.

    The CHANGE-decomposition and recency channels run ``temporal_filter=None``
    sub-searches and merge their results into the vector pool that feeds RRF.
    They are gated on the query-string-derived temporal signal, NOT on
    ``temporal_filter`` — so a CHANGE/RECENCY-classified query combined with a
    caller ``RecallFilter`` would otherwise smuggle filter-violating chunks into
    the fused top-k (the deterministic pre-filter contract is violated). Until
    ``filter_ast`` is threaded through those sub-searches (follow-up scope), they
    must be skipped entirely when a filter is in flight.
    """

    async def test_change_and_recency_skipped_under_filter(self) -> None:
        """With a filter present, neither adaptive sub-search runs (no merge)."""
        ns_id = uuid4()
        retriever = _change_recency_retriever(ns_id)

        await retriever.retrieve(
            "what changed in the config",
            ns_id,
            limit=10,
            temporal_signal=_CHANGE_SIGNAL,
            filter_ast=_build_filter_ast(),
        )

        retriever._decompose_change_query.assert_not_called()
        retriever._recency_channel_chunks.assert_not_awaited()

    async def test_change_and_recency_fire_without_filter(self) -> None:
        """Control: with NO filter and the SAME CHANGE signal, both adaptive
        sub-searches run — proving the ``filter_ast`` gate (not some unrelated
        condition) is what suppresses them above. Without this control the
        skip-assertion could pass vacuously."""
        ns_id = uuid4()
        retriever = _change_recency_retriever(ns_id)

        await retriever.retrieve(
            "what changed in the config",
            ns_id,
            limit=10,
            temporal_signal=_CHANGE_SIGNAL,
            filter_ast=None,
        )

        retriever._decompose_change_query.assert_called()
        retriever._recency_channel_chunks.assert_awaited()
