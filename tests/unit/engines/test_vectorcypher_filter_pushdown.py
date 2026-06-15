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
    "_recency_channel_chunks",
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
    list so the CHANGE gate is reachable. With those satisfied, both sub-searches
    run regardless of ``filter_ast`` and carry it through.
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
class TestFilterCarriesIntoAdaptiveSubSearches:
    """Adaptive sub-searches RUN under a ``filter_ast`` recall and carry it.

    The CHANGE-decomposition and recency channels run ``temporal_filter=None``
    sub-searches and merge their results into the vector pool that feeds RRF.
    They are gated on the query-string-derived temporal signal, NOT on
    ``temporal_filter`` — so a CHANGE/RECENCY-classified query combined with a
    caller ``RecallFilter`` still runs them, but now each one carries the caller
    ``filter_ast`` so no filter-violating chunk reaches RRF:

    * CHANGE-decomposition pushes ``filter_ast`` down through
      ``_vector_search_chunks`` (pgvector pushdown).
    * the recency channel reads the relational chunks table (no SQL pushdown), so
      it enforces ``filter_ast`` as an in-memory post-filter.
    """

    async def test_change_and_recency_run_under_filter_and_forward_it(self) -> None:
        """With a filter present, both adaptive sub-searches RUN and forward it.

        The CHANGE-decomposition sub-search pushes ``filter_ast`` down through
        ``_vector_search_chunks``; the recency channel receives ``filter_ast`` so
        it can post-filter in memory. (Inverts the previous skip-under-filter
        premise: the filter is now composed across both, not used to skip them.)
        """
        ns_id = uuid4()
        retriever = _change_recency_retriever(ns_id)
        # Spy the CHANGE-decomposition sub-search so we can read its kwargs.
        change_calls: dict[str, Any] = {}
        real_vector_search = retriever._vector_search_chunks

        async def _vector_spy(*args: Any, **kwargs: Any) -> Any:
            # The CHANGE-decomposition sub-search is the one that runs with
            # temporal_filter=None; capture its filter_ast.
            if kwargs.get("temporal_filter") is None:
                change_calls.setdefault("kwargs", kwargs)
            return await real_vector_search(*args, **kwargs)

        retriever._vector_search_chunks = _vector_spy  # type: ignore[method-assign]

        ast = _build_filter_ast()
        await retriever.retrieve(
            "what changed in the config",
            ns_id,
            limit=10,
            temporal_signal=_CHANGE_SIGNAL,
            filter_ast=ast,
        )

        # CHANGE-decomposition ran and forwarded the caller filter.
        retriever._decompose_change_query.assert_called()
        assert change_calls, "CHANGE-decomposition sub-search did not run under a filter"
        assert change_calls["kwargs"].get("filter_ast") is ast, (
            "CHANGE-decomposition sub-search dropped the caller filter_ast"
        )

        # Recency channel ran and received the caller filter to push into its
        # recency SQL.
        retriever._recency_channel_chunks.assert_awaited()
        recency_kwargs = retriever._recency_channel_chunks.await_args.kwargs
        assert recency_kwargs.get("filter_ast") is ast, "recency channel did not receive the caller filter_ast"

    async def test_change_and_recency_fire_without_filter(self) -> None:
        """Control: with NO filter and the SAME CHANGE signal, both adaptive
        sub-searches still run — and the recency channel receives ``filter_ast=None``.
        Proves the unfiltered behaviour is unchanged."""
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
        recency_kwargs = retriever._recency_channel_chunks.await_args.kwargs
        assert recency_kwargs.get("filter_ast") is None, "no-filter recall leaked a filter into the recency channel"

    async def test_recency_channel_excludes_violating_chunk(self) -> None:
        """A chunk that violates the caller filter never reaches the recency pool.

        The recency channel pushes the caller filter into the temporal store's
        ``search_recent_chunks`` SQL (GitHub issue #1223), so a recent chunk whose
        metadata does not satisfy the filter is excluded at the source and never
        reaches the pool. The stub store below simulates that SQL ``WHERE`` by
        applying the same compiled predicate the real backend would.
        """
        from khora.core.models import Chunk
        from khora.filter.compilers.python import compile_python
        from khora.filter.execute import build_compile_context
        from khora.filter.report import ChannelPlan

        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)

        # Two recent chunks with embeddings above the relevance floor: one matches
        # the filter's metadata.tag membership, one violates it.
        matching = Chunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=uuid4(),
            content="matching recent chunk",
            metadata={"tag": "urgent"},
        )
        matching.embedding = [0.1] * 1536
        violating = Chunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=uuid4(),
            content="violating recent chunk",
            metadata={"tag": "noise"},
        )
        violating.embedding = [0.1] * 1536

        # The store honors ``filter_ast`` exactly as the pgvector backend does:
        # it compiles the filter and drops non-matching rows before returning
        # them, then reports an honest ``pushed_keys`` plan. A violating row is
        # never fetched — there is no in-memory post-filter in the channel.
        async def _recent_with_pushdown(
            namespace_id: UUID,
            limit: int,
            *,
            created_after: Any = None,
            filter_ast: FilterNode | None = None,
            filter_plan_out: list[ChannelPlan] | None = None,
        ) -> list[tuple[Any, float | None]]:
            rows = [(matching, None), (violating, None)]
            if filter_ast is not None:
                predicate = compile_python(
                    filter_ast, build_compile_context("khora_chunks", on_unsupported="raise")
                ).predicate
                rows = [(chunk, score) for chunk, score in rows if predicate(chunk)]
                if filter_plan_out is not None:
                    filter_plan_out.append(ChannelPlan(pushed_keys=frozenset({"metadata.tag"})))
            return rows

        retriever._vector_store.search_recent_chunks = _recent_with_pushdown
        # Floor of 0.0 so both pass the cosine gate and only the SQL pushdown culls.
        retriever._config.temporal_query_relevance_floor = 0.0

        # A metadata-only filter so the pushed predicate (not the system-key
        # columns, which these synthetic chunks don't carry) is the sole factor.
        ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent", "release"]}}))
        result = await retriever._recency_channel_chunks(
            query_embedding=[0.1] * 1536,
            namespace_id=ns_id,
            temporal_filter=None,
            filter_ast=ast,
        )

        returned_ids = {cid for cid, _, _ in result}
        assert matching.id in returned_ids, "filter-matching recent chunk was wrongly dropped"
        assert violating.id not in returned_ids, "filter-violating recent chunk reached the recency pool"


# --------------------------------------------------------------------------- #
# Over-fetch conditional: a residual metadata predicate widens the graph/PPR
# fetch; a no-filter or system-key-only filter does not.
# --------------------------------------------------------------------------- #


@_pushdown_gate
class TestGraphChannelOverFetch:
    """The graph chunk fetch widens ONLY when a metadata predicate is residual.

    Metadata leaves cannot push down to Cypher, so the graph channel must
    over-fetch to leave the in-memory post-filter enough candidates to fuse.
    The fetch budget is ``min(limit * multiplier, 200)`` when a metadata leaf is
    residual, and the historical ``limit * 2`` otherwise. The probe runs the
    Cypher compiler in split mode (never raises) and compares its consumed keys
    to the leaf keys — system-key-only and no-filter recalls push down exactly,
    so they keep the unwidened ``limit * 2`` fetch.

    The fetch limit is observed at ``_fetch_chunks_from_entities`` (the non-PPR
    graph path, which ``_make_retriever`` leaves enabled — ``enable_ppr_retrieval``
    is off by default).
    """

    async def test_residual_metadata_filter_widens_fetch(self) -> None:
        """A metadata predicate -> fetch widens to min(limit*multiplier, 200)."""
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)
        # default multiplier is 3 (RetrieverConfig.metadata_overfetch_multiplier)
        assert retriever._config.metadata_overfetch_multiplier == 3
        ast = _build_filter_ast()  # carries metadata.tag (residual)

        await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

        retriever._fetch_chunks_from_entities.assert_awaited()
        fetch_limit = retriever._fetch_chunks_from_entities.await_args.kwargs["limit"]
        assert fetch_limit == min(10 * 3, 200), f"residual-metadata fetch not widened: limit={fetch_limit}"

    async def test_residual_metadata_overfetch_capped_at_200(self) -> None:
        """The widened fetch is capped at 200 regardless of limit * multiplier."""
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)
        ast = _build_filter_ast()  # residual metadata.tag

        # limit*multiplier = 100*3 = 300 -> capped to 200.
        await retriever.retrieve("alpha bravo charlie", ns_id, limit=100, filter_ast=ast)

        fetch_limit = retriever._fetch_chunks_from_entities.await_args.kwargs["limit"]
        assert fetch_limit == 200, f"over-fetch not capped at 200: limit={fetch_limit}"

    async def test_no_filter_keeps_unwidened_fetch(self) -> None:
        """No filter -> the historical ``limit * 2`` fetch, unchanged."""
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)

        await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=None)

        fetch_limit = retriever._fetch_chunks_from_entities.await_args.kwargs["limit"]
        assert fetch_limit == 10 * 2, f"no-filter recall changed the fetch budget: limit={fetch_limit}"

    async def test_system_key_only_filter_keeps_unwidened_fetch(self) -> None:
        """A system-key-only filter (occurred_at) pushes down fully -> no widening.

        The whole filter is Cypher-expressible, so nothing is residual and the
        in-memory post-filter has no extra work — the fetch stays at ``limit * 2``.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)
        ast = parse_to_ast(RecallFilter.model_validate({"occurred_at": {"$gte": "2026-04-05"}}))

        await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

        fetch_limit = retriever._fetch_chunks_from_entities.await_args.kwargs["limit"]
        assert fetch_limit == 10 * 2, f"system-key-only filter wrongly widened the fetch: limit={fetch_limit}"


# --------------------------------------------------------------------------- #
# Session fan-out: a caller metadata.channel constraint narrows the fan-out to
# the intersection with the discovered entity channels (or skips it entirely).
# --------------------------------------------------------------------------- #

_RECENCY_SIGNAL = TemporalSignal(
    is_temporal=True,
    category=TemporalCategory.RECENCY,
    confidence=0.9,
    source="dictionary",
)


def _session_aware_retriever(ns_id: UUID, discovered_channels: list[str]) -> VectorCypherRetriever:
    """A retriever wired for the session-aware fan-out path.

    ``enable_session_aware_search=True`` plus a temporal signal plus >=1 entry
    entity reaches the session-discovery block. ``get_entity_channels`` is mocked
    to return the supplied channels. PPR stays off so the merged session results
    flow through the normal (non-PPR) graph path.
    """
    retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)
    retriever._config = RetrieverConfig(
        enable_bm25_channel=True,
        enable_session_aware_search=True,
    )
    retriever._dual_nodes.get_entity_channels = AsyncMock(return_value=discovered_channels)
    return retriever


def _spy_vector_search_channels(retriever: VectorCypherRetriever, sink: list[str | None]) -> None:
    """Record the per-call ``temporal_filter.channel`` on ``_vector_search_chunks``.

    The session fan-out launches one ``_vector_search_chunks`` per fanned-out
    channel (each carrying a per-session ``TemporalFilter(channel=ch)``) plus one
    unscoped fallback (``channel`` None / inherited). Capturing the channel on
    every call lets the test read which sessions were actually fanned out.
    """
    real = retriever._vector_search_chunks

    async def _spy(*args: Any, **kwargs: Any) -> Any:
        tf = kwargs.get("temporal_filter")
        sink.append(getattr(tf, "channel", None) if tf is not None else None)
        return await real(*args, **kwargs)

    retriever._vector_search_chunks = _spy  # type: ignore[method-assign]


@_pushdown_gate
class TestSessionFanoutChannelIntersect:
    """A caller ``metadata.channel`` constraint narrows the session fan-out."""

    async def test_fanout_restricted_to_intersection(self) -> None:
        """Caller channels that intersect discovered channels -> fan out the overlap only.

        Discovered sessions {alpha, beta, gamma}; caller pins {alpha, beta}. The
        fan-out covers alpha + beta only (2 channels -> fan-out activates); gamma
        is excluded. filter_ast still rides on every per-session search.
        """
        ns_id = uuid4()
        retriever = _session_aware_retriever(ns_id, ["alpha", "beta", "gamma"])
        channels_seen: list[str | None] = []
        _spy_vector_search_channels(retriever, channels_seen)

        ast = parse_to_ast(RecallFilter.model_validate({"metadata.channel": {"$in": ["alpha", "beta"]}}))
        result = await retriever.retrieve(
            "what changed recently",
            ns_id,
            limit=10,
            temporal_signal=_RECENCY_SIGNAL,
            filter_ast=ast,
        )

        # Per-session searches fanned out over the intersection {alpha, beta} only.
        fanned = {ch for ch in channels_seen if ch is not None}
        assert fanned == {"alpha", "beta"}, f"fan-out did not narrow to the intersection: {sorted(fanned)}"
        assert "gamma" not in fanned, "fan-out included a channel outside the caller constraint"
        assert result.metadata["session_aware_activated"] is True

    async def test_disjoint_constraint_skips_fanout(self) -> None:
        """A caller constraint disjoint from discovered channels -> skip the fan-out.

        Discovered {alpha, beta}; caller pins {zeta}. The intersection is empty,
        so the fan-out (which needs >=2 channels) is skipped and the global
        filtered vector task is kept — no per-session channel searches run.
        """
        ns_id = uuid4()
        retriever = _session_aware_retriever(ns_id, ["alpha", "beta"])
        channels_seen: list[str | None] = []
        _spy_vector_search_channels(retriever, channels_seen)

        ast = parse_to_ast(RecallFilter.model_validate({"metadata.channel": "zeta"}))
        result = await retriever.retrieve(
            "what changed recently",
            ns_id,
            limit=10,
            temporal_signal=_RECENCY_SIGNAL,
            filter_ast=ast,
        )

        # No per-session (channel-scoped) search ran.
        assert all(ch is None for ch in channels_seen), (
            f"disjoint constraint still fanned out channel-scoped searches: {channels_seen}"
        )
        assert result.metadata["session_aware_activated"] is False

    async def test_no_channel_constraint_fans_out_all_discovered(self) -> None:
        """No channel constraint -> fan-out behaviour is unchanged (all discovered).

        A filter with no top-level ``metadata.channel`` leaf (or no filter) keeps
        the pre-feature behaviour: fan out over every discovered session.
        """
        ns_id = uuid4()
        retriever = _session_aware_retriever(ns_id, ["alpha", "beta"])
        channels_seen: list[str | None] = []
        _spy_vector_search_channels(retriever, channels_seen)

        # A metadata.tag filter has no channel pin -> caller_channels is None.
        ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent"]}}))
        result = await retriever.retrieve(
            "what changed recently",
            ns_id,
            limit=10,
            temporal_signal=_RECENCY_SIGNAL,
            filter_ast=ast,
        )

        fanned = {ch for ch in channels_seen if ch is not None}
        assert fanned == {"alpha", "beta"}, f"unconstrained fan-out did not cover all discovered: {sorted(fanned)}"
        assert result.metadata["session_aware_activated"] is True


# --------------------------------------------------------------------------- #
# Full-AST graph post-filter / split-mode: a metadata leaf inside an $or (whose
# Cypher side collapses to a non-constraining superset) still culls graph chunks;
# emptying the graph channel records a failure-observability degradation entry.
# --------------------------------------------------------------------------- #


def _graph_chunk(ns_id: UUID, *, tag: str) -> Any:
    """A chunk that the graph channel returns, carrying a metadata.tag value."""
    from khora.core.models import Chunk

    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=uuid4(),
        content=f"graph chunk tagged {tag}",
        metadata={"tag": tag},
    )


@_pushdown_gate
class TestGraphChannelFullAstPostFilter:
    """The graph channel re-checks the WHOLE AST in memory before fusion."""

    async def test_or_with_metadata_leaf_culls_graph_chunk(self) -> None:
        """A metadata leaf inside an ``$or`` still drops a violating graph chunk.

        The Cypher side of ``$or`` collapses the metadata branch to a
        non-constraining ``true`` (superset), so a violating chunk can survive the
        graph fetch. The full-AST in-memory post-filter must re-check the whole
        ``$or`` and drop a chunk that satisfies NEITHER branch — here a chunk
        whose tag is not in the metadata branch and whose source_name does not
        match the other branch.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)

        keep = _graph_chunk(ns_id, tag="urgent")  # satisfies the metadata branch
        drop = _graph_chunk(ns_id, tag="noise")  # satisfies neither branch
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(keep.id, 0.9, keep), (drop.id, 0.8, drop)])

        # $or: a metadata leaf (collapses to superset in Cypher) OR a system key
        # that none of the synthetic graph chunks carry.
        ast = parse_to_ast(
            RecallFilter.model_validate({"$or": [{"metadata.tag": "urgent"}, {"source_name": "nonexistent-source"}]})
        )
        result = await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

        graph_ids = result.metadata["search_methods"]["by_method"]["graph"]["chunk_ids"]
        assert str(keep.id) in graph_ids, "$or-satisfying graph chunk was wrongly dropped"
        assert str(drop.id) not in graph_ids, "graph chunk violating the $or reached fusion"

    async def test_graph_channel_emptied_records_degradation(self) -> None:
        """Emptying the graph channel under a filter records a degradation entry.

        When the metadata post-filter drops ALL graph chunks while the SQL-pushed
        vector / BM25 channels returned filtered rows, the graph side
        under-recalled relative to the completeness backstop: one degradation
        entry (component ``vectorcypher.graph_channel``, reason
        ``empty_under_filter``) is appended to the result's degradations list.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)

        # All graph chunks violate the filter -> post-filter empties the channel.
        violators = [_graph_chunk(ns_id, tag="noise"), _graph_chunk(ns_id, tag="other")]
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(c.id, 0.9, c) for c in violators])
        # The vector channel returns a row (it does in _make_retriever), so the
        # "vector/BM25 returned rows" arm of the degradation condition holds.

        ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent", "release"]}}))
        result = await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

        degradations = result.metadata.get("degradations", [])
        graph_empty = [
            d
            for d in degradations
            if d.get("component") == "vectorcypher.graph_channel" and d.get("reason") == "empty_under_filter"
        ]
        assert graph_empty, f"graph-channel-empty degradation not recorded; degradations={degradations}"
        # The graph channel did empty (no graph chunks in the fused provenance).
        assert result.metadata["graph_chunk_count"] == 0

    async def test_no_degradation_when_graph_channel_survives(self) -> None:
        """Control: when the post-filter keeps a graph chunk, no degradation fires."""
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)

        survivor = _graph_chunk(ns_id, tag="urgent")  # passes the filter
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(survivor.id, 0.9, survivor)])

        ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent", "release"]}}))
        result = await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

        degradations = result.metadata.get("degradations", [])
        graph_empty = [d for d in degradations if d.get("reason") == "empty_under_filter"]
        assert not graph_empty, f"degradation wrongly fired while a graph chunk survived: {degradations}"

    async def test_graph_channel_empty_increments_filter_counter(self, monkeypatch) -> None:
        """Emptying the graph channel under a filter fires the declared filter counter.

        The service-level ``khora.recall.filter.graph_channel_empty`` counter
        (owner: filter) must actually increment — not the prior engine-private
        duplicate. Spy on the ``record_graph_channel_empty`` helper the retriever
        calls.
        """
        import khora.engines.vectorcypher.retriever as rmod

        calls: list[int] = []
        monkeypatch.setattr(rmod, "record_graph_channel_empty", lambda: calls.append(1))

        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)
        violators = [_graph_chunk(ns_id, tag="noise")]
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(c.id, 0.9, c) for c in violators])

        ast = parse_to_ast(RecallFilter.model_validate({"metadata.tag": {"$in": ["urgent"]}}))
        await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

        assert calls, "record_graph_channel_empty not invoked when the graph channel emptied under a filter"


# --------------------------------------------------------------------------- #
# Partial failure: an unsupported-filter compile error surfaces to the caller
# (no vector-only fallback); a transient Neo4j error degrades to vector-only.
# --------------------------------------------------------------------------- #


@_pushdown_gate
class TestFilterUnsupportedPartialFailure:
    """A capability gap surfaces; a transient Neo4j error degrades gracefully."""

    async def test_unsupported_filter_raises_and_skips_vector_only_fallback(self) -> None:
        """``RecallFilterUnsupportedError`` propagates; ``_vector_only_fallback`` is not used.

        The Cypher compiler raises ``RecallFilterUnsupportedError`` for a
        predicate the graph backend cannot honor at all — deliberately OUTSIDE
        the transient-error handler so a capability gap is not masked as a Neo4j
        blip. ``retrieve`` only catches transient Neo4j errors, so this raises out
        of ``retrieve`` and the vector-only fallback is NEVER awaited.
        """
        from khora.filter.model import RecallFilterUnsupportedError

        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)
        retriever._fetch_chunks_from_entities = AsyncMock(
            side_effect=RecallFilterUnsupportedError("metadata.tag", "unsupported on this backend")
        )
        retriever._vector_only_fallback = AsyncMock()  # spy

        ast = _build_filter_ast()
        with pytest.raises(RecallFilterUnsupportedError):
            await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

        retriever._vector_only_fallback.assert_not_awaited()

    async def test_compile_cypher_raise_from_graph_fetch_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same contract, raised from the REAL graph-fetch Cypher compile.

        Drives the real ``_fetch_chunks_from_entities`` ->
        ``DualNodeManager.get_chunks_by_entities`` path, where the Cypher compiler
        is imported locally from ``khora.filter.compilers.cypher`` — so the patch
        targets that source module. The SAME compiler also runs earlier as the
        over-fetch probe in ``_vectorcypher_retrieve`` with an identical compile
        context, so an unconditional patch would raise at the probe instead. A
        call counter lets the probe (first call) compile normally and raises only
        on the graph-fetch call (second), which sits BEFORE any Neo4j session use
        (no live driver needed) and OUTSIDE the transient-error handler. The
        capability error escapes ``retrieve`` and the fallback is never awaited.
        """
        from khora.filter.compilers.cypher import compile_cypher as _real_compile
        from khora.filter.model import RecallFilterUnsupportedError

        calls = {"n": 0}

        def _raise_on_graph_fetch(ast: Any, ctx: Any, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            # Call 1 is the over-fetch probe; call 2 is the dual_nodes graph
            # fetch — raise only there so the error genuinely originates from the
            # graph-fetch chain, not the probe.
            if calls["n"] >= 2:
                raise RecallFilterUnsupportedError("metadata.tag", "unsupported on this backend")
            return _real_compile(ast, ctx, *args, **kwargs)

        monkeypatch.setattr("khora.filter.compilers.cypher.compile_cypher", _raise_on_graph_fetch)

        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)
        # Un-stub the graph fetch so the real DualNodeManager path runs and reaches
        # the patched compiler.
        del retriever._fetch_chunks_from_entities
        retriever._vector_only_fallback = AsyncMock()  # spy

        ast = _build_filter_ast()
        with pytest.raises(RecallFilterUnsupportedError):
            await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

        # The probe compiled, then the graph-fetch compile raised: proves the
        # raise originated from the graph fetch, not the probe.
        assert calls["n"] >= 2, "graph-fetch Cypher compile was never reached (raise came from the probe)"
        retriever._vector_only_fallback.assert_not_awaited()

    async def test_transient_neo4j_error_degrades_to_vector_only(self) -> None:
        """Control: a transient Neo4j error IS caught and degrades to vector-only.

        Proves the unsupported-filter case above is NOT simply "every graph error
        skips the fallback" — a genuine transient failure still reaches the
        vector-only fallback path.
        """
        from neo4j.exceptions import ServiceUnavailable

        ns_id = uuid4()
        retriever = _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)
        retriever._fetch_chunks_from_entities = AsyncMock(side_effect=ServiceUnavailable("neo4j down"))

        # Stub the fallback so we observe it was awaited without running the
        # whole simple-retrieve path.
        from khora.engines.vectorcypher.retriever import VectorCypherResult

        fallback_result = VectorCypherResult(
            chunks=[],
            entities=[],
            routing_decision=retriever._router.route.return_value,
            metadata={"fallback_mode": "vector_only"},
        )
        retriever._vector_only_fallback = AsyncMock(return_value=fallback_result)

        ast = _build_filter_ast()
        result = await retriever.retrieve("alpha bravo charlie", ns_id, limit=10, filter_ast=ast)

        retriever._vector_only_fallback.assert_awaited()
        assert result.metadata.get("fallback_mode") == "vector_only"


# --------------------------------------------------------------------------- #
# Restrictive-fallback skip: the unfiltered re-run is suppressed under a caller
# filter (it would smuggle filter-violating chunks into RRF).
# --------------------------------------------------------------------------- #


def _restrictive_fallback_retriever(ns_id: UUID) -> VectorCypherRetriever:
    """A retriever whose sparse vector results would normally trigger the
    restrictive temporal-filter fallback.

    The recency channel stays OFF (``_make_retriever`` default) so the ONLY
    ``_vector_search_chunks`` call with ``temporal_filter=None`` is the
    restrictive fallback re-run — making the spy unambiguous. The vector channel
    returns a single chunk, below ``limit // 2``, so the restrictive-fallback
    arm (``len(vector_chunks) < limit // 2``) is reached.
    """
    return _make_retriever(ns_id, complexity=QueryComplexity.MODERATE)


@_pushdown_gate
class TestRestrictiveFallbackSkippedUnderFilter:
    """The unfiltered restrictive-fallback re-run is suppressed under a filter."""

    async def test_restrictive_fallback_runs_without_filter(self) -> None:
        """Control: with NO caller filter, sparse results trigger the unfiltered re-run.

        With a real ``temporal_filter``, sparse vector results (1 < limit//2), a
        non-EXPLICIT temporal signal, and no caller filter, the engine re-runs the
        vector search with ``temporal_filter=None`` — observable as a
        ``_vector_search_chunks`` call carrying ``temporal_filter=None``.
        """
        ns_id = uuid4()
        retriever = _restrictive_fallback_retriever(ns_id)
        unfiltered_reruns: list[Any] = []
        real = retriever._vector_search_chunks

        async def _spy(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("temporal_filter") is None:
                unfiltered_reruns.append(kwargs)
            return await real(*args, **kwargs)

        retriever._vector_search_chunks = _spy  # type: ignore[method-assign]

        from khora.engines.skeleton.backends import TemporalFilter

        await retriever.retrieve(
            "what happened recently",
            ns_id,
            limit=10,
            temporal_filter=TemporalFilter(channel="ops"),
            temporal_signal=_RECENCY_SIGNAL,
            filter_ast=None,
        )

        assert unfiltered_reruns, "restrictive-fallback unfiltered re-run did not fire without a caller filter"

    async def test_restrictive_fallback_skipped_with_filter(self) -> None:
        """Under a caller filter, the unfiltered re-run is SKIPPED.

        Same restrictive conditions as the control, but a caller ``filter_ast`` is
        present: the re-run would drop the caller filter (it re-searches with
        ``temporal_filter=None`` and threads no ``filter_ast``), smuggling
        filter-violating chunks into RRF — so it must not run. No
        ``_vector_search_chunks`` call carries ``temporal_filter=None``.
        """
        ns_id = uuid4()
        retriever = _restrictive_fallback_retriever(ns_id)
        unfiltered_reruns: list[Any] = []
        real = retriever._vector_search_chunks

        async def _spy(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("temporal_filter") is None:
                unfiltered_reruns.append(kwargs)
            return await real(*args, **kwargs)

        retriever._vector_search_chunks = _spy  # type: ignore[method-assign]

        from khora.engines.skeleton.backends import TemporalFilter

        await retriever.retrieve(
            "what happened recently",
            ns_id,
            limit=10,
            temporal_filter=TemporalFilter(channel="ops"),
            temporal_signal=_RECENCY_SIGNAL,
            filter_ast=_build_filter_ast(),
        )

        assert not unfiltered_reruns, (
            "restrictive-fallback re-ran unfiltered under a caller filter (would smuggle violating chunks into RRF)"
        )
