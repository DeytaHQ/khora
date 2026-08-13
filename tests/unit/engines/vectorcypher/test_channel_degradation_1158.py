"""Entity-vector-search and BM25 channel degradation observability (#1158).

ADR-001 (failure-observability contract): a channel that catches an
exception and returns a default MUST record a ``Degradation`` so the
silent fallback is observable on ``RecallResult.engine_info['degradations']``
and via the ``khora.{engine}.{component}.degraded_total`` counter.

Two VectorCypher recall channels previously swallowed failures with a
bare WARNING + ``return []``:

- ``_vector_search_entities``: entry-entity discovery. When it fails, the
  graph-expansion channel of GRAPH/HYBRID recall silently collapses to
  vector-only (no entry seeds -> ``_simple_retrieve`` fallback) with no
  machine-readable signal.
- ``_bm25_search_chunks``: the independent lexical channel. When it fails,
  the BM25 contribution silently disappears from RRF fusion.

These are pure-unit tests with a mocked storage coordinator / vector store
- no embedded stack, no LLM.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.diagnostics import Degradation
from khora.core.temporal import TemporalChunk, TemporalSearchResult
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.query import SearchMode

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Channel-level unit tests (call the channel method directly)
# ---------------------------------------------------------------------------


async def test_vector_search_entities_records_degradation_on_failure() -> None:
    """When ``search_similar_entities`` raises, a Degradation is appended."""
    storage = MagicMock()
    storage.search_similar_entities = AsyncMock(side_effect=RuntimeError("pgvector down"))
    retriever = VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    degradations: list[Degradation] = []
    results = await retriever._vector_search_entities(
        query_embedding=[0.1] * 8,
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )

    # The channel degrades to empty rather than crashing ...
    assert results == []
    # ... and the silent fallback is now observable.
    assert len(degradations) == 1, f"expected one degradation, got {degradations!r}"
    deg = degradations[0]
    assert deg["component"] == "vectorcypher.entity_vector_search"
    assert deg["reason"] == "channel_exception"
    assert deg["exception"] == "RuntimeError"
    assert "pgvector down" in (deg.get("detail") or "")


async def test_vector_search_entities_no_degradation_when_sink_absent() -> None:
    """Without a ``degradations`` sink the channel still degrades cleanly."""
    storage = MagicMock()
    storage.search_similar_entities = AsyncMock(side_effect=RuntimeError("pgvector down"))
    retriever = VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    # ``degradations`` defaults to None - the guard must short-circuit.
    results = await retriever._vector_search_entities(
        query_embedding=[0.1] * 8,
        namespace_id=uuid4(),
        limit=10,
    )
    assert results == []


async def test_bm25_search_chunks_records_degradation_on_failure() -> None:
    """When the BM25 search raises, a Degradation is appended."""
    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(side_effect=RuntimeError("fulltext index missing"))
    vector_store = MagicMock()
    # No temporal-store fulltext method -> coordinator path is taken and raises.
    vector_store.search_fulltext = None
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    degradations: list[Degradation] = []
    results = await retriever._bm25_search_chunks(
        query="anything",
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )

    assert results == []
    assert len(degradations) == 1, f"expected one degradation, got {degradations!r}"
    deg = degradations[0]
    assert deg["component"] == "vectorcypher.bm25"
    assert deg["reason"] == "channel_exception"
    assert deg["exception"] == "RuntimeError"
    assert "fulltext index missing" in (deg.get("detail") or "")


async def test_bm25_search_chunks_no_degradation_when_sink_absent() -> None:
    """Without a ``degradations`` sink the BM25 channel still degrades cleanly."""
    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(side_effect=RuntimeError("fulltext index missing"))
    vector_store = MagicMock()
    vector_store.search_fulltext = None
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    results = await retriever._bm25_search_chunks(
        query="anything",
        namespace_id=uuid4(),
        limit=10,
    )
    assert results == []


# ---------------------------------------------------------------------------
# Empty-multitoken channel degradation (#1330): a >=2-token keyword query that
# returns ZERO BM25 rows is the OR-fix's residual failure mode. It does not
# raise, so it took its own reason value rather than channel_exception.
# ---------------------------------------------------------------------------


def _make_empty_bm25_retriever() -> VectorCypherRetriever:
    """A retriever whose BM25 channel returns 0 rows without raising."""
    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])
    vector_store = MagicMock()
    vector_store.search_fulltext = None
    return VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )


async def test_bm25_empty_multitoken_records_degradation() -> None:
    """A >=2-token query with 0 BM25 rows records an empty_multitoken_channel degradation."""
    retriever = _make_empty_bm25_retriever()
    degradations: list[Degradation] = []
    results = await retriever._bm25_search_chunks(
        query="status of MER-0001",
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )
    assert results == []
    assert len(degradations) == 1, f"expected one degradation, got {degradations!r}"
    deg = degradations[0]
    assert deg["component"] == "vectorcypher.bm25"
    assert deg["reason"] == "empty_multitoken_channel"


async def test_bm25_empty_singletoken_does_not_degrade() -> None:
    """A single-token (bare-ID) query with 0 rows is NOT a degradation.

    A bare ``MER-9999`` lookup that legitimately finds nothing is expected;
    only multi-token sentence queries that drop the whole lexical channel are
    the observable failure mode the #1330 fix targets.
    """
    retriever = _make_empty_bm25_retriever()
    degradations: list[Degradation] = []
    results = await retriever._bm25_search_chunks(
        query="MER-9999",
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )
    assert results == []
    assert degradations == []


async def test_bm25_empty_multitoken_under_filter_does_not_degrade() -> None:
    """Under a deterministic filter_ast, a 0-row multitoken result is a
    legitimate filtered miss (the predicate excluded every candidate), NOT a
    broken lexical channel. Flagging it would inflate the public counter with
    benign events (CodeRabbit on PR #1332)."""
    from khora.filter.ast import FilterNode, FilterOp

    storage = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])
    vector_store = MagicMock()
    # Filtered path goes through the temporal store's search_fulltext, which
    # honors the predicate and legitimately returns 0 rows.
    vector_store.search_fulltext = AsyncMock(return_value=[])
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(),
        storage=storage,
    )

    degradations: list[Degradation] = []
    results = await retriever._bm25_search_chunks(
        query="status of MER-0001",
        namespace_id=uuid4(),
        limit=10,
        filter_ast=FilterNode(op=FilterOp.AND),
        degradations=degradations,
    )
    assert results == []
    assert degradations == []


# ---------------------------------------------------------------------------
# Full retrieve() tests (degradation surfaces on the result, recall survives)
# ---------------------------------------------------------------------------


def _make_retriever_for_retrieve() -> VectorCypherRetriever:
    """A retriever wired for the COMPLEX (graph-expansion) retrieve path.

    The vector chunk channel returns one chunk so recall produces a result;
    everything else is mocked.
    """
    vector_store = AsyncMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    storage = AsyncMock()

    ns_id = uuid4()
    doc_id = uuid4()
    chunk_id = uuid4()

    mock_result = MagicMock()
    mock_result.chunk = MagicMock()
    mock_result.chunk.id = chunk_id
    mock_result.chunk.namespace_id = ns_id
    mock_result.chunk.content = "vector chunk"
    mock_result.chunk.document_id = doc_id
    mock_result.chunk.occurred_at = None
    mock_result.chunk.created_at = None
    mock_result.chunk.source_timestamp = None
    mock_result.chunk.metadata = {}
    mock_result.chunk.chunker_info = {}
    mock_result.combined_score = 0.85
    mock_result.similarity = 0.85
    vector_store.search = AsyncMock(return_value=[mock_result])

    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])

    config = RetrieverConfig(coherence_weight=0.0, enable_bm25_channel=False)

    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=embedder,
        config=config,
        storage=storage,
    )

    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=QueryComplexity.COMPLEX,
            use_graph=True,
            graph_depth=2,
            confidence=0.9,
            reasoning="complex query",
        )
    )
    retriever._router.compute_adaptive_depth = MagicMock(return_value=2)
    return retriever


async def test_retrieve_surfaces_entity_channel_degradation_and_degrades_to_vector() -> None:
    """An entity-vector-search failure surfaces a Degradation on the result.

    With no entry entities, recall degrades to the vector-only simple path
    instead of crashing, and the Degradation rides along on the result
    metadata (the engine spreads it onto engine_info['degradations']).
    """
    retriever = _make_retriever_for_retrieve()
    retriever._storage.search_similar_entities = AsyncMock(side_effect=RuntimeError("pgvector down"))

    result = await retriever.retrieve("Tell me about Alice", uuid4())

    assert isinstance(result, VectorCypherResult)
    # Recall survived and produced chunks via the vector-only fallback.
    assert len(result.chunks) > 0
    degradations = result.metadata.get("degradations") or []
    components = {d.get("component") for d in degradations}
    assert "vectorcypher.entity_vector_search" in components, f"degradations: {degradations!r}"
    entity_deg = next(d for d in degradations if d.get("component") == "vectorcypher.entity_vector_search")
    assert entity_deg["reason"] == "channel_exception"


async def test_retrieve_surfaces_bm25_channel_degradation() -> None:
    """A BM25 failure surfaces a Degradation while recall still returns chunks."""
    retriever = _make_retriever_for_retrieve()
    # Enable the BM25 channel and make it fail.
    retriever._config.enable_bm25_channel = True
    retriever._storage.search_similar_entities = AsyncMock(return_value=[(uuid4(), 0.9)])
    retriever._cypher_expand = AsyncMock(return_value=({}, {}))
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])
    retriever._storage.get_entities_batch = AsyncMock(return_value={})
    retriever._dual_nodes = None
    # The temporal store exposes a fulltext method that raises.
    retriever._vector_store.search_fulltext = AsyncMock(side_effect=RuntimeError("fulltext down"))

    result = await retriever.retrieve("lexical query", uuid4())

    assert isinstance(result, VectorCypherResult)
    degradations = result.metadata.get("degradations") or []
    components = {d.get("component") for d in degradations}
    assert "vectorcypher.bm25" in components, f"degradations: {degradations!r}"
    bm25_deg = next(d for d in degradations if d.get("component") == "vectorcypher.bm25")
    assert bm25_deg["reason"] == "channel_exception"


# ---------------------------------------------------------------------------
# #1574 — an inapplicable ``bm25_title_weight``.
#
# This one degrades without anything going wrong: the lexical channel runs,
# returns rows, and raises nothing. What is missing is only the requested title
# boost, because the store's FTS index has no ``title`` column (an embedded
# database built before #1574 — its DDL is ``IF NOT EXISTS``, so nothing ever
# migrates it to the 2-column shape).
#
# It is worth a record precisely because it is invisible otherwise. SQLite does
# not reject the surplus weight argument (measured on 3.53.4: extra bm25()
# weights are silently ignored), so the operator who set the knob gets ranking
# identical to not having set it, with nothing anywhere to say so.
# ---------------------------------------------------------------------------


class _FulltextStore:
    """A minimal lexical store: only what the channel actually reaches for.

    A ``MagicMock`` cannot express the third case below — the ``getattr`` probe
    would auto-create a truthy ``fts_has_title`` — so the shape is spelled out.
    ``fts_has_title`` is set as an *instance* attribute only when a case wants
    one, leaving it genuinely absent otherwise.
    """

    def __init__(self, *, fts_has_title: bool | None = None) -> None:
        self.calls: list[dict] = []
        self.search_calls: list[dict] = []
        if fts_has_title is not None:
            self.fts_has_title = fts_has_title

    async def search_fulltext(self, namespace_id, query, **kwargs):
        self.calls.append(kwargs)
        chunk = MagicMock()
        chunk.id = uuid4()
        return [(chunk, 0.9)]

    async def search(self, **kwargs):
        """The hybrid channel's entry point — same store, same missing column.

        Returns a real ``TemporalSearchResult`` rather than a mock because
        ``_vector_search_chunks`` reads nine chunk attributes off it and builds
        a domain ``Chunk``; a MagicMock would pass attribute access and fail
        construction, hiding the assertion behind an unrelated error.
        """
        self.search_calls.append(kwargs)
        return [
            TemporalSearchResult(
                chunk=TemporalChunk(
                    id=uuid4(),
                    namespace_id=uuid4(),
                    document_id=uuid4(),
                    content="floor panels dimensioned to the bay spacing",
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
                similarity=0.9,
            )
        ]


def _title_weight_retriever(store: _FulltextStore, weight: float) -> VectorCypherRetriever:
    return VectorCypherRetriever(
        vector_store=store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(bm25_title_weight=weight),
        storage=MagicMock(),
    )


async def _degradations_for(store: _FulltextStore, weight: float) -> list[Degradation]:
    retriever = _title_weight_retriever(store, weight)
    degradations: list[Degradation] = []
    results = await retriever._bm25_search_chunks(
        query="floor panels dimensioned",
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )
    # Always: the channel is NOT broken. Rows still come back — that is what
    # makes this a degradation rather than a failure, and asserting it here
    # keeps every case below honest about the distinction.
    assert len(results) == 1, f"the lexical channel must still return rows, got {results!r}"
    return degradations


async def test_title_weight_records_degradation_on_a_store_without_the_column() -> None:
    """The #1574 record: weight requested, store cannot apply it."""
    degradations = await _degradations_for(_FulltextStore(fts_has_title=False), 2.0)

    assert len(degradations) == 1, f"expected one degradation, got {degradations!r}"
    deg = degradations[0]
    assert deg["component"] == "vectorcypher.bm25_title_weight"
    assert deg["reason"] == "fts_missing_title_column"
    # The detail has to name the value that was ignored — an operator reading
    # this in a log needs to know which knob did nothing.
    assert "2.0" in (deg.get("detail") or "")


async def test_neutral_weight_on_the_same_store_does_not_degrade() -> None:
    """1.0 asks for nothing, so nothing is being dropped.

    Without this gate every recall on every un-upgraded embedded deployment
    would emit a degradation and bump a public counter, for a default none of
    them chose. That noise would make the signal above worthless.
    """
    assert await _degradations_for(_FulltextStore(fts_has_title=False), 1.0) == []


async def test_capable_store_does_not_degrade_at_a_raised_weight() -> None:
    """The weight applies, so there is nothing to report."""
    assert await _degradations_for(_FulltextStore(fts_has_title=True), 2.0) == []


async def test_store_without_the_probe_attribute_is_assumed_capable() -> None:
    """No ``fts_has_title`` -> no claim either way -> no record.

    Every non-embedded store is in this position: pgvector gets its title
    coverage from migration 058 and exposes no probe, and neither do the
    accept-and-ignore backends or a test double. Defaulting to "degraded" would
    fire the counter on the *normal* Postgres path, where the weight works.
    """
    store = _FulltextStore()
    assert not hasattr(store, "fts_has_title"), "the fixture must leave the attribute genuinely absent"

    assert await _degradations_for(store, 2.0) == []


async def test_title_weight_is_forwarded_to_the_store_on_every_call() -> None:
    """The kwarg is passed unconditionally, including at the default.

    Unconditional forwarding is what lets each backend decide for itself
    (apply / ignore / probe-and-warn) instead of the retriever guessing — and it
    is also the change that can break a hand-rolled test double with a strict
    signature, so it is pinned rather than assumed.
    """
    store = _FulltextStore(fts_has_title=True)
    await _degradations_for(store, 1.0)
    await _degradations_for(store, 3.0)

    assert [call.get("title_weight") for call in store.calls] == [1.0, 3.0]


async def test_no_degradation_sink_still_degrades_cleanly() -> None:
    """``degradations=None`` is the default; the guard must short-circuit."""
    retriever = _title_weight_retriever(_FulltextStore(fts_has_title=False), 2.0)

    results = await retriever._bm25_search_chunks(
        query="floor panels dimensioned",
        namespace_id=uuid4(),
        limit=10,
    )
    assert len(results) == 1


async def _hybrid_search(retriever: VectorCypherRetriever, degradations: list[Degradation]) -> list:
    """One hybrid vector-channel call, sharing the caller's per-recall sink."""
    return await retriever._vector_search_chunks(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        namespace_id=uuid4(),
        temporal_filter=None,
        query_text="floor panels dimensioned",
        limit=10,
        degradations=degradations,
    )


async def test_hybrid_path_records_the_title_weight_degradation() -> None:
    """The hybrid vector channel hands the store a title_weight too.

    Recording only on the dedicated lexical channel would leave the common
    configuration — BM25 channel off, hybrid blend on — silently unweighted:
    the store is asked for a title boost it cannot give, and nothing says so.
    """
    store = _FulltextStore(fts_has_title=False)
    retriever = _title_weight_retriever(store, 2.0)
    degradations: list[Degradation] = []

    chunks = await _hybrid_search(retriever, degradations)

    # Same distinction the lexical cases assert: rows still come back.
    assert len(chunks) == 1, f"the hybrid channel must still return rows, got {chunks!r}"
    assert store.search_calls[0].get("title_weight") == 2.0
    assert len(degradations) == 1, f"expected one degradation, got {degradations!r}"
    assert degradations[0]["component"] == "vectorcypher.bm25_title_weight"
    assert degradations[0]["reason"] == "fts_missing_title_column"


async def test_the_record_is_deduped_to_one_per_recall() -> None:
    """One recall, many channels, one record — and one counter increment.

    A recall runs the lexical channel and the hybrid channel against the SAME
    store, and the session fan-out calls the hybrid channel repeatedly (once
    per session, plus an unscoped fallback). Each observes the identical
    missing column. Without the dedupe an operator would read a counter that
    tracks channel invocations — so a fan-out over 8 sessions reports 9 —
    and could not tell one broken deployment from nine. The counter is bumped
    only on the append, so pinning the record count pins the metric too.
    """
    store = _FulltextStore(fts_has_title=False)
    retriever = _title_weight_retriever(store, 2.0)
    degradations: list[Degradation] = []

    await retriever._bm25_search_chunks(
        query="floor panels dimensioned",
        namespace_id=uuid4(),
        limit=10,
        degradations=degradations,
    )
    await _hybrid_search(retriever, degradations)
    await _hybrid_search(retriever, degradations)

    assert len(degradations) == 1, f"one record per recall, not per channel; got {degradations!r}"
    assert degradations[0]["component"] == "vectorcypher.bm25_title_weight"
    # The dedupe must not swallow OTHER components' records on the same list.
    assert [d["component"] for d in degradations] == ["vectorcypher.bm25_title_weight"]


def _simple_routing() -> RoutingDecision:
    return RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        use_graph=False,
        graph_depth=0,
        confidence=0.9,
        reasoning="simple query",
    )


async def _simple_retrieve_degradations(
    monkeypatch: pytest.MonkeyPatch, mode: SearchMode
) -> tuple[list[Degradation], list[dict], _FulltextStore]:
    """Run ``_simple_retrieve`` in ``mode``; return (degradations, bumps, store).

    The store comes back so each case can prove the search actually ran — a
    "no degradation" assertion is worthless if the path never reached the store.

    The counter is a no-op object without logfire, so it cannot be read back.
    Swapping the module-level counter for a recorder is what makes "the counter
    did not move" an assertion rather than an inference from the record count —
    the two are separate statements, and the guard has to satisfy both.
    """
    import khora.engines.vectorcypher.retriever as retriever_module

    bumps: list[dict] = []
    monkeypatch.setattr(
        retriever_module,
        "_BM25_TITLE_WEIGHT_DEGRADED_COUNTER",
        SimpleNamespace(add=lambda amount, attributes=None: bumps.append(attributes or {})),
    )

    store = _FulltextStore(fts_has_title=False)
    retriever = _title_weight_retriever(store, 2.0)
    retriever._storage = None  # no coordinator fallback; keep the path to the store
    degradations: list[Degradation] = []

    await retriever._simple_retrieve(
        query="floor panels dimensioned",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        namespace_id=uuid4(),
        temporal_filter=None,
        limit=10,
        routing=_simple_routing(),
        mode=mode,
        degradations=degradations,
    )
    return degradations, bumps, store


async def test_pure_vector_mode_does_not_record_an_unused_title_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SearchMode.VECTOR sets hybrid_alpha=None: the weight is never consumed.

    The store skips its internal BM25 entirely on that call, so the missing
    title column costs the query nothing. Recording it would put a degradation
    on a recall that was not degraded — and on the deployment where someone set
    the knob globally, EVERY pure-vector recall would carry one.
    """
    degradations, bumps, store = await _simple_retrieve_degradations(monkeypatch, SearchMode.VECTOR)

    # Anti-vacuity: the search must have RUN and been handed the weight. Without
    # this the test would also pass if the path short-circuited before the store.
    assert len(store.search_calls) == 1, f"the vector search must still run, got {store.search_calls!r}"
    assert store.search_calls[0].get("title_weight") == 2.0
    assert store.search_calls[0].get("hybrid_alpha") is None, "VECTOR mode must ask for a pure-vector search"

    title_records = [d for d in degradations if d["component"] == "vectorcypher.bm25_title_weight"]
    assert title_records == [], f"pure vector must not report an unused weight, got {title_records!r}"
    assert bumps == [], f"the counter must not move either, got {bumps!r}"


async def test_hybrid_mode_on_the_same_setup_does_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contrast half: identical store and weight, blend on -> one record.

    Paired with the VECTOR case above deliberately. Alone, either test passes
    for the wrong reason — a guard that never fires and a guard that always
    fires each satisfy one of them. Only the pair pins the behavior to the
    blend actually being in play.
    """
    degradations, bumps, store = await _simple_retrieve_degradations(monkeypatch, SearchMode.HYBRID)

    assert store.search_calls[0].get("hybrid_alpha") is not None, "HYBRID must ask for a blend"

    title_records = [d for d in degradations if d["component"] == "vectorcypher.bm25_title_weight"]
    assert len(title_records) == 1, f"expected exactly one record, got {title_records!r}"
    assert title_records[0]["reason"] == "fts_missing_title_column"
    assert bumps == [{"reason": "fts_missing_title_column"}], f"expected one counter bump, got {bumps!r}"


async def test_a_separate_recall_records_again() -> None:
    """Dedupe is per-sink, not per-process.

    The guard keys off the in-scope ``degradations`` list, so a fresh recall
    starts clean. If it were memoized on the retriever instead, a long-lived
    engine would report the fault once and then go quiet forever.
    """
    store = _FulltextStore(fts_has_title=False)
    retriever = _title_weight_retriever(store, 2.0)

    first: list[Degradation] = []
    second: list[Degradation] = []
    await _hybrid_search(retriever, first)
    await _hybrid_search(retriever, second)

    assert len(first) == 1
    assert len(second) == 1
