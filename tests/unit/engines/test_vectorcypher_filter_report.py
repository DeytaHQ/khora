"""VectorCypher emits a canonical ``FilterPushdownReport`` as ``engine_info["filter"]``.

Every ``VectorCypherEngine.recall`` stamps ``engine_info["filter"]`` with the
JSON projection of a :class:`khora.filter.report.FilterPushdownReport`, folded
from the per-channel :class:`~khora.filter.report.ChannelPlan` carriers the
retriever collected from each channel's ACTUAL compile this recall. This file
pins that public payload:

* the top-level partition (``pushed_down`` / ``post_filtered`` /
  ``pushed_keys`` / ``post_filtered_keys``), and
* the per-channel ``channels`` breakdown keyed by channel name
  (``"vector"`` / ``"bm25"`` / ``"graph"``).

A channel appears in ``channels`` ONLY when it actually ran and saw the filter
this recall — the report never fabricates a channel that no-opped. The recency
channel is the canonical example: it is config-gateable, but its SQL source
(``search_recent_chunks``) is absent on every wired temporal store, so it
early-returns and records nothing — it is therefore ABSENT from ``channels``
even when the config flag is on and a temporal query fires
(``TestRecencyChannelPresence``).

The representative recall filter (one system key the graph channel pushes, one
``metadata.*`` key the graph channel re-checks in memory)::

    {"source_name": "linear", "metadata.tier": "gold"}

Why engine-level (no live DB): ``recall`` is driven with a fully-wired retriever
whose backend seams are mocked so the channels run deterministically; the
retriever folds the plans into ``engine_info["filter"]`` through the real
``build_filter_report``. The BM25 seam mirrors a RAISE-MODE pg/surreal backend
(every leaf pushed); the split-mode sqlite_lance partial-pushdown shape and the
honest end-to-end row-set proof live in the embedded ``sqlite_lance`` matrix
lane (``tests/integration/matrix``). This file proves the canonical report SHAPE
the engine surfaces.

GATING vs DEFENSIVE (the report's per-leaf partition rule): the vector / bm25 /
graph channels all GATE — each independently feeds RRF fusion, so a leaf any
gating channel re-checks in memory lands in the top-level ``post_filtered_keys``
even when the SQL channels pushed it. NO-DEMOTE applies to a *defensive*
full-predicate re-check, which sets ``post_filtered=True`` without demoting a
fully-pushed leaf — pinned at the builder level AND end-to-end on the real graph
channel (it always runs a full-AST in-memory re-check, so a fully-Cypher-pushed
system-only filter reports ``pushed_down=True`` AND ``post_filtered=True``; see
``TestFilterEdgeCases.test_system_only_filter_is_fully_pushed``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from neo4j.exceptions import ServiceUnavailable

from khora.core.models import Chunk
from khora.engines.vectorcypher.engine import VectorCypherConfig, VectorCypherEngine
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.filter import FilterNode, RecallFilter, parse_to_ast
from khora.filter.execute import filter_leaf_keys
from khora.filter.report import ChannelPlan, FilterPushdownReport
from khora.query import SearchMode
from khora.query.temporal_detection import TemporalCategory, TemporalSignal

pytestmark = pytest.mark.unit


# The representative recall filter: ``source_name`` is a system key the Cypher
# (graph) compiler pushes; ``metadata.tier`` is a JSONB-blob key the graph
# channel cannot push and re-checks in memory. The SQL chunk channels (vector /
# bm25) push BOTH.
_RECALL_FILTER: dict[str, Any] = {"source_name": "linear", "metadata.tier": "gold"}


def _ast(spec: dict[str, Any] | None) -> FilterNode | None:
    """Build the canonical AST exactly as the public facade does (khora.py)."""
    if spec is None:
        return None
    return parse_to_ast(RecallFilter.model_validate(spec))


# ---------------------------------------------------------------------------
# A retriever wired so every channel runs deterministically under a filter.
#
# Mirrors the proven wiring in ``test_vectorcypher_filter_pushdown._make_retriever``
# but additionally makes the mocked VECTOR store append a real ``ChannelPlan`` to
# its ``filter_plan_out`` sink (the live pgvector / lance store does this from its
# own compile) and lets the caller supply graph chunks so the graph channel's
# plan is recorded at its post-filter site.
# ---------------------------------------------------------------------------


def _graph_chunk(ns_id: UUID, *, tier: str) -> Chunk:
    """A chunk the graph channel returns, carrying a ``metadata.tier`` value."""
    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=uuid4(),
        content=f"graph chunk tier={tier}",
        metadata={"tier": tier},
    )


def _make_retriever(
    ns_id: UUID,
    *,
    enable_bm25: bool = False,
    enable_recency: bool = False,
    graph_chunks: list[Chunk] | None = None,
    cypher_raises: bool = False,
    graph_cypher_pushes: bool = True,
) -> VectorCypherRetriever:
    """A MODERATE-routed retriever whose channels run without live backends.

    * VECTOR channel: ``vector_store.search`` appends a ``ChannelPlan`` that
      pushes every leaf to ``filter_plan_out`` (what the real SQL store does on
      its ``khora_chunks`` raise-mode compile), so the vector plan is recorded.
    * BM25 channel (``enable_bm25``): ``vector_store.search_fulltext`` is an
      explicit callable that mirrors a RAISE-MODE pg/surreal backend — it appends
      a ``ChannelPlan`` pushing every leaf to the per-call sink, exactly as the
      live temporal store records its own ``khora_chunks`` fulltext compile. The
      split-mode (sqlite_lance) partial-pushdown shape is proven on the embedded
      matrix lane, not reproducible in a unit mock.
    * GRAPH channel: ``_fetch_chunks_from_entities`` returns ``graph_chunks`` so
      the graph plan is recorded at the in-memory post-filter site, and (when
      ``graph_cypher_pushes``) appends the Cypher-pushed ``consumed_keys`` to the
      ``graph_pushed_keys_out`` sink — exactly as the live Neo4j fetch reports
      back through ``get_chunks_by_entities(pushed_keys_out=...)``, so the graph
      plan derives from the compile that actually ran (``source_name`` pushed,
      ``metadata.tier`` re-checked). With ``graph_cypher_pushes=False`` the fetch
      returns chunks but leaves the sink empty (a store that spliced nothing —
      PPR / storage fallback), so every leaf falls to ``post_filtered_keys``.
      The graph channel ALWAYS sets ``defensive_recheck=True`` (it always
      re-checks the full AST in memory). When ``cypher_raises`` is set,
      ``_cypher_expand`` raises a transient Neo4j error that the retriever catches
      internally (``graph_fallback=True``) — the graph channel then records NO
      plan (it never produced candidate chunks).
    * RECENCY channel (``enable_recency``): enabled in config, but
      ``_recency_channel_chunks`` is stubbed to return ``[]`` — mirroring the
      wired stack, where the recency SQL source (``search_recent_chunks``) is
      absent so the channel no-ops and records NO plan. Recency is therefore
      ABSENT from ``channels`` even when a temporal query fires
      (``TestRecencyChannelPresence``).
    """
    vector_store = AsyncMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.model_name = "test-model"
    embedder.dimension = 1536

    # VECTOR channel result + a real ChannelPlan appended to the sink.
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

    async def _vector_search(*_args: Any, **kwargs: Any) -> list[Any]:
        sink = kwargs.get("filter_plan_out")
        fast = kwargs.get("filter_ast")
        # The live SQL store appends the plan it built from the SAME compile its
        # search ran (raise mode -> every leaf pushed). Empty/None filter: no plan.
        if sink is not None and fast is not None:
            sink.append(ChannelPlan(pushed_keys=filter_leaf_keys(fast)))
        return [vec_result]

    vector_store.search = _vector_search

    # BM25 channel seam: ``search_fulltext`` mirrors a RAISE-MODE pg/surreal
    # backend — its ``khora_chunks`` fulltext compile consumes every leaf, so it
    # appends a ChannelPlan pushing all leaves to the per-call sink (the same way
    # the live temporal store records its own compile). A bare AsyncMock would
    # ignore ``filter_plan_out`` and the bm25 channel would never appear, so we
    # use an explicit callable that honours the sink. The partial-pushdown
    # (split-mode sqlite_lance) shape is proven on the embedded matrix lane, not
    # here — a unit mock cannot faithfully reproduce the lance WHERE + residual.
    bm25_chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="bm25 channel chunk")

    async def _fulltext(
        _namespace_id: UUID,
        _query: str,
        *,
        limit: int,
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
    ) -> list[Any]:
        if filter_ast is not None and filter_plan_out is not None:
            filter_plan_out.append(ChannelPlan(pushed_keys=filter_leaf_keys(filter_ast)))
        return [(bm25_chunk, 1.0)]

    vector_store.search_fulltext = _fulltext

    storage = AsyncMock()
    storage.search_similar_entities = AsyncMock(return_value=[(uuid4(), 0.9)])
    storage.get_entities_batch = AsyncMock(return_value={})
    storage.search_fulltext_chunks = AsyncMock(return_value=[(bm25_chunk, 1.0)])

    config = RetrieverConfig(
        enable_bm25_channel=enable_bm25,
        enable_session_aware_search=False,
        temporal_recency_channel_enabled=enable_recency,
        temporal_recency_floor_enabled=False,
    )
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=AsyncMock(),
        embedder=embedder,
        config=config,
        storage=storage,
    )

    # Route MODERATE -> ``_vectorcypher_retrieve`` (vector + bm25 + graph).
    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=QueryComplexity.MODERATE,
            use_graph=True,
            graph_depth=2,
            confidence=0.8,
            reasoning="moderate",
        )
    )
    retriever._router.compute_adaptive_depth = MagicMock(return_value=2)

    # Graph helpers: cypher expand succeeds (no fallback) unless cypher_raises,
    # in which case a transient Neo4j error is caught internally -> graph_fallback.
    if cypher_raises:
        retriever._cypher_expand = AsyncMock(side_effect=ServiceUnavailable("neo4j down"))
    else:
        retriever._cypher_expand = AsyncMock(return_value=({}, {}))
    gc = graph_chunks if graph_chunks is not None else []

    async def _fetch_entity_chunks(
        *_args: Any,
        filter_ast: FilterNode | None = None,
        graph_pushed_keys_out: list[frozenset[str]] | None = None,
        **_kwargs: Any,
    ) -> list[tuple[UUID, float, Chunk]]:
        # Mirror the Neo4j BFS fetch: it splices the Cypher system-key slice and
        # reports the consumed keys back through the sink (what the live
        # ``get_chunks_by_entities(pushed_keys_out=...)`` does). Computing the
        # consumed set with the real Cypher compiler keeps the reported graph
        # disposition identical to the executing path. ``graph_cypher_pushes=False``
        # simulates a fetch that spliced nothing (PPR / storage fallback): the
        # sink stays empty so every leaf is reported as post-filtered.
        if graph_cypher_pushes and filter_ast is not None and graph_pushed_keys_out is not None:
            from khora.filter.compilers.cypher import compile_cypher
            from khora.filter.execute import build_compile_context

            compiled = compile_cypher(
                filter_ast,
                build_compile_context("Chunk", table_alias="c", on_unsupported="split"),
            )
            if compiled.consumed_keys:
                graph_pushed_keys_out.append(compiled.consumed_keys)
        return [(c.id, 0.9, c) for c in gc]

    retriever._fetch_chunks_from_entities = _fetch_entity_chunks
    retriever._version_filter_entities = AsyncMock(return_value=[])

    # Stub the CHANGE-path graph helpers unconditionally: a CHANGE-classified
    # query reaches version-history fetch + decomposition (which touch Neo4j)
    # BEFORE the recency block, independent of the recency config flag.
    retriever._recency_channel_chunks = AsyncMock(return_value=[])
    retriever._fetch_version_history = AsyncMock(return_value=[{"entity": "x"}])
    retriever._decompose_change_query = MagicMock(return_value="current state of the config")

    return retriever


def _build_engine(retriever: VectorCypherRetriever) -> VectorCypherEngine:
    """A connected ``VectorCypherEngine`` driven by an injected retriever.

    Bypasses ``connect()`` (no live backends): the engine ``recall`` only needs
    ``_retriever``, ``_vc_config``, ``_config`` and ``_connected`` to fold the
    plans into ``engine_info["filter"]``.
    """
    engine = VectorCypherEngine.__new__(VectorCypherEngine)
    engine._config = MagicMock()
    # Abstention knobs (#1331) — recall() reads these off config.query and
    # compares them numerically, so a bare MagicMock would raise on the
    # ``chunk_count < min_chunks`` comparison.
    engine._config.query.abstention_min_chunks = 1
    engine._config.query.abstention_min_top_score = 0.3
    engine._config.query.abstention_combined_threshold = 0.5
    engine._config.query.abstention_weight_entities_empty = 0.3
    engine._config.query.abstention_weight_chunks_below_min = 0.4
    engine._config.query.abstention_weight_top_score_low = 0.3
    engine._config.query.abstention_mode = "cosine_floor"
    engine._config.query.abstention_confidence_target_cosine = 0.5
    engine._config.query.abstention_confidence_target_gap = 0.1
    engine._vc_config = VectorCypherConfig()
    engine._retriever = retriever
    engine._connected = True
    return engine


async def _recall_filter_report(
    engine: VectorCypherEngine,
    ns_id: UUID,
    *,
    query: str = "alpha bravo charlie",
    mode: SearchMode = SearchMode.HYBRID,
    spec: dict[str, Any] | None = _RECALL_FILTER,
) -> dict[str, Any]:
    """Run ``engine.recall`` and return the canonical ``engine_info["filter"]``.

    Asserts on every emitted report that (a) the private channel-plan carrier did
    not leak into public ``engine_info`` and (b) the payload round-trips through
    ``FilterPushdownReport.model_validate`` — the two invariants every test in
    this file shares.
    """
    result = await engine.recall(query, ns_id, limit=10, mode=mode, filter_ast=_ast(spec))
    engine_info = result.engine_info
    report = engine_info["filter"]
    assert "_filter_channel_plans" not in engine_info, "private channel-plan carrier leaked into engine_info"
    # Every emitted report is a valid FilterPushdownReport (REQUIRED case #4).
    FilterPushdownReport.model_validate(report)
    return report


# ---------------------------------------------------------------------------
# 1. HYBRID-mode recall: per-channel entries + top-level partition.
# ---------------------------------------------------------------------------


class TestHybridReport:
    """HYBRID recall folds vector / bm25 / graph plans into the canonical report."""

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): report honestly flags the "
        "filter unenforced against the uncovered entity surface until the #1457 fix filters it",
    )
    async def test_hybrid_vector_bm25_graph(self) -> None:
        """vector+bm25 push both keys; graph pushes source_name, post-filters metadata.tier.

        Top-level: ``source_name`` is pushed on every gating channel (vector,
        bm25, graph all pushed it) so it lands in ``pushed_keys``;
        ``metadata.tier`` was re-checked in memory on the graph channel so it
        lands in ``post_filtered_keys``. Not fully pushed -> ``pushed_down`` False.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, mode=SearchMode.HYBRID)

        # Per-channel breakdown (REQUIRED case #1).
        assert set(report["channels"]) == {"vector", "bm25", "graph"}
        assert report["channels"]["vector"] == {
            "pushed_keys": ["metadata.tier", "source_name"],
            "post_filtered_keys": [],
        }
        assert report["channels"]["bm25"] == {
            "pushed_keys": ["metadata.tier", "source_name"],
            "post_filtered_keys": [],
        }
        assert report["channels"]["graph"] == {
            "pushed_keys": ["source_name"],
            "post_filtered_keys": ["metadata.tier"],
        }

        # Top-level intersection / partition outcome.
        assert report["pushed_keys"] == ["source_name"]
        assert report["post_filtered_keys"] == ["metadata.tier"]
        assert report["pushed_down"] is False
        assert report["post_filtered"] is True

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): report honestly flags the "
        "filter unenforced against the uncovered entity surface until the #1457 fix filters it",
    )
    async def test_hybrid_vector_graph_default_no_bm25(self) -> None:
        """Default HYBRID (bm25 channel OFF) still reports vector + graph honestly.

        The bm25 channel only appears when ``enable_bm25_channel`` is on, so the
        default HYBRID recall folds only ``vector`` + ``graph``. The top-level
        partition is unchanged (graph still demotes ``metadata.tier``).
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=False, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, mode=SearchMode.HYBRID)

        assert set(report["channels"]) == {"vector", "graph"}
        assert report["pushed_keys"] == ["source_name"]
        assert report["post_filtered_keys"] == ["metadata.tier"]
        assert report["pushed_down"] is False
        assert report["post_filtered"] is True


# ---------------------------------------------------------------------------
# 2. GRAPH-mode recall: graph channel only.
# ---------------------------------------------------------------------------


class TestGraphModeReport:
    """GRAPH mode drops the vector + bm25 chunk channels; only graph gates."""

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): report honestly flags the "
        "filter unenforced against the uncovered entity surface until the #1457 fix filters it",
    )
    async def test_graph_only_disposition(self) -> None:
        """GRAPH mode: source_name pushed, metadata.tier post-filtered, by graph alone.

        ``mode=GRAPH`` skips the vector and bm25 chunk channels, so the report
        carries only the ``graph`` channel. Its disposition (cypher pushes
        ``source_name``, re-checks ``metadata.tier``) is the whole top-level
        partition.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, mode=SearchMode.GRAPH)

        assert set(report["channels"]) == {"graph"}
        assert report["channels"]["graph"] == {
            "pushed_keys": ["source_name"],
            "post_filtered_keys": ["metadata.tier"],
        }
        assert report["pushed_keys"] == ["source_name"]
        assert report["post_filtered_keys"] == ["metadata.tier"]
        assert report["pushed_down"] is False
        assert report["post_filtered"] is True

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): report honestly flags the "
        "filter unenforced against the uncovered entity surface until the #1457 fix filters it",
    )
    async def test_graph_fetch_that_spliced_nothing_pushes_no_keys(self) -> None:
        """A graph fetch that never touched the sink reports ``pushed_keys == []``.

        Because the graph plan's pushed keys now come from the sink the executing
        fetch fills (not the retriever-side probe compile), a fetch that spliced
        no Cypher ``WHERE`` (the PPR path, or a graph store that pushed nothing)
        leaves the sink empty — so EVERY constraint leaf falls to
        ``post_filtered_keys`` and the report cannot over-claim a pushdown that
        did not happen. The full-AST in-memory post-filter still enforces the
        filter, so ``defensive_recheck`` flips ``post_filtered`` true.
        """
        ns_id = uuid4()
        retriever = _make_retriever(
            ns_id,
            enable_bm25=True,
            graph_chunks=[_graph_chunk(ns_id, tier="gold")],
            graph_cypher_pushes=False,
        )
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, mode=SearchMode.GRAPH)

        assert set(report["channels"]) == {"graph"}
        assert report["channels"]["graph"] == {
            "pushed_keys": [],
            "post_filtered_keys": ["metadata.tier", "source_name"],
        }
        assert report["pushed_keys"] == []
        assert report["post_filtered_keys"] == ["metadata.tier", "source_name"]
        assert report["pushed_down"] is False
        assert report["post_filtered"] is True


# ---------------------------------------------------------------------------
# 3. No-filter carrier: report present, all-False, empty channels.
# ---------------------------------------------------------------------------


class TestNoFilterReport:
    """A no-filter recall still emits an honest all-False report."""

    @pytest.mark.parametrize("mode", [SearchMode.HYBRID, SearchMode.GRAPH], ids=["hybrid", "graph"])
    async def test_no_filter_all_false(self, mode: SearchMode) -> None:
        """``filter=None`` -> report present, nothing narrowed, ``channels`` empty.

        No channel ran a filter compile, so no plan is recorded and the builder
        yields the constraint-free all-False report with an empty ``channels``
        map (the carriers are absent, not zeroed).
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, mode=mode, spec=None)

        assert report == {
            "pushed_down": False,
            "post_filtered": False,
            "pushed_keys": [],
            "post_filtered_keys": [],
            "unenforced_keys": [],
            "channels": {},
        }


# ---------------------------------------------------------------------------
# 4. Validity round-trip is asserted on EVERY report via _recall_filter_report
#    (FilterPushdownReport.model_validate). This dedicated test makes the
#    REQUIRED guarantee explicit and self-documenting.
# ---------------------------------------------------------------------------


class TestReportAlwaysValidates:
    """``FilterPushdownReport.model_validate`` succeeds on every emitted report."""

    @pytest.mark.parametrize(
        "spec",
        [
            _RECALL_FILTER,
            {"source_name": "linear"},
            {"metadata.tier": "gold"},
            {},
            None,
        ],
        ids=["system+metadata", "system_only", "metadata_only", "empty", "no_filter"],
    )
    async def test_every_report_validates(self, spec: dict[str, Any] | None) -> None:
        """Round-trip the JSON payload back through the canonical model.

        ``_recall_filter_report`` already calls ``model_validate``; re-validating
        the returned dict here makes the contract explicit and guards a future
        engine change that emits a non-conforming dict.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, spec=spec)
        # A second, explicit validation of the exact dict surfaced on engine_info.
        FilterPushdownReport.model_validate(report)


# ---------------------------------------------------------------------------
# 6. Defensive NO-DEMOTE distinction.
#
#   * A gating channel that post-filters a leaf in memory DEMOTES it: a leaf
#     pushed by vector/bm25 but re-checked by the graph channel lands in the
#     top-level ``post_filtered_keys`` (see ``test_hybrid_vector_bm25_graph``).
#   * A leaf pushed by ALL gating channels stays in ``pushed_keys``.
#   * A *defensive* full-predicate re-check (``defensive_recheck=True``) sets
#     ``post_filtered=True`` WITHOUT demoting a fully-pushed leaf (NO-DEMOTE).
#
# The defensive NO-DEMOTE rule is pinned both at the builder level (below) and
# end-to-end through the real graph channel in
# ``TestFilterEdgeCases.test_system_only_filter_is_fully_pushed`` (the graph
# channel ALWAYS runs its full-AST in-memory re-check, so a fully-Cypher-pushed
# system-only filter reports ``pushed_down=True`` AND ``post_filtered=True``
# without demoting the pushed leaf). It is also exercised end-to-end on the
# embedded ``sqlite_lance`` matrix lane's vector channel.
# ---------------------------------------------------------------------------


class TestGatingVsDefensive:
    """A defensive re-check flips ``post_filtered`` but does not demote."""

    def test_defensive_recheck_does_not_demote(self) -> None:
        """A defensive full-predicate re-check flips ``post_filtered`` but keeps
        every fully-pushed leaf in ``pushed_keys`` (NO-DEMOTE).

        This is the embedded ``sqlite_lance`` vector channel's behaviour: its
        compiler pushes every leaf into the lance WHERE, and an always-on
        in-memory ``compile_python`` post-filter re-checks the full AST as a
        safety net. The report must report ``pushed_down=True`` AND
        ``post_filtered=True`` simultaneously, with both leaves still in
        ``pushed_keys`` — distinct from the gating recency case above. Pinned at
        the builder level (the canonical fold) so the distinction is unambiguous.
        """
        from khora.filter.report import build_filter_report

        ast = _ast(_RECALL_FILTER)
        assert ast is not None
        leaves = filter_leaf_keys(ast)

        report = build_filter_report(
            ast,
            {"vector": ChannelPlan(pushed_keys=leaves, defensive_recheck=True)},
        )

        # Every leaf stays pushed; post_filtered flipped by the defensive re-check.
        assert sorted(report.pushed_keys) == sorted(leaves)
        assert report.post_filtered_keys == []
        assert report.pushed_down is True
        assert report.post_filtered is True
        # And the defensive channel itself reports no in-memory post-filter keys.
        assert report.channels["vector"].post_filtered_keys == []
        assert sorted(report.channels["vector"].pushed_keys) == sorted(leaves)


# ---------------------------------------------------------------------------
# 7. Edge cases (devil's-advocate): empty {}, system-only, metadata-only.
# ---------------------------------------------------------------------------


class TestFilterEdgeCases:
    """Constraint-free, all-pushable, and all-residual filters."""

    async def test_empty_filter_carries_channels_but_no_leaves(self) -> None:
        """An empty ``{}`` filter narrows nothing: channels ran, no leaves recorded.

        ``parse_to_ast(RecallFilter.model_validate({}))`` is a childless AND root
        (not ``None``), so the channels still execute a (vacuous) compile and
        appear in ``channels`` with empty key lists. There are no constraint
        leaves, so ``pushed_down`` is False and ``pushed_keys`` /
        ``post_filtered_keys`` are empty. ``post_filtered`` is True, however: the
        graph channel ALWAYS runs its full-AST in-memory re-check
        (``defensive_recheck=True``), which flips the flag even though there is
        no leaf to demote (NO-DEMOTE — nothing lands in ``post_filtered_keys``).
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, spec={})

        assert report["pushed_down"] is False
        # Graph's always-on defensive re-check flips post_filtered; no leaf moves.
        assert report["post_filtered"] is True
        assert report["pushed_keys"] == []
        assert report["post_filtered_keys"] == []
        # Channels are present (they ran) but carry empty key lists.
        for ch in report["channels"].values():
            assert ch == {"pushed_keys": [], "post_filtered_keys": []}

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): report honestly flags the "
        "filter unenforced against the uncovered entity surface until the #1457 fix filters it",
    )
    async def test_system_only_filter_is_fully_pushed(self) -> None:
        """A system-key-only filter pushes down on every channel -> ``pushed_down``.

        ``source_name`` is Cypher-expressible, so the graph channel pushes it too
        (nothing residual): every gating channel pushed the single leaf, none
        moved it to ``post_filtered_keys``, so ``pushed_keys`` covers every leaf
        and ``pushed_down`` is True.

        ``post_filtered`` is nonetheless True — the graph channel ALWAYS runs its
        full-AST in-memory re-check (``defensive_recheck=True``). This is the
        NO-DEMOTE case proven on a REAL channel: the flag flips, but the
        fully-pushed ``source_name`` stays in ``pushed_keys`` (it is NOT demoted
        to ``post_filtered_keys``), so ``pushed_down`` and ``post_filtered`` are
        True simultaneously. Contrast a GATING in-memory post-filter (graph
        re-checking ``metadata.tier`` in ``test_hybrid_vector_bm25_graph``), which
        DOES demote.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, spec={"source_name": "linear"})

        assert report["pushed_keys"] == ["source_name"]
        assert report["post_filtered_keys"] == []
        assert report["pushed_down"] is True
        # Graph's always-on defensive re-check flips post_filtered without demoting.
        assert report["post_filtered"] is True
        # Graph pushed source_name (no residual); SQL channels too.
        assert report["channels"]["graph"] == {"pushed_keys": ["source_name"], "post_filtered_keys": []}

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): report honestly flags the "
        "filter unenforced against the uncovered entity surface until the #1457 fix filters it",
    )
    async def test_metadata_only_filter_is_post_filtered_on_graph(self) -> None:
        """A metadata-only filter: SQL channels push it, the graph channel re-checks it.

        ``metadata.tier`` is not Cypher-expressible, so the graph channel
        post-filters it in memory and demotes it to the top-level
        ``post_filtered_keys`` (it gates), even though vector / bm25 pushed it.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, spec={"metadata.tier": "gold"})

        assert report["channels"]["graph"] == {"pushed_keys": [], "post_filtered_keys": ["metadata.tier"]}
        assert report["channels"]["vector"] == {"pushed_keys": ["metadata.tier"], "post_filtered_keys": []}
        assert report["pushed_keys"] == []
        assert report["post_filtered_keys"] == ["metadata.tier"]
        assert report["pushed_down"] is False
        assert report["post_filtered"] is True


# ---------------------------------------------------------------------------
# Devil's-advocate scenario A: recency is ABSENT from ``channels`` on every
# wired stack — even with the config flag ON and a temporal query that the
# detector flags.
#
# The recency channel's SQL source (``search_recent_chunks``) is not present on
# any wired temporal store, so ``_recency_channel_chunks`` early-returns ``[]``
# and records no plan: a channel that never produced a candidate that gates in
# RRF is never credited with a filter disposition. Crediting it would be a
# fabrication. ``_make_retriever`` stubs ``_recency_channel_chunks`` to return
# ``[]`` precisely to mirror that real wired-stack no-op.
# ---------------------------------------------------------------------------


class TestRecencyChannelPresence:
    """The recency channel never appears in ``channels`` on the wired stack."""

    async def test_recency_enabled_and_temporal_absent(self) -> None:
        """Recency flag ON + temporal query -> recency channel STILL ABSENT.

        Deliverable #3: even with ``temporal_recency_channel_enabled=True`` AND a
        temporal query the detector classifies as CHANGE (a category carrying a
        default window, so the recency block is reached), the recency channel
        no-ops on the wired stack — its ``search_recent_chunks`` SQL source is
        absent, so it returns no candidates and records no plan. The report must
        NOT list a ``recency`` channel: doing so would credit a channel that
        never enforced anything this recall.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_recency=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, query="what changed in the config")

        assert "recency" not in report["channels"], (
            f"recency fabricated when it no-ops on the wired stack: {sorted(report['channels'])}"
        )

    async def test_recency_enabled_but_non_temporal_absent(self) -> None:
        """Recency flag ON + NON-temporal query -> recency channel ABSENT.

        The flag alone does not fire the channel: a non-temporal query has no
        default window, so the recency block is skipped and records no plan.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_recency=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        # "alpha bravo charlie" is non-temporal -> recency does not fire.
        report = await _recall_filter_report(engine, ns_id, query="alpha bravo charlie")

        assert "recency" not in report["channels"], (
            f"recency fabricated on a non-temporal query: {sorted(report['channels'])}"
        )

    async def test_recency_disabled_and_temporal_absent(self) -> None:
        """Recency flag OFF + temporal query -> recency channel ABSENT.

        The temporal query alone does not fire the channel either; the config
        flag must also be on.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_recency=False, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, query="what changed in the config")

        assert "recency" not in report["channels"], f"recency present while disabled: {sorted(report['channels'])}"

    async def test_recency_enabled_temporal_no_filter_empty_report(self) -> None:
        """Recency flag ON + temporal query + NO filter -> all-False, empty channels.

        With no filter no channel records a plan (the channels gate on
        ``filter_ast is not None``), so even the fired recency channel contributes
        nothing — the report is the constraint-free all-False shape.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_recency=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, query="what changed in the config", spec=None)

        assert report == {
            "pushed_down": False,
            "post_filtered": False,
            "pushed_keys": [],
            "post_filtered_keys": [],
            "unenforced_keys": [],
            "channels": {},
        }


# ---------------------------------------------------------------------------
# Devil's-advocate scenario B: graph fallback (Neo4j error) under a filter.
#
# When ``_cypher_expand`` raises a transient Neo4j error, the retriever catches
# it internally (``graph_fallback=True``) and produces no graph chunks. The graph
# channel must then be ABSENT from ``channels`` — the report does not claim a
# channel that never ran enforced the filter.
# ---------------------------------------------------------------------------


class TestGraphFallbackReport:
    """Under a graph fallback, the graph channel is absent (not fabricated)."""

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): report honestly flags the "
        "filter unenforced against the uncovered entity surface until the #1457 fix filters it",
    )
    async def test_graph_fallback_omits_graph_channel(self) -> None:
        """A transient Neo4j error during expand -> graph channel ABSENT.

        The vector channel still ran and pushed both leaves; with no graph
        channel to demote ``metadata.tier`` the report is fully pushed. The point
        is the HONESTY of ``channels``: graph is not listed because it never
        enforced anything this recall.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, graph_chunks=[_graph_chunk(ns_id, tier="gold")], cypher_raises=True)
        engine = _build_engine(retriever)

        report = await _recall_filter_report(engine, ns_id, mode=SearchMode.HYBRID)

        assert "graph" not in report["channels"], (
            f"graph channel fabricated under a Neo4j fallback: {sorted(report['channels'])}"
        )
        assert "vector" in report["channels"], "vector channel should still report under the fallback"
        # The vector channel pushed both leaves; nothing demoted them.
        assert report["channels"]["vector"] == {
            "pushed_keys": ["metadata.tier", "source_name"],
            "post_filtered_keys": [],
        }
        assert report["pushed_keys"] == ["metadata.tier", "source_name"]
        assert report["post_filtered_keys"] == []
        assert report["pushed_down"] is True


# ---------------------------------------------------------------------------
# Devil's-advocate scenario C: a TYPED_ENTITY_RECENT recall WITH a caller filter
# falls through to the FULL ``_vectorcypher_retrieve`` path and emits a REAL,
# per-channel report.
#
# The dispatch gate (``retrieve()``) enters the #569 fast path ONLY when
# ``filter_ast is None`` — the fast-path Cypher cannot enforce caller filters
# (chunk metadata is a serialized JSON property on the graph node, not queryable
# columns), so a filtered typed-recent recall is routed to
# ``_vectorcypher_retrieve`` instead, which enforces the filter per channel and
# folds REAL ``ChannelPlan`` carriers into ``engine_info["filter"]``. Routing
# TYPED_ENTITY_RECENT through the filtered fallback therefore exercises the same
# vector/bm25/graph channels as a MODERATE recall — the report is non-empty and
# validates. (Pre-#1181 this case asserted ``channels={}`` via the fast path; the
# gate makes that shape STRUCTURALLY UNREACHABLE, so it is flipped here.)
#
# The UNFILTERED fast path (``filter_ast=None``) still runs the #569 Cypher and
# stamps the all-false no-filter carrier (``_filter_channel_plans == {}``), which
# the engine folds into the constraint-free all-False report with empty
# ``channels`` — pinned below.
# ---------------------------------------------------------------------------


def _typed_entity_retriever(ns_id: UUID, rows: list[dict[str, Any]]) -> VectorCypherRetriever:
    """A retriever routed to the typed-entity-recent fast path.

    The router returns ``TYPED_ENTITY_RECENT``; ``_dual_nodes._session()`` yields
    a session whose ``execute_read`` returns ``rows`` so the REAL fast path runs
    (and stamps its no-filter ``_filter_channel_plans == {}`` carrier) rather than
    falling back to ``_vectorcypher_retrieve``. Used by the UNFILTERED case; the
    FILTERED case routes through the fully-wired ``_make_retriever`` channels.
    """
    vector_store = AsyncMock()
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.model_name = "test-model"
    embedder.dimension = 1536

    storage = AsyncMock()
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=AsyncMock(),
        embedder=embedder,
        config=RetrieverConfig(enable_session_aware_search=False),
        storage=storage,
    )
    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=QueryComplexity.TYPED_ENTITY_RECENT,
            use_graph=True,
            graph_depth=2,
            confidence=0.9,
            reasoning="typed_entity_recent",
        )
    )
    retriever._router.compute_adaptive_depth = MagicMock(return_value=2)

    session = AsyncMock()
    session.execute_read = AsyncMock(return_value=rows)

    @asynccontextmanager
    async def _fake_session() -> AsyncIterator[Any]:
        yield session

    dual_nodes = MagicMock()
    dual_nodes._session = _fake_session
    retriever._dual_nodes = dual_nodes
    return retriever


class TestTypedEntityFastPathHonesty:
    """Typed-recent reports: REAL report when filtered, no-filter carrier when not."""

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): report honestly flags the "
        "filter unenforced against the uncovered entity surface until the #1457 fix filters it",
    )
    async def test_filtered_typed_recent_emits_real_report(self) -> None:
        """TYPED_ENTITY_RECENT routed WITH a filter -> non-empty, valid report.

        The dispatch gate skips the #569 fast path under a filter and routes to
        ``_vectorcypher_retrieve``, which enforces the filter per channel. So the
        report is the FULL vector/bm25/graph disposition — non-empty ``channels``
        that pass ``FilterPushdownReport.model_validate`` — exactly the fallback
        coverage the devil's-advocate flagged: a typed-recent query WITH a caller
        filter exercised end-to-end through the enforcing path, not asserted via a
        mock-only ``channels={}`` claim.
        """
        ns_id = uuid4()
        # Fully-wired channels (vector + bm25 + graph) but routed TYPED_ENTITY_RECENT
        # so the gate's filter fall-through is the path actually under test.
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        retriever._router.route = AsyncMock(
            return_value=RoutingDecision(
                complexity=QueryComplexity.TYPED_ENTITY_RECENT,
                use_graph=True,
                graph_depth=2,
                confidence=0.9,
                reasoning="typed_entity_recent",
            )
        )
        engine = _build_engine(retriever)

        # "latest action items" would route to the fast path with no filter; under
        # a filter the gate falls through to the enforcing ``_vectorcypher_retrieve``.
        report = await _recall_filter_report(engine, ns_id, query="latest action items")

        # A REAL per-channel report (the fallback path ran end-to-end).
        assert report["channels"], "filtered typed-recent must emit a non-empty per-channel report"
        assert set(report["channels"]) == {"vector", "bm25", "graph"}
        assert report["channels"]["graph"] == {
            "pushed_keys": ["source_name"],
            "post_filtered_keys": ["metadata.tier"],
        }
        assert report["pushed_keys"] == ["source_name"]
        assert report["post_filtered_keys"] == ["metadata.tier"]
        assert report["post_filtered"] is True
        # ``_recall_filter_report`` already model_validate'd it; re-assert explicitly.
        FilterPushdownReport.model_validate(report)

    async def test_unfiltered_fast_path_carries_no_filter_marker(self) -> None:
        """TYPED_ENTITY_RECENT routed WITHOUT a filter -> all-False, empty channels.

        With no caller filter the gate runs the #569 fast path, which stamps the
        all-false no-filter carrier (``_filter_channel_plans == {}``). The engine
        folds that into the constraint-free all-False report whose ``channels``
        reflect no caller filter (empty map).
        """
        ns_id = uuid4()
        eid, cid, did = uuid4(), uuid4(), uuid4()
        rows = [
            {
                "entity": {"id": str(eid), "name": "ship v2", "entity_type": "ACTION_ITEM", "description": "d"},
                "evidence_chunk": {
                    "id": str(cid),
                    "document_id": str(did),
                    "content": "action item evidence chunk",
                    "occurred_at": None,
                    "chunker_info": None,
                },
            }
        ]
        retriever = _typed_entity_retriever(ns_id, rows)
        engine = _build_engine(retriever)

        # "latest action items" + no filter routes to the typed-entity fast path.
        report = await _recall_filter_report(engine, ns_id, query="latest action items", spec=None)

        assert report == {
            "pushed_down": False,
            "post_filtered": False,
            "pushed_keys": [],
            "post_filtered_keys": [],
            "unenforced_keys": [],
            "channels": {},
        }


# ---------------------------------------------------------------------------
# Devil's-advocate scenario D: simple-path (graph-less) mode channel skips.
#
# VECTOR / KEYWORD route to ``_simple_retrieve`` and skip a chunk channel each;
# the report must reflect exactly which channel ran. The plan is recorded only
# when the channel executed under a filter (vector skipped in KEYWORD, bm25
# skipped in VECTOR).
# ---------------------------------------------------------------------------


def _simple_retriever(ns_id: UUID, *, enable_bm25: bool) -> VectorCypherRetriever:
    """A SIMPLE-routed retriever exercising ``_simple_retrieve`` (vector + bm25, no graph).

    Both seams append a real ``ChannelPlan`` to their sink from their own compile
    (raise-mode -> every leaf pushed), mirroring the live SQL store.
    """
    vector_store = AsyncMock()
    sr = MagicMock()
    sr.chunk = MagicMock()
    sr.chunk.id = uuid4()
    sr.chunk.namespace_id = ns_id
    sr.chunk.document_id = uuid4()
    sr.chunk.content = "vector chunk"
    sr.chunk.occurred_at = None
    sr.chunk.created_at = None
    sr.chunk.source_timestamp = None
    sr.chunk.metadata = {}
    sr.chunk.chunker_info = {}
    sr.similarity = 0.8
    sr.combined_score = 0.8

    async def _search(*_args: Any, **kwargs: Any) -> list[Any]:
        sink = kwargs.get("filter_plan_out")
        fast = kwargs.get("filter_ast")
        if sink is not None and fast is not None:
            sink.append(ChannelPlan(pushed_keys=filter_leaf_keys(fast)))
        return [sr]

    vector_store.search = _search

    bm25_chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="bm25 chunk")

    async def _fulltext(
        _namespace_id: UUID,
        _query: str,
        *,
        limit: int,
        filter_ast: FilterNode | None = None,
        filter_plan_out: list[ChannelPlan] | None = None,
    ) -> list[Any]:
        # The temporal-store fulltext path records the bm25 plan from its compile.
        if filter_ast is not None and filter_plan_out is not None:
            filter_plan_out.append(ChannelPlan(pushed_keys=filter_leaf_keys(filter_ast)))
        return [(bm25_chunk, 1.0)]

    vector_store.search_fulltext = _fulltext

    storage = AsyncMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[(bm25_chunk, 1.0)])
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.model_name = "test-model"
    embedder.dimension = 1536

    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=AsyncMock(),
        embedder=embedder,
        config=RetrieverConfig(enable_bm25_channel=enable_bm25, enable_session_aware_search=False),
        storage=storage,
    )
    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.8,
            reasoning="simple",
        )
    )
    retriever._router.compute_adaptive_depth = MagicMock(return_value=0)
    return retriever


class TestSimplePathModeChannels:
    """VECTOR / KEYWORD / HYBRID select exactly the channels they run."""

    async def test_vector_mode_reports_vector_only(self) -> None:
        """``mode=VECTOR`` skips BM25 -> only the vector channel is in the report."""
        ns_id = uuid4()
        engine = _build_engine(_simple_retriever(ns_id, enable_bm25=True))

        report = await _recall_filter_report(engine, ns_id, mode=SearchMode.VECTOR)

        assert set(report["channels"]) == {"vector"}
        assert report["channels"]["vector"]["pushed_keys"] == ["metadata.tier", "source_name"]

    async def test_keyword_mode_reports_bm25_only(self) -> None:
        """``mode=KEYWORD`` skips the vector store -> only the bm25 channel reports."""
        ns_id = uuid4()
        engine = _build_engine(_simple_retriever(ns_id, enable_bm25=True))

        report = await _recall_filter_report(engine, ns_id, mode=SearchMode.KEYWORD)

        assert set(report["channels"]) == {"bm25"}
        assert report["channels"]["bm25"]["pushed_keys"] == ["metadata.tier", "source_name"]

    async def test_hybrid_simple_reports_both(self) -> None:
        """``mode=HYBRID`` on the simple path runs both -> vector + bm25 present."""
        ns_id = uuid4()
        engine = _build_engine(_simple_retriever(ns_id, enable_bm25=True))

        report = await _recall_filter_report(engine, ns_id, mode=SearchMode.HYBRID)

        assert set(report["channels"]) == {"vector", "bm25"}

    async def test_simple_path_entity_bearing_report_is_clean(self) -> None:
        """The simple path COVERS its entity surface, so ``unenforced_keys == []``.

        Coverage-by-derivation (the green counterpart to the graph-path leak): a
        ``simple_*`` search_mode makes the engine's covered set include
        ``{"entities", "relationships"}``, so even when the recall surfaces
        entities the surface-coverage rule stays inert and every filter leaf is
        enforced. Here the simple-path entity projection returns a genuine entity
        (its ``source_chunk_ids`` overlap the recalled chunk), so the assertion is
        not vacuous — a covered, non-empty entity surface still yields a clean,
        fully-pushed report. This is the PASSING invariant the graph/chronicle
        leak tests are the strict-xfail mirror of.
        """
        ns_id = uuid4()
        retriever = _simple_retriever(ns_id, enable_bm25=True)
        # Make the simple-path entity projection return an entity whose
        # source_chunk_ids overlap the recalled vector chunk, so recall surfaces
        # a non-empty (but COVERED) entity surface.
        recalled_id = uuid4()

        sr = MagicMock()
        sr.chunk = MagicMock()
        sr.chunk.id = recalled_id
        sr.chunk.namespace_id = ns_id
        sr.chunk.document_id = uuid4()
        sr.chunk.content = "vector chunk"
        sr.chunk.occurred_at = None
        sr.chunk.created_at = None
        sr.chunk.source_timestamp = None
        sr.chunk.metadata = {}
        sr.chunk.chunker_info = {}
        sr.similarity = 0.8
        sr.combined_score = 0.8

        async def _search(*_args: Any, **kwargs: Any) -> list[Any]:
            sink = kwargs.get("filter_plan_out")
            fast = kwargs.get("filter_ast")
            if sink is not None and fast is not None:
                sink.append(ChannelPlan(pushed_keys=filter_leaf_keys(fast)))
            return [sr]

        retriever._vector_store.search = _search

        entity = MagicMock()
        entity.id = uuid4()
        entity.name = "acme"
        entity.entity_type = "ORG"
        entity.description = "d"
        entity.attributes = {}
        entity.mention_count = 1
        entity.source_document_ids = []
        entity.source_chunk_ids = [recalled_id]
        retriever._storage.list_entities = AsyncMock(return_value=[entity])
        retriever._storage.list_relationships = AsyncMock(return_value=[])
        engine = _build_engine(retriever)

        result = await engine.recall("q", ns_id, limit=10, mode=SearchMode.HYBRID, filter_ast=_ast(_RECALL_FILTER))
        report = result.engine_info["filter"]
        FilterPushdownReport.model_validate(report)

        # Precondition: the simple path ran (``simple_*`` search_mode) AND surfaced
        # an entity — so the coverage-by-derivation claim is exercised, not vacuous.
        assert str(result.engine_info.get("search_mode", "")).startswith("simple_")
        assert result.entities, "simple-path entity projection surfaced no entity — coverage claim is vacuous"

        # The covered entity surface leaves the fully-pushed report clean.
        assert report["unenforced_keys"] == []
        assert report["pushed_keys"] == ["metadata.tier", "source_name"]
        assert report["pushed_down"] is True


# ---------------------------------------------------------------------------
# Devil's-advocate scenario E: session-aware fan-out records the vector plan ONCE.
#
# When entry entities span multiple sessions, the engine cancels the single
# global vector search and fans out N per-session searches + one unscoped
# fallback. All compile the identical ``khora_chunks`` WHERE, so the report must
# carry exactly ONE vector plan (captured from the unscoped fallback), not N.
# Driven at the retriever level so the temporal signal + fan-out are explicit.
# ---------------------------------------------------------------------------


def _session_aware_retriever(ns_id: UUID, discovered_channels: list[str]) -> VectorCypherRetriever:
    """A retriever wired for the session-aware fan-out path (>=2 discovered sessions).

    The vector store appends a ``ChannelPlan`` from whatever sink is passed; only
    the unscoped fallback search carries a sink, so a faithful impl records one
    vector plan regardless of the fan-out width.
    """
    vector_store = AsyncMock()
    sr = MagicMock()
    sr.chunk = MagicMock()
    sr.chunk.id = uuid4()
    sr.chunk.namespace_id = ns_id
    sr.chunk.document_id = uuid4()
    sr.chunk.content = "vector chunk"
    sr.chunk.occurred_at = None
    sr.chunk.created_at = None
    sr.chunk.source_timestamp = None
    sr.chunk.metadata = {}
    sr.chunk.chunker_info = {}
    sr.similarity = 0.8
    sr.combined_score = 0.8

    async def _search(*_args: Any, **kwargs: Any) -> list[Any]:
        sink = kwargs.get("filter_plan_out")
        fast = kwargs.get("filter_ast")
        if sink is not None and fast is not None:
            sink.append(ChannelPlan(pushed_keys=filter_leaf_keys(fast)))
        return [sr]

    vector_store.search = _search
    vector_store.search_fulltext = AsyncMock(return_value=[])

    storage = AsyncMock()
    storage.search_similar_entities = AsyncMock(return_value=[(uuid4(), 0.9)])
    storage.get_entities_batch = AsyncMock(return_value={})
    storage.search_fulltext_chunks = AsyncMock(return_value=[])

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.model_name = "test-model"
    embedder.dimension = 1536

    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=AsyncMock(),
        embedder=embedder,
        config=RetrieverConfig(enable_bm25_channel=False, enable_session_aware_search=True),
        storage=storage,
    )
    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=QueryComplexity.MODERATE,
            use_graph=True,
            graph_depth=2,
            confidence=0.8,
            reasoning="moderate",
        )
    )
    retriever._router.compute_adaptive_depth = MagicMock(return_value=2)
    retriever._dual_nodes.get_entity_channels = AsyncMock(return_value=discovered_channels)
    retriever._cypher_expand = AsyncMock(return_value=({}, {}))
    retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])
    retriever._version_filter_entities = AsyncMock(return_value=[])
    return retriever


_RECENCY_SIGNAL = TemporalSignal(
    is_temporal=True,
    category=TemporalCategory.RECENCY,
    confidence=0.9,
    source="dictionary",
)


class TestSessionAwareFanoutSingleVectorPlan:
    """Session fan-out records the vector channel exactly once."""

    async def test_fanout_records_vector_plan_once(self) -> None:
        """3 discovered sessions -> fan-out runs, but ONE vector plan in the report.

        The N per-session searches + unscoped fallback compile the identical
        WHERE; only the fallback carries a sink, so the report must show a single
        ``vector`` channel pushing every leaf — not three duplicate entries.
        Driven at the retriever level so the temporal signal is explicit; the
        plans are folded with the real ``build_filter_report``.
        """
        from khora.filter.report import build_filter_report

        ns_id = uuid4()
        retriever = _session_aware_retriever(ns_id, ["alpha", "beta", "gamma"])

        ast = _ast(_RECALL_FILTER)
        result = await retriever.retrieve(
            "what changed recently",
            ns_id,
            limit=10,
            temporal_signal=_RECENCY_SIGNAL,
            filter_ast=ast,
        )

        assert result.metadata["session_aware_activated"] is True
        plans = result.metadata["_filter_channel_plans"]
        # Exactly one vector ChannelPlan carrier (the fan-out did not triplicate it).
        assert "vector" in plans
        assert sorted(plans["vector"].pushed_keys) == ["metadata.tier", "source_name"]

        # The folded report carries a single vector channel entry.
        report = build_filter_report(ast, plans).model_dump(mode="json")
        FilterPushdownReport.model_validate(report)
        assert report["channels"]["vector"] == {
            "pushed_keys": ["metadata.tier", "source_name"],
            "post_filtered_keys": [],
        }


# ---------------------------------------------------------------------------
# Devil's-advocate scenario F: concurrent recalls on a shared retriever do not
# cross-contaminate. The per-call sinks (fresh per ``_vectorcypher_retrieve``
# invocation) keep each report scoped to its own filter.
# ---------------------------------------------------------------------------


class TestConcurrentRecallsNoCrossContamination:
    """Concurrent recalls with distinct filters keep distinct reports."""

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): report honestly flags the "
        "filter unenforced against the uncovered entity surface until the #1457 fix filters it",
    )
    async def test_five_concurrent_recalls_distinct_keys(self) -> None:
        """5 concurrent recalls, each a different filter -> each report is scoped.

        A leaked sink would surface another recall's keys; using DISTINCT key
        sets per recall makes any cross-contamination observable. The fresh
        per-call sinks must keep every report carrying exactly its own keys.
        """
        ns_id = uuid4()
        # One shared retriever; graph fetch returns nothing so only the vector
        # channel reports (isolating the per-call sink under test).
        retriever = _make_retriever(ns_id)
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])
        engine = _build_engine(retriever)

        specs: list[dict[str, Any]] = [
            {"source_name": "x"},
            {"metadata.tier": "gold"},
            {"source_name": "y", "metadata.team": "z"},
            {"metadata.k1": "v"},
            {"metadata.a": "1", "metadata.b": "2"},
        ]
        expected = [sorted(filter_leaf_keys(_ast(s))) for s in specs]  # type: ignore[arg-type]

        results = await asyncio.gather(
            *(engine.recall("q", ns_id, limit=10, mode=SearchMode.HYBRID, filter_ast=_ast(s)) for s in specs)
        )

        for i, result in enumerate(results):
            report = result.engine_info["filter"]
            FilterPushdownReport.model_validate(report)
            assert report["channels"]["vector"]["pushed_keys"] == expected[i], (
                f"recall {i} cross-contaminated: {report['channels']['vector']['pushed_keys']} != {expected[i]}"
            )
            assert report["pushed_keys"] == expected[i]


# ---------------------------------------------------------------------------
# The #1457 repro shape, pinned: a date-keyed filter drives the graph path via
# EXPLICIT temporal synthesis (engine.py's ``filter_constrains_date_key`` gate),
# so the recall surfaces graph-derived entities the chunk filter never touched.
# The engine passes ``covered_surfaces={"chunks"}`` on the graph path, so the
# uncovered entity surface forces the date leaf into ``unenforced_keys``.
#
# This is the same leak the Task-of-record graph/hybrid tests above encode; this
# test pins the SPECIFIC trigger the fix targets — a date predicate on a
# beyond-corpus horizon (2027 against a 2026-ish corpus) — so the #1457 fix, which
# filters the entity surface, flips it to xpass.
# ---------------------------------------------------------------------------


class TestDateKeyedExplicitSynthesisLeak:
    """A date-keyed filter on the graph path leaks the uncovered entity surface."""

    @pytest.mark.xfail(
        strict=True,
        reason="graph-path entity/relationship filter leak (#1457): a date-keyed filter forces the "
        "EXPLICIT graph path, which surfaces an uncovered entity surface the filter never constrained; "
        "the report honestly flags the date leaf unenforced until the #1457 fix filters the surface",
    )
    async def test_date_filter_graph_path_report_is_clean(self) -> None:
        """A ``occurred_at $gte 2027`` filter on the graph path reports a clean pushdown.

        The date predicate trips ``filter_constrains_date_key`` -> EXPLICIT
        temporal synthesis -> the full graph path (``search_mode`` is NOT
        ``simple_*``). The mocked entity search returns an entity, so the recall's
        ``entities`` surface is non-empty and uncovered on the graph path. The
        CLEAN report the #1457 fix restores enforces the single ``occurred_at``
        leaf on the (then-filtered) entity surface, so nothing is unenforced.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[_graph_chunk(ns_id, tier="gold")])
        engine = _build_engine(retriever)

        result = await engine.recall(
            "alpha bravo charlie",
            ns_id,
            limit=10,
            mode=SearchMode.HYBRID,
            filter_ast=_ast({"occurred_at": {"$gte": "2027-01-01T00:00:00Z"}}),
        )
        report = result.engine_info["filter"]
        assert "_filter_channel_plans" not in result.engine_info
        FilterPushdownReport.model_validate(report)

        # Precondition (holds before AND after the fix): the recall genuinely ran
        # the GRAPH path (non-``simple_*`` search_mode or graph chunks spliced)
        # and surfaced entities — so the surface-coverage rule has real fuel and
        # the leak below is not a vacuous fall-through to the simple path.
        search_mode = str(result.engine_info.get("search_mode", ""))
        assert result.entities, "graph path surfaced no entities — the surface-coverage rule is inert"
        assert not search_mode.startswith("simple_") or result.engine_info.get("graph_chunk_count", 0) > 0, (
            f"recall fell through to the simple path (search_mode={search_mode!r}, "
            f"graph_chunk_count={result.engine_info.get('graph_chunk_count')}) — the graph-path leak "
            "is not exercised"
        )

        # The single date leaf is enforced (nothing unenforced) on a clean report.
        assert report["unenforced_keys"] == []
        assert report["pushed_keys"] == ["occurred_at"]
