"""Regression coverage for the #1457 ∃-over-provenance entity/relationship filter.

VectorCypher's graph path surfaces entities / relationships assembled from graph
traversal the caller's chunk filter never touched. GitHub #1457 closes that leak:
``VectorCypherRetriever._filter_surfaces_by_provenance`` re-applies the SAME
``compile_python("Chunk")`` predicate the graph chunk channel uses to each item's
provenance chunks — an entity survives iff ≥1 of its ``source_chunk_ids`` chunk
passes; a relationship with its own provenance follows the same ∃ rule, else the
endpoint-survival fallback. When the provenance fetch succeeds the engine marks
the entity / relationship surfaces COVERED, so the honest ``engine_info["filter"]``
report stops forcing their leaves into ``unenforced_keys``.

This file is the engine-level (no live DB) regression suite for that behavior. It
reuses the fully-wired retriever + engine builders from the sibling report suite
(:mod:`tests.unit.engines.test_vectorcypher_filter_report`) and the
``_wire_graph_entity`` seam that gives the graph path a REAL ``Entity`` with
provenance (the default ``_make_retriever`` builds an empty-provenance stub, which
always drops under a filter). Each case sets its provenance chunks' fields so they
pass — or fail — the filter under test.

The four scenarios (GitHub #1457):

* **Case B** — a beyond-corpus date filter (``occurred_at $gte 2027`` on a 2026
  corpus) empties ALL FOUR result surfaces (chunks / entities / relationships /
  documents).
* **Partial match** — a ``source``-keyed filter narrows the entity surface to
  exactly the entities whose provenance carries the matching source.
* **Truncation** — an entity whose only filter-passing provenance chunk is
  truncated OUT of the returned top-k is STILL returned (entities need not come
  from the returned chunks — the ∃ filter fetches provenance separately).
* **Fail-closed** — a ``get_chunks_batch`` raise DROPS the unverified entity and
  records a ``Degradation``; because every returned item is verified, an
  unverified surface is never returned or reported as enforced (ADR-001).

Two devil's-advocate completeness cases round out the branch coverage:

* **Mixed-provenance relationships (same endpoints)** — the two relationship rules
  compose: on a shared surviving-endpoint pair, an edge whose OWN provenance fails
  drops while a provenance-less legacy edge survives on the endpoint rule.
* **Multi-page fail-closed** — provenance spanning >1 fetch page with a LATER page
  raising keeps the item already decided on an earlier page and drops only the
  still-undecided one (exercised directly against ``filter_items_by_provenance``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk
from khora.core.models.document import DocumentSource
from khora.core.models.entity import Entity, Relationship
from khora.filter.report import FilterPushdownReport
from khora.query import SearchMode
from tests.unit.engines.test_vectorcypher_filter_report import (
    _ast,
    _build_engine,
    _graph_chunk,
    _make_retriever,
    _wire_graph_entity,
)

pytestmark = pytest.mark.unit


# The beyond-corpus horizon: a 2027 lower bound against a 2026-ish corpus.
_DATE_FILTER: dict[str, Any] = {"occurred_at": {"$gte": "2027-01-01T00:00:00Z"}}


def _prov_chunk(ns_id: Any, *, year: int, source: str | None = None) -> Chunk:
    """A provenance chunk whose ``occurred_at`` sits in ``year`` (+ optional source).

    ``occurred_at`` decides whether the chunk clears the ``$gte 2027`` horizon;
    ``source`` (threaded via a ``DocumentSource`` projection, where the ``"Chunk"``
    predicate resolves the denormalized ``source`` key) decides a ``source``-keyed
    filter.
    """
    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=uuid4(),
        content=f"provenance chunk ({year})",
        occurred_at=datetime(year, 6, 1, tzinfo=UTC),
        source_document=(DocumentSource(id=uuid4(), source=source) if source is not None else None),
    )


# ---------------------------------------------------------------------------
# Case B: a beyond-corpus date filter empties all four result surfaces.
# ---------------------------------------------------------------------------


class TestBeyondCorpusDateFilterEmptiesAllSurfaces:
    """``occurred_at $gte 2027`` on a 2026 corpus -> chunks/entities/rels/docs all empty."""

    async def test_case_b_four_empty_lists(self) -> None:
        """A 2027 lower bound against 2026 provenance drops every surface.

        Every channel's candidate is 2026-dated, so nothing clears the horizon:
        the vector / bm25 stores push down and match zero rows, the graph channel's
        full-AST post-filter drops its 2026 chunk, and the ∃-over-provenance filter
        drops the entity (its sole provenance chunk is 2026). Relationships fall
        with the entity (endpoint fallback). Documents are derived from the (now
        empty) chunk + entity surfaces, so they are empty too. The report stays
        clean — nothing surfaced that the filter did not constrain.
        """
        ns_id = uuid4()
        # Graph channel returns a 2026 chunk (its post-filter drops it under the 2027 bound).
        graph_2026 = _graph_chunk(ns_id, tier="gold")
        graph_2026.occurred_at = datetime(2026, 6, 1, tzinfo=UTC)
        retriever = _make_retriever(ns_id, enable_bm25=True, graph_chunks=[graph_2026])
        # The vector + bm25 store pushdowns matched zero rows (2027 bound, 2026 corpus).
        retriever._vector_store.search = AsyncMock(return_value=[])
        retriever._vector_store.search_fulltext = AsyncMock(return_value=[])
        # One entity whose sole provenance chunk is 2026 -> dropped by the ∃ filter.
        prov_2026 = _prov_chunk(ns_id, year=2026)
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="Ancient",
            entity_type="EVENT",
            source_chunk_ids=[prov_2026.id],
        )
        _wire_graph_entity(retriever, entity, provenance_chunks=[prov_2026])
        engine = _build_engine(retriever)

        result = await engine.recall(
            "alpha bravo charlie",
            ns_id,
            limit=10,
            mode=SearchMode.HYBRID,
            filter_ast=_ast(_DATE_FILTER),
        )

        # Four empty result surfaces.
        assert result.chunks == []
        assert result.entities == []
        assert result.relationships == []
        assert result.documents == []

        # The ∃ filter ran and the surfaces are covered -> the report is clean.
        assert result.engine_info.get("provenance_filtered_surfaces") is True
        report = result.engine_info["filter"]
        FilterPushdownReport.model_validate(report)
        assert report["unenforced_keys"] == []


# ---------------------------------------------------------------------------
# Partial match: a source-keyed filter narrows entities by provenance source.
# ---------------------------------------------------------------------------


class TestPartialMatchNarrowsBySourceProvenance:
    """A ``source``-keyed filter keeps only entities whose provenance matches."""

    async def test_source_filter_keeps_only_matching_source_entity(self) -> None:
        """``{"source": "linear", ...}`` -> only the linear-provenance entity survives.

        The filter is composed with a date key so it constrains a date system key
        (``filter_constrains_date_key`` -> EXPLICIT temporal synthesis routes the
        graph path). BOTH entities' provenance clears the date, so ONLY the
        ``source`` leaf differentiates them: the entity whose provenance chunk
        carries ``source="linear"`` survives; the ``source="slack"`` one drops.
        This pins that the ∃ filter narrows to exactly the matching-source
        provenance, not merely "any surviving chunk".
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True)
        # Two 2027 provenance chunks differing only in source.
        linear_chunk = _prov_chunk(ns_id, year=2027, source="linear")
        slack_chunk = _prov_chunk(ns_id, year=2027, source="slack")
        linear_entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="LinearEvent",
            entity_type="EVENT",
            source_chunk_ids=[linear_chunk.id],
        )
        slack_entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="SlackEvent",
            entity_type="EVENT",
            source_chunk_ids=[slack_chunk.id],
        )
        retriever._storage.search_similar_entities = AsyncMock(
            return_value=[(linear_entity.id, 0.9), (slack_entity.id, 0.8)]
        )
        retriever._storage.get_entities_batch = AsyncMock(
            return_value={linear_entity.id: linear_entity, slack_entity.id: slack_entity}
        )
        retriever._storage.get_chunks_batch = AsyncMock(
            return_value={linear_chunk.id: linear_chunk, slack_chunk.id: slack_chunk}
        )
        # Two surviving-eligible entities reach the Neo4j relationship fetch; stub
        # it (this case is about the entity surface, not relationships).
        retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])
        engine = _build_engine(retriever)

        result = await engine.recall(
            "alpha bravo charlie",
            ns_id,
            limit=10,
            mode=SearchMode.HYBRID,
            filter_ast=_ast({"source": "linear", "occurred_at": {"$gte": "2027-01-01T00:00:00Z"}}),
        )

        # Exactly the linear-provenance entity survives; the slack one is dropped.
        assert [e.name for e in result.entities] == ["LinearEvent"]
        assert result.engine_info.get("provenance_filtered_surfaces") is True
        report = result.engine_info["filter"]
        FilterPushdownReport.model_validate(report)
        assert report["unenforced_keys"] == []


# ---------------------------------------------------------------------------
# Truncation: an entity whose passing provenance chunk is not in the returned
# top-k is still returned (entities need not come from the returned chunks).
# ---------------------------------------------------------------------------


class TestEntityNeedNotComeFromReturnedChunks:
    """A truncated-out provenance chunk still keeps its entity (∃ fetches separately)."""

    async def test_entity_kept_when_its_provenance_chunk_is_truncated_out(self) -> None:
        """``limit=1`` truncates the provenance chunk out of the top-k; the entity survives.

        The entity's ONLY filter-passing provenance chunk is fetched separately by
        the ∃ filter (via ``get_chunks_batch``), NOT drawn from the returned chunk
        set. With ``limit=1`` the returned chunks are a single vector-channel chunk
        that is NOT the entity's provenance chunk — yet the entity is still returned,
        pinning that entities need not come from the returned top-k chunks.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True)
        prov = _prov_chunk(ns_id, year=2027)
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="Kept",
            entity_type="EVENT",
            source_chunk_ids=[prov.id],
        )
        _wire_graph_entity(retriever, entity, provenance_chunks=[prov])
        engine = _build_engine(retriever)

        result = await engine.recall(
            "alpha bravo charlie",
            ns_id,
            limit=1,
            mode=SearchMode.HYBRID,
            filter_ast=_ast(_DATE_FILTER),
        )

        # The entity survives.
        assert [e.name for e in result.entities] == ["Kept"]
        # Its provenance chunk is NOT among the returned (truncated) top-k chunks.
        returned_chunk_ids = {c.id for c in result.chunks}
        assert prov.id not in returned_chunk_ids, (
            "the entity's provenance chunk leaked into the returned top-k — the truncation invariant is not exercised"
        )
        # Precondition: the top-k was genuinely truncated to the single vector chunk.
        assert len(result.chunks) == 1, f"expected limit=1 to truncate to one chunk, got {len(result.chunks)}"
        assert result.engine_info.get("provenance_filtered_surfaces") is True


# ---------------------------------------------------------------------------
# Fail-closed: a provenance fetch failure DROPS the unverified entity and records
# a Degradation — an unverified surface is never returned (ADR-001).
# ---------------------------------------------------------------------------


class TestProvenanceFetchFailureIsFailClosed:
    """A ``get_chunks_batch`` raise fail-closes: unverified entities dropped + Degradation."""

    async def test_fetch_failure_drops_entity_and_records_degradation(self) -> None:
        """A provenance fetch raise -> the unverified entity is DROPPED + a Degradation.

        ADR-001 fail-closed (``khora.filter.provenance.filter_items_by_provenance``):
        the entity's provenance could not be verified, so it is DROPPED rather than
        returned unverified (which would re-introduce the #1457 leak). A structured
        ``Degradation`` (``component="vectorcypher.entity_filter"``,
        ``reason="provenance_fetch_failed"``) rides ``engine_info["degradations"]``.
        Because every RETURNED entity is verified, the surface is still legitimately
        enforced — the honest report is clean (``unenforced_keys == []``) and the
        engine marks the surface covered.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True)
        prov = _prov_chunk(ns_id, year=2027)
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="Unverifiable",
            entity_type="EVENT",
            source_chunk_ids=[prov.id],
        )
        retriever._storage.search_similar_entities = AsyncMock(return_value=[(entity.id, 0.9)])
        retriever._storage.get_entities_batch = AsyncMock(return_value={entity.id: entity})
        # The provenance fetch raises -> the ∃ filter cannot verify the surface.
        retriever._storage.get_chunks_batch = AsyncMock(side_effect=RuntimeError("provenance store down"))
        engine = _build_engine(retriever)

        result = await engine.recall(
            "alpha bravo charlie",
            ns_id,
            limit=10,
            mode=SearchMode.HYBRID,
            filter_ast=_ast(_DATE_FILTER),
        )

        # Fail-closed: the unverified entity is DROPPED, not returned.
        assert result.entities == [], "an unverified entity leaked into the result on the degraded path"

        # A structured Degradation names the entity-filter component + reason.
        degradations = result.engine_info.get("degradations", [])
        provenance_degs = [
            d
            for d in degradations
            if d.get("component") == "vectorcypher.entity_filter" and d.get("reason") == "provenance_fetch_failed"
        ]
        assert provenance_degs, (
            f"expected a vectorcypher.entity_filter / provenance_fetch_failed Degradation, "
            f"got: {[(d.get('component'), d.get('reason')) for d in degradations]}"
        )

        # Every returned entity is verified (here: none), so the surface is
        # legitimately enforced — the honest report stays clean.
        assert result.engine_info.get("provenance_filtered_surfaces") is True
        report = result.engine_info["filter"]
        FilterPushdownReport.model_validate(report)
        assert report["unenforced_keys"] == []


# ---------------------------------------------------------------------------
# Relationship endpoint fallback: a provenance-less legacy edge survives iff
# BOTH its endpoints survived the ∃ filter (the impl's endpoint rule).
# ---------------------------------------------------------------------------


class TestRelationshipEndpointFallback:
    """A provenance-less relationship follows the endpoint-survival rule."""

    async def test_legacy_edge_kept_iff_both_endpoints_survive(self) -> None:
        """A provenance-less edge between a surviving + a dropped entity is dropped.

        A relationship carrying no ``source_chunk_ids`` cannot use the ∃-over-
        provenance rule, so the impl keeps it iff BOTH endpoints survived. Here one
        endpoint's provenance clears the date and the other's does not, so the edge
        is dropped even though one endpoint survives — pinning the endpoint-survival
        fallback for legacy edges.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True)
        keep_chunk = _prov_chunk(ns_id, year=2027)
        drop_chunk = _prov_chunk(ns_id, year=2026)
        keep_entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="Survivor",
            entity_type="EVENT",
            source_chunk_ids=[keep_chunk.id],
        )
        drop_entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="Casualty",
            entity_type="EVENT",
            source_chunk_ids=[drop_chunk.id],
        )
        # A provenance-less legacy edge between the two.
        legacy_edge = Relationship(
            id=uuid4(),
            namespace_id=ns_id,
            source_entity_id=keep_entity.id,
            target_entity_id=drop_entity.id,
            relationship_type="RELATES_TO",
            source_chunk_ids=[],
        )
        retriever._storage.search_similar_entities = AsyncMock(
            return_value=[(keep_entity.id, 0.9), (drop_entity.id, 0.8)]
        )
        retriever._storage.get_entities_batch = AsyncMock(
            return_value={keep_entity.id: keep_entity, drop_entity.id: drop_entity}
        )
        retriever._storage.get_chunks_batch = AsyncMock(
            return_value={keep_chunk.id: keep_chunk, drop_chunk.id: drop_chunk}
        )
        # Feed the legacy edge through the retriever's provenance filter directly:
        # the ∃ filter is the unit under test, exercised via its own seam so the
        # relationship-fetch mock shape is not load-bearing.
        kept_entities, kept_rels, filtered = await retriever._filter_surfaces_by_provenance(
            [(keep_entity, 0.9), (drop_entity, 0.8)],
            [(legacy_edge, 0.9)],
            _ast(_DATE_FILTER),
            ns_id,
            [],
        )

        assert filtered is True
        assert {e.name for e, _ in kept_entities} == {"Survivor"}
        # The edge drops: one endpoint (Casualty) did not survive.
        assert kept_rels == []

    async def test_mixed_provenance_edges_same_endpoints(self) -> None:
        """Two edges between the SAME surviving endpoints: the ∃ rule splits them.

        Both endpoints survive (both provenance chunks clear the date), so the
        endpoint rule alone would keep BOTH edges. But one edge carries its own
        provenance that FAILS the date (2026) and the other carries none: the
        provenance-bearing edge is dropped by the ∃-over-own-provenance rule while
        the provenance-less legacy edge survives on the endpoint rule. This pins
        that the two relationship rules compose (``endpoint AND (no-prov OR ∃)``),
        isolating them on a shared-endpoint pair so ONLY the provenance status
        differs.
        """
        ns_id = uuid4()
        retriever = _make_retriever(ns_id, enable_bm25=True)
        e1_chunk = _prov_chunk(ns_id, year=2027)
        e2_chunk = _prov_chunk(ns_id, year=2027)
        e1 = Entity(id=uuid4(), namespace_id=ns_id, name="E1", entity_type="EVENT", source_chunk_ids=[e1_chunk.id])
        e2 = Entity(id=uuid4(), namespace_id=ns_id, name="E2", entity_type="EVENT", source_chunk_ids=[e2_chunk.id])
        # A provenance-bearing edge whose own provenance FAILS the date (2026).
        stale_prov_chunk = _prov_chunk(ns_id, year=2026)
        edge_with_stale_prov = Relationship(
            id=uuid4(),
            namespace_id=ns_id,
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relationship_type="OWN_PROV",
            source_chunk_ids=[stale_prov_chunk.id],
        )
        # A provenance-less legacy edge between the SAME endpoints.
        legacy_edge = Relationship(
            id=uuid4(),
            namespace_id=ns_id,
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            relationship_type="LEGACY",
            source_chunk_ids=[],
        )
        retriever._storage.get_chunks_batch = AsyncMock(
            return_value={
                e1_chunk.id: e1_chunk,
                e2_chunk.id: e2_chunk,
                stale_prov_chunk.id: stale_prov_chunk,
            }
        )

        _kept_entities, kept_rels, filtered = await retriever._filter_surfaces_by_provenance(
            [(e1, 0.9), (e2, 0.8)],
            [(edge_with_stale_prov, 0.9), (legacy_edge, 0.85)],
            _ast(_DATE_FILTER),
            ns_id,
            [],
        )

        assert filtered is True
        # Both endpoints survived, so only the ∃-over-own-provenance rule splits the
        # edges: the stale-provenance edge drops, the legacy (endpoint-rule) edge stays.
        assert [rel.relationship_type for rel, _ in kept_rels] == ["LEGACY"]


# ---------------------------------------------------------------------------
# Multi-page fail-closed: provenance spanning >1 fetch page, with a LATER page
# raising, keeps the items already decided on an earlier page (QA gap #1).
# ---------------------------------------------------------------------------


class _ProvItem:
    """A minimal ``_HasProvenance`` item: an id + a provenance chunk-id sequence."""

    def __init__(self, chunk_ids: list[UUID]) -> None:
        self.id = uuid4()
        self.source_chunk_ids = chunk_ids


class TestMultiPageFailClosedKeepsDecidedItems:
    """A later-page fetch failure drops only the still-undecided items."""

    async def test_page_two_failure_keeps_page_one_survivor(self) -> None:
        """Item decided on page 1 survives; item pending on page 2 drops on the raise.

        ``filter_items_by_provenance`` fetches provenance in ``_PAGE_SIZE`` (500)
        pages. Item A carries a full page of passing (2027) provenance chunks, so it
        is decided on page 1; Item B's single provenance chunk lands on page 2, which
        raises. The fail-closed rule drops only the still-undecided B — A (already
        proven) is kept — and records exactly ONE Degradation. This closes the
        devil's-advocate gap the earlier fail-closed test left (that one fails the
        FIRST fetch, so every item is undecided).
        """
        from khora.filter.provenance import _PAGE_SIZE, filter_items_by_provenance

        ns_id = uuid4()
        # Item A: a full first page of passing (2027) provenance chunks.
        a_chunks = [_prov_chunk(ns_id, year=2027) for _ in range(_PAGE_SIZE)]
        a_map = {c.id: c for c in a_chunks}
        item_a = _ProvItem([c.id for c in a_chunks])
        # Item B: a single provenance chunk that lands on page 2.
        b_chunk = _prov_chunk(ns_id, year=2027)
        item_b = _ProvItem([b_chunk.id])

        calls = {"n": 0}

        async def _paged_fetch(chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
            calls["n"] += 1
            if calls["n"] == 1:
                return {cid: a_map[cid] for cid in chunk_ids if cid in a_map}
            raise RuntimeError("page 2 store down")

        storage = AsyncMock()
        storage.get_chunks_batch = _paged_fetch
        degradations: list[Any] = []

        kept = await filter_items_by_provenance(
            [item_a, item_b],
            _ast(_DATE_FILTER),
            namespace_id=ns_id,
            storage=storage,
            component="vectorcypher.entity_filter",
            degradations=degradations,
        )

        # Two fetch pages were attempted (page 1 succeeded, page 2 raised).
        assert calls["n"] == 2
        # A (decided on page 1) survives; B (pending on page 2) is dropped fail-closed.
        assert kept == [item_a]
        # Exactly one Degradation names the entity-filter component + reason.
        assert [(d.get("component"), d.get("reason")) for d in degradations] == [
            ("vectorcypher.entity_filter", "provenance_fetch_failed")
        ]


# ---------------------------------------------------------------------------
# chunk_record_adapter: the entity surface enforces the filter with the SAME
# field semantics its chunk channel uses. Chronicle's chunks carry their event
# time only in ``source_timestamp`` (raw ``occurred_at`` is NULL), so its
# ``_chunk_to_record`` COALESCE adapter is what lets an ``occurred_at`` predicate
# the chunk channel satisfied keep the provenance entity (GitHub #1458).
# ---------------------------------------------------------------------------


class TestChunkRecordAdapterCoalescesOccurredAt:
    """The real chronicle ``_chunk_to_record`` adapter aligns entity + chunk enforcement.

    Drives ``filter_items_by_provenance`` with chronicle's ACTUAL
    ``_chunk_to_record`` (not a stand-in) so the entity surface enforces
    ``occurred_at`` with the SAME ``COALESCE(occurred_at, source_timestamp)``
    semantics chronicle's chunk post-filter uses. Both directions have teeth:
    a source_timestamp-only chunk that CLEARS the horizon keeps its entity; a
    chunk whose event time (via either field) FAILS the horizon drops it.
    """

    async def test_source_timestamp_only_chunk_needs_adapter_to_survive(self) -> None:
        """A source_timestamp-only chunk drops under the raw predicate, survives via the adapter.

        This is the chronicle shape: ``chunk.occurred_at`` is NULL and the real
        event time lives in ``source_timestamp``. Without an adapter the raw chunk's
        ``occurred_at`` resolves to ``None`` -> the ``$gte`` predicate fails ->
        the entity false-drops even though the chunk channel (which reads via
        ``_chunk_to_record``) kept the chunk. The real ``_chunk_to_record`` adapter
        closes that gap.
        """
        from khora.engines.chronicle.engine import _chunk_to_record
        from khora.filter.provenance import filter_items_by_provenance

        ns_id = uuid4()
        # A chunk carrying its event time ONLY in source_timestamp (occurred_at NULL).
        chunk = Chunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=uuid4(),
            content="source_timestamp-only provenance chunk",
            occurred_at=None,
            source_timestamp=datetime(2027, 6, 1, tzinfo=UTC),
        )
        item = _ProvItem([chunk.id])

        storage = AsyncMock()
        storage.get_chunks_batch = AsyncMock(return_value={chunk.id: chunk})

        # Without an adapter: raw occurred_at is None -> the entity is dropped.
        dropped = await filter_items_by_provenance(
            [item],
            _ast(_DATE_FILTER),
            namespace_id=ns_id,
            storage=storage,
            component="chronicle.entity_filter",
            degradations=[],
        )
        assert dropped == [], "raw occurred_at=None should fail the $gte predicate without a COALESCE adapter"

        # With chronicle's real _chunk_to_record adapter: the entity survives, because
        # the record's occurred_at is COALESCE(None, source_timestamp) = 2027 >= 2027.
        kept = await filter_items_by_provenance(
            [item],
            _ast(_DATE_FILTER),
            namespace_id=ns_id,
            storage=storage,
            component="chronicle.entity_filter",
            degradations=[],
            chunk_record_adapter=_chunk_to_record,
        )
        assert kept == [item], "the COALESCE adapter should let the source_timestamp event time satisfy the filter"

    async def test_chunk_failing_both_fields_is_dropped_even_with_adapter(self) -> None:
        """A chunk whose event time fails the horizon (via either field) drops WITH the adapter.

        The adapter is not a blanket "keep everything": it only changes WHICH field
        supplies the event time, not whether the predicate is enforced. A chunk whose
        source_timestamp (COALESCE'd into occurred_at) is 2026 does NOT clear the 2027
        horizon, so its entity is correctly dropped — the enforcement still has teeth.
        """
        from khora.engines.chronicle.engine import _chunk_to_record
        from khora.filter.provenance import filter_items_by_provenance

        ns_id = uuid4()
        stale = Chunk(
            id=uuid4(),
            namespace_id=ns_id,
            document_id=uuid4(),
            content="pre-horizon provenance chunk",
            occurred_at=None,
            source_timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        )
        item = _ProvItem([stale.id])

        storage = AsyncMock()
        storage.get_chunks_batch = AsyncMock(return_value={stale.id: stale})

        kept = await filter_items_by_provenance(
            [item],
            _ast(_DATE_FILTER),
            namespace_id=ns_id,
            storage=storage,
            component="chronicle.entity_filter",
            degradations=[],
            chunk_record_adapter=_chunk_to_record,
        )
        assert kept == [], "a 2026 event time must not clear the 2027 horizon even through the COALESCE adapter"
