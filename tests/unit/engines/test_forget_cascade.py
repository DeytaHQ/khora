"""Unit tests for the shared forget-cascade (#923, review fixes).

Fast, no-Docker tests over mock stores. Cover the orphan/survivor refcount
predicates, the strip dispatch priority (batch > single > base), the
source-document-scoped lookup vs bounded scan path (+ scan-cap degradation),
and the silent-no-op fallbacks (a store that can list+delete but cannot strip
must surface a Degradation, not silently skip).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines._forget_cascade import (
    _SCAN_LIMIT,
    _is_orphan,
    _is_survivor,
    cascade_forget_extraction,
)


def _rec(rec_id, source_document_ids):
    return SimpleNamespace(id=rec_id, source_document_ids=list(source_document_ids))


# ---------------------------------------------------------------------------
# Refcount predicates
# ---------------------------------------------------------------------------


class TestPredicates:
    def test_orphan_none_sources(self) -> None:
        doc = uuid4()
        assert _is_orphan(SimpleNamespace(source_document_ids=None), doc) is False
        assert _is_survivor(SimpleNamespace(source_document_ids=None), doc) is False

    def test_orphan_empty_sources(self) -> None:
        doc = uuid4()
        assert _is_orphan(_rec(uuid4(), []), doc) is False
        assert _is_survivor(_rec(uuid4(), []), doc) is False

    def test_orphan_sole_source(self) -> None:
        doc = uuid4()
        rec = _rec(uuid4(), [doc])
        assert _is_orphan(rec, doc) is True
        assert _is_survivor(rec, doc) is False

    def test_survivor_multi_source(self) -> None:
        doc = uuid4()
        rec = _rec(uuid4(), [doc, uuid4()])
        assert _is_orphan(rec, doc) is False
        assert _is_survivor(rec, doc) is True

    def test_untouched_other_source_only(self) -> None:
        doc = uuid4()
        rec = _rec(uuid4(), [uuid4()])
        assert _is_orphan(rec, doc) is False
        assert _is_survivor(rec, doc) is False


# ---------------------------------------------------------------------------
# Store builders
# ---------------------------------------------------------------------------


def _pgvector_like(entities=None, relationships=None, *, source_scoped=False):
    """Vector store with the pgvector method surface (single-arg strip)."""
    spec = [
        "list_entities",
        "list_relationships",
        "delete_entities_batch",
        "delete_relationships_batch",
        "remove_document_from_entity_sources",
        "remove_document_from_relationship_sources",
    ]
    if source_scoped:
        spec += ["list_entities_by_source_document", "list_relationships_by_source_document"]
    store = MagicMock(spec=spec)
    store.list_entities = AsyncMock(return_value=list(entities or []))
    store.list_relationships = AsyncMock(return_value=list(relationships or []))
    store.delete_entities_batch = AsyncMock()
    store.delete_relationships_batch = AsyncMock()
    store.remove_document_from_entity_sources = AsyncMock()
    store.remove_document_from_relationship_sources = AsyncMock()
    if source_scoped:
        store.list_entities_by_source_document = AsyncMock(return_value=list(entities or []))
        store.list_relationships_by_source_document = AsyncMock(return_value=list(relationships or []))
    return store


def _neo4j_like():
    """Mirror graph store with the Neo4j batch-strip surface."""
    store = MagicMock(
        spec=[
            "delete_entities_batch",
            "delete_relationships_batch",
            "remove_document_from_entity_sources_batch",
            "remove_document_from_relationship_sources_batch",
        ]
    )
    store.delete_entities_batch = AsyncMock()
    store.delete_relationships_batch = AsyncMock()
    store.remove_document_from_entity_sources_batch = AsyncMock()
    store.remove_document_from_relationship_sources_batch = AsyncMock()
    return store


def _base_graph_like(entities=None, relationships=None):
    """GraphBackendBase-style store: list+delete+strip_document_from_* fallback."""
    store = MagicMock(
        spec=[
            "list_entities",
            "list_relationships",
            "delete_entities_batch",
            "delete_relationships_batch",
            "strip_document_from_entities",
            "strip_document_from_relationships",
        ]
    )
    store.list_entities = AsyncMock(return_value=list(entities or []))
    store.list_relationships = AsyncMock(return_value=list(relationships or []))
    store.delete_entities_batch = AsyncMock()
    store.delete_relationships_batch = AsyncMock()
    store.strip_document_from_entities = AsyncMock()
    store.strip_document_from_relationships = AsyncMock()
    return store


def _list_delete_only(entities=None, relationships=None):
    """Store that can list+delete but has NO source-strip primitive at all."""
    store = MagicMock(
        spec=[
            "list_entities",
            "list_relationships",
            "delete_entities_batch",
            "delete_relationships_batch",
        ]
    )
    store.list_entities = AsyncMock(return_value=list(entities or []))
    store.list_relationships = AsyncMock(return_value=list(relationships or []))
    store.delete_entities_batch = AsyncMock()
    store.delete_relationships_batch = AsyncMock()
    return store


# ---------------------------------------------------------------------------
# Orphan / survivor handling
# ---------------------------------------------------------------------------


class TestOrphanSurvivor:
    @pytest.mark.asyncio
    async def test_orphan_entity_deleted_both_stores(self) -> None:
        doc, ns, eid = uuid4(), uuid4(), uuid4()
        vector = _pgvector_like(entities=[_rec(eid, [doc])])
        graph = _neo4j_like()

        deg = await cascade_forget_extraction(graph=graph, vector=vector, document_id=doc, namespace_id=ns, engine="t")

        assert deg == []
        vector.delete_entities_batch.assert_awaited_once_with([eid], namespace_id=ns)
        graph.delete_entities_batch.assert_awaited_once_with([eid], namespace_id=ns)
        vector.remove_document_from_entity_sources.assert_not_called()

    @pytest.mark.asyncio
    async def test_survivor_entity_stripped_with_namespace(self) -> None:
        doc, ns, eid = uuid4(), uuid4(), uuid4()
        vector = _pgvector_like(entities=[_rec(eid, [doc, uuid4()])])
        graph = _neo4j_like()

        deg = await cascade_forget_extraction(graph=graph, vector=vector, document_id=doc, namespace_id=ns, engine="t")

        assert deg == []
        vector.delete_entities_batch.assert_not_called()
        vector.remove_document_from_entity_sources.assert_awaited_once_with([eid], doc, ns)
        graph.remove_document_from_entity_sources_batch.assert_awaited_once_with([eid], doc, ns)

    @pytest.mark.asyncio
    async def test_untouched_entity_not_modified(self) -> None:
        doc, ns = uuid4(), uuid4()
        vector = _pgvector_like(entities=[_rec(uuid4(), [uuid4()])])
        graph = _neo4j_like()

        deg = await cascade_forget_extraction(graph=graph, vector=vector, document_id=doc, namespace_id=ns, engine="t")

        assert deg == []
        vector.delete_entities_batch.assert_not_called()
        vector.remove_document_from_entity_sources.assert_not_called()
        graph.remove_document_from_entity_sources_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_store_with_primitives_degrades(self) -> None:
        doc, ns = uuid4(), uuid4()
        # Graph has no cleanup helpers; vector is None.
        graph = MagicMock(spec=["something_else"])
        deg = await cascade_forget_extraction(graph=graph, vector=None, document_id=doc, namespace_id=ns, engine="t")
        assert len(deg) == 1
        assert deg[0]["reason"] == "no_store_with_primitives"


# ---------------------------------------------------------------------------
# Strip dispatch priority: batch > single > base
# ---------------------------------------------------------------------------


class TestStripDispatchPriority:
    @pytest.mark.asyncio
    async def test_prefers_batch_over_single(self) -> None:
        doc, ns, eid = uuid4(), uuid4(), uuid4()
        # A store exposing BOTH batch and single must use batch.
        store = MagicMock(
            spec=[
                "list_entities",
                "list_relationships",
                "delete_entities_batch",
                "delete_relationships_batch",
                "remove_document_from_entity_sources_batch",
                "remove_document_from_entity_sources",
                "remove_document_from_relationship_sources_batch",
                "remove_document_from_relationship_sources",
            ]
        )
        store.list_entities = AsyncMock(return_value=[_rec(eid, [doc, uuid4()])])
        store.list_relationships = AsyncMock(return_value=[])
        store.delete_entities_batch = AsyncMock()
        store.delete_relationships_batch = AsyncMock()
        store.remove_document_from_entity_sources_batch = AsyncMock()
        store.remove_document_from_entity_sources = AsyncMock()
        store.remove_document_from_relationship_sources_batch = AsyncMock()
        store.remove_document_from_relationship_sources = AsyncMock()

        await cascade_forget_extraction(graph=None, vector=store, document_id=doc, namespace_id=ns, engine="t")

        store.remove_document_from_entity_sources_batch.assert_awaited_once_with([eid], doc, ns)
        store.remove_document_from_entity_sources.assert_not_called()

    @pytest.mark.asyncio
    async def test_base_strip_fallback_used(self) -> None:
        doc, ns, eid = uuid4(), uuid4(), uuid4()
        graph = _base_graph_like(entities=[_rec(eid, [doc, uuid4()])])

        deg = await cascade_forget_extraction(graph=graph, vector=None, document_id=doc, namespace_id=ns, engine="t")

        assert deg == []
        graph.strip_document_from_entities.assert_awaited_once_with([eid], doc, namespace_id=ns)


# ---------------------------------------------------------------------------
# Silent-no-op fallback (M4): list+delete but no strip -> Degradation
# ---------------------------------------------------------------------------


class TestStripUnsupportedDegrades:
    @pytest.mark.asyncio
    async def test_survivor_entity_no_strip_primitive_degrades(self) -> None:
        doc, ns, eid = uuid4(), uuid4(), uuid4()
        store = _list_delete_only(entities=[_rec(eid, [doc, uuid4()])])

        deg = await cascade_forget_extraction(graph=None, vector=store, document_id=doc, namespace_id=ns, engine="t")

        assert len(deg) == 1
        assert deg[0]["reason"] == "strip_unsupported"
        store.delete_entities_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_strip_exception_degrades(self) -> None:
        doc, ns, eid = uuid4(), uuid4(), uuid4()
        store = _pgvector_like(entities=[_rec(eid, [doc, uuid4()])])
        store.remove_document_from_entity_sources = AsyncMock(side_effect=RuntimeError("boom"))

        deg = await cascade_forget_extraction(graph=None, vector=store, document_id=doc, namespace_id=ns, engine="t")

        assert len(deg) == 1
        assert deg[0]["reason"] == "strip_failed"
        assert "boom" in deg[0]["exception"]


# ---------------------------------------------------------------------------
# Source-document-scoped lookup vs bounded scan + cap degradation (H1)
# ---------------------------------------------------------------------------


class TestCandidateSelection:
    @pytest.mark.asyncio
    async def test_prefers_source_scoped_lookup(self) -> None:
        doc, ns, eid = uuid4(), uuid4(), uuid4()
        vector = _pgvector_like(entities=[_rec(eid, [doc])], source_scoped=True)

        await cascade_forget_extraction(graph=None, vector=vector, document_id=doc, namespace_id=ns, engine="t")

        vector.list_entities_by_source_document.assert_awaited_once_with(ns, doc)
        vector.list_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_bounded_scan_used_when_no_source_lookup(self) -> None:
        doc, ns = uuid4(), uuid4()
        vector = _pgvector_like(entities=[])

        await cascade_forget_extraction(graph=None, vector=vector, document_id=doc, namespace_id=ns, engine="t")

        vector.list_entities.assert_awaited_once_with(ns, limit=_SCAN_LIMIT)

    @pytest.mark.asyncio
    async def test_scan_cap_hit_degrades(self) -> None:
        doc, ns = uuid4(), uuid4()
        # Return exactly _SCAN_LIMIT entities (all untouched) -> cap hit warning.
        capped = [_rec(uuid4(), [uuid4()]) for _ in range(_SCAN_LIMIT)]
        vector = _pgvector_like(entities=capped)

        deg = await cascade_forget_extraction(graph=None, vector=vector, document_id=doc, namespace_id=ns, engine="t")

        reasons = {d["reason"] for d in deg}
        assert "scan_cap_hit" in reasons
