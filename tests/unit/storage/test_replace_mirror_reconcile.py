"""Unit tests for the #1430 replace graph-mirror reconciler.

Covers the three legs of the durable-recovery contract:

1. A post-PG-commit graph failure persists the computed graph plan on
   ``documents.graph_mirror_pending`` (and reports ``pending_persisted``
   on the typed #884 exception).
2. ``reconcile_replace_graph_mirror`` replays the persisted plan against
   the graph and clears the marker on success; a still-failing plan stays
   queued and surfaces an ADR-001 degradation.
3. The drain runs at the start of the next ``replace_document_extraction``
   in the namespace - the same trigger shape as the dream reconciler
   (#1272) - and a successful replace clears any stale marker in-tx.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Document, Entity, Relationship
from khora.exceptions import GraphMirrorFailedAfterPGCommitError
from khora.storage.coordinator import StorageCoordinator
from khora.storage.replace_mirror import (
    apply_replace_mirror_payload,
    build_replace_mirror_payload,
)


class _FakeTxnSession:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False
        self.closed = False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        self.closed = True


def _relational_backend() -> MagicMock:
    rel = MagicMock()
    rel.update_document = AsyncMock()
    rel.partial_update_document = AsyncMock(return_value=1)
    rel.list_documents_with_graph_mirror_pending = AsyncMock(return_value=[])
    return rel


def _vector_backend() -> MagicMock:
    vec = MagicMock()
    vec.delete_chunks_by_document = AsyncMock(return_value=0)
    vec.create_chunks_batch = AsyncMock(return_value=[])
    # Route entity upserts through the coordinator's graph-only branch and
    # skip the embedding write-through (both probed via hasattr).
    del vec.upsert_entities_batch
    del vec.update_entity_embeddings_batch
    return vec


def _graph_backend(*, old_entity_records: list[dict] | None = None) -> MagicMock:
    graph = MagicMock()
    graph.fetch_document_extraction_state = AsyncMock(return_value=(old_entity_records or [], []))
    graph.retire_orphaned_entities_batch = AsyncMock(return_value=1)
    graph.retire_orphaned_relationships_batch = AsyncMock(return_value=1)
    graph.remap_source_document_ids_batch = AsyncMock(return_value=None)
    graph.remove_document_from_entity_sources_batch = AsyncMock(return_value=None)
    graph.remove_document_from_relationship_sources_batch = AsyncMock(return_value=None)
    graph.upsert_entities_batch = AsyncMock(side_effect=lambda ns, ents, **kw: [(e, True) for e in ents])
    graph.create_relationships_batch = AsyncMock(side_effect=lambda rels, **kw: [(r, True) for r in rels])
    return graph


def _make_coordinator(
    *, relational: MagicMock, vector: MagicMock, graph: MagicMock
) -> tuple[StorageCoordinator, _FakeTxnSession]:
    session = _FakeTxnSession()
    relational._session_factory = lambda: session
    coord = StorageCoordinator(relational=relational, vector=vector, graph=graph)
    return coord, session


def _orphan_entity_record(namespace_id) -> dict:
    return {
        "id": str(uuid4()),
        "name": "bob",
        "entity_type": "PERSON",
        "namespace_id": str(namespace_id),
        "source_document_count": 1,
    }


@pytest.mark.unit
class TestPendingMarkerWrite:
    @pytest.mark.asyncio
    async def test_graph_failure_persists_pending_marker(self) -> None:
        """A post-commit graph failure durably queues the computed plan."""
        namespace_id = uuid4()
        old_doc_id = uuid4()
        new_doc = Document(namespace_id=namespace_id, content="new body")
        new_chunks = [Chunk(namespace_id=namespace_id, document_id=new_doc.id)]
        net_new_entity = Entity(
            namespace_id=namespace_id,
            name="carol",
            entity_type="PERSON",
            embedding=[0.1, 0.2, 0.3],
            source_document_ids=[new_doc.id],
        )
        net_new_rel = Relationship(
            namespace_id=namespace_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="KNOWS",
        )

        rel_backend = _relational_backend()
        vec_backend = _vector_backend()
        graph_backend = _graph_backend(old_entity_records=[_orphan_entity_record(namespace_id)])
        graph_backend.retire_orphaned_entities_batch = AsyncMock(side_effect=RuntimeError("neo4j down"))

        coord, session = _make_coordinator(relational=rel_backend, vector=vec_backend, graph=graph_backend)

        with pytest.raises(GraphMirrorFailedAfterPGCommitError) as exc_info:
            await coord.replace_document_extraction(
                namespace_id=namespace_id,
                old_document_id=old_doc_id,
                new_document=new_doc,
                new_chunks=new_chunks,
                new_entities=[net_new_entity],
                new_relationships=[net_new_rel],
            )

        assert exc_info.value.pending_persisted is True
        # PG committed; marker write is the LAST partial_update call (the
        # first is the in-tx stale-marker clear).
        assert session.committed is True
        last_call = rel_backend.partial_update_document.await_args_list[-1]
        assert last_call.args == (new_doc.id,)
        payload = last_call.kwargs["graph_mirror_pending"]
        assert payload["version"] == 1
        assert payload["old_document_id"] == str(old_doc_id)
        assert payload["exception"] == "RuntimeError"
        assert len(payload["entity_retirement_rows"]) == 1
        assert payload["net_new_entities"][0]["name"] == "carol"
        assert payload["net_new_entities"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert payload["net_new_relationships"][0]["relationship_type"] == "KNOWS"
        # The payload must be JSON-serializable (it goes into a JSONB column).
        json.dumps(payload)
        # In-memory Document mirrors the marker for callers that hold the row.
        assert new_doc.graph_mirror_pending == payload

    @pytest.mark.asyncio
    async def test_marker_write_failure_degrades_to_884_contract(self) -> None:
        """A failed marker write must not mask the graph failure."""
        namespace_id = uuid4()
        new_doc = Document(namespace_id=namespace_id, content="new body")
        new_chunks = [Chunk(namespace_id=namespace_id, document_id=new_doc.id)]

        rel_backend = _relational_backend()
        # In-tx clear succeeds; the post-failure marker write raises.
        rel_backend.partial_update_document = AsyncMock(side_effect=[1, RuntimeError("pg down too")])
        vec_backend = _vector_backend()
        graph_backend = _graph_backend(old_entity_records=[_orphan_entity_record(namespace_id)])
        graph_backend.retire_orphaned_entities_batch = AsyncMock(side_effect=RuntimeError("neo4j down"))

        coord, _session = _make_coordinator(relational=rel_backend, vector=vec_backend, graph=graph_backend)

        with pytest.raises(GraphMirrorFailedAfterPGCommitError) as exc_info:
            await coord.replace_document_extraction(
                namespace_id=namespace_id,
                old_document_id=uuid4(),
                new_document=new_doc,
                new_chunks=new_chunks,
                new_entities=[],
                new_relationships=[],
            )

        assert exc_info.value.pending_persisted is False
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "neo4j down"
        assert new_doc.graph_mirror_pending is None


@pytest.mark.unit
class TestReconcileReplaceGraphMirror:
    def _pending_document(self, namespace_id) -> tuple[Document, dict]:
        old_doc_id = uuid4()
        net_new = Entity(
            namespace_id=namespace_id,
            name="carol",
            entity_type="PERSON",
            embedding=[0.5, 0.6],
            source_document_ids=[uuid4()],
        )
        payload = build_replace_mirror_payload(
            old_document_id=old_doc_id,
            entity_retirement_rows=[
                {
                    "current_id": str(uuid4()),
                    "snapshot_id": str(uuid4()),
                    "namespace_id": str(namespace_id),
                    "retired_at": "2026-07-06T00:00:00+00:00",
                }
            ],
            relationship_retirement_rows=[],
            entity_survivor_remap_rows=[],
            relationship_survivor_remap_rows=[],
            entity_survivor_strip_ids=[uuid4()],
            relationship_survivor_strip_ids=[],
            net_new_entities=[net_new],
            net_new_relationships=[],
            exception=RuntimeError("original failure"),
        )
        doc = Document(namespace_id=namespace_id, content="body", graph_mirror_pending=payload)
        return doc, payload

    @pytest.mark.asyncio
    async def test_reconcile_replays_and_clears_marker(self) -> None:
        namespace_id = uuid4()
        doc, payload = self._pending_document(namespace_id)

        rel_backend = _relational_backend()
        rel_backend.list_documents_with_graph_mirror_pending = AsyncMock(return_value=[doc])
        vec_backend = _vector_backend()
        graph_backend = _graph_backend()

        coord, _ = _make_coordinator(relational=rel_backend, vector=vec_backend, graph=graph_backend)

        degradations = await coord.reconcile_replace_graph_mirror(namespace_id)

        assert degradations == []
        # Replay hit the persisted retire rows and net-new entity upsert.
        graph_backend.retire_orphaned_entities_batch.assert_awaited_once_with(payload["entity_retirement_rows"])
        strip_call = graph_backend.remove_document_from_entity_sources_batch.await_args
        assert [str(u) for u in strip_call.args[0]] == payload["entity_survivor_strip_ids"]
        upsert_call = graph_backend.upsert_entities_batch.await_args
        replayed_entity = upsert_call.args[1][0]
        assert replayed_entity.name == "carol"
        assert replayed_entity.embedding == [0.5, 0.6]
        # Marker cleared.
        rel_backend.partial_update_document.assert_awaited_once_with(
            doc.id, namespace_id=namespace_id, graph_mirror_pending=None
        )

    @pytest.mark.asyncio
    async def test_reconcile_failure_keeps_marker_and_degrades(self) -> None:
        namespace_id = uuid4()
        doc, _payload = self._pending_document(namespace_id)

        rel_backend = _relational_backend()
        rel_backend.list_documents_with_graph_mirror_pending = AsyncMock(return_value=[doc])
        vec_backend = _vector_backend()
        graph_backend = _graph_backend()
        graph_backend.retire_orphaned_entities_batch = AsyncMock(side_effect=RuntimeError("still down"))

        coord, _ = _make_coordinator(relational=rel_backend, vector=vec_backend, graph=graph_backend)

        degradations = await coord.reconcile_replace_graph_mirror(namespace_id)

        assert len(degradations) == 1
        assert degradations[0]["component"] == "coordinator.replace_mirror.reconcile"
        assert degradations[0]["reason"] == "graph_mirror_reconcile_failed"
        assert degradations[0]["exception"] == "RuntimeError"
        # Marker NOT cleared - stays queued for the next drain.
        rel_backend.partial_update_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reconcile_never_raises_on_pending_read_failure(self) -> None:
        namespace_id = uuid4()
        rel_backend = _relational_backend()
        rel_backend.list_documents_with_graph_mirror_pending = AsyncMock(side_effect=RuntimeError("pg read failed"))
        coord, _ = _make_coordinator(relational=rel_backend, vector=_vector_backend(), graph=_graph_backend())

        degradations = await coord.reconcile_replace_graph_mirror(namespace_id)

        assert len(degradations) == 1
        assert degradations[0]["reason"] == "graph_mirror_pending_read_failed"

    @pytest.mark.asyncio
    async def test_reconcile_skips_when_relational_lacks_capability(self) -> None:
        """Non-PG relational backends (no marker support) drain to a no-op."""
        namespace_id = uuid4()
        rel_backend = _relational_backend()
        del rel_backend.list_documents_with_graph_mirror_pending
        coord, _ = _make_coordinator(relational=rel_backend, vector=_vector_backend(), graph=_graph_backend())

        assert await coord.reconcile_replace_graph_mirror(namespace_id) == []


@pytest.mark.unit
class TestDrainAtReplaceStart:
    @pytest.mark.asyncio
    async def test_replace_drains_prior_pending_before_prefetch(self) -> None:
        """The next replace in the namespace replays a prior failure's plan
        first (same trigger shape as the dream reconciler at apply start)."""
        namespace_id = uuid4()
        prior_doc = Document(
            namespace_id=namespace_id,
            content="prior",
            graph_mirror_pending=build_replace_mirror_payload(
                old_document_id=uuid4(),
                entity_retirement_rows=[],
                relationship_retirement_rows=[],
                entity_survivor_remap_rows=[],
                relationship_survivor_remap_rows=[],
                entity_survivor_strip_ids=[],
                relationship_survivor_strip_ids=[],
                net_new_entities=[Entity(namespace_id=namespace_id, name="dave", entity_type="PERSON")],
                net_new_relationships=[],
                exception=RuntimeError("prior failure"),
            ),
        )

        rel_backend = _relational_backend()
        rel_backend.list_documents_with_graph_mirror_pending = AsyncMock(return_value=[prior_doc])
        vec_backend = _vector_backend()
        graph_backend = _graph_backend()

        call_order: list[str] = []
        graph_backend.upsert_entities_batch = AsyncMock(
            side_effect=lambda ns, ents, **kw: (call_order.append(f"upsert:{ents[0].name}"), [(e, True) for e in ents])[
                1
            ]
        )
        graph_backend.fetch_document_extraction_state = AsyncMock(
            side_effect=lambda *a, **kw: (call_order.append("prefetch"), ([], []))[1]
        )

        coord, session = _make_coordinator(relational=rel_backend, vector=vec_backend, graph=graph_backend)

        new_doc = Document(namespace_id=namespace_id, content="new body")
        result = await coord.replace_document_extraction(
            namespace_id=namespace_id,
            old_document_id=uuid4(),
            new_document=new_doc,
            new_chunks=[],
            new_entities=[Entity(namespace_id=namespace_id, name="erin", entity_type="PERSON")],
            new_relationships=[],
        )

        # Drain (dave, from the prior payload) ran before this replace's
        # graph prefetch; the replace's own upsert (erin) came after.
        assert call_order == ["upsert:dave", "prefetch", "upsert:erin"]
        assert result.degradations == []
        # Prior marker cleared post-replay; current doc's stale-marker slot
        # cleared in-tx alongside the new content.
        cleared = [
            c
            for c in rel_backend.partial_update_document.await_args_list
            if c.kwargs.get("graph_mirror_pending") is None
        ]
        assert {c.args[0] for c in cleared} == {prior_doc.id, new_doc.id}
        in_tx = [c for c in cleared if c.args[0] == new_doc.id]
        assert in_tx[0].kwargs["session"] is session

    @pytest.mark.asyncio
    async def test_drain_failure_rides_the_exception_when_own_mirror_also_fails(self) -> None:
        """Compound failure: the drain degrades AND the current replace's own
        graph mirror fails. The failure path returns no ReplaceResult, so the
        drain degradations must ride GraphMirrorFailedAfterPGCommitError."""
        namespace_id = uuid4()
        prior_doc = Document(
            namespace_id=namespace_id,
            content="prior",
            graph_mirror_pending=build_replace_mirror_payload(
                old_document_id=uuid4(),
                entity_retirement_rows=[
                    {
                        "current_id": str(uuid4()),
                        "snapshot_id": str(uuid4()),
                        "namespace_id": str(namespace_id),
                        "retired_at": "2026-07-06T00:00:00+00:00",
                    }
                ],
                relationship_retirement_rows=[],
                entity_survivor_remap_rows=[],
                relationship_survivor_remap_rows=[],
                entity_survivor_strip_ids=[],
                relationship_survivor_strip_ids=[],
                net_new_entities=[],
                net_new_relationships=[],
                exception=RuntimeError("prior failure"),
            ),
        )

        rel_backend = _relational_backend()
        rel_backend.list_documents_with_graph_mirror_pending = AsyncMock(return_value=[prior_doc])
        vec_backend = _vector_backend()
        # Retire fails for the drained payload AND for the current replace's
        # own mirror (its prefetch reports an orphan below).
        graph_backend = _graph_backend(old_entity_records=[_orphan_entity_record(namespace_id)])
        graph_backend.retire_orphaned_entities_batch = AsyncMock(side_effect=RuntimeError("still down"))

        coord, _ = _make_coordinator(relational=rel_backend, vector=vec_backend, graph=graph_backend)

        with pytest.raises(GraphMirrorFailedAfterPGCommitError) as exc_info:
            await coord.replace_document_extraction(
                namespace_id=namespace_id,
                old_document_id=uuid4(),
                new_document=Document(namespace_id=namespace_id, content="new body"),
                new_chunks=[],
                new_entities=[],
                new_relationships=[],
            )

        assert len(exc_info.value.drain_degradations) == 1
        assert exc_info.value.drain_degradations[0]["reason"] == "graph_mirror_reconcile_failed"

    @pytest.mark.asyncio
    async def test_drain_failure_surfaces_on_replace_result(self) -> None:
        """A still-failing prior marker degrades the current ReplaceResult
        instead of failing the (independent) replace."""
        namespace_id = uuid4()
        prior_doc = Document(
            namespace_id=namespace_id,
            content="prior",
            graph_mirror_pending=build_replace_mirror_payload(
                old_document_id=uuid4(),
                entity_retirement_rows=[
                    {
                        "current_id": str(uuid4()),
                        "snapshot_id": str(uuid4()),
                        "namespace_id": str(namespace_id),
                        "retired_at": "2026-07-06T00:00:00+00:00",
                    }
                ],
                relationship_retirement_rows=[],
                entity_survivor_remap_rows=[],
                relationship_survivor_remap_rows=[],
                entity_survivor_strip_ids=[],
                relationship_survivor_strip_ids=[],
                net_new_entities=[],
                net_new_relationships=[],
                exception=RuntimeError("prior failure"),
            ),
        )

        rel_backend = _relational_backend()
        rel_backend.list_documents_with_graph_mirror_pending = AsyncMock(return_value=[prior_doc])
        vec_backend = _vector_backend()
        graph_backend = _graph_backend()
        # Retire fails only for the drained payload (the current replace has
        # no retirement rows because prefetch returns empty state).
        graph_backend.retire_orphaned_entities_batch = AsyncMock(side_effect=RuntimeError("still down"))

        coord, _ = _make_coordinator(relational=rel_backend, vector=vec_backend, graph=graph_backend)

        result = await coord.replace_document_extraction(
            namespace_id=namespace_id,
            old_document_id=uuid4(),
            new_document=Document(namespace_id=namespace_id, content="new body"),
            new_chunks=[],
            new_entities=[],
            new_relationships=[],
        )

        assert len(result.degradations) == 1
        assert result.degradations[0]["reason"] == "graph_mirror_reconcile_failed"


@pytest.mark.unit
class TestPayloadVersionGate:
    @pytest.mark.asyncio
    async def test_apply_rejects_unknown_payload_version(self) -> None:
        """A payload written by an unknown schema version must not run graph
        mutations - it stays queued and degrades at the reconcile layer."""
        graph = _graph_backend()
        coord = StorageCoordinator(relational=MagicMock(), vector=None, graph=graph)
        payload = {"version": 999, "old_document_id": str(uuid4())}

        with pytest.raises(ValueError, match="payload version"):
            await apply_replace_mirror_payload(coord, payload, namespace_id=uuid4())

        graph.retire_orphaned_entities_batch.assert_not_awaited()
        graph.upsert_entities_batch.assert_not_awaited()


@pytest.mark.unit
class TestPayloadRoundTrip:
    @pytest.mark.asyncio
    async def test_apply_reconstructs_types_from_json(self) -> None:
        """UUID / datetime / embedding fidelity across the JSON round-trip
        (the payload is persisted to JSONB and read back as plain dicts)."""
        namespace_id = uuid4()
        old_doc_id = uuid4()
        rel_id = uuid4()
        entity = Entity(
            namespace_id=namespace_id,
            name="carol",
            entity_type="PERSON",
            embedding=[0.1, 0.2],
            source_document_ids=[uuid4()],
            source_chunk_ids=[uuid4()],
            confidence=0.9,
        )
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="KNOWS",
            weight=0.7,
        )
        payload = build_replace_mirror_payload(
            old_document_id=old_doc_id,
            entity_retirement_rows=[],
            relationship_retirement_rows=[
                {
                    "relationship_id": rel_id,
                    "old_doc_id": old_doc_id,
                    "retired_at": datetime(2026, 7, 6, tzinfo=UTC),
                }
            ],
            entity_survivor_remap_rows=[],
            relationship_survivor_remap_rows=[],
            entity_survivor_strip_ids=[],
            relationship_survivor_strip_ids=[],
            net_new_entities=[entity],
            net_new_relationships=[relationship],
            exception=ValueError("boom"),
        )
        # Simulate the JSONB round-trip.
        payload = json.loads(json.dumps(payload))

        graph = _graph_backend()
        coord = StorageCoordinator(relational=MagicMock(), vector=None, graph=graph)

        counts = await apply_replace_mirror_payload(coord, payload, namespace_id=namespace_id)

        retire_rows = graph.retire_orphaned_relationships_batch.await_args.args[0]
        assert retire_rows[0]["relationship_id"] == rel_id
        assert retire_rows[0]["old_doc_id"] == old_doc_id
        assert retire_rows[0]["retired_at"].year == 2026

        replayed_entity = graph.upsert_entities_batch.await_args.args[1][0]
        assert replayed_entity.id == entity.id
        assert replayed_entity.namespace_id == namespace_id
        assert replayed_entity.embedding == [0.1, 0.2]
        assert replayed_entity.source_document_ids == entity.source_document_ids
        assert replayed_entity.source_chunk_ids == entity.source_chunk_ids

        replayed_rel = graph.create_relationships_batch.await_args.args[0][0]
        assert replayed_rel.id == relationship.id
        assert replayed_rel.source_entity_id == relationship.source_entity_id
        assert replayed_rel.weight == 0.7

        assert counts == {
            "entities_retired": 0,
            "relationships_retired": 1,
            "entities_upserted": 1,
            "relationships_created": 1,
        }
