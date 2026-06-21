"""VectorCypher remember path emits entity.created / relationship.created hooks (#978).

Chronicle's shared ingest flow (``pipelines/flows/ingest.py``) dispatches
``ENTITY_CREATED`` / ``RELATIONSHIP_CREATED`` through ``storage.dispatch_hook``
at the entity/relationship persistence points. VectorCypher uses its own
``_run_skeleton_extraction`` write path, which historically never called
``dispatch_hook`` — so subscribers to ``entity.created`` / ``relationship.created``
saw zero callbacks on the default graph engine.

These tests drive ``_run_skeleton_extraction`` directly with stubbed storage +
mocked extraction (matching the harness style of ``test_chunk_entities_resolved``)
and assert the hooks fire, with created-vs-updated honoured from the
``upsert_entities_batch`` ``is_new`` flag (no double-fire on dedup-merge).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.core.models import Entity, Relationship
from khora.core.models.event import EventType
from khora.core.temporal import TemporalChunk


def _make_engine(upsert_results, *, rels_created: int = 0):
    """Build a minimally-wired VectorCypherEngine for _run_skeleton_extraction."""
    from khora.config import KhoraConfig
    from khora.engines.vectorcypher.engine import VectorCypherConfig, VectorCypherEngine

    engine = VectorCypherEngine.__new__(VectorCypherEngine)
    engine._config = KhoraConfig()
    engine._vc_config = VectorCypherConfig()

    storage = MagicMock()
    storage.upsert_entities_batch = AsyncMock(return_value=upsert_results)
    storage.create_relationships_batch = AsyncMock(return_value=rels_created)
    storage.dispatch_hook = AsyncMock()
    storage.get_entity_by_name = AsyncMock(return_value=None)
    engine._storage = storage

    engine._dual_nodes = None  # sqlite_lance / unified path: no MENTIONED_IN edges

    embedder = MagicMock()
    embedder.model_name = "fake-embed"
    embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts])
    engine._embedder = embedder

    return engine, storage


def _make_chunk(ns_id: UUID, doc_id: UUID, content: str = "Acme acquired Beta in 2025.") -> TemporalChunk:
    return TemporalChunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=doc_id,
        content=content,
        embedding=[0.1] * 8,
        occurred_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )


def _make_entity(ns_id: UUID, name: str, entity_type: str, chunk_id: UUID) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        description=f"{name} desc",
        source_chunk_ids=[chunk_id],
        confidence=0.9,
    )


def _events_of(storage_mock, event_type: EventType):
    return [c.args[0] for c in storage_mock.dispatch_hook.await_args_list if c.args[0].event_type == event_type]


@pytest.mark.unit
class TestVectorCypherSemanticHooks:
    @pytest.mark.asyncio
    async def test_remember_fires_entity_and_relationship_created(self) -> None:
        ns_id, doc_id = uuid4(), uuid4()
        chunk = _make_chunk(ns_id, doc_id)
        acme = _make_entity(ns_id, "Acme", "ORGANIZATION", chunk.id)
        beta = _make_entity(ns_id, "Beta", "ORGANIZATION", chunk.id)
        entities = [acme, beta]
        rel = Relationship(
            id=uuid4(),
            namespace_id=ns_id,
            source_entity_id=acme.id,
            target_entity_id=beta.id,
            relationship_type="ACQUIRED",
            confidence=0.9,
        )

        engine, storage = _make_engine([(acme, True), (beta, True)], rels_created=1)

        with patch(
            "khora.pipelines.tasks.extract.extract_entities",
            new=AsyncMock(return_value=(entities, [rel])),
        ):
            n_ent, n_rel = await engine._run_skeleton_extraction(
                [chunk],
                ns_id,
                entity_types=["ORGANIZATION"],
                relationship_types=["ACQUIRED"],
            )

        assert n_ent == 2
        assert n_rel == 1

        entity_events = _events_of(storage, EventType.ENTITY_CREATED)
        rel_events = _events_of(storage, EventType.RELATIONSHIP_CREATED)
        assert len(entity_events) == 2
        assert len(rel_events) == 1

        assert {e.resource_id for e in entity_events} == {acme.id, beta.id}
        names = {e.data["name"] for e in entity_events}
        assert names == {"Acme", "Beta"}
        for e in entity_events:
            assert e.resource_type == "entity"
            assert e.data["is_new"] is True
            assert e.data["document_id"] == str(doc_id)

        revt = rel_events[0]
        assert revt.resource_type == "relationship"
        assert revt.resource_id == rel.id
        assert revt.data["relationship_type"] == "ACQUIRED"
        assert revt.data["source_entity_id"] == str(acme.id)
        assert revt.data["target_entity_id"] == str(beta.id)

    @pytest.mark.asyncio
    async def test_dedup_merge_emits_entity_updated_not_created(self) -> None:
        """Re-ingesting an existing entity (is_new=False) fires entity.updated, not created."""
        ns_id, doc_id = uuid4(), uuid4()
        chunk = _make_chunk(ns_id, doc_id)
        acme = _make_entity(ns_id, "Acme", "ORGANIZATION", chunk.id)
        beta = _make_entity(ns_id, "Beta", "ORGANIZATION", chunk.id)

        # Acme already existed (is_new=False); Beta is genuinely new.
        engine, storage = _make_engine([(acme, False), (beta, True)])

        with patch(
            "khora.pipelines.tasks.extract.extract_entities",
            new=AsyncMock(return_value=([acme, beta], [])),
        ):
            await engine._run_skeleton_extraction(
                [chunk],
                ns_id,
                entity_types=["ORGANIZATION"],
                relationship_types=[],
            )

        created = _events_of(storage, EventType.ENTITY_CREATED)
        updated = _events_of(storage, EventType.ENTITY_UPDATED)
        assert {e.resource_id for e in created} == {beta.id}
        assert {e.resource_id for e in updated} == {acme.id}

    @pytest.mark.asyncio
    async def test_write_path_invokes_dispatch_hook(self) -> None:
        """dispatch_hook is the gate; a real no-subscriber coordinator short-circuits
        internally. Confirm the write path still routes through dispatch_hook."""
        ns_id, doc_id = uuid4(), uuid4()
        chunk = _make_chunk(ns_id, doc_id)
        acme = _make_entity(ns_id, "Acme", "ORGANIZATION", chunk.id)
        engine, storage = _make_engine([(acme, True)])

        with patch(
            "khora.pipelines.tasks.extract.extract_entities",
            new=AsyncMock(return_value=([acme], [])),
        ):
            await engine._run_skeleton_extraction(
                [chunk],
                ns_id,
                entity_types=["ORGANIZATION"],
                relationship_types=[],
            )

        assert storage.dispatch_hook.await_count >= 1
