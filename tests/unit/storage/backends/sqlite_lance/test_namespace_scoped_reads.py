"""IDOR guard: ``namespace_id``-tightened reads on the sqlite_lance backend.

Uses the real migrated-SQLite fixture so the recursive CTE and point lookups
are exercised end-to-end. Asserts that a query with the *wrong* namespace
returns the empty form (``None`` / ``[]`` / ``{...}`` with empty bags) and
that an in-namespace query returns the row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from khora.core.models import Entity, Episode, Relationship
from khora.core.models.event import EventType, MemoryEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(ns: UUID, *, name: str = "alice", entity_type: str = "PERSON") -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=ns,
        name=name,
        entity_type=entity_type,
        description="",
        attributes={},
        mention_count=1,
        confidence=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_relationship(ns: UUID, src: UUID, tgt: UUID, *, rel_type: str = "RELATES_TO") -> Relationship:
    return Relationship(
        id=uuid4(),
        namespace_id=ns,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description="",
        confidence=1.0,
        weight=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_event(ns: UUID, resource_id: UUID, resource_type: str = "document") -> MemoryEvent:
    return MemoryEvent(
        id=uuid4(),
        namespace_id=ns,
        event_type=EventType.DOCUMENT_CREATED,
        timestamp=datetime.now(UTC),
        resource_type=resource_type,
        resource_id=resource_id,
        data={},
        actor_id="test",
        actor_type="system",
        version=1,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Graph adapter — wrong-namespace reads return the empty form
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSQLiteLanceGraphNamespaceFilter:
    @pytest.fixture
    async def adapter(self, migrated_sqlite_db: Path, tmp_path: Path):
        from khora.storage.backends.sqlite_lance.connection import (
            EmbeddedStorageHandle,
            EmbeddedStorageHandleConfig,
        )
        from khora.storage.backends.sqlite_lance.graph import SQLiteLanceGraphAdapter

        cfg = EmbeddedStorageHandleConfig(
            db_path=str(migrated_sqlite_db),
            lance_path=str(tmp_path / "ns.lance"),
            embedding_dimension=8,
            use_halfvec=False,
        )
        h = EmbeddedStorageHandle(cfg)
        await h.connect()
        await h.sqlite.execute("PRAGMA foreign_keys = OFF")
        await h.sqlite.commit()
        a = SQLiteLanceGraphAdapter(h)
        await a.connect()
        yield a
        await h.disconnect()

    @pytest.mark.asyncio
    async def test_get_entity_wrong_namespace_returns_none(self, adapter) -> None:
        owner, attacker = uuid4(), uuid4()
        e = _make_entity(owner)
        await adapter.create_entity(e)

        assert await adapter.get_entity(e.id, namespace_id=owner) is not None
        assert await adapter.get_entity(e.id, namespace_id=attacker) is None

    @pytest.mark.asyncio
    async def test_get_relationship_wrong_namespace_returns_none(self, adapter) -> None:
        owner, attacker = uuid4(), uuid4()
        a, b = _make_entity(owner, name="a"), _make_entity(owner, name="b")
        await adapter.create_entity(a)
        await adapter.create_entity(b)
        r = _make_relationship(owner, a.id, b.id)
        await adapter.create_relationship(r)

        assert await adapter.get_relationship(r.id, namespace_id=owner) is not None
        assert await adapter.get_relationship(r.id, namespace_id=attacker) is None

    @pytest.mark.asyncio
    async def test_get_entity_relationships_wrong_namespace_returns_empty(self, adapter) -> None:
        owner, attacker = uuid4(), uuid4()
        a, b = _make_entity(owner, name="a"), _make_entity(owner, name="b")
        await adapter.create_entity(a)
        await adapter.create_entity(b)
        await adapter.create_relationship(_make_relationship(owner, a.id, b.id))

        good = await adapter.get_entity_relationships(a.id, namespace_id=owner)
        bad = await adapter.get_entity_relationships(a.id, namespace_id=attacker)
        assert len(good) == 1
        assert bad == []

    @pytest.mark.asyncio
    async def test_get_episode_wrong_namespace_returns_none(self, adapter) -> None:
        owner, attacker = uuid4(), uuid4()
        ep = Episode(
            id=uuid4(),
            namespace_id=owner,
            name="login",
            description="",
            occurred_at=datetime.now(UTC),
            duration_seconds=None,
            entity_ids=[],
            source_document_ids=[],
            source_chunk_ids=[],
            metadata={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        await adapter.create_episode(ep)

        assert await adapter.get_episode(ep.id, namespace_id=owner) is not None
        assert await adapter.get_episode(ep.id, namespace_id=attacker) is None

    @pytest.mark.asyncio
    async def test_neighborhood_does_not_cross_namespaces(self, adapter) -> None:
        """An attacker-namespace query against owner's seed returns nothing,
        and a misrouted edge stored in owner's namespace cannot be used to
        reach an entity belonging to a different namespace."""
        owner = uuid4()
        other = uuid4()
        attacker = uuid4()

        a = _make_entity(owner, name="A")
        b_other = _make_entity(other, name="B_OTHER")
        await adapter.create_entity(a)
        await adapter.create_entity(b_other)
        # Misrouted edge in owner's namespace that points to an entity in other.
        await adapter.create_relationship(_make_relationship(owner, a.id, b_other.id))

        # Attacker sees nothing.
        attacker_view = await adapter.get_neighborhood(a.id, namespace_id=attacker)
        assert attacker_view == {"entities": [], "relationships": []}

        # Owner can see the edge but the entity it points at is filtered out
        # by the defense-in-depth entity load.
        owner_view = await adapter.get_neighborhood(a.id, namespace_id=owner)
        ent_names = {e.name for e in owner_view["entities"]}
        assert "B_OTHER" not in ent_names

    @pytest.mark.asyncio
    async def test_neighborhoods_batch_wrong_namespace_returns_empty_per_seed(self, adapter) -> None:
        owner, attacker = uuid4(), uuid4()
        a, b = _make_entity(owner, name="a"), _make_entity(owner, name="b")
        await adapter.create_entity(a)
        await adapter.create_entity(b)
        await adapter.create_relationship(_make_relationship(owner, a.id, b.id))

        out = await adapter.get_neighborhoods_batch([a.id], namespace_id=attacker)
        # Seed key is preserved in the result dict but its bags are empty.
        assert out == {a.id: {"entities": [], "relationships": []}}


# ---------------------------------------------------------------------------
# Event store
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSQLiteLanceEventStoreNamespaceFilter:
    @pytest.fixture
    async def store(self, migrated_sqlite_db: Path, tmp_path: Path):
        from khora.storage.backends.sqlite_lance.connection import (
            EmbeddedStorageHandle,
            EmbeddedStorageHandleConfig,
        )
        from khora.storage.backends.sqlite_lance.event_store import (
            SQLiteLanceEventStoreAdapter,
        )

        cfg = EmbeddedStorageHandleConfig(
            db_path=str(migrated_sqlite_db),
            lance_path=str(tmp_path / "ev.lance"),
            embedding_dimension=8,
            use_halfvec=False,
        )
        h = EmbeddedStorageHandle(cfg)
        await h.connect()
        await h.sqlite.execute("PRAGMA foreign_keys = OFF")
        await h.sqlite.commit()
        s = SQLiteLanceEventStoreAdapter(h)
        await s.connect()
        yield s
        await h.disconnect()

    @pytest.mark.asyncio
    async def test_events_for_resource_wrong_namespace_returns_empty(self, store) -> None:
        owner, attacker = uuid4(), uuid4()
        resource = uuid4()
        await store.append_event(_make_event(owner, resource))

        good = await store.get_events_for_resource("document", resource, namespace_id=owner)
        bad = await store.get_events_for_resource("document", resource, namespace_id=attacker)

        assert len(good) == 1
        assert bad == []

    @pytest.mark.asyncio
    async def test_latest_event_wrong_namespace_returns_none(self, store) -> None:
        owner, attacker = uuid4(), uuid4()
        resource = uuid4()
        await store.append_event(_make_event(owner, resource))

        assert await store.get_latest_event("document", resource, namespace_id=owner) is not None
        assert await store.get_latest_event("document", resource, namespace_id=attacker) is None
