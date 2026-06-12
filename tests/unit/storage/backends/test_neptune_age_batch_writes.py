"""Regression tests for #1149 - Neptune/AGE batch graph writes.

StorageCoordinator gates ``upsert_entities_batch`` /
``create_relationships_batch`` behind ``hasattr`` checks. NeptuneBackend and
AGEBackend defined neither method, so on the standard ingest path entities
were written only to the pgvector mirror and relationships were written
NOWHERE - the coordinator silently returned 0 / synthetic results.

The fix adds default N+1 implementations to ``GraphBackendBase`` built on
the per-item primitives every graph backend already provides
(``get_entity_by_name`` / ``create_entity`` / ``update_entity`` /
``create_relationship``), honoring the #806 id-remap contract: an input
entity matched to an existing row gets its ``id`` synced in place so
relationship endpoints built from extraction-time ids still resolve.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Entity, Relationship
from khora.storage.backends.age import AGEBackend
from khora.storage.backends.mixins import GraphBackendBase
from khora.storage.backends.neptune import NeptuneBackend
from khora.storage.coordinator import StorageCoordinator

_NS = uuid4()


# ---------------------------------------------------------------------------
# In-memory stub backend exercising the GraphBackendBase defaults
# ---------------------------------------------------------------------------


class _StubGraphBackend(GraphBackendBase):
    """Minimal per-item-only graph backend (the Neptune/AGE shape)."""

    def __init__(self) -> None:
        self.entities: dict[tuple[UUID, str, str], Entity] = {}
        self.relationships: list[Relationship] = []
        self.probe_calls = 0

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        self.probe_calls += 1
        return self.entities.get((namespace_id, name, entity_type))

    async def create_entity(self, entity: Entity) -> Entity:
        self.entities[(entity.namespace_id, entity.name, entity.entity_type)] = entity
        return entity

    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> Entity:
        self.entities[(namespace_id, entity.name, entity.entity_type)] = entity
        return entity

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        self.relationships.append(relationship)
        return relationship


def _entity(name: str, **kwargs: Any) -> Entity:
    return Entity(namespace_id=_NS, name=name, entity_type="PERSON", **kwargs)


# ---------------------------------------------------------------------------
# GraphBackendBase default semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_default_upsert_creates_new_entities() -> None:
    backend = _StubGraphBackend()
    entities = [_entity("Alice"), _entity("Bob")]

    results = await backend.upsert_entities_batch(_NS, entities)

    assert results == [(entities[0], True), (entities[1], True)]
    assert len(backend.entities) == 2


@pytest.mark.unit
async def test_default_upsert_merges_existing_and_syncs_id() -> None:
    """The #806 id-remap contract: on match, the INPUT entity's id is synced
    to the stored id so relationships built from extraction-time ids resolve."""
    backend = _StubGraphBackend()
    doc_a, doc_b = uuid4(), uuid4()
    existing = _entity(
        "Alice",
        description="short",
        mention_count=2,
        confidence=0.9,
        source_document_ids=[doc_a],
    )
    await backend.create_entity(existing)

    incoming = _entity(
        "Alice",
        description="a much longer description",
        mention_count=3,
        confidence=0.5,
        source_document_ids=[doc_a, doc_b],
        attributes={"role": "engineer"},
    )
    assert incoming.id != existing.id

    results = await backend.upsert_entities_batch(_NS, [incoming])

    assert results == [(incoming, False)]
    # Input entity id synced in place to the stored id (#806).
    assert incoming.id == existing.id
    stored = backend.entities[(_NS, "Alice", "PERSON")]
    assert stored.id == existing.id
    assert stored.description == "a much longer description"  # longest wins
    assert stored.mention_count == 5  # summed
    assert stored.confidence == 0.9  # max wins
    assert stored.source_document_ids == [doc_a, doc_b]  # union, no dupes
    assert stored.attributes == {"role": "engineer"}  # input replaces


@pytest.mark.unit
async def test_default_upsert_bulk_mode_skips_existence_probe() -> None:
    backend = _StubGraphBackend()

    results = await backend.upsert_entities_batch(_NS, [_entity("Alice")], bulk_mode=True)

    assert results[0][1] is True
    assert backend.probe_calls == 0


@pytest.mark.unit
async def test_default_relationships_batch_writes_every_relationship() -> None:
    backend = _StubGraphBackend()
    rels = [Relationship(namespace_id=_NS) for _ in range(3)]

    count = await backend.create_relationships_batch(rels)

    assert count == 3
    assert backend.relationships == rels


@pytest.mark.unit
async def test_default_relationships_batch_failure_raises_not_silent() -> None:
    """A mid-batch failure must propagate (#868 pattern), never silently drop."""
    backend = _StubGraphBackend()
    rels = [Relationship(namespace_id=_NS), Relationship(namespace_id=_NS)]

    original = backend.create_relationship

    async def _fail_on_second(rel: Relationship) -> Relationship:
        if rel is rels[1]:
            raise RuntimeError("edge write failed")
        return await original(rel)

    backend.create_relationship = _fail_on_second  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="edge write failed"):
        await backend.create_relationships_batch(rels)

    assert backend.relationships == [rels[0]]


# ---------------------------------------------------------------------------
# Coordinator ingest batch path against stubbed Neptune / AGE backends
# ---------------------------------------------------------------------------


def _stub_per_item_methods(backend: Any) -> None:
    """Stub the per-item primitives so no driver / DB is needed."""
    backend.get_entity_by_name = AsyncMock(return_value=None)
    backend.create_entity = AsyncMock(side_effect=lambda e: e)
    backend.create_relationship = AsyncMock(side_effect=lambda r: r)


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_backend",
    [
        lambda: NeptuneBackend("bolt://cluster:8182"),
        lambda: AGEBackend(database_url="postgresql://localhost/test"),
    ],
    ids=["neptune", "age"],
)
async def test_coordinator_batch_upsert_reaches_backend(make_backend: Any) -> None:
    """Every entity on the coordinator batch path must reach the graph backend.

    On main, NeptuneBackend / AGEBackend had no ``upsert_entities_batch`` so
    the coordinator's ``hasattr`` gate skipped the graph entirely and
    fabricated synthetic ``(entity, True)`` results - silent data loss.
    """
    backend = make_backend()
    _stub_per_item_methods(backend)
    coordinator = StorageCoordinator(graph=backend)
    entities = [_entity("Alice"), _entity("Bob"), _entity("Carol")]

    results = await coordinator.upsert_entities_batch(_NS, entities)

    assert len(results) == 3
    created = {call.args[0].name for call in backend.create_entity.await_args_list}
    assert created == {"Alice", "Bob", "Carol"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_backend",
    [
        lambda: NeptuneBackend("bolt://cluster:8182"),
        lambda: AGEBackend(database_url="postgresql://localhost/test"),
    ],
    ids=["neptune", "age"],
)
async def test_coordinator_batch_relationships_reach_backend(make_backend: Any) -> None:
    """Every relationship on the coordinator batch path must reach the backend.

    On main, the coordinator's ``hasattr`` gate failed for Neptune / AGE and
    silently returned 0 - relationships were written NOWHERE.
    """
    backend = make_backend()
    _stub_per_item_methods(backend)
    coordinator = StorageCoordinator(graph=backend)
    rels = [Relationship(namespace_id=_NS) for _ in range(3)]

    count = await coordinator.create_relationships_batch(rels)

    assert count == 3
    written = [call.args[0] for call in backend.create_relationship.await_args_list]
    assert written == rels


# ---------------------------------------------------------------------------
# Real backend wiring through mocked drivers (no per-item stubbing)
# ---------------------------------------------------------------------------


def _make_neptune_backend() -> tuple[NeptuneBackend, AsyncMock]:
    """NeptuneBackend with a mocked bolt driver (existing coverage pattern)."""
    result = MagicMock()
    result.data = AsyncMock(return_value=[])
    result.single = AsyncMock(return_value=None)
    session = AsyncMock()
    session.run = AsyncMock(return_value=result)

    driver = MagicMock()

    @asynccontextmanager
    async def _session_ctx():  # type: ignore[no-untyped-def]
        yield session

    driver.session = MagicMock(side_effect=_session_ctx)
    backend = NeptuneBackend("bolt://cluster:8182")
    backend._driver = driver
    return backend, session


@pytest.mark.unit
async def test_neptune_batch_methods_issue_cypher() -> None:
    backend, session = _make_neptune_backend()

    results = await backend.upsert_entities_batch(_NS, [_entity("Alice"), _entity("Bob")])
    assert [is_new for _, is_new in results] == [True, True]
    creates = [c.args[0] for c in session.run.await_args_list if "CREATE (e:Entity" in c.args[0]]
    assert len(creates) == 2

    session.run.reset_mock()
    count = await backend.create_relationships_batch([Relationship(namespace_id=_NS), Relationship(namespace_id=_NS)])
    assert count == 2
    rel_creates = [c.args[0] for c in session.run.await_args_list if "CREATE (source)-[r:" in c.args[0]]
    assert len(rel_creates) == 2


def _make_age_backend() -> tuple[AGEBackend, AsyncMock]:
    """AGEBackend with a mocked SQLAlchemy session (existing coverage pattern)."""
    result = MagicMock()
    result.fetchall = MagicMock(return_value=[])
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)

    @asynccontextmanager
    async def _begin():  # type: ignore[no-untyped-def]
        yield None

    session.begin = MagicMock(side_effect=_begin)

    backend = AGEBackend(database_url="postgresql://localhost/test")

    @asynccontextmanager
    async def _session_ctx():  # type: ignore[no-untyped-def]
        yield session

    backend._session_factory = MagicMock(side_effect=_session_ctx)  # type: ignore[assignment]
    return backend, session


@pytest.mark.unit
async def test_age_batch_methods_issue_cypher() -> None:
    backend, session = _make_age_backend()

    results = await backend.upsert_entities_batch(_NS, [_entity("Alice"), _entity("Bob")])
    assert [is_new for _, is_new in results] == [True, True]
    statements = [str(c.args[0]) for c in session.execute.await_args_list]
    creates = [s for s in statements if "CREATE (e:Entity" in s]
    assert len(creates) == 2

    session.execute.reset_mock()
    count = await backend.create_relationships_batch([Relationship(namespace_id=_NS), Relationship(namespace_id=_NS)])
    assert count == 2
    statements = [str(c.args[0]) for c in session.execute.await_args_list]
    rel_creates = [s for s in statements if "CREATE (source)-[r:" in s]
    assert len(rel_creates) == 2
