"""Protocol compliance matrix for storage backends.

Runs all storage backends through the same parameterized test matrix to
prove feature parity with the four core protocols:

* :class:`RelationalBackendProtocol`
* :class:`GraphBackendProtocol`
* :class:`VectorBackendProtocol`
* :class:`EventStoreProtocol`

**v1 scope**: the matrix parameterizes over ``sqlite_lance`` only. The
fixture structure is ready to extend — adding Postgres / Neo4j / SurrealDB
is a matter of new backend-specific fixtures and extra ``params=[...]``
entries on the protocol-level fixtures. Full cross-backend compliance runs
in integration tests with live services; this file keeps the
unit suite hermetic.

The existing per-adapter tests in ``tests/unit/storage/backends/sqlite_lance/``
cover implementation details. This suite complements them by asserting the
*protocol contract*: edge cases (empty namespaces, missing IDs, unicode,
pagination boundaries, etc.) that must hold regardless of the backend
behind the interface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:  # pragma: no cover — skip path
    _HAS_EMBEDDED = False

pytestmark = pytest.mark.skipif(
    not _HAS_EMBEDDED,
    reason="sqlite_lance backend unavailable (aiosqlite/lancedb not installed)",
)


if _HAS_EMBEDDED:
    from khora.core.models import (
        Chunk,
        ChunkMetadata,
        Document,
        DocumentMetadata,
        Entity,
        Episode,
        MemoryNamespace,
        Relationship,
        TenancyMode,
    )
    from khora.core.models.document import DocumentStatus
    from khora.core.models.event import EventType, MemoryEvent
    from khora.db.session import run_migrations
    from khora.storage.backends.base import (
        EventStoreProtocol,
        GraphBackendProtocol,
        RelationalBackendProtocol,
        VectorBackendProtocol,
    )
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )
    from khora.storage.backends.sqlite_lance.event_store import (
        SQLiteLanceEventStoreAdapter,
    )
    from khora.storage.backends.sqlite_lance.graph import SQLiteLanceGraphAdapter
    from khora.storage.backends.sqlite_lance.relational import (
        SQLiteLanceRelationalAdapter,
    )
    from khora.storage.backends.sqlite_lance.vector import SQLiteLanceVectorAdapter


# ---------------------------------------------------------------------------
# Shared DB setup
# ---------------------------------------------------------------------------


async def _migrate(db_path: Path) -> None:
    """Run Alembic migrations against ``db_path``.

    The adapters speak the exact schema the migrations produce
    (``chunks.metadata``, no ``embedding`` column on ``chunks`` /
    ``entities``, external-content FTS5 with triggers, 32-char UUID
    hex) — no reshape shim required.
    """
    url = f"sqlite+aiosqlite:///{db_path}"
    result = await run_migrations(url)
    assert result.success, f"migrations failed: {result.error}"


# ---------------------------------------------------------------------------
# Backend-specific fixtures (one per role × backend)
# ---------------------------------------------------------------------------


@pytest.fixture
async def sqlite_lance_handle(tmp_path: Path) -> AsyncIterator[EmbeddedStorageHandle]:
    """Shared embedded handle backed by the real Alembic-migrated schema."""
    db_path = tmp_path / "compliance.db"
    lance_path = tmp_path / "compliance.lance"
    await _migrate(db_path)

    cfg = EmbeddedStorageHandleConfig(
        db_path=str(db_path),
        lance_path=str(lance_path),
        embedding_dimension=8,
        use_halfvec=False,
    )
    handle = EmbeddedStorageHandle(cfg)
    await handle.connect()
    # Protocol tests use bare namespace/document UUIDs without seeding
    # the parent rows.  FK enforcement is exercised in integration tests
    # (test_sqlite_lance_ingest.py / _fk_enforcement) against a coordinator
    # that creates namespaces through the relational adapter.
    await handle.sqlite.execute("PRAGMA foreign_keys = OFF")
    await handle.sqlite.commit()
    try:
        yield handle
    finally:
        await handle.disconnect()


@pytest.fixture
async def sqlite_lance_relational(
    sqlite_lance_handle: EmbeddedStorageHandle,
) -> AsyncIterator[SQLiteLanceRelationalAdapter]:
    adapter = SQLiteLanceRelationalAdapter(sqlite_lance_handle)
    await adapter.connect()
    try:
        yield adapter
    finally:
        await adapter.disconnect()


@pytest.fixture
async def sqlite_lance_graph(
    sqlite_lance_handle: EmbeddedStorageHandle,
) -> AsyncIterator[SQLiteLanceGraphAdapter]:
    adapter = SQLiteLanceGraphAdapter(sqlite_lance_handle)
    await adapter.connect()
    try:
        yield adapter
    finally:
        await adapter.disconnect()


@pytest.fixture
async def sqlite_lance_vector(
    sqlite_lance_handle: EmbeddedStorageHandle,
) -> AsyncIterator[SQLiteLanceVectorAdapter]:
    adapter = SQLiteLanceVectorAdapter(sqlite_lance_handle)
    await adapter.connect()
    try:
        yield adapter
    finally:
        await adapter.disconnect()


@pytest.fixture
async def sqlite_lance_event_store(
    sqlite_lance_handle: EmbeddedStorageHandle,
) -> AsyncIterator[SQLiteLanceEventStoreAdapter]:
    adapter = SQLiteLanceEventStoreAdapter(sqlite_lance_handle)
    try:
        yield adapter
    finally:
        # EventStoreAdapter borrows the shared handle, no adapter teardown
        # of its own.
        pass


# ---------------------------------------------------------------------------
# Protocol-level parameterized fixtures
#
# To add another backend (e.g. postgres), register a backend-specific
# fixture above and extend ``params=[...]`` here. Gate availability with
# ``pytest.skip`` inside the branch when the driver/service is missing.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["sqlite_lance"])
async def relational_backend(
    request: pytest.FixtureRequest, sqlite_lance_relational: SQLiteLanceRelationalAdapter
) -> SQLiteLanceRelationalAdapter:
    if request.param == "sqlite_lance":
        return sqlite_lance_relational
    raise AssertionError(f"unknown backend: {request.param}")  # pragma: no cover


@pytest.fixture(params=["sqlite_lance"])
async def graph_backend(
    request: pytest.FixtureRequest, sqlite_lance_graph: SQLiteLanceGraphAdapter
) -> SQLiteLanceGraphAdapter:
    if request.param == "sqlite_lance":
        return sqlite_lance_graph
    raise AssertionError(f"unknown backend: {request.param}")  # pragma: no cover


@pytest.fixture(params=["sqlite_lance"])
async def vector_backend(
    request: pytest.FixtureRequest, sqlite_lance_vector: SQLiteLanceVectorAdapter
) -> SQLiteLanceVectorAdapter:
    if request.param == "sqlite_lance":
        return sqlite_lance_vector
    raise AssertionError(f"unknown backend: {request.param}")  # pragma: no cover


@pytest.fixture
async def entity_seeder(vector_backend, sqlite_lance_graph: SQLiteLanceGraphAdapter):
    """Factory that seeds an entity through both graph and vector adapters.

    On sqlite_lance, entity SQLite rows are graph-owned; the vector
    adapter only writes the LanceDB embedding.  ``entity_exists`` /
    ``update_entity_embedding`` read SQLite, so the protocol tests that
    call ``vector.create_entity`` and then expect the row to exist must
    seed via the graph adapter too.  On unified or entity-owning
    backends (e.g. pgvector), the graph call is a harmless duplicate.
    """

    async def _seed(entity):
        await sqlite_lance_graph.create_entity(entity)
        await vector_backend.create_entity(entity)

    return _seed


@pytest.fixture(params=["sqlite_lance"])
async def event_store_backend(
    request: pytest.FixtureRequest, sqlite_lance_event_store: SQLiteLanceEventStoreAdapter
) -> SQLiteLanceEventStoreAdapter:
    if request.param == "sqlite_lance":
        return sqlite_lance_event_store
    raise AssertionError(f"unknown backend: {request.param}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(dim: int, idx: int = 0) -> list[float]:
    v = [0.0] * dim
    v[idx % dim] = 1.0
    return v


def _make_namespace() -> MemoryNamespace:
    nid = uuid4()
    return MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED)


def _make_document(
    namespace_id: UUID,
    *,
    checksum: str = "abc",
    title: str = "Doc",
    content: str = "hello world",
) -> Document:
    return Document(
        namespace_id=namespace_id,
        content=content,
        metadata=DocumentMetadata(
            source="file:///tmp/a.txt",
            source_type="file",
            title=title,
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
        ),
    )


def _make_entity(
    namespace_id: UUID,
    *,
    name: str = "alice",
    entity_type: str = "PERSON",
    attributes: dict | None = None,
    embedding: list[float] | None = None,
) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=namespace_id,
        name=name,
        entity_type=entity_type,
        description="",
        attributes=attributes or {},
        embedding=embedding,
        embedding_model="test-model" if embedding else "",
        mention_count=1,
        confidence=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_relationship(
    namespace_id: UUID,
    src: UUID,
    tgt: UUID,
    *,
    rel_type: str = "RELATES_TO",
    description: str = "",
) -> Relationship:
    return Relationship(
        id=uuid4(),
        namespace_id=namespace_id,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description=description,
        confidence=1.0,
        weight=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_chunk(
    namespace_id: UUID,
    document_id: UUID,
    *,
    content: str = "test content",
    embedding: list[float] | None = None,
    index: int = 0,
    created_at: datetime | None = None,
) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=document_id,
        content=content,
        metadata=ChunkMetadata(document_id=document_id, chunk_index=index),
        embedding=embedding,
        embedding_model="test-model" if embedding else "",
        created_at=created_at or datetime.now(UTC),
    )


def _make_event(
    namespace_id: UUID,
    *,
    event_type: EventType | None = None,
    resource_id: UUID | None = None,
    timestamp: datetime | None = None,
    data: dict | None = None,
) -> MemoryEvent:
    et = event_type if event_type is not None else EventType.DOCUMENT_CREATED
    return MemoryEvent(
        id=uuid4(),
        namespace_id=namespace_id,
        event_type=et,
        timestamp=timestamp or datetime.now(UTC),
        resource_type=et.value.split(".")[0],
        resource_id=resource_id or uuid4(),
        data=data or {"k": "v"},
    )


# ===========================================================================
# Relational protocol
# ===========================================================================


class TestRelationalProtocol:
    """Contract tests for :class:`RelationalBackendProtocol`."""

    async def test_runtime_protocol_membership(self, relational_backend):
        assert isinstance(relational_backend, RelationalBackendProtocol)

    async def test_is_healthy(self, relational_backend):
        assert await relational_backend.is_healthy() is True

    async def test_create_get_namespace(self, relational_backend):
        ns = _make_namespace()
        created = await relational_backend.create_namespace(ns)
        assert created.id == ns.id

        fetched = await relational_backend.get_namespace(ns.id)
        assert fetched is not None
        assert fetched.namespace_id == ns.namespace_id
        assert fetched.is_active is True

    async def test_get_namespace_missing_returns_none(self, relational_backend):
        assert await relational_backend.get_namespace(uuid4()) is None

    async def test_list_namespaces_pagination(self, relational_backend):
        created_ids: list[UUID] = []
        for _ in range(5):
            ns = _make_namespace()
            await relational_backend.create_namespace(ns)
            created_ids.append(ns.id)

        all_page = await relational_backend.list_namespaces(limit=10, offset=0)
        assert all_page.total == 5
        assert len(all_page.items) == 5

        first_page = await relational_backend.list_namespaces(limit=2, offset=0)
        assert len(first_page.items) == 2
        assert first_page.total == 5

        last_page = await relational_backend.list_namespaces(limit=2, offset=4)
        assert len(last_page.items) == 1
        assert last_page.total == 5

    async def test_namespace_version_bump(self, relational_backend):
        ns = _make_namespace()
        await relational_backend.create_namespace(ns)

        v2 = await relational_backend.create_namespace_version(previous_version=ns)
        assert v2.version == 2
        assert v2.is_active is True
        assert v2.namespace_id == ns.namespace_id
        assert v2.id != ns.id

        prior = await relational_backend.get_namespace(ns.id)
        assert prior is not None
        assert prior.is_active is False

        # resolve_namespace returns the ACTIVE row id.
        resolved = await relational_backend.resolve_namespace(ns.namespace_id)
        assert resolved == v2.id

    async def test_resolve_namespace_missing_raises(self, relational_backend):
        with pytest.raises(ValueError):
            await relational_backend.resolve_namespace(uuid4())

    async def test_document_crud(self, relational_backend):
        ns = _make_namespace()
        await relational_backend.create_namespace(ns)
        doc = _make_document(ns.id, checksum="crud", title="Before")
        created = await relational_backend.create_document(doc)
        assert created.id == doc.id

        fetched = await relational_backend.get_document(doc.id)
        assert fetched is not None
        assert fetched.metadata.title == "Before"

        fetched.metadata.title = "After"
        fetched.status = DocumentStatus.COMPLETED
        await relational_backend.update_document(fetched)

        refreshed = await relational_backend.get_document(doc.id)
        assert refreshed is not None
        assert refreshed.metadata.title == "After"
        assert refreshed.status == DocumentStatus.COMPLETED

        assert await relational_backend.delete_document(doc.id) is True
        assert await relational_backend.get_document(doc.id) is None
        # Idempotent — second delete is a no-op returning False.
        assert await relational_backend.delete_document(doc.id) is False

    async def test_document_checksum_dedup(self, relational_backend):
        ns = _make_namespace()
        await relational_backend.create_namespace(ns)
        doc = _make_document(ns.id, checksum="dedup-me")
        await relational_backend.create_document(doc)

        hit = await relational_backend.get_document_by_checksum(ns.id, "dedup-me")
        assert hit is not None
        assert hit.id == doc.id

        miss = await relational_backend.get_document_by_checksum(ns.id, "does-not-exist")
        assert miss is None

    async def test_get_documents_batch(self, relational_backend):
        ns = _make_namespace()
        await relational_backend.create_namespace(ns)
        docs = [_make_document(ns.id, checksum=f"b{i}") for i in range(3)]
        for d in docs:
            await relational_backend.create_document(d)

        result = await relational_backend.get_documents_batch([d.id for d in docs] + [uuid4()])
        assert set(result.keys()) == {d.id for d in docs}

        # Empty input is a no-op.
        assert await relational_backend.get_documents_batch([]) == {}

    async def test_get_documents_by_checksums(self, relational_backend):
        ns = _make_namespace()
        await relational_backend.create_namespace(ns)
        await relational_backend.create_document(_make_document(ns.id, checksum="ck1"))
        await relational_backend.create_document(_make_document(ns.id, checksum="ck2"))

        matches = await relational_backend.get_documents_by_checksums(ns.id, ["ck1", "ck2", "missing"])
        assert set(matches.keys()) == {"ck1", "ck2"}

        assert await relational_backend.get_documents_by_checksums(ns.id, []) == {}

    async def test_sync_checkpoint_roundtrip(self, relational_backend):
        ns = _make_namespace()
        await relational_backend.create_namespace(ns)

        # Missing checkpoint returns None.
        assert await relational_backend.get_sync_checkpoint(ns.id, "github") is None

        await relational_backend.set_sync_checkpoint(ns.id, "github", "cursor-1")
        assert await relational_backend.get_sync_checkpoint(ns.id, "github") == "cursor-1"

        # Upsert — must not create a duplicate row.
        await relational_backend.set_sync_checkpoint(ns.id, "github", "cursor-2")
        assert await relational_backend.get_sync_checkpoint(ns.id, "github") == "cursor-2"

        # Different source lives in its own slot.
        await relational_backend.set_sync_checkpoint(ns.id, "notion", "ntn-1")
        assert await relational_backend.get_sync_checkpoint(ns.id, "notion") == "ntn-1"
        assert await relational_backend.get_sync_checkpoint(ns.id, "github") == "cursor-2"

    async def test_count_documents_and_stats(self, relational_backend):
        ns = _make_namespace()
        await relational_backend.create_namespace(ns)

        # Empty namespace contract.
        assert await relational_backend.count_documents(ns.id) == 0
        assert await relational_backend.get_last_activity_at(ns.id) is None
        count, last = await relational_backend.get_document_stats(ns.id)
        assert count == 0 and last is None

        await relational_backend.create_document(_make_document(ns.id, checksum="x"))
        assert await relational_backend.count_documents(ns.id) == 1
        last_activity = await relational_backend.get_last_activity_at(ns.id)
        assert last_activity is not None
        assert isinstance(last_activity, datetime)

    async def test_edge_case_empty_namespace(self, relational_backend):
        ns = _make_namespace()
        await relational_backend.create_namespace(ns)
        assert await relational_backend.list_documents(ns.id) == []
        assert await relational_backend.count_documents(ns.id) == 0

    async def test_edge_case_nonexistent_id(self, relational_backend):
        assert await relational_backend.get_document(uuid4()) is None
        assert await relational_backend.delete_document(uuid4()) is False

    async def test_unicode_content_roundtrip(self, relational_backend):
        ns = _make_namespace()
        await relational_backend.create_namespace(ns)
        payload = "héllo ✨ 世界 \n\t 你好 — naïve"
        doc = _make_document(ns.id, checksum="u", title=payload, content=payload)
        await relational_backend.create_document(doc)

        fetched = await relational_backend.get_document(doc.id)
        assert fetched is not None
        assert fetched.content == payload
        assert fetched.metadata.title == payload


# ===========================================================================
# Graph protocol
# ===========================================================================


class TestGraphProtocol:
    """Contract tests for :class:`GraphBackendProtocol`."""

    async def test_runtime_protocol_membership(self, graph_backend):
        assert isinstance(graph_backend, GraphBackendProtocol)

    async def test_is_healthy(self, graph_backend):
        assert await graph_backend.is_healthy() is True

    async def test_entity_crud(self, graph_backend):
        ns = uuid4()
        e = _make_entity(ns, name="Alice", attributes={"role": "eng"})
        await graph_backend.create_entity(e)

        fetched = await graph_backend.get_entity(e.id)
        assert fetched is not None
        assert fetched.id == e.id
        assert fetched.attributes["role"] == "eng"

        e.description = "updated"
        await graph_backend.update_entity(e)
        refreshed = await graph_backend.get_entity(e.id)
        assert refreshed is not None and refreshed.description == "updated"

        assert await graph_backend.delete_entity(e.id) is True
        assert await graph_backend.get_entity(e.id) is None
        # Deleting again is a no-op.
        assert await graph_backend.delete_entity(e.id) is False

    async def test_get_entity_by_name(self, graph_backend):
        ns = uuid4()
        e = _make_entity(ns, name="Bob", entity_type="PERSON")
        await graph_backend.create_entity(e)

        hit = await graph_backend.get_entity_by_name(ns, "Bob", "PERSON")
        assert hit is not None and hit.id == e.id

        # Wrong type → miss.
        assert await graph_backend.get_entity_by_name(ns, "Bob", "ORG") is None
        # Wrong name → miss.
        assert await graph_backend.get_entity_by_name(ns, "Missing", "PERSON") is None

    async def test_upsert_entities_batch_new_vs_existing(self, graph_backend):
        ns = uuid4()

        # Seed 3 existing.
        seed = [_make_entity(ns, name=f"S{i}") for i in range(3)]
        seeded = await graph_backend.upsert_entities_batch(ns, seed)
        assert all(is_new for _, is_new in seeded)

        # Mix 2 new + 3 colliding-by-name/type.
        new = [_make_entity(ns, name=f"N{i}") for i in range(2)]
        collide = [_make_entity(ns, name=f"S{i}") for i in range(3)]
        results = await graph_backend.upsert_entities_batch(ns, new + collide)

        new_flags = [flag for _, flag in results]
        assert new_flags[:2] == [True, True]
        assert new_flags[2:] == [False, False, False]

        # Total row count is 5 — MERGE semantics preserved.
        assert await graph_backend.count_entities(ns) == 5

    async def test_upsert_entities_batch_empty(self, graph_backend):
        assert await graph_backend.upsert_entities_batch(uuid4(), []) == []

    async def test_relationship_crud(self, graph_backend):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await graph_backend.create_entity(a)
        await graph_backend.create_entity(b)

        r = _make_relationship(ns, a.id, b.id, rel_type="KNOWS")
        await graph_backend.create_relationship(r)

        fetched = await graph_backend.get_relationship(r.id)
        assert fetched is not None
        assert fetched.relationship_type == "KNOWS"

        assert await graph_backend.delete_relationship(r.id) is True
        assert await graph_backend.get_relationship(r.id) is None
        assert await graph_backend.delete_relationship(r.id) is False

    async def test_get_entity_relationships_directions(self, graph_backend):
        ns = uuid4()
        hub = _make_entity(ns, name="Hub")
        n1 = _make_entity(ns, name="N1")
        n2 = _make_entity(ns, name="N2")
        for e in (hub, n1, n2):
            await graph_backend.create_entity(e)
        await graph_backend.create_relationship(_make_relationship(ns, hub.id, n1.id))
        await graph_backend.create_relationship(_make_relationship(ns, n2.id, hub.id))

        out = await graph_backend.get_entity_relationships(hub.id, direction="outgoing")
        assert len(out) == 1
        inc = await graph_backend.get_entity_relationships(hub.id, direction="incoming")
        assert len(inc) == 1
        both = await graph_backend.get_entity_relationships(hub.id, direction="both")
        assert len(both) == 2

    async def test_find_paths(self, graph_backend):
        ns = uuid4()
        a = _make_entity(ns, name="a")
        b = _make_entity(ns, name="b")
        c = _make_entity(ns, name="c")
        for e in (a, b, c):
            await graph_backend.create_entity(e)
        await graph_backend.create_relationship(_make_relationship(ns, a.id, b.id))
        await graph_backend.create_relationship(_make_relationship(ns, b.id, c.id))

        paths = await graph_backend.find_paths(ns, a.id, c.id, max_depth=3)
        assert len(paths) == 1
        assert len(paths[0]) == 2

        # Disconnected — no path.
        lone = _make_entity(ns, name="lone")
        await graph_backend.create_entity(lone)
        assert await graph_backend.find_paths(ns, a.id, lone.id, max_depth=3) == []

    async def test_get_neighborhood_depths(self, graph_backend):
        ns = uuid4()
        entities = [_make_entity(ns, name=f"n{i}") for i in range(4)]
        for e in entities:
            await graph_backend.create_entity(e)
        # n0 -> n1 -> n2 -> n3
        for i in range(3):
            await graph_backend.create_relationship(_make_relationship(ns, entities[i].id, entities[i + 1].id))

        nb1 = await graph_backend.get_neighborhood(entities[0].id, depth=1, limit=10)
        names_d1 = {e.name for e in nb1["entities"]}
        assert names_d1 == {"n1"}

        nb2 = await graph_backend.get_neighborhood(entities[0].id, depth=2, limit=10)
        names_d2 = {e.name for e in nb2["entities"]}
        assert {"n1", "n2"}.issubset(names_d2)
        assert "n3" not in names_d2

    async def test_get_neighborhoods_batch_no_n_plus_one(self, graph_backend):
        """Batched neighborhood override must not degenerate to per-seed walks."""
        ns = uuid4()
        hub_a = _make_entity(ns, name="hub_a")
        hub_b = _make_entity(ns, name="hub_b")
        n1 = _make_entity(ns, name="n1")
        n2 = _make_entity(ns, name="n2")
        for e in (hub_a, hub_b, n1, n2):
            await graph_backend.create_entity(e)
        await graph_backend.create_relationship(_make_relationship(ns, hub_a.id, n1.id))
        await graph_backend.create_relationship(_make_relationship(ns, hub_b.id, n2.id))

        result = await graph_backend.get_neighborhoods_batch([hub_a.id, hub_b.id], depth=1, limit_per_entity=10)
        assert set(result.keys()) == {hub_a.id, hub_b.id}
        assert len(result[hub_a.id]["entities"]) == 1
        assert len(result[hub_b.id]["entities"]) == 1

        # Empty batch is a no-op.
        assert await graph_backend.get_neighborhoods_batch([]) == {}

    async def test_search_entities_by_attribute(self, graph_backend):
        ns = uuid4()
        await graph_backend.create_entity(_make_entity(ns, name="Eng1", attributes={"role": "eng"}))
        await graph_backend.create_entity(_make_entity(ns, name="PM1", attributes={"role": "pm"}))

        engs = await graph_backend.search_entities_by_attribute(ns, "role", "eng")
        assert len(engs) == 1
        assert engs[0].name == "Eng1"

        assert await graph_backend.search_entities_by_attribute(ns, "role", "ceo") == []

    async def test_episode_crud(self, graph_backend):
        ns = uuid4()
        ep = Episode(
            id=uuid4(),
            namespace_id=ns,
            name="Kickoff",
            occurred_at=datetime.now(UTC),
            duration_seconds=600,
            entity_ids=[uuid4()],
        )
        await graph_backend.create_episode(ep)

        fetched = await graph_backend.get_episode(ep.id)
        assert fetched is not None
        assert fetched.name == "Kickoff"
        assert fetched.duration_seconds == 600

        listed = await graph_backend.list_episodes(ns, limit=10)
        assert any(e.id == ep.id for e in listed)

        # Missing episode returns None.
        assert await graph_backend.get_episode(uuid4()) is None

    async def test_relationship_label_sanitization(self, graph_backend):
        """Cypher label sanitization must hold for all graph backends."""
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await graph_backend.create_entity(a)
        await graph_backend.create_entity(b)

        r = _make_relationship(ns, a.id, b.id, rel_type="likes; DROP TABLE")
        await graph_backend.create_relationship(r)

        fetched = await graph_backend.get_relationship(r.id)
        assert fetched is not None
        # At minimum, the sanitized label must be upper-case and contain
        # no SQL metacharacters that could escape a quoted identifier.
        assert fetched.relationship_type.isupper()
        assert ";" not in fetched.relationship_type
        assert " " not in fetched.relationship_type

    async def test_count_entities_and_relationships(self, graph_backend):
        ns = uuid4()
        assert await graph_backend.count_entities(ns) == 0
        assert await graph_backend.count_relationships(ns) == 0

        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await graph_backend.create_entity(a)
        await graph_backend.create_entity(b)
        await graph_backend.create_relationship(_make_relationship(ns, a.id, b.id))

        assert await graph_backend.count_entities(ns) == 2
        assert await graph_backend.count_relationships(ns) == 1

        # Isolation — a different namespace reports zero.
        other_ns = uuid4()
        assert await graph_backend.count_entities(other_ns) == 0

    async def test_edge_case_unicode_entity_name(self, graph_backend):
        ns = uuid4()
        name = "naïve café — 你好"
        e = _make_entity(ns, name=name)
        await graph_backend.create_entity(e)

        fetched = await graph_backend.get_entity_by_name(ns, name, "PERSON")
        assert fetched is not None
        assert fetched.name == name


# ===========================================================================
# Vector protocol
# ===========================================================================


class TestVectorProtocol:
    """Contract tests for :class:`VectorBackendProtocol`."""

    async def test_runtime_protocol_membership(self, vector_backend):
        assert isinstance(vector_backend, VectorBackendProtocol)

    async def test_is_healthy(self, vector_backend):
        assert await vector_backend.is_healthy() is True

    async def test_chunk_crud(self, vector_backend):
        ns, doc = uuid4(), uuid4()
        c = _make_chunk(ns, doc, embedding=_unit(8, 0))
        created = await vector_backend.create_chunk(c)
        assert created.id == c.id

        fetched = await vector_backend.get_chunk(c.id, namespace_id=ns)
        assert fetched is not None
        # Backends where metadata and embedding live in separate stores
        # (e.g. sqlite_lance — SQLite for metadata, LanceDB for vectors)
        # return ``embedding=None`` from metadata reads; the vector shows
        # up through similarity search.
        assert fetched.embedding in (None, c.embedding)

        assert await vector_backend.get_chunk(uuid4(), namespace_id=ns) is None

    async def test_create_chunks_batch(self, vector_backend):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(4)]
        result = await vector_backend.create_chunks_batch(chunks)
        assert len(result) == 4
        assert await vector_backend.count_chunks(ns) == 4

        # Empty batch is a no-op.
        assert await vector_backend.create_chunks_batch([]) == []

    async def test_get_chunks_by_document(self, vector_backend):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i), index=i) for i in range(3)]
        await vector_backend.create_chunks_batch(chunks)

        fetched = await vector_backend.get_chunks_by_document(doc, namespace_id=ns)
        assert len(fetched) == 3
        # Contract: ordered by chunk_index ascending.
        assert [c.metadata.chunk_index for c in fetched] == [0, 1, 2]

    async def test_delete_chunks_by_document(self, vector_backend):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(3)]
        await vector_backend.create_chunks_batch(chunks)

        deleted = await vector_backend.delete_chunks_by_document(doc)
        assert deleted == 3
        assert await vector_backend.count_chunks(ns) == 0

        # Deleting an empty document is a no-op.
        assert await vector_backend.delete_chunks_by_document(uuid4()) == 0

    async def test_delete_chunks_by_document_with_session(self, vector_backend):
        """When ``session`` is provided, the caller owns the transaction.

        For backends that don't use SQLAlchemy (sqlite_lance, surrealdb),
        any non-None session is treated as a signal to defer commit; the
        row count returned must still reflect the work done.
        """
        ns, doc = uuid4(), uuid4()
        await vector_backend.create_chunks_batch([_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(2)])

        sentinel = object()  # Non-SQLAlchemy opaque session.
        deleted = await vector_backend.delete_chunks_by_document(doc, session=sentinel)  # type: ignore[arg-type]
        assert deleted == 2

    async def test_search_similar_top_k(self, vector_backend):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(3)]
        await vector_backend.create_chunks_batch(chunks)

        results = await vector_backend.search_similar(ns, _unit(8, 0), limit=3)
        assert len(results) > 0
        top_chunk, top_score = results[0]
        assert top_chunk.id == chunks[0].id
        assert 0.0 <= top_score <= 1.0 + 1e-6

    async def test_search_similar_min_similarity(self, vector_backend):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(2)]
        await vector_backend.create_chunks_batch(chunks)

        results = await vector_backend.search_similar(ns, _unit(8, 0), limit=10, min_similarity=0.5)
        # Only the self-match clears a 0.5 threshold for orthogonal basis.
        assert len(results) == 1
        assert results[0][0].id == chunks[0].id

    async def test_search_similar_filter_document_ids(self, vector_backend):
        ns = uuid4()
        doc_a, doc_b = uuid4(), uuid4()
        await vector_backend.create_chunks_batch(
            [
                _make_chunk(ns, doc_a, embedding=_unit(8, 0)),
                _make_chunk(ns, doc_b, embedding=_unit(8, 1)),
            ]
        )

        filtered = await vector_backend.search_similar(ns, _unit(8, 0), limit=10, filter_document_ids=[doc_b])
        assert len(filtered) == 1
        assert filtered[0][0].document_id == doc_b

    async def test_search_similar_namespace_isolation(self, vector_backend):
        ns_a, ns_b, doc = uuid4(), uuid4(), uuid4()
        await vector_backend.create_chunks_batch(
            [
                _make_chunk(ns_a, doc, embedding=_unit(8, 0)),
                _make_chunk(ns_b, doc, embedding=_unit(8, 0)),
            ]
        )

        hits_a = await vector_backend.search_similar(ns_a, _unit(8, 0), limit=10)
        assert all(c.namespace_id == ns_a for c, _ in hits_a)

    async def test_search_fulltext(self, vector_backend):
        ns, doc = uuid4(), uuid4()
        await vector_backend.create_chunks_batch(
            [
                _make_chunk(ns, doc, content="quick brown fox", index=0),
                _make_chunk(ns, doc, content="slow green turtle", index=1),
            ]
        )

        results = await vector_backend.search_fulltext(ns, "quick fox", limit=10)
        assert len(results) == 1
        matched, score = results[0]
        assert "quick" in matched.content
        assert score != 0.0  # Non-trivial rank.

    async def test_entity_embedding_crud(self, vector_backend, entity_seeder):
        ns = uuid4()
        e = _make_entity(ns, name="Alice", embedding=_unit(8, 0))
        await entity_seeder(e)

        assert await vector_backend.entity_exists(e.id) is True
        assert await vector_backend.entity_exists(uuid4()) is False

        # Update → search still hits the row.
        e.description = "updated"
        await vector_backend.update_entity(e)
        hits = await vector_backend.search_similar_entities(ns, _unit(8, 0), limit=5)
        assert any(eid == e.id for eid, _ in hits)

    async def test_update_entity_embedding(self, vector_backend, entity_seeder):
        ns = uuid4()
        e = _make_entity(ns, name="Alice", embedding=_unit(8, 0))
        await entity_seeder(e)

        await vector_backend.update_entity_embedding(e.id, _unit(8, 3), "new-model")
        hits = await vector_backend.search_similar_entities(ns, _unit(8, 3), limit=5)
        assert hits
        assert hits[0][0] == e.id

    async def test_search_similar_entities(self, vector_backend, entity_seeder):
        ns = uuid4()
        await entity_seeder(_make_entity(ns, name="a", embedding=_unit(8, 0)))
        await entity_seeder(_make_entity(ns, name="b", embedding=_unit(8, 1)))

        # min_similarity keeps only the self-match on orthogonal basis.
        hits = await vector_backend.search_similar_entities(ns, _unit(8, 0), min_similarity=0.5)
        assert len(hits) == 1

    async def test_count_chunks_and_list_pagination(self, vector_backend):
        ns, doc = uuid4(), uuid4()
        assert await vector_backend.count_chunks(ns) == 0

        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(4)]
        await vector_backend.create_chunks_batch(chunks)

        assert await vector_backend.count_chunks(ns) == 4

        page1 = await vector_backend.list_chunks(ns, limit=2, offset=0)
        page2 = await vector_backend.list_chunks(ns, limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # No overlap between pages.
        assert {c.id for c in page1}.isdisjoint({c.id for c in page2})


# ===========================================================================
# Event store protocol
# ===========================================================================


class TestEventStoreProtocol:
    """Contract tests for :class:`EventStoreProtocol`."""

    async def test_runtime_protocol_membership(self, event_store_backend):
        assert isinstance(event_store_backend, EventStoreProtocol)

    async def test_append_event(self, event_store_backend):
        ns = uuid4()
        evt = _make_event(ns, data={"title": "hello"})
        returned = await event_store_backend.append_event(evt)
        assert returned.id == evt.id

        rows = await event_store_backend.get_events(ns, limit=10)
        assert len(rows) == 1
        assert rows[0].data == {"title": "hello"}

    async def test_append_events_batch(self, event_store_backend):
        ns = uuid4()
        base = datetime.now(UTC)
        events = [_make_event(ns, timestamp=base + timedelta(seconds=i), data={"i": i}) for i in range(10)]
        returned = await event_store_backend.append_events_batch(events)
        assert len(returned) == 10
        assert await event_store_backend.count_events(ns) == 10

        # Empty batch is a no-op.
        assert await event_store_backend.append_events_batch([]) == []

    async def test_list_events_pagination(self, event_store_backend):
        ns = uuid4()
        base = datetime.now(UTC)
        events = [_make_event(ns, timestamp=base + timedelta(seconds=i), data={"i": i}) for i in range(20)]
        await event_store_backend.append_events_batch(events)

        first = await event_store_backend.get_events(ns, limit=10, offset=0)
        second = await event_store_backend.get_events(ns, limit=10, offset=10)
        assert len(first) == 10
        assert len(second) == 10

        # No overlap in page contents.
        ids = {e.id for e in first} | {e.id for e in second}
        assert len(ids) == 20

    async def test_get_events_time_boundaries(self, event_store_backend):
        ns = uuid4()
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        events = [_make_event(ns, timestamp=t0 + timedelta(hours=i)) for i in range(5)]
        await event_store_backend.append_events_batch(events)

        cutoff = t0 + timedelta(hours=2)
        after = await event_store_backend.get_events(ns, after=cutoff, limit=100)
        # Strict > semantics — the exact boundary is excluded.
        assert len(after) == 2

        before = await event_store_backend.get_events(ns, before=cutoff, limit=100)
        assert len(before) == 2

    async def test_filter_by_event_types(self, event_store_backend):
        ns = uuid4()
        await event_store_backend.append_events_batch(
            [
                _make_event(ns, event_type=EventType.DOCUMENT_CREATED),
                _make_event(ns, event_type=EventType.DOCUMENT_UPDATED),
                _make_event(ns, event_type=EventType.ENTITY_CREATED),
                _make_event(ns, event_type=EventType.ENTITY_CREATED),
            ]
        )

        ents = await event_store_backend.get_events(ns, event_types=[EventType.ENTITY_CREATED], limit=100)
        assert len(ents) == 2
        assert all(e.event_type == EventType.ENTITY_CREATED for e in ents)

        # String values work too.
        docs = await event_store_backend.get_events(ns, event_types=["document.created", "document.updated"], limit=100)
        assert len(docs) == 2

    async def test_count_events(self, event_store_backend):
        ns_a, ns_b = uuid4(), uuid4()
        await event_store_backend.append_events_batch(
            [
                _make_event(ns_a, event_type=EventType.DOCUMENT_CREATED),
                _make_event(ns_a, event_type=EventType.DOCUMENT_CREATED),
                _make_event(ns_b, event_type=EventType.DOCUMENT_CREATED),
            ]
        )

        assert await event_store_backend.count_events(ns_a) == 2
        assert await event_store_backend.count_events(ns_b) == 1
        assert await event_store_backend.count_events(ns_a, event_types=[EventType.DOCUMENT_CREATED]) == 2
        assert await event_store_backend.count_events(ns_a, event_types=[EventType.ENTITY_CREATED]) == 0

    async def test_json_payload_unicode_nested_null(self, event_store_backend):
        """Complex JSON must roundtrip losslessly."""
        ns = uuid4()
        payload = {
            "unicode": "héllo ✨ 世界 ⚡",
            "nested": {"a": [1, 2, {"b": None}], "c": True},
            "null_field": None,
            "empty_list": [],
            "empty_dict": {},
        }
        evt = _make_event(ns, data=payload)
        await event_store_backend.append_event(evt)

        rows = await event_store_backend.get_events(ns, limit=1)
        assert len(rows) == 1
        assert rows[0].data == payload

    async def test_get_events_for_resource(self, event_store_backend):
        ns = uuid4()
        target = uuid4()
        other = uuid4()
        t0 = datetime(2026, 5, 1, tzinfo=UTC)
        await event_store_backend.append_events_batch(
            [
                _make_event(
                    ns,
                    event_type=EventType.DOCUMENT_CREATED,
                    resource_id=target,
                    timestamp=t0,
                ),
                _make_event(
                    ns,
                    event_type=EventType.DOCUMENT_UPDATED,
                    resource_id=target,
                    timestamp=t0 + timedelta(hours=1),
                ),
                _make_event(
                    ns,
                    event_type=EventType.DOCUMENT_CREATED,
                    resource_id=other,
                    timestamp=t0,
                ),
            ]
        )

        hits = await event_store_backend.get_events_for_resource("document", target, limit=10)
        assert len(hits) == 2
        assert all(e.resource_id == target for e in hits)

        latest = await event_store_backend.get_latest_event("document", target)
        assert latest is not None
        assert latest.event_type == EventType.DOCUMENT_UPDATED

        # Unknown resource → None.
        assert await event_store_backend.get_latest_event("document", uuid4()) is None

    async def test_append_only_no_mutation_methods(self, event_store_backend):
        """An append-only event store must not expose update/delete hooks."""
        forbidden = {"update_event", "delete_event", "upsert_event", "remove_event", "patch_event"}
        attrs = {name for name in dir(event_store_backend) if not name.startswith("_")}
        assert forbidden.isdisjoint(attrs), f"unexpected mutation API: {forbidden & attrs}"
