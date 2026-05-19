"""Coverage tests for khora.storage.backends.memgraph.

Exercises init, URL/SecretStr handling, lifecycle (connect/disconnect/health),
record-to-domain converters, and the Cypher-building paths for every CRUD /
traversal method using a mocked neo4j async driver.  No real Memgraph.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from khora.config.schema import MemgraphConfig
from khora.core.models import Entity, Episode, Relationship
from khora.storage.backends.memgraph import MemgraphBackend

# IGR-221/223: read-side methods now require a kwarg-only ``namespace_id`` so
# the backend can scope every Cypher MATCH to the caller's tenant.  Tests use
# this fixed UUID across the file so assertions against the generated query
# parameters can pin the value.
_NS = uuid4()

# ---------------------------------------------------------------------------
# Mock driver / session plumbing
# ---------------------------------------------------------------------------


def _make_session_with_records(
    records: list[dict[str, Any]] | None = None, single: dict[str, Any] | None = None
) -> AsyncMock:
    """Build a mocked neo4j async session.

    ``result.data()`` returns ``records`` (list of dicts).  ``result.single()``
    returns ``single``.  ``session.run`` is awaited and returns this result.
    """
    result = MagicMock()
    result.data = AsyncMock(return_value=records or [])
    result.single = AsyncMock(return_value=single)
    session = AsyncMock()
    session.run = AsyncMock(return_value=result)
    return session


def _make_driver(session: AsyncMock) -> MagicMock:
    """Build a mocked neo4j async driver whose session() is an async ctx manager."""
    driver = MagicMock()

    @asynccontextmanager
    async def _session_ctx():  # type: ignore[no-untyped-def]
        yield session

    driver.session = MagicMock(side_effect=_session_ctx)
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    return driver


def _connected_backend(session: AsyncMock) -> MemgraphBackend:
    """Skip connect() — bolt the mocked driver directly onto the backend."""
    backend = MemgraphBackend("bolt://localhost:7687")
    backend._driver = _make_driver(session)
    return backend


# ---------------------------------------------------------------------------
# __init__ / from_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_stores_attributes() -> None:
    b = MemgraphBackend("bolt://h:7687", user="alice", password="pw", max_connection_pool_size=11)
    assert b._url == "bolt://h:7687"
    assert b._user == "alice"
    assert b._password == "pw"
    assert b._max_connection_pool_size == 11
    assert b._driver is None


@pytest.mark.unit
def test_init_defaults() -> None:
    b = MemgraphBackend("bolt://h:7687")
    assert b._user == "memgraph"
    assert b._password == ""
    assert b._max_connection_pool_size == 50


@pytest.mark.unit
def test_from_config_plain_values() -> None:
    cfg = MemgraphConfig(url="bolt://mg:7687", user="u", password="p")
    b = MemgraphBackend.from_config(cfg)
    assert b._url == "bolt://mg:7687"
    assert b._user == "u"
    assert b._password == "p"


@pytest.mark.unit
def test_from_config_unwraps_secretstr() -> None:
    cfg = MemgraphConfig(
        url=SecretStr("bolt://secret:7687"),
        password=SecretStr("hidden"),
    )
    b = MemgraphBackend.from_config(cfg)
    assert b._url == "bolt://secret:7687"
    assert b._password == "hidden"


@pytest.mark.unit
def test_from_config_url_none_defaults_to_localhost() -> None:
    cfg = MemgraphConfig()  # url=None
    b = MemgraphBackend.from_config(cfg)
    assert b._url == "bolt://localhost:7687"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_connect_initializes_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _make_session_with_records()
    fake_driver = _make_driver(fake_session)

    # Patch the neo4j module so connect() doesn't try a real bolt handshake.
    import neo4j

    monkeypatch.setattr(neo4j.AsyncGraphDatabase, "driver", lambda *a, **kw: fake_driver)
    b = MemgraphBackend("bolt://h:7687", user="u", password="p")
    await b.connect()
    assert b._driver is fake_driver
    fake_driver.verify_connectivity.assert_awaited()


@pytest.mark.unit
async def test_connect_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second connect() should not replace the existing driver."""
    fake_driver = _make_driver(_make_session_with_records())
    b = MemgraphBackend("bolt://h:7687")
    b._driver = fake_driver

    # Spy on AsyncGraphDatabase.driver — it must NOT be called again.
    import neo4j

    called = []
    monkeypatch.setattr(neo4j.AsyncGraphDatabase, "driver", lambda *a, **kw: called.append(1) or fake_driver)

    await b.connect()
    assert called == []
    assert b._driver is fake_driver


@pytest.mark.unit
async def test_disconnect_closes_and_clears() -> None:
    fake_driver = _make_driver(_make_session_with_records())
    b = MemgraphBackend("bolt://h:7687")
    b._driver = fake_driver

    await b.disconnect()
    fake_driver.close.assert_awaited()
    assert b._driver is None


@pytest.mark.unit
async def test_disconnect_noop_when_not_connected() -> None:
    b = MemgraphBackend("bolt://h:7687")
    await b.disconnect()  # must not raise
    assert b._driver is None


@pytest.mark.unit
async def test_is_healthy_false_when_disconnected() -> None:
    b = MemgraphBackend("bolt://h:7687")
    assert await b.is_healthy() is False


@pytest.mark.unit
async def test_is_healthy_true_on_success() -> None:
    fake_driver = _make_driver(_make_session_with_records())
    b = MemgraphBackend("bolt://h:7687")
    b._driver = fake_driver
    assert await b.is_healthy() is True
    fake_driver.verify_connectivity.assert_awaited()


@pytest.mark.unit
async def test_is_healthy_false_on_error() -> None:
    fake_driver = _make_driver(_make_session_with_records())
    fake_driver.verify_connectivity = AsyncMock(side_effect=RuntimeError("down"))
    b = MemgraphBackend("bolt://h:7687")
    b._driver = fake_driver
    assert await b.is_healthy() is False


@pytest.mark.unit
def test_get_driver_raises_when_disconnected() -> None:
    b = MemgraphBackend("bolt://h:7687")
    with pytest.raises(RuntimeError, match="not connected"):
        b._get_driver()


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_indexes_runs_seven_statements() -> None:
    session = _make_session_with_records()
    b = MemgraphBackend("bolt://h:7687")
    b._driver = _make_driver(session)
    await b._create_indexes()
    # 7 indexes defined in the implementation.
    assert session.run.await_count == 7
    statements = [c.args[0] for c in session.run.await_args_list]
    assert all(s.startswith("CREATE INDEX ON :") for s in statements)


@pytest.mark.unit
async def test_create_indexes_swallows_errors() -> None:
    session = AsyncMock()
    session.run = AsyncMock(side_effect=RuntimeError("already exists"))
    b = MemgraphBackend("bolt://h:7687")
    b._driver = _make_driver(session)
    # Should NOT raise — Memgraph throws on duplicate index, by design.
    await b._create_indexes()


@pytest.mark.unit
async def test_create_indexes_skipped_when_no_driver() -> None:
    b = MemgraphBackend("bolt://h:7687")
    await b._create_indexes()  # must be a silent no-op


# ---------------------------------------------------------------------------
# Record-to-domain converters
# ---------------------------------------------------------------------------


def _entity_node(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "name": "Alice",
        "entity_type": "PERSON",
        "description": "a person",
        "attributes": "{}",
        "source_document_ids": [],
        "source_chunk_ids": [],
        "mention_count": 2,
        "confidence": 0.9,
        "metadata": "{}",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_record_to_entity_full() -> None:
    b = MemgraphBackend("bolt://h:7687")
    node = _entity_node()
    ent = b._record_to_entity(node)
    assert isinstance(ent, Entity)
    assert ent.name == "Alice"
    assert ent.entity_type == "PERSON"
    assert ent.confidence == 0.9
    assert ent.mention_count == 2


@pytest.mark.unit
def test_record_to_entity_with_valid_window() -> None:
    b = MemgraphBackend("bolt://h:7687")
    node = _entity_node(
        valid_from="2026-01-01T00:00:00+00:00",
        valid_until="2026-12-31T00:00:00+00:00",
    )
    ent = b._record_to_entity(node)
    assert ent.valid_from is not None and ent.valid_from.year == 2026
    assert ent.valid_until is not None and ent.valid_until.year == 2026


@pytest.mark.unit
def test_record_to_entity_missing_timestamps_uses_now() -> None:
    b = MemgraphBackend("bolt://h:7687")
    node = _entity_node()
    del node["created_at"]
    del node["updated_at"]
    ent = b._record_to_entity(node)
    assert isinstance(ent.created_at, datetime)
    assert isinstance(ent.updated_at, datetime)


@pytest.mark.unit
def test_record_to_relationship() -> None:
    b = MemgraphBackend("bolt://h:7687")
    rel_id = str(uuid4())
    ns_id = str(uuid4())
    src = str(uuid4())
    tgt = str(uuid4())
    rel = b._record_to_relationship(
        {
            "id": rel_id,
            "namespace_id": ns_id,
            "description": "works at",
            "properties": "{}",
            "source_document_ids": [],
            "source_chunk_ids": [],
            "confidence": 0.5,
            "weight": 1.5,
            "metadata": "{}",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        src,
        tgt,
        "WORKS_AT",
    )
    assert isinstance(rel, Relationship)
    assert rel.id == UUID(rel_id)
    assert rel.source_entity_id == UUID(src)
    assert rel.target_entity_id == UUID(tgt)
    assert rel.relationship_type == "WORKS_AT"
    assert rel.weight == 1.5


@pytest.mark.unit
def test_record_to_episode() -> None:
    b = MemgraphBackend("bolt://h:7687")
    ep_id = str(uuid4())
    ns_id = str(uuid4())
    ep = b._record_to_episode(
        {
            "id": ep_id,
            "namespace_id": ns_id,
            "name": "Meeting",
            "description": "weekly sync",
            "occurred_at": "2026-01-15T10:00:00+00:00",
            "duration_seconds": 900,
            "entity_ids": [],
            "source_document_ids": [],
            "source_chunk_ids": [],
            "metadata": "{}",
            "created_at": "2026-01-15T10:00:00+00:00",
            "updated_at": "2026-01-15T10:00:00+00:00",
        }
    )
    assert isinstance(ep, Episode)
    assert ep.id == UUID(ep_id)
    assert ep.duration_seconds == 900


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_entity_sends_expected_params() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    ent = Entity(name="Bob", entity_type="PERSON", description="d")
    result = await b.create_entity(ent)
    assert result is ent

    # Inspect Cypher + params
    call = session.run.await_args
    cypher = call.args[0]
    kwargs = call.kwargs
    assert "CREATE (e:Entity" in cypher
    assert kwargs["name"] == "Bob"
    assert kwargs["entity_type"] == "PERSON"
    assert kwargs["id"] == str(ent.id)


@pytest.mark.unit
async def test_get_entity_returns_none_when_missing() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    got = await b.get_entity(uuid4(), namespace_id=_NS)
    assert got is None


@pytest.mark.unit
async def test_get_entity_returns_domain_model() -> None:
    node = _entity_node()
    session = _make_session_with_records(single={"e": node})
    b = _connected_backend(session)
    got = await b.get_entity(UUID(node["id"]), namespace_id=UUID(node["namespace_id"]))
    assert got is not None
    assert got.name == "Alice"


@pytest.mark.unit
async def test_get_entity_by_name_returns_none_when_missing() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    got = await b.get_entity_by_name(uuid4(), "Alice", "PERSON")
    assert got is None


@pytest.mark.unit
async def test_get_entity_by_name_returns_entity() -> None:
    node = _entity_node()
    session = _make_session_with_records(single={"e": node})
    b = _connected_backend(session)
    got = await b.get_entity_by_name(UUID(node["namespace_id"]), node["name"], node["entity_type"])
    assert got is not None
    assert got.name == node["name"]


@pytest.mark.unit
async def test_get_entities_batch_empty_short_circuits() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    result = await b.get_entities_batch([], namespace_id=_NS)
    assert result == {}
    session.run.assert_not_called()


@pytest.mark.unit
async def test_get_entities_batch_returns_mapping() -> None:
    node = _entity_node()
    session = _make_session_with_records(records=[{"e": node}])
    b = _connected_backend(session)
    result = await b.get_entities_batch([UUID(node["id"])], namespace_id=UUID(node["namespace_id"]))
    assert UUID(node["id"]) in result


@pytest.mark.unit
async def test_update_entity_sends_set_clause() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    ent = Entity(name="Bob", entity_type="PERSON")
    result = await b.update_entity(ent, namespace_id=ent.namespace_id)
    assert result is ent
    cypher = session.run.await_args.args[0]
    assert "MATCH (e:Entity" in cypher
    assert "SET" in cypher


@pytest.mark.unit
async def test_delete_entity_returns_true_when_deleted() -> None:
    session = _make_session_with_records(single={"deleted": 1})
    b = _connected_backend(session)
    assert await b.delete_entity(uuid4(), namespace_id=uuid4()) is True


@pytest.mark.unit
async def test_delete_entity_returns_false_when_missing() -> None:
    session = _make_session_with_records(single={"deleted": 0})
    b = _connected_backend(session)
    assert await b.delete_entity(uuid4(), namespace_id=uuid4()) is False


@pytest.mark.unit
async def test_delete_entity_returns_false_when_no_record() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    assert await b.delete_entity(uuid4(), namespace_id=uuid4()) is False


@pytest.mark.unit
async def test_list_entities_no_filter_builds_query_without_where() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.list_entities(uuid4(), limit=5, offset=2)
    assert out == []
    cypher = session.run.await_args.args[0]
    assert "WHERE" not in cypher  # entity_type=None → no filter
    assert "SKIP" in cypher


@pytest.mark.unit
async def test_list_entities_with_entity_type_filter() -> None:
    node = _entity_node()
    session = _make_session_with_records(records=[{"e": node}])
    b = _connected_backend(session)
    out = await b.list_entities(uuid4(), entity_type="PERSON")
    assert len(out) == 1
    cypher = session.run.await_args.args[0]
    assert "WHERE e.entity_type" in cypher


@pytest.mark.unit
async def test_count_entities_returns_value() -> None:
    session = _make_session_with_records(single={"cnt": 42})
    b = _connected_backend(session)
    assert await b.count_entities(uuid4()) == 42


@pytest.mark.unit
async def test_count_entities_returns_zero_when_no_record() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    assert await b.count_entities(uuid4()) == 0


@pytest.mark.unit
async def test_count_relationships_raises_not_implemented() -> None:
    b = _connected_backend(_make_session_with_records())
    with pytest.raises(NotImplementedError):
        await b.count_relationships(uuid4())


# ---------------------------------------------------------------------------
# Relationship operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_relationship_sanitizes_label() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    rel = Relationship(relationship_type="works at!")
    result = await b.create_relationship(rel)
    assert result is rel
    cypher = session.run.await_args.args[0]
    # ``sanitize_cypher_label`` UPPER_SNAKE_CASEs the label.
    assert "WORKS_AT_" in cypher


@pytest.mark.unit
async def test_get_relationship_returns_none_when_missing() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    got = await b.get_relationship(uuid4(), namespace_id=_NS)
    assert got is None


@pytest.mark.unit
async def test_get_relationship_returns_domain_model() -> None:
    rel_props = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "description": "d",
        "properties": "{}",
        "source_document_ids": [],
        "source_chunk_ids": [],
        "confidence": 1.0,
        "weight": 1.0,
        "metadata": "{}",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    session = _make_session_with_records(
        single={
            "r": rel_props,
            "source_id": str(uuid4()),
            "target_id": str(uuid4()),
            "rel_type": "KNOWS",
        }
    )
    b = _connected_backend(session)
    got = await b.get_relationship(uuid4(), namespace_id=_NS)
    assert got is not None
    assert got.relationship_type == "KNOWS"


@pytest.mark.unit
async def test_delete_relationship_true_when_deleted() -> None:
    session = _make_session_with_records(single={"deleted": 1})
    b = _connected_backend(session)
    assert await b.delete_relationship(uuid4(), namespace_id=uuid4()) is True


@pytest.mark.unit
async def test_delete_relationship_false_when_missing() -> None:
    session = _make_session_with_records(single={"deleted": 0})
    b = _connected_backend(session)
    assert await b.delete_relationship(uuid4(), namespace_id=uuid4()) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "direction,expected_fragment",
    [
        # IGR-223: each pattern node carries ``{namespace_id: $namespace_id}``
        # so the legacy ``(e)-[r`` form no longer appears.  Pin on the direction
        # arrow instead.
        ("outgoing", "]->(other:Entity"),
        ("incoming", "]->(e:Entity"),
        ("both", "]-(other:Entity"),
    ],
)
async def test_get_entity_relationships_direction_pattern(direction: str, expected_fragment: str) -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.get_entity_relationships(uuid4(), namespace_id=_NS, direction=direction)
    assert out == []
    cypher = session.run.await_args.args[0]
    assert expected_fragment in cypher
    # Bound parameter carries the per-tenant namespace.
    assert session.run.await_args.kwargs.get("namespace_id") == str(_NS)


@pytest.mark.unit
async def test_get_entity_relationships_rel_type_filter() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    await b.get_entity_relationships(uuid4(), namespace_id=_NS, relationship_types=["likes", "KNOWS"])
    cypher = session.run.await_args.args[0]
    # Both labels sanitized and joined with |
    assert "LIKES" in cypher
    assert "KNOWS" in cypher
    assert "|" in cypher


@pytest.mark.unit
async def test_list_relationships_no_filter() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.list_relationships(uuid4())
    assert out == []
    cypher = session.run.await_args.args[0]
    assert "[r]" in cypher  # no rel-type filter inside brackets


@pytest.mark.unit
async def test_list_relationships_with_type_filter() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    await b.list_relationships(uuid4(), relationship_type="connects to")
    cypher = session.run.await_args.args[0]
    assert "[r:CONNECTS_TO]" in cypher


# ---------------------------------------------------------------------------
# Episode operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_episode_without_entity_ids() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    ep = Episode(name="meeting", occurred_at=datetime.now(UTC))
    result = await b.create_episode(ep)
    assert result is ep
    # Only the CREATE Episode statement, no INVOLVES link.
    assert session.run.await_count == 1


@pytest.mark.unit
async def test_create_episode_with_entity_ids_emits_involves() -> None:
    session = _make_session_with_records()
    b = _connected_backend(session)
    ep = Episode(name="meeting", occurred_at=datetime.now(UTC), entity_ids=[uuid4(), uuid4()])
    await b.create_episode(ep)
    # Episode CREATE + the INVOLVES batch — 2 calls total.
    assert session.run.await_count == 2
    second_call = session.run.await_args_list[1]
    assert "INVOLVES" in second_call.args[0]


@pytest.mark.unit
async def test_get_episode_returns_none_when_missing() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    got = await b.get_episode(uuid4(), namespace_id=_NS)
    assert got is None


@pytest.mark.unit
async def test_get_episode_returns_domain_model() -> None:
    ep_node = {
        "id": str(uuid4()),
        "namespace_id": str(uuid4()),
        "name": "ep",
        "description": "",
        "occurred_at": "2026-01-15T10:00:00+00:00",
        "duration_seconds": None,
        "entity_ids": [],
        "source_document_ids": [],
        "source_chunk_ids": [],
        "metadata": "{}",
        "created_at": "2026-01-15T10:00:00+00:00",
        "updated_at": "2026-01-15T10:00:00+00:00",
    }
    session = _make_session_with_records(single={"ep": ep_node})
    b = _connected_backend(session)
    got = await b.get_episode(UUID(ep_node["id"]), namespace_id=UUID(ep_node["namespace_id"]))
    assert got is not None
    assert got.name == "ep"


@pytest.mark.unit
async def test_list_episodes_no_time_filters() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.list_episodes(uuid4())
    assert out == []
    cypher = session.run.await_args.args[0]
    assert "WHERE" not in cypher


@pytest.mark.unit
async def test_list_episodes_with_start_and_end() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.list_episodes(
        uuid4(),
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 12, 31, tzinfo=UTC),
    )
    assert out == []
    cypher = session.run.await_args.args[0]
    assert "ep.occurred_at >= $start_time" in cypher
    assert "ep.occurred_at <= $end_time" in cypher
    params = session.run.await_args.kwargs
    assert "start_time" in params
    assert "end_time" in params


# ---------------------------------------------------------------------------
# Graph traversal — find_paths / get_neighborhood
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_find_paths_empty_result() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    paths = await b.find_paths(uuid4(), uuid4(), namespace_id=uuid4())
    assert paths == []
    cypher = session.run.await_args.args[0]
    assert "MATCH path" in cypher


@pytest.mark.unit
async def test_find_paths_with_rel_filter() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    await b.find_paths(uuid4(), uuid4(), namespace_id=uuid4(), relationship_types=["KNOWS"], max_depth=5)
    cypher = session.run.await_args.args[0]
    assert ":KNOWS" in cypher
    assert "*1..5" in cypher


@pytest.mark.unit
async def test_get_neighborhood_empty() -> None:
    session = _make_session_with_records(single={"nodes": [], "relationships": []})
    b = _connected_backend(session)
    result = await b.get_neighborhood(uuid4(), namespace_id=_NS, depth=2)
    assert result == {"entities": [], "relationships": []}


@pytest.mark.unit
async def test_get_neighborhood_returns_none_record() -> None:
    session = _make_session_with_records(single=None)
    b = _connected_backend(session)
    result = await b.get_neighborhood(uuid4(), namespace_id=_NS)
    assert result == {"entities": [], "relationships": []}


@pytest.mark.unit
async def test_get_neighborhood_with_rel_types() -> None:
    session = _make_session_with_records(single={"nodes": [], "relationships": []})
    b = _connected_backend(session)
    await b.get_neighborhood(uuid4(), namespace_id=_NS, relationship_types=["likes"])
    cypher = session.run.await_args.args[0]
    assert ":LIKES" in cypher


# ---------------------------------------------------------------------------
# search_entities_by_attribute
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_search_entities_by_attribute_passes_params() -> None:
    session = _make_session_with_records(records=[])
    b = _connected_backend(session)
    out = await b.search_entities_by_attribute(uuid4(), "role", "admin")
    assert out == []
    kwargs = session.run.await_args.kwargs
    assert kwargs["attribute_name"] == "role"
    assert kwargs["attribute_value"] == "admin"
