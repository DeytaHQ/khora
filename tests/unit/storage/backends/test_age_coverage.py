"""Coverage tests for khora.storage.backends.age.

Augments the existing ``tests/unit/test_age_backend.py`` (which covers pure
helpers — escape / serialize / parse_agtype / from_config) with lifecycle,
session helpers, and the Cypher-builder paths for CRUD / traversal methods
using a mocked SQLAlchemy async session.  No real PostgreSQL AGE.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Entity, Episode, Relationship
from khora.storage.backends.age import AGEBackend

# IGR-221/223: read-side methods now require a kwarg-only ``namespace_id`` so
# the backend can scope every match to the caller's tenant.  Tests use this
# fixed UUID rather than a fresh ``uuid4()`` per call so assertions against the
# generated Cypher can pin the embedded literal.
_NS = uuid4()

# ---------------------------------------------------------------------------
# Mock session plumbing
# ---------------------------------------------------------------------------


def _make_session(rows: list[Any] | None = None) -> AsyncMock:
    """Build a mocked SQLAlchemy AsyncSession.

    ``session.execute(...)`` returns a result whose ``fetchall()`` yields
    ``rows``.  ``session.begin()`` is an async context manager.
    """
    result = MagicMock()
    result.fetchall = MagicMock(return_value=rows or [])
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)

    @asynccontextmanager
    async def _begin():  # type: ignore[no-untyped-def]
        yield None

    session.begin = MagicMock(side_effect=_begin)
    return session


def _connected_backend(session: AsyncMock) -> AGEBackend:
    """Bolt a session-factory that yields the mocked session directly."""
    backend = AGEBackend(database_url="postgresql://localhost/test")

    @asynccontextmanager
    async def _session_ctx():  # type: ignore[no-untyped-def]
        yield session

    # The session_factory is called as ``self._session_factory()`` and must
    # return an async context manager.
    backend._session_factory = MagicMock(side_effect=_session_ctx)  # type: ignore[assignment]
    return backend


# ---------------------------------------------------------------------------
# __init__ / lifecycle / connection plumbing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_owns_engine_when_none_passed() -> None:
    b = AGEBackend("postgresql://x/y")
    assert b._owns_engine is True
    assert b._engine is None
    assert b._session_factory is None
    assert b._graph_name == "khora_graph"


@pytest.mark.unit
def test_init_does_not_own_external_engine() -> None:
    engine = MagicMock()
    b = AGEBackend("postgresql://x/y", engine=engine)
    assert b._owns_engine is False
    assert b._engine is engine


@pytest.mark.unit
async def test_connect_rewrites_postgresql_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fake_engine = MagicMock()

    def _factory(url, **kw):  # type: ignore[no-untyped-def]
        captured["url"] = url
        captured.update(kw)
        return fake_engine

    monkeypatch.setattr("khora.storage.backends.age.create_async_engine", _factory)

    # Pre-build a fake session that succeeds for the bootstrap statements.
    fake_session = _make_session()
    monkeypatch.setattr(
        "khora.storage.backends.age.async_sessionmaker",
        lambda *a, **kw: MagicMock(side_effect=lambda: _async_ctx(fake_session)),
    )

    b = AGEBackend("postgresql://h/db")
    await b.connect()
    assert captured["url"].startswith("postgresql+asyncpg://")
    assert b._engine is fake_engine
    assert b._session_factory is not None


@asynccontextmanager
async def _async_ctx(session):  # type: ignore[no-untyped-def]
    yield session


@pytest.mark.unit
async def test_connect_rewrites_postgres_short_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _factory(url, **kw):  # type: ignore[no-untyped-def]
        captured["url"] = url
        return MagicMock()

    monkeypatch.setattr("khora.storage.backends.age.create_async_engine", _factory)
    fake_session = _make_session()
    monkeypatch.setattr(
        "khora.storage.backends.age.async_sessionmaker",
        lambda *a, **kw: MagicMock(side_effect=lambda: _async_ctx(fake_session)),
    )

    b = AGEBackend("postgres://h/db")
    await b.connect()
    assert captured["url"].startswith("postgresql+asyncpg://")


@pytest.mark.unit
async def test_connect_passthrough_asyncpg(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _factory(url, **kw):  # type: ignore[no-untyped-def]
        captured["url"] = url
        return MagicMock()

    monkeypatch.setattr("khora.storage.backends.age.create_async_engine", _factory)
    fake_session = _make_session()
    monkeypatch.setattr(
        "khora.storage.backends.age.async_sessionmaker",
        lambda *a, **kw: MagicMock(side_effect=lambda: _async_ctx(fake_session)),
    )

    b = AGEBackend("postgresql+asyncpg://h/db")
    await b.connect()
    assert captured["url"] == "postgresql+asyncpg://h/db"


@pytest.mark.unit
async def test_connect_is_idempotent() -> None:
    b = AGEBackend("postgresql://x/y")
    b._session_factory = MagicMock()  # type: ignore[assignment]
    before = b._session_factory
    await b.connect()
    assert b._session_factory is before


@pytest.mark.unit
async def test_connect_swallows_create_graph_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``create_graph(...)`` call raises when the graph already exists;
    connect() must swallow that exact error and continue."""
    fake_engine = MagicMock()
    monkeypatch.setattr("khora.storage.backends.age.create_async_engine", lambda *a, **kw: fake_engine)

    # Build a session whose 4th execute() call (the create_graph) raises.
    call_count = {"n": 0}

    async def _execute(stmt):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if "create_graph" in str(stmt):
            raise RuntimeError("already exists")
        return MagicMock(fetchall=MagicMock(return_value=[]))

    session = AsyncMock()
    session.execute = _execute

    @asynccontextmanager
    async def _begin():  # type: ignore[no-untyped-def]
        yield None

    session.begin = MagicMock(side_effect=_begin)

    monkeypatch.setattr(
        "khora.storage.backends.age.async_sessionmaker",
        lambda *a, **kw: MagicMock(side_effect=lambda: _async_ctx(session)),
    )

    b = AGEBackend("postgresql://h/db")
    # Must not raise.
    await b.connect()


@pytest.mark.unit
async def test_disconnect_disposes_when_owns_engine() -> None:
    b = AGEBackend("postgresql://x/y")
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()
    b._engine = fake_engine
    b._owns_engine = True
    b._session_factory = MagicMock()  # type: ignore[assignment]
    await b.disconnect()
    fake_engine.dispose.assert_awaited()
    assert b._engine is None
    assert b._session_factory is None


@pytest.mark.unit
async def test_disconnect_skips_dispose_when_external_engine() -> None:
    b = AGEBackend("postgresql://x/y")
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()
    b._engine = fake_engine
    b._owns_engine = False
    b._session_factory = MagicMock()  # type: ignore[assignment]
    await b.disconnect()
    fake_engine.dispose.assert_not_called()
    assert b._engine is None


@pytest.mark.unit
async def test_disconnect_safe_when_disconnected() -> None:
    b = AGEBackend("postgresql://x/y")
    await b.disconnect()  # must not raise


@pytest.mark.unit
async def test_is_healthy_false_when_disconnected() -> None:
    b = AGEBackend("postgresql://x/y")
    assert await b.is_healthy() is False


@pytest.mark.unit
async def test_is_healthy_true_on_success() -> None:
    session = _make_session()
    b = _connected_backend(session)
    assert await b.is_healthy() is True


@pytest.mark.unit
async def test_is_healthy_false_on_error() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=RuntimeError("db down"))

    @asynccontextmanager
    async def _begin():  # type: ignore[no-untyped-def]
        yield None

    session.begin = MagicMock(side_effect=_begin)
    b = AGEBackend("postgresql://x/y")

    @asynccontextmanager
    async def _ctx():  # type: ignore[no-untyped-def]
        yield session

    b._session_factory = MagicMock(side_effect=_ctx)  # type: ignore[assignment]
    assert await b.is_healthy() is False


@pytest.mark.unit
def test_get_session_factory_raises_when_disconnected() -> None:
    b = AGEBackend("postgresql://x/y")
    with pytest.raises(RuntimeError, match="not connected"):
        b._get_session_factory()


# ---------------------------------------------------------------------------
# _cypher — SQL wrapping & defense
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_cypher_wraps_with_uniquely_tagged_dollar_quote() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    out = await b._cypher(session, "MATCH (n) RETURN n", columns=["v agtype"])
    assert out == []
    sql = str(session.execute.await_args.args[0])
    assert "$khora_age$" in sql
    assert "cypher('khora_graph'" in sql
    assert "MATCH (n) RETURN n" in sql


@pytest.mark.unit
async def test_cypher_rejects_reserved_tag_in_payload() -> None:
    """The defense-in-depth check rejects payloads with the literal tag."""
    session = _make_session()
    b = _connected_backend(session)
    with pytest.raises(ValueError, match="reserved dollar-quote tag"):
        await b._cypher(session, "MATCH (n) RETURN $khora_age$", columns=["v agtype"])


@pytest.mark.unit
async def test_cypher_parses_rows_via_parse_agtype() -> None:
    """Each row's columns go through ``_parse_agtype``."""
    session = _make_session(rows=[(42,), (None,)])
    b = _connected_backend(session)
    out = await b._cypher(session, "RETURN 1", columns=["v agtype"])
    assert out == [{"v": 42}, {"v": None}]


@pytest.mark.unit
async def test_cypher_default_columns_clause() -> None:
    """When ``columns`` is None, the implementation falls back to ``['v agtype']``."""
    session = _make_session(rows=[])
    b = _connected_backend(session)
    await b._cypher(session, "MATCH (n) RETURN n")
    sql = str(session.execute.await_args.args[0])
    assert "AS (v agtype)" in sql


# ---------------------------------------------------------------------------
# Session helper paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_age_session_sets_search_path() -> None:
    """``_age_session`` returns a session with search_path already set."""
    session = AsyncMock()
    session.execute = AsyncMock()

    b = AGEBackend("postgresql://x/y")
    b._session_factory = MagicMock(return_value=session)  # type: ignore[assignment]
    got = await b._age_session()
    assert got is session
    # First execute() set the search_path.
    session.execute.assert_awaited()
    sql = str(session.execute.await_args.args[0])
    assert "search_path" in sql and "ag_catalog" in sql


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_entity_builds_cypher_and_returns_entity() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    ent = Entity(name="Alice", entity_type="PERSON")
    result = await b.create_entity(ent)
    # When the AGE result has no rows the implementation falls back to the
    # input ``entity`` — the parse path is exercised in test_age_backend.py.
    assert result is ent
    # The Cypher CREATE statement is in the SQL.
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any("CREATE (e:Entity" in sql for sql in assembled)


@pytest.mark.unit
async def test_get_entity_returns_none_when_no_rows() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    out = await b.get_entity(uuid4(), namespace_id=_NS)
    assert out is None


@pytest.mark.unit
async def test_get_entity_by_name_returns_none_when_no_rows() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    out = await b.get_entity_by_name(uuid4(), "Alice", "PERSON")
    assert out is None


@pytest.mark.unit
async def test_update_entity_returns_input() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    ent = Entity(name="Alice", entity_type="PERSON")
    out = await b.update_entity(ent)
    assert out is ent
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any("MATCH (e:Entity" in sql and "SET" in sql for sql in assembled)


@pytest.mark.unit
async def test_delete_entity_false_when_no_rows() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    assert await b.delete_entity(uuid4()) is False


@pytest.mark.unit
async def test_delete_entity_uses_detach_delete() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    await b.delete_entity(uuid4())
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any("DETACH DELETE e" in sql for sql in assembled)


@pytest.mark.unit
async def test_list_entities_empty_result() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    out = await b.list_entities(uuid4())
    assert out == []


@pytest.mark.unit
async def test_list_entities_with_type_filter_builds_where() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    await b.list_entities(uuid4(), entity_type="PERSON")
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any("WHERE e.entity_type" in sql for sql in assembled)


@pytest.mark.unit
async def test_count_entities_zero_when_no_rows() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    assert await b.count_entities(uuid4()) == 0


@pytest.mark.unit
async def test_count_relationships_raises_not_implemented() -> None:
    b = _connected_backend(_make_session())
    with pytest.raises(NotImplementedError):
        await b.count_relationships(uuid4())


# ---------------------------------------------------------------------------
# Relationship operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_relationship_returns_input_and_sanitizes_label() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    rel = Relationship(relationship_type="reports-to!")
    out = await b.create_relationship(rel)
    assert out is rel
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    # ``_sanitize_label`` replaces non-[A-Za-z0-9_] with `_`.
    assert any("reports_to_" in sql for sql in assembled)


@pytest.mark.unit
async def test_get_relationship_none_when_no_rows() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    assert await b.get_relationship(uuid4(), namespace_id=_NS) is None


@pytest.mark.unit
async def test_delete_relationship_false_when_no_rows() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    assert await b.delete_relationship(uuid4()) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "direction,expected_fragment",
    [
        # IGR-223: patterns now embed ``{namespace_id: '<uuid>'}`` on every
        # node, so the bare ``(e)-[r`` form no longer appears.  Each fragment
        # below is the smallest substring unique to the chosen direction.
        ("outgoing", "]->(other:Entity"),
        ("incoming", "]->(e:Entity"),
        ("both", "]-(other:Entity"),
    ],
)
async def test_get_entity_relationships_direction(direction: str, expected_fragment: str) -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    out = await b.get_entity_relationships(uuid4(), namespace_id=_NS, direction=direction)
    assert out == []
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any(expected_fragment in sql for sql in assembled)
    # And the per-tenant namespace literal appears on the pattern.
    assert any(f"namespace_id: '{_NS}'" in sql for sql in assembled)


@pytest.mark.unit
async def test_get_entity_relationships_with_rel_types_join() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    await b.get_entity_relationships(uuid4(), namespace_id=_NS, relationship_types=["knows", "WORKS_AT"])
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any("knows|WORKS_AT" in sql for sql in assembled)


@pytest.mark.unit
async def test_list_relationships_no_filter() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    out = await b.list_relationships(uuid4())
    assert out == []


@pytest.mark.unit
async def test_list_relationships_with_type_filter() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    await b.list_relationships(uuid4(), relationship_type="OWNS")
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any("[r:OWNS]" in sql for sql in assembled)


# ---------------------------------------------------------------------------
# Episode operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_episode_no_entity_links() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    ep = Episode(name="x", occurred_at=datetime.now(UTC))
    out = await b.create_episode(ep)
    assert out is ep
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any("CREATE (ep:Episode" in sql for sql in assembled)
    # No INVOLVES link statement.
    assert not any("INVOLVES" in sql for sql in assembled)


@pytest.mark.unit
async def test_create_episode_with_entity_ids_emits_involves() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    ep = Episode(name="x", occurred_at=datetime.now(UTC), entity_ids=[uuid4(), uuid4()])
    await b.create_episode(ep)
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert sum("INVOLVES" in sql for sql in assembled) >= 2


@pytest.mark.unit
async def test_get_episode_none_when_no_rows() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    assert await b.get_episode(uuid4(), namespace_id=_NS) is None


@pytest.mark.unit
async def test_list_episodes_no_filters() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    out = await b.list_episodes(uuid4())
    assert out == []
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert not any("WHERE" in sql for sql in assembled if "MATCH (ep:Episode" in sql)


@pytest.mark.unit
async def test_list_episodes_with_time_filters() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    await b.list_episodes(
        uuid4(),
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 12, 31, tzinfo=UTC),
    )
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any("ep.occurred_at >=" in sql and "ep.occurred_at <=" in sql for sql in assembled)


# ---------------------------------------------------------------------------
# Traversal — find_paths / get_neighborhood
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_find_paths_empty_result() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    paths = await b.find_paths(uuid4(), uuid4(), uuid4())
    assert paths == []


@pytest.mark.unit
async def test_find_paths_with_rel_filter_and_depth() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    await b.find_paths(uuid4(), uuid4(), uuid4(), relationship_types=["KNOWS", "WORKS_AT"], max_depth=4)
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any(":KNOWS|WORKS_AT" in sql for sql in assembled)
    assert any("*1..4" in sql for sql in assembled)


@pytest.mark.unit
async def test_get_neighborhood_empty() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    result = await b.get_neighborhood(uuid4(), namespace_id=_NS)
    assert result == {"entities": [], "relationships": []}


@pytest.mark.unit
async def test_get_neighborhood_with_rel_types() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    await b.get_neighborhood(uuid4(), namespace_id=_NS, relationship_types=["RELATES"], depth=3)
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    assert any(":RELATES" in sql for sql in assembled)
    assert any("*1..3" in sql for sql in assembled)


# ---------------------------------------------------------------------------
# search_entities_by_attribute — verifies escape goes into the Cypher
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_search_entities_by_attribute_escapes_payload() -> None:
    session = _make_session(rows=[])
    b = _connected_backend(session)
    out = await b.search_entities_by_attribute(uuid4(), "role", "it's-admin")
    assert out == []
    assembled = [str(call.args[0]) for call in session.execute.await_args_list]
    # The single-quote in the value is escaped by ``_escape``.
    assert any("it\\'s-admin" in sql for sql in assembled)


# ---------------------------------------------------------------------------
# Result parsing edge cases for get_neighborhood
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_neighborhood_with_nested_relationship_lists() -> None:
    """get_neighborhood flattens nested relationship lists from agtype path output."""
    session = _make_session(
        rows=[
            (
                '[{"id": "n1", "label": "Entity"}]',
                '[[{"id": "r1", "label": "REL"}], {"id": "r2", "label": "REL"}]',
            )
        ]
    )
    b = _connected_backend(session)
    result = await b.get_neighborhood(uuid4(), namespace_id=_NS)
    # 1 node, 2 flattened relationships.
    assert isinstance(result, dict)
    assert "entities" in result and "relationships" in result
    assert len(result["entities"]) == 1
    # Both nested elements and the non-list element are flattened.
    assert len(result["relationships"]) == 2


@pytest.mark.unit
async def test_get_neighborhood_with_non_list_nodes_falls_back_to_empty() -> None:
    """When the parsed value for ``nodes`` isn't a list, default to ``[]``."""
    session = _make_session(rows=[("not-a-list-just-a-string", '"also-not-a-list"')])
    b = _connected_backend(session)
    result = await b.get_neighborhood(uuid4(), namespace_id=_NS)
    assert result["entities"] == []
    assert result["relationships"] == []
