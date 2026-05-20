"""IDOR guard: ``namespace_id``-tightened GraphBackend reads (IDOR family).

Each graph-backend read method now requires ``namespace_id`` and filters at
the query layer. These tests use mock drivers / SQL captures and verify two
things per backend without spinning up a database:

1. The new keyword arg ``namespace_id`` is actually bound into the
   Cypher/SQL parameters (so the filter reaches the engine).
2. When the underlying driver returns no rows (simulating "wrong namespace
   — driver finds nothing"), the public method returns the empty form
   (``None`` / ``{}`` / ``[]``).

Backed by a real SQLite DB: see ``sqlite_lance/test_namespace_scoped_reads.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Helpers: lightweight driver/session mocks shared across backend tests
# ---------------------------------------------------------------------------


def _mock_session(single_record: object | None = None, data_records: list | None = None) -> MagicMock:
    """Build a mock driver-session object that ``.run()`` yields a mock result on.

    Also supports ``session.execute_read(work)`` / ``execute_write(work)`` —
    Neo4j's managed-transaction helpers — by calling the supplied work function
    with a transaction handle whose ``.run()`` returns the same mock result.
    """
    result = MagicMock()
    result.single = AsyncMock(return_value=single_record)
    result.data = AsyncMock(return_value=data_records or [])

    tx = MagicMock()
    tx.run = AsyncMock(return_value=result)

    async def _execute(work, *args, **kwargs):
        return await work(tx)

    session = MagicMock()
    session.run = AsyncMock(return_value=result)
    session.execute_read = AsyncMock(side_effect=_execute)
    session.execute_write = AsyncMock(side_effect=_execute)
    # Surface the inner-tx run for tests that need to inspect Cypher bound
    # inside execute_read.
    session._tx = tx  # noqa: SLF001
    return session


def _driver_with_session(session: MagicMock) -> MagicMock:
    """Build a mock neo4j-shaped driver whose ``.session()`` ctx yields ``session``."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver = MagicMock()
    driver.session = MagicMock(return_value=ctx)
    return driver


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeo4jNamespaceFilter:
    """Neo4jBackend reads filter on ``namespace_id`` via bound parameters."""

    def _backend(self, session: MagicMock):
        from khora.storage.backends.neo4j import Neo4jBackend

        driver = _driver_with_session(session)
        return Neo4jBackend.from_driver(driver, query_timeout=None)

    @pytest.mark.asyncio
    async def test_get_entity_binds_namespace_and_returns_none_on_miss(self) -> None:
        session = _mock_session(single_record=None)
        b = self._backend(session)
        ns = uuid4()
        eid = uuid4()

        result = await b.get_entity(eid, namespace_id=ns)

        assert result is None
        _, kwargs = session.run.call_args
        assert kwargs["id"] == str(eid)
        assert kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_get_entities_batch_filters_by_namespace(self) -> None:
        session = _mock_session(data_records=[])
        b = self._backend(session)
        ns = uuid4()
        ids = [uuid4(), uuid4()]

        result = await b.get_entities_batch(ids, namespace_id=ns)

        assert result == {}
        _, kwargs = session.run.call_args
        assert kwargs["namespace_id"] == str(ns)
        cypher = session.run.call_args[0][0]
        assert "namespace_id" in cypher
        assert "$ids" in cypher

    @pytest.mark.asyncio
    async def test_get_relationship_binds_namespace(self) -> None:
        session = _mock_session(single_record=None)
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_relationship(uuid4(), namespace_id=ns)

        assert out is None
        assert session.run.call_args.kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_get_entity_relationships_constrains_both_endpoints(self) -> None:
        session = _mock_session(data_records=[])
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_entity_relationships(uuid4(), namespace_id=ns)

        assert out == []
        cypher = session.run.call_args[0][0]
        # Both endpoint Entity patterns must carry the namespace filter.
        assert cypher.count("namespace_id: $namespace_id") >= 2
        assert session.run.call_args.kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_get_episode_binds_namespace(self) -> None:
        session = _mock_session(single_record=None)
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_episode(uuid4(), namespace_id=ns)

        assert out is None
        assert session.run.call_args.kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_get_neighborhood_filters_every_hop(self) -> None:
        session = _mock_session(single_record=None)
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_neighborhood(uuid4(), namespace_id=ns)

        assert out == {"entities": [], "relationships": []}
        # Neo4j wraps the query in session.execute_read(_work); the actual
        # Cypher hits tx.run(...) inside _work.
        cypher = session._tx.run.call_args[0][0]
        assert "center:Entity {id: $entity_id, namespace_id: $namespace_id}" in cypher
        assert "other:Entity {namespace_id: $namespace_id}" in cypher
        assert session._tx.run.call_args.kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_get_neighborhoods_batch_filters_every_hop(self) -> None:
        session = _mock_session(data_records=[])
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_neighborhoods_batch([uuid4()], namespace_id=ns)

        assert out == {}
        cypher = session._tx.run.call_args[0][0]
        assert "center:Entity {id: eid, namespace_id: $namespace_id}" in cypher
        assert "other:Entity {namespace_id: $namespace_id}" in cypher
        assert session._tx.run.call_args.kwargs["namespace_id"] == str(ns)


# ---------------------------------------------------------------------------
# Memgraph (same mock shape as Neo4j — both use the neo4j driver)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemgraphNamespaceFilter:
    def _backend(self, session: MagicMock):
        from khora.storage.backends.memgraph import MemgraphBackend

        b = MemgraphBackend("bolt://localhost:7687")
        b._driver = _driver_with_session(session)
        return b

    @pytest.mark.asyncio
    async def test_get_entity_binds_namespace(self) -> None:
        session = _mock_session(single_record=None)
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_entity(uuid4(), namespace_id=ns)

        assert out is None
        assert session.run.call_args.kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_get_entities_batch_filters_by_namespace(self) -> None:
        session = _mock_session(data_records=[])
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_entities_batch([uuid4()], namespace_id=ns)

        assert out == {}
        assert "e.namespace_id = $namespace_id" in session.run.call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_entity_relationships_constrains_both_endpoints(self) -> None:
        session = _mock_session(data_records=[])
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_entity_relationships(uuid4(), namespace_id=ns)

        assert out == []
        cypher = session.run.call_args[0][0]
        assert cypher.count("namespace_id: $namespace_id") >= 2

    @pytest.mark.asyncio
    async def test_get_neighborhood_filters_every_hop(self) -> None:
        session = _mock_session(single_record=None)
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_neighborhood(uuid4(), namespace_id=ns)

        assert out == {"entities": [], "relationships": []}
        cypher = session.run.call_args[0][0]
        assert "center:Entity {id: $entity_id, namespace_id: $namespace_id}" in cypher
        assert "other:Entity {namespace_id: $namespace_id}" in cypher


# ---------------------------------------------------------------------------
# Neptune (mirrors Memgraph shape)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeptuneNamespaceFilter:
    def _backend(self, session: MagicMock):
        from khora.storage.backends.neptune import NeptuneBackend

        b = NeptuneBackend("bolt://localhost:8182")
        b._driver = _driver_with_session(session)
        return b

    @pytest.mark.asyncio
    async def test_get_entity_binds_namespace(self) -> None:
        session = _mock_session(single_record=None)
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_entity(uuid4(), namespace_id=ns)

        assert out is None
        assert session.run.call_args.kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_get_relationship_binds_namespace(self) -> None:
        session = _mock_session(single_record=None)
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_relationship(uuid4(), namespace_id=ns)

        assert out is None
        assert session.run.call_args.kwargs["namespace_id"] == str(ns)

    @pytest.mark.asyncio
    async def test_get_neighborhood_filters_both_center_and_neighbors(self) -> None:
        session = _mock_session(single_record=None)
        b = self._backend(session)
        ns = uuid4()

        out = await b.get_neighborhood(uuid4(), namespace_id=ns)

        assert out == {"entities": [], "relationships": []}
        cypher = session.run.call_args[0][0]
        assert "center:Entity {id: $entity_id, namespace_id: $namespace_id}" in cypher
        assert "other:Entity {namespace_id: $namespace_id}" in cypher


# ---------------------------------------------------------------------------
# AGE — f-string-interpolated Cypher; assert the namespace UUID is in the SQL text.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAGENamespaceFilter:
    """AGE's Cypher is f-string-interpolated today (the IDOR family will migrate to
    bind parameters). The namespace_id is a type-safe UUID at the call site,
    so all we need to verify is that the rendered Cypher text *contains*
    the caller's namespace.
    """

    def _backend(self):
        from khora.storage.backends.age import AGEBackend

        b = AGEBackend.__new__(AGEBackend)
        b._cypher = AsyncMock(return_value=[])  # type: ignore[attr-defined]

        class _DummySession:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False

            def begin(self_inner):  # noqa: D401
                class _Tx:
                    async def __aenter__(self_tx):
                        return self_tx

                    async def __aexit__(self_tx, *exc):
                        return False

                return _Tx()

            async def execute(self_inner, *_args, **_kwargs):
                return None

        b._get_session_factory = lambda: lambda: _DummySession()  # type: ignore[attr-defined]
        return b

    @pytest.mark.asyncio
    async def test_get_entity_namespace_baked_into_cypher(self) -> None:
        b = self._backend()
        ns = uuid4()
        eid = uuid4()

        out = await b.get_entity(eid, namespace_id=ns)

        assert out is None
        rendered = b._cypher.call_args[0][1]
        assert f"namespace_id: '{ns}'" in rendered
        assert str(eid) in rendered

    @pytest.mark.asyncio
    async def test_get_relationship_namespace_baked_into_cypher(self) -> None:
        b = self._backend()
        ns = uuid4()
        rid = uuid4()

        out = await b.get_relationship(rid, namespace_id=ns)

        assert out is None
        rendered = b._cypher.call_args[0][1]
        assert rendered.count(f"namespace_id: '{ns}'") >= 2

    @pytest.mark.asyncio
    async def test_get_entity_relationships_constrains_both_endpoints(self) -> None:
        b = self._backend()
        ns = uuid4()

        out = await b.get_entity_relationships(uuid4(), namespace_id=ns)

        assert out == []
        rendered = b._cypher.call_args[0][1]
        assert rendered.count(f"namespace_id: '{ns}'") >= 2

    @pytest.mark.asyncio
    async def test_get_neighborhood_filters_every_hop(self) -> None:
        b = self._backend()
        ns = uuid4()

        out = await b.get_neighborhood(uuid4(), namespace_id=ns)

        assert out == {"entities": [], "relationships": []}
        rendered = b._cypher.call_args[0][1]
        assert "center:Entity {id: '" in rendered
        assert f"namespace_id: '{ns}'" in rendered
        assert f"other:Entity {{namespace_id: '{ns}'}}" in rendered

    @pytest.mark.asyncio
    async def test_get_episode_namespace_baked_into_cypher(self) -> None:
        b = self._backend()
        ns = uuid4()

        out = await b.get_episode(uuid4(), namespace_id=ns)

        assert out is None
        rendered = b._cypher.call_args[0][1]
        assert f"namespace_id: '{ns}'" in rendered
