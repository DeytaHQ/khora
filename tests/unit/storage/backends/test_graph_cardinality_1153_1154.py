"""Regression tests for #1153 and #1154 across the four Cypher graph backends.

#1153 - ``search_entities_by_attribute`` subscripted the ``attributes`` JSON
string as a Cypher map (``e.attributes[$attribute_name]``), which never matches
because the property is stored as a serialized JSON *string*. The fix prefilters
candidates server-side with ``CONTAINS`` on the key name and does the exact
key/value match in Python after deserializing the attribute dict, so an entity
whose JSON attribute holds the key/value is actually found (Neo4j, Memgraph,
Neptune). AGE already used a CONTAINS workaround and is unchanged for #1153.

#1154 - ``get_neighborhood`` applied ``LIMIT`` *after* ``collect(...)``
aggregation, which always yields a single row, so the limit never bounded the
neighborhood. The fix slices inside the projection
(``collect(DISTINCT other)[0..$limit]``) exactly as ``get_neighborhoods_batch``
already does, across Neo4j, Memgraph, Neptune and AGE.

All tests are mock-driven: they assert the generated Cypher shape and the
Python-side deserialize/filter behavior. No live database.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

_NS = uuid4()


# ---------------------------------------------------------------------------
# neo4j-driver shaped mock (Neo4j + Memgraph + Neptune all use the neo4j driver)
# ---------------------------------------------------------------------------


def _entity_node(attributes: dict) -> dict:
    """A node dict shaped like what the neo4j driver hands back for an Entity."""
    return {
        "id": str(uuid4()),
        "namespace_id": str(_NS),
        "name": "Alice",
        "entity_type": "PERSON",
        "description": "",
        "attributes": json.dumps(attributes),
        "metadata": json.dumps({}),
        "source_document_ids": [],
        "source_chunk_ids": [],
        "mention_count": 1,
        "confidence": 1.0,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _make_session(*, single=None, data_records=None) -> MagicMock:
    """neo4j-shaped session.

    ``run().single()`` / ``run().data()`` are awaitables. ``execute_read(work)``
    (Neo4j's managed-transaction helper) calls ``work`` with a tx whose
    ``.run()`` yields the same result, exposed as ``session._tx`` so callers can
    inspect the Cypher bound inside ``_work``.
    """
    result = MagicMock()
    result.single = AsyncMock(return_value=single)
    result.data = AsyncMock(return_value=data_records or [])

    tx = MagicMock()
    tx.run = AsyncMock(return_value=result)

    async def _execute(work, *args, **kwargs):
        return await work(tx)

    session = MagicMock()
    session.run = AsyncMock(return_value=result)
    session.execute_read = AsyncMock(side_effect=_execute)
    session._tx = tx  # noqa: SLF001
    return session


def _driver_with_session(session: MagicMock) -> MagicMock:
    @asynccontextmanager
    async def _ctx(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        yield session

    driver = MagicMock()
    driver.session = MagicMock(side_effect=_ctx)
    return driver


def _neo4j_backend(session: MagicMock):
    from khora.storage.backends.neo4j import Neo4jBackend

    return Neo4jBackend.from_driver(_driver_with_session(session), query_timeout=None)


def _memgraph_backend(session: MagicMock):
    from khora.storage.backends.memgraph import MemgraphBackend

    b = MemgraphBackend("bolt://localhost:7687")
    b._driver = _driver_with_session(session)
    return b


def _neptune_backend(session: MagicMock):
    from khora.storage.backends.neptune import NeptuneBackend

    b = NeptuneBackend("bolt://localhost:8182")
    b._driver = _driver_with_session(session)
    return b


def _age_backend():
    """AGE backend with ``_cypher`` mocked, so we can read the rendered Cypher."""
    from khora.storage.backends.age import AGEBackend

    b = AGEBackend.__new__(AGEBackend)
    b._cypher = AsyncMock(return_value=[])  # type: ignore[attr-defined]

    class _DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def begin(self):
            class _Tx:
                async def __aenter__(self_tx):
                    return self_tx

                async def __aexit__(self_tx, *exc):
                    return False

            return _Tx()

        async def execute(self, *_a, **_k):
            return None

    b._get_session_factory = lambda: lambda: _DummySession()  # type: ignore[attr-defined]
    return b


# ===========================================================================
# #1154 - get_neighborhood must bound BEFORE aggregation (slice in projection)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neo4j_get_neighborhood_slices_before_collect() -> None:
    session = _make_session(single=None)
    b = _neo4j_backend(session)

    await b.get_neighborhood(uuid4(), namespace_id=_NS, limit=7)

    # Neo4j runs the query inside execute_read(_work) -> tx.run.
    cypher = session._tx.run.call_args[0][0]
    # The slice must be applied to the collected list, mirroring the batch
    # variant. A bare ``LIMIT $limit`` after ``collect(...)`` is a no-op.
    assert "collect(DISTINCT other)[0..$limit]" in cypher
    assert session._tx.run.call_args.kwargs["limit"] == 7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memgraph_get_neighborhood_slices_before_collect() -> None:
    session = _make_session(single=None)
    b = _memgraph_backend(session)

    await b.get_neighborhood(uuid4(), namespace_id=_NS, limit=7)

    cypher = session.run.call_args[0][0]
    assert "collect(DISTINCT other)[0..$limit]" in cypher
    assert session.run.call_args.kwargs["limit"] == 7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neptune_get_neighborhood_slices_before_collect() -> None:
    session = _make_session(single=None)
    b = _neptune_backend(session)

    await b.get_neighborhood(uuid4(), namespace_id=_NS, limit=7)

    cypher = session.run.call_args[0][0]
    assert "collect(DISTINCT other)[0..$limit]" in cypher
    assert session.run.call_args.kwargs["limit"] == 7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_age_get_neighborhood_slices_before_collect() -> None:
    b = _age_backend()

    await b.get_neighborhood(uuid4(), namespace_id=_NS, limit=7)

    rendered = b._cypher.call_args[0][1]
    # AGE interpolates the limit literal; the slice must wrap the collected list.
    assert "collect(DISTINCT other)[0..7]" in rendered


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neo4j_get_neighborhood_returns_at_most_limit() -> None:
    """Behavioral guard: a hub whose (server-sliced) result has ``limit`` nodes
    is returned verbatim - the public method does not re-expand beyond it."""
    limit = 3
    record = {
        "nodes": [_entity_node({}) for _ in range(limit)],
        "relationships": [],
    }
    session = _make_session(single=record)
    b = _neo4j_backend(session)

    out = await b.get_neighborhood(uuid4(), namespace_id=_NS, limit=limit)

    assert len(out["entities"]) == limit


# ===========================================================================
# #1153 - search_entities_by_attribute must match the JSON-string storage format
# ===========================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neo4j_search_by_attribute_finds_matching_json_entity() -> None:
    """An entity whose serialized JSON attribute holds role=admin IS found."""
    match = _entity_node({"role": "admin", "team": "infra"})
    nonmatch = _entity_node({"role": "viewer"})
    session = _make_session(data_records=[{"e": match}, {"e": nonmatch}])
    b = _neo4j_backend(session)

    out = await b.search_entities_by_attribute(_NS, "role", "admin")

    assert len(out) == 1
    assert out[0].attributes["role"] == "admin"
    cypher = session.run.call_args[0][0]
    # No more map-subscript on the JSON string.
    assert "e.attributes[$attribute_name]" not in cypher
    assert "CONTAINS" in cypher


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memgraph_search_by_attribute_finds_matching_json_entity() -> None:
    match = _entity_node({"role": "admin"})
    nonmatch = _entity_node({"role": "viewer"})
    session = _make_session(data_records=[{"e": match}, {"e": nonmatch}])
    b = _memgraph_backend(session)

    out = await b.search_entities_by_attribute(_NS, "role", "admin")

    assert len(out) == 1
    assert out[0].attributes["role"] == "admin"
    cypher = session.run.call_args[0][0]
    assert "e.attributes[$attribute_name]" not in cypher
    assert "CONTAINS" in cypher


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neptune_search_by_attribute_finds_matching_json_entity() -> None:
    match = _entity_node({"role": "admin"})
    nonmatch = _entity_node({"role": "viewer"})
    session = _make_session(data_records=[{"e": match}, {"e": nonmatch}])
    b = _neptune_backend(session)

    out = await b.search_entities_by_attribute(_NS, "role", "admin")

    assert len(out) == 1
    assert out[0].attributes["role"] == "admin"
    cypher = session.run.call_args[0][0]
    assert "e.attributes[$attribute_name]" not in cypher
    assert "CONTAINS" in cypher


@pytest.mark.unit
@pytest.mark.asyncio
async def test_neo4j_search_by_attribute_matches_non_string_value() -> None:
    """A numeric attribute value matches by exact equality, not substring."""
    match = _entity_node({"level": 5})
    nonmatch = _entity_node({"level": 50})  # substring "5" would false-match
    session = _make_session(data_records=[{"e": match}, {"e": nonmatch}])
    b = _neo4j_backend(session)

    out = await b.search_entities_by_attribute(_NS, "level", 5)

    assert len(out) == 1
    assert out[0].attributes["level"] == 5
