"""Namespace-scoping tests for SurrealDB backend read methods (IGR-221 / IGR-223).

These tests assert that read methods filter at the SurrealQL layer on
``namespace_id`` (or the equivalent ``namespace`` record link) and that
the wrong-namespace case returns the empty / not-found result *without*
relying on the underlying connection to enforce isolation.

The connection is mocked: we feed it pre-shaped SurrealDB rows and
verify the adapter's SQL + bindings carry the namespace predicate, and
that any row not belonging to the caller's namespace is dropped.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

pytest.importorskip("surrealdb")

from khora.storage.backends.surrealdb.event_store import (  # noqa: E402
    SurrealDBEventStoreAdapter,
)
from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter  # noqa: E402
from khora.storage.backends.surrealdb.relational import (  # noqa: E402
    SurrealDBRelationalAdapter,
)
from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter  # noqa: E402

NS_A = UUID("11111111-1111-1111-1111-111111111111")
NS_B = UUID("22222222-2222-2222-2222-222222222222")

DOC_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
ENT_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
REL_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
EPI_ID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
RES_ID = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")


def _make_conn() -> MagicMock:
    """Build a SurrealDBConnection mock with the three usual entry points."""
    conn = MagicMock()
    conn.connected = True
    conn.query = AsyncMock(return_value=[])
    conn.query_one = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value=None)
    return conn


def _last_call(mock: AsyncMock) -> tuple[str, dict]:
    """Return (sql, bindings) for the most recent call to the mock."""
    call = mock.await_args_list[-1]
    sql = call.args[0]
    bindings = call.args[1] if len(call.args) > 1 else call.kwargs.get("bindings", {})
    return sql, bindings


# ---------------------------------------------------------------------------
# Relational adapter
# ---------------------------------------------------------------------------


async def test_get_document_filters_on_namespace_in_sql() -> None:
    conn = _make_conn()
    adapter = SurrealDBRelationalAdapter(conn)

    # Wrong-namespace case: connection returns no row.
    conn.query_one = AsyncMock(return_value=None)
    result = await adapter.get_document(DOC_ID, namespace_id=NS_B)
    assert result is None

    sql, bindings = _last_call(conn.query_one)
    assert "namespace_id = $ns" in sql
    assert bindings["ns"] == str(NS_B)


async def test_get_documents_batch_filters_on_namespace_in_sql() -> None:
    conn = _make_conn()
    adapter = SurrealDBRelationalAdapter(conn)

    conn.query = AsyncMock(return_value=[])  # nothing matches
    result = await adapter.get_documents_batch([DOC_ID], namespace_id=NS_B)
    assert result == {}

    sql, bindings = _last_call(conn.query)
    assert "namespace_id = $ns" in sql
    assert bindings["ns"] == str(NS_B)


async def test_get_documents_batch_returns_empty_on_empty_input() -> None:
    conn = _make_conn()
    adapter = SurrealDBRelationalAdapter(conn)

    result = await adapter.get_documents_batch([], namespace_id=NS_A)
    assert result == {}
    # No query was issued for an empty input.
    assert conn.query.await_count == 0


async def test_get_document_sources_batch_filters_on_namespace_in_sql() -> None:
    conn = _make_conn()
    adapter = SurrealDBRelationalAdapter(conn)

    conn.query = AsyncMock(return_value=[])
    result = await adapter.get_document_sources_batch([DOC_ID], namespace_id=NS_B)
    assert result == {}

    sql, bindings = _last_call(conn.query)
    assert "namespace_id = $ns" in sql
    assert bindings["ns"] == str(NS_B)


# ---------------------------------------------------------------------------
# Vector adapter
# ---------------------------------------------------------------------------


async def test_vector_entity_exists_filters_on_namespace() -> None:
    conn = _make_conn()
    adapter = SurrealDBVectorAdapter(conn)

    # Connection returns count=0 for the wrong namespace.
    conn.query_one = AsyncMock(return_value={"cnt": 0})
    result = await adapter.entity_exists(ENT_ID, namespace_id=NS_B)
    assert result is False

    sql, bindings = _last_call(conn.query_one)
    # We MUST be filtering on namespace, not just on id.
    assert "namespace" in sql
    assert bindings["ns_str"] == str(NS_B)


async def test_vector_get_entity_requires_and_filters_namespace() -> None:
    """Regression: the kwarg was previously ignored — assert it now filters."""
    conn = _make_conn()
    adapter = SurrealDBVectorAdapter(conn)

    conn.query_one = AsyncMock(return_value=None)
    result = await adapter.get_entity(ENT_ID, namespace_id=NS_B)
    assert result is None

    sql, bindings = _last_call(conn.query_one)
    # The SQL must reference namespace, not just the bare id.
    assert "namespace" in sql
    assert bindings["ns_str"] == str(NS_B)
    assert bindings["ns_rid"] is not None


def test_vector_get_entity_kwarg_is_required() -> None:
    """The namespace_id kwarg is now required — no silent default."""
    conn = _make_conn()
    adapter = SurrealDBVectorAdapter(conn)

    with pytest.raises(TypeError):
        # positional ``namespace_id`` is rejected by the keyword-only signature
        # and omitting it entirely raises TypeError at call time.
        adapter.get_entity(ENT_ID)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Graph adapter
# ---------------------------------------------------------------------------


async def test_graph_get_entity_filters_on_namespace() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    conn.query_one = AsyncMock(return_value=None)
    result = await adapter.get_entity(ENT_ID, namespace_id=NS_B)
    assert result is None

    sql, bindings = _last_call(conn.query_one)
    assert "namespace" in sql
    assert bindings["ns_str"] == str(NS_B)


async def test_graph_get_entities_batch_filters_on_namespace() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    conn.query = AsyncMock(return_value=[])
    result = await adapter.get_entities_batch([ENT_ID], namespace_id=NS_B)
    assert result == {}

    sql, bindings = _last_call(conn.query)
    assert "namespace" in sql
    assert bindings["ns_str"] == str(NS_B)


async def test_graph_get_relationship_filters_on_namespace() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    conn.query_one = AsyncMock(return_value=None)
    result = await adapter.get_relationship(REL_ID, namespace_id=NS_B)
    assert result is None

    sql, bindings = _last_call(conn.query_one)
    assert "namespace_id = $ns" in sql
    assert bindings["ns"] == str(NS_B)


async def test_graph_get_entity_relationships_filters_on_namespace() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    conn.query = AsyncMock(return_value=[])
    result = await adapter.get_entity_relationships(ENT_ID, namespace_id=NS_B)
    assert result == []

    sql, bindings = _last_call(conn.query)
    # Edge-side filter on namespace_id must be present.
    assert "namespace_id = $ns" in sql
    assert bindings["ns"] == str(NS_B)


async def test_graph_get_episode_filters_on_namespace() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    conn.query_one = AsyncMock(return_value=None)
    result = await adapter.get_episode(EPI_ID, namespace_id=NS_B)
    assert result is None

    sql, bindings = _last_call(conn.query_one)
    assert "namespace" in sql
    assert bindings["ns_str"] == str(NS_B)


async def test_graph_get_neighborhood_returns_empty_when_seed_in_other_ns() -> None:
    """If the seed entity lives in another namespace, return an empty result."""
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    # The seed-entity verification SELECT returns nothing — i.e. the seed
    # does not belong to NS_B. The adapter must short-circuit.
    conn.query_one = AsyncMock(return_value=None)
    result = await adapter.get_neighborhood(ENT_ID, namespace_id=NS_B, depth=1, limit=5)
    assert result == {"entities": [], "relationships": []}

    # The seed lookup MUST filter on namespace.
    sql, bindings = _last_call(conn.query_one)
    assert "namespace" in sql
    assert bindings["ns_str"] == str(NS_B)
    # And the traversal SELECT must not have been issued.
    assert conn.query.await_count == 0


async def test_graph_get_neighborhood_traversal_filters_each_hop() -> None:
    """When the seed exists, every hop must carry the namespace predicate."""
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    # Pretend the seed exists.
    conn.query_one = AsyncMock(
        return_value={
            "id": f"entity:⟨{ENT_ID}⟩",
            "name": "seed",
            "entity_type": "CONCEPT",
            "namespace": f"memory_namespace:⟨{NS_A}⟩",
            "namespace_id": str(NS_A),
        }
    )
    conn.query = AsyncMock(return_value=[])

    await adapter.get_neighborhood(ENT_ID, namespace_id=NS_A, depth=2, limit=5)

    # Inspect the traversal SELECT.
    assert conn.query.await_count >= 1
    sql, bindings = conn.query.await_args_list[0].args[0], conn.query.await_args_list[0].args[1]
    # Each hop (and the relationship filter) must reference namespace.
    assert "namespace_id = $ns_str" in sql
    assert "namespace = $ns_rid OR namespace.namespace_id = $ns_str" in sql
    assert bindings["ns_str"] == str(NS_A)


async def test_graph_get_neighborhood_drops_foreign_namespace_rows_defensively() -> None:
    """Even if the connection returns a foreign-namespace neighbour, drop it."""
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    foreign_id = UUID("ff111111-1111-1111-1111-111111111111")
    # Seed exists in NS_A.
    conn.query_one = AsyncMock(
        return_value={
            "id": f"entity:⟨{ENT_ID}⟩",
            "name": "seed",
            "entity_type": "CONCEPT",
            "namespace": f"memory_namespace:⟨{NS_A}⟩",
            "namespace_id": str(NS_A),
        }
    )
    # Traversal returns a neighbour that claims a different namespace.
    conn.query = AsyncMock(
        side_effect=[
            [
                {
                    "out_neighbors": [
                        {
                            "id": f"entity:⟨{foreign_id}⟩",
                            "name": "foreign",
                            "namespace_id": str(NS_B),
                        }
                    ],
                    "in_neighbors": [],
                }
            ],
            [],  # Relationship fetch returns nothing.
        ]
    )

    result = await adapter.get_neighborhood(ENT_ID, namespace_id=NS_A, depth=1, limit=5)
    # Foreign-namespace neighbour must be dropped client-side as a defence
    # in depth, regardless of what the SurrealDB filter happened to do.
    assert result["entities"] == []


async def test_graph_get_neighborhoods_batch_filters_on_namespace() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    conn.query = AsyncMock(return_value=[])  # no seeds in NS_B
    result = await adapter.get_neighborhoods_batch([ENT_ID], namespace_id=NS_B, depth=1, limit_per_entity=5)
    # Seeds outside NS_B are dropped — the result map keeps each requested
    # id mapped to an empty neighborhood.
    assert result == {ENT_ID: {"entities": [], "relationships": []}}

    sql, bindings = _last_call(conn.query)
    assert "namespace = $ns_rid OR namespace.namespace_id = $ns_str" in sql
    assert bindings["ns_str"] == str(NS_B)


# ---------------------------------------------------------------------------
# Event store adapter
# ---------------------------------------------------------------------------


async def test_event_store_get_events_for_resource_filters_on_namespace() -> None:
    conn = _make_conn()
    adapter = SurrealDBEventStoreAdapter(conn)

    conn.query = AsyncMock(return_value=[])
    result = await adapter.get_events_for_resource("document", RES_ID, namespace_id=NS_B)
    assert result == []

    sql, bindings = _last_call(conn.query)
    assert "namespace_id = $namespace_id" in sql
    assert bindings["namespace_id"] == str(NS_B)


async def test_event_store_get_latest_event_filters_on_namespace() -> None:
    conn = _make_conn()
    adapter = SurrealDBEventStoreAdapter(conn)

    conn.query_one = AsyncMock(return_value=None)
    result = await adapter.get_latest_event("document", RES_ID, namespace_id=NS_B)
    assert result is None

    sql, bindings = _last_call(conn.query_one)
    assert "namespace_id = $namespace_id" in sql
    assert bindings["namespace_id"] == str(NS_B)
