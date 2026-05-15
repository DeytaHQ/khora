"""Regression tests for SurrealDB graph adapter UUID-in-SQL quoting (issue #635).

The SurrealQL parser splits bare record-ID tokens like
``entity:f6c351e7-eb0f-4ef4-a90b-bd20ff90e728`` on the hyphens, interpreting
the right-hand side as arithmetic and crashing with
``Parse error: Invalid token, found unexpected character ...``.

The fix routes every record ID through either parameter binding (``$param``)
or Unicode escape brackets (``⟨...⟩``).  These tests assert the SQL that the
adapter actually emits never contains a bare hyphen-bearing record ID.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Relationship  # noqa: E402
from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter  # noqa: E402

# A UUID guaranteed to expose the bug — every UUID has hyphens, but this one
# also has a leading hex letter, which is the exact shape that produced the
# original ``Invalid token, found unexpected character 'a' after number token``
# message reported in issue #635.
_BUGGY_UUID = UUID("f6c351e7-eb0f-4ef4-a90b-bd20ff90e728")


def _bare_record_id_pattern(table: str, uid: UUID) -> re.Pattern[str]:
    """Match a bare ``table:<uuid>`` token (i.e. not wrapped in ⟨...⟩).

    The negative lookbehind / lookahead reject the SurrealDB escape brackets
    so that the SAFE ``table:⟨<uuid>⟩`` form passes.
    """
    return re.compile(rf"(?<!⟨){re.escape(table)}:{re.escape(str(uid))}(?!⟩)")


def _make_conn() -> MagicMock:
    conn = MagicMock()
    conn.connected = True
    conn.query = AsyncMock(return_value=[])
    conn.query_one = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value=None)
    return conn


def _collect_sql(mock_method: AsyncMock) -> list[str]:
    """Return every SQL string passed to a mocked query/execute method."""
    sqls: list[str] = []
    for call in mock_method.await_args_list:
        if call.args:
            sqls.append(call.args[0])
    return sqls


# ---------------------------------------------------------------------------
# get_neighborhood — the reported bug site
# ---------------------------------------------------------------------------


async def test_get_neighborhood_does_not_interpolate_bare_uuid() -> None:
    """Issue #635: ``get_neighborhood`` must not splice a bare RecordID into SQL."""
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    await adapter.get_neighborhood(_BUGGY_UUID, depth=1, limit=10)

    bare_pattern = _bare_record_id_pattern("entity", _BUGGY_UUID)
    for sql in _collect_sql(conn.query):
        assert not bare_pattern.search(sql), (
            f"SurrealQL would parse-error on bare ``entity:<uuid>`` token. Offending SQL: {sql!r}"
        )
        # And: every SQL must either bind ``$eid`` or wrap the id in
        # SurrealDB's Unicode escape brackets ⟨...⟩.
        assert ("$eid" in sql) or ("⟨" in sql), f"SQL must use $eid binding or ⟨...⟩ brackets; got: {sql!r}"


async def test_get_neighborhood_binds_center_record() -> None:
    """The center entity id should arrive as a ``$eid`` parameter, not in the SQL."""
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    await adapter.get_neighborhood(_BUGGY_UUID, depth=1, limit=10)

    # First query is the neighbourhood SELECT; second (if any) is the
    # relationships fetch.  Both must bind ``$eid``.
    first_call = conn.query.await_args_list[0]
    sql = first_call.args[0]
    bindings = first_call.args[1] if len(first_call.args) > 1 else first_call.kwargs.get("bindings")
    assert "$eid" in sql
    assert bindings is not None and "eid" in bindings


async def test_get_neighborhood_relationships_fetch_uses_bindings() -> None:
    """The follow-up relationships fetch must also bind ids, not interpolate."""
    conn = _make_conn()
    # Make the first SELECT return one neighbour so the relationship fetch runs.
    neighbour_uuid = UUID("11111111-2222-3333-4444-555555555555")
    conn.query = AsyncMock(
        side_effect=[
            [{"out_neighbors": [{"id": f"entity:⟨{neighbour_uuid}⟩", "name": "n"}]}],
            [],
        ]
    )
    adapter = SurrealDBGraphAdapter(conn)

    await adapter.get_neighborhood(_BUGGY_UUID, depth=1, limit=10)

    assert conn.query.await_count == 2
    rel_sql = conn.query.await_args_list[1].args[0]
    rel_bindings = conn.query.await_args_list[1].args[1]
    assert not _bare_record_id_pattern("entity", _BUGGY_UUID).search(rel_sql)
    assert not _bare_record_id_pattern("entity", neighbour_uuid).search(rel_sql)
    assert "$eid" in rel_sql
    assert "$neighbor_rids" in rel_sql
    assert "eid" in rel_bindings and "neighbor_rids" in rel_bindings


# ---------------------------------------------------------------------------
# Sibling defects in the same file (audit scope of issue #635)
# ---------------------------------------------------------------------------


async def test_get_neighborhoods_batch_does_not_interpolate_bare_uuid() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)
    ids = [_BUGGY_UUID, UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")]

    await adapter.get_neighborhoods_batch(ids, depth=1, limit_per_entity=5)

    for sql in _collect_sql(conn.query):
        for uid in ids:
            assert not _bare_record_id_pattern("entity", uid).search(sql), f"Bare entity:<uuid> in SQL: {sql!r}"


async def test_create_relationship_does_not_interpolate_bare_uuid() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)
    src_id = _BUGGY_UUID
    tgt_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    rel = Relationship(
        id=UUID("99999999-8888-7777-6666-555555555555"),
        namespace_id=UUID("12345678-1234-1234-1234-123456789abc"),
        source_entity_id=src_id,
        target_entity_id=tgt_id,
        relationship_type="RELATES_TO",
        description="",
    )

    await adapter.create_relationship(rel)

    for sql in _collect_sql(conn.execute):
        assert not _bare_record_id_pattern("entity", src_id).search(sql)
        assert not _bare_record_id_pattern("entity", tgt_id).search(sql)
        # Should bind both endpoints
        assert "$src" in sql and "$tgt" in sql


async def test_get_entities_batch_does_not_interpolate_bare_uuid() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)
    ids = [_BUGGY_UUID, UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")]

    await adapter.get_entities_batch(ids)

    for sql in _collect_sql(conn.query):
        for uid in ids:
            assert not _bare_record_id_pattern("entity", uid).search(sql)
        assert "$eids" in sql


async def test_get_temporal_neighbors_does_not_interpolate_bare_uuid() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)

    await adapter.get_temporal_neighbors(
        _BUGGY_UUID,
        UUID("12345678-1234-1234-1234-123456789abc"),
        max_hops=1,
        limit=5,
    )

    for sql in _collect_sql(conn.query):
        assert not _bare_record_id_pattern("entity", _BUGGY_UUID).search(sql)
        assert "$eid" in sql


async def test_create_session_links_does_not_interpolate_bare_namespace_uuid() -> None:
    conn = _make_conn()
    adapter = SurrealDBGraphAdapter(conn)
    ns_id = _BUGGY_UUID

    await adapter.create_session_links(ns_id)

    for sql in _collect_sql(conn.query):
        assert not _bare_record_id_pattern("memory_namespace", ns_id).search(sql)
        assert "$ns_rid" in sql
