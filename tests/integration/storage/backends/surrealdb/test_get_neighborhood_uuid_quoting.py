"""End-to-end repro of issue #635 against an in-memory SurrealDB instance.

The unit test :mod:`tests.unit.storage.backends.surrealdb.test_graph_uuid_quoting`
covers the SQL-shape contract.  This test verifies the *actual SurrealQL
parser* does not reject the queries the adapter emits — the original symptom
was a ``RuntimeError: Parse error: Invalid token, found unexpected character
'a' after number token`` raised from inside the SDK, which only surfaces when
the query reaches the real engine.
"""

from __future__ import annotations

from uuid import UUID

import pytest

pytest.importorskip("surrealdb")

from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter  # noqa: E402

pytestmark = pytest.mark.integration

# UUID that historically triggered the parse error (issue #635).
_BUGGY_UUID = UUID("f6c351e7-eb0f-4ef4-a90b-bd20ff90e728")
_NS_ID = UUID("11111111-2222-3333-4444-555555555555")


async def test_get_neighborhood_does_not_parse_error_on_memory_surrealdb() -> None:
    """Direct repro of the ticket: ``get_neighborhood`` against memory:// SurrealDB.

    Before the fix this raised ``RuntimeError: There was a problem with the
    database: Parse error: Invalid token, found unexpected character`` because
    the centre RecordID was interpolated directly into the SQL, exposing the
    UUID hyphens to the SurrealQL tokenizer.

    No entity needs to exist for this assertion — the parser runs before any
    row lookup, so an empty database is sufficient to reproduce the failure
    (and to prove the fix).
    """
    conn = SurrealDBConnection(mode="memory", namespace="test", database="test")
    await conn.connect()
    adapter = SurrealDBGraphAdapter(conn)
    try:
        result = await adapter.get_neighborhood(_BUGGY_UUID, namespace_id=_NS_ID, depth=1, limit=10)
        assert isinstance(result, dict)
        assert result == {"entities": [], "relationships": []}
    finally:
        await conn.disconnect()


async def test_get_neighborhoods_batch_does_not_parse_error_on_memory_surrealdb() -> None:
    """Same coverage for the batch variant — same defect lived there too."""
    conn = SurrealDBConnection(mode="memory", namespace="test", database="test")
    await conn.connect()
    adapter = SurrealDBGraphAdapter(conn)
    try:
        result = await adapter.get_neighborhoods_batch(
            [_BUGGY_UUID, UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")],
            namespace_id=_NS_ID,
            depth=1,
            limit_per_entity=5,
        )
        assert isinstance(result, dict)
        # Empty graph → every requested id maps to an empty neighbourhood.
        for nb in result.values():
            assert nb == {"entities": [], "relationships": []}
    finally:
        await conn.disconnect()
