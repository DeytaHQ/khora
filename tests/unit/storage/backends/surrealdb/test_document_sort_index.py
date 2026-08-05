"""Regression tests for the ``document`` sort index DDL (PR #1581).

The SurrealDB half of that change is two lines of DDL inside
``_TABLE_DEFINITIONS``: it defines ``idx_document_ns_created`` on
``(namespace_id, created_at)`` and removes ``idx_document_namespace``, the
one-field index that is a strict prefix of it.

Nothing caught either statement failing. ``initialize_schema`` hands the whole
multi-statement blob to ``conn.execute()``, which does not surface a
per-statement DDL error: a ``DEFINE`` that never applies still leaves
``SurrealDB schema initialized successfully`` in the log. These tests assert
the two indexes' *actual* post-init state via ``INFO FOR TABLE document``, so
a statement that stops parsing (the fate of the ``FLEXIBLE TYPE`` clauses on
3.x servers, #1584) fails here instead of silently degrading a deployment.

Read the scope honestly: these are *existence* assertions, not *selection*
assertions. ``idx_document_ns_created`` is **not selected by any planner
reachable from this repo**. On embedded 2.0.0 a bare ``WHERE namespace_id =
$ns`` prefix-matches whichever ``(namespace_id, …)`` composite sorts first by
name, which is ``idx_document_ns_checksum``. A test pinning the sort index as
the chosen plan fails today, on this branch. So a green run here means the DDL
applied — it does not mean the index earns its write cost. The selection test
becomes meaningful only on a server that can parse this DDL (#1584); it is
deliberately not written as one now, because pinning the current winner would
harden an undocumented, name-ordered planner tie-break into a test.

They run against ``memory://`` — no server, no API key.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

pytest.importorskip("surrealdb")

from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402

pytestmark = pytest.mark.unit

SORT_INDEX = "idx_document_ns_created"
LEGACY_INDEX = "idx_document_namespace"


async def _document_indexes(conn: SurrealDBConnection) -> dict[str, str]:
    """Return ``{index_name: definition}`` for the ``document`` table."""
    info: Any = await conn.query_one("INFO FOR TABLE document")
    return dict(info["indexes"])


async def _connected() -> SurrealDBConnection:
    """A ``memory://`` connection with khora's real schema applied."""
    conn = SurrealDBConnection(mode="memory", sync_data=False)
    await conn.connect()
    return conn


async def test_sort_index_is_actually_defined() -> None:
    """``idx_document_ns_created`` exists on ``(namespace_id, created_at)``."""
    conn = await _connected()
    try:
        indexes = await _document_indexes(conn)
        assert SORT_INDEX in indexes, (
            f"{SORT_INDEX} missing after schema init; document indexes are "
            f"{sorted(indexes)}. initialize_schema swallows per-statement DDL "
            "errors, so a non-parsing DEFINE looks like success in the log."
        )
        definition = indexes[SORT_INDEX]
        assert "namespace_id" in definition and "created_at" in definition, definition
    finally:
        await conn.disconnect()


async def test_legacy_prefix_index_is_removed() -> None:
    """The one-field ``idx_document_namespace`` is gone after schema init.

    It is a strict prefix of the sort index. Leaving it in place costs write
    maintenance for no read benefit, and it is the finding the raw-SQLite half
    of #1581 closes by dropping its own equivalent.

    Weak by construction: on a *fresh* database the index was never defined,
    so this passes whether or not the ``REMOVE`` statement is present — it was
    confirmed to survive deleting that statement. It guards only against the
    ``DEFINE`` coming back. The load-bearing checks are
    ``test_existing_database_drops_the_legacy_index_on_connect`` (behavioural)
    and ``test_ddl_creates_before_it_drops_and_keeps_no_stale_define``
    (source-level).
    """
    conn = await _connected()
    try:
        indexes = await _document_indexes(conn)
        assert LEGACY_INDEX not in indexes, (
            f"{LEGACY_INDEX} still defined after schema init; document indexes are {sorted(indexes)}"
        )
    finally:
        await conn.disconnect()


async def test_existing_database_drops_the_legacy_index_on_connect() -> None:
    """An older database that already carries the legacy index converges.

    The blob re-executes on every ``connect()``, so an upgrade path is the
    normal path here: the pre-existing index must be removed, and the rows it
    used to serve must all still be reachable afterwards.
    """
    conn = await _connected()
    try:
        # Recreate the pre-#1581 state on a database that already holds rows.
        await conn.execute(f"DEFINE INDEX IF NOT EXISTS {LEGACY_INDEX} ON document FIELDS namespace_id")
        assert LEGACY_INDEX in await _document_indexes(conn)
        for i in range(20):
            await conn.execute(
                "CREATE type::thing('document', $id) SET namespace_id = $ns, title = $t, created_at = time::now()",
                {"id": f"doc{i:03d}", "ns": "ns-a", "t": f"t{i}"},
            )

        # Re-run schema init the way a later connect() would.
        from khora.storage.backends.surrealdb.schema import initialize_schema

        await initialize_schema(conn)

        indexes = await _document_indexes(conn)
        assert LEGACY_INDEX not in indexes, sorted(indexes)
        assert SORT_INDEX in indexes, sorted(indexes)

        rows = await conn.query(
            "SELECT id, created_at FROM document WHERE namespace_id = $ns ORDER BY created_at DESC, id DESC",
            {"ns": "ns-a"},
        )
        assert len(rows) == 20, f"rows lost after dropping {LEGACY_INDEX}: {len(rows)}/20"
    finally:
        await conn.disconnect()


def test_ddl_creates_before_it_drops_and_keeps_no_stale_define() -> None:
    """Source-level guard on statement order and on the removed ``DEFINE``.

    ``INFO FOR TABLE`` cannot see either of these: both a swapped
    create/drop order and a legacy ``DEFINE`` left beside the ``REMOVE``
    converge on the same final index set. The order matters because a
    connect that dies between the two statements should leave more index
    coverage than needed, not less; the stale ``DEFINE`` matters because the
    blob re-executes on every ``connect()``, so side by side the pair would
    rebuild and destroy the index each time.
    """
    from khora.storage.backends.surrealdb.schema import _TABLE_DEFINITIONS

    statements = [s.strip() for s in _TABLE_DEFINITIONS.split("\n") if s.strip() and not s.strip().startswith("--")]
    defines = [i for i, s in enumerate(statements) if re.match(rf"DEFINE INDEX (IF NOT EXISTS )?{SORT_INDEX}\b", s)]
    removes = [i for i, s in enumerate(statements) if re.match(rf"REMOVE INDEX (IF EXISTS )?{LEGACY_INDEX}\b", s)]
    stale = [s for s in statements if re.match(rf"DEFINE INDEX .*\b{LEGACY_INDEX}\b", s)]

    assert len(defines) == 1, f"expected exactly one DEFINE of {SORT_INDEX}, got {defines}"
    assert len(removes) == 1, f"expected exactly one REMOVE of {LEGACY_INDEX}, got {removes}"
    assert not stale, f"legacy DEFINE left beside the REMOVE: {stale}"
    assert defines[0] < removes[0], (
        f"{SORT_INDEX} must be defined before {LEGACY_INDEX} is removed "
        f"(define at {defines[0]}, remove at {removes[0]})"
    )


async def test_namespace_only_query_still_uses_an_index() -> None:
    """A bare ``namespace_id`` filter must not fall back to a table scan.

    Dropping the one-field index only works because the planner prefix-matches
    the leading column of a two-field composite. This is the guard on that: if
    a future engine stops prefix-matching, ``list_documents`` silently becomes
    a full table scan, and this test is what says so.
    """
    conn = await _connected()
    try:
        for i in range(20):
            await conn.execute(
                "CREATE type::thing('document', $id) SET namespace_id = $ns, created_at = time::now()",
                {"id": f"doc{i:03d}", "ns": "ns-a"},
            )
        plan: Any = await conn.query(
            "SELECT * FROM document WHERE namespace_id = $ns ORDER BY created_at DESC, id DESC LIMIT 5 START 0 EXPLAIN",
            {"ns": "ns-a"},
        )
        iterate = plan[0]
        assert iterate["operation"] == "Iterate Index", (
            f"namespace-only document listing no longer uses an index: {plan}"
        )
        # Which composite wins is chosen by index name on the engines reachable
        # today, so this deliberately does not pin the specific index.
        assert iterate["detail"]["plan"]["index"].startswith("idx_document_ns"), plan
    finally:
        await conn.disconnect()
