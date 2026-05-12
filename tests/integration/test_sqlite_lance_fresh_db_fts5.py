"""Fresh-DB FTS5 trigger verification for the sqlite_lance backend.

Silent-empty FTS5 search is one of the worst user-facing bugs: every
``recall`` call returns zero BM25 hits with no error message. The
chunks_fts virtual table and three triggers (AFTER INSERT/UPDATE/DELETE)
come from migration 002. If any of them is silently skipped, BM25 search
goes dark forever.

This module runs ``run_migrations`` against an empty tmp_path SQLite file
and asserts (a) the FTS5 table exists, (b) all three triggers exist, and
(c) inserting a chunk via SQL actually propagates into chunks_fts (the
trigger fires).
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import aiosqlite  # noqa: F401

    _HAS_AIOSQLITE = True
except ImportError:
    _HAS_AIOSQLITE = False

from khora.db.session import run_migrations

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_AIOSQLITE, reason="aiosqlite not installed"),
]


async def _migrate(tmp_path: Path) -> str:
    db_path = str(tmp_path / "khora.db")
    result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    assert result.success, f"migrations failed: {result.error}"
    return db_path


async def test_fresh_db_creates_fts5_virtual_table(tmp_path: Path) -> None:
    """Migration 002 must create ``chunks_fts`` as a virtual FTS5 table."""
    import aiosqlite

    db_path = await _migrate(tmp_path)

    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT type, sql FROM sqlite_master WHERE name = 'chunks_fts'")
        rows = await cur.fetchall()

    assert rows, "chunks_fts virtual table is missing — migration 002 didn't run"
    obj_type, sql = rows[0]
    assert obj_type == "table"
    assert sql is not None and "VIRTUAL TABLE" in sql.upper()
    assert "fts5" in sql.lower(), f"chunks_fts isn't an fts5 table: {sql!r}"


@pytest.mark.parametrize("trigger_name", ["chunks_ai", "chunks_au", "chunks_ad"])
async def test_fresh_db_creates_fts5_triggers(tmp_path: Path, trigger_name: str) -> None:
    """Each of the three AFTER INSERT/UPDATE/DELETE triggers must exist."""
    import aiosqlite

    db_path = await _migrate(tmp_path)

    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (trigger_name,),
        )
        row = await cur.fetchone()

    assert row is not None, (
        f"FTS5 sync trigger {trigger_name!r} missing — chunks_fts will go stale "
        f"on insert/update/delete and BM25 search will silently return empty"
    )


async def test_fresh_db_insert_propagates_to_fts5(tmp_path: Path) -> None:
    """Inserting a chunk via raw SQL must propagate the content to chunks_fts.

    This is the end-to-end check: if any of the trigger statements is malformed
    or skipped, the INSERT trigger fires but the row never appears in the FTS5
    table and ``MATCH 'token'`` returns nothing.
    """
    import aiosqlite

    db_path = await _migrate(tmp_path)
    ns_id = "11111111-1111-1111-1111-111111111111"
    doc_id = "22222222-2222-2222-2222-222222222222"
    chunk_id = "33333333-3333-3333-3333-333333333333"

    async with aiosqlite.connect(db_path) as conn:
        # We're testing the FTS5 trigger, not FK enforcement — disabling FKs
        # lets us insert directly into chunks without the workspaces /
        # memory_namespaces / documents chain. The trigger fires on
        # ``AFTER INSERT ON chunks`` regardless.
        await conn.execute("PRAGMA foreign_keys = OFF")
        await conn.execute(
            "INSERT INTO chunks (id, namespace_id, document_id, content, chunk_index, "
            "metadata, created_at) "
            "VALUES (?, ?, ?, ?, 0, '{}', datetime('now'))",
            (chunk_id, ns_id, doc_id, "zettabyte mentions appear here"),
        )
        await conn.commit()

        # The AFTER INSERT trigger should have propagated content into chunks_fts.
        cur = await conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("zettabyte",),
        )
        fts_hits = await cur.fetchall()

    assert fts_hits, (
        "AFTER INSERT trigger didn't fire (or chunks_fts is empty): inserting a chunk "
        "with content 'zettabyte mentions...' did not produce an FTS5 match. BM25 "
        "search is silently broken."
    )
