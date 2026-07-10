"""Coverage for migration ``053_khora_chunks_bookkeeping_to_chunker_info``.

The writer/reader refactor (khora#1491) moved the four chunk-bookkeeping keys
``chunk_index`` / ``start_char`` / ``end_char`` / ``token_count`` out of
``khora_chunks.metadata`` and into ``khora_chunks.chunker_info``, and the
temporal-chunk reader now sources them from ``chunker_info`` *exclusively*
(no metadata fallback). Migration 053 is the backfill companion: it relocates
those keys on every existing ``khora_chunks`` row, on BOTH Postgres and SQLite.

Two tiers, keyed on whether the row has a twin in the main ``chunks`` table
(``chunks.id = khora_chunks.id``):

* Tier 1 (twin exists): copy the twin's *typed* column values into
  ``chunker_info`` (merged after the twin's own ``chunker_info``, bookkeeping
  stamped last), and strip a metadata key ONLY where its value equals the
  typed column value — a differing value is a user key that stays.
* Tier 2 (no twin): read from ``metadata`` itself, moving a key only when its
  value is number-typed; a string-typed collision stays. Strip exactly the
  moved keys.

Both tiers are guarded so the relocation is convergent — re-running touches
zero rows (``metadata`` still carries a key AND ``chunker_info`` lacks
``chunk_index``).

The archetypes exercised on both dialects:

1. Tier-1, metadata values == typed columns → all four stripped from metadata;
   chunker_info = twin chunker_info merged with the four column values.
2. Tier-1 user collision (metadata ``chunk_index`` differs from the column) →
   that metadata key PRESERVED; chunker_info still gets the column value.
3. Tier-2 twinless, number-typed metadata → moved into chunker_info + stripped.
4. Tier-2 twinless, string-typed ``chunk_index`` → left in metadata untouched.
5. Post-fix clean row (chunker_info already has ``chunk_index``, user metadata
   carries a ``chunk_index`` of its own) → untouched by the guard.
6. Three-way merge precedence: the same bookkeeping key (``start_char``) is
   present in kc.chunker_info (stale base), the twin's chunker_info, AND the
   bookkeeping overlay (the typed column) with three distinct values → the
   column value wins last, pinning the ``kc || c.chunker_info || bookkeeping``
   nesting order.

The SQLite lane runs the real relocation with the JSON1 functions and needs no
Docker — it is the primary coverage and is marked ``unit``. The Postgres lane
runs the same archetypes against real JSONB and skips when Postgres is
unreachable.

Run the Postgres lane locally::

    make dev    # postgres on port 5434
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/db/test_migration_053_bookkeeping_to_chunker_info.py \
        -v -m integration --no-cov
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from khora.db.session import run_migrations

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


def _pg_reachable() -> bool:
    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"

_HEAD_REVISION = "053_khora_chunks_bookkeeping_to_chunker_info"
_PREV_REVISION = "052_entities_source_chunk_ids_gin"

_TSV_TRIGGER = "khora_chunks_content_tsv_update"

# Stable UUIDs for the archetype rows so assertions can target each.
_ID_TIER1_MATCH = UUID("00000000-0000-0000-0000-000000000001")
_ID_TIER1_COLLISION = UUID("00000000-0000-0000-0000-000000000002")
_ID_TIER2_NUMBER = UUID("00000000-0000-0000-0000-000000000003")
_ID_TIER2_STRING = UUID("00000000-0000-0000-0000-000000000004")
_ID_CLEAN = UUID("00000000-0000-0000-0000-000000000005")
# Three-way precedence archetype: the SAME bookkeeping key (start_char) appears
# in kc.chunker_info (stale base), the twin's chunker_info, AND the bookkeeping
# overlay (the typed column) with three DISTINCT values — the column value must
# win last. Uses start_char (not chunk_index) as the colliding key so the
# ``NOT (chunker_info ? 'chunk_index')`` guard still fires (chunk_index is
# deliberately absent from this row's seeded chunker_info).
_ID_TIER1_PRECEDENCE = UUID("00000000-0000-0000-0000-000000000006")

# Three distinct start_char values for the precedence row.
_PREC_BASE_START_CHAR = 111  # kc.chunker_info (stale base) — must lose
_PREC_TWIN_START_CHAR = 222  # twin chunks.chunker_info — must lose
_PREC_COLUMN_START_CHAR = 333  # bookkeeping (typed column c.start_char) — WINS

_NS = UUID("00000000-0000-0000-0000-0000000000aa")


def _make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    # Alembic uses configparser.BasicInterpolation; escape any literal '%' in
    # the URL so it isn't read as a config-interpolation token.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


# ---------------------------------------------------------------------------
# SQLite lane — the real relocation via JSON1, no Docker (primary coverage)
# ---------------------------------------------------------------------------

# A runtime-shaped ``khora_chunks`` on SQLite: only the columns 053 touches or
# joins on. ``id`` is stored as TEXT (matching how the embedded store persists
# UUIDs) so the twin join is a plain string equality.
_SQLITE_KHORA_CHUNKS_DDL = """
CREATE TABLE khora_chunks (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    chunker_info TEXT NOT NULL DEFAULT '{}'
)
"""

# A minimal ``chunks`` main table with the four typed columns + chunker_info.
_SQLITE_CHUNKS_DDL = """
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    start_char INTEGER NOT NULL DEFAULT 0,
    end_char INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    chunker_info TEXT NOT NULL DEFAULT '{}'
)
"""

# khora_chunks is NOT Alembic-managed, so alembic never creates it. We build the
# two tables ourselves, stamp the version at 052, then run ``upgrade 053``.
_ALEMBIC_VERSION_DDL = """
CREATE TABLE khora_alembic_version (
    version_num VARCHAR(64) NOT NULL,
    CONSTRAINT khora_alembic_version_pkc PRIMARY KEY (version_num)
)
"""


async def _seed_sqlite(url: str) -> None:
    """Create both tables (stamped at 052) and seed the archetype rows."""
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text(_ALEMBIC_VERSION_DDL))
            await conn.execute(
                sa.text("INSERT INTO khora_alembic_version (version_num) VALUES (:v)"),
                {"v": _PREV_REVISION},
            )
            await conn.execute(sa.text(_SQLITE_KHORA_CHUNKS_DDL))
            await conn.execute(sa.text(_SQLITE_CHUNKS_DDL))

            # Twin in the main chunks table for the two Tier-1 rows.
            for cid in (_ID_TIER1_MATCH, _ID_TIER1_COLLISION):
                await conn.execute(
                    sa.text(
                        "INSERT INTO chunks "
                        "(id, namespace_id, document_id, content, chunk_index, "
                        " start_char, end_char, token_count, chunker_info) "
                        "VALUES (:id, :ns, :doc, 'c', 3, 10, 40, 7, :ci)"
                    ),
                    {
                        "id": str(cid),
                        "ns": str(_NS),
                        "doc": str(uuid4()),
                        "ci": json.dumps({"chunker": "sentence", "version": "1"}),
                    },
                )

            # 1. Tier-1 match: metadata values == typed columns (+ a user key).
            await _insert_khora_chunk(
                conn,
                _ID_TIER1_MATCH,
                metadata={"chunk_index": 3, "start_char": 10, "end_char": 40, "token_count": 7, "author": "alice"},
                chunker_info={},
            )
            # 2. Tier-1 collision: metadata chunk_index differs from the column.
            await _insert_khora_chunk(
                conn,
                _ID_TIER1_COLLISION,
                metadata={"chunk_index": 999, "start_char": 10, "end_char": 40, "token_count": 7},
                chunker_info={},
            )
            # 3. Tier-2 twinless, number-typed metadata (no chunks row).
            await _insert_khora_chunk(
                conn,
                _ID_TIER2_NUMBER,
                metadata={"chunk_index": 5, "start_char": 0, "end_char": 12, "token_count": 4, "topic": "x"},
                chunker_info={},
            )
            # 4. Tier-2 twinless, string-typed chunk_index → stays in metadata.
            await _insert_khora_chunk(
                conn,
                _ID_TIER2_STRING,
                metadata={"chunk_index": "abc"},
                chunker_info={},
            )
            # 5. Post-fix clean row: chunker_info already has chunk_index; a
            #    user metadata.chunk_index of its own must be left untouched.
            await _insert_khora_chunk(
                conn,
                _ID_CLEAN,
                metadata={"chunk_index": 42},
                chunker_info={"chunk_index": 1, "start_char": 0, "end_char": 5, "token_count": 2},
            )

            # 6. Three-way precedence: dedicated twin whose typed start_char
            #    column (C=333) differs from both the twin's own chunker_info
            #    start_char (B=222) and the kc.chunker_info base (A=111). The
            #    column value must win last (kc || twin || bookkeeping).
            await conn.execute(
                sa.text(
                    "INSERT INTO chunks "
                    "(id, namespace_id, document_id, content, chunk_index, "
                    " start_char, end_char, token_count, chunker_info) "
                    "VALUES (:id, :ns, :doc, 'c', 1, :sc, 40, 7, :ci)"
                ),
                {
                    "id": str(_ID_TIER1_PRECEDENCE),
                    "ns": str(_NS),
                    "doc": str(uuid4()),
                    "sc": _PREC_COLUMN_START_CHAR,
                    "ci": json.dumps({"chunker": "sentence", "start_char": _PREC_TWIN_START_CHAR}),
                },
            )
            await _insert_khora_chunk(
                conn,
                _ID_TIER1_PRECEDENCE,
                # metadata carries a bookkeeping key (chunk_index) so the row is
                # eligible; chunker_info holds the STALE start_char base (A) but
                # deliberately NO chunk_index, so the guard fires.
                metadata={"chunk_index": 1, "start_char": 10, "author": "bob"},
                chunker_info={"start_char": _PREC_BASE_START_CHAR},
            )
    finally:
        await engine.dispose()


async def _insert_khora_chunk(conn: AsyncConnection, cid: UUID, *, metadata: dict, chunker_info: dict) -> None:
    await conn.execute(
        sa.text(
            "INSERT INTO khora_chunks "
            "(id, namespace_id, document_id, content, metadata, chunker_info) "
            "VALUES (:id, :ns, :doc, 'body', :md, :ci)"
        ),
        {
            "id": str(cid),
            "ns": str(_NS),
            "doc": str(uuid4()),
            "md": json.dumps(metadata),
            "ci": json.dumps(chunker_info),
        },
    )


async def _read_row(url: str, cid: UUID) -> tuple[dict, dict]:
    """Return ``(metadata, chunker_info)`` for a khora_chunks row as dicts."""
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    sa.text("SELECT metadata, chunker_info FROM khora_chunks WHERE id = :id"),
                    {"id": str(cid)},
                )
            ).one()
            return _as_dict(row[0]), _as_dict(row[1])
    finally:
        await engine.dispose()


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    return json.loads(value)


async def _step_version_back_sqlite(url: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text("UPDATE khora_alembic_version SET version_num = :v"),
                {"v": _PREV_REVISION},
            )
    finally:
        await engine.dispose()


@pytest.mark.unit
class TestMigration053OnSqlite:
    @pytest.fixture
    def sqlite_url(self, tmp_path: Path) -> str:
        return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    def test_archetypes(self, sqlite_url: str) -> None:
        asyncio.run(_seed_sqlite(sqlite_url))
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, _HEAD_REVISION)

        # 1. Tier-1 match: all four stripped from metadata (user key survives);
        #    chunker_info carries the twin's info + the four column values.
        md, ci = asyncio.run(_read_row(sqlite_url, _ID_TIER1_MATCH))
        assert "chunk_index" not in md and "start_char" not in md
        assert "end_char" not in md and "token_count" not in md
        assert md == {"author": "alice"}
        assert ci["chunk_index"] == 3
        assert ci["start_char"] == 10
        assert ci["end_char"] == 40
        assert ci["token_count"] == 7
        assert ci["chunker"] == "sentence"  # twin chunker_info preserved

        # 2. Tier-1 collision: the differing metadata.chunk_index stays; the
        #    other three (which matched) are stripped; chunker_info gets the
        #    authoritative column value (3), not the user value (999).
        md, ci = asyncio.run(_read_row(sqlite_url, _ID_TIER1_COLLISION))
        assert md == {"chunk_index": 999}
        assert ci["chunk_index"] == 3
        assert ci["start_char"] == 10 and ci["end_char"] == 40 and ci["token_count"] == 7

        # 3. Tier-2 number-typed: moved + stripped, user key preserved.
        md, ci = asyncio.run(_read_row(sqlite_url, _ID_TIER2_NUMBER))
        assert md == {"topic": "x"}
        assert ci == {"chunk_index": 5, "start_char": 0, "end_char": 12, "token_count": 4}

        # 4. Tier-2 string-typed chunk_index: left in metadata; not moved.
        md, ci = asyncio.run(_read_row(sqlite_url, _ID_TIER2_STRING))
        assert md == {"chunk_index": "abc"}
        assert "chunk_index" not in ci

        # 5. Clean row: chunker_info already has chunk_index → guard skips it,
        #    the user metadata.chunk_index is left untouched.
        md, ci = asyncio.run(_read_row(sqlite_url, _ID_CLEAN))
        assert md == {"chunk_index": 42}
        assert ci == {"chunk_index": 1, "start_char": 0, "end_char": 5, "token_count": 2}

        # 6. Three-way precedence: start_char collides across kc.chunker_info
        #    (111), the twin's chunker_info (222), and the bookkeeping column
        #    (333). The column value MUST win last (kc || twin || bookkeeping),
        #    pinning the nesting order so a future regression in the merge
        #    precedence is caught.
        md, ci = asyncio.run(_read_row(sqlite_url, _ID_TIER1_PRECEDENCE))
        assert ci["start_char"] == _PREC_COLUMN_START_CHAR, (
            f"precedence broken: expected column value {_PREC_COLUMN_START_CHAR}, got {ci['start_char']} "
            f"(base was {_PREC_BASE_START_CHAR}, twin was {_PREC_TWIN_START_CHAR})"
        )
        assert ci["chunk_index"] == 1  # bookkeeping column value landed
        assert ci["chunker"] == "sentence"  # twin's non-colliding key preserved

    def test_idempotent_rerun(self, sqlite_url: str) -> None:
        asyncio.run(_seed_sqlite(sqlite_url))
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, _HEAD_REVISION)

        first = {
            cid: asyncio.run(_read_row(sqlite_url, cid))
            for cid in (
                _ID_TIER1_MATCH,
                _ID_TIER1_COLLISION,
                _ID_TIER2_NUMBER,
                _ID_TIER2_STRING,
                _ID_CLEAN,
                _ID_TIER1_PRECEDENCE,
            )
        }

        # Step the version back and re-run: the guarded UPDATEs must touch zero
        # rows (chunker_info now carries chunk_index everywhere it was moved).
        asyncio.run(_step_version_back_sqlite(sqlite_url))
        command.upgrade(cfg, _HEAD_REVISION)

        second = {cid: asyncio.run(_read_row(sqlite_url, cid)) for cid in first}
        assert first == second, "re-run changed an already-migrated row"


# ---------------------------------------------------------------------------
# Postgres lane — same archetypes against real JSONB (skips without a bind)
# ---------------------------------------------------------------------------


_PG_KHORA_CHUNKS_DDL = """
CREATE TABLE khora_chunks (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    chunker_info JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_tsv TSVECTOR
)
"""

_PG_TSV_FUNCTION_DDL = """
CREATE OR REPLACE FUNCTION khora_chunks_content_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql
"""

_PG_TSV_TRIGGER_DDL = """
CREATE TRIGGER khora_chunks_content_tsv_update
BEFORE INSERT OR UPDATE ON khora_chunks
FOR EACH ROW EXECUTE FUNCTION khora_chunks_content_tsv_trigger()
"""


async def _reset_public_schema(eng: AsyncEngine) -> None:
    """Wipe ``public`` and pre-create the wide khora_alembic_version table.

    Mirrors the sibling migration tests: alembic creates
    ``khora_alembic_version`` as ``VARCHAR(32)`` but several revision ids are
    wider, so pre-create it at VARCHAR(64) and let the chain apply cleanly.
    """
    async with eng.begin() as conn:
        r = await conn.execute(
            sa.text("SELECT typname FROM pg_type WHERE typnamespace = 'public'::regnamespace AND typtype = 'e'")
        )
        for (typname,) in r.fetchall():
            await conn.execute(sa.text(f"DROP TYPE IF EXISTS public.{typname} CASCADE"))
        await conn.execute(sa.text("DROP SCHEMA public CASCADE"))
        await conn.execute(sa.text("CREATE SCHEMA public"))
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(
            sa.text(
                "CREATE TABLE khora_alembic_version ("
                "  version_num VARCHAR(64) NOT NULL,"
                "  CONSTRAINT khora_alembic_version_pkc PRIMARY KEY (version_num)"
                ")"
            )
        )


async def _seed_pg(url: str) -> None:
    """Re-migrate the schema to head, then drop+recreate a runtime-shaped
    ``khora_chunks`` (with its content_tsv trigger), seed the archetype
    rows into it and the twin ``chunks`` rows, and step the version to 052."""
    eng = create_async_engine(url)
    try:
        await _reset_public_schema(eng)
    finally:
        await eng.dispose()
    result = await run_migrations(url)
    assert result.success, f"migrations failed: {result.error}"

    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text("DROP TABLE IF EXISTS khora_chunks CASCADE"))
            await conn.execute(sa.text(_PG_KHORA_CHUNKS_DDL))
            await conn.execute(sa.text(_PG_TSV_FUNCTION_DDL))
            await conn.execute(sa.text(_PG_TSV_TRIGGER_DDL))

            await conn.execute(
                sa.text(
                    "INSERT INTO memory_namespaces "
                    "(id, namespace_id, version, is_active, created_at, updated_at) "
                    "VALUES (:id, :id, 1, TRUE, NOW(), NOW())"
                ),
                {"id": _NS},
            )

            for cid in (_ID_TIER1_MATCH, _ID_TIER1_COLLISION):
                doc_id = uuid4()
                await _pg_seed_document(conn, doc_id)
                await conn.execute(
                    sa.text(
                        "INSERT INTO chunks "
                        "(id, namespace_id, document_id, content, chunk_index, "
                        " start_char, end_char, token_count, chunker_info) "
                        "VALUES (:id, :ns, :doc, 'c', 3, 10, 40, 7, "
                        ' \'{"chunker": "sentence", "version": "1"}\'::jsonb)'
                    ),
                    {"id": cid, "ns": _NS, "doc": doc_id},
                )

            await _pg_insert_khora_chunk(
                conn,
                _ID_TIER1_MATCH,
                metadata={"chunk_index": 3, "start_char": 10, "end_char": 40, "token_count": 7, "author": "alice"},
                chunker_info={},
            )
            await _pg_insert_khora_chunk(
                conn,
                _ID_TIER1_COLLISION,
                metadata={"chunk_index": 999, "start_char": 10, "end_char": 40, "token_count": 7},
                chunker_info={},
            )
            await _pg_insert_khora_chunk(
                conn,
                _ID_TIER2_NUMBER,
                metadata={"chunk_index": 5, "start_char": 0, "end_char": 12, "token_count": 4, "topic": "x"},
                chunker_info={},
            )
            await _pg_insert_khora_chunk(
                conn,
                _ID_TIER2_STRING,
                metadata={"chunk_index": "abc"},
                chunker_info={},
            )
            await _pg_insert_khora_chunk(
                conn,
                _ID_CLEAN,
                metadata={"chunk_index": 42},
                chunker_info={"chunk_index": 1, "start_char": 0, "end_char": 5, "token_count": 2},
            )

            # 6. Three-way precedence: dedicated twin whose typed start_char
            #    column (C=333) differs from both the twin's own chunker_info
            #    start_char (B=222) and the kc.chunker_info base (A=111). The
            #    column value must win last (kc || c.chunker_info || bookkeeping).
            prec_doc = uuid4()
            await _pg_seed_document(conn, prec_doc)
            await conn.execute(
                sa.text(
                    "INSERT INTO chunks "
                    "(id, namespace_id, document_id, content, chunk_index, "
                    " start_char, end_char, token_count, chunker_info) "
                    "VALUES (:id, :ns, :doc, 'c', 1, :sc, 40, 7, CAST(:ci AS JSONB))"
                ),
                {
                    "id": _ID_TIER1_PRECEDENCE,
                    "ns": _NS,
                    "doc": prec_doc,
                    "sc": _PREC_COLUMN_START_CHAR,
                    "ci": json.dumps({"chunker": "sentence", "start_char": _PREC_TWIN_START_CHAR}),
                },
            )
            await _pg_insert_khora_chunk(
                conn,
                _ID_TIER1_PRECEDENCE,
                metadata={"chunk_index": 1, "start_char": 10, "author": "bob"},
                chunker_info={"start_char": _PREC_BASE_START_CHAR},
            )

            await conn.execute(
                sa.text("UPDATE khora_alembic_version SET version_num = :v"),
                {"v": _PREV_REVISION},
            )
    finally:
        await engine.dispose()


async def _pg_seed_document(conn: AsyncConnection, doc_id: UUID) -> None:
    await conn.execute(
        sa.text(
            "INSERT INTO documents "
            "(id, namespace_id, content, status, source_type, source_name, "
            " external_id, content_type, source, title, created_at, updated_at) "
            "VALUES (:id, :ns, 'body', 'completed', 'slack', 'general', "
            " :eid, 'text/plain', 's', 't', NOW(), NOW())"
        ),
        {"id": doc_id, "ns": _NS, "eid": str(doc_id)},
    )


async def _pg_insert_khora_chunk(conn: AsyncConnection, cid: UUID, *, metadata: dict, chunker_info: dict) -> None:
    await conn.execute(
        sa.text(
            "INSERT INTO khora_chunks "
            "(id, namespace_id, document_id, content, metadata, chunker_info) "
            "VALUES (:id, :ns, :doc, 'body', CAST(:md AS JSONB), CAST(:ci AS JSONB))"
        ),
        {
            "id": cid,
            "ns": _NS,
            "doc": uuid4(),
            "md": json.dumps(metadata),
            "ci": json.dumps(chunker_info),
        },
    )


@pytest.mark.integration
class TestMigration053OnPostgres:
    @pytest.fixture
    def pg_url(self) -> str:
        if not _pg_reachable():
            pytest.skip("PostgreSQL not reachable (run `make dev` first)")
        asyncio.run(_seed_pg(DATABASE_URL))
        return DATABASE_URL

    def test_archetypes(self, pg_url: str) -> None:
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        # 1. Tier-1 match.
        md, ci = asyncio.run(_read_row(pg_url, _ID_TIER1_MATCH))
        assert md == {"author": "alice"}
        assert ci["chunk_index"] == 3 and ci["start_char"] == 10
        assert ci["end_char"] == 40 and ci["token_count"] == 7
        assert ci["chunker"] == "sentence"

        # 2. Tier-1 collision.
        md, ci = asyncio.run(_read_row(pg_url, _ID_TIER1_COLLISION))
        assert md == {"chunk_index": 999}
        assert ci["chunk_index"] == 3

        # 3. Tier-2 number-typed.
        md, ci = asyncio.run(_read_row(pg_url, _ID_TIER2_NUMBER))
        assert md == {"topic": "x"}
        assert ci == {"chunk_index": 5, "start_char": 0, "end_char": 12, "token_count": 4}

        # 4. Tier-2 string-typed.
        md, ci = asyncio.run(_read_row(pg_url, _ID_TIER2_STRING))
        assert md == {"chunk_index": "abc"}
        assert "chunk_index" not in ci

        # 5. Clean row untouched.
        md, ci = asyncio.run(_read_row(pg_url, _ID_CLEAN))
        assert md == {"chunk_index": 42}
        assert ci == {"chunk_index": 1, "start_char": 0, "end_char": 5, "token_count": 2}

        # 6. Three-way precedence: start_char in kc.chunker_info (111), twin
        #    chunker_info (222), and bookkeeping column (333) — column wins last.
        md, ci = asyncio.run(_read_row(pg_url, _ID_TIER1_PRECEDENCE))
        assert ci["start_char"] == _PREC_COLUMN_START_CHAR, (
            f"precedence broken: expected column value {_PREC_COLUMN_START_CHAR}, got {ci['start_char']} "
            f"(base was {_PREC_BASE_START_CHAR}, twin was {_PREC_TWIN_START_CHAR})"
        )
        assert ci["chunk_index"] == 1
        assert ci["chunker"] == "sentence"

    def test_idempotent_rerun_and_trigger_restored(self, pg_url: str) -> None:
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        first = {
            cid: asyncio.run(_read_row(pg_url, cid))
            for cid in (
                _ID_TIER1_MATCH,
                _ID_TIER1_COLLISION,
                _ID_TIER2_NUMBER,
                _ID_TIER2_STRING,
                _ID_CLEAN,
                _ID_TIER1_PRECEDENCE,
            )
        }

        # The content_tsv trigger must be re-enabled after the relocation.
        async def _trigger_state() -> str:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    return (
                        await conn.execute(
                            sa.text(
                                "SELECT t.tgenabled::text FROM pg_trigger t "
                                "JOIN pg_class c ON c.oid = t.tgrelid "
                                "WHERE c.relname = 'khora_chunks' AND t.tgname = :trg"
                            ),
                            {"trg": _TSV_TRIGGER},
                        )
                    ).scalar()
            finally:
                await engine.dispose()

        assert asyncio.run(_trigger_state()) == "O", "content_tsv trigger not re-enabled"

        # Step version back and re-run: guarded UPDATEs touch zero rows.
        async def _step_back() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.begin() as conn:
                    await conn.execute(
                        sa.text("UPDATE khora_alembic_version SET version_num = :v"),
                        {"v": _PREV_REVISION},
                    )
            finally:
                await engine.dispose()

        asyncio.run(_step_back())
        command.upgrade(cfg, "head")

        second = {cid: asyncio.run(_read_row(pg_url, cid)) for cid in first}
        assert first == second, "re-run changed an already-migrated row"
