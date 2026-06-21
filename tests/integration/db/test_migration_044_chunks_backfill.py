"""Coverage for migration ``044_khora_chunks_backfill_denormalized``.

Migration 041 *added* eight nullable, denormalized document-grained columns
to the runtime-managed ``khora_chunks`` temporal store (later widened so
``source`` / ``external_id`` match the ``documents`` widths). Migration 044
*populates* them on existing rows from the parent ``documents`` row and builds
the five filterable-subset indexes so recall filters resolve without a join:

    source_type, source_name, source_url, source_timestamp,
    external_id, content_type, source, title

``khora_chunks`` is NOT part of the Alembic-managed schema — it is created at
runtime by ``PgVectorTemporalStore.connect()``. The migration is therefore
Postgres-only, guarded by ``has_table("khora_chunks")``, and runs entirely in
autocommit (per-namespace batched backfill → ``CREATE INDEX CONCURRENTLY`` →
``VACUUM (ANALYZE)``). These tests cover all of that surface:

1. Postgres backfill happy path: a legacy ``khora_chunks`` (eight cols NULL) +
   a parent ``documents`` row with known values incl. a >255-char ``source``
   and >255-char ``external_id``; step the version back to 043; ``upgrade
   head``; assert each chunk col equals the document value in full (the
   widened columns hold the entire producer value verbatim).
2. Multi-namespace batching: chunks across ≥2 distinct ``namespace_id``s, each
   with its own parent document — assert **all** namespaces are backfilled
   (proves the ``SELECT DISTINCT namespace_id`` loop covers every ns).
3. Idempotent / restartable re-run: run the upgrade twice; assert the second
   run leaves values unchanged (the ``source_type IS NULL`` sentinel).
4. Indexes created: all five ``ix_khora_chunks_ns_*`` exist in ``pg_indexes``,
   the timestamp one carries the partial ``WHERE source_timestamp IS NOT
   NULL``, and all are VALID (``indisvalid`` — the VACUUM ran after the build).
5. Trigger restored: ``khora_chunks_content_tsv_update`` is ENABLED
   (``pg_trigger.tgenabled = 'O'``) after the upgrade — the ``finally``
   re-enabled it even though the backfill ran in autocommit.
6. ``content_tsv`` untouched: a row with a known ``content_tsv`` keeps it
   through the backfill (proves the trigger was disabled, so the backfill
   UPDATE did not recompute it).
7. Missing-table no-op (fresh deploy): no ``khora_chunks`` → chain reaches head
   044, table still absent.
8. SQLite: the migration is a clean no-op; chain reaches head
   ``044_khora_chunks_backfill_denormalized`` and downgrades to 043 cleanly.

Run locally::

    make dev    # postgres on port 5434
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/db/test_migration_044_chunks_backfill.py \
        -v -m integration --no-cov
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

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


pytestmark = pytest.mark.integration


_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"

_PREV_REVISION = "043_khora_chunks_metadata_backfill"
_HEAD_REVISION = "048_dream_conflicts_reconcile"

_TSV_TRIGGER = "khora_chunks_content_tsv_update"

# The five filter indexes 044 builds CONCURRENTLY — names byte-identical to the
# runtime ``PgVectorTemporalStore.connect()``.
_FILTER_INDEXES = (
    "ix_khora_chunks_ns_source_type",
    "ix_khora_chunks_ns_source_name",
    "ix_khora_chunks_ns_source_timestamp",
    "ix_khora_chunks_ns_external_id",
    "ix_khora_chunks_ns_content_type",
)


def _make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    # Alembic uses configparser.BasicInterpolation; escape any literal '%' in
    # the URL so it isn't read as a config-interpolation token.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


# A runtime-shaped ``khora_chunks`` table: the identity/temporal/content
# columns the runtime always creates, the eight denormalized columns 041 added
# and the widen migration resized (present, NULL), plus ``content_tsv`` and the
# BEFORE INSERT OR UPDATE trigger that recomputes it. Creating this before the
# upgrade exercises 044's existing-deployment backfill path including the
# trigger DISABLE/ENABLE.
_RUNTIME_KHORA_CHUNKS_DDL = """
CREATE TABLE khora_chunks (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    occurred_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ,
    source_type VARCHAR(64),
    source_name VARCHAR(255),
    source_url TEXT,
    source_timestamp TIMESTAMPTZ,
    external_id VARCHAR(512),
    content_type VARCHAR(128),
    source TEXT,
    title TEXT,
    content_tsv TSVECTOR
)
"""

# The runtime's content_tsv trigger, byte-equivalent to
# ``PgVectorTemporalStore.connect()``. Seeding it lets us assert the backfill
# disabled it (content_tsv untouched) and the finally re-enabled it.
_TSV_FUNCTION_DDL = """
CREATE OR REPLACE FUNCTION khora_chunks_content_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql
"""

_TSV_TRIGGER_DDL = """
CREATE TRIGGER khora_chunks_content_tsv_update
BEFORE INSERT OR UPDATE ON khora_chunks
FOR EACH ROW EXECUTE FUNCTION khora_chunks_content_tsv_trigger()
"""


async def _reset_public_schema(eng: AsyncEngine) -> None:
    """Wipe ``public`` and pre-create the wide khora_alembic_version table.

    Mirrors ``test_migration_033_bitemporal.py``: alembic creates
    ``khora_alembic_version`` with the default ``VARCHAR(32)`` but several
    revision ids are wider. Pre-create the table with VARCHAR(64) so the chain
    applies cleanly.
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


@pytest.fixture
def pg_url() -> Iterator[str]:
    """Wipe ``public``, re-migrate to head, and yield the base Postgres URL.

    The schema-wipe makes the tests safe to run against shared PostgreSQL (the
    integration job migrates the service DB to head up front) without leaking
    state. ``khora_chunks`` is runtime-managed (not Alembic), so after the
    re-migrate it does not exist — each test seeds the runtime-shaped table
    (with its content_tsv trigger) itself.
    """
    if not _pg_reachable():
        pytest.skip("PostgreSQL not reachable (run `make dev` first)")

    async def _setup() -> None:
        eng = create_async_engine(DATABASE_URL)
        try:
            await _reset_public_schema(eng)
        finally:
            await eng.dispose()
        result = await run_migrations(DATABASE_URL)
        assert result.success, f"migrations failed: {result.error}"

    asyncio.run(_setup())
    yield DATABASE_URL


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"


# ---------------------------------------------------------------------------
# Seed helpers (Postgres)
# ---------------------------------------------------------------------------


async def _seed_namespace(conn: AsyncConnection, ns_row_id) -> None:
    """Insert a ``memory_namespaces`` row so ``documents.namespace_id`` FK holds."""
    await conn.execute(
        sa.text(
            "INSERT INTO memory_namespaces "
            "(id, namespace_id, version, is_active, created_at, updated_at) "
            "VALUES (:id, :id, 1, TRUE, NOW(), NOW())"
        ),
        {"id": ns_row_id},
    )


def _coerce_source_timestamp(values: dict) -> dict:
    """Return a copy of ``values`` with any ISO-string ``source_timestamp``
    parsed into a ``datetime`` (asyncpg binds TIMESTAMPTZ from datetime, not str)."""
    ts = values.get("source_timestamp")
    if isinstance(ts, str):
        return {**values, "source_timestamp": datetime.fromisoformat(ts)}
    return values


async def _seed_document(conn: AsyncConnection, doc_id, ns_row_id, values: dict) -> None:
    """Insert a ``documents`` row carrying the provenance the backfill copies."""
    await conn.execute(
        sa.text(
            "INSERT INTO documents "
            "(id, namespace_id, content, status, source_type, source_name, "
            " source_url, source_timestamp, external_id, content_type, source, "
            " title, created_at, updated_at) "
            "VALUES (:id, :ns, :content, 'completed', :source_type, :source_name, "
            " :source_url, :source_timestamp, :external_id, :content_type, "
            " :source, :title, NOW(), NOW())"
        ),
        # asyncpg binds ``source_timestamp`` as a real TIMESTAMPTZ and rejects a
        # str — parse any ISO string the caller passes into a datetime first.
        {"id": doc_id, "ns": ns_row_id, **_coerce_source_timestamp(values)},
    )


async def _seed_chunk(
    conn: AsyncConnection,
    chunk_id,
    ns_id,
    doc_id,
    content: str,
) -> None:
    """Insert a legacy ``khora_chunks`` row: eight denormalized columns NULL.

    The INSERT fires the seeded BEFORE-INSERT trigger, which populates
    ``content_tsv`` from ``content`` — exactly as the runtime would.
    """
    await conn.execute(
        sa.text(
            "INSERT INTO khora_chunks "
            "(id, namespace_id, document_id, content, occurred_at, created_at) "
            "VALUES (:id, :ns, :doc, :content, NOW(), NOW())"
        ),
        {"id": chunk_id, "ns": ns_id, "doc": doc_id, "content": content},
    )


async def _install_chunks_table_and_trigger(conn: AsyncConnection) -> None:
    """Create the runtime-shaped khora_chunks table + content_tsv trigger."""
    await conn.execute(sa.text("DROP TABLE IF EXISTS khora_chunks CASCADE"))
    await conn.execute(sa.text(_RUNTIME_KHORA_CHUNKS_DDL))
    await conn.execute(sa.text(_TSV_FUNCTION_DDL))
    await conn.execute(sa.text(_TSV_TRIGGER_DDL))


async def _step_version_back(conn: AsyncConnection) -> None:
    """Step ``khora_alembic_version`` from head (044) back to 043 so the next
    ``upgrade head`` re-runs 044 against the table we just seeded."""
    await conn.execute(
        sa.text("UPDATE khora_alembic_version SET version_num = :prev"),
        {"prev": _PREV_REVISION},
    )
    # The step-back above is metadata-only, so chunks.occurred_at (added by
    # migration 046 during the first upgrade-to-head) survives. Drop it so the
    # replayed ``upgrade head`` re-applies 046 cleanly — mirrors the
    # drop+recreate of khora_chunks the seed helpers perform.
    await conn.execute(sa.text("ALTER TABLE chunks DROP COLUMN IF EXISTS occurred_at"))


# A >255-char source and external_id: each fits in full now that the columns
# are widened (source -> TEXT, external_id -> VARCHAR(512)), so the backfill
# copies the whole value verbatim.
_LONG_SOURCE = "src://" + ("x" * 400)
_LONG_EXTERNAL_ID = "ext-" + ("y" * 400)


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


class TestMigration044OnPostgres:
    def test_backfill_happy_path(self, pg_url: str) -> None:
        """Each denormalized chunk col gets the parent document value verbatim,
        including the long ``source`` / ``external_id`` (widened columns hold
        the full value)."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        ns_id = uuid4()
        doc_id = uuid4()
        chunk_id = uuid4()
        doc_values = {
            "content": "parent document body",
            "source_type": "slack",
            "source_name": "general",
            "source_url": "https://slack.example/general/123",
            "source_timestamp": "2026-01-02T03:04:05+00:00",
            "external_id": _LONG_EXTERNAL_ID,
            "content_type": "text/markdown",
            "source": _LONG_SOURCE,
            "title": "Weekly sync notes",
        }

        async def _seed() -> None:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await _install_chunks_table_and_trigger(conn)
                    await _seed_namespace(conn, ns_id)
                    await _seed_document(conn, doc_id, ns_id, doc_values)
                    await _seed_chunk(conn, chunk_id, ns_id, doc_id, "chunk body text")
                    await _step_version_back(conn)
            finally:
                await engine.dispose()

        asyncio.run(_seed())
        command.upgrade(cfg, "head")  # re-runs 044 → backfills

        async def _check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    row = (
                        await conn.execute(
                            sa.text(
                                "SELECT source_type, source_name, source_url, "
                                "source_timestamp, external_id, content_type, "
                                "source, title FROM khora_chunks WHERE id = :id"
                            ),
                            {"id": chunk_id},
                        )
                    ).one()
                    (
                        source_type,
                        source_name,
                        source_url,
                        source_timestamp,
                        external_id,
                        content_type,
                        source,
                        title,
                    ) = row

                    assert source_type == doc_values["source_type"]
                    assert source_name == doc_values["source_name"]
                    assert source_url == doc_values["source_url"]
                    assert source_timestamp is not None  # copied verbatim from doc
                    assert content_type == doc_values["content_type"]
                    assert title == doc_values["title"]

                    # The widened columns hold the full producer value now
                    # that source is TEXT and external_id is VARCHAR(512).
                    assert external_id == _LONG_EXTERNAL_ID
                    assert len(external_id) == len(_LONG_EXTERNAL_ID)
                    assert source == _LONG_SOURCE
                    assert len(source) == len(_LONG_SOURCE)
            finally:
                await engine.dispose()

        asyncio.run(_check())

    def test_multi_namespace_batching(self, pg_url: str) -> None:
        """Two distinct namespaces both get backfilled — proves the per-ns
        ``SELECT DISTINCT namespace_id`` loop covers every namespace, not just
        the first."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        ns_a, ns_b = uuid4(), uuid4()
        doc_a, doc_b = uuid4(), uuid4()
        chunk_a, chunk_b = uuid4(), uuid4()

        async def _seed() -> None:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await _install_chunks_table_and_trigger(conn)
                    for ns, doc, chunk, st in (
                        (ns_a, doc_a, chunk_a, "alpha-source"),
                        (ns_b, doc_b, chunk_b, "beta-source"),
                    ):
                        await _seed_namespace(conn, ns)
                        await _seed_document(
                            conn,
                            doc,
                            ns,
                            {
                                "content": "doc body",
                                "source_type": st,
                                "source_name": "chan",
                                "source_url": None,
                                "source_timestamp": None,
                                "external_id": "eid-" + st,
                                "content_type": "text/plain",
                                "source": st,
                                "title": "t-" + st,
                            },
                        )
                        await _seed_chunk(conn, chunk, ns, doc, "chunk for " + st)
                    await _step_version_back(conn)
            finally:
                await engine.dispose()

        asyncio.run(_seed())
        command.upgrade(cfg, "head")

        async def _check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    rows = (
                        await conn.execute(
                            sa.text("SELECT id, source_type FROM khora_chunks WHERE id = ANY(:ids)"),
                            {"ids": [chunk_a, chunk_b]},
                        )
                    ).fetchall()
                    by_id = {r[0]: r[1] for r in rows}
                    # BOTH namespaces backfilled (loop covered every ns).
                    assert by_id[chunk_a] == "alpha-source"
                    assert by_id[chunk_b] == "beta-source"
            finally:
                await engine.dispose()

        asyncio.run(_check())

    def test_idempotent_rerun(self, pg_url: str) -> None:
        """Re-running the upgrade (step version back, keep data) leaves
        already-backfilled values unchanged — the ``source_type IS NULL``
        sentinel makes the second pass a no-op (models resume-after-crash)."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        ns_id = uuid4()
        doc_id = uuid4()
        chunk_id = uuid4()

        async def _seed() -> None:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await _install_chunks_table_and_trigger(conn)
                    await _seed_namespace(conn, ns_id)
                    await _seed_document(
                        conn,
                        doc_id,
                        ns_id,
                        {
                            "content": "doc body",
                            "source_type": "email",
                            "source_name": "inbox",
                            "source_url": "https://mail.example/1",
                            "source_timestamp": None,
                            "external_id": "msg-001",
                            "content_type": "message/rfc822",
                            "source": "imap://inbox",
                            "title": "Re: hello",
                        },
                    )
                    await _seed_chunk(conn, chunk_id, ns_id, doc_id, "chunk body")
                    await _step_version_back(conn)
            finally:
                await engine.dispose()

        asyncio.run(_seed())
        command.upgrade(cfg, "head")  # first backfill

        async def _snapshot() -> tuple:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    return tuple(
                        (
                            await conn.execute(
                                sa.text(
                                    "SELECT source_type, source_name, external_id, "
                                    "source, title FROM khora_chunks WHERE id = :id"
                                ),
                                {"id": chunk_id},
                            )
                        ).one()
                    )
            finally:
                await engine.dispose()

        first = asyncio.run(_snapshot())

        # Step version back and re-run: the second pass must touch zero rows.
        async def _step_back_only() -> None:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await _step_version_back(conn)
            finally:
                await engine.dispose()

        asyncio.run(_step_back_only())
        command.upgrade(cfg, "head")  # second backfill (idempotent)

        second = asyncio.run(_snapshot())
        assert first == second, "re-run changed an already-backfilled row"
        assert second[0] == "email"  # sanity: still populated

    def test_filter_indexes_created_and_valid(self, pg_url: str) -> None:
        """All five filter indexes exist after the upgrade, the timestamp one
        is partial (``WHERE source_timestamp IS NOT NULL``), and all are VALID
        (``indisvalid`` — the post-build VACUUM completed)."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        ns_id = uuid4()
        doc_id = uuid4()
        chunk_id = uuid4()

        async def _seed() -> None:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await _install_chunks_table_and_trigger(conn)
                    await _seed_namespace(conn, ns_id)
                    await _seed_document(
                        conn,
                        doc_id,
                        ns_id,
                        {
                            "content": "doc body",
                            "source_type": "library",
                            "source_name": "kb",
                            "source_url": None,
                            "source_timestamp": "2026-03-01T00:00:00+00:00",
                            "external_id": "kb-1",
                            "content_type": "text/plain",
                            "source": "kb",
                            "title": "KB entry",
                        },
                    )
                    await _seed_chunk(conn, chunk_id, ns_id, doc_id, "chunk body")
                    await _step_version_back(conn)
            finally:
                await engine.dispose()

        asyncio.run(_seed())
        command.upgrade(cfg, "head")

        async def _check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    rows = (
                        await conn.execute(
                            sa.text(
                                "SELECT indexname, indexdef FROM pg_indexes "
                                "WHERE tablename = 'khora_chunks' "
                                "AND indexname = ANY(:names)"
                            ),
                            {"names": list(_FILTER_INDEXES)},
                        )
                    ).fetchall()
                    defs = {r[0]: r[1] for r in rows}
                    # All five present.
                    assert set(defs.keys()) == set(_FILTER_INDEXES), (
                        f"missing: {set(_FILTER_INDEXES) - set(defs.keys())}"
                    )
                    # The timestamp index is partial.
                    ts_def = defs["ix_khora_chunks_ns_source_timestamp"]
                    assert "source_timestamp IS NOT NULL" in ts_def, ts_def
                    # The others are not partial.
                    for name in _FILTER_INDEXES:
                        if name != "ix_khora_chunks_ns_source_timestamp":
                            assert "WHERE" not in defs[name].upper(), defs[name]

                    # All five are VALID (CONCURRENTLY can leave INVALID on
                    # failure; the VACUUM ran after a clean build).
                    valid = (
                        await conn.execute(
                            sa.text(
                                "SELECT c.relname, i.indisvalid "
                                "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                                "WHERE c.relname = ANY(:names)"
                            ),
                            {"names": list(_FILTER_INDEXES)},
                        )
                    ).fetchall()
                    for name, indisvalid in valid:
                        assert indisvalid is True, f"{name} is INVALID"
            finally:
                await engine.dispose()

        asyncio.run(_check())

    def test_tsv_trigger_re_enabled(self, pg_url: str) -> None:
        """``khora_chunks_content_tsv_update`` is ENABLED (``tgenabled = 'O'``)
        after the upgrade — the ``finally`` re-enabled it even though the
        backfill ran in autocommit (load-bearing, not defensive)."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        ns_id = uuid4()
        doc_id = uuid4()
        chunk_id = uuid4()

        async def _seed() -> None:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await _install_chunks_table_and_trigger(conn)
                    await _seed_namespace(conn, ns_id)
                    await _seed_document(
                        conn,
                        doc_id,
                        ns_id,
                        {
                            "content": "doc body",
                            "source_type": "slack",
                            "source_name": "c",
                            "source_url": None,
                            "source_timestamp": None,
                            "external_id": "x",
                            "content_type": "text/plain",
                            "source": "s",
                            "title": "t",
                        },
                    )
                    await _seed_chunk(conn, chunk_id, ns_id, doc_id, "chunk body")
                    await _step_version_back(conn)
            finally:
                await engine.dispose()

        asyncio.run(_seed())
        command.upgrade(cfg, "head")

        async def _check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    tgenabled = (
                        await conn.execute(
                            sa.text(
                                # Cast to text so the PG ``"char"`` ``tgenabled`` comes
                                # back as a str (asyncpg returns the raw type as bytes).
                                "SELECT t.tgenabled::text FROM pg_trigger t "
                                "JOIN pg_class c ON c.oid = t.tgrelid "
                                "WHERE c.relname = 'khora_chunks' "
                                "AND t.tgname = :trigger"
                            ),
                            {"trigger": _TSV_TRIGGER},
                        )
                    ).scalar()
                    # 'O' = trigger fires in "origin" (normal) mode, i.e. ENABLED.
                    assert tgenabled == "O", f"trigger not re-enabled: tgenabled={tgenabled!r}"
            finally:
                await engine.dispose()

        asyncio.run(_check())

    def test_content_tsv_untouched_by_backfill(self, pg_url: str) -> None:
        """The backfill UPDATE must not recompute ``content_tsv`` — proves the
        trigger was disabled across the backfill loop. We seed a row, capture
        its ``content_tsv``, run the backfill, and assert it is byte-identical
        (the backfill never touches ``content``)."""
        cfg = _make_config(pg_url)
        command.upgrade(cfg, "head")

        ns_id = uuid4()
        doc_id = uuid4()
        chunk_id = uuid4()

        async def _seed_and_capture() -> str:
            engine = create_async_engine(pg_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    await _install_chunks_table_and_trigger(conn)
                    await _seed_namespace(conn, ns_id)
                    await _seed_document(
                        conn,
                        doc_id,
                        ns_id,
                        {
                            "content": "doc body",
                            "source_type": "slack",
                            "source_name": "c",
                            "source_url": None,
                            "source_timestamp": None,
                            "external_id": "x",
                            "content_type": "text/plain",
                            "source": "s",
                            "title": "t",
                        },
                    )
                    await _seed_chunk(conn, chunk_id, ns_id, doc_id, "the quick brown fox jumps")
                    before = (
                        await conn.execute(
                            sa.text("SELECT content_tsv::text FROM khora_chunks WHERE id = :id"),
                            {"id": chunk_id},
                        )
                    ).scalar()
                    await _step_version_back(conn)
                    return before
            finally:
                await engine.dispose()

        before = asyncio.run(_seed_and_capture())
        # The seeded trigger populated content_tsv on INSERT.
        assert before, "seed precondition: content_tsv should be populated on INSERT"

        command.upgrade(cfg, "head")  # backfill (trigger disabled across loop)

        async def _capture_after() -> str:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    return (
                        await conn.execute(
                            sa.text("SELECT content_tsv::text FROM khora_chunks WHERE id = :id"),
                            {"id": chunk_id},
                        )
                    ).scalar()
            finally:
                await engine.dispose()

        after = asyncio.run(_capture_after())
        assert after == before, "backfill recomputed content_tsv (trigger was not disabled)"

    def test_no_op_when_table_absent(self, pg_url: str) -> None:
        """Fresh DB with no ``khora_chunks``: the ``has_table`` guard
        early-returns, the chain reaches head, and the migration does not
        create the runtime table."""
        cfg = _make_config(pg_url)
        # pg_url fixture already dropped khora_chunks; just upgrade.
        command.upgrade(cfg, "head")

        async def _check() -> None:
            engine = create_async_engine(pg_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == _HEAD_REVISION

                    result = await conn.execute(
                        sa.text(
                            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'khora_chunks')"
                        )
                    )
                    assert result.scalar() is False
            finally:
                await engine.dispose()

        asyncio.run(_check())


# ---------------------------------------------------------------------------
# SQLite — Postgres-only migration must early-return and stay green
# ---------------------------------------------------------------------------


class TestMigration044OnSqlite:
    def test_chain_reaches_head_on_sqlite(self, sqlite_url: str) -> None:
        """Migration 044 is a clean no-op on SQLite; chain reaches head."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == _HEAD_REVISION
            finally:
                await engine.dispose()

        asyncio.run(check())

    def test_downgrade_is_clean_on_sqlite(self, sqlite_url: str) -> None:
        """upgrade head → downgrade to 043 is a clean no-op on SQLite."""
        cfg = _make_config(sqlite_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, _PREV_REVISION)

        async def check() -> None:
            engine = create_async_engine(sqlite_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(sa.text("SELECT version_num FROM khora_alembic_version"))
                    assert result.scalar() == _PREV_REVISION
            finally:
                await engine.dispose()

        asyncio.run(check())
