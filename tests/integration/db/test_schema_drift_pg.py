"""ORM-vs-live schema drift gate, PostgreSQL leg.

The SQLite leg (``tests/unit/test_migration_drift.py``) runs on every PR and
carries the coverage rationale for the whole gate. This module runs the same
diff against a real PostgreSQL database, where nothing is exempt — including
the three config-conditional hnsw indexes, which are checked here because the
dimension this module migrates at is pinned below pgvector's ceiling (see
``_MIGRATION_EMBEDDING_DIMENSION``).

Why a second leg is not redundant
---------------------------------
Twelve ORM declarations are structurally invisible on the SQLite leg and
are only ever checked here:

* Six indexes — three ``postgresql_using="hnsw"`` vector indexes, the
  ``gin`` index on ``chunks.content_tsv``, the ``brin`` index on
  ``temporal_edges.occurred_at``, and the expression-based
  ``ix_entities_namespace_mentions`` (``mention_count.desc()``).
* Six ``Vector`` / ``TSVECTOR`` columns.

A declaration invisible to the gate is exactly the bug class this gate
exists to catch. One of those six indexes is in fact drifting:
``ix_entities_namespace_mentions`` is built by no migration, and only this
leg can see it.

That is why the index ledger is not simply shared. ``INDEX_BASELINE`` holds
the drift both legs observe; ``PG_ONLY_INDEX_BASELINE`` holds the drift only
this leg can observe, and is unioned in here alone. It cannot go in the
shared ledger: ``assert_ratchet`` asserts in both directions, so an entry
the SQLite leg exempts would fail that leg in the stale direction. The
nullability ledger is genuinely shared — that dimension agrees across both.

The helper's docstring lists what this gate does not check. A green run
here is not "the schemas match".

Run locally::

    make dev    # postgres on port 5434
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/db/test_schema_drift_pg.py -v \
        -m integration --no-cov
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from khora.db.migrations._schema_config import DEFAULT_EMBEDDING_DIMENSION
from khora.db.session import run_migrations
from tests.test_helpers.schema_drift import (
    INDEX_BASELINE,
    NULLABILITY_BASELINE,
    PG_ONLY_INDEX_BASELINE,
    assert_ratchet,
    collect_drift,
)

pytestmark = pytest.mark.integration

# The dimension this module migrates at.
#
# This fixture was ALREADY at 1536 before the constant existed, but only
# implicitly: ``run_migrations(url)`` with no ``embedding_dimension`` never
# sets the ``embedding_dimension`` Alembic attribute at all
# (``_run_migrations_sync`` guards the assignment with ``is not None``), and
# ``configured_embedding_dimension()`` then falls back to
# ``DEFAULT_EMBEDDING_DIMENSION``. So the value was decided two modules away
# by an omission. Naming it here makes it a stated precondition of the gate
# instead of an accident, which matters because the gate's result depends on
# it: migrations 002 / 007 / 024 build the three ``*_embedding_hnsw`` indexes
# only when ``full_precision_hnsw_supported()`` holds, i.e. at or below
# pgvector's 2000-dim ceiling for the ``vector`` HNSW opclass.
#
# Deliberately NOT paired with an exemption rule in ``collect_drift``. Pinning
# makes the claim well-defined — "at this dimension, every ORM index the chain
# can build, it does build" — whereas exempting the three would also hide a
# genuine regression at every dimension above the ceiling. If this pin ever
# needs to move above ``VECTOR_HNSW_MAX_DIM``, the three indexes have to be
# reasoned about explicitly rather than silently dropped from the gate;
# ``TestConfigConditionalIndexes`` on the SQLite leg fails first if it does.
_MIGRATION_EMBEDDING_DIMENSION = DEFAULT_EMBEDDING_DIMENSION


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


_PG_AVAILABLE = _pg_reachable()


async def _reset_public_schema() -> None:
    """Wipe ``public`` so the chain builds the schema from a clean slate.

    Mirrors ``test_migration_replay_idempotent.py``: drop leftover enum
    types, recreate the schema, re-create the ``vector`` extension, and
    pre-create ``khora_alembic_version`` with VARCHAR(64) so the wide
    revision ids apply cleanly.
    """
    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.begin() as conn:
            r = await conn.execute(
                text("SELECT typname FROM pg_type WHERE typnamespace = 'public'::regnamespace AND typtype = 'e'")
            )
            for (typname,) in r.fetchall():
                await conn.execute(text(f"DROP TYPE IF EXISTS public.{typname} CASCADE"))
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    "CREATE TABLE khora_alembic_version ("
                    "  version_num VARCHAR(64) NOT NULL,"
                    "  CONSTRAINT khora_alembic_version_pkc PRIMARY KEY (version_num)"
                    ")"
                )
            )
    finally:
        await eng.dispose()


@pytest.fixture(scope="module")
def pg_head_drift():
    """Build the schema from scratch on PostgreSQL, then diff it.

    Module-scoped: a full chain run against a real server is the expensive
    part and every test below wants the same schema.
    """
    if not _PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable (run `make dev` first)")

    import asyncio

    async def build():
        await _reset_public_schema()
        # Pass the dimension explicitly rather than relying on the implicit
        # fallback — see ``_MIGRATION_EMBEDDING_DIMENSION``.
        result = await run_migrations(DATABASE_URL, embedding_dimension=_MIGRATION_EMBEDDING_DIMENSION)
        assert result.success is True, f"migration chain failed: {result.error}"
        assert result.skipped is False, "chain unexpectedly took the ahead-skip path"

        eng = create_async_engine(DATABASE_URL)
        try:
            async with eng.connect() as conn:
                return await conn.run_sync(lambda sync_conn: collect_drift(sa.inspect(sync_conn), sqlite=False))
        finally:
            await eng.dispose()

    return asyncio.run(build())


class TestSchemaDriftPostgres:
    """Reflect the migrated PostgreSQL schema and diff it against the ORM."""

    def test_no_missing_tables(self, pg_head_drift):
        """Every ORM table must be built by the chain. No baseline — zero."""
        assert pg_head_drift.missing_tables == set()

    def test_no_missing_columns(self, pg_head_drift):
        """Every ORM column must exist, including Vector and TSVECTOR."""
        assert pg_head_drift.missing_columns == set()

    def test_orm_indexes_are_built(self, pg_head_drift):
        """Every ORM index must exist, including the six SQLite-exempt ones.

        Ledger is the shared one plus this leg's delta — see the module
        docstring for why the delta cannot live in the shared frozenset.
        """
        assert_ratchet(pg_head_drift.missing_indexes, INDEX_BASELINE | PG_ONLY_INDEX_BASELINE, "index")

    def test_indexes_cover_the_declared_columns(self, pg_head_drift):
        """Every index the chain builds must match its declaration exactly."""
        assert pg_head_drift.wrong_index_columns == set()

    def test_orm_not_null_is_installed(self, pg_head_drift):
        """``nullable=False`` in the ORM must be NOT NULL in the schema."""
        assert_ratchet(pg_head_drift.nullable_in_live, NULLABILITY_BASELINE, "nullability")

    def test_documents_alignment_is_not_in_the_ledger(self, pg_head_drift):
        """The two drifts migration 055 fixes must be gated, not ledgered.

        Ledgering them would make the gate green whether or not 054 exists.
        Their absence from both ledgers is what makes removing 054 fail the
        two tests above by name.
        """
        assert "documents.ix_documents_namespace_source_type" not in INDEX_BASELINE
        assert "documents.source_type" not in NULLABILITY_BASELINE
        assert "documents.ix_documents_namespace_source_type" not in pg_head_drift.missing_indexes
        assert "documents.source_type" not in pg_head_drift.nullable_in_live
