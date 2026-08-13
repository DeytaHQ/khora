"""``058_khora_chunks_title_fts`` — PostgreSQL lane (#1574).

Migration 041 added the denormalized ``khora_chunks.title`` column and 044
backfilled it, but ``khora_chunks_content_tsv_trigger()`` only ever computed
``to_tsvector('english', NEW.content)``. A chunk whose *title* was the only
place a term appeared was invisible to the lexical channel. 058 swaps the
function to a weighted concatenation (``title`` -> ``'A'``, ``content`` ->
``'B'``) and recomputes every stored vector.

Postgres-only by construction, and there is no SQLite twin: ``khora_chunks``
does not exist on the embedded stack — that tier gets its title FTS from the
sqlite_lance store's own DDL, covered by
``tests/unit/storage/test_sqlite_lance_title_fts.py``. The one thing this
revision shares with the runtime store — the trigger-function body, which both
sites ``CREATE OR REPLACE`` — is pinned byte-for-byte in the unit lane by
``tests/unit/db/test_migration_058_lockstep.py``, so that contract is checked on
every PR even without a server.

What is actually exercised here, none of which a string test can reach:

1. The 0-hit repro, end to end through ``PgVectorTemporalStore.search_fulltext``
   — before the migration a title-only query returns nothing, after it returns
   the chunk. Seeded with the verbatim repro title
   ``Floor Panels_Dimensioned_20260213``, which doubles as a tokenizer guard:
   the ``english`` configuration must split it on the underscores.
2. The stored vector carries the ``A`` / ``B`` weight labels.
3. **Rank equality** — the assertion that makes "the default is not a behavior
   change" a measured fact rather than an argument. Postgres' implicit
   ``ts_rank_cd`` weights are ``{0.1, 0.2, 0.4, 1.0}``, so relabelling content
   from unlabelled (``D`` = 0.1) to ``'B'`` (= 0.4) would quadruple every
   content hit's score. The store therefore passes the vector explicitly, with
   ``B`` pinned back to ``0.1``. This compares the post-migration score under
   that vector against the pre-migration score under the *implicit* default on
   the *unlabelled* vector, on the same row and the same query, and requires
   them to be equal.
4. ``downgrade()`` genuinely reverses: title tokens stop matching and the labels
   are gone.

Why the pre-migration state is set up by hand
---------------------------------------------
Building the schema at 057 is not enough to produce a "before". ``khora_chunks``
is runtime-managed, and the runtime's ``connect()`` installs the *new* function
unconditionally — so a store connected at 057 already indexes titles. The state
058 exists to repair is "table created by an OLD khora runtime", which this
module reproduces by installing the content-only body (imported from the
revision's own ``downgrade`` constant, so it cannot drift from what a downgrade
produces) over the function the store just created.

Each class owns a throwaway database (``tests/test_helpers/pg_scratch_db.py``)
rather than touching the shared dev one: these tests downgrade and re-upgrade,
and CI runs the integration job with ``--timeout-method=thread``, which kills
the process outright so ``finally`` blocks never run.

Run explicitly::

    KHORA_DATABASE_URL="postgresql://khora:khora@localhost:5434/khora" \\
        UV_NO_SYNC=1 uv run pytest \\
        tests/integration/db/test_migration_058_khora_chunks_title_fts.py \\
        -o addopts="" --no-cov -q
"""

from __future__ import annotations

import asyncio
import importlib
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import create_async_engine

from khora.config.schema import KhoraConfig
from khora.core.temporal import TemporalChunk
from khora.storage.temporal.pgvector import PgVectorTemporalStore
from tests.test_helpers.pg_scratch_db import pg_reachable, scratch_database

pytestmark = [pytest.mark.integration]

skip_no_pg = pytest.mark.skipif(
    not pg_reachable(),
    reason="PostgreSQL not reachable (run `make dev` first)",
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "src" / "khora" / "db" / "migrations"

REVISION = "058_khora_chunks_title_fts"
PREV_REVISION = "057_drop_documents_created_at_index"

#: The pre-#1574 function body, taken from the revision's own ``downgrade``
#: constant. Imported rather than re-spelled so "the state before the upgrade"
#: and "the state after a downgrade" are the same string by construction — a
#: local copy could drift and quietly make the before/after comparison vacuous.
_MIGRATION = importlib.import_module(f"khora.db.migrations.versions.{REVISION}")
CONTENT_ONLY_FUNCTION_SQL: str = _MIGRATION._TSV_FUNCTION_SQL_CONTENT_ONLY

#: Verbatim from the repro, and a tokenizer regression guard: the ``english``
#: text-search configuration must split on ``_``, yielding ``floor`` / ``panel``
#: / ``dimens`` / ``20260213``. A configuration that kept the underscores would
#: index one unmatchable token and every title assertion below would fail.
TITLE = "Floor Panels_Dimensioned_20260213"

#: Shares no vocabulary with :data:`TITLE` — load-bearing, or a title hit and a
#: content hit would be indistinguishable.
BODY = "the assembly drawing revision notes for the north wing"

#: A query whose terms live only in :data:`BODY`. The rank-equality comparison
#: has to be a *content-only* match: it is asking whether an already-working
#: content hit still scores what it scored before.
CONTENT_QUERY = "assembly drawing"

#: Queries whose terms live only in :data:`TITLE`.
TITLE_QUERY = "floor panels dimensioned"
TITLE_NUMERIC_QUERY = "20260213"

#: The tsquery ``_bm25_search`` builds for :data:`CONTENT_QUERY` — same
#: alnum-split + ``|`` join, so the rank numbers below come from the query the
#: store actually issues rather than a lookalike.
CONTENT_TSQUERY = "assembly | drawing"

#: The vector the store passes at ``title_weight=1.0``, in Postgres'
#: ``{D, C, B, A}`` order.
NEUTRAL_WEIGHTS = "{0.1, 0.2, 0.1, 0.1}"


def make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # Alembic uses configparser.BasicInterpolation; escape any literal '%' so it
    # is not read as an interpolation token.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg


def _async_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


async def _scalar(url: str, statement: str, params: dict | None = None):
    engine = create_async_engine(_async_url(url))
    try:
        async with engine.connect() as conn:
            result = await conn.execute(sa.text(statement), params or {})
            return result.scalar()
    finally:
        await engine.dispose()


async def _execute(url: str, statement: str) -> None:
    engine = create_async_engine(_async_url(url))
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.text(statement))
    finally:
        await engine.dispose()


def _store_config(url: str) -> KhoraConfig:
    config = KhoraConfig()
    config.storage.postgresql_url = SecretStr(_async_url(url))
    # Small vectors: the store builds an HNSW index at connect() and nothing
    # here searches by vector. Keeps the fixture's setup cost near zero.
    config.llm.embedding_dimension = 8
    return config


async def _connected_store(url: str) -> PgVectorTemporalStore:
    store = PgVectorTemporalStore(_store_config(url))
    await store.connect()
    return store


def _chunk(namespace_id: UUID, *, content: str, title: str | None) -> TemporalChunk:
    return TemporalChunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=uuid4(),
        content=content,
        embedding=[0.0] * 7 + [1.0],
        occurred_at=datetime.now(UTC),
        created_at=datetime.now(UTC),
        title=title,
    )


class _Seeded:
    """A scratch database sitting at 057 in the genuine pre-#1574 state."""

    def __init__(self, url: str, namespace_id: UUID, chunk_id: UUID) -> None:
        self.url = url
        self.namespace_id = namespace_id
        self.chunk_id = chunk_id
        self.config = make_config(url)


async def _search(url: str, seeded: _Seeded, query: str, **kwargs) -> list[UUID]:
    """Chunk ids the lexical channel returns, best-ranked first.

    Goes through the real store rather than raw SQL so the assertions cover the
    query the product issues — the ``|``-joined tsquery, the explicit weights
    vector, and the ``@@`` predicate — not a hand-written approximation.

    Note what connecting a store costs: ``connect()`` reinstalls the *new*
    trigger function every time, so after the first call here the content-only
    body is gone. That does not weaken anything below, because the trigger only
    fires on INSERT/UPDATE and every assertion in this module reads *stored*
    vectors — which only the migration's recompute rewrites. It does mean the
    downgrade test proves the downgrade's **recompute**, not that its function
    swap survives a subsequent runtime boot; that it cannot survive one is the
    documented mixed-version window, and the function bodies themselves are
    pinned in the unit lane. Do not "fix" this by moving the reinstall.
    """
    store = await _connected_store(url)
    try:
        rows = await store.search_fulltext(seeded.namespace_id, query, limit=10, **kwargs)
        return [chunk.id for chunk, _score in rows]
    finally:
        await store.disconnect()


# A ``khora_chunks`` predating migration 041 — the identity/content/tsv columns
# and NOTHING else. Hand-written rather than produced by the store, because the
# store's ``connect()`` builds the current (title-bearing) shape and there is no
# chain over this runtime-managed table to walk backwards. The trigger installed
# alongside it is the content-only formula, which is what such a database has.
_TITLE_LESS_KHORA_CHUNKS_DDL = (
    """
    CREATE TABLE khora_chunks (
        id UUID PRIMARY KEY,
        namespace_id UUID NOT NULL,
        document_id UUID NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ,
        content_tsv TSVECTOR
    )
    """,
    CONTENT_ONLY_FUNCTION_SQL,
    """
    CREATE TRIGGER khora_chunks_content_tsv_update
    BEFORE INSERT OR UPDATE ON khora_chunks
    FOR EACH ROW EXECUTE FUNCTION khora_chunks_content_tsv_trigger()
    """,
    """
    INSERT INTO khora_chunks (id, namespace_id, document_id, content, created_at)
    VALUES (gen_random_uuid(), gen_random_uuid(), gen_random_uuid(), 'legacy chunk body', NOW())
    """,
)

#: A write issued *after* the migration ran. If 058 had swapped the function in,
#: this INSERT would raise ``UndefinedColumnError: record "new" has no field
#: "title"`` — the failure that outlives the revision.
_TITLE_LESS_SECOND_INSERT_SQL = """
INSERT INTO khora_chunks (id, namespace_id, document_id, content, created_at)
VALUES (gen_random_uuid(), gen_random_uuid(), gen_random_uuid(), 'written after the migration', NOW())
"""


async def _install_title_less_khora_chunks(url: str) -> None:
    for statement in _TITLE_LESS_KHORA_CHUNKS_DDL:
        await _execute(url, statement)


async def _seed_rows(url: str) -> tuple[UUID, UUID]:
    """Create the runtime table, revert its function, and insert one chunk.

    ``connect()`` creates the runtime table, its indexes and its trigger; the
    content-only function is then installed over the new one so the trigger that
    fires on the seed INSERT computes the pre-#1574 vector. That is what an
    old khora instance would have left behind, and it is the only starting state
    from which "before the migration" means anything.
    """
    store = await _connected_store(url)
    try:
        namespace_id = uuid4()
        await _execute(url, CONTENT_ONLY_FUNCTION_SQL)
        chunk = _chunk(namespace_id, content=BODY, title=TITLE)
        await store.create_chunk(chunk)
    finally:
        await store.disconnect()
    return namespace_id, chunk.id


@pytest.fixture
def seeded() -> Iterator[_Seeded]:
    """A throwaway database at 057, seeded, in the pre-migration state.

    Function-scoped: every test here moves the schema, and the setup is a few
    seconds against an empty database — cheaper than the coupling a shared,
    rewound fixture would introduce.

    ``command.upgrade`` runs OUTSIDE the ``asyncio.run`` below on purpose:
    Alembic's async ``env.py`` opens its own event loop, so calling it from
    inside a coroutine raises ``asyncio.run() cannot be called from a running
    event loop``. Every test in this module keeps the same split.
    """
    with scratch_database("mig058") as url:
        command.upgrade(make_config(url), PREV_REVISION)
        namespace_id, chunk_id = asyncio.run(_seed_rows(url))
        yield _Seeded(url, namespace_id, chunk_id)


@skip_no_pg
class TestMigration058TitleFts:
    def test_title_is_not_searchable_before_the_upgrade(self, seeded: _Seeded) -> None:
        """The bug, reproduced. Body words find the chunk; title words do not.

        The body query is the control: it proves the seed is searchable at all,
        so the two empty results below are the missing title tokens and not an
        empty table or a mis-scoped namespace.
        """
        assert asyncio.run(_search(seeded.url, seeded, CONTENT_QUERY)) == [seeded.chunk_id]
        assert asyncio.run(_search(seeded.url, seeded, TITLE_QUERY)) == []
        assert asyncio.run(_search(seeded.url, seeded, TITLE_NUMERIC_QUERY)) == []

    def test_upgrade_makes_title_searchable(self, seeded: _Seeded) -> None:
        """After 058 the SAME rows answer a title-only query.

        No re-ingest: the revision recomputes ``content_tsv`` in place, so this
        also proves the backfill ran rather than only the function swap.
        """
        command.upgrade(seeded.config, REVISION)

        assert asyncio.run(_search(seeded.url, seeded, TITLE_QUERY)) == [seeded.chunk_id]
        # The bare numeric token separately: it is the half a tokenizer that
        # swallowed the surrounding underscores would lose.
        assert asyncio.run(_search(seeded.url, seeded, TITLE_NUMERIC_QUERY)) == [seeded.chunk_id]
        # And content is still searchable — the concatenation added a label to
        # the content tokens, it did not replace them.
        assert asyncio.run(_search(seeded.url, seeded, CONTENT_QUERY)) == [seeded.chunk_id]

    def test_upgrade_labels_the_stored_vector(self, seeded: _Seeded) -> None:
        """``title`` tokens carry ``A``, ``content`` tokens carry ``B``.

        Read off ``content_tsv::text`` because the labels are what the weights
        vector addresses; a title-token search passing tells you the terms are
        present, not that they are separable from the body.
        """
        before = asyncio.run(_scalar(seeded.url, "SELECT content_tsv::text FROM khora_chunks"))
        assert not re.search(r":\d+[A-C]", before), f"pre-migration vector should be unlabelled: {before}"

        command.upgrade(seeded.config, REVISION)

        after = asyncio.run(_scalar(seeded.url, "SELECT content_tsv::text FROM khora_chunks"))
        assert re.search(r"'dimens':\d+A", after), f"title tokens must be weighted 'A': {after}"
        assert re.search(r"'assembl':\d+B", after), f"content tokens must be weighted 'B': {after}"

    def test_default_weights_reproduce_the_pre_migration_rank_exactly(self, seeded: _Seeded) -> None:
        """The compatibility claim, measured on one row across the upgrade.

        Before: the unlabelled vector under ``ts_rank_cd``'s implicit default.
        After: the relabelled vector under the ``{0.1, 0.2, 0.1, 0.1}`` the
        store passes at ``title_weight=1.0``.

        Equality is the whole point of pinning ``B`` back to ``0.1``: Postgres'
        implicit default weights ``B`` at 0.4, so an implementation that simply
        relabelled and kept relying on the default would have multiplied every
        content hit's score by four — a silent, global ranking change shipped
        under a feature that is supposed to be off by default. The control below
        makes that concrete rather than hypothetical.
        """
        rank_sql = "SELECT ts_rank_cd({0}content_tsv, to_tsquery('english', :q)) FROM khora_chunks"
        params = {"q": CONTENT_TSQUERY}

        before = asyncio.run(_scalar(seeded.url, rank_sql.format(""), params))
        assert before > 0, "the pre-migration content match must score something to compare against"

        command.upgrade(seeded.config, REVISION)

        after = asyncio.run(_scalar(seeded.url, rank_sql.format(f"'{NEUTRAL_WEIGHTS}'::float4[], "), params))
        assert after == pytest.approx(before, rel=1e-9), (
            f"title_weight=1.0 must reproduce the pre-058 ranking exactly: {before} -> {after}"
        )

        # Control: the same relabelled vector under the IMPLICIT default is a
        # different number. Without this, an implementation that dropped the
        # explicit vector entirely would still pass the equality above if the
        # two happened to coincide.
        implicit = asyncio.run(_scalar(seeded.url, rank_sql.format(""), params))
        assert implicit != pytest.approx(before, rel=1e-9), (
            "relabelling content to 'B' must change the IMPLICIT-default score — "
            "otherwise this test is not measuring what it claims"
        )

    def test_raised_title_weight_lifts_only_the_title_match(self, seeded: _Seeded) -> None:
        """The knob does something once the labels exist — and only to titles.

        Two chunks matching the same query, one through its title and one
        through its body. Raising ``title_weight`` moves the ``A`` slot of the
        weights vector and leaves ``B`` alone, so the exact expected effect is:
        the title match's score rises, the body match's score does not move at
        all, and the order flips. Asserting the scores rather than only the
        order is what makes this non-vacuous — an order assertion alone would
        pass if the titled chunk happened to lead at the neutral weight too.

        Seeded after the upgrade so both rows go through the new trigger.
        """
        command.upgrade(seeded.config, REVISION)

        async def _run() -> tuple[dict[UUID, float], dict[UUID, float], UUID, UUID]:
            store = await _connected_store(seeded.url)
            try:
                ns = uuid4()
                titled = _chunk(ns, content="unrelated filler prose", title="Gasket Torque Specification")
                bodied = _chunk(ns, content="gasket torque specification and nothing else", title=None)
                await store.create_chunks_batch([titled, bodied])
                query = "gasket torque"
                neutral = {c.id: s for c, s in await store.search_fulltext(ns, query, limit=10)}
                weighted = {c.id: s for c, s in await store.search_fulltext(ns, query, limit=10, title_weight=8.0)}
                return neutral, weighted, titled.id, bodied.id
            finally:
                await store.disconnect()

        neutral, weighted, titled_id, bodied_id = asyncio.run(_run())

        assert set(neutral) == {titled_id, bodied_id}, "both chunks must match; the weight orders, it does not filter"
        assert set(weighted) == set(neutral)
        assert weighted[titled_id] > neutral[titled_id], "the title match must score higher at a raised weight"
        assert weighted[bodied_id] == pytest.approx(neutral[bodied_id]), (
            "title_weight moves only the 'A' slot — a body-only match must be untouched"
        )
        assert max(weighted, key=weighted.__getitem__) == titled_id

    def test_downgrade_restores_content_only_matching(self, seeded: _Seeded) -> None:
        """058 is fully reversible — the vector is derived data.

        Both halves are asserted: title tokens stop matching (the terms were
        really removed, not merely relabelled) and the labels are gone from the
        stored vector, while the content match survives untouched.
        """
        command.upgrade(seeded.config, REVISION)
        assert asyncio.run(_search(seeded.url, seeded, TITLE_QUERY)) == [seeded.chunk_id]

        command.downgrade(seeded.config, PREV_REVISION)

        assert asyncio.run(_search(seeded.url, seeded, TITLE_QUERY)) == []
        assert asyncio.run(_search(seeded.url, seeded, TITLE_NUMERIC_QUERY)) == []
        assert asyncio.run(_search(seeded.url, seeded, CONTENT_QUERY)) == [seeded.chunk_id]

        vector = asyncio.run(_scalar(seeded.url, "SELECT content_tsv::text FROM khora_chunks"))
        assert not re.search(r":\d+[A-C]", vector), f"downgrade must leave an unlabelled vector: {vector}"

    def test_re_running_the_upgrade_is_idempotent(self, seeded: _Seeded) -> None:
        """Down-and-up returns the same vector, byte for byte.

        The revision has no sentinel distinguishing an already-recomputed row,
        so a re-run rewrites the whole table — deliberately. What must hold is
        that the *outcome* is stable; the cost is documented, not asserted.
        """
        command.upgrade(seeded.config, REVISION)
        first = asyncio.run(_scalar(seeded.url, "SELECT content_tsv::text FROM khora_chunks"))

        command.downgrade(seeded.config, PREV_REVISION)
        command.upgrade(seeded.config, REVISION)
        second = asyncio.run(_scalar(seeded.url, "SELECT content_tsv::text FROM khora_chunks"))

        assert second == first

    def test_null_title_does_not_blank_the_vector(self, seeded: _Seeded) -> None:
        """The ``coalesce`` guard: ``to_tsvector(NULL)`` is NULL, and NULL
        concatenated with anything is NULL — which would erase the content
        tokens of every untitled chunk. ``title`` is nullable and 044's backfill
        leaves it NULL wherever the parent document had none, so this is a real
        row shape, not a contrived one."""
        command.upgrade(seeded.config, REVISION)

        async def _run() -> list[UUID]:
            store = await _connected_store(seeded.url)
            try:
                ns = uuid4()
                untitled = _chunk(ns, content=BODY, title=None)
                await store.create_chunk(untitled)
                rows = await store.search_fulltext(ns, CONTENT_QUERY, limit=10)
                return [c.id for c, _ in rows]
            finally:
                await store.disconnect()

        assert asyncio.run(_run()), "an untitled chunk must still be content-searchable"


@skip_no_pg
class TestMigration058NoOpPaths:
    def test_upgrade_is_a_no_op_without_khora_chunks(self) -> None:
        """A fresh Postgres deploy reaches 058 before any store has connected.

        ``khora_chunks`` is runtime-managed, so on a database that has never run
        the app the table does not exist and the revision must pass through
        cleanly — the store's own ``connect()`` installs the new function
        moments later and there are no rows to recompute.
        """
        with scratch_database("mig058_fresh") as url:
            cfg = make_config(url)
            command.upgrade(cfg, PREV_REVISION)
            assert asyncio.run(_scalar(url, "SELECT to_regclass('public.khora_chunks')")) is None

            command.upgrade(cfg, REVISION)

            assert asyncio.run(_scalar(url, "SELECT version_num FROM khora_alembic_version")) == REVISION
            assert asyncio.run(_scalar(url, "SELECT to_regclass('public.khora_chunks')")) is None

    def test_downgrade_is_a_no_op_without_khora_chunks(self) -> None:
        """Same guard on the way back down, so a rewind cannot fail on a
        database that never ran the app."""
        with scratch_database("mig058_fresh_down") as url:
            cfg = make_config(url)
            command.upgrade(cfg, REVISION)

            command.downgrade(cfg, PREV_REVISION)

            assert asyncio.run(_scalar(url, "SELECT version_num FROM khora_alembic_version")) == PREV_REVISION

    def test_upgrade_is_a_no_op_when_the_table_has_no_title_column(self) -> None:
        """A ``khora_chunks`` that predates 041 must be skipped, not broken.

        The second half of the precondition, and the one with teeth. plpgsql
        resolves ``NEW.title`` when the trigger *fires*, not when the function is
        created — so against a title-less table the ``CREATE OR REPLACE
        FUNCTION`` and ``CREATE TRIGGER`` both succeed and the recompute
        ``UPDATE`` is what raises ``UndefinedColumnError: record "new" has no
        field "title"``. Without the gate the damage would also outlive the
        statement: the swapped function stays installed, so every later INSERT
        or UPDATE to ``khora_chunks`` fails too.

        Three assertions, in increasing order of what they would have caught:
        the upgrade completes and stamps 058; the seeded row's vector is
        byte-unchanged (nothing was recomputed); and — the one that matters — a
        row written *after* the migration still gets a content-only vector,
        proving the weighted function was never installed. Only the third
        distinguishes "gated" from "swapped the function but happened not to
        rewrite any rows".
        """
        with scratch_database("mig058_no_title_col") as url:
            cfg = make_config(url)
            command.upgrade(cfg, PREV_REVISION)
            asyncio.run(_install_title_less_khora_chunks(url))

            before = asyncio.run(_scalar(url, "SELECT content_tsv::text FROM khora_chunks"))
            assert before, "precondition: the seeded trigger must populate content_tsv on INSERT"
            assert not re.search(r":\d+[A-C]", before), f"precondition: seed must be unlabelled: {before}"

            command.upgrade(cfg, REVISION)

            assert asyncio.run(_scalar(url, "SELECT version_num FROM khora_alembic_version")) == REVISION
            after = asyncio.run(_scalar(url, "SELECT content_tsv::text FROM khora_chunks"))
            assert after == before, "the gate must skip the recompute on a title-less table"

            asyncio.run(_execute(url, _TITLE_LESS_SECOND_INSERT_SQL))
            written_after = asyncio.run(
                _scalar(url, "SELECT content_tsv::text FROM khora_chunks WHERE content = 'written after the migration'")
            )
            assert written_after, "the post-migration INSERT must succeed and populate content_tsv"
            assert not re.search(r":\d+[A-C]", written_after), (
                f"the weighted function was installed on a title-less table: {written_after}"
            )

    def test_downgrade_is_a_no_op_when_the_table_has_no_title_column(self) -> None:
        """The gate lives in the shared ``_run``, so both directions skip the
        same databases. A downgrade that tried to recompute a table the upgrade
        had skipped would fail identically, and for the same reason."""
        with scratch_database("mig058_no_title_col_down") as url:
            cfg = make_config(url)
            command.upgrade(cfg, PREV_REVISION)
            asyncio.run(_install_title_less_khora_chunks(url))
            command.upgrade(cfg, REVISION)
            before = asyncio.run(_scalar(url, "SELECT content_tsv::text FROM khora_chunks"))

            command.downgrade(cfg, PREV_REVISION)

            assert asyncio.run(_scalar(url, "SELECT version_num FROM khora_alembic_version")) == PREV_REVISION
            assert asyncio.run(_scalar(url, "SELECT content_tsv::text FROM khora_chunks")) == before
