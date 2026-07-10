"""Move chunk bookkeeping keys from khora_chunks.metadata into chunker_info.

Revision ID: 053_khora_chunks_bookkeeping_to_chunker_info
Revises: 052_entities_source_chunk_ids_gin
Create Date: 2026-07-10

Data change only — NO schema change. The four chunk-bookkeeping keys
``chunk_index`` / ``start_char`` / ``end_char`` / ``token_count`` used to
live in ``khora_chunks.metadata`` (the temporal store's user/document
metadata blob). The writer/reader refactor (khora#1491) moved them into
``khora_chunks.chunker_info`` at every chunk-writer site and the
temporal-chunk reader now sources the four fields from ``chunker_info``
**exclusively, with no metadata fallback** — a legacy row that still
carries them only in ``metadata`` reads back as zeros. This migration is
the backfill companion: it relocates the four keys on every existing row
so those rows keep reading correctly after the reader change lands. It
must run before release for the same reason.

Cross-dialect: unlike 043/044 this migration runs on **both** dialects.
``khora_chunks`` exists on Postgres (``PgVectorTemporalStore``) and on the
sqlite_lance embedded stack, so the SQLite branch performs the same
relocation with the JSON1 functions (``json_patch`` / ``json_remove`` /
``json_extract``) instead of the Postgres JSONB operators. The
sqlite_lance fixture stack runs the full Alembic chain, so a SQLite no-op
would leave embedded rows unmigrated — the SQLite branch is load-bearing,
not a green-keeper.

Two tiers per row, keyed on whether the row has a twin in the main
``chunks`` table (``chunks.id = khora_chunks.id``, the same UUID):

* **Tier 1 (twin exists).** The main ``chunks`` row carries the four
  values in *typed* columns (``chunk_index`` etc., ``NOT NULL`` integers)
  plus its own ``chunker_info``. The relocation copies the typed column
  values into ``khora_chunks.chunker_info`` (merged after the twin's
  ``chunker_info``) and strips a metadata key **only where its metadata
  value equals the typed column value**. A metadata key whose value
  differs from the column is a user key that merely collides on name — it
  is left in ``metadata`` untouched, while ``chunker_info`` still gets the
  authoritative column value.

* **Tier 2 (no twin).** No typed columns to trust, so the values are read
  from ``metadata`` itself. A key is moved only when its metadata value is
  number-typed (``jsonb_typeof = 'number'`` on Postgres, ``json_type =
  'integer'`` on SQLite); a string-typed collision is a user key and stays
  in ``metadata``. Exactly the moved keys are stripped. No synthetic
  ``chunker`` name is fabricated — only the four numeric keys move.

Merge precedence (both tiers, both dialects): the row's own existing
``chunker_info`` is the base and always wins over nothing, the twin's
``chunker_info`` (Tier 1) layers on top, and the four bookkeeping keys are
stamped **last** so they win any collision — mirroring the writer, which
stamps the bookkeeping keys last. On Postgres this is
``kc.chunker_info || c.chunker_info || bookkeeping`` (right operand wins in
JSONB concat). On SQLite ``json_patch(a, b)`` lets ``b`` win, so the calls
nest ``json_patch(json_patch(kc.chunker_info, c.chunker_info),
bookkeeping)`` to reproduce the same base → twin → bookkeeping order.

Idempotency / convergence (both tiers, both dialects): the UPDATE is
guarded by ``metadata ?| ARRAY[four keys] AND NOT (chunker_info ?
'chunk_index')`` — a row is visited only while it still carries any of the
four keys in ``metadata`` *and* has not yet had ``chunk_index`` written to
``chunker_info``. Once relocated, ``chunker_info`` carries ``chunk_index``
so the row is skipped, and re-running the migration touches zero rows. A
row where a differing-value user key legitimately remains in ``metadata``
is still skipped on re-run because ``chunker_info`` now has ``chunk_index``.

Scale (~1M chunk rows): the relocation runs namespace-batched — one Tier-1
UPDATE and one Tier-2 UPDATE per distinct ``khora_chunks.namespace_id`` —
so no single statement spans the whole table. The whole Alembic chain runs
inside ONE transaction (``env.py:do_run_migrations``), so batching bounds
per-statement cost and planning, not lock-hold duration.

Trigger suppression (Postgres): ``khora_chunks`` carries a
``BEFORE INSERT OR UPDATE`` trigger (``khora_chunks_content_tsv_update``)
that recomputes ``to_tsvector('english', content)`` on every updated row.
This migration touches only ``metadata`` / ``chunker_info`` and never
``content``, so re-deriving the tsvector is wasted CPU on a ~1M-row table.
The trigger is disabled for the relocation and re-enabled in a ``finally``
so it is restored on every exit path. ``DISABLE TRIGGER`` / ``ENABLE
TRIGGER`` is transactional, so a rolled-back migration leaves the trigger
enabled. SQLite has no such trigger and skips the toggle.

Lock-timeout safety (Postgres): a ``SET lock_timeout = '5s'`` bounds the
brief ``ACCESS EXCLUSIVE`` lock ``DISABLE TRIGGER`` takes and the row
locks the UPDATEs take. On lock-timeout the upgrade logs
``khora.migration.applied`` at ERROR with ``lock_timeout_tripped=True``
(Postgres SQLSTATE ``55P03`` on ``OperationalError.orig.pgcode``); any
other ``OperationalError`` logs ``lock_timeout_tripped=False`` so
dashboards don't conflate deadlocks / connection drops with the
lock-timeout signal. Either path re-raises so Alembic rolls back.

Runtime-def convergence: ``khora_chunks`` is created at runtime by
``PgVectorTemporalStore.connect()`` (and the embedded store's DDL), not by
the Alembic-managed schema, and the main ``chunks`` table is
Alembic-managed. Both ``has_table`` guards early-return ``(0, 0)`` when
either table is absent (fresh deploys where the temporal store hasn't
booted yet) — new chunks already carry the split from the ingest path, so
there is nothing to relocate.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from loguru import logger
from sqlalchemy.exc import OperationalError

revision: str = "053_khora_chunks_bookkeeping_to_chunker_info"
down_revision: str | Sequence[str] | None = "052_entities_source_chunk_ids_gin"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# PostgreSQL SQLSTATE for "lock_not_available" — what `lock_timeout` raises
# when an acquisition exceeds the configured timeout.
_PG_LOCK_NOT_AVAILABLE = "55P03"

# Name of the BEFORE INSERT OR UPDATE trigger on khora_chunks that recomputes
# content_tsv. Defined at runtime in
# ``src/khora/storage/temporal/pgvector.py``.
_CONTENT_TSV_TRIGGER = "khora_chunks_content_tsv_update"


# ---------------------------------------------------------------------------
# Postgres statements
# ---------------------------------------------------------------------------

# Tier 1 — the row has a twin in the main ``chunks`` table. Copy the twin's
# typed column values into chunker_info (merged after the twin's own
# chunker_info, bookkeeping stamped last so it wins), and strip a metadata key
# ONLY where its metadata value equals the typed column value (a differing
# value is a user key that merely collides on name and stays put).
_PG_TIER1_SQL = sa.text(
    """
    UPDATE khora_chunks kc
    SET chunker_info = kc.chunker_info || c.chunker_info || jsonb_build_object(
            'chunk_index', c.chunk_index, 'start_char', c.start_char,
            'end_char', c.end_char, 'token_count', c.token_count),
        metadata = kc.metadata - ARRAY(
            SELECT t.k FROM (VALUES
                ('chunk_index', to_jsonb(c.chunk_index)),
                ('start_char',  to_jsonb(c.start_char)),
                ('end_char',    to_jsonb(c.end_char)),
                ('token_count', to_jsonb(c.token_count))) AS t(k, v)
            WHERE kc.metadata -> t.k = t.v)
    FROM chunks c
    WHERE c.id = kc.id AND kc.namespace_id = :namespace_id
      AND kc.metadata ?| ARRAY['chunk_index','start_char','end_char','token_count']
      AND NOT (kc.chunker_info ? 'chunk_index')
    """
)

# Tier 2 — no twin. Read the values from metadata itself, moving a key only
# when its metadata value is number-typed (a string-typed collision is a user
# key and stays). Strip exactly the moved keys. No synthetic chunker name.
_PG_TIER2_SQL = sa.text(
    """
    UPDATE khora_chunks kc
    SET chunker_info = kc.chunker_info || (
            SELECT COALESCE(jsonb_object_agg(t.k, kc.metadata -> t.k), '{}'::jsonb)
            FROM (VALUES ('chunk_index'), ('start_char'), ('end_char'), ('token_count')) AS t(k)
            WHERE jsonb_typeof(kc.metadata -> t.k) = 'number'),
        metadata = kc.metadata - ARRAY(
            SELECT t.k FROM (VALUES
                ('chunk_index'), ('start_char'), ('end_char'), ('token_count')) AS t(k)
            WHERE jsonb_typeof(kc.metadata -> t.k) = 'number')
    WHERE kc.namespace_id = :namespace_id
      AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.id = kc.id)
      AND kc.metadata ?| ARRAY['chunk_index','start_char','end_char','token_count']
      AND NOT (kc.chunker_info ? 'chunk_index')
    """
)


# ---------------------------------------------------------------------------
# SQLite statements (JSON1)
# ---------------------------------------------------------------------------

# Tier 1 — SQLite JSON1. ``json_patch(a, b)`` lets ``b`` win, so the nesting
# base(kc.chunker_info) → twin(c.chunker_info) → bookkeeping reproduces the
# Postgres ``||`` precedence. The bookkeeping object carries the twin's typed
# column values. A metadata key is stripped only where its value equals the
# typed column value (compared via json_extract on both sides). ``id`` is TEXT
# on SQLite, so the twin join is a plain equality. json_type() returns
# 'integer' for whole numbers (not 'number'); the guard on chunker_info uses
# the same ``json_extract(... , '$.chunk_index') IS NULL`` skip as the Postgres
# ``NOT (chunker_info ? 'chunk_index')``.
#
# DIVERGENCE from Postgres ``||``: ``json_patch`` follows RFC 7386 and DELETES
# a key whose patch value is JSON null, where Postgres ``||`` would keep it as
# an explicit null. This is moot in practice — the bookkeeping overlay carries
# concrete integers from the twin's NOT NULL typed columns (never null), and a
# twin's ``chunker_info`` values (chunker-name string + ints) are non-null too
# — so no special-casing is added; the note stands only to explain why the
# two dialects can be trusted to converge on real data.
#
# NOTE on the strip: SQLite ``json_remove(x, path)`` returns NULL if *any*
# path argument is NULL, but is a plain no-op when the path does not exist. So
# the strip is expressed as nested ``json_remove`` calls, each keying the real
# path only when the metadata value equals the twin column and otherwise a
# guaranteed-absent sentinel path (a no-op) — never a NULL path arg (which
# would null out the NOT NULL ``metadata`` column). This mirrors the Postgres
# ``metadata - ARRAY(... WHERE value = column)`` value-equality strip.
_SQLITE_TIER1_SQL = sa.text(
    """
    UPDATE khora_chunks AS kc
    SET chunker_info = json_patch(
            json_patch(kc.chunker_info, c.chunker_info),
            json_object(
                'chunk_index', c.chunk_index, 'start_char', c.start_char,
                'end_char', c.end_char, 'token_count', c.token_count)),
        metadata = json_remove(json_remove(json_remove(json_remove(
            kc.metadata,
            CASE WHEN json_extract(kc.metadata, '$.chunk_index') = c.chunk_index
                 THEN '$.chunk_index' ELSE '$."__khora_noop__"' END),
            CASE WHEN json_extract(kc.metadata, '$.start_char') = c.start_char
                 THEN '$.start_char' ELSE '$."__khora_noop__"' END),
            CASE WHEN json_extract(kc.metadata, '$.end_char') = c.end_char
                 THEN '$.end_char' ELSE '$."__khora_noop__"' END),
            CASE WHEN json_extract(kc.metadata, '$.token_count') = c.token_count
                 THEN '$.token_count' ELSE '$."__khora_noop__"' END)
    FROM chunks AS c
    WHERE c.id = kc.id AND kc.namespace_id = :namespace_id
      AND (json_extract(kc.metadata, '$.chunk_index') IS NOT NULL
           OR json_extract(kc.metadata, '$.start_char') IS NOT NULL
           OR json_extract(kc.metadata, '$.end_char') IS NOT NULL
           OR json_extract(kc.metadata, '$.token_count') IS NOT NULL)
      AND json_extract(kc.chunker_info, '$.chunk_index') IS NULL
    """
)

# Tier 2 — no twin. Move a key only when its metadata value is integer-typed
# (SQLite ``json_type`` returns 'integer' for whole numbers, not 'number').
#
# The values are set with nested ``json_set`` guarded per key so ONLY
# integer-typed keys are ever set — never introducing a JSON ``null`` value.
# This matters: ``json_patch`` follows RFC 7386 and DELETES a target key whose
# patch value is ``null``, so a ``json_object`` with ``CASE ... ELSE NULL``
# would silently drop keys. ``json_set`` with a NULL *path* is a clean no-op
# (the value is ignored), so a non-integer key contributes nothing; this
# mirrors the Postgres ``jsonb_object_agg(... WHERE jsonb_typeof = 'number')``,
# which likewise only aggregates the number-typed keys.
#
# The strip uses the same nested-``json_remove`` + absent-path-sentinel pattern
# as Tier 1 (``json_remove`` returns NULL on a NULL path but no-ops on an
# absent path, so the two functions need different sentinels) — a non-integer
# key never contributes a NULL path arg that would null out the NOT NULL
# ``metadata`` column.
_SQLITE_TIER2_SQL = sa.text(
    """
    UPDATE khora_chunks AS kc
    SET chunker_info = json_set(json_set(json_set(json_set(
            kc.chunker_info,
            CASE WHEN json_type(kc.metadata, '$.chunk_index') = 'integer'
                 THEN '$.chunk_index' ELSE NULL END,
            json_extract(kc.metadata, '$.chunk_index')),
            CASE WHEN json_type(kc.metadata, '$.start_char') = 'integer'
                 THEN '$.start_char' ELSE NULL END,
            json_extract(kc.metadata, '$.start_char')),
            CASE WHEN json_type(kc.metadata, '$.end_char') = 'integer'
                 THEN '$.end_char' ELSE NULL END,
            json_extract(kc.metadata, '$.end_char')),
            CASE WHEN json_type(kc.metadata, '$.token_count') = 'integer'
                 THEN '$.token_count' ELSE NULL END,
            json_extract(kc.metadata, '$.token_count')),
        metadata = json_remove(json_remove(json_remove(json_remove(
            kc.metadata,
            CASE WHEN json_type(kc.metadata, '$.chunk_index') = 'integer'
                 THEN '$.chunk_index' ELSE '$."__khora_noop__"' END),
            CASE WHEN json_type(kc.metadata, '$.start_char') = 'integer'
                 THEN '$.start_char' ELSE '$."__khora_noop__"' END),
            CASE WHEN json_type(kc.metadata, '$.end_char') = 'integer'
                 THEN '$.end_char' ELSE '$."__khora_noop__"' END),
            CASE WHEN json_type(kc.metadata, '$.token_count') = 'integer'
                 THEN '$.token_count' ELSE '$."__khora_noop__"' END)
    WHERE kc.namespace_id = :namespace_id
      AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.id = kc.id)
      AND (json_extract(kc.metadata, '$.chunk_index') IS NOT NULL
           OR json_extract(kc.metadata, '$.start_char') IS NOT NULL
           OR json_extract(kc.metadata, '$.end_char') IS NOT NULL
           OR json_extract(kc.metadata, '$.token_count') IS NOT NULL)
      AND json_extract(kc.chunker_info, '$.chunk_index') IS NULL
    """
)


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _is_lock_timeout(exc: OperationalError) -> bool:
    """Distinguish a real lock_timeout trip from any other OperationalError.

    OperationalError is a broad SQLAlchemy class — it wraps deadlocks,
    connection drops, syntax errors, server shutdowns, AND lock_timeout
    failures. The structured log field ``lock_timeout_tripped`` must
    only be True for the latter so monitoring dashboards aren't misled.
    """
    orig = getattr(exc, "orig", None)
    if orig is None:
        return False
    return getattr(orig, "pgcode", None) == _PG_LOCK_NOT_AVAILABLE


def _upgrade_impl() -> tuple[int, int]:
    """Run the namespace-batched relocation. Returns ``(namespaces, rows)``.

    Both dialects share the same two-tier, per-namespace shape; only the SQL
    statements differ (JSONB operators vs JSON1 functions) and the Postgres
    branch adds the lock-timeout / trigger-toggle safety.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("khora_chunks") or not inspector.has_table("chunks"):
        # ``khora_chunks`` is created at runtime by the temporal store and the
        # main ``chunks`` table is Alembic-managed. On fresh deploys / before
        # the temporal store has booted one or both may be absent; new chunks
        # already carry the bookkeeping in chunker_info from the ingest path,
        # so there is nothing to relocate here.
        return 0, 0

    is_pg = _is_postgres()
    tier1_sql = _PG_TIER1_SQL if is_pg else _SQLITE_TIER1_SQL
    tier2_sql = _PG_TIER2_SQL if is_pg else _SQLITE_TIER2_SQL

    if is_pg:
        # Bound every lock acquisition — the brief ACCESS EXCLUSIVE lock that
        # ``DISABLE TRIGGER`` takes and the row locks the UPDATEs take — so a
        # stuck pg_stat_activity entry on khora_chunks cannot stall the deploy
        # past 5s. Issued before the trigger toggle.
        op.execute("SET lock_timeout = '5s'")
        # The content_tsv trigger only matters when ``content`` changes; this
        # relocation touches only ``metadata`` / ``chunker_info``, so
        # re-deriving the tsvector for every updated row is wasted CPU. Disable
        # it for the relocation and restore it in the ``finally``.
        op.execute(f"ALTER TABLE khora_chunks DISABLE TRIGGER {_CONTENT_TSV_TRIGGER}")

    try:
        namespace_ids = [row[0] for row in bind.execute(sa.text("SELECT DISTINCT namespace_id FROM khora_chunks"))]
        total_rows = 0
        for namespace_id in namespace_ids:
            # Tier 1 (twin) then Tier 2 (twinless) per namespace. The guard
            # predicates are disjoint per row (twin exists vs NOT EXISTS), so a
            # row is touched by at most one tier.
            r1 = bind.execute(tier1_sql, {"namespace_id": namespace_id})
            r2 = bind.execute(tier2_sql, {"namespace_id": namespace_id})
            total_rows += int(r1.rowcount or 0) + int(r2.rowcount or 0)
        return len(namespace_ids), total_rows
    finally:
        if is_pg:
            op.execute(f"ALTER TABLE khora_chunks ENABLE TRIGGER {_CONTENT_TSV_TRIGGER}")


def upgrade() -> None:
    start = time.monotonic()
    # Initialize log fields up-front so the error path always emits a uniform
    # event for dashboards / alerts.
    namespaces_migrated = 0
    rows_migrated = 0
    try:
        namespaces_migrated, rows_migrated = _upgrade_impl()
    except OperationalError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.bind(
            migration_id=revision,
            duration_ms=duration_ms,
            lock_timeout_tripped=_is_lock_timeout(exc),
            namespaces_migrated=namespaces_migrated,
            rows_migrated=rows_migrated,
        ).error("khora.migration.applied")
        # Bare ``raise`` re-raises the active exception with the original
        # traceback preserved. env.py's wrapping context.begin_transaction()
        # rolls back the UPDATEs from this revision (and the trigger toggle).
        raise

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.bind(
        migration_id=revision,
        duration_ms=duration_ms,
        lock_timeout_tripped=False,
        namespaces_migrated=namespaces_migrated,
        rows_migrated=rows_migrated,
    ).info("khora.migration.applied")


def downgrade() -> None:
    """Irreversible relocation — no-op.

    The upgrade moves the four bookkeeping keys from ``metadata`` into
    ``chunker_info`` (stripping only the keys whose metadata value matched
    the authoritative source). Once moved, the migration cannot distinguish
    a bookkeeping key that originated in ``metadata`` from one the writer
    always wrote to ``chunker_info``, so the relocation cannot be cleanly
    un-done. The downgrade is therefore a no-op on both dialects.
    """
