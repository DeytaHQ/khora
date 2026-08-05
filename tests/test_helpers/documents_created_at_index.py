"""Shared constants for the ``ix_documents_created_at`` drop migration.

Migration ``057_drop_documents_created_at_index`` has two lanes that run in
two different CI jobs, so the revision pins and the index name live here
rather than in either module:

* ``tests/unit/db/test_migration_057_drop_documents_created_at_index.py`` —
  the SQLite lane. Plain DDL, no server, so it belongs in the unit job and
  runs on every PR and for contributors without Docker.
* ``tests/integration/db/test_migration_057_drop_documents_created_at_index.py``
  — the Postgres lane, which drives the ``SET LOCAL lock_timeout`` branch
  against a real server and is selected by the integration job's
  ``-m integration``.

Splitting them is what makes both demonstrably execute, and the rule is: a
lane goes in the directory whose job selects it. The unit job selects by path
(``tests/unit/``); the integration job selects by marker
(``-m "integration and not filter_conformance"``). A ``unit``-marked class
sitting under ``tests/integration/`` therefore matches neither, and is run by
no CI job at all.

That is not hypothetical — it happened partway through the change that
introduced migration 055 and was fixed before merge, which is why 055 is
already split this way today and both of its lanes run. The commit that split
them records it. (Squash-merge means ``git log --diff-filter=A`` shows both
055 modules added together, so it cannot show the intermediate state.)

The revision pins are shared rather than duplicated because they are the one
thing that must not drift between the lanes: ``_ORIGIN`` in particular is the
revision whose unconditional drop makes 057's downgrade load-bearing, and a
lane pinning the wrong one would still pass while covering nothing.
"""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config

__all__ = [
    "BELOW_ORIGIN_REVISION",
    "HEAD_REVISION",
    "INDEX_NAME",
    "MIGRATIONS_DIR",
    "ORIGIN_REVISION",
    "PREV_REVISION",
    "UNCONDITIONAL_DROP_SQL",
    "indexed_columns",
    "make_config",
]

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "khora" / "db" / "migrations"

HEAD_REVISION = "057_drop_documents_created_at_index"
PREV_REVISION = "055_documents_source_type_alignment"

#: The revision that CREATED the index, and the one immediately below it.
#: ``ORIGIN_REVISION``'s ``downgrade()`` drops the index with a bare
#: ``op.drop_index(...)`` — no ``if_exists`` — so walking to
#: ``BELOW_ORIGIN_REVISION`` forces that statement to actually execute, which
#: is the whole reason 057's ``downgrade()`` has to restore the index.
ORIGIN_REVISION = "009_temporal_search_indexes"
BELOW_ORIGIN_REVISION = "008_entity_dedup_and_indexes"

INDEX_NAME = "ix_documents_created_at"

#: Exactly what ``ORIGIN_REVISION``'s ``downgrade()`` emits. Replayed verbatim
#: by both lanes against the post-downgrade schema. Replaying the statement is
#: the point: asserting ``INDEX_NAME in <reflected indexes>`` instead would
#: only restate 057's own source, and would still pass if the restore built
#: something the bare drop cannot find.
UNCONDITIONAL_DROP_SQL = f"DROP INDEX {INDEX_NAME}"


def indexed_columns(sql: str | None) -> list[str]:
    """Ordered key list out of a ``CREATE INDEX`` statement.

    Handles both spellings the lanes read: SQLite's ``sqlite_master.sql`` and
    Postgres' ``pg_indexes.indexdef``. In both, the key list is the first
    balanced parenthesised group.

    Naive ``sql[sql.index("(") + 1 : sql.rindex(")")]`` is wrong for two real
    shapes, and the SQLite lane maps this over EVERY index on the table rather
    than a hand-picked one, so it will meet them:

    * an expression key — ``(namespace_id, COALESCE(a, b))`` — where the last
      ``)`` is not the end of the key list and the top-level split must not
      break inside the nested call;
    * a partial index whose predicate contains parentheses —
      ``(a) WHERE lower(b) IS NOT NULL`` — where ``rindex(")")`` reaches past
      the key list entirely and yields ``['a) WHERE lower(b']``.

    Neither shape exists on ``documents`` today; ``ix_chunks_ns_temporal`` is
    the first and nothing is the second. Returning a wrong parse would not
    have raised — it would have silently failed to match, which is the kind of
    quiet wrongness a comparison helper should not have.

    A third shape is handled but does **not** exist anywhere in this schema:
    a parenthesis or comma inside a quoted region. Counting depth naively
    would treat the ``)`` in ``COALESCE(label, ')')`` as structural and close
    the key list early, silently dropping every key after it. No index in this
    chain currently quotes anything — the two partial predicates
    (``WHERE external_id IS NOT NULL``, ``WHERE status != 'failed'``) quote a
    literal with no parenthesis or comma in it, and every key is a bare
    identifier — so this is guarded against a future index rather than a
    present bug. Both SQL quoting forms are tracked: ``'`` for literals and
    ``"`` for identifiers, each escaped by doubling.

    Returns ``[]`` when there is no parenthesised group at all, which is the
    case for SQLite's implicit ``sqlite_autoindex_*`` entries (NULL ``sql``).
    """
    if not sql:
        return []

    keys: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    position = 0
    length = len(sql)

    while position < length:
        char = sql[position]

        # Inside a quoted literal or identifier nothing is structural: a
        # parenthesis or comma here is data, not syntax.
        if quote is not None:
            if depth:
                current.append(char)
            if char == quote:
                # SQL escapes a quote by doubling it, so a doubled quote
                # continues the region rather than ending it.
                if position + 1 < length and sql[position + 1] == quote:
                    if depth:
                        current.append(sql[position + 1])
                    position += 2
                    continue
                quote = None
            position += 1
            continue

        if char in ("'", '"'):
            quote = char
            if depth:
                current.append(char)
        elif char == "(":
            depth += 1
            if depth > 1:
                current.append(char)
        elif char == ")":
            depth -= 1
            if depth == 0:
                keys.append("".join(current).strip())
                return keys
            current.append(char)
        elif char == "," and depth == 1:
            keys.append("".join(current).strip())
            current = []
        elif depth:
            current.append(char)
        position += 1

    # No balanced group closed - an unterminated quote or unbalanced parens.
    return []
    return keys


def make_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # Alembic uses configparser.BasicInterpolation; escape any literal '%' in
    # the URL so it isn't read as a config-interpolation token.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    cfg.attributes["database_url"] = url
    return cfg
