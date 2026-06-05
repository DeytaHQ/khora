"""Postgres-backed row-set test for the recall-filter compiler (§4).

Where ``tests/unit/filter/test_compile_postgres.py`` asserts the *shape* of the emitted SQL, this
file asserts its *behavior* against a real PostgreSQL ``khora_chunks`` table seeded
with ADVERSARIAL rows. For each representative filter we compile the AST to a
SQLAlchemy predicate, run ``SELECT id FROM khora_chunks WHERE ns = :ns AND
<predicate>``, and assert the EXACT set of returned chunk ids. The hand-authored
expected set is the oracle — there is exactly one Postgres compiler, so the
behavior is fully determined and a divergence is a real bug.

Adversarial seed (one namespace, one row per "id" label below). Every row carries
a ``score`` and a ``tier`` metadata field plus a ``sent_at`` string, chosen to break
naive ``->>`` text comparisons:

* ``num10``   — metadata.score = 10  (number)
* ``num2``    — metadata.score = 2   (number; "10" < "2" as TEXT, 2 < 10 as number)
* ``numstr``  — metadata.score = "10" (numeric-looking STRING — must NOT count as a number)
* ``strabc``  — metadata.score = "abc" (non-numeric string)
* ``arr``     — metadata.score = [1, 2] (ARRAY value)
* ``missing`` — metadata has no ``score`` key at all
* ``sysnull`` — system column ``source_name`` IS NULL
* ``baddate`` — metadata.sent_at = "not-a-date" (malformed; khora_try_timestamptz → NULL)

Docker Compose (this repo's ``compose.yaml``, Postgres on port 5434) provides the
database; the module skips cleanly when Postgres is unreachable.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from khora.filter.compilers.postgres import compile_postgres
from khora.filter.context import CompileContext

# Hard import (NOT importorskip): the compiler is on the branch, so an import
# failure must be a LOUD test error — never a silent module skip. (This module
# still skips when Postgres is unreachable, via the pytestmark skipif below.)


# This repo's compose puts Postgres on 5434 (see compose.yaml). Honor an explicit
# override, else default to the compose port — never another project's container.
_DEFAULT_URL = "postgresql+asyncpg://khora:khora@localhost:5434/khora"


def _database_url() -> str:
    url = os.environ.get("KHORA_DATABASE_URL", _DEFAULT_URL)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def _pg_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(_database_url().replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _pg_reachable(), reason="PostgreSQL not reachable (run `make dev`)"),
]


# The compiler qualifies columns with ``ctx.backend_target`` (``_col`` derives
# ``<backend_target>.<col>`` from ctx alone), so a predicate built with
# ``backend_target="khora_chunks"`` references ``khora_chunks.<col>``. To run that
# predicate against real data WITHOUT touching any production ``khora_chunks``, we
# create a session-local ``TEMPORARY TABLE khora_chunks`` on a single dedicated
# connection: a temp table shadows any permanent same-named table for this session
# only (``pg_temp`` is first on the search_path), is invisible to every other
# session, and is dropped automatically when the connection closes. That satisfies
# the test-isolation rule (never reuse / clobber another project's or the real
# khora_chunks) while letting FROM and the compiled predicate agree on the name.
_TEST_TABLE = "khora_chunks"

_CREATE_TABLE = f"""
CREATE TEMPORARY TABLE {_TEST_TABLE} (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    occurred_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ,
    source_timestamp TIMESTAMPTZ,
    source_type VARCHAR(64),
    source_name VARCHAR(255),
    source_url TEXT,
    external_id VARCHAR(512),
    content_type VARCHAR(128),
    source TEXT,
    title TEXT,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
)
"""

# khora_try_timestamptz (migration 045) — the safe text→timestamptz cast the
# $date metadata path depends on. Created in pg_temp so it is session-local and
# never collides with (or shadows-then-orphans) a real installed function; the
# compiler calls it unqualified, which resolves via pg_temp first on search_path.
# Self-sufficient even on a DB that has not run the Alembic chain.
_CREATE_TRY_TS = """
CREATE FUNCTION pg_temp.khora_try_timestamptz(txt text)
RETURNS timestamptz
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
BEGIN
    RETURN txt::timestamptz;
EXCEPTION WHEN others THEN
    RETURN NULL;
END;
$$;
"""


# Row label → (system-column overrides, metadata dict). namespace_id is filled in
# at seed time. Unspecified system columns default to a benign non-null value so a
# row only differs on the dimension under test.
_ROWS: dict[str, dict] = {
    "num10": {"metadata": {"score": 10, "tier": "gold", "sent_at": "2026-01-10T00:00:00Z"}},
    "num2": {"metadata": {"score": 2, "tier": "silver", "sent_at": "2026-01-02T00:00:00Z"}},
    "numstr": {"metadata": {"score": "10", "tier": "bronze"}},
    "strabc": {"metadata": {"score": "abc", "tier": "gold"}},
    "arr": {"metadata": {"score": [1, 2], "tier": ["gold", "silver"]}},
    "missing": {"metadata": {"tier": "gold"}},
    "sysnull": {"system": {"source_name": None}, "metadata": {"score": 5, "tier": "gold"}},
    "baddate": {"metadata": {"score": 7, "tier": "silver", "sent_at": "not-a-date"}},
}

_ALL_IDS = set(_ROWS)


@pytest.fixture
async def seeded() -> AsyncIterator[tuple[AsyncConnection, UUID, dict[str, UUID]]]:
    """Create the session-local temp table + function, seed the adversarial rows.

    Everything happens on ONE connection because the ``khora_chunks`` temp table
    and the ``pg_temp`` cast function are connection-scoped; the same connection
    then runs the compiled predicates. The temp objects vanish when the
    connection closes, so there is no cross-test or cross-project leakage.
    """
    eng = create_async_engine(_database_url())
    ns = uuid4()
    ids: dict[str, UUID] = {label: uuid4() for label in _ROWS}
    try:
        async with eng.connect() as conn:
            await conn.execute(sa.text(_CREATE_TABLE))
            await conn.execute(sa.text(_CREATE_TRY_TS))
            for label, spec in _ROWS.items():
                system = spec.get("system", {})
                await conn.execute(
                    sa.text(
                        # _TEST_TABLE is a hardcoded module constant, not user input.
                        f"INSERT INTO {_TEST_TABLE} "  # noqa: S608
                        "(id, namespace_id, source_name, source_type, occurred_at, metadata) "
                        "VALUES (:id, :ns, :source_name, :source_type, :occurred_at, "
                        "CAST(:metadata AS jsonb))"
                    ),
                    {
                        "id": ids[label],
                        "ns": ns,
                        "source_name": system.get("source_name", "linear"),
                        "source_type": system.get("source_type", "slack"),
                        "occurred_at": system.get("occurred_at", "2026-01-05T00:00:00Z"),
                        "metadata": json.dumps(spec["metadata"]),
                    },
                )
            await conn.commit()
            yield conn, ns, ids
    finally:
        await eng.dispose()


async def _matching_labels(
    seeded: tuple[AsyncConnection, UUID, dict[str, UUID]],
    wire: dict,
) -> set[str]:
    """Compile ``wire`` and return the set of ROW LABELS whose rows match.

    Runs the compiled predicate against the seeded temp ``khora_chunks`` on the
    same connection. FROM is a lightweight ``sa.table("khora_chunks", ...)`` whose
    name matches the qualifier the compiler emits (``ctx.backend_target``), so the
    statement's table and the predicate's ``khora_chunks.<col>`` references agree.
    """
    conn, ns, ids = seeded
    id_to_label = {v: k for k, v in ids.items()}

    ast = parse_to_ast(RecallFilter.model_validate(wire))
    ctx = CompileContext(backend_target=_TEST_TABLE)
    predicate = compile_postgres(ast, ctx).predicate

    table = sa.table(_TEST_TABLE, sa.column("id"), sa.column("namespace_id"))
    stmt = sa.select(table.c.id).where(sa.and_(table.c.namespace_id == ns, predicate))
    result = await conn.execute(stmt)
    return {id_to_label[r[0]] for r in result.fetchall()}


# ===========================================================================
# Numeric range — excludes array, "abc", numeric STRING; orders numerically.
# ===========================================================================


async def test_numeric_gte_excludes_non_numbers_and_orders_numerically(seeded) -> None:
    # metadata.score >= 5 must include the numbers 10, 7, 5 — and EXCLUDE the
    # numeric-looking string "10" (it is text, not a number), the non-numeric
    # "abc", the array [1,2], and the row with no score key. Crucially num2
    # (score=2) is excluded: numeric ordering, not "10" < "2" text ordering.
    labels = await _matching_labels(seeded, {"metadata.score": {"$gte": 5}})
    assert labels == {"num10", "sysnull", "baddate"}


async def test_numeric_lt_orders_numerically_not_lexically(seeded) -> None:
    # metadata.score < 5 → only num2 (=2). If the compiler compared as TEXT,
    # "10" < "5" would be true and numstr would wrongly appear; the jsonb_typeof
    # gate + numeric cast keeps it out, and "10"-the-string is excluded anyway.
    labels = await _matching_labels(seeded, {"metadata.score": {"$lt": 5}})
    assert labels == {"num2"}


async def test_numeric_range_band(seeded) -> None:
    # 3 <= score <= 9 → only the numbers 5 (sysnull) and 7 (baddate).
    labels = await _matching_labels(seeded, {"metadata.score": {"$gte": 3, "$lte": 9}})
    assert labels == {"sysnull", "baddate"}


# ===========================================================================
# $ne / $nin include the missing/NULL rows (Rule 2).
# ===========================================================================


async def test_metadata_ne_includes_missing_key(seeded) -> None:
    # tier != "gold" compiles to NOT(metadata @> {"tier":"gold"}). Containment is
    # array-aware, so the array-tier row ("arr": ["gold","silver"]) DOES contain
    # "gold" → it is excluded by the $ne, exactly like the scalar-gold rows.
    # The $ne includes rows whose tier is a *different* scalar; there is no
    # missing-tier row here (every row has a tier), but the polarity is still
    # exercised — a hypothetical missing-tier row would be admitted because
    # containment is FALSE (not NULL) for an absent key, and NOT(FALSE) = TRUE.
    labels = await _matching_labels(seeded, {"metadata.tier": {"$ne": "gold"}})
    # @>{"tier":"gold"} is TRUE for num10/strabc/missing/sysnull (scalar gold) and
    # arr (array contains gold). NOT(...) keeps the rest:
    assert labels == {"num2", "numstr", "baddate"}


async def test_metadata_ne_admits_absent_key(seeded) -> None:
    # Direct Rule-2 proof: $ne on a key NO row carries ("nope" is absent
    # everywhere) must match EVERY row — NOT(metadata @> {"nope": "x"}) is
    # NOT(FALSE) = TRUE for an absent key, never NULL-dropped.
    labels = await _matching_labels(seeded, {"metadata.nope": {"$ne": "x"}})
    assert labels == _ALL_IDS


async def test_system_ne_includes_null_column(seeded) -> None:
    # source_name != "linear" must INCLUDE the row whose source_name IS NULL
    # (a missing value is "not equal"). All non-sysnull rows have "linear", so
    # only sysnull qualifies.
    labels = await _matching_labels(seeded, {"source_name": {"$ne": "linear"}})
    assert labels == {"sysnull"}


async def test_not_eq_admits_null_row_like_ne(seeded) -> None:
    # SEMANTICS over tokens (architect note): $not($eq x) is NULL-INCLUSIVE — it
    # must admit a row whose column IS NULL, exactly like $ne. The child equality
    # is total (`coalesce(col = x, false)`), so NOT(false) = TRUE flips the NULL
    # row IN. Proven by the row-set, not the rendered operator: $not($eq "linear")
    # returns the SAME set as {"$ne": "linear"} — only the NULL-valued row.
    not_eq = await _matching_labels(seeded, {"$not": {"source_name": "linear"}})
    ne = await _matching_labels(seeded, {"source_name": {"$ne": "linear"}})
    assert not_eq == {"sysnull"}
    assert not_eq == ne  # $not($eq) ≡ $ne at the row-set level (NULL-inclusion)


# ===========================================================================
# $in — array-contains membership.
# ===========================================================================


async def test_metadata_in_matches_scalar_and_array_membership(seeded) -> None:
    # tier $in ["silver"] → every row whose tier is the scalar "silver" (num2,
    # baddate) AND the array-tier row (arr contains "silver"). Containment
    # membership spans both the scalar and array shapes.
    labels = await _matching_labels(seeded, {"metadata.tier": {"$in": ["silver"]}})
    assert labels == {"num2", "baddate", "arr"}


async def test_metadata_in_multiple_values(seeded) -> None:
    labels = await _matching_labels(seeded, {"metadata.tier": {"$in": ["silver", "bronze"]}})
    # silver: num2, baddate, arr (contains silver). bronze: numstr.
    assert labels == {"num2", "baddate", "numstr", "arr"}


# ===========================================================================
# $date — excludes malformed via khora_try_timestamptz.
# ===========================================================================


async def test_metadata_date_range_excludes_malformed(seeded) -> None:
    # sent_at >= 2026-01-05 → num10 (2026-01-10). num2 (2026-01-02) is before.
    # baddate ("not-a-date") must be EXCLUDED — khora_try_timestamptz returns
    # NULL so the comparison is NULL (non-match) rather than erroring the query.
    # numstr/strabc/arr/missing/sysnull have no sent_at at all → excluded.
    labels = await _matching_labels(seeded, {"metadata.sent_at": {"$gte": {"$date": "2026-01-05T00:00:00Z"}}})
    assert labels == {"num10"}


async def test_metadata_date_eq_excludes_malformed(seeded) -> None:
    labels = await _matching_labels(seeded, {"metadata.sent_at": {"$date": "2026-01-02T00:00:00Z"}})
    assert labels == {"num2"}


# ===========================================================================
# $exists — key presence, true/false (Rule 4).
# ===========================================================================


async def test_metadata_exists_true(seeded) -> None:
    # score key present → all rows EXCEPT "missing".
    labels = await _matching_labels(seeded, {"metadata.score": {"$exists": True}})
    assert labels == _ALL_IDS - {"missing"}


async def test_metadata_exists_false(seeded) -> None:
    # score key absent → only "missing".
    labels = await _matching_labels(seeded, {"metadata.score": {"$exists": False}})
    assert labels == {"missing"}


async def test_metadata_exists_true_includes_null_valued_array_and_string(seeded) -> None:
    # $exists is KEY-PRESENCE (Rule 4): a present-but-array ("arr") and a
    # present-but-string ("strabc"/"numstr") value all count as existing — a ->>
    # value extraction would have mishandled the array/typed values.
    labels = await _matching_labels(seeded, {"metadata.score": {"$exists": True}})
    assert {"arr", "strabc", "numstr"} <= labels


# ===========================================================================
# Array containment scalar $eq spans scalar + array fields.
# ===========================================================================


async def test_metadata_scalar_eq_matches_array_membership(seeded) -> None:
    # tier == "gold" via @> containment matches the scalar-gold rows AND the
    # array-tier row that contains "gold".
    labels = await _matching_labels(seeded, {"metadata.tier": "gold"})
    # scalar gold: num10, strabc, missing, sysnull. array containing gold: arr.
    assert labels == {"num10", "strabc", "missing", "sysnull", "arr"}


# ===========================================================================
# Empty filter — match everything.
# ===========================================================================


async def test_empty_filter_matches_all_rows(seeded) -> None:
    labels = await _matching_labels(seeded, {})
    assert labels == _ALL_IDS


# ===========================================================================
# System $exists is a CONSTANT (a denormalized column is structurally always
# present), and NULL-ness on a system column is reached via {key: null}.
# ===========================================================================


async def test_system_exists_true_matches_all_rows(seeded) -> None:
    # $exists:true on a system column is a tautology — the column is always
    # present on the row, including the NULL-valued one (postgres.py:182-184).
    labels = await _matching_labels(seeded, {"source_name": {"$exists": True}})
    assert labels == _ALL_IDS


async def test_system_exists_false_matches_nothing(seeded) -> None:
    # ...and $exists:false matches NOTHING, even the NULL-valued row — a present
    # column is never "absent". This is the deliberate divergence from metadata
    # $exists; NULL-ness is asserted via {source_name: null} below.
    labels = await _matching_labels(seeded, {"source_name": {"$exists": False}})
    assert labels == set()


async def test_system_null_operand_finds_null_column(seeded) -> None:
    # {"source_name": null} is the active null-or-missing match → IS NULL. THIS is
    # how a NULL system column is found (not $exists:false). Only sysnull matches.
    labels = await _matching_labels(seeded, {"source_name": None})
    assert labels == {"sysnull"}


async def test_system_eq_excludes_null_column(seeded) -> None:
    # A positive scalar $eq on a system column excludes the NULL-valued row
    # (NULL = 'linear' is NULL, not TRUE) — the complement of the $ne polarity.
    labels = await _matching_labels(seeded, {"source_name": "linear"})
    assert labels == _ALL_IDS - {"sysnull"}
