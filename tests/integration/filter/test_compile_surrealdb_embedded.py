"""Embedded-SurrealDB integration test for the recall-filter compiler.

The unit corpus (``tests/unit/filter/test_compile_surrealdb.py``) pins the
*emitted string* shape; this test pins the *row-set semantics* by running the
compiled predicate against a real embedded SurrealDB (``memory://``). It is the
cheapest guard against the negation/totality confusion recurring: SurrealQL's
NONE-boolean algebra means the compiler emits NO ``coalesce`` wrapper, and the
only way to prove that is total (and that the metadata type-gate actually
excludes wrong-typed values) is to ask the engine for the real row-set.

Each test seeds a few ``temporal_chunk`` rows, compiles a wire filter through the
real :func:`~khora.filter.compilers.surrealdb.compile_surrealdb` (using the live
recall path's :class:`CompileContext`), splices the predicate into a
``SELECT ... WHERE`` exactly as the skeleton engine does, and asserts on the
``content`` values that come back.

The ``surrealdb`` SDK is an optional dependency, so this module self-skips when
it is absent (``pytest.importorskip``) — it is NOT a hard dependency of the main
test job and adds no Docker requirement (embedded ``memory://`` runs in-process).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.filter import RecallFilter  # noqa: E402
from khora.filter.ast import parse_to_ast  # noqa: E402
from khora.filter.compilers.python import compile_python  # noqa: E402
from khora.filter.compilers.surrealdb import compile_surrealdb  # noqa: E402
from khora.filter.context import CompileContext, RecallFilterUnsupportedError  # noqa: E402
from khora.storage.backends.surrealdb._helpers import _rid  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.temporal import TemporalFilter  # noqa: E402
from khora.storage.temporal.surrealdb import SurrealDBTemporalStore  # noqa: E402

pytestmark = pytest.mark.integration


# The live recall path's context: the two BACKED system keys (occurred_at /
# created_at) map to their bare columns, the ``metadata`` root → physical
# ``metadata_`` column. The eight unbacked keys are absent from the mapping (rejected).
_CTX = CompileContext(
    backend_target="temporal_chunk",
    field_mapping={"occurred_at": "occurred_at", "created_at": "created_at", "metadata": "metadata_"},
)

# A trimmed temporal_chunk shape — only the columns these row-set assertions
# touch. Keeping it minimal (no namespace/document record links, no HNSW index)
# avoids needing the full skeleton schema while exercising the exact column kinds
# the compiler emits against: bare string/datetime system columns and the
# FLEXIBLE ``metadata_`` object.
_SCHEMA = """
DEFINE TABLE IF NOT EXISTS temporal_chunk SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS content ON temporal_chunk TYPE string;
DEFINE FIELD IF NOT EXISTS occurred_at ON temporal_chunk TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS metadata_ ON temporal_chunk FLEXIBLE TYPE option<object>;
"""


def _seed_rows() -> list[dict]:
    """Rows that exercise absent-key, explicit-null, matching, non-matching, and
    wrong-typed metadata values, plus a deeply-nested object path.

    ``content`` is the row's stable label — every assertion compares the returned
    ``content`` set.
    """
    return [
        # ``score`` key entirely absent from the object.
        {"id": _rid("temporal_chunk", uuid4()), "content": "absent", "metadata_": {}},
        # ``score`` present as an explicit JSON null.
        {"id": _rid("temporal_chunk", uuid4()), "content": "null", "metadata_": {"score": None}},
        # numeric ``score`` that satisfies ``> 5``.
        {"id": _rid("temporal_chunk", uuid4()), "content": "ten", "metadata_": {"score": 10}},
        # numeric ``score`` that does NOT satisfy ``> 5``.
        {"id": _rid("temporal_chunk", uuid4()), "content": "three", "metadata_": {"score": 3}},
        # ``score`` present but the WRONG type (string) — must be gated out of a
        # numeric range without erroring.
        {"id": _rid("temporal_chunk", uuid4()), "content": "mismatch", "metadata_": {"score": "high"}},
        # a deeply-nested object path for the dot-descent assertion.
        {"id": _rid("temporal_chunk", uuid4()), "content": "deep", "metadata_": {"a": {"b": {"c": "x"}}}},
    ]


@pytest.fixture
async def conn() -> AsyncIterator[SurrealDBConnection]:
    """An embedded (in-memory) SurrealDB connection seeded with the test rows.

    ``memory://`` runs in-process — no server, no Docker. Each test gets its own
    fresh database. The connection's auto schema-init runs the full khora schema;
    we add the trimmed temporal_chunk table on top (idempotent).
    """
    connection = SurrealDBConnection(mode="memory")
    await connection.connect()
    try:
        await connection.execute(_SCHEMA)
        await connection.execute("INSERT INTO temporal_chunk $records", {"records": _seed_rows()})
        yield connection
    finally:
        await connection.disconnect()


async def _matching_content(connection: SurrealDBConnection, wire: dict) -> list[str]:
    """Compile ``wire`` through the real compiler, run it, return sorted contents.

    The predicate + binds are spliced into the ``WHERE`` exactly as
    ``SurrealDBTemporalStore._search_inner`` does — this exercises the live
    compile path, not a hand-written predicate.
    """
    ast = parse_to_ast(RecallFilter.model_validate(wire))
    compiled = compile_surrealdb(ast, _CTX)
    rows = await connection.query(
        f"SELECT content FROM temporal_chunk WHERE {compiled.predicate}",  # noqa: S608 - predicate is compiler-emitted, values bind
        compiled.params,
    )
    return sorted(row["content"] for row in rows)


async def _legacy_matching_content(connection: SurrealDBConnection, additional: dict) -> list[str]:
    """Run a legacy ``TemporalFilter.additional`` predicate, return sorted contents.

    Builds the WHERE through ``_build_filter_clauses`` (the legacy channel, which
    routes each ``additional`` key through the recall-filter compiler's guarded
    builder), drops the namespace-scoping clause/binds, and queries — proving the
    legacy path's row-set semantics, not just the emitted string.
    """
    clauses, binds = SurrealDBTemporalStore._build_filter_clauses(
        uuid4(),
        TemporalFilter(additional=additional),
    )
    legacy_clauses = [c for c in clauses if "namespace" not in c]
    legacy_binds = {k: v for k, v in binds.items() if k.startswith("af_")}
    where = " AND ".join(legacy_clauses)
    rows = await connection.query(
        f"SELECT content FROM temporal_chunk WHERE {where}",  # noqa: S608 - clauses are compiler-emitted, values bind
        legacy_binds,
    )
    return sorted(row["content"] for row in rows)


# ===========================================================================
# (1) Nested dot-path descent returns the right row.
# ===========================================================================


async def test_nested_dot_path_descent_matches_only_the_nested_row(conn: SurrealDBConnection) -> None:
    # metadata.a.b.c descends to metadata_.a.b.c — only the "deep" row carries
    # that nested value. Proves the path is NOT collapsed/mangled into a single
    # key that would match nothing (or the wrong row).
    assert await _matching_content(conn, {"metadata.a.b.c": "x"}) == ["deep"]


# ===========================================================================
# (2) A type-mismatched value is EXCLUDED by a numeric range (the spike's gap).
# ===========================================================================


async def test_numeric_range_gate_excludes_wrong_typed_value(conn: SurrealDBConnection) -> None:
    # ``metadata.score > 5`` is type-gated: only the numeric "ten" row qualifies.
    # The string "mismatch" row, the explicit-null row, and the absent-key row are
    # all excluded by the ``type::is::number`` gate (never erroring on the wrong
    # type), and "three" (3 > 5 is false) is excluded by the compare. This is the
    # exact behaviour the original spike could not achieve.
    assert await _matching_content(conn, {"metadata.score": {"$gt": 5}}) == ["ten"]


# ===========================================================================
# (3) $not over a metadata range KEEPS absent + null rows, EXCLUDES the match.
# ===========================================================================


async def test_not_over_range_keeps_absent_and_null_excludes_match(conn: SurrealDBConnection) -> None:
    # ``$not(metadata.score > 5)`` flips the total gated leaf. SurrealQL's
    # NONE-boolean algebra (no coalesce) means the negation admits the absent-key
    # row, the explicit-null row, the wrong-typed row, and the non-matching "three"
    # row — and EXCLUDES the one row that matched the inner range ("ten"). A
    # coalesce-style or NULL-propagating negation would wrongly drop the absent /
    # null rows; the only row that must be gone is "ten".
    kept = await _matching_content(conn, {"$not": {"metadata.score": {"$gt": 5}}})
    assert "ten" not in kept
    assert set(kept) == {"absent", "null", "mismatch", "three", "deep"}


# ===========================================================================
# (4) An unbacked system key FAILS LOUD — it never reaches the database.
# ===========================================================================
#
# ``source_url`` (one of the eight denormalized document keys) is NOT a column on
# the ``temporal_chunk`` table. Pre-fix the compiler emitted a bare ``source_url``
# field ref, which read NONE at query time and — via SurrealQL's total-false
# absent-compare — silently dropped every row (an $eq) or kept every row (a $ne),
# quietly returning a WRONG row-set instead of the intended filter. The compiler now
# refuses to emit that predicate: ``compile_surrealdb`` raises
# ``RecallFilterUnsupportedError`` BEFORE any query runs, so the row-set bug is
# impossible. These assert the raise on the same live compile path the row-set tests
# use (no SELECT is ever issued).


async def test_unbacked_system_key_eq_raises(conn: SurrealDBConnection) -> None:
    # The $eq form raised — the predicate that pre-fix silently dropped every row.
    ast = parse_to_ast(RecallFilter.model_validate({"source_url": "anything"}))
    with pytest.raises(RecallFilterUnsupportedError, match="does not back") as exc:
        compile_surrealdb(ast, _CTX)
    # Load-bearing invariant: the gate named the offending key (robust to re-wording).
    assert "source_url" in str(exc.value)


async def test_unbacked_system_key_ne_raises(conn: SurrealDBConnection) -> None:
    # The $ne form raised too — the predicate that pre-fix silently kept every row.
    # The rejection is op-independent: an unbacked column is never queryable.
    ast = parse_to_ast(RecallFilter.model_validate({"source_url": {"$ne": "anything"}}))
    with pytest.raises(RecallFilterUnsupportedError, match="does not back") as exc:
        compile_surrealdb(ast, _CTX)
    assert "source_url" in str(exc.value)


# ===========================================================================
# (5) Legacy ``additional`` path — a range op EXCLUDES a wrong-typed value.
# ===========================================================================
#
# The legacy ``TemporalFilter.additional`` channel routes every key through the
# same guarded compiler builder, so the type-gate that excludes wrong-typed
# values must hold here too. This is the row-set proof for the legacy path the
# behavioral unit tests (tests/unit/engines/.../test_surrealdb_legacy_additional.py)
# pin at the emitted-string level — it guards the eq-path regression that slipped
# through for lack of any test on this channel.


async def test_legacy_additional_range_excludes_wrong_typed_value(conn: SurrealDBConnection) -> None:
    # Legacy ``{"score": {"gt": 5}}`` is type-gated: only the numeric "ten" row
    # qualifies. The string "mismatch" row is gated out (not lexicographically
    # compared, not an error), and the absent-key / non-matching "three" rows fall
    # away — matching the deterministic-filter path's exclusion semantics.
    assert await _legacy_matching_content(conn, {"score": {"gt": 5}}) == ["ten"]


async def test_legacy_additional_eq_matches_only_exact_value(conn: SurrealDBConnection) -> None:
    # The legacy eq path (the one the injection regression touched) binds its value
    # and matches only the exact row — the wrong-typed "mismatch" row does not
    # coincidentally match a bound scalar.
    assert await _legacy_matching_content(conn, {"score": 10}) == ["ten"]


# ===========================================================================
# (6) Array-valued metadata fields + cross-backend parity with the oracle.
# ===========================================================================
#
# The canonical connector metadata shape is ``tags: list[str]``, so a scalar
# ``$eq`` / ``$in`` must match an ARRAY field that *contains* the value, not only
# a scalar field equal to it (SurrealQL ``=`` / ``INSIDE`` are scalar). These
# tests pin the array-aware row-sets AND assert they equal the ``compile_python``
# oracle's row-set for the same corpus — the parity check that keeps the five
# compilers from drifting (and that would have caught the scalar-only bug).

# (content, metadata) corpus exercising scalar/array/missing/null/wrong-type plus
# the substring traps (``CONTAINS`` does substring on a string operand, so a
# scalar ``"xyz"`` and an array element ``"xyz"`` must NOT match operand ``"x"``).
_TAGS_CORPUS: list[tuple[str, dict]] = [
    ("scalar", {"tags": "x"}),
    ("scalar_other", {"tags": "z"}),
    ("scalar_super", {"tags": "xyz"}),  # substring trap
    ("array", {"tags": ["x", "y"]}),
    ("array_other", {"tags": ["z"]}),
    ("array_super", {"tags": ["xyz"]}),  # element-substring trap
    ("missing", {"foo": 1}),
    ("null", {"tags": None}),
    ("num", {"tags": 5}),
]

# The oracle reads ``record["metadata"]`` and resolves system keys by identity,
# so it runs under a default context (no ``metadata`` → ``metadata_`` remap).
_ORACLE_CTX = CompileContext(backend_target="temporal_chunk")


@pytest.fixture
async def tags_conn() -> AsyncIterator[SurrealDBConnection]:
    """An embedded connection seeded with the array-valued ``tags`` corpus."""
    connection = SurrealDBConnection(mode="memory")
    await connection.connect()
    try:
        await connection.execute(_SCHEMA)
        rows = [
            {"id": _rid("temporal_chunk", uuid4()), "content": content, "metadata_": md} for content, md in _TAGS_CORPUS
        ]
        await connection.execute("INSERT INTO temporal_chunk $records", {"records": rows})
        yield connection
    finally:
        await connection.disconnect()


def _oracle_content(wire: dict) -> list[str]:
    """Row-set the ``compile_python`` oracle accepts over ``_TAGS_CORPUS``."""
    predicate = compile_python(parse_to_ast(RecallFilter.model_validate(wire)), _ORACLE_CTX).predicate
    return sorted(content for content, md in _TAGS_CORPUS if predicate({"metadata": md}))


async def test_scalar_eq_matches_scalar_and_array_fields(tags_conn: SurrealDBConnection) -> None:
    # A scalar operand matches BOTH the scalar field equal to it AND the array
    # field containing it — but NOT the substring traps (scalar "xyz" / element
    # "xyz") which a naive string CONTAINS would wrongly include.
    assert await _matching_content(tags_conn, {"metadata.tags": "x"}) == ["array", "scalar"]


async def test_scalar_in_is_contains_any_over_scalar_and_array(tags_conn: SurrealDBConnection) -> None:
    # $in is contains-any: scalar membership ("scalar"=x, "scalar_other"=z) plus
    # array share-any (["x","y"], ["z"]). Substring traps stay excluded.
    assert await _matching_content(tags_conn, {"metadata.tags": {"$in": ["x", "z"]}}) == sorted(
        ["scalar", "scalar_other", "array", "array_other"]
    )


@pytest.mark.parametrize(
    "wire",
    [
        {"metadata.tags": "x"},  # scalar $eq vs array field
        {"metadata.tags": {"$in": ["x", "z"]}},  # contains-any
        {"metadata.tags": {"$ne": "x"}},  # negated containment (admits absent/wrong-type)
        {"metadata.tags": {"$nin": ["x", "z"]}},  # negated contains-any
    ],
)
async def test_parity_with_python_oracle(tags_conn: SurrealDBConnection, wire: dict) -> None:
    # The SurrealDB compiler must return the SAME rows as the compile_python
    # oracle for the same corpus — the cross-backend parity guard. A divergence
    # (e.g. scalar-only $eq missing array fields) fails here immediately. Scoped to
    # the array-aware equality / membership ops: ``$exists`` is intentionally
    # excluded because SurrealDB drops an explicit-null field inside a FLEXIBLE
    # object on write (so present-null differs from the in-memory oracle) — a
    # storage representation difference, not a compiler divergence.
    assert await _matching_content(tags_conn, wire) == _oracle_content(wire)
