"""Unit tests for the four ``documents``-tier recall-filter compile contexts.

``@internal``. Each relational store module builds the
:class:`~khora.filter.context.CompileContext` that the document-enumeration path
hands to its backend compiler. The context is the ONLY place the physical
``documents`` schema is declared, so these tests pin what a compiler emits when
it is driven by that context: the physical table qualifier, the physical metadata
column, the pushdown/residual split, and the two known gaps a caller must defend
against.

Truth table these tests lock (verified against the four shipped builders):

===============  ================  ==================  ==================
context          backend_target    metadata column     compiler
===============  ================  ==================  ==================
postgresql       ``documents``     ``metadata``        ``compile_postgres``
sqlite (raw)     ``documents``     ``metadata_``       ``compile_lance``
sqlite_lance     ``documents``     ``metadata``        ``compile_lance``
surrealdb        ``document``      ``metadata_``       ``compile_surrealdb``
===============  ================  ==================  ==================

**Why every metadata assertion goes through a delimited matcher.** ``metadata``
is a PREFIX of ``metadata_``, so the obvious ``"documents.metadata" in sql``
passes against BOTH spellings — it would stay green against exactly the defect
this module exists to catch (a context wired to the other backend's column), and
its mirror image ``"metadata" not in sql`` is unsatisfiable where ``metadata_``
is the correct spelling. Every column assertion therefore uses
:func:`_emits_column`, whose trailing ``\\b`` refuses to match a longer
identifier, and every test asserts the OPPOSITE spelling is absent through the
same matcher. :func:`test_metadata_column_matcher_rejects_the_swapped_mapping`
is the standing proof that the matcher discriminates: it runs the shipped
assertion against a deliberately swapped mapping and requires it to fail there.

**Why the SQLite metadata tests build a local context.** Both SQLite builders
derive ``sqlite_json1`` from a runtime probe of the host's ``sqlite3`` build.
``compile_lance`` gates ALL metadata pushdown on that flag, so on a build without
JSON1 the shipped context emits no metadata column at all and a column assertion
against it would fail for a reason that has nothing to do with the mapping. The
column tests therefore build a local ``sqlite_json1=True`` context carrying the
shipped ``field_mapping``; the probe-driven behaviour of the shipped context is
pinned separately, in a form that is self-consistent under either probe answer.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.elements import ColumnElement

from khora.filter import RecallFilter
from khora.filter.ast import FilterNode, parse_to_ast
from khora.filter.compilers.lance import compile_lance
from khora.filter.compilers.postgres import compile_postgres
from khora.filter.compilers.surrealdb import compile_surrealdb
from khora.filter.context import CompileContext, SchemaCapabilities
from khora.filter.model import SYSTEM_KEYS
from khora.filter.registry import CompiledFilter
from khora.storage.backends.postgresql import _documents_compile_context as postgres_context
from khora.storage.backends.sqlite import _documents_compile_context as sqlite_context
from khora.storage.backends.sqlite import _sqlite_has_json1 as sqlite_has_json1
from khora.storage.backends.sqlite_lance.relational import _documents_compile_context as sqlite_lance_context
from khora.storage.backends.sqlite_lance.relational import _sqlite_has_json1 as sqlite_lance_has_json1
from khora.storage.backends.surrealdb.relational import _documents_compile_context as surrealdb_context

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

_DT = datetime(2026, 1, 31, 12, 30, tzinfo=UTC)

# The nine system keys a ``documents`` row backs with a real column, restated
# here independently of the store modules' own ``_BACKED_SYSTEM_KEYS`` — a test
# that imported the constant under test would assert nothing.
_BACKED_KEYS: frozenset[str] = frozenset(
    {
        "created_at",
        "source_timestamp",
        "source_type",
        "source_name",
        "source_url",
        "external_id",
        "content_type",
        "source",
        "title",
    }
)


def _ast(wire: dict) -> FilterNode:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


def _norm(sql: str) -> str:
    """Collapse whitespace and lowercase for resilient matching."""
    return " ".join(sql.split()).lower()


def _postgres_sql(wire: dict, ctx: CompileContext) -> str:
    """Compile with ``compile_postgres`` and render the predicate as inline SQL."""
    predicate = compile_postgres(_ast(wire), ctx).predicate
    assert isinstance(predicate, ColumnElement), f"not a SQLAlchemy element: {type(predicate)!r}"
    return _norm(str(predicate.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})))


def _lance_sql(wire: dict, ctx: CompileContext) -> str:
    return _norm(compile_lance(_ast(wire), ctx).predicate)


def _surrealdb_sql(wire: dict, ctx: CompileContext) -> str:
    return _norm(compile_surrealdb(_ast(wire), ctx).predicate)


def _emits_column(sql: str, column: str) -> bool:
    """Whether ``sql`` references exactly ``column`` — never a longer identifier.

    The trailing ``\\b`` is the whole point: ``metadata`` is a prefix of
    ``metadata_``, so a plain substring test cannot tell the two physical columns
    apart and would pass against a context wired to the wrong one. ``_`` is a
    word character, so ``metadata\\b`` does not match inside ``metadata_``.
    """
    return re.search(rf"\b{re.escape(column)}\b", sql) is not None


def _json1_variant(ctx: CompileContext) -> CompileContext:
    """The same context with JSON1 forced on, so metadata pushdown is exercised.

    The shipped SQLite contexts derive ``sqlite_json1`` from a runtime probe.
    Column-mapping tests must not depend on the host's SQLite build, so they run
    against this local variant, which carries the shipped ``field_mapping``
    verbatim and only pins the capability flag.
    """
    return CompileContext(
        backend_target=ctx.backend_target,
        table_alias=ctx.table_alias,
        param_namespace=ctx.param_namespace,
        field_mapping=ctx.field_mapping,
        schema_capabilities=SchemaCapabilities(sqlite_json1=True),
        on_unsupported=ctx.on_unsupported,
    )


# Every context builder, with its expected physical target/metadata column, and
# the SQL-rendering helper for the compiler it is registered against.
_ALL_CONTEXTS: tuple[tuple[str, Callable[[], CompileContext], str, str], ...] = (
    ("postgresql", postgres_context, "documents", "metadata"),
    ("sqlite", sqlite_context, "documents", "metadata_"),
    ("sqlite_lance", sqlite_lance_context, "documents", "metadata"),
    ("surrealdb", surrealdb_context, "document", "metadata_"),
)

# The two ``compile_lance``-driven contexts, which share every behaviour that
# depends on the JSON1 capability flag.
_SQLITE_CONTEXTS: tuple[tuple[str, Callable[[], CompileContext], str], ...] = (
    ("sqlite", sqlite_context, "metadata_"),
    ("sqlite_lance", sqlite_lance_context, "metadata"),
)


# ===========================================================================
# Context shape.
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "builder", "target", "metadata_column"),
    _ALL_CONTEXTS,
    ids=[case[0] for case in _ALL_CONTEXTS],
)
def test_context_declares_the_physical_schema(
    name: str,
    builder: Callable[[], CompileContext],
    target: str,
    metadata_column: str,
) -> None:
    ctx = builder()
    # ``backend_target`` is the PHYSICAL table name, which is the singular
    # ``document`` on SurrealDB even though the registry key is plural.
    assert ctx.backend_target == target
    # No documents read path aliases the table, so the qualifier comes from
    # ``backend_target`` alone.
    assert ctx.table_alias is None
    assert ctx.param_namespace == "f"
    # A document enumeration always has an in-memory post-filter available, so an
    # unpushable leaf is left unconsumed rather than raising.
    assert ctx.on_unsupported == "split"

    mapping = ctx.field_mapping
    assert mapping is not None
    # Exactly the nine backed system keys plus the ``metadata`` root — the
    # ``metadata`` entry must be present even where it is identity, because
    # ``compile_postgres`` resolves it eagerly.
    assert set(mapping) == set(_BACKED_KEYS) | {"metadata"}
    assert all(mapping[key] == key for key in _BACKED_KEYS)
    assert mapping["metadata"] == metadata_column


@pytest.mark.parametrize(
    ("name", "builder"),
    [(case[0], case[1]) for case in _ALL_CONTEXTS],
    ids=[case[0] for case in _ALL_CONTEXTS],
)
def test_occurred_at_is_absent_from_every_documents_mapping(
    name: str,
    builder: Callable[[], CompileContext],
) -> None:
    mapping = builder().field_mapping
    assert mapping is not None
    assert "occurred_at" not in mapping
    # The omission is only meaningful because ``occurred_at`` IS a filterable
    # system key: it can reach a compiler as an AST leaf, it just has no
    # ``documents`` column behind it.
    assert "occurred_at" in SYSTEM_KEYS
    assert set(_BACKED_KEYS) | {"occurred_at"} == set(SYSTEM_KEYS)


def test_only_the_sqlite_contexts_declare_a_capability() -> None:
    # Neither ``compile_postgres`` nor ``compile_surrealdb`` reads
    # ``schema_capabilities``, so their contexts leave it at the conservative
    # all-False default.
    assert postgres_context().schema_capabilities == SchemaCapabilities.DEFAULTS
    assert surrealdb_context().schema_capabilities == SchemaCapabilities.DEFAULTS

    # The two ``compile_lance`` contexts declare exactly one capability —
    # ``sqlite_json1``, whose value comes from a runtime probe of the host's
    # SQLite build and is therefore asserted as a bool, not as a fixed answer.
    for _name, builder, _column in _SQLITE_CONTEXTS:
        capabilities = builder().schema_capabilities
        assert isinstance(capabilities.sqlite_json1, bool)
        assert not capabilities.jsonb_path_query
        assert not capabilities.full_text
        assert not capabilities.native_map_metadata


# ===========================================================================
# System-key leaves — the physical table qualifier.
# ===========================================================================


def test_postgres_system_key_is_qualified_by_the_documents_table() -> None:
    sql = _postgres_sql({"source_type": "email"}, postgres_context())
    assert _emits_column(sql, "documents.source_type")


@pytest.mark.parametrize(
    ("name", "builder", "metadata_column"),
    _SQLITE_CONTEXTS,
    ids=[case[0] for case in _SQLITE_CONTEXTS],
)
def test_sqlite_system_key_is_qualified_by_the_documents_table(
    name: str,
    builder: Callable[[], CompileContext],
    metadata_column: str,
) -> None:
    sql = _lance_sql({"source_type": "email"}, builder())
    assert _emits_column(sql, "documents.source_type")
    # Positional binds, not named — ``compile_lance`` returns an ordered arg list.
    compiled = compile_lance(_ast({"source_type": "email"}), builder())
    assert compiled.params == {"args": ["email"]}


def test_surrealdb_system_key_is_a_bare_field() -> None:
    ctx = surrealdb_context()
    compiled = compile_surrealdb(_ast({"source_type": "email"}), ctx)
    sql = _norm(compiled.predicate)
    # SurrealQL selects from the table directly, so fields are unqualified —
    # a ``document.`` / ``documents.`` prefix here would be a syntax error.
    assert _emits_column(sql, "source_type")
    assert "document." not in sql
    assert "documents." not in sql
    # Named binds under the context's ``param_namespace``.
    assert compiled.params == {"f_0": "email"}
    assert "$f_0" in sql


# ===========================================================================
# Metadata leaves — the physical metadata column (delimited matcher).
# ===========================================================================


def test_postgres_metadata_leaf_addresses_the_metadata_column() -> None:
    sql = _postgres_sql({"metadata.tier": {"$eq": "gold"}}, postgres_context())
    assert _emits_column(sql, "documents.metadata")
    # ...and NOT the raw-SQLite spelling. Both spellings render distinguishably;
    # only a delimited matcher keeps that information.
    assert not _emits_column(sql, "documents.metadata_")


@pytest.mark.parametrize(
    ("name", "builder", "metadata_column"),
    _SQLITE_CONTEXTS,
    ids=[case[0] for case in _SQLITE_CONTEXTS],
)
def test_sqlite_metadata_leaf_addresses_its_own_metadata_column(
    name: str,
    builder: Callable[[], CompileContext],
    metadata_column: str,
) -> None:
    # Local JSON1 context: the shipped flag is probe-derived, and with JSON1 off
    # there is no metadata column in the fragment at all (pinned separately).
    ctx = _json1_variant(builder())
    sql = _lance_sql({"metadata.tier": {"$eq": "gold"}}, ctx)
    other = "metadata" if metadata_column == "metadata_" else "metadata_"
    assert _emits_column(sql, f"documents.{metadata_column}")
    assert not _emits_column(sql, f"documents.{other}")
    # The JSON path is bound, not interpolated.
    assert "json_each(" in sql


def test_surrealdb_metadata_leaf_addresses_the_metadata_underscore_field() -> None:
    sql = _surrealdb_sql({"metadata.tier": {"$eq": "gold"}}, surrealdb_context())
    assert _emits_column(sql, "metadata_.tier")
    assert not _emits_column(sql, "metadata.tier")


def test_metadata_column_matcher_rejects_the_swapped_mapping() -> None:
    """The column assertions above must FAIL against the defect they guard.

    ``metadata`` is a prefix of ``metadata_``, so a substring assertion stays
    green when a context is wired to the other backend's column. This test runs
    each backend's shipped assertion against a context whose metadata mapping has
    been deliberately swapped to the wrong spelling and requires it to reject —
    if this ever passes trivially, every metadata assertion above is worthless.
    """

    def _swapped(ctx: CompileContext, wrong_column: str) -> CompileContext:
        mapping = dict(ctx.field_mapping or {}) | {"metadata": wrong_column}
        return CompileContext(
            backend_target=ctx.backend_target,
            table_alias=ctx.table_alias,
            param_namespace=ctx.param_namespace,
            field_mapping=mapping,
            schema_capabilities=ctx.schema_capabilities,
            on_unsupported=ctx.on_unsupported,
        )

    wire = {"metadata.tier": {"$eq": "gold"}}

    # Postgres: shipped column is ``metadata``; swap it to ``metadata_``.
    sql = _postgres_sql(wire, _swapped(postgres_context(), "metadata_"))
    assert not _emits_column(sql, "documents.metadata")
    assert _emits_column(sql, "documents.metadata_")

    # Raw SQLite: shipped column is ``metadata_``; swap it to ``metadata``.
    sql = _lance_sql(wire, _swapped(_json1_variant(sqlite_context()), "metadata"))
    assert not _emits_column(sql, "documents.metadata_")
    assert _emits_column(sql, "documents.metadata")

    # sqlite_lance: shipped column is ``metadata``; swap it to ``metadata_``.
    sql = _lance_sql(wire, _swapped(_json1_variant(sqlite_lance_context()), "metadata_"))
    assert not _emits_column(sql, "documents.metadata")
    assert _emits_column(sql, "documents.metadata_")

    # SurrealDB: shipped field is ``metadata_``; swap it to ``metadata``.
    sql = _surrealdb_sql(wire, _swapped(surrealdb_context(), "metadata"))
    assert not _emits_column(sql, "metadata_.tier")
    assert _emits_column(sql, "metadata.tier")


# ===========================================================================
# Pushdown / residual split.
# ===========================================================================


# Both wire spellings of a ``$date`` metadata comparison that the validator
# accepts: ``$date`` as the operator key over a scalar operand (the form the
# existing compiler suite uses), and a ``$date``-wrapped OPERAND under a range
# operator. Both are inexpressible in SQLite and must split the same way. The
# third conceivable spelling — an operator nested INSIDE ``$date``
# (``{"$date": {"$gt": ...}}``) — is rejected by the validator, so it can never
# reach a compiler and is not a compiler concern.
_DATE_METADATA_WIRES: tuple[tuple[str, dict], ...] = (
    ("date-as-operator", {"metadata.due": {"$date": "2026-01-01T00:00:00Z"}}),
    ("date-wrapped-operand", {"metadata.due": {"$gt": {"$date": "2026-01-01T00:00:00Z"}}}),
)


@pytest.mark.parametrize(
    ("name", "builder", "metadata_column"),
    _SQLITE_CONTEXTS,
    ids=[case[0] for case in _SQLITE_CONTEXTS],
)
@pytest.mark.parametrize(
    ("wire_id", "date_wire"),
    _DATE_METADATA_WIRES,
    ids=[case[0] for case in _DATE_METADATA_WIRES],
)
def test_sqlite_date_metadata_compare_is_a_residual_not_a_raise(
    wire_id: str,
    date_wire: dict,
    name: str,
    builder: Callable[[], CompileContext],
    metadata_column: str,
) -> None:
    """A ``$date`` metadata compare is inexpressible in SQLite and must split.

    Run against a local JSON1 context so the contrast is real: with JSON1 on, a
    plain metadata ``$eq`` IS consumed, while the ``$date`` compare on the very
    same column is not. Against the shipped probe-derived context this pair could
    both be residuals for the unrelated reason that the host has no JSON1, which
    would prove nothing about ``$date`` — the control is what makes the residual
    assertion mean something.
    """
    ctx = _json1_variant(builder())

    # CONTROL — a plain metadata leaf on the same column DOES push down here.
    consumed_leaf = compile_lance(_ast({"metadata.tier": {"$eq": "gold"}}), ctx)
    assert consumed_leaf.consumed_keys == frozenset({"metadata.tier"})
    assert _emits_column(_norm(consumed_leaf.predicate), f"documents.{metadata_column}")

    # ``on_unsupported="split"`` — the unsupported leaf comes back as a residual,
    # never as a ``RecallFilterUnsupportedError``.
    residual = compile_lance(_ast(date_wire), ctx)
    assert residual.consumed_keys == frozenset()
    # Non-constraining placeholder: the caller's post-filter re-checks the leaf.
    assert residual.predicate.strip() == "(1)"
    assert residual.params == {"args": []}
    # No column is addressed at all — nothing to get wrong, and nothing pushed.
    assert not _emits_column(_norm(residual.predicate), f"documents.{metadata_column}")

    # Both leaves in one filter: the split is per-leaf, not per-filter.
    both = compile_lance(_ast({"metadata.tier": {"$eq": "gold"}} | date_wire), ctx)
    assert both.consumed_keys == frozenset({"metadata.tier"})
    assert _emits_column(_norm(both.predicate), f"documents.{metadata_column}")


@pytest.mark.parametrize(
    ("name", "builder", "metadata_column"),
    _SQLITE_CONTEXTS,
    ids=[case[0] for case in _SQLITE_CONTEXTS],
)
def test_shipped_sqlite_context_is_self_consistent_under_either_json1_answer(
    name: str,
    builder: Callable[[], CompileContext],
    metadata_column: str,
) -> None:
    """The shipped context's metadata behaviour matches its own probe result.

    ``sqlite_json1`` is probed from the host's SQLite build, so this asserts the
    IMPLICATION rather than a fixed outcome: JSON1 on ⇒ the metadata leaf pushes
    down to this backend's column; JSON1 off ⇒ every metadata leaf is a residual
    the caller post-filters. Green on a host either way; a mapping or gating
    regression still fails it.
    """
    ctx = builder()
    compiled: CompiledFilter[Any] = compile_lance(_ast({"metadata.tier": {"$eq": "gold"}}), ctx)
    sql = _norm(compiled.predicate)

    if ctx.schema_capabilities.sqlite_json1:
        assert compiled.consumed_keys == frozenset({"metadata.tier"})
        assert _emits_column(sql, f"documents.{metadata_column}")
    else:
        assert compiled.consumed_keys == frozenset()
        assert sql.strip() == "(1)"
        assert not _emits_column(sql, "documents.metadata")
        assert not _emits_column(sql, "documents.metadata_")

    # A system key pushes down either way — the capability gates metadata only.
    system = compile_lance(_ast({"source_type": "email"}), ctx)
    assert system.consumed_keys == frozenset({"source_type"})


# ===========================================================================
# The JSON1 probe — the only new runtime logic behind these contexts.
# ===========================================================================

# Both SQLite store modules carry their own copy of the probe (their schemas
# genuinely diverge, so they share no code); both are covered.
_JSON1_PROBES: tuple[tuple[str, Callable[[], bool]], ...] = (
    ("sqlite", sqlite_has_json1),
    ("sqlite_lance", sqlite_lance_has_json1),
)


def _direct_json1_probe() -> bool:
    """Ask this interpreter's ``sqlite3`` build about JSON1, independently."""
    conn = sqlite3.connect(":memory:")
    try:
        return conn.execute("SELECT json_valid('{}')").fetchone()[0] == 1
    except sqlite3.Error:
        return False
    finally:
        conn.close()


@pytest.mark.parametrize(("name", "probe"), _JSON1_PROBES, ids=[case[0] for case in _JSON1_PROBES])
def test_json1_probe_agrees_with_the_hosts_sqlite_build(name: str, probe: Callable[[], bool]) -> None:
    """The probe reports what this host's SQLite actually supports.

    Asserted against an independent probe rather than a hardcoded ``True``, so
    the test states the contract (the flag tracks the runtime) instead of the
    accident of which SQLite this host shipped. ``aiosqlite`` and SQLAlchemy's
    ``sqlite+aiosqlite`` both run on this same in-process library, which is what
    makes a process-wide answer correct.
    """
    result = probe()
    assert isinstance(result, bool)
    assert result is _direct_json1_probe()


@pytest.mark.parametrize(("name", "probe"), _JSON1_PROBES, ids=[case[0] for case in _JSON1_PROBES])
def test_json1_probe_is_stable_across_calls(name: str, probe: Callable[[], bool]) -> None:
    # Memoized: the answer cannot change within a process, and a context built
    # twice must not disagree with itself.
    assert probe() is probe()


@pytest.mark.parametrize(("name", "probe"), _JSON1_PROBES, ids=[case[0] for case in _JSON1_PROBES])
def test_json1_probe_fails_closed_when_the_query_errors(
    name: str,
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed probe reports NO JSON1 — the conservative answer.

    This is the branch that decides what happens on a SQLite build without the
    JSON1 functions, and it is unreachable on a host that has them, so it is
    driven here with a connection whose query raises. ``False`` is the only safe
    answer: under ``on_unsupported="split"`` it sends every metadata leaf to the
    caller's post-filter (less pushdown, same rows), whereas a wrong ``True``
    would emit ``json_extract`` against a build that cannot run it.

    Without this case the probe tests could not distinguish a real probe from a
    hardcoded ``True`` on a JSON1-enabled host.
    """

    class _FailingConnection:
        def execute(self, _sql: str) -> object:
            raise sqlite3.Error("no such function: json_valid")

        def close(self) -> None:
            return None

    monkeypatch.setattr(sqlite3, "connect", lambda *_args, **_kwargs: _FailingConnection())
    # The probe is memoized, so the real answer must be evicted before and after
    # this case or the fake result would leak into the rest of the session.
    probe.cache_clear()
    try:
        assert probe() is False
    finally:
        probe.cache_clear()

    monkeypatch.undo()
    # Back to the host's real answer for every later test in this process.
    assert probe() is _direct_json1_probe()


def test_both_sqlite_probes_reach_the_same_answer() -> None:
    # Two independent copies of the same probe against the same in-process
    # library — a divergence would mean one of them stopped probing.
    assert sqlite_has_json1() is sqlite_lance_has_json1()


@pytest.mark.parametrize(
    ("name", "builder", "metadata_column"),
    _SQLITE_CONTEXTS,
    ids=[case[0] for case in _SQLITE_CONTEXTS],
)
def test_context_capability_carries_the_probe_result(
    name: str,
    builder: Callable[[], CompileContext],
    metadata_column: str,
) -> None:
    # The probe is only meaningful because the context hands its answer to the
    # compiler — this is the wire between the two.
    assert builder().schema_capabilities.sqlite_json1 is _direct_json1_probe()


# ===========================================================================
# KNOWN GAPS — pinned as they behave today, NOT as they should behave.
# ===========================================================================


def test_occurred_at_is_defended_on_surrealdb() -> None:
    # ``compile_surrealdb`` treats the ``field_mapping`` key set as its
    # declared+pushable whitelist, so the undeclared ``occurred_at`` leaf never
    # reaches the query: it emits the non-constraining literal and is left out of
    # ``consumed_keys`` for the caller to post-filter. This is the one backend
    # where the context alone prevents the bad predicate.
    ctx = surrealdb_context()
    compiled = compile_surrealdb(_ast({"occurred_at": {"$gte": _DT}}), ctx)
    assert compiled.predicate.strip() == "(true)"
    assert compiled.consumed_keys == frozenset()
    assert compiled.params == {}


def test_occurred_at_is_not_defended_on_sql_backends() -> None:
    """KNOWN GAP — pins today's behaviour so the caller contract is written down.

    ``compile_postgres`` and ``compile_lance`` fall back to identity for a key
    that is absent from ``field_mapping``, so an ``occurred_at`` leaf compiles to
    a column that does not exist on ``documents`` AND is reported in
    ``consumed_keys`` — the caller is told the leaf was pushed down and will not
    post-filter it, while the statement itself fails at execute time. Nothing
    expressible in a compile context prevents this; until the compilers honour
    the key set as a whitelist, the enumeration caller MUST strip ``occurred_at``
    before compiling. These assertions encode the gap, not the desired behaviour.

    **This expectation INVERTS when the pushdown whitelist gate lands.**
    ``consumed_keys`` becomes ``frozenset()`` and no ``documents.occurred_at``
    column is emitted at all, matching what ``compile_surrealdb`` already does
    (see the test above) — that is the fix landing, not a regression. Update the
    assertions; do NOT restore green by declaring ``occurred_at`` in the
    ``field_mapping``, which would point the leaf at a column no ``documents``
    table has.
    """
    wire = {"occurred_at": {"$gte": _DT}}

    sql = _postgres_sql(wire, postgres_context())
    assert _emits_column(sql, "documents.occurred_at")
    assert compile_postgres(_ast(wire), postgres_context()).consumed_keys == frozenset({"occurred_at"})

    for _name, builder, _column in _SQLITE_CONTEXTS:
        compiled = compile_lance(_ast(wire), builder())
        assert _emits_column(_norm(compiled.predicate), "documents.occurred_at")
        assert compiled.consumed_keys == frozenset({"occurred_at"})


def test_surrealdb_placeholder_inverts_under_a_negated_or() -> None:
    """KNOWN GAP — an undeclared key under ``$not``/``$or`` matches nothing.

    The placeholder ``compile_surrealdb`` emits for an unsupported leaf is the
    literal ``true``, which is non-constraining only inside a positive
    conjunction. Under a negated OR it inverts: ``!(... OR true)`` is always
    false, so the query silently returns no rows while ``consumed_keys`` still
    reports the leaf as the caller's to post-filter — and post-filtering can only
    narrow, never recover the dropped rows. Pinned as it behaves today; the fix
    is a soundness gate in the compiler, not in this context.

    **This expectation INVERTS when that soundness gate lands.** A gated
    ``compile_surrealdb`` defers the whole ``OR`` node rather than letting a
    match-all placeholder invert, so ``or (true)`` disappears from the emitted
    predicate and ``title`` stops being consumed — that is the fix landing, not a
    regression. Update the assertions to the deferred shape; do NOT restore green
    by relaxing them back to accepting an inverted placeholder.
    """
    compiled = compile_surrealdb(
        _ast({"$not": {"$or": [{"title": "x"}, {"occurred_at": {"$gt": _DT}}]}}),
        surrealdb_context(),
    )
    sql = _norm(compiled.predicate)
    assert sql.startswith("!(")
    assert "or (true)" in sql
    assert compiled.consumed_keys == frozenset({"title"})


def test_sqlite_lance_datetime_bind_does_not_match_the_stored_format() -> None:
    """KNOWN GAP — the pushed-down bind format differs from the stored format.

    ``documents`` rows on this stack are written through SQLAlchemy's SQLite
    ``DATETIME`` type, which stores ``'2026-01-31 12:30:00.000000'`` (SPACE
    separator, no offset), while ``compile_lance`` binds a datetime operand as
    ``.isoformat()`` (``'T'`` separator, offset included) and relies on
    lexicographic ISO comparison. ``' '`` (0x20) sorts before ``'T'`` (0x54), so a
    pushed-down ``created_at`` / ``source_timestamp`` bound silently excludes rows
    from its own day. The bind value is pinned here so the mismatch is visible in
    a test rather than only in a docstring; the caller must strip the date-valued
    system keys until the bind format is fixed upstream.

    **This expectation INVERTS when the pushdown whitelist gate lands.**
    ``consumed_keys`` becomes ``frozenset()`` once ``compile_lance`` honours the
    ``field_mapping`` key set and the two date keys are dropped from the
    sqlite_lance mapping — that is the fix landing, not a regression. Update the
    assertion; do NOT restore green by re-declaring ``created_at`` in the
    mapping, which would reinstate the silent mismatch this test documents.
    """
    compiled = compile_lance(_ast({"created_at": {"$gte": _DT}}), sqlite_lance_context())
    assert compiled.params == {"args": ["2026-01-31T12:30:00+00:00"]}
    # Stored form for the same instant, which the bind above does NOT equal.
    assert "2026-01-31 12:30:00.000000" != compiled.params["args"][0]
    assert compiled.consumed_keys == frozenset({"created_at"})


# ===========================================================================
# Field-mapping typing sanity (cheap, catches a Mapping/str mix-up).
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "builder"),
    [(case[0], case[1]) for case in _ALL_CONTEXTS],
    ids=[case[0] for case in _ALL_CONTEXTS],
)
def test_field_mapping_is_a_plain_string_to_string_mapping(
    name: str,
    builder: Callable[[], CompileContext],
) -> None:
    mapping = builder().field_mapping
    assert isinstance(mapping, Mapping)
    assert all(isinstance(key, str) and isinstance(value, str) for key, value in mapping.items())
