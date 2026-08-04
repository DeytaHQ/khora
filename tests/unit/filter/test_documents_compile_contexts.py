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

import dataclasses
import re
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.elements import ColumnElement

from khora.db.models import DocumentModel
from khora.filter import RecallFilter
from khora.filter.ast import FilterNode, parse_to_ast
from khora.filter.compilers.lance import compile_lance
from khora.filter.compilers.postgres import compile_postgres
from khora.filter.compilers.python import compile_python
from khora.filter.compilers.surrealdb import compile_surrealdb
from khora.filter.context import CompileContext, SchemaCapabilities
from khora.filter.execute import build_compile_context
from khora.filter.model import SYSTEM_KEYS
from khora.filter.registry import CompiledFilter
from khora.storage.backends._sqlite_capabilities import sqlite_has_json1
from khora.storage.backends.postgresql import _documents_compile_context as postgres_context
from khora.storage.backends.sqlite import _SCHEMA_SQL
from khora.storage.backends.sqlite import _documents_compile_context as sqlite_context
from khora.storage.backends.sqlite_lance.relational import _documents_compile_context as sqlite_lance_context
from khora.storage.backends.surrealdb.relational import _documents_compile_context as surrealdb_context

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

_DT = datetime(2026, 1, 31, 12, 30, tzinfo=UTC)

# The nine system keys a ``documents`` row backs with a real column, restated
# here independently of the store modules' own ``_PUSHABLE_SYSTEM_KEYS`` — a test
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

# Backed by a column but deliberately NOT declared, per context. Since the
# compilers honour a non-``None`` ``field_mapping`` key set as the pushdown
# whitelist, an omission is now enforced rather than advisory — so "which keys
# are withheld" is a behavioural claim, not bookkeeping.
#
# Both SQLite-backed stores withhold the two date-valued keys: they write
# timestamps as strings whose format does not order lexicographically against
# the ISO-8601 bind ``compile_lance`` emits, so a pushed comparison silently
# returns wrong rows (pinned below, and proved against a real table in
# :func:`test_sqlite_lance_date_predicate_survives_the_day_boundary`). Postgres
# and SurrealDB compare real timestamp values and withhold nothing.
_WITHHELD_KEYS: Mapping[str, frozenset[str]] = {
    "postgresql": frozenset(),
    "sqlite": frozenset({"created_at", "source_timestamp"}),
    "sqlite_lance": frozenset({"created_at", "source_timestamp"}),
    "surrealdb": frozenset(),
}


def _declared_keys(name: str) -> frozenset[str]:
    """The system keys ``name``'s context declares — backed minus withheld."""
    return _BACKED_KEYS - _WITHHELD_KEYS[name]


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

    ``dataclasses.replace`` rather than a field-by-field rebuild: a new field on
    :class:`CompileContext` would otherwise be silently dropped from every test
    routed through this helper, with nothing to flag it.
    """
    return dataclasses.replace(ctx, schema_capabilities=SchemaCapabilities(sqlite_json1=True))


# Every context builder, with its expected physical target/metadata column, and
# the SQL-rendering helper for the compiler it is registered against.
_ALL_CONTEXTS: tuple[tuple[str, Callable[[], CompileContext], str, str], ...] = (
    ("postgresql", postgres_context, "documents", "metadata"),
    ("sqlite", sqlite_context, "documents", "metadata_"),
    ("sqlite_lance", sqlite_lance_context, "documents", "metadata"),
    ("surrealdb", surrealdb_context, "document", "metadata_"),
)

# The enumeration contract specifies ``"split"`` everywhere, and every context
# now ships it. SurrealDB previously shipped ``"raise"`` as an interim posture,
# because ``compile_surrealdb``'s unsupported-leaf placeholder inverted under
# ``$not``/``$or``; the all-or-nothing gate in
# :mod:`khora.filter.compilers._split` closed that, so the context is back on
# ``"split"`` with the rest.
_EXPECTED_UNSUPPORTED_MODE: Mapping[str, str] = {
    "postgresql": "split",
    "sqlite": "split",
    "sqlite_lance": "split",
    "surrealdb": "split",
}

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
    # unpushable leaf is normally left unconsumed rather than raising — except on
    # SurrealDB, whose compiler is not yet sound under split (see the map above).
    assert ctx.on_unsupported == _EXPECTED_UNSUPPORTED_MODE[name]

    mapping = ctx.field_mapping
    assert mapping is not None
    # Exactly the declared system keys plus the ``metadata`` root — the
    # ``metadata`` entry must be present even where it is identity, because
    # ``compile_postgres`` resolves it eagerly.
    declared = _declared_keys(name)
    assert set(mapping) == set(declared) | {"metadata"}
    assert all(mapping[key] == key for key in declared)
    assert mapping["metadata"] == metadata_column
    # The withheld keys are absent — and that omission is what routes them to
    # the caller's post-filter, since the key set IS the pushdown whitelist.
    assert not set(mapping) & _WITHHELD_KEYS[name]


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


def _real_columns(name: str) -> frozenset[str]:
    """The physical column/field names each store's own schema actually defines.

    Read from each store's schema artifact rather than restated here, so this is
    a genuine cross-check of the mappings instead of a comparison against a copy
    of them. Needs no database server and no fixtures.
    """
    if name in {"postgresql", "sqlite_lance"}:
        # Both reuse the shared declarative model. ``.c`` is keyed by PHYSICAL
        # column name, so the metadata column reads as ``metadata`` even though
        # the mapped attribute is ``metadata_``.
        return frozenset(DocumentModel.__table__.c.keys())
    if name == "sqlite":
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(_SCHEMA_SQL)
            return frozenset(row[1] for row in conn.execute("PRAGMA table_info(documents)"))
        finally:
            conn.close()
    if name == "surrealdb":
        schema = Path(
            str(__import__("khora.storage.backends.surrealdb.schema", fromlist=["__file__"]).__file__)
        ).read_text()
        return frozenset(re.findall(r"DEFINE FIELD (?:IF NOT EXISTS )?(\w+) ON document\b", schema))
    raise AssertionError(f"unknown store {name!r}")


@pytest.mark.parametrize(
    ("name", "builder", "target", "metadata_column"),
    _ALL_CONTEXTS,
    ids=[case[0] for case in _ALL_CONTEXTS],
)
def test_field_mapping_matches_the_real_physical_columns(
    name: str,
    builder: Callable[[], CompileContext],
    target: str,
    metadata_column: str,
) -> None:
    """Every declared physical name must exist in the store's own schema.

    The other context assertions compare the mapping to literals restated in
    this module, so they would happily agree with a mapping that names a column
    no table has. This one reads each store's real schema and is the guard that
    actually fails when a column is renamed or dropped underneath a context —
    the exact drift this whole module exists to prevent.
    """
    columns = _real_columns(name)
    assert columns, f"failed to read any columns for {name}"

    mapping = builder().field_mapping
    assert mapping is not None
    missing = set(mapping.values()) - columns
    assert not missing, f"{name} maps to columns its schema does not define: {sorted(missing)}"


@pytest.mark.parametrize(
    ("name", "builder", "target", "metadata_column"),
    _ALL_CONTEXTS,
    ids=[case[0] for case in _ALL_CONTEXTS],
)
def test_occurred_at_is_absent_from_every_real_documents_schema(
    name: str,
    builder: Callable[[], CompileContext],
    target: str,
    metadata_column: str,
) -> None:
    # The reason ``occurred_at`` is excluded from every mapping, asserted against
    # the schemas themselves rather than left as prose: it is chunk event-time
    # and no documents table has a column for it on any backend.
    assert "occurred_at" not in _real_columns(name)


@pytest.mark.parametrize(
    ("name", "builder", "target", "metadata_column"),
    _ALL_CONTEXTS,
    ids=[case[0] for case in _ALL_CONTEXTS],
)
def test_withheld_keys_are_real_columns_withheld_for_soundness(
    name: str,
    builder: Callable[[], CompileContext],
    target: str,
    metadata_column: str,
) -> None:
    """The withheld keys DO have columns — they are held back, not missing.

    This is what separates the two reasons a key can be undeclared, which the
    mapping alone cannot distinguish. ``occurred_at`` is undeclared because no
    ``documents`` table has the column; ``created_at`` / ``source_timestamp`` on
    the two SQLite stores are undeclared even though the columns exist, because
    the stored string format does not order against this compiler's binds.
    Asserting the columns are real is what keeps the second reason from being
    quietly re-read as the first — and what would fail if someone "fixed" a
    withheld key by deleting the column instead of the pushdown.
    """
    columns = _real_columns(name)
    assert columns, f"failed to read any columns for {name}"
    for key in _WITHHELD_KEYS[name]:
        assert key in columns, f"{name} withholds {key!r}, which is not even a column"


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
    """``_emits_column`` must DISCRIMINATE the two physical spellings.

    ``metadata`` is a prefix of ``metadata_``, so a plain substring assertion is
    green for both and cannot fail against the defect this module exists to
    guard. This case pins that the word-boundary matcher tells them apart:
    compile through a context whose metadata mapping is the wrong spelling and
    require the shipped-column assertion to reject it.

    Scope, stated precisely: both the shipped and the wrong spelling are
    hardcoded here, so this proves the MATCHER discriminates — not that any
    shipped mapping is correct. The mapping itself is guarded by
    :func:`test_field_mapping_matches_the_real_physical_columns` (against each
    store's real schema) and by ``test_context_declares_the_physical_schema``.
    """

    def _swapped(ctx: CompileContext, wrong_column: str) -> CompileContext:
        mapping = dict(ctx.field_mapping or {}) | {"metadata": wrong_column}
        return dataclasses.replace(ctx, field_mapping=mapping)

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


# Both SQLite stores share ONE probe: it interrogates the process's stdlib
# ``sqlite3`` build, not either store's schema, so the two cannot disagree.
def _direct_json1_probe() -> bool:
    """Ask this interpreter's ``sqlite3`` build about JSON1, independently."""
    conn = sqlite3.connect(":memory:")
    try:
        return conn.execute("SELECT json_valid('{}')").fetchone()[0] == 1
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def test_json1_probe_agrees_with_the_hosts_sqlite_build() -> None:
    """The probe reports what this host's SQLite actually supports.

    Asserted against an independent probe rather than a hardcoded ``True``, so
    the test states the contract (the flag tracks the runtime) instead of the
    accident of which SQLite this host shipped. ``aiosqlite`` and SQLAlchemy's
    ``sqlite+aiosqlite`` both run on this same in-process library, which is what
    makes a single process-wide answer correct for both stores.
    """
    result = sqlite_has_json1()
    assert isinstance(result, bool)
    assert result is _direct_json1_probe()


def test_json1_probe_is_stable_across_calls() -> None:
    # Memoized: the answer cannot change within a process, and a context built
    # twice must not disagree with itself.
    assert sqlite_has_json1() is sqlite_has_json1()


def test_json1_probe_interrogates_json1_specifically(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe must ask about JSON1 — not merely run *some* query successfully.

    Without this, a probe rewritten to ``SELECT 1`` passes every other case on a
    JSON1-enabled host: the fail-closed case below drives its negative branch
    with a connection that raises for ANY SQL, so it cannot tell the two apart.
    A ``SELECT 1`` probe would report ``True`` on a build lacking JSON1 and
    ``compile_lance`` would then emit ``json_extract`` / ``json_each`` against a
    build that cannot run them — exactly what the probe exists to prevent.

    Orthogonal to :func:`test_context_reads_the_probe_rather_than_hardcoding_it`:
    that one pins the context→probe wire, this one pins what the probe asks.
    Fixing either leaves the other's mutant alive.
    """
    executed: list[str] = []
    # Bind the real factory before patching, or the wrapper below re-enters the
    # patched one and recurses.
    real_connect = sqlite3.connect

    class _RecordingConnection:
        def __init__(self) -> None:
            self._conn = real_connect(":memory:")

        def execute(self, sql: str) -> object:
            executed.append(sql)
            return self._conn.execute(sql)

        def close(self) -> None:
            self._conn.close()

    monkeypatch.setattr(sqlite3, "connect", lambda *_a, **_k: _RecordingConnection())
    sqlite_has_json1.cache_clear()
    try:
        sqlite_has_json1()
    finally:
        sqlite_has_json1.cache_clear()
    monkeypatch.undo()

    assert executed, "the probe ran no query at all"
    assert any("json_valid" in sql for sql in executed), f"probe must interrogate a JSON1 function, but ran: {executed}"
    # Restore the host's real answer for every later test in this process.
    assert sqlite_has_json1() is _direct_json1_probe()


def test_json1_probe_fails_closed_when_the_query_errors(
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
    sqlite_has_json1.cache_clear()
    try:
        assert sqlite_has_json1() is False
    finally:
        sqlite_has_json1.cache_clear()

    monkeypatch.undo()
    # Back to the host's real answer for every later test in this process.
    assert sqlite_has_json1() is _direct_json1_probe()


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


@pytest.mark.parametrize(
    ("name", "builder", "metadata_column"),
    _SQLITE_CONTEXTS,
    ids=[case[0] for case in _SQLITE_CONTEXTS],
)
def test_context_reads_the_probe_rather_than_hardcoding_it(
    name: str,
    builder: Callable[[], CompileContext],
    metadata_column: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A context that hardcoded ``sqlite_json1=True`` must fail here.

    The case above compares the context's flag to a live probe, so on a
    JSON1-enabled host both sides are ``True`` and it cannot distinguish "reads
    the probe" from "hardcoded". Forcing the probe to the opposite answer is
    what makes the wire observable.
    """
    # Patch the name in the STORE module, not in ``_sqlite_capabilities``: each
    # store does ``from ... import sqlite_has_json1``, which binds the function
    # object into its own namespace at import time, so patching the source
    # module would not be observed here.
    store_module = builder.__module__
    for forced in (False, True):
        monkeypatch.setattr(f"{store_module}.sqlite_has_json1", lambda _v=forced: _v)
        try:
            assert builder().schema_capabilities.sqlite_json1 is forced
        finally:
            monkeypatch.undo()

    # The real probe is memoized and was never displaced, but assert the host's
    # answer is intact so nothing leaks into later tests.
    assert builder().schema_capabilities.sqlite_json1 is _direct_json1_probe()


# ===========================================================================
# The undeclared-key defences — each of these INVERTED a pin from the previous
# revision, where the same shape was recorded as a known gap.
# ===========================================================================


def test_occurred_at_is_defended_on_surrealdb() -> None:
    """The shipped SurrealDB context DEFERS an undeclared system key.

    ``compile_surrealdb`` treats the ``field_mapping`` key set as its
    declared+pushable whitelist, so the undeclared ``occurred_at`` leaf never
    reaches the query. This used to surface as a ``RecallFilterUnsupportedError``
    because the context shipped ``on_unsupported="raise"`` — an interim posture
    that existed only to keep the placeholder inversion below unreachable. With
    the all-or-nothing gate in :mod:`khora.filter.compilers._split` closing that,
    the context is back on ``"split"`` and the leaf takes the ordinary residual
    path: a match-all placeholder, nothing consumed, the caller's post-filter
    enforces it.
    """
    compiled = compile_surrealdb(_ast({"occurred_at": {"$gte": _DT}}), surrealdb_context())
    # Non-constraining placeholder, not a predicate against a field the
    # ``document`` table does not have.
    assert compiled.predicate.strip() == "(true)"
    assert compiled.params == {}
    assert compiled.consumed_keys == frozenset()
    assert not _emits_column(_norm(compiled.predicate), "occurred_at")


def test_occurred_at_is_defended_on_sql_backends() -> None:
    """The three SQL contexts defer ``occurred_at`` instead of inventing a column.

    Previously ``compile_postgres`` / ``compile_lance`` fell back to identity for
    a key absent from ``field_mapping``, so this leaf compiled to
    ``documents.occurred_at`` — a column no ``documents`` table has — AND was
    reported in ``consumed_keys``, telling the caller it had been pushed while
    the statement itself would fail at execute time. Both compilers now honour a
    non-``None`` key set as the pushdown whitelist, so the undeclared key emits
    the match-all placeholder and stays a residual.

    Do NOT restore green here by declaring ``occurred_at`` in a ``field_mapping``
    — that points the leaf back at a column that does not exist.
    """
    wire = {"occurred_at": {"$gte": _DT}}

    sql = _postgres_sql(wire, postgres_context())
    assert not _emits_column(sql, "documents.occurred_at")
    assert compile_postgres(_ast(wire), postgres_context()).consumed_keys == frozenset()

    for _name, builder, _column in _SQLITE_CONTEXTS:
        compiled = compile_lance(_ast(wire), builder())
        assert not _emits_column(_norm(compiled.predicate), "documents.occurred_at")
        assert compiled.predicate.strip() == "(1)"
        assert compiled.consumed_keys == frozenset()


def test_surrealdb_defers_a_negated_or_rather_than_inverting_a_placeholder() -> None:
    """An undeclared key under ``$not``/``$or`` defers the node, it does not invert.

    The placeholder ``compile_surrealdb`` emits for an unsupported leaf is the
    literal ``true``, which is non-constraining only inside a positive
    conjunction. Under a negated OR it used to invert: ``!(... OR true)`` is
    always false, so the query silently returned NO rows while ``consumed_keys``
    still reported ``title`` as pushed — and a post-filter only narrows, so the
    wrongly-excluded rows were unrecoverable. The gate now defers the whole
    ``$not`` subtree instead, emitting the bare match-all and consuming nothing.

    Run against the SHIPPED context, not a split-mode copy of it. The previous
    revision had to build its own context because the shipped one was
    ``"raise"``; asserting against the real one is what makes this the alarm for
    a future ``"split"`` flip landing without the gate.
    """
    compiled = compile_surrealdb(
        _ast({"$not": {"$or": [{"title": "x"}, {"occurred_at": {"$gt": _DT}}]}}),
        surrealdb_context(),
    )
    sql = _norm(compiled.predicate)
    # The whole negation is deferred: the bare placeholder, with no negation and
    # no inverted disjunct left in the fragment at all.
    assert sql.strip() == "true"
    assert "!(" not in sql
    assert "or" not in sql
    assert compiled.params == {}
    # ``title`` IS declared and would push on its own — it is unconsumed here
    # only because the gate deferred the subtree containing it.
    assert compiled.consumed_keys == frozenset()
    assert compile_surrealdb(_ast({"title": "x"}), surrealdb_context()).consumed_keys == frozenset({"title"})


def test_sqlite_lance_datetime_bind_does_not_match_the_stored_format() -> None:
    """The date-valued keys are withheld, so the mismatched bind never ships.

    ``documents`` rows on this stack are written through SQLAlchemy's SQLite
    ``DATETIME`` type, which stores ``'2026-01-31 12:30:00.000000'`` (SPACE
    separator, no offset), while ``compile_lance`` binds a datetime operand as
    ``.isoformat()`` (``'T'`` separator, offset included) and relies on
    lexicographic ISO comparison. ``' '`` (0x20) sorts before ``'T'`` (0x54), so a
    pushed-down ``created_at`` / ``source_timestamp`` bound silently excludes rows
    from its own day.

    The previous revision pinned ``consumed_keys == {"created_at"}`` here,
    because the mapping declared the key and ``compile_lance`` ignored the key
    set anyway. Dropping the two date keys from ``_PUSHABLE_SYSTEM_KEYS`` is what
    makes that assertion fail, and the inverted assertion below — the key is a
    RESIDUAL — is the standing guard that the drop stays dropped. Do NOT restore
    green by re-declaring ``created_at``: that reinstates the silent mismatch.

    The second half compiles the same leaf through a context that DOES declare
    the key, so the defective bind is still visible in a test rather than only in
    a docstring. That is what keeps the withholding legible as load-bearing —
    without it, nothing here shows what re-declaring the key would cost.
    """
    compiled = compile_lance(_ast({"created_at": {"$gte": _DT}}), sqlite_lance_context())
    assert compiled.consumed_keys == frozenset()
    assert compiled.predicate.strip() == "(1)"
    assert compiled.params == {"args": []}
    assert not _emits_column(_norm(compiled.predicate), "documents.created_at")

    # What re-declaring the key would emit: the ``'T'``-separated, offset-bearing
    # bind that does not compare against the stored ``' '``-separated form.
    ctx = sqlite_lance_context()
    redeclared = dataclasses.replace(ctx, field_mapping=dict(ctx.field_mapping or {}) | {"created_at": "created_at"})
    pushed = compile_lance(_ast({"created_at": {"$gte": _DT}}), redeclared)
    assert pushed.consumed_keys == frozenset({"created_at"})
    assert pushed.params == {"args": ["2026-01-31T12:30:00+00:00"]}
    # Stored form for the same instant, which the bind above does NOT equal.
    assert "2026-01-31 12:30:00.000000" != pushed.params["args"][0]


def test_sqlite_lance_date_predicate_survives_the_day_boundary() -> None:
    """A same-day row survives a ``$gte`` on midnight — against a REAL table.

    The end-to-end form of the mismatch above, which is the only form that shows
    the consequence rather than the bind string. A row written at 09:00 on the
    filter's own day is materialized through the SAME SQLAlchemy column type the
    ``documents`` model declares, so the stored spelling is authentic rather than
    restated here; then the compiled fragment is executed as the ``WHERE`` clause
    it would be in production.

    The CONTROL is what makes this a proof: compiled through a context that
    re-declares ``created_at`` — i.e. the mapping as it shipped before the two
    date keys were withheld — the very same row is EXCLUDED, because
    ``'2026-08-04 09:00:00.000000' >= '2026-08-04T00:00:00+00:00'`` is false
    lexicographically (``' '`` 0x20 sorts before ``'T'`` 0x54). Under the shipped
    mapping the leaf is not pushed at all, the row reaches the caller, and the
    full-AST post-filter keeps it.

    Deliberately not a lancedb test: only the relational half is in play, so this
    stays a hermetic in-memory SQLite case with no optional extra required.
    """
    row_at = datetime(2026, 8, 4, 9, 0, tzinfo=UTC)
    # A two-column stand-in, but ``created_at`` carries the model's OWN type
    # object — the stored format is decided by that type plus the SQLite
    # dialect, so reusing it is what keeps this faithful. Building the full
    # ``DocumentModel`` table is not possible here: its ``JSONB`` columns have no
    # SQLite DDL rendering outside the migration chain, and none of them matter
    # to a datetime comparison.
    metadata = sa.MetaData()
    documents = sa.Table(
        "documents",
        metadata,
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("created_at", DocumentModel.__table__.c.created_at.type, nullable=False),
    )
    engine = sa.create_engine("sqlite://")
    try:
        metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(sa.insert(documents).values(id="doc-1", created_at=row_at))
            stored = conn.exec_driver_sql("SELECT created_at FROM documents").scalar_one()
        # The stored spelling, read back from the table rather than asserted from
        # memory: space-separated, no offset.
        assert stored == "2026-08-04 09:00:00.000000"

        ast = _ast({"created_at": {"$gte": "2026-08-04T00:00:00+00:00"}})

        def _survivors(ctx: CompileContext) -> list[str]:
            compiled = compile_lance(ast, ctx)
            with engine.begin() as conn:
                rows = conn.exec_driver_sql(
                    f"SELECT id FROM documents WHERE {compiled.predicate}",  # noqa: S608 - compiled fragment, binds are parameterized
                    tuple(compiled.params["args"]),
                ).fetchall()
            return [row[0] for row in rows]

        ctx = sqlite_lance_context()
        # SHIPPED: the leaf is withheld, so the prefilter does not narrow and the
        # row reaches the caller.
        assert _survivors(ctx) == ["doc-1"]

        # CONTROL: re-declare the key and the same row disappears — the defect
        # this withholding exists to prevent, executed rather than described.
        redeclared = dataclasses.replace(
            ctx, field_mapping=dict(ctx.field_mapping or {}) | {"created_at": "created_at"}
        )
        assert _survivors(redeclared) == []
    finally:
        engine.dispose()

    # The caller's post-filter evaluates the FULL AST and keeps the row, so the
    # shipped path returns it — correct rows across the day boundary.
    post_filter = compile_python(ast, build_compile_context("documents", on_unsupported="split")).predicate
    assert post_filter({"id": "doc-1", "created_at": row_at}) is True


# ===========================================================================
# Split-mode soundness — the all-or-nothing OR/NOT gate, per documents context.
# ===========================================================================


# The unconsumable leaf every documents context agrees on: ``occurred_at`` is a
# system key (so it reaches a compiler as an AST leaf) that no documents mapping
# declares. Using one shape across all four keeps the gate the only variable.
_UNDECLARED_LEAF: dict = {"occurred_at": {"$gt": _DT}}

# ``(compiler, context builder, the match-all placeholder that compiler emits)``.
_SPLIT_TARGETS: tuple[tuple[str, Callable, Callable[[], CompileContext], str], ...] = (
    ("postgresql", compile_postgres, postgres_context, "true"),
    ("sqlite", compile_lance, sqlite_context, "1"),
    ("sqlite_lance", compile_lance, sqlite_lance_context, "1"),
    ("surrealdb", compile_surrealdb, surrealdb_context, "true"),
)


def _rendered(name: str, compiled: CompiledFilter[Any]) -> str:
    """The emitted fragment as a string, rendering SQLAlchemy elements inline.

    ``literal_binds`` matters here rather than being cosmetic: SQLAlchemy folds
    ``or_(x, true())`` down to ``true`` during compilation, so whether the gate
    fired is only observable AFTER rendering. Reading ``str(element)`` would show
    an un-short-circuited tree and hide it.
    """
    if name == "postgresql":
        predicate = compiled.predicate
        assert isinstance(predicate, ColumnElement), f"not a SQLAlchemy element: {type(predicate)!r}"
        return _norm(str(predicate.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})))
    return _norm(compiled.predicate)


@pytest.mark.parametrize(
    ("name", "compiler", "builder", "placeholder"),
    _SPLIT_TARGETS,
    ids=[case[0] for case in _SPLIT_TARGETS],
)
def test_negated_or_defers_whole_rather_than_inverting(
    name: str,
    compiler: Callable,
    builder: Callable[[], CompileContext],
    placeholder: str,
) -> None:
    """``$not`` over an ``$or`` holding an undeclared key defers the WHOLE node.

    The soundness property behind every documents context, stated per compiler.
    A match-all placeholder is superset-safe only in positive position: under a
    negation it inverts to match-NOTHING, which empties the result set — and a
    post-filter only narrows, so those rows are unrecoverable. The gate therefore
    refuses to push an ``$or`` / ``$not`` unless its entire subtree is
    consumable.

    The assertion is on the RENDERED fragment: it must be the bare match-all, not
    a negation and not a ``false``. ``title`` is declared on every documents
    mapping and pushes on its own, so its absence from ``consumed_keys`` here is
    the gate deferring the subtree — not the key being unsupported.
    """
    ctx = builder()
    assert ctx.on_unsupported == "split", "this case only means anything under split"

    compiled = compiler(_ast({"$not": {"$or": [{"title": "x"}, _UNDECLARED_LEAF]}}), ctx)
    sql = _rendered(name, compiled)

    # The bare match-all placeholder — the whole negation deferred.
    assert sql.strip() == placeholder
    # Nothing that could empty the result set survived into the fragment.
    assert "not" not in sql
    assert "!(" not in sql
    assert " or " not in sql
    assert "false" not in sql
    assert "0" not in sql.replace(placeholder, "")
    assert compiled.consumed_keys == frozenset()

    # CONTROL — ``title`` alone DOES push here, so the empty ``consumed_keys``
    # above is the gate firing rather than a uniformly unsupported leaf.
    assert compiler(_ast({"title": "x"}), ctx).consumed_keys == frozenset({"title"})


@pytest.mark.parametrize(
    ("name", "compiler", "builder", "placeholder"),
    _SPLIT_TARGETS,
    ids=[case[0] for case in _SPLIT_TARGETS],
)
def test_key_in_both_a_pushed_leaf_and_a_deferred_subtree_is_not_consumed(
    name: str,
    compiler: Callable,
    builder: Callable[[], CompileContext],
    placeholder: str,
) -> None:
    """A path pushed in one branch and deferred in another is NOT reported consumed.

    ``consumed_keys`` is a set of dotted paths, but a path can occur many times
    in one AST. Here the same key sits in a conjunctive leaf that IS pushed and
    again inside a ``$not`` the gate defers wholesale. Reporting it consumed
    would tell a caller differencing ``leaf_keys - consumed_keys`` that the
    deferred occurrence is already enforced, so it would never be post-filtered
    and the query would return rows the filter excludes.

    Both halves of the emitted fragment are asserted, because the interesting
    failure is not "nothing was pushed": the conjunctive leaf SHOULD still push
    (that is the pushdown being preserved) while the key is still reported as a
    residual. A test that only checked ``consumed_keys`` would pass against a
    compiler that had simply stopped pushing anything.
    """
    ctx = builder()
    key, pushed_leaf, deferred_leaf = _repeated_key_case(name)

    compiled = compiler(
        _ast({"$and": [pushed_leaf, {"$not": {"$or": [deferred_leaf, _UNDECLARED_LEAF]}}]}),
        ctx,
    )
    sql = _rendered(name, compiled)

    # The conjunctive occurrence still pushed — pushdown is preserved.
    assert key in sql, f"expected the conjunctive {key!r} leaf to still push: {sql}"
    # ...and the key is nonetheless a residual, because its other occurrence was
    # deferred with the ``$not``.
    assert key not in compiled.consumed_keys
    assert compiled.consumed_keys == frozenset()

    # CONTROL — the same key with only the conjunctive occurrence IS consumed,
    # so the exclusion above is per-occurrence accounting and not the key being
    # unpushable on this context.
    assert compiler(_ast(pushed_leaf), ctx).consumed_keys == frozenset({key})


def _repeated_key_case(name: str) -> tuple[str, dict, dict]:
    """A declared key plus two leaves on it, for the repeated-path case above.

    The key has to be one the context actually declares, or the case degenerates
    to "nothing was pushed" and proves nothing. The two SQLite contexts withhold
    both date-valued keys, so they use a string key and equality leaves instead
    of a range.
    """
    if name in {"sqlite", "sqlite_lance"}:
        return "source_type", {"source_type": "email"}, {"source_type": "web"}
    return (
        "created_at",
        {"created_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
        {"created_at": {"$lt": "2026-02-01T00:00:00+00:00"}},
    )


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
