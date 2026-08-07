"""Execution-seam spy for the document scans — is the reported split HONEST?

``DocumentScanStep.consumed_keys`` is a self-report. Every existing scan test
reads it, and none of them can tell the difference between the three things it
might mean:

* **honest** — the leaf really is in the statement the driver ran;
* **a perf lie** — the compiler claims a leaf was pushed, SQL never saw it, and
  the caller's post-filter silently covers for it. Correct rows, and the whole
  point of pushdown (a narrowed window, fewer rows fetched) quietly gone. No
  row-level assertion anywhere can see this.
* **a correctness lie** — the compiler claims a leaf was pushed, and the caller
  *believes* it: a caller that treated ``consumed_keys`` as permission to skip
  those leaves would then enforce them nowhere at all. Today's coordinator always
  re-checks the full AST, so this shows up as nothing at all on the result
  surface — which is exactly why it needs a seam-level check rather than a
  row-level one.

So this module captures at the real driver boundary, per tier, and the tests
assert against the values actually bound. Four stores, three seams:

* **PostgreSQL** and **sqlite_lance** both hold a SQLAlchemy ``AsyncEngine`` at
  ``store._engine`` — :func:`sqlalchemy_sql_log` attaches a
  ``before_cursor_execute`` listener to its ``sync_engine`` (the same reason
  ``sqlite_lance/relational.py``'s pragma listener attaches there) and detaches on
  exit.
* **raw SQLite** issues ``store._conn.execute(sql, params)`` on an aiosqlite
  connection, and **SurrealDB** issues ``store._conn.query(sql, bindings)``:
  :func:`method_sql_log` wraps either by name.

**Why the assertion is value-based and not column-name-based.** Grepping the
statement text for ``source_type`` proves nothing: ``created_at``, ``id``,
``namespace_id`` and ``status`` appear in the ORDER BY, the keyset predicate and
the namespace scope of *every* statement whether or not a filter leaf pushed, and
a deferred leaf's column name can still show up in a ``SELECT *`` expansion or a
comment. The bound *operand* is the discriminating evidence: a pushed leaf's
literal reaches the driver, and a deferred one's cannot.

Use a string-valued system key as the probe (``source_type`` is the natural one —
every documents context declares it, and all four compilers push it) with a
globally distinctive literal, :data:`PROBE_VALUE`, so a match cannot be some other
column's coincidental value. Do **not** probe a date key by value: PostgreSQL
binds a ``datetime`` object, the SQLite tiers bind their own string
serializations, and SurrealDB binds a ``datetime`` again, so "is this value in the
params" is a dialect quiz rather than a split check. Date-key pushdown is covered
by the compile-split recompute instead (``step.consumed_keys`` against a fresh
compile with the same compiler and context).

Not to be confused with :mod:`tests.test_helpers.filter_spy`, which watches a
different seam for a different question: it captures the ``filter_ast`` crossing an
engine/compiler boundary and compares its ``canonical_hash``, proving the validated
AST was *threaded* unchanged. This module sits one level lower — after compilation,
at the driver — and asks whether the compiled predicate's operands were actually
*bound*. A path can thread the AST faithfully and still push none of it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from sqlalchemy import event

# The probe literal. Deliberately unlike any real ``source_type`` and unlike any
# other value the seed corpus writes, so finding it among the bound params can
# only mean the probe leaf was pushed.
PROBE_VALUE = "ZZZ_e3_marker"

# One captured statement: the SQL text and whatever parameter container the seam
# handed the driver (positional sequence, bind dict, or None).
SqlLog = list[tuple[str, Any]]


@contextmanager
def sqlalchemy_sql_log(engine: Any) -> Iterator[SqlLog]:
    """Record ``(statement, parameters)`` for every cursor execution on ``engine``.

    ``before_cursor_execute`` fires after the dialect has compiled and bound the
    statement, so the parameters recorded are the ones the DBAPI receives — not
    the SQLAlchemy-level construct. The listener goes on ``engine.sync_engine``
    because the async engine is a facade over it (same reason as the pragma
    listener in ``sqlite_lance/relational.py``), and is removed on exit so a
    fixture-scoped engine does not accumulate listeners across tests.
    """
    log: SqlLog = []

    def _record(_conn, _cursor, statement, parameters, _context, _executemany) -> None:  # noqa: ANN001
        log.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", _record)
    try:
        yield log
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _record)


@contextmanager
def method_sql_log(obj: Any, method_name: str) -> Iterator[SqlLog]:
    """Record ``(sql, params)`` for every call to ``obj.method_name(sql, params)``.

    The seam for the two stores that talk to their driver directly: aiosqlite's
    ``execute(sql, params)`` and ``SurrealDBConnection.query(sql, bindings)``. The
    second positional (or its keyword form) is the parameter container in both.

    A resumed SurrealDB step issues up to TWO statements — the Q1 tie block and
    the Q2 strictly-older range — and both land in the log; assertions here are
    over the union of every statement in the step, which is the right granularity
    (a leaf pushed into only one of the two legs is still pushed).
    """
    log: SqlLog = []
    original: Callable[..., Any] = getattr(obj, method_name)
    # Whether the name was the object's OWN attribute before we shadowed it. Both
    # seams define theirs on the class, so the correct restore is to delete the
    # instance attribute rather than leave a bound copy shadowing the class.
    had_own_attr = method_name in vars(obj)

    @functools.wraps(original)
    async def _recording(sql: str, *args: Any, **kwargs: Any) -> Any:
        params = args[0] if args else next((kwargs[k] for k in ("params", "bindings") if k in kwargs), None)
        log.append((sql, params))
        return await original(sql, *args, **kwargs)

    setattr(obj, method_name, _recording)
    try:
        yield log
    finally:
        if had_own_attr:
            setattr(obj, method_name, original)
        else:
            delattr(obj, method_name)


def flatten_params(sql_log: SqlLog) -> set[str]:
    """Every bound value in ``sql_log``, stringified, as one set.

    Containers nest differently per seam — a positional tuple, a bind dict, a list
    of tuples for an executemany — so this walks them all and stringifies the
    leaves. Stringifying is what makes one assertion work across dialects that
    bind the same logical operand as different Python types; it is sound for the
    string-valued probe this module is for, and is the reason date keys are out of
    scope (see the module docstring).
    """
    values: set[str] = set()

    def _walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            for item in value.values():
                _walk(item)
            return
        if isinstance(value, (str, bytes)):
            values.add(str(value))
            return
        if isinstance(value, Sequence):
            for item in value:
                _walk(item)
            return
        values.add(str(value))

    for _statement, parameters in sql_log:
        _walk(parameters)
    return values


def _statement_text(sql_log: SqlLog) -> str:
    """Every captured statement's SQL text, joined — searched alongside the params.

    A pushed operand should arrive as a *bind*, and on all four seams today it does:
    ``compile_lance`` never inlines a caller value (its ``?`` placeholders are
    rewritten to ``:kf0``… and bound by ``_lance_fragment_to_text``), the postgres
    compiler emits a SQLAlchemy expression, and the SurrealDB compiler emits
    ``$f_0``-style bindings. So searching the text is a **tripwire, not a live
    hazard** — but it is the direction that matters for the residual half: if a
    compiler ever started inlining a literal, "the value is absent from the params"
    would become a false pass while SQL enforced the leaf anyway. Including the text
    costs nothing and closes that.
    """
    return "\n".join(statement for statement, _parameters in sql_log)


def assert_split_honest(
    sql_log: SqlLog,
    *,
    pushed_values: Sequence[Any] = (),
    residual_values: Sequence[Any] = (),
    must_touch: str = "document",
) -> None:
    """Assert the executed statements bound exactly the operands they claim to.

    ``pushed_values`` must all appear in the executed statements (bound param or
    SQL text) — a value missing is the perf lie: reported as pushed, absent from
    the statement, the post-filter quietly covering. ``residual_values`` must all
    be absent — a value present is the correctness lie's fingerprint: reported as
    deferred, yet enforced in SQL, so the two halves of the split disagree about
    who owns the leaf.

    **Two vacuity guards, and they are the point of this function being a function.**
    Every assertion here is over the statements that ran, so a spy attached to the
    wrong seam captures nothing and satisfies the whole ``residual_values`` half
    unconditionally — a green run that proves nothing and looks identical to a real
    one. So: the log must be non-empty, AND at least one statement must mention
    ``must_touch``. The default ``"document"`` is a substring of the ``documents``
    table on three tiers and the exact ``document`` table on SurrealDB, so one
    default covers all four; it rejects a log that captured only a namespace lookup,
    a ``PRAGMA``, or a ``BEGIN``.
    """
    assert sql_log, "the spy captured no statements — it is attached to the wrong seam, or the scan never ran"
    text = _statement_text(sql_log)
    assert must_touch in text, (
        f"no captured statement mentions {must_touch!r}, so the spy did not observe a documents scan — "
        f"captured: {[statement[:120] for statement, _ in sql_log]}"
    )
    params = flatten_params(sql_log)

    def _present(value: Any) -> bool:
        # Exact match against a bound param (the expected route), OR a substring of
        # the SQL text (the inlined-literal tripwire above).
        needle = str(value)
        return needle in params or needle in text

    for value in pushed_values:
        assert _present(value), (
            f"{value!r} was reported as pushed down but appears in neither the bound params nor the SQL text — "
            f"SQL never enforced it and the post-filter is silently covering for the report"
        )
    for value in residual_values:
        assert not _present(value), (
            f"{value!r} was reported as post-filtered (residual) but IS in the executed statement "
            f"(params or SQL text) — the statement enforced a leaf the split says it deferred"
        )


def force_residual(monkeypatch: Any, backend_module: Any, drop_key: str = "source_type") -> None:
    """Make ``drop_key`` unpushable on ``backend_module``, on every scan path.

    The **uniform** forced-residual lever across all four stores, and it is not the
    obvious one. A :class:`~khora.filter.context.SchemaCapabilities` override does
    NOT work here: only ``compile_lance`` gates on a capability flag
    (``sqlite_json1``), and the postgres / surrealdb documents compilers never read
    ``schema_capabilities`` at all. What every one of them *does* read is the
    ``field_mapping`` key set as the pushdown whitelist — so dropping a key from it
    defers that leaf on all four under ``on_unsupported="split"``.

    Patching is at the module level because all four stores call
    ``_documents_compile_context()`` as a bare module-global name (``postgresql.py``,
    ``sqlite_lance/relational.py``, ``sqlite.py``, and ``surrealdb/relational.py``
    inside ``_documents_where``), so one ``setattr`` covers every scan path in that
    store — including the second statement of a resumed SurrealDB step.
    """
    from dataclasses import replace

    original = backend_module._documents_compile_context  # noqa: SLF001

    def _patched() -> Any:
        ctx = original()
        mapping = dict(ctx.field_mapping or {})
        assert drop_key in mapping, (
            f"{drop_key!r} is not in this store's field_mapping, so dropping it forces nothing — "
            f"the probe key must be one the store normally pushes"
        )
        del mapping[drop_key]
        return replace(ctx, field_mapping=mapping)

    monkeypatch.setattr(backend_module, "_documents_compile_context", _patched)


__all__ = [
    "PROBE_VALUE",
    "assert_split_honest",
    "flatten_params",
    "force_residual",
    "method_sql_log",
    "sqlalchemy_sql_log",
]
