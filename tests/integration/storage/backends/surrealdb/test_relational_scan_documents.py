"""``SurrealDBRelationalAdapter.scan_documents`` — the bounded keyset scan.

Runs against an in-memory SurrealDB (``mode="memory"``) — no docker required,
same fixture shape as :mod:`tests.integration.storage.backends.surrealdb.test_list_documents_order`.
Skipped when the ``surrealdb`` extra is not installed.

This leg is not a transcription of the SQLAlchemy pair (khora #1586). Three
things are structurally different on this store and each gets its own coverage
below:

* **The keyset predicate is two clauses, not a row-value compare.** SurrealQL
  has no ``(a, b) < (x, y)``, so the resume position is expressed as
  ``created_at < $ts OR (created_at = $ts AND id < $id)`` — a top-level ``OR``
  sitting in a ``WHERE`` that also carries the namespace scope. That makes
  *grouping* a correctness property of this store rather than a style choice,
  and it makes mid-tie resume the interesting case.
* **``id`` is a ``RecordID``, not a UUID column.** ``id < $rid`` is a record-id
  compare, and nothing outside these tests establishes that it agrees with the
  statement's own ``ORDER BY id DESC``. It was the ticket's highest-risk
  unknown; :func:`test_walk_visits_every_document_exactly_once_in_total_order`
  is the answer, and it is kept as a regression guard now that the answer is
  known.
* **Cursor and ``updated_before`` operands bind as Python objects.** A
  stringified datetime does not compare against a ``TYPE datetime`` field at
  all on this store — it matches nothing rather than mis-ordering. Measured:
  no bound = 6 rows, ``datetime`` bind = 6 rows, ``.isoformat()`` bind = 0 rows.

Seeding goes through ``create_document``, the production write API, so every row
is serialized by the same path production writes take. Timestamps are pinned to
a whole second on purpose; see :mod:`tests.test_helpers.document_scan`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, NamedTuple
from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import MemoryNamespace, TenancyMode  # noqa: E402
from khora.core.models.document import DocumentStatus  # noqa: E402
from khora.filter import (  # noqa: E402
    CompiledFilter,
    CompileError,
    CompilerRegistry,
    RecallFilterUnsupportedError,
)
from khora.filter.compilers.python import compile_python  # noqa: E402

# The raw-SQLite store, imported into this SurrealDB module on purpose: §8's
# parity checkbox is a claim about two stores agreeing, and a claim about two
# stores cannot be checked from inside one of them. Both run in-process with no
# container (``:memory:`` and ``mode="memory"``), so this costs a fixture, not a
# lane.
from khora.storage.backends.sqlite import SQLiteRelationalBackend  # noqa: E402
from khora.storage.backends.sqlite import _documents_compile_context as _sqlite_documents_context  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402

# ``_documents_compile_context`` is private, and imported on purpose: the
# superset test below must compile with the *same* context the scan itself uses,
# or it would prove a property of some other context.
from khora.storage.backends.surrealdb.relational import (  # noqa: E402
    SurrealDBRelationalAdapter,
    _documents_compile_context,
)
from tests.test_helpers.document_scan import (  # noqa: E402
    WHOLE_SECOND,
    ScanSeed,
    as_utc,
    scan_seed,
    seed_documents,
    seed_varied,
    walk_scan,
    wire_to_ast,
    write_document,
)

pytestmark = pytest.mark.integration

_COMPILER_KEY = ("relational.surrealdb", "documents")


@pytest.fixture
async def adapter():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="doc_scan")
    await conn.connect()
    adapter = SurrealDBRelationalAdapter(conn)
    try:
        yield adapter
    finally:
        await conn.disconnect()


async def _make_namespace(adapter: Any) -> MemoryNamespace:
    nid = uuid4()
    return await adapter.create_namespace(MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED))


@pytest.fixture
async def namespace(adapter):
    return await _make_namespace(adapter)


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def _ladder(prefix: str, discriminator: str, n: int) -> list[UUID]:
    return [UUID(f"{prefix}{discriminator}{i:08x}") for i in range(n)]


def _two_namespace_ladders(n: int = 6) -> tuple[list[UUID], list[UUID]]:
    """Return ``(scanned_ids, foreign_ids)`` where EVERY foreign id sorts BELOW
    every scanned id.

    **This ordering is the whole point, and it must not be left to chance.**
    The keyset-grouping tripwire below resumes from a cursor inside the scanned
    namespace's tie block and asserts the ungrouped form leaks foreign rows. The
    unscoped right disjunct is ``created_at = $ts AND id < $cursor_id``, so it
    can only match foreign rows whose ids sort *below* the cursor. Measured on
    this store, with the two namespaces' ids in the two possible arrangements:

    * foreign ids above the cursor — the mutant leaks **0** rows and the test
      passes while proving nothing;
    * foreign ids below the cursor — the mutant leaks **4** rows and the test
      bites.

    ``id_ladder`` draws a fresh random 24-hex prefix per call, so building the
    two namespaces from two independent ``scan_seed()`` calls decides that
    arrangement by coin flip: the test would pass for the wrong reason about
    half the time and flake the rest. Here both ladders share one random
    **23**-hex-character head and differ at the very next nibble — ``0`` for the
    foreign rows, ``1`` for the scanned ones — followed by the same 8-hex
    counter, so the two ladders are 32 hex characters wide like any UUID and the
    first character that can differ is the discriminator. That fixes the relative
    order by construction while keeping the ids unique across runs, which a fixed
    all-zeros prefix would not: these tests run against a fresh ``memory://``
    instance today, but a persistent one would collide.

    Do NOT "simplify" this back to two ``id_ladder`` / ``scan_seed`` calls.
    """
    prefix = uuid4().hex[:23]
    return _ladder(prefix, "1", n), _ladder(prefix, "0", n)


def _seeded(n: int = 6) -> ScanSeed:
    """A single-namespace seed; ids may be random because nothing compares across
    namespaces."""
    return scan_seed(ids=_ladder(uuid4().hex[:23], "1", n))


# --------------------------------------------------------------------------- #
# The window bound
# --------------------------------------------------------------------------- #


async def test_scan_limit_bounds_the_window(adapter, namespace) -> None:
    seed = _seeded()
    await seed_documents(adapter, namespace.id, seed)

    step = await adapter.scan_documents(namespace.id, scan_limit=2)

    assert [d.id for d in step.documents] == seed.expected[:2]
    assert step.last_scanned == (step.documents[-1].created_at, step.documents[-1].id)
    assert step.exhausted is False


async def test_a_full_window_is_not_yet_exhausted(adapter, namespace) -> None:
    """``exhausted`` means SurrealQL ran short, not "the caller has seen everything".

    A window filled exactly to the bound cannot distinguish "six rows and no
    more" from "six rows and a seventh waiting", so it must report not-exhausted
    and let the next step find the empty tail. Reporting exhaustion here would
    silently truncate every namespace whose size is a multiple of the bound. The
    short window is the other half of the same contract.
    """
    seed = _seeded()
    await seed_documents(adapter, namespace.id, seed)

    exact = await adapter.scan_documents(namespace.id, scan_limit=6)
    assert len(exact.documents) == 6
    assert exact.exhausted is False

    short = await adapter.scan_documents(namespace.id, scan_limit=7)
    assert len(short.documents) == 6
    assert short.exhausted is True


async def test_exhausted_describes_the_raw_window_not_what_survives_a_post_filter(adapter, namespace) -> None:
    """A full window that the caller's post-filter empties is still not exhausted.

    ``exhausted`` derives from ``len(rows) < scan_limit`` — the RAW window — and
    it has to, because it is the walk's only termination signal. The filter here
    matches nothing at all in memory (no row carries that ``source_type``, and no
    ``document`` backs ``occurred_at``) while pushing nothing to SurrealQL, so
    the raw window fills to the bound and the caller's post-filter then rejects
    every row it contains.

    Deriving ``exhausted`` from the surviving subset instead would report ``True``
    here and end the walk at the first window — silently truncating every
    namespace whose leading rows happen not to match.

    **Honest scope.** ``scan_documents`` post-filters nothing, so it has no
    surviving subset to derive the wrong answer *from*; no mutation of this
    method makes the two values differ, and measured, the mutant
    ``exhausted=not rows`` leaves this test green (it is caught by
    :func:`test_a_full_window_is_not_yet_exhausted` instead). What this test
    establishes is the property one tier up: a full window really can post-filter
    to zero on real rows, so a *caller* must not read "no matching documents" as
    exhaustion. It guards the coordinator loop that consumes this method, and a
    future refactor that moved post-filtering inside it — not today's arithmetic.
    """
    seed = _seeded()
    await seed_varied(adapter, namespace.id, seed)

    wire = {
        "$or": [
            {"source_type": {"$eq": "no-such-source-type"}},
            {"occurred_at": {"$gte": "2999-01-01T00:00:00+00:00"}},
        ]
    }
    ast = wire_to_ast(wire)
    step = await adapter.scan_documents(namespace.id, filter_ast=ast, scan_limit=6)

    # Nothing pushed, so the raw window is the whole namespace, filled to the bound.
    assert step.consumed_keys == frozenset()
    assert len(step.documents) == 6
    assert step.exhausted is False

    # …and the caller's post-filter keeps none of it. The two really do diverge.
    matches = compile_python(ast, _documents_compile_context()).predicate
    assert [d for d in step.documents if matches(d)] == []


async def test_scan_limit_below_one_is_rejected_before_anything_is_compiled(adapter, namespace, monkeypatch) -> None:
    """A zero bound would return an empty window that reports neither a resume
    position nor exhaustion — the one pair a walking caller cannot act on.

    The bound is validated *first*, before the filter is compiled, so a bad bound
    surfaces as its own ``ValueError`` rather than being masked by whatever the
    compiler happens to raise on the same call. Asserted by counting compiler
    invocations rather than by reading the source: a later reordering that moved
    the guard below the compile would still raise ``ValueError`` here for a
    filter that compiles cleanly, and only the call count notices.
    """
    calls: list[Any] = []

    def counting_compiler(ast, ctx):
        calls.append(ast)
        raise AssertionError("the filter was compiled despite an invalid scan_limit")

    monkeypatch.setitem(CompilerRegistry._registry, _COMPILER_KEY, counting_compiler)  # noqa: SLF001

    with pytest.raises(ValueError, match="scan_limit"):
        await adapter.scan_documents(
            namespace.id,
            filter_ast=wire_to_ast({"source_type": {"$eq": "report"}}),
            scan_limit=0,
        )

    assert calls == []


# --------------------------------------------------------------------------- #
# The keyset cursor
# --------------------------------------------------------------------------- #


async def test_walk_visits_every_document_exactly_once_in_total_order(adapter, namespace) -> None:
    """``id < $rid`` on a ``RecordID`` agrees with ``ORDER BY id DESC``.

    This was the ticket's highest-risk unknown, and it is the reason this module
    exists separately from the SQLAlchemy pair. On the two SQL stores ``id`` is a
    UUID column and the keyset comparison is the same comparison the ``ORDER BY``
    uses, by construction. Here ``id`` is a ``RecordID`` (``document:<uuid>``):
    the scan resumes with ``id < $rid`` while the statement sorts with
    ``ORDER BY id DESC``, and if SurrealDB ordered record ids by anything other
    than the comparison it uses in a ``WHERE`` — a different collation, the table
    part weighing in, the UUID part compared as text — a resumed walk would skip
    rows or repeat them.

    ``scan_limit=1`` puts a cursor boundary between every pair of rows, including
    between the four that share a ``created_at``, so every resume here is a
    mid-tie resume decided solely by the ``id DESC`` leg. The assertion is
    exact-equality against the seed's single correct enumeration, not a set
    comparison: a wrong-but-complete order has to fail too.
    """
    seed = _seeded()
    await seed_documents(adapter, namespace.id, seed)

    steps = await walk_scan(adapter.scan_documents, namespace.id, scan_limit=1)
    seen = [d.id for step in steps for d in step.documents]

    assert len(seen) == len(set(seen))  # no document served twice
    assert set(seen) == set(seed.expected)  # every document served
    assert seen == seed.expected  # and in one total order across the concatenation
    assert steps[-1].documents == []
    assert steps[-1].last_scanned is None
    assert steps[-1].exhausted is True


async def test_cursor_excludes_its_own_row_and_keeps_its_tie_mates(adapter, namespace) -> None:
    """A mid-tie cursor is strict on its own row and inclusive of the rest of the block.

    The two-clause keyset form has one failure mode in each direction and neither
    raises. If the ``created_at = $ts AND id < $id`` leg were ``<=`` (or the
    first leg ``<=``), the cursor's own row comes back and a walk chaining
    ``last_scanned`` never advances. If the first leg were ``<=`` without the
    id guard — or if the disjunction collapsed to ``created_at < $ts`` — the
    cursor's tie-mates are skipped and a walk silently loses rows. Both are
    asserted, plus the exact remaining suffix.
    """
    seed = _seeded()
    await seed_documents(adapter, namespace.id, seed)

    full = await adapter.scan_documents(namespace.id, scan_limit=10)
    assert [d.id for d in full.documents] == seed.expected

    cursor_doc = next(d for d in full.documents if d.id == seed.tied_ids[0])
    assert as_utc(cursor_doc.created_at) == seed.tie_instant

    step = await adapter.scan_documents(
        namespace.id,
        after=(cursor_doc.created_at, cursor_doc.id),
        scan_limit=10,
    )
    ids = [d.id for d in step.documents]

    assert cursor_doc.id not in ids, "the cursor's own row came back — a resumed walk would never advance"
    assert seed.tied_ids[1] in ids, "the cursor's tie-mate was skipped — a resumed walk would lose rows"
    assert ids == seed.expected[seed.expected.index(cursor_doc.id) + 1 :]


async def test_filtered_walk_puts_a_cursor_and_a_compiled_fragment_in_one_statement(adapter, namespace) -> None:
    """The only place a cursor and a pushdown fragment share a statement.

    ``scan_documents`` merges the compiler's binds into its own ``params`` with a
    bare ``dict.update``, resting on the two families being disjoint by
    construction: the scan names its own ``ns`` / ``lim`` / ``after_created_at`` /
    ``after_id``, the compiler names its own ``f_0`` … ``f_N``. Nothing exercises
    that merge unless both families are present in one call, which happens only
    when a cursor and a compiled fragment appear in the same ``SELECT``. A
    collision would silently overwrite a bind — the namespace scope among them —
    rather than raise, so it has to be caught on rows.

    The filter is a two-leaf disjunction on purpose, so it compiles to two binds
    rather than one, and the walk runs at ``scan_limit=1`` so every step past the
    first carries both families at once.
    """
    seed = _seeded()
    await seed_varied(adapter, namespace.id, seed)

    full = await adapter.scan_documents(namespace.id, scan_limit=10)
    wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
    expected = [d.id for d in full.documents if d.source_type == "report" or d.title == "doc-1"]
    assert 1 < len(expected) < len(full.documents), "the filter must narrow, but not to a single row"

    steps = await walk_scan(
        adapter.scan_documents,
        namespace.id,
        scan_limit=1,
        filter_ast=wire_to_ast(wire),
    )
    seen = [d.id for step in steps for d in step.documents]

    assert len(seen) == len(set(seen))
    assert seen == expected
    assert steps[-1].exhausted is True


async def test_empty_window_reports_exhausted_without_a_position(adapter, namespace) -> None:
    """Both the never-seeded namespace and the tail past the last row."""
    empty = await adapter.scan_documents(namespace.id, scan_limit=5)
    assert empty.documents == []
    assert empty.last_scanned is None
    assert empty.exhausted is True

    seed = _seeded()
    await seed_documents(adapter, namespace.id, seed)
    full = await adapter.scan_documents(namespace.id, scan_limit=10)
    oldest = full.documents[-1]

    tail = await adapter.scan_documents(namespace.id, after=(oldest.created_at, oldest.id), scan_limit=5)
    assert tail.documents == []
    assert tail.last_scanned is None
    assert tail.exhausted is True


# --------------------------------------------------------------------------- #
# The compile split
# --------------------------------------------------------------------------- #


async def test_split_reports_only_the_leaves_surrealql_enforced(adapter, namespace) -> None:
    """A forced-residual AST: one pushable leaf, one that must reach the post-filter.

    ``occurred_at`` is a recall-chunk key with no ``document`` field behind it,
    so it is absent from this store's ``field_mapping`` and
    ``compile_surrealdb`` defers it under ``on_unsupported="split"``. The
    conjunction still pushes its other leaf, so ``consumed_keys`` must name
    ``source_type`` and only ``source_type``.

    Both halves are asserted on rows as well as on the reported set: the pushed
    leaf really did narrow the window, and the deferred one narrowed nothing —
    every row survives a bound that would have excluded all of them had it been
    pushed. Reporting alone would pass against a compiler that pushed
    ``occurred_at`` against a missing field, which on this SCHEMAFULL table reads
    ``NONE`` and returns an empty result set that looks exactly like a
    legitimate no-match.
    """
    seed = _seeded()
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await write_document(
            adapter, namespace.id, doc_id, created_at, source_type="report" if i % 2 == 0 else "library"
        )

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast(
            {"source_type": {"$eq": "report"}, "occurred_at": {"$gte": "2999-01-01T00:00:00+00:00"}}
        ),
        scan_limit=10,
    )

    assert step.consumed_keys == frozenset({"source_type"})
    assert {d.source_type for d in step.documents} == {"report"}
    assert len(step.documents) == 3


class _Shape(NamedTuple):
    """One superset-property parametrization, with what it is expected to exercise.

    ``consumed`` is the exact ``consumed_keys`` this shape is expected to produce,
    measured on this store (see the table below). It is not a convenience field: a
    shape that consumes NOTHING cannot fail the superset comparison at all, so
    which side of that line a shape sits on has to be pinned separately from the
    comparison itself — see the test's docstring on mode B.

    **The exact set rather than a non-empty/empty category, deliberately.** The
    plan specified a category assertion, and a category would be enough for the
    mode-B purpose. This is the stronger form, chosen for cross-module consistency:
    the two sibling scan modules (``tests/unit/test_sqlite_scan_documents.py`` and
    ``tests/unit/storage/backends/sqlite_lance/test_relational_scan_documents.py``)
    landed the exact-set form independently, and three near-identical records with
    three different shapes is exactly the drift this ticket is about. The exact set
    strictly implies the category, so nothing is lost. What it costs: a deliberate
    compiler improvement that starts pushing one more key fails here — which is
    the correct place to notice a pushdown change, and the failure names the shape.

    ``oracle_empty`` marks a shape whose oracle is empty *by construction* on this
    store, so it is asserted empty rather than being exempted from the
    anti-vacuity check.
    """

    wire: dict[str, Any]
    consumed: frozenset[str]
    oracle_empty: bool = False


# The pushdown must never reject a row the full filter would keep. Shapes are
# chosen for the ways a compiler can get that wrong, not for operator coverage:
# the ones wrapping an unpushable leaf in a disjunction or a negation matter
# most, because a match-all placeholder left inside a negation inverts into a
# match-nothing and excludes rows.
#
# ``pushable_exists`` was ``{"source_url": {"$exists": False}}`` and was VACUOUS.
# ``$exists: False`` on a system key is constant-false — system keys are
# documented always-present — so the oracle was empty for any seed and
# ``oracle <= window`` could not fail. It was blind precisely where it looked
# strongest, being one of the few shapes that both consumes a key and returns an
# empty window, i.e. the "the pushdown rejected every row" direction this test
# exists to catch. The replacement asks the same question of a metadata path,
# where absence is a real property of the corpus rather than a tautology.
#
# EVERY figure below was measured on **embedded SurrealDB** (``mode="memory"``,
# the module's own fixture — no container, no compose stack) over
# ``seed_varied``'s 6-row corpus, in this tree on this branch. That provenance is
# stated rather than left implicit: this is an integration lane, so "measured on
# this store" otherwise invites the fair question of whether the lane ran. It did,
# embedded. A remote (``ws://``) instance is NOT covered by these numbers.
#
# | shape               | oracle | window | consumed_keys      |
# | ---                 | ---    | ---    | ---                |
# | pushable_eq         | 3      | 3      | source_type        |
# | pushable_ne         | 3      | 3      | source_type        |
# | pushable_nin        | 3      | 3      | source_type        |
# | pushable_exists     | 4      | 4      | metadata.tier      |  (was 0 / 0)
# | metadata_eq         | 2      | 2      | metadata.tier      |
# | pushable_date       | 5      | 5      | created_at         |
# | unpushable_key      | 0      | 6      | (none)             |
# | or_over_unpushable  | 3      | 6      | (none)             |
# | not_over_pushable   | 3      | 3      | source_type        |
# | not_over_unpushable | 6      | 6      | (none)             |
# | and_of_in_and_not   | 5      | 5      | source_type, title |
_SUPERSET_SHAPES: dict[str, _Shape] = {
    "pushable_eq": _Shape({"source_type": {"$eq": "report"}}, consumed=frozenset({"source_type"})),
    "pushable_ne": _Shape({"source_type": {"$ne": "report"}}, consumed=frozenset({"source_type"})),
    "pushable_nin": _Shape({"source_type": {"$nin": ["report"]}}, consumed=frozenset({"source_type"})),
    "pushable_exists": _Shape({"metadata.tier": {"$exists": False}}, consumed=frozenset({"metadata.tier"})),
    "metadata_eq": _Shape({"metadata.tier": {"$eq": "gold"}}, consumed=frozenset({"metadata.tier"})),
    "pushable_date": _Shape(
        {"created_at": {"$gte": "2026-01-31T12:30:00+00:00"}},
        consumed=frozenset({"created_at"}),
    ),
    # ``occurred_at`` has no ``document`` field behind it, so the oracle drops
    # every row: empty by construction, not by seed.
    "unpushable_key": _Shape(
        {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
        consumed=frozenset(),
        oracle_empty=True,
    ),
    "or_over_unpushable": _Shape(
        {
            "$or": [
                {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
                {"source_type": {"$eq": "report"}},
            ]
        },
        consumed=frozenset(),
    ),
    "not_over_pushable": _Shape({"$not": {"source_type": {"$eq": "report"}}}, consumed=frozenset({"source_type"})),
    "not_over_unpushable": _Shape(
        {"$not": {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}}},
        consumed=frozenset(),
    ),
    "and_of_in_and_not": _Shape(
        {
            "$and": [
                {"source_type": {"$in": ["report", "library"]}},
                {"$not": {"title": {"$eq": "doc-0"}}},
            ]
        },
        consumed=frozenset({"source_type", "title"}),
    ),
}


@pytest.mark.parametrize("shape", _SUPERSET_SHAPES.values(), ids=_SUPERSET_SHAPES.keys())
async def test_pushdown_never_rejects_a_row_the_full_filter_would_keep(adapter, namespace, shape: _Shape) -> None:
    """The superset property the resume contract depends on.

    Resuming past the rows a pushdown rejected is sound only because a rejected
    row could not have satisfied the full filter either. The ``scan_documents``
    docstring names that as an assumption about the *compiler*; this checks the
    consequence where it actually lands, by comparing the scan's window against
    the in-process ``compile_python`` evaluation of the same AST over the same
    corpus. If it ever fails, a walk is silently and permanently dropping
    documents — a post-filter can only narrow, never recover a row the window
    never returned.

    **Scope, counted from a run on this store rather than asserted.** Eleven
    shapes are parametrized, but they do not all exercise the property, and the
    docstring this replaced claimed they did. Measured ``consumed_keys`` per
    shape (embedded ``memory://``, see the table above the shape dict): **eight**
    put a fragment into the SurrealQL ``WHERE`` and are therefore capable of
    failing the comparison; **three** — ``unpushable_key``, ``or_over_unpushable``,
    ``not_over_unpushable`` — consume nothing, because ``compile_surrealdb``'s
    all-or-nothing gate defers the whole node.

    **Those three are structurally unfailable for the superset property, and no
    seed fixes that.** When nothing is pushed, the window IS the namespace, and
    ``oracle`` is computed by filtering that same window — so ``oracle <= window``
    holds for every possible oracle value. What they still pin is their
    ``consumed_keys``: asserting it is empty catches a compile-context or
    ``field_mapping`` edit that starts pushing one of them, which is the change
    that would make the deferral silently stop happening. That is the whole of
    their value here; the superset property for deferred subtrees belongs to the
    compilers and to the forced-residual conformance corpus.

    Note the mode-B discriminator is ``consumed_keys``, **not**
    ``window == corpus``. Those are not the same test: a pushdown bug that wrongly
    *rejects* rows shrinks the window below the corpus while still consuming a
    key, and ``oracle <= window`` then does fire — which is the whole point.

    **Two anti-vacuity checks, because they cover different modes.**
    ``assert oracle`` catches an oracle that went empty — an empty set is a
    subset of anything. Mutation-verified, and reverted afterwards: hollowing out
    ``seed_varied`` so every row is written ``source_type="library"`` fails
    **2 of these 11 parametrizations** (``pushable_eq`` and
    ``or_over_unpushable``, whose oracles both drop from 3 rows to 0) on the
    ``assert oracle`` line rather than passing green. The ``consumed_keys``
    assertion catches the other mode, above. Neither subsumes the other, and
    ``assert oracle`` alone would have left all three of the deferring shapes
    looking tested.
    """
    seed = _seeded()
    await seed_varied(adapter, namespace.id, seed)
    ast = wire_to_ast(shape.wire)

    step = await adapter.scan_documents(namespace.id, filter_ast=ast, scan_limit=100)
    # Precondition: the comparison below is only meaningful if this one window
    # covered the whole namespace. Without it, growing the corpus past the bound
    # would fail the test for a reason that has nothing to do with the pushdown.
    assert step.exhausted is True

    all_docs = (await adapter.scan_documents(namespace.id, scan_limit=100)).documents
    matches = compile_python(ast, _documents_compile_context()).predicate
    oracle = {d.id for d in all_docs if matches(d)}

    # Anti-vacuity, mode A: an empty oracle is a subset of anything.
    if shape.oracle_empty:
        assert not oracle, "declared empty-by-construction, but the oracle kept rows — reclassify the shape"
    else:
        assert oracle, "the oracle kept no rows, so the comparison below cannot fail — the corpus stopped feeding it"

    # Anti-vacuity, mode B: a shape that pushes nothing cannot reject anything, so
    # the comparison is unfailable for it. Pinning the EXACT measured set (not
    # merely non-empty vs empty) also catches a shape drifting across that line,
    # and matches the two sibling scan modules — see ``_Shape``.
    assert step.consumed_keys == shape.consumed

    assert oracle <= {d.id for d in step.documents}


# --------------------------------------------------------------------------- #
# What the position means
# --------------------------------------------------------------------------- #


async def test_last_scanned_is_the_final_raw_row_not_the_last_match(adapter, namespace) -> None:
    """Resume from the last row SCANNED, not from the last row that matches.

    The seed is arranged so the two genuinely differ — otherwise the assertion is
    vacuous. The window deliberately ENDS on a row the caller's post-filter will
    reject: the filter is an ``$or`` mixing a pushable leaf with an unbacked one,
    which ``compile_surrealdb``'s all-or-nothing gate defers wholesale rather
    than pushing half a disjunction, so SurrealQL narrows nothing and the oldest
    row — a ``library`` row that does not satisfy the filter — is the final row
    of the raw window.

    A walk that resumed from the last *matching* row instead would re-scan the
    rejected gap on every step, and when a whole window is rejected there is no
    matching row to resume from at all, so such a walk cannot advance past a run
    of non-matching rows longer than one window. Taking the position from the raw
    window is what lets ``exhausted`` be the only termination signal.
    """
    newest, middle, oldest = (uuid4() for _ in range(3))
    base = WHOLE_SECOND
    await write_document(adapter, namespace.id, newest, base + timedelta(seconds=2), source_type="report")
    await write_document(adapter, namespace.id, middle, base + timedelta(seconds=1), source_type="report")
    await write_document(adapter, namespace.id, oldest, base, source_type="library")

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast(
            {
                "$or": [
                    {"source_type": {"$eq": "report"}},
                    {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
                ]
            }
        ),
        scan_limit=10,
    )

    # Nothing was pushed, so the raw window is the whole namespace and its last
    # row is the one the post-filter will drop.
    assert step.consumed_keys == frozenset()
    assert [d.id for d in step.documents] == [newest, middle, oldest]
    assert step.documents[-1].source_type == "library"

    last_row = step.documents[-1]
    assert step.last_scanned == (last_row.created_at, last_row.id)

    last_match = step.documents[1]
    assert step.last_scanned != (last_match.created_at, last_match.id)


async def test_last_scanned_carries_a_uuid_not_a_record_id(adapter, namespace) -> None:
    """``DocumentScanKey`` declares ``tuple[datetime, UUID]``, and the id half is
    the easy one to get wrong.

    **The derivation is SPLIT, and that split is the contract this pins.**
    :func:`~khora.storage.backends.surrealdb.relational._scan_key` takes
    ``created_at`` off the **raw row** — the row already carries a real
    ``datetime`` (``TYPE datetime``), so parsing it would only introduce the
    ``_parse_dt`` -> ``None`` -> ``datetime.now(UTC)`` coalesce that puts the
    cursor above every row — while the ``id`` half is **strictly converted**,
    because the raw ``id`` is a ``RecordID`` and the declared key type is a
    ``UUID``. Neither half may be taken the other's way. An earlier revision
    built the whole key from the converted ``Document``, which is what
    reintroduces the coalesce.

    **What makes the id half worth its own test even so.** A ``RecordID`` seated
    in a ``DocumentScanKey`` round-trips back into *this same store* — it is the
    shape the write path binds — so a type error there is not loud on every path;
    it is silently green on some, and only a direct type assertion states the
    contract.

    Scope note, re-measured on this branch rather than inherited. The mutant that
    derives BOTH halves from the raw row wholesale
    (``return (row["created_at"], row["id"])``) does not stay green here: it fails
    **7 of the 33 tests in this module**, the walks dying inside the SDK at
    ``surrealdb/connections/async_embedded.py:119`` with ``ValueError: Failed to
    decode CBOR request`` when the ``RecordID`` is bound back in as a cursor
    operand. So this test is a *clearer* statement of the contract, not the only
    thing standing between a ``RecordID`` key and a green suite — keep it for the
    former reason.

    Two prior versions of this count were wrong and the corrections are recorded
    so nobody re-derives them: the original said "five" (never measured, and it
    described a whole-``Document`` derivation the code no longer uses), and the
    "6 of 30" that replaced it was measured against the module BEFORE this
    branch added three tests. The seventh failure is
    :func:`test_a_non_uuid_record_id_raises_instead_of_inventing_a_position`,
    which the wholesale form fails for a different reason than the walks do — it
    seats the ``RecordID`` in the key without raising at all. Re-run the mutant
    rather than trusting this number; the denominator moves whenever the module
    grows.
    """
    seed = _seeded()
    await seed_varied(adapter, namespace.id, seed)

    step = await adapter.scan_documents(namespace.id, scan_limit=3)
    assert step.last_scanned is not None
    created_at, doc_id = step.last_scanned
    assert isinstance(doc_id, UUID)
    assert isinstance(created_at, datetime)

    filtered = await adapter.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast({"source_type": {"$eq": "report"}}),
        scan_limit=2,
    )
    assert filtered.last_scanned is not None
    assert isinstance(filtered.last_scanned[1], UUID)


async def test_a_hyphenated_metadata_key_in_a_deferred_subtree_does_not_raise(adapter, namespace) -> None:
    """A hyphenated key inside a wholesale-deferred ``$or`` also just works.

    **Position-independence is now the whole point, and this test is what shows
    the two positions agree.** Before §8 the mapping was sibling-dependent: the
    injection guard fired only when the emit walk *reached* the offending leaf, so
    the same key raised in conjunctive position while working inside an ``$or``
    that the all-or-nothing gate deferred for an unrelated reason (here the
    unbacked ``occurred_at`` sibling). §8 removed that split by making the leaf
    non-consumable, so both positions now take the residual route. Its conjunctive
    counterpart is
    :func:`test_a_hyphenated_metadata_key_matches_the_oracle_and_raw_sqlite`.

    **Scope, stated rather than implied: row-level parity is checked in the
    CONJUNCTIVE position only.** That is the position §8 rules on, and it is the
    one that changed. This ``$or`` test asserts only that the filter does not
    raise and consumes nothing — it deliberately does NOT compare rows against an
    oracle or against another store, so do not read the pair as establishing
    parity "in every position". Position coverage at the *compile* level (three
    positions × four stores, exact ``consumed_keys``) lives in
    :mod:`tests.unit.filter.test_documents_compile_contexts`.

    This one stays a deliberately weak assertion for a structural reason: the gate
    defers the whole node for the ``occurred_at`` sibling regardless of the
    metadata leaf, so it cannot distinguish "deferred because unrenderable" from
    "deferred because of the sibling". Measured, that deferral also costs
    SurrealDB the pushable sibling — in ``$or`` position it consumes nothing at
    all and scans the whole namespace. The conjunctive test is the one with teeth.
    """
    seed = _seeded()
    await seed_varied(adapter, namespace.id, seed)

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast(
            {
                "$or": [
                    {"metadata.due-date": {"$eq": "2026-01-01"}},
                    {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
                ]
            }
        ),
        scan_limit=50,
    )

    # Deferred wholesale: nothing consumed, nothing narrowed, no exception.
    assert step.consumed_keys == frozenset()
    assert len(step.documents) == 6


# --------------------------------------------------------------------------- #
# The non-filter narrowing legs
# --------------------------------------------------------------------------- #


async def test_status_and_updated_before_narrow_the_window(adapter, namespace) -> None:
    """``updated_before`` binds a ``datetime`` OBJECT, and that is why it narrows.

    This store's ``updated_at`` is ``TYPE datetime``. A stringified operand does
    not compare against it at all — it matches no row, so a walk reports itself
    exhausted at the first step and the caller sees an empty namespace rather
    than an error. Measured on an in-memory instance over this exact corpus: no
    bound = 6 rows, ``datetime`` bind = 6 rows, ``.isoformat()`` bind = 0 rows.

    That is why the assertion below is on *which* rows came back, not merely on
    a count: the string form's 0 rows and a correct 4 are both "narrower than 6",
    and only an exact row set separates a working bound from a broken one.

    ``scan_documents`` therefore diverges deliberately from ``list_documents`` on
    this same adapter, which binds ``.isoformat()`` and is defective for it. That
    defect is tracked separately and is deliberately not touched or tested here.
    """
    seed = _seeded()
    cutoff = seed.tie_instant + timedelta(hours=1)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await write_document(
            adapter,
            namespace.id,
            doc_id,
            created_at,
            status=DocumentStatus.COMPLETED if i % 2 == 0 else DocumentStatus.PENDING,
            updated_at=cutoff - timedelta(minutes=1) if i < 4 else cutoff + timedelta(minutes=1),
        )

    by_status = await adapter.scan_documents(namespace.id, status=DocumentStatus.COMPLETED.value, scan_limit=10)
    assert {d.id for d in by_status.documents} == {doc_id for i, (doc_id, _) in enumerate(seed.writes) if i % 2 == 0}

    unbounded = await adapter.scan_documents(namespace.id, scan_limit=10)
    assert len(unbounded.documents) == 6

    by_updated = await adapter.scan_documents(namespace.id, updated_before=cutoff, scan_limit=10)
    assert {d.id for d in by_updated.documents} == {doc_id for i, (doc_id, _) in enumerate(seed.writes) if i < 4}


async def test_every_narrowing_leg_composes_in_one_statement(adapter, namespace) -> None:
    """Cursor + compiled fragment + ``status`` + ``updated_before``, all at once.

    Each leg has its own test above; none of those puts more than two of them in
    the same ``WHERE``. This is the shape that a grouping or bind-merge mistake
    actually reaches in production, and the one where a mis-composed conjunct
    hides: with four conditions joined by ``AND``, a predicate absorbed into a
    disjunct or a bind quietly overwritten still returns *some* plausible subset.

    The expectation is computed from the rows the store returns rather than from
    the seeding loop's counter, and each leg is asserted to be doing work — a
    leg that narrowed nothing would make its own conjunct untested while the
    overall assertion still passed.
    """
    seed = _seeded()
    cutoff = seed.tie_instant + timedelta(hours=1)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await write_document(
            adapter,
            namespace.id,
            doc_id,
            created_at,
            title=f"doc-{i}",
            source_type="report" if i % 2 == 0 else "library",
            status=DocumentStatus.COMPLETED if i < 5 else DocumentStatus.PENDING,
            updated_at=cutoff - timedelta(minutes=1) if i < 4 else cutoff + timedelta(minutes=1),
        )

    everything = await adapter.scan_documents(namespace.id, scan_limit=50)
    assert len(everything.documents) == 6

    # Resume from inside the tie block, so the keyset's right disjunct is live.
    cursor_doc = next(d for d in everything.documents if d.id == seed.tied_ids[0])
    after_cursor = everything.documents[everything.documents.index(cursor_doc) + 1 :]

    def surviving(docs: list[Any]) -> list[UUID]:
        return [
            d.id
            for d in docs
            if d.source_type == "report" and d.status == DocumentStatus.COMPLETED and as_utc(d.updated_at) < cutoff
        ]

    expected = surviving(after_cursor)

    # Every leg must actually remove something, or its conjunct is untested here.
    assert expected, "the four-way conjunction must keep at least one row"
    assert len(after_cursor) < len(everything.documents), "the cursor narrowed nothing"
    assert len(expected) < len(after_cursor), "the filter/status/updated_before legs narrowed nothing"

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast({"source_type": {"$eq": "report"}}),
        status=DocumentStatus.COMPLETED.value,
        updated_before=cutoff,
        after=(cursor_doc.created_at, cursor_doc.id),
        scan_limit=50,
    )

    assert [d.id for d in step.documents] == expected
    assert step.consumed_keys == frozenset({"source_type"})


# --------------------------------------------------------------------------- #
# Cross-store parity on a hyphenated metadata key (ticket §8)
# --------------------------------------------------------------------------- #


@pytest.fixture
async def sqlite_backend():
    """An in-memory raw-SQLite store, for the two-store half of the §8 parity claim."""
    store = SQLiteRelationalBackend(":memory:")
    await store.connect()
    try:
        yield store
    finally:
        await store.disconnect()


# One row per outcome, so the filter's two leaves are each load-bearing and each
# in a different direction. ``discriminator`` is the row that separates a real
# parity check from a vacuous one: the metadata leaf rejects it while the
# ``source_type`` sibling admits it, so it is in SurrealDB's raw window and must
# be removed by the residual.
_PARITY_ROWS: dict[str, dict[str, Any]] = {
    "match": {"source_type": "library", "metadata": {"foo-bar": "x"}},
    "discriminator": {"source_type": "library", "metadata": {"foo-bar": "y"}},
    "wrong_source_type": {"source_type": "report", "metadata": {"foo-bar": "x"}},
    "metadata_absent": {"source_type": "library", "metadata": {}},
}
_PARITY_WIRE: dict[str, Any] = {"metadata.foo-bar": {"$eq": "x"}, "source_type": {"$eq": "library"}}


async def _seed_parity(store: Any, namespace_id: UUID) -> dict[str, UUID]:
    """Write :data:`_PARITY_ROWS` to ``store``; return name -> id."""
    ids: dict[str, UUID] = {}
    for offset, (name, fields) in enumerate(_PARITY_ROWS.items()):
        doc_id = uuid4()
        ids[name] = doc_id
        await write_document(store, namespace_id, doc_id, WHOLE_SECOND + timedelta(seconds=offset), **fields)
    return ids


async def test_a_hyphenated_metadata_key_matches_the_oracle_and_raw_sqlite(adapter, namespace, sqlite_backend) -> None:
    """A hyphenated metadata key returns the SAME ROWS on both stores (ticket §8).

    ``metadata.foo-bar`` is legal, common JSON, and SurrealQL has no bind form for
    a field *name*, so the leaf is simply unpushable on SurrealDB. §8 ruled that
    this makes it an **unpushable leaf, not an error**: it defers to the caller's
    ``compile_python`` residual like any other, and the pre-§8 behaviour this test
    replaces — ``RecallFilterUnsupportedError`` in conjunctive position, working
    rows inside a deferred ``$or`` — is gone rather than relabelled.

    **What "parity" does and does not mean, because the tempting assertion is
    wrong.** The two stores do NOT agree step-for-step and never will: measured in
    this tree, raw-sqlite / sqlite_lance / postgresql all compile this filter with
    ``consumed_keys == {"metadata.foo-bar", "source_type"}`` while SurrealDB
    reports ``{"source_type"}``. So SurrealDB's raw window is a **superset** of
    raw-sqlite's, and ``assert surreal_rows == sqlite_rows`` on the raw windows
    would be asserting something false. Parity holds one tier up — *after the
    residual the caller is contractually required to apply* — and that is what is
    asserted here. The ``consumed_keys`` asymmetry is asserted too, in both
    directions, because it is real, permanent, and the thing a future reader is
    most likely to "fix".

    **Anti-vacuity, in the shape §4 of this ticket demands elsewhere.** If the
    residual removed nothing, SurrealDB's raw window would already equal the
    oracle and this test would pass without exercising the deferral at all — the
    same defect as the ``pushable_exists`` parametrization §4 replaced. So the
    seed carries ``discriminator`` (``foo-bar = "y"``, ``source_type =
    "library"``): admitted by the pushed leaf, rejected by the deferred one. Both
    facts are asserted rather than assumed — the oracle is non-empty, and the raw
    window is strictly larger than it.
    """
    surreal_ids = await _seed_parity(adapter, namespace.id)
    sqlite_ns = await sqlite_backend.create_namespace(
        MemoryNamespace(id=(sid := uuid4()), namespace_id=sid, tenancy_mode=TenancyMode.SHARED)
    )
    sqlite_ids = await _seed_parity(sqlite_backend, sqlite_ns.id)

    ast = wire_to_ast(_PARITY_WIRE)
    surreal_step = await adapter.scan_documents(namespace.id, filter_ast=ast, scan_limit=50)
    sqlite_step = await sqlite_backend.scan_documents(sqlite_ns.id, filter_ast=ast, scan_limit=50)

    # The asymmetry, both directions. SurrealDB defers the unrenderable leaf…
    assert "metadata.foo-bar" not in surreal_step.consumed_keys
    assert surreal_step.consumed_keys == frozenset({"source_type"})
    # …while raw-sqlite pushes it into SQL via JSON1.
    assert "metadata.foo-bar" in sqlite_step.consumed_keys
    assert _sqlite_documents_context().schema_capabilities.sqlite_json1, (
        "raw-sqlite deferred the metadata leaf too — this build lacks JSON1, so the "
        "asymmetry this test documents is not the one being exercised"
    )

    # The residual every caller of scan_documents is required to apply.
    matches = compile_python(ast, _documents_compile_context()).predicate
    surreal_raw = {d.id for d in surreal_step.documents}
    surreal_rows = {d.id for d in surreal_step.documents if matches(d)}
    sqlite_rows = {d.id for d in sqlite_step.documents if matches(d)}

    # Anti-vacuity FIRST, so a hollowed-out seed reports the property it broke
    # rather than a downstream bookkeeping mismatch.
    assert surreal_rows, "the oracle kept no row — the comparison below cannot fail"
    assert surreal_raw > surreal_rows, "the residual removed nothing — the deferral is not being exercised"
    assert surreal_rows == {surreal_ids["match"]}
    assert surreal_ids["discriminator"] in surreal_raw

    # The parity itself, named by row rather than by count so a wrong-but-equal-sized
    # answer fails too.
    assert sqlite_rows == {sqlite_ids["match"]}
    assert {n for n, i in surreal_ids.items() if i in surreal_rows} == {
        n for n, i in sqlite_ids.items() if i in sqlite_rows
    }


async def test_an_unrelated_compile_error_still_propagates(adapter, namespace, monkeypatch) -> None:
    """There is NO mapping layer, so a compiler fault escapes as itself.

    **This test was rewritten because §8 made its previous subject disappear, and
    the rewrite is the point.** It used to assert that the ``try/except`` in
    ``scan_documents`` was *narrow* — that it mapped one exception subclass and let
    the base class through. §8 deleted that ``try/except`` entirely, which left the
    test passing for a reason its docstring no longer described: nothing catches
    anything, so of course the error propagates. A test that passes trivially while
    documenting a mechanism that no longer exists is precisely the stale artifact
    this ticket was opened over, so it asserts the post-§8 property instead.

    That property is still worth pinning, and it is not vacuous. Re-adding any
    ``except CompileError: raise RecallFilterUnsupportedError`` here would relabel
    every genuine compiler bug as a caller-input problem and bury it behind a
    4xx-shaped error — the inversion §8 removed. This fails against that
    regression, because the two classes are siblings under ``KhoraError`` rather
    than parent and child, so a mapped error would not satisfy ``pytest.raises``
    below. The explicit ``not isinstance`` assertion states the direction that
    matters rather than leaving it implicit in the match.

    Nothing in the filter language provokes a ``CompileError`` on this path any
    more (§8 routes the one caller-reachable guard to the residual instead), so the
    only way to reach the branch is to inject a fault through the registry — which
    is also how the real lookup finds its compiler.
    """

    def broken_compiler(ast, ctx):
        raise CompileError("internal compiler invariant violated while emitting a node")

    monkeypatch.setitem(CompilerRegistry._registry, _COMPILER_KEY, broken_compiler)  # noqa: SLF001

    with pytest.raises(CompileError, match="internal compiler invariant") as excinfo:
        await adapter.scan_documents(
            namespace.id,
            filter_ast=wire_to_ast({"source_type": {"$eq": "report"}}),
            scan_limit=10,
        )

    # The direction that matters: a compiler fault must NOT arrive dressed as a
    # caller-input problem. Siblings under ``KhoraError``, not parent and child.
    assert not isinstance(excinfo.value, RecallFilterUnsupportedError)


# --------------------------------------------------------------------------- #
# The strict cursor extractor
# --------------------------------------------------------------------------- #


async def test_a_non_uuid_record_id_raises_instead_of_inventing_a_position(adapter, namespace) -> None:
    """A foreign-shaped record id fails the scan loudly rather than yielding a UUID5.

    ``DocumentScanKey`` is ``tuple[datetime, UUID]``, and SurrealDB record ids are
    a tagged union — ``document:⟨uuid⟩`` is what this store writes, but
    ``document:not-a-uuid`` is a legal row that a user can create directly, and
    SurrealDB *is* a backend people write to directly. The obvious conversion
    routes through ``_helpers._parse_uuid``, whose documented
    ``uuid5(NAMESPACE_URL, raw)`` fallback returns a perfectly well-formed UUID
    for such an id: measured, ``_parse_uuid("not-a-uuid")`` yields
    ``6892d5c5-a618-5b00-a8f6-5e57b68e64b9``. That value is a *fictional*
    position — it corresponds to no row, compares against nothing real, and
    round-trips into this same store without complaint, so a walk resuming from
    it silently reads the wrong window. ``_scan_key`` therefore converts strictly
    and raises.

    The row is seeded with ``CREATE type::thing('document', $sid)`` because
    ``create_document`` cannot produce this shape at all — it routes every id
    through ``_record_id`` — which is exactly why the hazard is latent rather
    than reachable through khora's own writes.

    **What this raise does and does not cover — both halves, because it is
    tempting to over-claim.** ``_scan_key`` is called on the window's **LAST row
    only**, the one row a cursor is ever built from. So a foreign-shaped id
    anywhere *else* in the table is untouched by this guard, and it still
    desynchronises the resume predicate ``id < $after_id`` from
    ``ORDER BY id DESC`` — the record-id variants order relative to one another
    rather than by content — with no error and no trip. That residue is why the
    record-id homogeneity precondition stays documented on ``scan_documents``
    rather than being marked closed. This test pins the cursor half; the
    homogeneity half remains a precondition, not an enforced invariant.
    """
    seed = _seeded()
    await seed_documents(adapter, namespace.id, seed)

    # Oldest instant in the corpus, so this row sorts LAST under
    # ``ORDER BY created_at DESC`` and is therefore the row the cursor is built
    # from.
    await adapter._conn.query(  # noqa: SLF001
        "CREATE type::thing('document', $sid) CONTENT { namespace_id: $ns, content: 'x', checksum: 'foreign-id', "
        "created_at: $ca, updated_at: $ca, status: 'pending', source_type: 'library' }",
        {"sid": "not-a-uuid", "ns": str(namespace.id), "ca": seed.tie_instant - timedelta(hours=1)},
    )

    with pytest.raises(ValueError, match="not a UUID"):
        await adapter.scan_documents(namespace.id, scan_limit=50)

    # The message names the offending id and refuses to substitute a position; it
    # must not have quietly returned the UUID5 instead.
    with pytest.raises(ValueError) as excinfo:
        await adapter.scan_documents(namespace.id, scan_limit=50)
    assert "6892d5c5-a618-5b00-a8f6-5e57b68e64b9" not in str(excinfo.value)

    # A window that stops SHORT of the bad row still works — the guard is about
    # the cursor, not about the table containing such a row.
    ok = await adapter.scan_documents(namespace.id, scan_limit=2)
    assert [d.id for d in ok.documents] == seed.expected[:2]


# --------------------------------------------------------------------------- #
# The bind-merge guard
# --------------------------------------------------------------------------- #


async def test_a_compiled_bind_colliding_with_a_scan_bind_is_rejected(adapter, namespace, monkeypatch) -> None:
    """A compiled bind named ``ns`` would replace the tenant scope. It must raise.

    ``scan_documents`` merges the compiler's binds over its own with
    ``params.update(compiled.params)``, so the compiled side WINS. An overlapping
    name is therefore not a clash but a *silent substitution*: a compiled bind
    called ``ns`` replaces the value in ``namespace_id = $ns`` and the statement
    still executes normally, so a walk carries on over another tenant's rows with
    no error and nothing in the logs.

    **Verified by execution against the pre-guard behaviour.** With the
    ``collisions`` check deleted, this test and its reserved-name sibling are the
    only two of the 33 in this module that fail — so they do pin the guard rather
    than passing for an unrelated reason.

    On the leak's magnitude, stated carefully because the number depends on the
    stub's own predicate and an earlier version of this docstring conflated two
    runs. Measured on a two-namespace corpus of 6 rows each, guard deleted:

    * with **this** stub (``(title = $f_0)``, and exactly one row per namespace
      carries ``title = 'doc-0'``) the scan of namespace A returns **1 row, and it
      is namespace B's** — 0 of A's own;
    * with a match-all fragment in place of that predicate, it returns **6 rows,
      all 6 of them namespace B's** — again 0 of A's.

    So the row COUNT tracks the fragment, but the tenant substitution is **total
    either way**: the scope predicate is not weakened, it is replaced, and A
    contributes nothing. Both runs came back with a populated ``last_scanned`` and
    ``exhausted=True``. A caller walking that has no signal of any kind that it is
    reading the wrong tenant.

    **What this guard is, stated exactly, because the tempting framing is
    false.** It is a tripwire against a compiler that **hand-writes** bind names
    outside the ``{param_namespace}_{counter}`` convention. It is *not* a fix for
    a reachable configuration: ``compile_surrealdb._bind`` has a single
    assignment site and names every bind ``f"{param_namespace}_{counter}"``, so
    every compiled bind ends in ``_<digits>`` and no reserved scan bind has that
    shape. No value of ``param_namespace`` can collide — ``"ns"`` yields
    ``ns_0``, not ``ns``. Do not read this test as evidence that the knob is
    dangerous; it is evidence that the convention is enforced rather than
    assumed, which is the same posture the fragment parentheses one line earlier
    already took.

    The stub keeps one bind in the legitimate ``f_N`` family alongside the
    colliding one, so what is rejected is the collision and not merely "a
    compiler returned an unexpected bind set".
    """
    seed = _seeded()
    await seed_varied(adapter, namespace.id, seed)
    other = await _make_namespace(adapter)
    await seed_varied(adapter, other.id, _seeded())

    def colliding_compiler(ast, ctx):
        return CompiledFilter(
            predicate="(title = $f_0)",
            params={"f_0": "doc-0", "ns": str(other.id)},
            consumed_keys=frozenset({"title"}),
            consumed_slice_hash="bind-collision-tripwire",
        )

    monkeypatch.setitem(CompilerRegistry._registry, _COMPILER_KEY, colliding_compiler)  # noqa: SLF001

    with pytest.raises(RuntimeError) as excinfo:
        await adapter.scan_documents(
            namespace.id,
            filter_ast=wire_to_ast({"title": {"$eq": "doc-0"}}),
            scan_limit=50,
        )

    message = str(excinfo.value)
    assert "ns" in message, "the guard must name the colliding key"
    # Names only, never values: the binds carry document content and user filter
    # values, and the foreign namespace id is exactly what must not be echoed.
    assert str(other.id) not in message
    assert "doc-0" not in message
    # And it is not presented as a filter-capability outcome — a cross-tenant
    # tripwire must not look recoverable to a caller catching the public error.
    assert not isinstance(excinfo.value, RecallFilterUnsupportedError)
    assert not isinstance(excinfo.value, CompileError)


async def test_a_reserved_bind_is_rejected_even_when_this_step_does_not_use_it(adapter, namespace, monkeypatch) -> None:
    """``after_id`` is rejected with ``after=None`` — the case a live-key check misses.

    The guard checks the compiled binds against a reserved NAME SET
    (``_SCAN_BIND_NAMES``) rather than against the binds this particular call
    happens to have built. That choice is the whole point of this test, and it is
    the one place the two forms differ observably.

    ``after_id`` is only bound on a *resumed* step. Against a compiler that
    hand-writes a bare ``{"after_id": ...}``, the live-key form
    (``params.keys() & compiled.params.keys()``) finds nothing on step one —
    the key is simply absent from ``params`` — and fires only once a walk
    resumes. So the same misbehaving compiler would be accepted or rejected
    depending on which step of a walk you happened to be on, and a first-step
    smoke test would pass. The reserved-name form rejects the *configuration*,
    step-independently.

    This test would pass against the narrower live-key form only by accident of
    never reaching step two, which is why it pins the un-resumed call explicitly:
    ``after`` is ``None`` here, deliberately.
    """
    seed = _seeded()
    await seed_varied(adapter, namespace.id, seed)

    def hand_written_bind_compiler(ast, ctx):
        return CompiledFilter(
            predicate="(title = $after_id)",
            params={"after_id": "doc-0"},
            consumed_keys=frozenset({"title"}),
            consumed_slice_hash="reserved-name-tripwire",
        )

    monkeypatch.setitem(CompilerRegistry._registry, _COMPILER_KEY, hand_written_bind_compiler)  # noqa: SLF001

    with pytest.raises(RuntimeError, match="after_id"):
        await adapter.scan_documents(
            namespace.id,
            filter_ast=wire_to_ast({"title": {"$eq": "doc-0"}}),
            after=None,  # explicit: this step builds no ``after_id`` bind of its own
            scan_limit=50,
        )


# --------------------------------------------------------------------------- #
# Namespace isolation and the two ungrouped-OR tripwires
# --------------------------------------------------------------------------- #
#
# Everything above runs in a single-namespace fixture, so no assertion up there
# can notice a scan that ignores its namespace scope. The three tests below are
# the ones that make the mutants fail; each records the mutation that was run
# against it and what it measured.


async def test_scan_never_returns_another_namespaces_rows(adapter, namespace) -> None:
    """A filtered walk over one namespace must not see a byte of the other.

    The second namespace is seeded with the SAME varied corpus, so every row in
    it matches the same filter — if the namespace predicate is dropped (or stops
    AND-composing with the fragment), the foreign rows are not merely reachable,
    they are guaranteed hits. Walked at ``scan_limit=1`` so the keyset predicate
    is exercised across pages too: the cursor is namespace-blind on its own, and
    only the scope predicate keeps a resume inside its tenant.

    **Mutation-verified, and reverted afterwards.** Neutralising the namespace
    predicate — ``conditions = ["namespace_id = $ns"]`` becoming
    ``conditions = ["true"]``, the faithful SurrealQL form of "delete the
    scope", since an empty list would emit ``WHERE`` with no operand and fail as
    a syntax error rather than as wrong rows — fails this test. Measured: the
    filtered walk returns **8 rows instead of 4**, four of them the other
    tenant's, and the unfiltered read below returns **12 instead of 6**. The two
    grouping tripwires fail on the same mutant.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    await seed_varied(adapter, namespace.id, scan_seed(ids=scanned_ids))
    other = await _make_namespace(adapter)
    await seed_varied(adapter, other.id, scan_seed(ids=foreign_ids))

    wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
    steps = await walk_scan(adapter.scan_documents, namespace.id, scan_limit=1, filter_ast=wire_to_ast(wire))
    seen = [d for step in steps for d in step.documents]

    assert seen, "the filter must match rows in the scanned namespace for this test to bite"
    assert all(d.namespace_id == namespace.id for d in seen)
    assert set(foreign_ids).isdisjoint({d.id for d in seen})

    unfiltered = await adapter.scan_documents(namespace.id, scan_limit=50)
    assert len(unfiltered.documents) == 6
    assert all(d.namespace_id == namespace.id for d in unfiltered.documents)


async def test_ungrouped_or_fragment_cannot_absorb_the_namespace_scope(adapter, namespace, monkeypatch) -> None:
    """The parentheses around the spliced fragment are load-bearing.

    ``scan_documents`` joins its conditions with ``AND`` and wraps the compiled
    fragment in parentheses at the splice. Ungrouped, ``AND`` binds tighter than
    ``OR``, so ``namespace_id = $ns AND a = $x OR b = $y`` parses as
    ``(namespace_id = $ns AND a = $x) OR (b = $y)`` — the right disjunct is
    unscoped and returns every tenant's rows. It fails as somebody else's data,
    not as an error.

    **Read the scope of this test narrowly: it is the store's half only.** No
    compiler emits an ungrouped fragment today (``compile_surrealdb`` self-groups
    its boolean nodes), so no compiled-filter test can reach this; the registered
    compiler is therefore replaced with one that emits the bare ungrouped shape,
    and the real ``scan_documents`` is called. Both namespaces' rows satisfy the
    right disjunct, so an absorbed scope predicate yields foreign rows
    deterministically rather than by luck.

    What that monkeypatch costs is precisely the realistic regression: a *real*
    compiler that started emitting ungrouped output would leave this test green,
    because the fake one is installed over it. So this test pins "**IF** an
    ungrouped fragment ever arrives, the store's parentheses contain it" and
    nothing more. The complementary half — "the compilers never emit one" — is
    :mod:`tests.unit.filter.test_fragment_splice_safety`, which drives the real
    ``compile_surrealdb`` and ``compile_lance`` over the conformance corpus. Both
    are needed; neither subsumes the other, and this one is not the one that
    catches a compiler regression.

    **Mutation-verified, and reverted afterwards.** Removing the parentheses at
    the splice — ``conditions.append(f"({compiled.predicate})")`` becoming
    ``conditions.append(compiled.predicate)`` — fails this test, and the measured
    magnitude is **6 rows becomes 12**: a full cross-tenant read of both
    namespaces, with no error and nothing in the logs.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    await seed_varied(adapter, namespace.id, scan_seed(ids=scanned_ids))
    other = await _make_namespace(adapter)
    await seed_varied(adapter, other.id, scan_seed(ids=foreign_ids))

    def ungrouped_compiler(ast, ctx):
        # Bind names deliberately in the compiler's own ``f_N`` family so the
        # scan's collision guard stays out of the way; the shape under test is
        # the missing parentheses, nothing else.
        return CompiledFilter(
            predicate="title = $f_0 OR content = $f_1",
            params={"f_0": "doc-0", "f_1": "scanned content"},
            consumed_keys=frozenset({"title", "content"}),
            consumed_slice_hash="ungrouped-or-tripwire",
        )

    monkeypatch.setitem(CompilerRegistry._registry, _COMPILER_KEY, ungrouped_compiler)  # noqa: SLF001

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast({"title": {"$eq": "doc-0"}}),
        scan_limit=50,
    )

    assert step.documents, "the fragment must match rows in the scanned namespace for this test to bite"
    assert len(step.documents) == 6
    assert all(d.namespace_id == namespace.id for d in step.documents)
    assert set(foreign_ids).isdisjoint({d.id for d in step.documents})


async def test_ungrouped_keyset_disjunction_cannot_absorb_the_namespace_scope(adapter, namespace) -> None:
    """The parentheses around the keyset disjunction are load-bearing too.

    This is the store-specific half of the same hazard. SurrealQL has no
    row-value comparison, so the resume predicate is a top-level ``OR``:
    ``created_at < $ts OR (created_at = $ts AND id < $id)``. Ungrouped, the
    namespace scope is absorbed into the left disjunct and the right one —
    ``created_at = $ts AND id < $cursor_id`` — reads every tenant's tie block.
    The SQLAlchemy stores cannot have this bug at all; it exists only because the
    predicate had to be written out by hand here.

    The seed is built by :func:`_two_namespace_ladders` so the foreign ids sort
    BELOW the cursor deterministically. That is not cosmetic: measured, with the
    foreign ids above the cursor the mutant leaks **0** rows and this test passes
    while proving nothing; below, it leaks **4**. Read that function's docstring
    before touching the seed.

    **Mutation-verified on this branch, and reverted afterwards.** Dropping the
    outer pair from the ``conditions`` entry — ``"(created_at <
    $after_created_at OR (created_at = $after_created_at AND id < $after_id))"``
    becoming the same string without its enclosing parentheses — fails this test:
    the resumed window returns **8 rows instead of 4**, the four extra being the
    other tenant's tie block.

    Across the module that mutant fails **4 of the 33 tests**, and all four are
    named here because an earlier version of this docstring named three and left
    the fourth to be rediscovered: this test, plus
    :func:`test_filtered_walk_puts_a_cursor_and_a_compiled_fragment_in_one_statement`,
    :func:`test_scan_never_returns_another_namespaces_rows` and
    :func:`test_every_narrowing_leg_composes_in_one_statement` — every test in
    the module that resumes from a cursor across pages.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    seed = scan_seed(ids=scanned_ids)
    await seed_documents(adapter, namespace.id, seed)
    other = await _make_namespace(adapter)
    await seed_documents(adapter, other.id, scan_seed(ids=foreign_ids))

    full = await adapter.scan_documents(namespace.id, scan_limit=50)
    assert [d.id for d in full.documents] == seed.expected

    # Resume from INSIDE the tie block: the right disjunct is only reachable at
    # a cursor whose ``created_at`` some other row also carries.
    cursor_doc = next(d for d in full.documents if d.id == seed.tied_ids[0])
    step = await adapter.scan_documents(
        namespace.id,
        after=(cursor_doc.created_at, cursor_doc.id),
        scan_limit=50,
    )

    assert [d.id for d in step.documents] == seed.expected[seed.expected.index(cursor_doc.id) + 1 :]
    assert all(d.namespace_id == namespace.id for d in step.documents)
    assert set(foreign_ids).isdisjoint({d.id for d in step.documents})
