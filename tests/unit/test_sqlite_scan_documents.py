"""``SQLiteRelationalBackend.scan_documents`` — the bounded keyset scan.

Runs against a real in-memory SQLite database through the store's own
``_SCHEMA_SQL`` (no Alembic chain behind this backend, no Docker, no services),
which matters more here than usual: this store keeps ``created_at`` as TEXT and
compares it lexicographically, so the scan's cursor is only correct if it is
bound in the store's own serialization. That cannot be checked against a mock.

This leg is not a transcription of the SQLAlchemy pair (khora #1586). Three
things are structurally different on this store, and each of them inverts a trap
that the sibling modules encode the other way round:

* **Binds are positional, so conjunct order IS bind order.** The statement is
  built as SQL text with ``?`` placeholders and a parallel ``params`` list —
  namespace, ``status``, ``updated_before``, the two cursor operands, the
  fragment's ``args``, the row bound. Appending a condition and its bind out of
  step shifts every later bind by one, and SQLite reports that as wrong rows
  whenever the shifted values happen to be type-compatible. That is what
  :func:`test_every_narrowing_leg_composes_in_one_statement` is for.
* **The cursor id is DASHED here, and ``uuid.hex`` is the mistake.** This store
  writes ``str(document.id)`` — the 36-character dashed form — so the cursor must
  bind ``str(cursor_id)``. sqlite_lance holds 32 undashed hex characters and the
  trap points the other way there. See the id-ladder note below.
* **``created_at`` is written by a bare ``dt.isoformat()``**, which OMITS the
  microsecond field at exactly ``.000000`` and emits six digits otherwise. Both
  polarities are seeded here; see the whole-second note below.

Seeding goes through ``create_document``, the production write API, so every row
is serialized by the same path production writes take. The three malformed-row
tests at the bottom are the deliberate exception: they write with a direct
``INSERT``, because ``create_document`` cannot produce the values they pin.

**Why the ``id_ladder`` seed is non-negotiable.**
:func:`tests.test_helpers.document_scan.scan_seed` draws its ids from
:func:`tests.test_helpers.document_order.id_ladder`, whose ids share a 24-hex
prefix and differ only in a trailing 8-hex counter. Do NOT "simplify" this seed
to plain ``uuid4``. The shared prefix is what makes the id half of the cursor
observable on this store: ``str(uuid)`` puts a dash at index 8, inside that
shared prefix, and ``-`` (0x2D) sorts BELOW every hex digit. So an undashed
``cursor_id.hex`` sorts ABOVE every stored id it shares leading hex with, and
``(created_at, id) < (?, ?)`` then matches strictly MORE rows — including the
cursor's own row and the row above it, so a resumed walk runs backwards.
Measured on this store from a mid-tie cursor at ``…0003``: the correct dashed
form yields ``[…0002, …0001, …0005]``; ``.hex`` yields ``[…0004, …0003, …0002,
…0001, …0005]``.

**Only HALF of that disappears under ``uuid4`` ids, and an earlier draft of this
docstring overstated it as "the whole defect is invisible".** ``str(u) < u.hex``
holds for *every* UUID — index 8 is the dash against a hex digit — so the
cursor's own row re-matches under any seed whatsoever. What needs the shared
prefix is the rows-ABOVE leg: reaching a row other than the cursor's own
requires two ids agreeing on eight leading hex characters, which random ids
essentially never do. The ladder is therefore what makes the *backwards-walk*
symptom observable, and the argument for keeping it stands as written — it is
the claimed blast radius that was too wide, not the conclusion.

**Nor does a backwards walk necessarily hang; that too was overstated.**
``exhausted`` is ``len(rows) < scan_limit`` over the RAW count, so a backwards
window that is merely short still ends the walk. Re-measured under the ``.hex``
mutant on this 6-row ladder, walking to a 30-step cap: ``scan_limit=10`` ends in
**1 step** with all 6 rows and ``exhausted=True`` — the defect is *invisible*,
because the FIRST step of a walk carries no cursor operand and the mutant cannot
apply to it; ``scan_limit=6`` ends in 2 steps, 7 rows yielded for 6 unique;
``scan_limit=5`` fills its resumed window (5 rows), reports ``exhausted=False``
and *still* ends, in 3 steps; only ``scan_limit=4`` and below fails to
terminate, capped at 30 steps having yielded 120 rows and reached 5 of the 6.
So a full window is **necessary but not sufficient** for a hang, and the two
window sizes quoted above are different measurements that must not be
conflated: 6 is a *first* step (no cursor), 5 is a *resumed* mid-tie step. A
test that fixes a generous ``scan_limit`` can watch this mutant and see nothing,
which is why the cursor tests below walk at a small bound.

**Why both microsecond polarities are seeded.** The cursor binds through the same
``dt.isoformat()`` the writer used, and ``isoformat()`` omits the ``.ffffff``
field entirely when the microsecond is zero. That makes the two plausible
mis-serializations fail on OPPOSITE seeds, so a single-polarity corpus silently
loses half the coverage:

* a cursor forced to six digits (``'…T12:30:00.000000+00:00'``) is byte-identical
  to the stored form at µs != 0 and therefore INVISIBLE there, but at µs == 0 it
  sorts above the stored ``'…T12:30:00+00:00'`` (``.`` 0x2E > ``+`` 0x2B) and the
  cursor's own row comes back;
* a cursor that truncates the microsecond is byte-identical at µs == 0 and
  therefore INVISIBLE there, but at µs != 0 it sorts below its tie-mates and
  silently skips the rest of the tie block.

Note this is the exact reverse of the sqlite_lance polarity note in
:mod:`tests.test_helpers.document_scan`: that store's ORM writes six digits
unconditionally, so ``.000000`` is the divergent case there and the agreeing case
here. Porting either seed unchanged would prove nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any, NamedTuple
from uuid import UUID, uuid4

import pytest

from khora.core.models import MemoryNamespace
from khora.core.models.document import DocumentStatus
from khora.core.models.tenancy import TenancyMode
from khora.filter import CompiledFilter, CompilerRegistry
from khora.filter.compilers.python import compile_python

# ``_documents_compile_context`` is private, and imported on purpose: the
# superset test below must compile with the *same* context the scan itself uses,
# or it would prove a property of some other context.
from khora.storage.backends.sqlite import SQLiteRelationalBackend, _documents_compile_context
from tests.test_helpers.document_scan import (
    WHOLE_SECOND,
    scan_seed,
    seed_documents,
    seed_varied,
    walk_scan,
    wire_to_ast,
    write_document,
)

# Matches the two sibling scan modules, which declare their lane
# (``pytest.mark.integration`` on the SurrealDB one, ``[pytest.mark.embedded,
# skipif]`` on sqlite_lance) while this one declared nothing. No behavioural
# change today — both lanes select these modules by path, not by marker.
pytestmark = pytest.mark.unit

_COMPILER_KEY = ("relational.sqlite", "documents")

# The two microsecond polarities, both live in every cursor assertion below. See
# the module docstring for why one of them is not enough.
_SUB_SECOND = WHOLE_SECOND.replace(microsecond=123456)
_INSTANTS: dict[str, datetime] = {"whole_second": WHOLE_SECOND, "sub_second": _SUB_SECOND}


@pytest.fixture
async def backend():
    store = SQLiteRelationalBackend(":memory:")
    await store.connect()
    try:
        yield store
    finally:
        await store.disconnect()


async def _make_namespace(store: SQLiteRelationalBackend) -> MemoryNamespace:
    nid = uuid4()
    return await store.create_namespace(MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED))


@pytest.fixture
async def namespace(backend):
    return await _make_namespace(backend)


def _two_namespace_ladders(n: int = 6) -> tuple[list[UUID], list[UUID]]:
    """Return ``(scanned_ids, foreign_ids)`` with every foreign id sorting BELOW
    every scanned id.

    Both ladders share one random 23-hex head and differ at the very next nibble
    — ``f`` for the scanned rows, ``0`` for the foreign ones — followed by the
    same 8-hex counter. That fixes the relative order by construction while
    keeping the ids unique across runs, which the fixed all-zeros prefix an
    earlier draft used would not: these tests run against ``:memory:`` today, but
    a file-backed database that is not truncated between runs would collide.
    The shared-prefix property ``id_ladder`` provides is preserved, so the
    undashed-cursor trap stays catchable.

    **Determinism only — NOT load-bearing on this store, and do not write a
    docstring claiming otherwise.** The SurrealDB sibling needs the fixed
    arrangement because its resume position expands to a top-level ``OR`` whose
    unscoped disjunct can only reach rows below the cursor, so the leak size
    depends on which tenant's ids sort lower. SQLite has a native row-value
    comparison, so the cursor here is one ``(created_at, id) < (?, ?)`` conjunct
    with no disjunction in it, and both mutants below leak on predicates every
    row in both namespaces satisfies. Measured: the namespace and grouping
    mutants each failed on independent random ladders too. The pair is used here
    so the corpus is identical run to run, not because the order decides anything.
    """
    head = uuid4().hex[:23]
    return (
        [UUID(f"{head}f{i:08x}") for i in range(n)],
        [UUID(f"{head}0{i:08x}") for i in range(n)],
    )


# --------------------------------------------------------------------------- #
# The window bound
# --------------------------------------------------------------------------- #


async def test_scan_limit_bounds_the_window(backend, namespace) -> None:
    seed = scan_seed(6)
    await seed_documents(backend, namespace.id, seed)

    step = await backend.scan_documents(namespace.id, scan_limit=2)

    assert [d.id for d in step.documents] == seed.expected[:2]
    assert step.last_scanned == (step.documents[-1].created_at, step.documents[-1].id)
    assert step.exhausted is False


async def test_a_full_window_is_not_yet_exhausted(backend, namespace) -> None:
    """``exhausted`` means SQL ran short, not "the caller has seen everything".

    A window filled exactly to the bound cannot distinguish "six rows and no
    more" from "six rows and a seventh waiting", so it must report not-exhausted
    and let the next step find the empty tail. Reporting exhaustion here would
    silently truncate every namespace whose size is a multiple of the bound. The
    short window is the other half of the same contract.
    """
    seed = scan_seed(6)
    await seed_documents(backend, namespace.id, seed)

    exact = await backend.scan_documents(namespace.id, scan_limit=6)
    assert len(exact.documents) == 6
    assert exact.exhausted is False

    short = await backend.scan_documents(namespace.id, scan_limit=7)
    assert len(short.documents) == 6
    assert short.exhausted is True


async def test_exhausted_describes_the_raw_window_not_what_survives_a_post_filter(backend, namespace) -> None:
    """A full window that the caller's post-filter empties is still not exhausted.

    ``exhausted`` derives from ``len(rows) < scan_limit`` — the RAW window — and
    it has to, because it is the walk's only termination signal. The filter here
    matches nothing at all in memory while pushing nothing to SQLite (``$or``
    over an unbacked ``occurred_at``, which ``compile_lance``'s all-or-nothing
    gate defers wholesale), so the raw window fills to the bound and the caller's
    post-filter then rejects every row it contains.

    Deriving ``exhausted`` from the surviving subset instead would report ``True``
    here and end the walk at the first window — silently truncating every
    namespace whose leading rows happen not to match.

    Scope note, stated rather than implied: ``scan_documents`` runs no
    post-filter of its own, so this cannot fail against a mutant that swaps
    ``len(rows)`` for ``len(documents)`` — those are the same list here by
    construction. What it does pin is that the window and the caller's own
    evaluation of the same AST genuinely diverge, which is the premise the
    raw-window contract rests on.
    """
    seed = scan_seed(6)
    await seed_varied(backend, namespace.id, seed)

    ast = wire_to_ast(
        {
            "$or": [
                {"source_type": {"$eq": "no-such-source-type"}},
                {"occurred_at": {"$gte": "2999-01-01T00:00:00+00:00"}},
            ]
        }
    )
    step = await backend.scan_documents(namespace.id, filter_ast=ast, scan_limit=6)

    # Nothing pushed, so the raw window is the whole namespace, filled to the bound.
    assert step.consumed_keys == frozenset()
    assert len(step.documents) == 6
    assert step.exhausted is False

    # …and the caller's post-filter keeps none of it. The two really do diverge.
    matches = compile_python(ast, _documents_compile_context()).predicate
    assert [d for d in step.documents if matches(d)] == []


async def test_scan_limit_below_one_is_rejected_before_anything_is_compiled(backend, namespace, monkeypatch) -> None:
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
        await backend.scan_documents(
            namespace.id,
            filter_ast=wire_to_ast({"source_type": {"$eq": "report"}}),
            scan_limit=0,
        )

    assert calls == []


# --------------------------------------------------------------------------- #
# The keyset cursor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("instant", _INSTANTS.values(), ids=_INSTANTS.keys())
async def test_walk_visits_every_document_exactly_once_in_total_order(backend, namespace, instant) -> None:
    """One row per step across a tie block, chaining ``last_scanned``.

    ``scan_limit=1`` puts a cursor boundary between every pair of rows, including
    between the four that share a ``created_at`` — so every resume here is a
    mid-tie resume decided solely by the ``id DESC`` leg, over the dashed stored
    form. The assertion is exact-equality against the seed's single correct
    enumeration, not a set comparison: a wrong-but-complete order has to fail too.

    Run at both microsecond polarities because the cursor's timestamp half is
    serialized by ``isoformat()``, whose output shape changes at ``.000000``; see
    the module docstring. ``walk_scan`` raises rather than hangs if a cursor fails
    to advance, which is the shape both serialization defects take.
    """
    seed = scan_seed(6, instant=instant)
    await seed_documents(backend, namespace.id, seed)

    steps = await walk_scan(backend.scan_documents, namespace.id, scan_limit=1)
    seen = [d.id for step in steps for d in step.documents]

    assert len(seen) == len(set(seen))  # no document served twice
    assert set(seen) == set(seed.expected)  # every document served
    assert seen == seed.expected  # and in one total order across the concatenation
    assert steps[-1].documents == []
    assert steps[-1].last_scanned is None
    assert steps[-1].exhausted is True


@pytest.mark.parametrize("instant", _INSTANTS.values(), ids=_INSTANTS.keys())
async def test_cursor_excludes_its_own_row_and_keeps_its_tie_mates(backend, namespace, instant) -> None:
    """A mid-tie cursor is strict on its own row and inclusive of the rest of the block.

    Both cursor operands are hand-serialized on this store, and each has a
    failure mode in either direction; none of the four raises.

    The cursor is taken from the SECOND row of the tie block, not the first, so
    that the block has members on both sides of it. A cursor that matched too
    much then returns rows ABOVE itself as well as its own row, and the walk runs
    backwards rather than merely stalling — the assertions below separate those
    two symptoms.

    *The timestamp half.* The bind is ``created_at.isoformat()``, byte-for-byte
    what ``create_document`` wrote into a TEXT column SQLite compares
    lexicographically. **Measured, by mutating the implementation and reverting
    it** — the correct window here is 3 rows, ``[…0002, …0001, …0005]``:

    * forcing six microsecond digits (``isoformat(timespec="microseconds")``) —
      at ``whole_second`` the window becomes 5 rows,
      ``[…0004, …0003, …0002, …0001, …0005]``: the cursor's own row and the one
      above it, and ``walk_scan`` raises "scan cursor did not advance". At
      ``sub_second`` the mutant is byte-identical to the stored form and every
      ``sub_second`` parametrization stays green (8 tests in this module fail
      under it, all of them seeded at a whole second).
    * truncating the microsecond (``.replace(microsecond=0)``) — the exact
      reverse: green at ``whole_second``, and at ``sub_second`` the window drops
      to 1 row, ``[…0005]``, losing the cursor's entire tie block. Across the
      module that mutant fails **only** the two ``sub_second`` parametrizations;
      every other test here is seeded at a whole second and never sees it.

    Neither polarity alone catches both, which is why this test is parametrized
    rather than pinned to one instant.

    *The id half.* The bind is ``str(cursor_id)`` — dashed, matching the stored
    form. Mutating it to ``cursor_id.hex`` gives the same 5-row backwards window
    as the first case above, at BOTH polarities (re-measured on this revision: it
    fails 10 tests in this module). See the module docstring for why the
    ``id_ladder`` seed is what makes that visible at all, and for what the
    ``.hex`` mutant does and does not do to a walk's termination.
    """
    seed = scan_seed(6, instant=instant)
    await seed_documents(backend, namespace.id, seed)

    full = await backend.scan_documents(namespace.id, scan_limit=10)
    assert [d.id for d in full.documents] == seed.expected

    cursor_doc = next(d for d in full.documents if d.id == seed.tied_ids[1])
    # The instant survived the write/read round trip, so whichever divergence
    # this polarity carries is live in this corpus.
    assert cursor_doc.created_at == seed.tie_instant
    assert cursor_doc.created_at.microsecond == instant.microsecond

    step = await backend.scan_documents(
        namespace.id,
        after=(cursor_doc.created_at, cursor_doc.id),
        scan_limit=10,
    )
    ids = [d.id for d in step.documents]
    position = seed.expected.index(cursor_doc.id)

    assert cursor_doc.id not in ids, "the cursor's own row came back — a resumed walk would never advance"
    assert set(ids).isdisjoint(seed.expected[:position]), (
        "a row that enumerates ABOVE the cursor came back — a resumed walk runs backwards"
    )
    assert seed.tied_ids[2] in ids, "the cursor's tie-mate was skipped — a resumed walk would lose rows"
    assert ids == seed.expected[position + 1 :]


async def test_filtered_walk_puts_a_cursor_and_a_compiled_fragment_in_one_statement(backend, namespace) -> None:
    """The only place a cursor and a pushdown fragment share one bind list.

    Binds are positional here, so the fragment's ``args`` must be appended after
    the cursor's two operands and before the row bound — the order the conditions
    themselves are joined in. Nothing exercises that unless both families are
    present in one call, which happens only when a cursor and a compiled fragment
    appear in the same ``SELECT``. A mis-ordered append does not raise: it shifts
    later binds by one and returns a wrong-but-plausible row set.

    The filter is a two-leaf disjunction on purpose, so it compiles to two binds
    rather than one, and the walk runs at ``scan_limit=1`` so every step past the
    first carries both families at once.
    """
    seed = scan_seed(6)
    await seed_varied(backend, namespace.id, seed)

    full = await backend.scan_documents(namespace.id, scan_limit=10)
    wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
    expected = [d.id for d in full.documents if d.source_type == "report" or d.title == "doc-1"]
    assert 1 < len(expected) < len(full.documents), "the filter must narrow, but not to a single row"

    steps = await walk_scan(
        backend.scan_documents,
        namespace.id,
        scan_limit=1,
        filter_ast=wire_to_ast(wire),
    )
    seen = [d.id for step in steps for d in step.documents]

    assert len(seen) == len(set(seen))
    assert seen == expected
    assert steps[-1].exhausted is True


async def test_mid_tie_resume_returns_the_exact_next_row(backend, namespace) -> None:
    """A one-row window resumed from mid-tie yields the next row, not any row.

    ``scan_limit=1`` on top of a mid-tie cursor is the narrowest statement of the
    resume contract: the ordering and the cursor predicate have to agree on which
    single row is next, and a bounded window gives the mistake nowhere to hide.
    A cursor that only compared ``created_at`` would return an arbitrary member
    of the tie block here and still look "sorted".
    """
    seed = scan_seed(6)
    await seed_documents(backend, namespace.id, seed)

    full = await backend.scan_documents(namespace.id, scan_limit=10)
    assert [d.id for d in full.documents] == seed.expected

    for position, cursor_id in enumerate(seed.expected[:-1]):
        cursor_doc = next(d for d in full.documents if d.id == cursor_id)
        step = await backend.scan_documents(
            namespace.id,
            after=(cursor_doc.created_at, cursor_doc.id),
            scan_limit=1,
        )
        assert [d.id for d in step.documents] == [seed.expected[position + 1]]
        assert step.exhausted is False


async def test_empty_window_reports_exhausted_without_a_position(backend, namespace) -> None:
    """Both the never-seeded namespace and the tail past the last row."""
    empty = await backend.scan_documents(namespace.id, scan_limit=5)
    assert empty.documents == []
    assert empty.last_scanned is None
    assert empty.exhausted is True

    seed = scan_seed(6)
    await seed_documents(backend, namespace.id, seed)
    full = await backend.scan_documents(namespace.id, scan_limit=10)
    oldest = full.documents[-1]

    tail = await backend.scan_documents(namespace.id, after=(oldest.created_at, oldest.id), scan_limit=5)
    assert tail.documents == []
    assert tail.last_scanned is None
    assert tail.exhausted is True


# --------------------------------------------------------------------------- #
# The total order, with the planner's help taken away
# --------------------------------------------------------------------------- #


async def test_tie_break_survives_without_the_covering_index(backend, namespace) -> None:
    """``ORDER BY created_at DESC, id DESC`` — asserted where the planner cannot fake it.

    Every other ordering assertion in this module runs against a table carrying
    ``idx_docs_ns_created_id`` (``documents(namespace_id, created_at, id)``,
    created in ``_SCHEMA_SQL``). SQLite reads that index backwards to satisfy the
    ``DESC`` order, and the trailing ``id`` key comes along for free — so
    dropping ``, id DESC`` from the statement leaves the *plan* unchanged and the
    rows still arrive in the correct order. **Measured before this test existed:
    the mutant ``ORDER BY created_at DESC`` left the entire module green,
    including the test whose docstring claims "a wrong-but-complete order has to
    fail too". The seed was never the problem; the planner was masking it.**

    Dropping the index on the fixture connection removes the free tie-break, and
    the mutant then produces a genuinely different order. Measured on this
    fixture's ladder, both re-derived here rather than copied from the ticket:
    correct ``[…0000, …0004, …0003, …0002, …0001, …0005]`` against the mutant's
    ``[…0000, …0001, …0004, …0002, …0003, …0005]`` — the same divergence the
    ticket reported. Mutation-verified by execution: with
    ``ORDER BY created_at DESC`` in ``sqlite.py`` this test fails and every other
    test in this module still passes; reverted, it passes.

    The drop is local to this in-memory fixture connection and nothing else in
    the module depends on it. Measured on the production statement shape
    (``SELECT *``) with fresh SQL text: with the index,
    ``SEARCH documents USING INDEX idx_docs_ns_created_id (namespace_id=?)`` —
    **not** ``COVERING``, since ``SELECT *`` needs columns the index does not
    carry, and a narrower projection is what reports ``COVERING INDEX``; after the
    drop, ``SCAN documents`` plus ``USE TEMP B-TREE FOR ORDER BY``, so the sort
    becomes the statement's own responsibility, which is the whole point. (If you
    ever assert a plan here, note that ``EXPLAIN QUERY PLAN`` re-run with SQL text
    already executed on this connection returned the *pre-drop* plan text in my
    run — a statement-cache artifact, not a failed drop. Vary the whitespace.)
    """
    seed = scan_seed(6)
    await seed_documents(backend, namespace.id, seed)

    await backend._conn.execute("DROP INDEX idx_docs_ns_created_id")  # noqa: SLF001 — fixture-local, no public DDL API
    await backend._conn.commit()  # noqa: SLF001

    step = await backend.scan_documents(namespace.id, scan_limit=10)
    assert [d.id for d in step.documents] == seed.expected

    # And across a paged walk, where each resume re-derives the same order.
    steps = await walk_scan(backend.scan_documents, namespace.id, scan_limit=1)
    assert [d.id for step_ in steps for d in step_.documents] == seed.expected


# --------------------------------------------------------------------------- #
# The compile split
# --------------------------------------------------------------------------- #


async def test_split_reports_only_the_leaves_sql_enforced(backend, namespace) -> None:
    """A forced-residual AST: one pushable leaf, one that must reach the post-filter.

    ``occurred_at`` is a recall-chunk key with no ``documents`` column behind it,
    so it is absent from this store's ``field_mapping`` and ``compile_lance``
    defers it under ``on_unsupported="split"``. The conjunction still pushes its
    other leaf, so ``consumed_keys`` must name ``source_type`` and only
    ``source_type``.

    Both halves are asserted on rows as well as on the reported set: the pushed
    leaf really did narrow the window, and the deferred one narrowed nothing —
    every ``report`` row survives a bound that would have excluded all of them
    had it been pushed. Reporting alone would pass against a compiler that
    emitted a predicate against a column this table does not have, which SQLite
    would answer with an error rather than with rows.
    """
    seed = scan_seed(6)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await write_document(
            backend, namespace.id, doc_id, created_at, source_type="report" if i % 2 == 0 else "library"
        )

    step = await backend.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast(
            {"source_type": {"$eq": "report"}, "occurred_at": {"$gte": "2999-01-01T00:00:00+00:00"}}
        ),
        scan_limit=10,
    )

    assert step.consumed_keys == frozenset({"source_type"})
    assert {d.source_type for d in step.documents} == {"report"}
    assert len(step.documents) == 3


async def test_date_system_keys_are_not_pushed_down_by_this_store(backend, namespace) -> None:
    """``created_at`` reaches the caller's post-filter here, and that is intended.

    This store withholds the date-valued system keys from ``_PUSHABLE_SYSTEM_KEYS``
    because ``_dt_to_str`` preserves the writer's offset with no UTC coercion,
    while ``compile_lance`` binds a UTC-normalized ISO string — so a pushed
    comparison would be on wall clock rather than on instant, and its
    false-EXCLUDE half is unrecoverable once the key is reported consumed. The
    leaf therefore compiles to a match-all placeholder, stays out of
    ``consumed_keys``, and narrows nothing — asserted positively, because a
    widened whitelist would show up here as a suddenly-shorter window rather than
    as an error.
    """
    seed = scan_seed(6)
    await seed_documents(backend, namespace.id, seed)

    step = await backend.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast({"created_at": {"$gte": "2999-01-01T00:00:00+00:00"}}),
        scan_limit=10,
    )

    assert step.consumed_keys == frozenset()
    # Every row is older than the bound, so a pushed-down comparison would have
    # returned nothing at all.
    assert [d.id for d in step.documents] == seed.expected


class _Shape(NamedTuple):
    """One superset parametrization, carrying what makes it non-vacuous.

    ``consumed`` and the ``oracle_empty`` flag are MEASURED on this store, and
    the numbers are per store — the sqlite_lance module measures its own even
    for the shapes whose wire form is identical, because the two compile
    contexts differ.
    """

    wire: dict[str, Any]
    consumed: frozenset[str]
    """The exact ``consumed_keys`` this shape reports here. Empty means the whole
    filter deferred to the caller's post-filter."""

    oracle_empty: bool = False
    """Set only where the oracle is empty *by construction*, so the
    ``assert oracle`` anti-vacuity guard below has to be inverted instead."""


# The pushdown must never reject a row the full filter would keep. Shapes are
# chosen for the ways a compiler can get that wrong, not for operator coverage:
# the ones wrapping an unpushable leaf in a disjunction or a negation matter
# most, because a match-all placeholder left inside a negation inverts into a
# match-nothing and excludes rows.
#
# ``pushable_exists`` was ``{"source_url": {"$exists": False}}`` until this
# revision, measured at oracle 0 / window 0: ``source_url`` is a NOT-NULL-ish
# system column ``create_document`` always writes, so ``$exists: False`` is
# constant-false and the parametrization could not fail for any seed. Its
# replacement measures 4 oracle / 4 window here and exercises the
# metadata-presence path rather than the constant-false system-key one.
# ``{"source_url": {"$exists": True}}`` (6/6, consumed ``source_url``) also has
# teeth and was the alternative.
_SUPERSET_SHAPES: dict[str, _Shape] = {
    "pushable_eq": _Shape({"source_type": {"$eq": "report"}}, frozenset({"source_type"})),
    "pushable_ne": _Shape({"source_type": {"$ne": "report"}}, frozenset({"source_type"})),
    "pushable_nin": _Shape({"source_type": {"$nin": ["report"]}}, frozenset({"source_type"})),
    "pushable_exists": _Shape({"metadata.tier": {"$exists": False}}, frozenset({"metadata.tier"})),
    "metadata_eq": _Shape({"metadata.tier": {"$eq": "gold"}}, frozenset({"metadata.tier"})),
    "unpushable_date": _Shape({"created_at": {"$gte": "2026-01-31T12:30:00+00:00"}}, frozenset()),
    # Constant-empty oracle, and not repairable from the corpus: ``occurred_at``
    # is a recall-chunk key with neither a ``documents`` column nor a
    # ``Document`` attribute behind it, so ``compile_python`` matches 0 of 6 rows
    # however they are seeded. Kept for the compile-split half it does pin —
    # that the leaf defers rather than being pushed against a column this table
    # does not have.
    "unpushable_key": _Shape({"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}}, frozenset(), oracle_empty=True),
    "or_over_unpushable": _Shape(
        {
            "$or": [
                {"created_at": {"$gte": "2026-01-31T12:30:00+00:00"}},
                {"source_type": {"$eq": "report"}},
            ]
        },
        frozenset(),
    ),
    "not_over_pushable": _Shape({"$not": {"source_type": {"$eq": "report"}}}, frozenset({"source_type"})),
    "not_over_unpushable": _Shape({"$not": {"created_at": {"$gte": "2026-01-31T12:30:00+00:00"}}}, frozenset()),
    "and_of_in_and_not": _Shape(
        {
            "$and": [
                {"source_type": {"$in": ["report", "library"]}},
                {"$not": {"title": {"$eq": "doc-0"}}},
            ]
        },
        frozenset({"source_type", "title"}),
    ),
}


@pytest.mark.parametrize("shape", _SUPERSET_SHAPES.values(), ids=_SUPERSET_SHAPES.keys())
async def test_pushdown_never_rejects_a_row_the_full_filter_would_keep(backend, namespace, shape) -> None:
    """The superset property the resume contract depends on.

    Resuming past the rows a pushdown rejected is sound only because a rejected
    row could not have satisfied the full filter either. The ``scan_documents``
    docstring names that as an assumption about the *compiler*; this checks the
    consequence where it actually lands, by comparing the scan's window against
    the in-process ``compile_python`` evaluation of the same AST over the same
    corpus. If it ever fails, a walk is silently and permanently dropping
    documents — a post-filter can only narrow, never recover a row the window
    never returned.

    **Scope, counted from this module's own instrumentation rather than
    asserted: of the eleven shapes, SEVEN can fail the subset assertion and
    four cannot.** Read the parametrization as a tripwire over seven shapes,
    never as a proof over the operator space — that belongs to the compilers and
    to the forced-residual conformance corpus. There are two independent ways a
    parametrization here goes toothless, and it takes two assertions to keep
    them out:

    * **Mode A — an empty oracle.** The empty set is a subset of anything, so
      ``oracle <= window`` holds whatever the window is. ``assert oracle``
      catches it, and it catches a real mutation: hollow out
      :func:`~tests.test_helpers.document_scan.seed_varied`
      so no row is ``source_type="report"`` and ``pushable_eq`` goes from 3/3 to
      0/0 while staying green. One shape is in mode A *by construction*
      (``unpushable_key``, measured 0 oracle / 6 window) and declares it via
      ``_Shape.oracle_empty``.
    * **Mode B — nothing was pushed, so the window IS the whole namespace.** The
      oracle is then computed by filtering that same window, which makes
      ``oracle <= window`` **mathematically incapable of failing for any oracle
      value**, and ``assert oracle`` does nothing about it. Measured here:
      ``unpushable_date`` 5/6, ``or_over_unpushable`` 5/6,
      ``not_over_unpushable`` 1/6, ``unpushable_key`` 0/6 — all four report
      ``consumed_keys == frozenset()``. **Those four remain structurally
      unfailable on the subset assertion and this docstring is the only place
      that says so.**
      Do NOT try to discriminate mode B by comparing the window against the
      corpus: a pushdown bug that wrongly *rejected* rows would shrink the
      window, and then ``oracle <= window`` would fire — so an equal-sized
      window is not evidence the test is incapable. The discriminator is
      ``consumed_keys == frozenset()``: no fragment reached the ``WHERE``, so
      there is no pushdown that could have rejected anything.

    What gives the mode-B shapes their value is therefore the per-shape
    ``consumed_keys`` equality below, not the subset assertion. It pins *this
    shape defers rather than pushes*, and it is the only thing that fails when a
    compile-context or ``field_mapping`` edit silently moves a shape from
    has-teeth to mode B (or the reverse: a widened whitelist pushing
    ``created_at`` here would flip three shapes at once).

    Measured window/oracle sizes over the 6-row ``seed_varied`` corpus, this
    store, this revision: ``pushable_eq`` 3/3, ``pushable_ne`` 3/3,
    ``pushable_nin`` 3/3, ``pushable_exists`` 4/4, ``metadata_eq`` 2/2,
    ``not_over_pushable`` 3/3, ``and_of_in_and_not`` 5/5 (the seven with teeth),
    then the four mode-B shapes listed above.
    """
    seed = scan_seed(6)
    await seed_varied(backend, namespace.id, seed)
    ast = wire_to_ast(shape.wire)

    step = await backend.scan_documents(namespace.id, filter_ast=ast, scan_limit=100)
    # Precondition: the comparison below is only meaningful if this one window
    # covered the whole namespace. Without it, growing the corpus past the bound
    # would fail the test for a reason that has nothing to do with the pushdown.
    assert step.exhausted is True

    all_docs = (await backend.scan_documents(namespace.id, scan_limit=100)).documents
    matches = compile_python(ast, _documents_compile_context()).predicate
    oracle = {d.id for d in all_docs if matches(d)}

    # Anti-vacuity, mode A. Inverted for the one shape whose oracle is empty by
    # construction, so that flag cannot be left on a shape that grew teeth.
    if shape.oracle_empty:
        assert oracle == set(), "declared constant-empty but the oracle now matches rows — drop the flag"
    else:
        assert oracle, "the oracle is empty, so the subset assertion below cannot fail — re-seed or re-shape"
    # Anti-vacuity, mode B: which side of the pushdown split this shape is on.
    assert step.consumed_keys == shape.consumed

    assert oracle <= {d.id for d in step.documents}


# --------------------------------------------------------------------------- #
# What the position means
# --------------------------------------------------------------------------- #


async def test_last_scanned_is_the_final_raw_row_not_the_last_match(backend, namespace) -> None:
    """Resume from the last row SCANNED, not from the last row that matches.

    The seed is arranged so the two genuinely differ — otherwise the assertion is
    vacuous. The window deliberately ENDS on a row the caller's post-filter will
    reject: the filter is an ``$or`` mixing a pushable leaf with an unbacked one,
    which ``compile_lance``'s all-or-nothing gate defers wholesale rather than
    pushing half a disjunction, so SQLite narrows nothing and the oldest row — a
    ``library`` row that does not satisfy the filter — is the final row of the raw
    window.

    A walk that resumed from the last *matching* row would re-scan the rejected
    gap on every step, and when a whole window is rejected there is no matching
    row to resume from at all, so such a walk cannot advance past a run of
    non-matching rows longer than one window. Taking the position from the raw
    window is what lets ``exhausted`` be the only termination signal.

    **Scope, measured rather than asserted past.** Against the shipped
    implementation ``last_scanned == documents[-1]`` is true by construction —
    ``scan_documents`` post-filters nothing — so no single-line edit of the
    current code can separate the two, and this test cannot be read as covering
    the current expression. What it does kill is the *plausible wrong
    implementation* the #1586 comment warns against, and that was verified rather
    than assumed: re-implementing the tail as ``matching = [d for d in documents
    if compile_python(filter_ast, …)(d)]`` and keying off ``matching[-1]`` fails
    this test and **only** this test — the entire rest of the module, walks
    included, stays green, because on every other seed here the final raw row
    also matches. That is the whole reason for the deliberately non-matching
    final row.

    The "only this test" figure is scoped to that exact construction, which
    post-filters solely when ``filter_ast`` is not ``None``. A variant that
    post-filters unconditionally moves the key on the *unfiltered* call too, and
    is therefore caught a second time by the
    ``(created_at, doc_id) == (documents[-1].created_at, documents[-1].id)``
    assertion in :func:`test_last_scanned_carries_a_datetime_and_a_uuid`
    (independently measured at 2 failures). Treat 1 as the conservative floor for
    this family of mutant, not as a claim that no other assertion can notice one —
    a bare "only this test fails" is the kind of figure that later gets quoted as
    licence to delete whatever looked redundant.
    """
    newest, middle, oldest = (uuid4() for _ in range(3))
    base = WHOLE_SECOND
    await write_document(backend, namespace.id, newest, base + timedelta(seconds=2), source_type="report")
    await write_document(backend, namespace.id, middle, base + timedelta(seconds=1), source_type="report")
    await write_document(backend, namespace.id, oldest, base, source_type="library")

    ast = wire_to_ast(
        {
            "$or": [
                {"source_type": {"$eq": "report"}},
                {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
            ]
        }
    )
    step = await backend.scan_documents(namespace.id, filter_ast=ast, scan_limit=10)

    # Nothing was pushed, so the raw window is the whole namespace and its last
    # row is the one the post-filter will drop.
    assert step.consumed_keys == frozenset()
    assert [d.id for d in step.documents] == [newest, middle, oldest]
    assert step.documents[-1].source_type == "library"

    # The premise the assertion rests on: the caller's own evaluation of the same
    # AST really does reject that final row, so the two candidate positions are
    # different rows rather than the same row named twice.
    matches = compile_python(ast, _documents_compile_context()).predicate
    surviving = [d for d in step.documents if matches(d)]
    assert surviving == step.documents[:2]

    last_row = step.documents[-1]
    assert step.last_scanned == (last_row.created_at, last_row.id)
    assert step.last_scanned != (surviving[-1].created_at, surviving[-1].id)


async def test_last_scanned_carries_a_datetime_and_a_uuid(backend, namespace) -> None:
    """``DocumentScanKey`` declares ``tuple[datetime, UUID]``, and the timestamp
    half is the one this store can get wrong silently.

    ``created_at`` lives in the raw row as TEXT, and the key is built off that
    raw row by :func:`~khora.storage.backends.sqlite._scan_key`, which parses it
    strictly rather than binding the string through. (Until this change the key came
    from the converted :class:`Document`, where ``_parse_dt(...) or now()`` had
    already run — that is what the raise in ``_scan_key`` replaced, and the
    ``''`` tests at the bottom of this module pin the difference.)

    **Measured by mutation, and the mutant has to be named exactly, because three
    plausible readings of "bind the raw TEXT straight through" give three
    different counts.** All three were run on this revision with the tree hashed
    before and after each run:

    * replace ``_scan_key``'s whole body with
      ``return (row["created_at"], UUID(row["id"]))`` — guards gone — **20 tests
      fail**: this one on the type assertion, the walks with ``AttributeError:
      'str' object has no attribute 'isoformat'`` when the raw string is bound
      back through ``_dt_to_str``, and the whole malformed-position section
      below, which pins the two guards and cannot pass without them.
    * replace only the ``return`` line, leaving both guards intact — **16 fail**.
      The 4-test difference IS the malformed-position section, so this variant
      isolates "the key carries the wrong type" from "the guards are gone".
    * replace the call site instead (``key=_scan_key`` ->
      ``key=lambda row: (row["created_at"], UUID(row["id"]))``) — **21 fail**.
      The extra one is
      :func:`test_a_lenient_key_extractor_repeats_the_same_window_forever`, which
      monkeypatches the module-level ``_scan_key``; bypass the call site and that
      monkeypatch stops reaching production. An earlier draft of this docstring
      quoted 21 against the body-replacement wording, which is why the number is
      now pinned to its exact replacement text.

    The point stands under all three: the mutant is **not silent** on this store,
    contrary to the premise this test was originally written under. So keep this
    test as the direct, legible statement of the declared contract, not as the
    only thing standing between a TEXT key and a green suite.

    Its sibling is :func:`test_last_scanned_is_the_final_raw_row_not_the_last_match`
    above, which pins *which row* the key comes from while this one pins *what
    type* it carries. The two are not redundant and neither subsumes the other:
    the type mutant leaves the right row and the wrong type, the last-match mutant
    leaves the right type and the wrong row.

    The unfiltered assertion below is also what separates the two constructions of
    that last-match mutant — see that test's docstring for the 1-vs-2 figure. A
    variant that post-filters unconditionally moves the key on this call too and
    trips the equality here; one guarded on ``filter_ast is not None`` does not.
    Do not weaken it to the filtered call alone.
    """
    seed = scan_seed(6)
    await seed_varied(backend, namespace.id, seed)

    step = await backend.scan_documents(namespace.id, scan_limit=3)
    assert step.last_scanned is not None
    created_at, doc_id = step.last_scanned
    assert isinstance(created_at, datetime)
    assert isinstance(doc_id, UUID)
    assert (created_at, doc_id) == (step.documents[-1].created_at, step.documents[-1].id)

    filtered = await backend.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast({"source_type": {"$eq": "report"}}),
        scan_limit=2,
    )
    assert filtered.last_scanned is not None
    assert isinstance(filtered.last_scanned[0], datetime)
    assert isinstance(filtered.last_scanned[1], UUID)


# --------------------------------------------------------------------------- #
# Malformed stored positions
# --------------------------------------------------------------------------- #
#
# Everything above seeds through ``create_document``. These tests do not, and
# cannot: ``create_document`` serializes ``created_at`` through ``_dt_to_str``
# and coalesces a missing value to now(), so none of the stored forms below is
# reachable through it. That is the same reachability class ``_scan_key``'s
# docstring claims for them — an out-of-band writer only — and the reason the
# guard raises rather than coping.
#
# The two guards are separate lines with separate failure modes, and a reader
# should not take one for the other: ``fromisoformat`` raising covers text that
# is not a timestamp at all (``''``), while the round-trip identity check covers
# text that parses perfectly and then binds back as a DIFFERENT string.


async def _direct_insert(store: Any, namespace_id: UUID, doc_id: UUID, created_at_text: str) -> None:
    """Write one row with a hand-chosen ``created_at`` TEXT, bypassing the write API."""
    await store._conn.execute(  # noqa: SLF001 — out-of-band on purpose; see the section comment
        "INSERT INTO documents (id, namespace_id, content, checksum, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(doc_id),
            str(namespace_id),
            "scanned content",
            f"scan-{doc_id.hex}",
            created_at_text,
            WHOLE_SECOND.isoformat(),
        ),
    )
    await store._conn.commit()  # noqa: SLF001


async def test_a_blank_stored_created_at_raises_instead_of_inventing_a_position(backend, namespace) -> None:
    """``''`` satisfies ``TEXT NOT NULL`` and is the value behind the unbounded walk.

    The schema's ``NOT NULL`` argument covers *SQL evaluating a NULL key*; it
    says nothing about the Python converter firing on a non-NULL value that
    satisfies the constraint. ``''`` is such a value, and ``_row_to_document``
    maps it to ``datetime.now(UTC)`` — a position above every row in the table,
    so the next step re-matches the whole window. ``_scan_key`` parses with
    ``datetime.fromisoformat`` rather than ``_parse_dt`` precisely so that this
    row raises instead: ``_parse_dt``'s ``if not val`` guard maps ``''`` to
    ``None``, and a ``None`` seated in a ``tuple[datetime, UUID]`` is a type
    violation the checker cannot see through the tuple.

    ``ValueError`` comes from the stdlib here, not from khora — the message is
    ``Invalid isoformat string: ''``.

    **Scope, asserted rather than left implied.** The key is built from the
    window's LAST row only, so the guard fires exactly when the malformed row
    terminates a window — measured below: it raises at ``scan_limit=50`` (the
    ``''`` row sorts below every other stored value, so it ends the window) and
    does NOT raise at ``scan_limit=2``, which stops short of it. A malformed row
    sitting in the middle of a window is therefore still returned as a
    ``Document`` with an invented ``created_at``; that residue belongs to
    ``_row_to_document``, not to the cursor.
    """
    seed = scan_seed(6)
    await seed_documents(backend, namespace.id, seed)
    await _direct_insert(backend, namespace.id, uuid4(), "")

    with pytest.raises(ValueError, match="isoformat"):
        await backend.scan_documents(namespace.id, scan_limit=50)

    # The same corpus, a window that stops before the malformed row: no raise.
    short = await backend.scan_documents(namespace.id, scan_limit=2)
    assert [d.id for d in short.documents] == seed.expected[:2]


_NON_ROUND_TRIPPING = {
    # stored form -> which way the round-tripped cursor sorts against it
    "space_separator": ("2026-01-31 12:30:00+00:00", "above"),
    "explicit_zero_micros": ("2026-01-31T12:30:00.000000+00:00", "below"),
    "z_suffix": ("2026-01-31T12:30:00Z", "below"),
}


@pytest.mark.parametrize("stored,direction", _NON_ROUND_TRIPPING.values(), ids=_NON_ROUND_TRIPPING.keys())
async def test_a_stored_created_at_that_does_not_survive_the_round_trip_raises(
    backend, namespace, stored, direction
) -> None:
    """Parsing successfully is not the invariant — binding back byte-identical is.

    The cursor is bound through ``_dt_to_str`` (``datetime.isoformat()``) against
    a TEXT column SQLite compares **lexicographically**, so a stored value that
    ``fromisoformat`` accepts but ``isoformat`` does not reproduce yields a cursor
    that disagrees with ``ORDER BY``. Both directions are broken and neither is
    loud, which is why ``_scan_key`` asserts identity rather than parseability:

    * ``'2026-01-31 12:30:00+00:00'`` (space separator) round-trips to a ``T``
      form that sorts **ABOVE** the stored text (``T`` 0x54 > space 0x20), so the
      cursor's own row re-matches forever — the unbounded walk this guard exists
      to prevent.
    * ``'…T12:30:00.000000+00:00'`` and ``'…T12:30:00Z'`` round-trip to
      ``'…T12:30:00+00:00'``, which sorts **BELOW** the stored text (``+`` 0x2B
      is below both ``.`` 0x2E and ``Z`` 0x5A), so the cursor silently skips the
      rest of its own tie block.

    The direction is recomputed in the test body rather than only asserted in
    prose, so a stdlib change to either function fails here instead of quietly
    invalidating the reasoning. The three forms that DO hold identity are covered
    by :func:`test_every_created_at_shape_the_write_api_produces_scans_clean`.
    """
    parsed = datetime.fromisoformat(stored)  # accepted — that is the trap
    round_tripped = parsed.isoformat()
    assert round_tripped != stored
    assert (round_tripped > stored) is (direction == "above")

    await _direct_insert(backend, namespace.id, uuid4(), stored)

    with pytest.raises(ValueError, match="round-trip"):
        await backend.scan_documents(namespace.id, scan_limit=50)


_WRITE_API_SHAPES = {
    "aware_whole_second": WHOLE_SECOND,
    "aware_micros": _SUB_SECOND,
    "naive_whole_second": WHOLE_SECOND.replace(tzinfo=None),
    "naive_micros": WHOLE_SECOND.replace(tzinfo=None, microsecond=1),
    "positive_offset": WHOLE_SECOND.replace(tzinfo=timezone(timedelta(hours=2))),
    "negative_offset_micros": WHOLE_SECOND.replace(microsecond=987654, tzinfo=timezone(-timedelta(hours=5))),
    "epoch": datetime(1970, 1, 1, tzinfo=UTC),
}


@pytest.mark.parametrize("created_at", _WRITE_API_SHAPES.values(), ids=_WRITE_API_SHAPES.keys())
async def test_every_created_at_shape_the_write_api_produces_scans_clean(backend, namespace, created_at) -> None:
    """The benign direction, so the round-trip guard cannot be over-tight.

    A guard that raises on values this store writes itself would be worse than
    the defect it closes. Seven shapes reachable through ``create_document``,
    measured: both microsecond polarities, naive and aware, a positive and a
    negative non-UTC offset, and the epoch. Each must scan without raising AND
    its key must bind back to exactly the bytes stored — asserted against the
    stored TEXT read straight off the table, not against the ``Document``.

    Note what this pins about the taxonomy: ``_scan_key`` returns an **aware or
    naive** ``datetime`` depending on what the writer passed, deliberately
    un-normalized, and that is safe *because* of the identity check rather than
    because the value passes through untouched (it does not — the path is TEXT ->
    ``fromisoformat`` -> ``datetime`` -> ``_dt_to_str`` -> TEXT).
    """
    doc_id = uuid4()
    await write_document(backend, namespace.id, doc_id, created_at)

    cursor = await backend._conn.execute("SELECT created_at FROM documents WHERE id = ?", [str(doc_id)])  # noqa: SLF001
    stored = (await cursor.fetchone())["created_at"]

    step = await backend.scan_documents(namespace.id, scan_limit=10)
    assert step.last_scanned is not None
    assert step.last_scanned == (created_at, doc_id)
    assert step.last_scanned[0].isoformat() == stored


async def test_a_namespace_mixing_naive_and_aware_timestamps_walks_in_stored_text_order(backend, namespace) -> None:
    """One table, both tz shapes, walked so that every step is a resumed step.

    ``_dt_to_str`` applies no UTC coercion, so a single namespace can hold naive
    and aware ``created_at`` text at once. Each cursor is only ever bound back
    against its own stored value, so the walk needs no common tz shape — but that
    is an argument, and this is the measurement: 6 rows alternating naive and
    aware, walked at ``scan_limit=1`` so every cursor in the corpus is exercised,
    must yield 6 unique rows in exactly the order the unpaged
    ``ORDER BY created_at DESC, id DESC`` produces.

    The enumeration is **stored-text order, not instant order** — that is this
    store's documented, pre-existing property and the reason the date system keys
    stay unpushable. The two positions here are not even mutually comparable in
    Python (``TypeError: can't compare offset-naive and offset-aware
    datetimes``), which constrains anything collecting positions across rows but
    not the walk itself.
    """
    for i in range(6):
        stamp = WHOLE_SECOND + timedelta(seconds=i)
        await write_document(backend, namespace.id, uuid4(), stamp if i % 2 == 0 else stamp.replace(tzinfo=None))

    cursor = await backend._conn.execute(  # noqa: SLF001 — the unpaged order is the oracle here
        "SELECT id FROM documents WHERE namespace_id = ? ORDER BY created_at DESC, id DESC",
        [str(namespace.id)],
    )
    unpaged = [UUID(row["id"]) for row in await cursor.fetchall()]

    steps = await walk_scan(backend.scan_documents, namespace.id, scan_limit=1)
    walked = [d.id for step in steps for d in step.documents]

    assert len(walked) == 6
    assert len(set(walked)) == 6
    assert walked == unpaged


async def test_a_lenient_key_extractor_repeats_the_same_window_forever(backend, namespace, monkeypatch) -> None:
    """The counterfactual the strict extractor exists to rule out, run rather than described.

    ``_scan_key`` is replaced with the pre-hardening shape — ``_parse_dt(...) or
    datetime.now(UTC)``, which is what reading the position off the converted
    :class:`Document` amounts to — over a corpus containing one ``''`` row. The
    ``''`` row sorts below every other stored value, so it terminates the window,
    and its invented ``now()`` position sits above every row in the table: the
    next step re-matches the whole window.

    Measured against a 7-row corpus (6 well-formed + the ``''`` row) with 12
    steps allowed, and the bound decides the symptom — the same
    "necessary-but-not-sufficient" shape the module docstring records for the
    ``.hex`` mutant:

    * ``scan_limit=7`` — the SAME window 12 times, 84 rows yielded for 7 unique,
      ``exhausted`` never true. The reviewer's shape exactly, and what this test
      asserts.
    * ``scan_limit=5`` and ``scan_limit=3`` — **terminate**, in 2 and 3 steps,
      7 rows for 7 unique. The malformed row lands in a SHORT window, which ends
      the walk before the invented position can be chained.
    * ``scan_limit=1`` — every window is full, so the walk cycles through 7
      distinct windows and never terminates.

    Note ``walk_scan``'s cursor-did-not-advance guard does NOT catch any of this,
    because ``now()`` differs on every call — the failure it raises is the
    ``max_steps`` backstop instead, which is what the assertion below matches on.
    """
    import khora.storage.backends.sqlite as sqlite_module

    seed = scan_seed(6)
    await seed_documents(backend, namespace.id, seed)
    await _direct_insert(backend, namespace.id, uuid4(), "")

    def lenient_key(row: Any) -> Any:
        return (sqlite_module._parse_dt(row["created_at"]) or datetime.now(UTC), UUID(row["id"]))  # noqa: SLF001

    monkeypatch.setattr(sqlite_module, "_scan_key", lenient_key)

    windows = []
    after = None
    for _ in range(4):
        step = await backend.scan_documents(namespace.id, after=after, scan_limit=7)
        windows.append([d.id for d in step.documents])
        assert step.exhausted is False, "a full window is what makes the repeat unbounded"
        after = step.last_scanned

    assert windows[0] == windows[1] == windows[2] == windows[3]
    assert len(windows[0]) == 7

    with pytest.raises(AssertionError, match="did not report exhausted"):
        await walk_scan(backend.scan_documents, namespace.id, scan_limit=7, max_steps=5)


# --------------------------------------------------------------------------- #
# The non-filter narrowing legs
# --------------------------------------------------------------------------- #


async def test_status_and_updated_before_narrow_the_window(backend, namespace) -> None:
    """Both optional legs bind positionally, ahead of the cursor and the fragment.

    Asserted on exact row sets rather than on counts: a bind shifted by one
    position produces a narrower-but-wrong window just as readily as a correct
    one, and only the identities separate them.
    """
    seed = scan_seed(6)
    cutoff = seed.tie_instant + timedelta(hours=1)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await write_document(
            backend,
            namespace.id,
            doc_id,
            created_at,
            status=DocumentStatus.COMPLETED if i % 2 == 0 else DocumentStatus.PENDING,
            updated_at=cutoff - timedelta(minutes=1) if i < 4 else cutoff + timedelta(minutes=1),
        )

    unbounded = await backend.scan_documents(namespace.id, scan_limit=10)
    assert len(unbounded.documents) == 6

    by_status = await backend.scan_documents(namespace.id, status=DocumentStatus.COMPLETED.value, scan_limit=10)
    assert {d.id for d in by_status.documents} == {doc_id for i, (doc_id, _) in enumerate(seed.writes) if i % 2 == 0}

    by_updated = await backend.scan_documents(namespace.id, updated_before=cutoff, scan_limit=10)
    assert {d.id for d in by_updated.documents} == {doc_id for i, (doc_id, _) in enumerate(seed.writes) if i < 4}


async def test_every_narrowing_leg_composes_in_one_statement(backend, namespace) -> None:
    """Cursor + compiled fragment + ``status`` + ``updated_before``, all at once.

    Each leg has its own test above; none of those puts more than two of them in
    the same ``WHERE``. On a positional-bind store this is the shape where an
    append-order mistake actually lands: with six binds ahead of the row bound, a
    conjunct appended out of step shifts every later value by one and SQLite
    answers with a plausible subset rather than an error.

    The expectation is computed from the rows the store returns rather than from
    the seeding loop's counter, and each leg is asserted to be doing work — a leg
    that narrowed nothing would make its own conjunct untested while the overall
    assertion still passed.
    """
    seed = scan_seed(6)
    cutoff = seed.tie_instant + timedelta(hours=1)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await write_document(
            backend,
            namespace.id,
            doc_id,
            created_at,
            title=f"doc-{i}",
            source_type="report" if i % 2 == 0 else "library",
            status=DocumentStatus.COMPLETED if i < 5 else DocumentStatus.PENDING,
            updated_at=cutoff - timedelta(minutes=1) if i < 4 else cutoff + timedelta(minutes=1),
        )

    everything = await backend.scan_documents(namespace.id, scan_limit=50)
    assert len(everything.documents) == 6

    # Resume from inside the tie block, so the cursor's id leg is live.
    cursor_doc = next(d for d in everything.documents if d.id == seed.tied_ids[0])
    after_cursor = everything.documents[everything.documents.index(cursor_doc) + 1 :]

    def surviving(docs: list[Any]) -> list[UUID]:
        return [
            d.id
            for d in docs
            if d.source_type == "report" and d.status == DocumentStatus.COMPLETED and d.updated_at < cutoff
        ]

    expected = surviving(after_cursor)

    # Every leg must actually remove something, or its conjunct is untested here.
    assert expected, "the four-way conjunction must keep at least one row"
    assert len(after_cursor) < len(everything.documents), "the cursor narrowed nothing"
    assert len(expected) < len(after_cursor), "the filter/status/updated_before legs narrowed nothing"

    step = await backend.scan_documents(
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
# Namespace isolation and the ungrouped-OR tripwire
# --------------------------------------------------------------------------- #
#
# Everything above runs in a single-namespace fixture, so no assertion up there
# can notice a scan that ignores its namespace scope. The two tests below are the
# ones that make the mutants fail; each records the mutation that was run against
# it and what it measured.
#
# Both seed their two namespaces through :func:`_two_namespace_ladders`, so the
# corpus is identical run to run. Read that function before touching the seed:
# the fixed relative order is determinism, NOT a correctness requirement on this
# store, and the reason it is load-bearing on the SurrealDB sibling does not
# transfer here.


async def test_scan_never_returns_another_namespaces_rows(backend, namespace) -> None:
    """A filtered walk over one namespace must not see a byte of the other.

    The second namespace is seeded with the SAME varied corpus, so every row in
    it matches the same filter — if the namespace predicate is dropped (or stops
    AND-composing with the fragment), the foreign rows are not merely reachable,
    they are guaranteed hits. Walked at ``scan_limit=1`` so the keyset predicate
    is exercised across pages too: the cursor is namespace-blind on its own, and
    only the scope predicate keeps a resume inside its tenant.

    **Mutation-verified, and reverted afterwards.** Neutralising the namespace
    predicate in :func:`~khora.storage.backends.sqlite._documents_where` —
    ``conditions = ["namespace_id = ?"]`` / ``params = [str(namespace_id)]``
    becoming ``conditions = ["1"]`` / ``params = []``, the faithful SQLite form of
    "delete the scope", since an empty condition list would emit a bare ``WHERE``
    and fail as a syntax error rather than as wrong rows — fails this test, and
    the walk assertion fires before the unfiltered read below is reached.

    **Re-measured on this revision, and the magnitude an earlier draft recorded
    (8 rows) was the wrong measurement.** The filtered WALK returns **6 rows
    instead of 4**, three of them the other tenant's. A *one-shot* filtered
    window over the same 12-row corpus returns **8 instead of 4** (4 own + 4
    foreign) — so the walk sees only 6 of the 8 rows the mutant leaks, losing one
    matching row per namespace. Observed rather than explained: the mutant's own
    window comes back in an order that is not ``(created_at DESC, id DESC)``
    within the tie block, so the keyset cursor steps over rows between pages.
    That is a second symptom of the same mutant and not what this test is for;
    quote 6-vs-4 for the walk and 8-vs-4 for a single window, never one figure
    for both.

    The grouping tripwire that follows fails on the same mutant, at **12 rows
    instead of 6**; those two are the only failures the mutant produces across
    this module, which is exactly why both of them exist.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    await seed_varied(backend, namespace.id, scan_seed(ids=scanned_ids))
    other = await _make_namespace(backend)
    # Same tie instant, same varied corpus: every foreign row is a guaranteed hit
    # for the filter below, so a dropped predicate fails on every run rather than
    # on a lucky arrangement.
    await seed_varied(backend, other.id, scan_seed(ids=foreign_ids))

    wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
    steps = await walk_scan(backend.scan_documents, namespace.id, scan_limit=1, filter_ast=wire_to_ast(wire))
    seen = [d for step in steps for d in step.documents]

    assert seen, "the filter must match rows in the scanned namespace for this test to bite"
    assert len(seen) == 4
    assert all(d.namespace_id == namespace.id for d in seen)
    assert set(foreign_ids).isdisjoint({d.id for d in seen})

    unfiltered = await backend.scan_documents(namespace.id, scan_limit=50)
    assert len(unfiltered.documents) == 6
    assert all(d.namespace_id == namespace.id for d in unfiltered.documents)


async def test_ungrouped_or_fragment_cannot_absorb_the_namespace_scope(backend, namespace, monkeypatch) -> None:
    """The parentheses around the spliced fragment are load-bearing.

    ``scan_documents`` joins its conditions with ``AND`` and wraps the compiled
    fragment in parentheses at the splice. Ungrouped, ``AND`` binds tighter than
    ``OR``, so ``namespace_id = ? AND title = ? OR content = ?`` parses as
    ``(namespace_id = ? AND title = ?) OR (content = ?)`` — the right disjunct is
    unscoped and returns every tenant's rows. It fails as somebody else's data,
    not as an error.

    No compiler emits an ungrouped fragment today (``compile_lance``
    self-parenthesizes every boolean node it emits), so no compiled-filter test
    can reach this. The registered compiler is therefore replaced with one that
    emits the bare ungrouped shape and the real ``scan_documents`` is driven
    through it — deliberately, rather than adding a fragment-rendering seam to
    production code just to make the parentheses reachable from a test. Both
    namespaces' rows satisfy the right disjunct, so an absorbed scope predicate
    yields foreign rows deterministically rather than by luck.

    **Mutation-verified, and reverted afterwards.** Removing the parentheses at
    the splice — ``conditions.append(f"({compiled.predicate})")`` becoming
    ``conditions.append(compiled.predicate)`` — fails this test, and the measured
    magnitude is **12 rows where the grouped form returns 6**: a literal 2x
    cross-tenant leak, a full read of both namespaces, with no error and nothing
    in the logs. It is the ONLY test in this module that mutant fails.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    await seed_varied(backend, namespace.id, scan_seed(ids=scanned_ids))
    other = await _make_namespace(backend)
    await seed_varied(backend, other.id, scan_seed(ids=foreign_ids))

    def ungrouped_compiler(ast, ctx):
        # Positional binds, in emit order, exactly as ``compile_lance`` returns
        # them. The shape under test is the missing parentheses, nothing else.
        return CompiledFilter(
            predicate="title = ? OR content = ?",
            params={"args": ["doc-0", "scanned content"]},
            consumed_keys=frozenset({"title", "content"}),
            consumed_slice_hash="ungrouped-or-tripwire",
        )

    monkeypatch.setitem(CompilerRegistry._registry, _COMPILER_KEY, ungrouped_compiler)  # noqa: SLF001

    step = await backend.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast({"title": {"$eq": "doc-0"}}),
        scan_limit=50,
    )

    assert step.documents, "the fragment must match rows in the scanned namespace for this test to bite"
    assert len(step.documents) == 6
    assert all(d.namespace_id == namespace.id for d in step.documents)
    assert set(foreign_ids).isdisjoint({d.id for d in step.documents})
