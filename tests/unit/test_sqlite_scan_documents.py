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
is serialized by the same path production writes take.

**Why the ``id_ladder`` seed is non-negotiable.**
:func:`tests.test_helpers.document_scan.scan_seed` draws its ids from
:func:`tests.test_helpers.document_order.id_ladder`, whose ids share a 24-hex
prefix and differ only in a trailing 8-hex counter. Do NOT "simplify" this seed
to plain ``uuid4``. The shared prefix is the only thing that makes the id half of
the cursor observable on this store: ``str(uuid)`` puts a dash at index 8, inside
that shared prefix, and ``-`` (0x2D) sorts BELOW every hex digit. So an undashed
``cursor_id.hex`` sorts ABOVE every stored id it shares leading hex with, and
``(created_at, id) < (?, ?)`` then matches strictly MORE rows — including the
cursor's own row and the row above it, so a chained walk runs backwards and never
terminates. Measured on this store from a mid-tie cursor at ``…0003``: the
correct dashed form yields ``[…0002, …0001, …0005]``; ``.hex`` yields ``[…0004,
…0003, …0002, …0001, …0005]``. With ``uuid4`` ids the comparison is decided long
before index 8 (it would need two ids agreeing on eight leading hex characters),
and the whole defect is invisible.

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

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from khora.core.models import Document, MemoryNamespace
from khora.core.models.document import DocumentStatus
from khora.core.models.tenancy import TenancyMode
from khora.filter import CompiledFilter, CompilerRegistry, RecallFilter
from khora.filter.ast import parse_to_ast
from khora.filter.compilers.python import compile_python

# ``_documents_compile_context`` is private, and imported on purpose: the
# superset test below must compile with the *same* context the scan itself uses,
# or it would prove a property of some other context.
from khora.storage.backends.sqlite import SQLiteRelationalBackend, _documents_compile_context
from tests.test_helpers.document_order import seed_order
from tests.test_helpers.document_scan import WHOLE_SECOND, ScanSeed, scan_seed, walk_scan

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


def _filter_ast(wire: dict[str, Any]) -> Any:
    return parse_to_ast(RecallFilter.model_validate(wire))


async def _write(store: Any, namespace_id: UUID, doc_id: UUID, created_at: datetime, **fields: Any) -> None:
    """Insert one document through the production write API."""
    await store.create_document(
        Document(
            id=doc_id,
            namespace_id=namespace_id,
            content="scanned content",
            checksum=f"scan-{doc_id.hex}",
            created_at=created_at,
            updated_at=fields.pop("updated_at", created_at),
            **fields,
        )
    )


async def _seed(store: Any, namespace_id: UUID, seed: ScanSeed) -> None:
    for doc_id, created_at in seed.writes:
        await _write(store, namespace_id, doc_id, created_at)


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


def _seed_from_ladder(ids: list[UUID], *, instant: datetime = WHOLE_SECOND) -> ScanSeed:
    """Build a :class:`ScanSeed` over a caller-supplied ascending id ladder.

    Identical in construction to :func:`tests.test_helpers.document_scan.scan_seed`
    — ``total - 2`` rows share ``instant`` so only the ``id DESC`` leg can order
    them, and the two rows outside that block carry timestamp and id in
    deliberate conflict. The one difference is that the ids come from the caller
    instead of from a fresh random ladder, so the two namespaces below can be
    seeded at the same tie instant with a pinned relative id order.
    """
    if len(ids) < 5:
        raise ValueError(f"need a tie block of at least 3 rows to resume from the middle of, got {len(ids)}")

    newest_id, oldest_id = ids[0], ids[-1]
    tied = ids[1:-1]
    stamps = dict.fromkeys(tied, instant)
    stamps[newest_id] = instant + timedelta(seconds=1)
    stamps[oldest_id] = instant - timedelta(seconds=1)

    return ScanSeed(
        writes=[(doc_id, stamps[doc_id]) for doc_id in seed_order(ids)],
        expected=[newest_id, *reversed(tied), oldest_id],
        tied_ids=list(reversed(tied)),
        newest_id=newest_id,
        oldest_id=oldest_id,
        tie_instant=instant,
    )


async def _seed_varied(store: Any, namespace_id: UUID, seed: ScanSeed) -> None:
    """Seed the same corpus with attribute variety, so a filter can split it.

    Attributes are assigned by *write* index, which is deliberately not the
    enumeration order — every expectation below is therefore derived from the
    rows a scan actually returns, never from this loop's counter.
    """
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await _write(
            store,
            namespace_id,
            doc_id,
            created_at,
            title=f"doc-{i}",
            source_type="report" if i % 2 == 0 else "library",
            metadata={"tier": "gold"} if i < 2 else {},
        )


# --------------------------------------------------------------------------- #
# The window bound
# --------------------------------------------------------------------------- #


async def test_scan_limit_bounds_the_window(backend, namespace) -> None:
    seed = scan_seed(6)
    await _seed(backend, namespace.id, seed)

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
    await _seed(backend, namespace.id, seed)

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
    await _seed_varied(backend, namespace.id, seed)

    ast = _filter_ast(
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
            filter_ast=_filter_ast({"source_type": {"$eq": "report"}}),
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
    await _seed(backend, namespace.id, seed)

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
      ``sub_second`` the mutant is byte-identical to the stored form and the
      whole module stays green.
    * truncating the microsecond (``.replace(microsecond=0)``) — the exact
      reverse: green at ``whole_second``, and at ``sub_second`` the window drops
      to 1 row, ``[…0005]``, losing the cursor's entire tie block. Across the
      module that mutant fails **only** the two ``sub_second`` parametrizations;
      every other test here is seeded at a whole second and never sees it.

    Neither polarity alone catches both, which is why this test is parametrized
    rather than pinned to one instant.

    *The id half.* The bind is ``str(cursor_id)`` — dashed, matching the stored
    form. Mutating it to ``cursor_id.hex`` gives the same 5-row backwards window
    as the first case above, at BOTH polarities (it fails 8 tests in this
    module). See the module docstring for why the ``id_ladder`` seed is what
    makes that visible at all.
    """
    seed = scan_seed(6, instant=instant)
    await _seed(backend, namespace.id, seed)

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
    await _seed_varied(backend, namespace.id, seed)

    full = await backend.scan_documents(namespace.id, scan_limit=10)
    wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
    expected = [d.id for d in full.documents if d.source_type == "report" or d.title == "doc-1"]
    assert 1 < len(expected) < len(full.documents), "the filter must narrow, but not to a single row"

    steps = await walk_scan(
        backend.scan_documents,
        namespace.id,
        scan_limit=1,
        filter_ast=_filter_ast(wire),
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
    await _seed(backend, namespace.id, seed)

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
    await _seed(backend, namespace.id, seed)
    full = await backend.scan_documents(namespace.id, scan_limit=10)
    oldest = full.documents[-1]

    tail = await backend.scan_documents(namespace.id, after=(oldest.created_at, oldest.id), scan_limit=5)
    assert tail.documents == []
    assert tail.last_scanned is None
    assert tail.exhausted is True


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
        await _write(backend, namespace.id, doc_id, created_at, source_type="report" if i % 2 == 0 else "library")

    step = await backend.scan_documents(
        namespace.id,
        filter_ast=_filter_ast(
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
    await _seed(backend, namespace.id, seed)

    step = await backend.scan_documents(
        namespace.id,
        filter_ast=_filter_ast({"created_at": {"$gte": "2999-01-01T00:00:00+00:00"}}),
        scan_limit=10,
    )

    assert step.consumed_keys == frozenset()
    # Every row is older than the bound, so a pushed-down comparison would have
    # returned nothing at all.
    assert [d.id for d in step.documents] == seed.expected


# The pushdown must never reject a row the full filter would keep. Shapes are
# chosen for the ways a compiler can get that wrong, not for operator coverage:
# the ones wrapping an unpushable leaf in a disjunction or a negation matter
# most, because a match-all placeholder left inside a negation inverts into a
# match-nothing and excludes rows.
_SUPERSET_SHAPES: dict[str, dict[str, Any]] = {
    "pushable_eq": {"source_type": {"$eq": "report"}},
    "pushable_ne": {"source_type": {"$ne": "report"}},
    "pushable_nin": {"source_type": {"$nin": ["report"]}},
    "pushable_exists": {"source_url": {"$exists": False}},
    "metadata_eq": {"metadata.tier": {"$eq": "gold"}},
    "unpushable_date": {"created_at": {"$gte": "2026-01-31T12:30:00+00:00"}},
    "unpushable_key": {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
    "or_over_unpushable": {
        "$or": [
            {"created_at": {"$gte": "2026-01-31T12:30:00+00:00"}},
            {"source_type": {"$eq": "report"}},
        ]
    },
    "not_over_pushable": {"$not": {"source_type": {"$eq": "report"}}},
    "not_over_unpushable": {"$not": {"created_at": {"$gte": "2026-01-31T12:30:00+00:00"}}},
    "and_of_in_and_not": {
        "$and": [
            {"source_type": {"$in": ["report", "library"]}},
            {"$not": {"title": {"$eq": "doc-0"}}},
        ]
    },
}


@pytest.mark.parametrize("wire", _SUPERSET_SHAPES.values(), ids=_SUPERSET_SHAPES.keys())
async def test_pushdown_never_rejects_a_row_the_full_filter_would_keep(backend, namespace, wire) -> None:
    """The superset property the resume contract depends on.

    Resuming past the rows a pushdown rejected is sound only because a rejected
    row could not have satisfied the full filter either. The ``scan_documents``
    docstring names that as an assumption about the *compiler*; this checks the
    consequence where it actually lands, by comparing the scan's window against
    the in-process ``compile_python`` evaluation of the same AST over the same
    corpus. If it ever fails, a walk is silently and permanently dropping
    documents — a post-filter can only narrow, never recover a row the window
    never returned.

    Scope, so a green run is not read as more than it is: eleven shapes on one
    store is a tripwire, not a proof over the operator space. The general
    property belongs to the compilers and to the forced-residual conformance
    corpus.
    """
    seed = scan_seed(6)
    await _seed_varied(backend, namespace.id, seed)
    ast = _filter_ast(wire)

    step = await backend.scan_documents(namespace.id, filter_ast=ast, scan_limit=100)
    # Precondition: the comparison below is only meaningful if this one window
    # covered the whole namespace. Without it, growing the corpus past the bound
    # would fail the test for a reason that has nothing to do with the pushdown.
    assert step.exhausted is True

    all_docs = (await backend.scan_documents(namespace.id, scan_limit=100)).documents
    matches = compile_python(ast, _documents_compile_context()).predicate
    oracle = {d.id for d in all_docs if matches(d)}

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
    await _write(backend, namespace.id, newest, base + timedelta(seconds=2), source_type="report")
    await _write(backend, namespace.id, middle, base + timedelta(seconds=1), source_type="report")
    await _write(backend, namespace.id, oldest, base, source_type="library")

    ast = _filter_ast(
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

    ``created_at`` lives in the raw row as TEXT, and the implementation builds the
    key from the converted :class:`Document` — where ``_parse_dt`` has already run
    — rather than off ``rows[-1]["created_at"]``.

    **Measured, by mutating the implementation to ``rows[-1]["created_at"]`` and
    reverting it:** that mutant is not silent on this store, contrary to the
    premise this test was originally written under. It fails 7 tests here — this
    one on the type assertion, and every walk with ``AttributeError: 'str' object
    has no attribute 'isoformat'`` when the raw string is bound back in through
    ``_dt_to_str``. So keep this test as the direct, legible statement of the
    declared contract, not as the only thing standing between a TEXT key and a
    green suite.

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
    await _seed_varied(backend, namespace.id, seed)

    step = await backend.scan_documents(namespace.id, scan_limit=3)
    assert step.last_scanned is not None
    created_at, doc_id = step.last_scanned
    assert isinstance(created_at, datetime)
    assert isinstance(doc_id, UUID)
    assert not isinstance(doc_id, bool)  # UUID has no bool subclass; guards a stub returning a truthy sentinel
    assert (created_at, doc_id) == (step.documents[-1].created_at, step.documents[-1].id)

    filtered = await backend.scan_documents(
        namespace.id,
        filter_ast=_filter_ast({"source_type": {"$eq": "report"}}),
        scan_limit=2,
    )
    assert filtered.last_scanned is not None
    assert isinstance(filtered.last_scanned[0], datetime)
    assert isinstance(filtered.last_scanned[1], UUID)


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
        await _write(
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
        await _write(
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
        filter_ast=_filter_ast({"source_type": {"$eq": "report"}}),
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
    predicate — ``conditions = ["namespace_id = ?"]`` / ``params = [str(namespace_id)]``
    becoming ``conditions = ["1"]`` / ``params = []``, the faithful SQLite form of
    "delete the scope", since an empty condition list would emit a bare ``WHERE``
    and fail as a syntax error rather than as wrong rows — fails this test.
    Measured: the filtered walk returns **8 rows instead of 4**, four of them the
    other tenant's, and the walk assertion fires before the unfiltered read below
    is reached. The grouping tripwire that follows fails on the same mutant, at
    **12 rows instead of 6**; those two are the only failures the mutant produces
    across this module, which is exactly why both of them exist.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    await _seed_varied(backend, namespace.id, _seed_from_ladder(scanned_ids))
    other = await _make_namespace(backend)
    # Same tie instant, same varied corpus: every foreign row is a guaranteed hit
    # for the filter below, so a dropped predicate fails on every run rather than
    # on a lucky arrangement.
    await _seed_varied(backend, other.id, _seed_from_ladder(foreign_ids))

    wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
    steps = await walk_scan(backend.scan_documents, namespace.id, scan_limit=1, filter_ast=_filter_ast(wire))
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
    await _seed_varied(backend, namespace.id, _seed_from_ladder(scanned_ids))
    other = await _make_namespace(backend)
    await _seed_varied(backend, other.id, _seed_from_ladder(foreign_ids))

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
        filter_ast=_filter_ast({"title": {"$eq": "doc-0"}}),
        scan_limit=50,
    )

    assert step.documents, "the fragment must match rows in the scanned namespace for this test to bite"
    assert len(step.documents) == 6
    assert all(d.namespace_id == namespace.id for d in step.documents)
    assert set(foreign_ids).isdisjoint({d.id for d in step.documents})
