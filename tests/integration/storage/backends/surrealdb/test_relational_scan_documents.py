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
* **Cursor and ``updated_before`` operands bind as Python objects.** Under the
  ``<`` compare both predicates here use, a stringified datetime matches
  nothing against a ``TYPE datetime`` field rather than mis-ordering. That is a
  property of the *operator*, not of the type pair — ``>=`` against the same
  pair is unconditionally true, so it pins the opposite way; see the
  datetime-binds note atop ``khora/storage/backends/surrealdb/relational.py``
  for all six operators. Measured: no bound = 6 rows, ``datetime`` bind = 6
  rows, ``.isoformat()`` bind = 0 rows.

Seeding goes through ``create_document``, the production write API, so every row
is serialized by the same path production writes take. Timestamps are pinned to
a whole second on purpose; see :mod:`tests.test_helpers.document_scan`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Document, MemoryNamespace, TenancyMode  # noqa: E402
from khora.core.models.document import DocumentStatus  # noqa: E402
from khora.filter import (  # noqa: E402
    CompiledFilter,
    CompileError,
    CompilerRegistry,
    RecallFilter,
    RecallFilterUnsupportedError,
)
from khora.filter.ast import parse_to_ast  # noqa: E402
from khora.filter.compilers.python import compile_python  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402

# ``_documents_compile_context`` is private, and imported on purpose: the
# superset test below must compile with the *same* context the scan itself uses,
# or it would prove a property of some other context.
from khora.storage.backends.surrealdb.relational import (  # noqa: E402
    SurrealDBRelationalAdapter,
    _documents_compile_context,
)
from tests.test_helpers.document_order import seed_order  # noqa: E402
from tests.test_helpers.document_scan import WHOLE_SECOND, ScanSeed, as_utc, walk_scan  # noqa: E402

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


def _filter_ast(wire: dict[str, Any]) -> Any:
    return parse_to_ast(RecallFilter.model_validate(wire))


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def _seed_from_ladder(ids: list[UUID], *, instant: datetime = WHOLE_SECOND) -> ScanSeed:
    """Build a :class:`ScanSeed` over a caller-supplied ascending id ladder.

    Identical in construction to :func:`tests.test_helpers.document_scan.scan_seed`
    — ``total - 2`` rows share ``instant`` so only the ``id DESC`` leg can order
    them, and the two rows outside that block carry timestamp and id in
    deliberate conflict. The one difference is that the ids come from the
    caller instead of from a fresh random ladder, which is what
    :func:`_two_namespace_ladders` needs to pin the relative order of two
    namespaces' ids. See that function for why that matters.
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


async def _write(adapter: Any, namespace_id: UUID, doc_id: UUID, created_at: datetime, **fields: Any) -> None:
    """Insert one document through the production write API."""
    await adapter.create_document(
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


async def _seed(adapter: Any, namespace_id: UUID, seed: ScanSeed) -> None:
    for doc_id, created_at in seed.writes:
        await _write(adapter, namespace_id, doc_id, created_at)


async def _seed_varied(adapter: Any, namespace_id: UUID, seed: ScanSeed) -> None:
    """Seed the same corpus with attribute variety, so a filter can split it.

    Attributes are assigned by *write* index, which is deliberately not the
    enumeration order — every expectation below is therefore derived from the
    rows a scan actually returns, never from this loop's counter.
    """
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await _write(
            adapter,
            namespace_id,
            doc_id,
            created_at,
            title=f"doc-{i}",
            source_type="report" if i % 2 == 0 else "library",
            metadata={"tier": "gold"} if i < 2 else {},
        )


def _seeded(n: int = 6) -> ScanSeed:
    """A single-namespace seed; ids may be random because nothing compares across
    namespaces."""
    return _seed_from_ladder(_ladder(uuid4().hex[:23], "1", n))


# --------------------------------------------------------------------------- #
# The window bound
# --------------------------------------------------------------------------- #


async def test_scan_limit_bounds_the_window(adapter, namespace) -> None:
    seed = _seeded()
    await _seed(adapter, namespace.id, seed)

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
    await _seed(adapter, namespace.id, seed)

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
    await _seed_varied(adapter, namespace.id, seed)

    wire = {
        "$or": [
            {"source_type": {"$eq": "no-such-source-type"}},
            {"occurred_at": {"$gte": "2999-01-01T00:00:00+00:00"}},
        ]
    }
    ast = _filter_ast(wire)
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
            filter_ast=_filter_ast({"source_type": {"$eq": "report"}}),
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
    await _seed(adapter, namespace.id, seed)

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
    await _seed(adapter, namespace.id, seed)

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
    await _seed_varied(adapter, namespace.id, seed)

    full = await adapter.scan_documents(namespace.id, scan_limit=10)
    wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
    expected = [d.id for d in full.documents if d.source_type == "report" or d.title == "doc-1"]
    assert 1 < len(expected) < len(full.documents), "the filter must narrow, but not to a single row"

    steps = await walk_scan(
        adapter.scan_documents,
        namespace.id,
        scan_limit=1,
        filter_ast=_filter_ast(wire),
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
    await _seed(adapter, namespace.id, seed)
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
        await _write(adapter, namespace.id, doc_id, created_at, source_type="report" if i % 2 == 0 else "library")

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=_filter_ast(
            {"source_type": {"$eq": "report"}, "occurred_at": {"$gte": "2999-01-01T00:00:00+00:00"}}
        ),
        scan_limit=10,
    )

    assert step.consumed_keys == frozenset({"source_type"})
    assert {d.source_type for d in step.documents} == {"report"}
    assert len(step.documents) == 3


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
    "pushable_date": {"created_at": {"$gte": "2026-01-31T12:30:00+00:00"}},
    "unpushable_key": {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
    "or_over_unpushable": {
        "$or": [
            {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}},
            {"source_type": {"$eq": "report"}},
        ]
    },
    "not_over_pushable": {"$not": {"source_type": {"$eq": "report"}}},
    "not_over_unpushable": {"$not": {"occurred_at": {"$gte": "2026-01-01T00:00:00+00:00"}}},
    "and_of_in_and_not": {
        "$and": [
            {"source_type": {"$in": ["report", "library"]}},
            {"$not": {"title": {"$eq": "doc-0"}}},
        ]
    },
}


@pytest.mark.parametrize("wire", _SUPERSET_SHAPES.values(), ids=_SUPERSET_SHAPES.keys())
async def test_pushdown_never_rejects_a_row_the_full_filter_would_keep(adapter, namespace, wire) -> None:
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
    seed = _seeded()
    await _seed_varied(adapter, namespace.id, seed)
    ast = _filter_ast(wire)

    step = await adapter.scan_documents(namespace.id, filter_ast=ast, scan_limit=100)
    # Precondition: the comparison below is only meaningful if this one window
    # covered the whole namespace. Without it, growing the corpus past the bound
    # would fail the test for a reason that has nothing to do with the pushdown.
    assert step.exhausted is True

    all_docs = (await adapter.scan_documents(namespace.id, scan_limit=100)).documents
    matches = compile_python(ast, _documents_compile_context()).predicate
    oracle = {d.id for d in all_docs if matches(d)}

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
    await _write(adapter, namespace.id, newest, base + timedelta(seconds=2), source_type="report")
    await _write(adapter, namespace.id, middle, base + timedelta(seconds=1), source_type="report")
    await _write(adapter, namespace.id, oldest, base, source_type="library")

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=_filter_ast(
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

    ``id`` on this store is a ``RecordID`` (``document:<uuid>``) in the raw row,
    and the implementation builds the key from the converted ``Document`` — where
    ``_row_to_document`` has already run the ``RecordID`` -> ``UUID`` conversion —
    rather than off ``rows[-1]["id"]``. This asserts the declared type directly,
    on both a mid-walk key and one taken under a filter.

    Scope note, because it was proposed to me as a mutant that "stays green
    across every walk test": on this store it does not. Building the key from the
    raw row instead fails five tests in this module, the walks dying inside the
    SDK with ``ValueError: Failed to decode CBOR request`` when the ``RecordID``
    is bound back in as a cursor operand, and the raw-window test above failing
    on the tuple compare. So this test is a *clearer* statement of the contract,
    not the only thing standing between a ``RecordID`` key and a green suite.
    Keep it for the former reason.
    """
    seed = _seeded()
    await _seed_varied(adapter, namespace.id, seed)

    step = await adapter.scan_documents(namespace.id, scan_limit=3)
    assert step.last_scanned is not None
    created_at, doc_id = step.last_scanned
    assert isinstance(doc_id, UUID)
    assert not isinstance(doc_id, bool)  # UUID has no bool subclass; guards a stub returning a truthy sentinel
    assert isinstance(created_at, datetime)

    filtered = await adapter.scan_documents(
        namespace.id,
        filter_ast=_filter_ast({"source_type": {"$eq": "report"}}),
        scan_limit=2,
    )
    assert filtered.last_scanned is not None
    assert isinstance(filtered.last_scanned[1], UUID)


async def test_a_hyphenated_metadata_key_in_a_deferred_subtree_does_not_raise(adapter, namespace) -> None:
    """The ``RecallFilterUnsupportedError`` mapping is sibling-dependent, by design.

    The injection guard fires only when the emit walk *reaches* the offending
    leaf. Inside an ``$or`` that ``compile_surrealdb``'s all-or-nothing gate
    defers wholesale for an unrelated reason — here the unbacked ``occurred_at``
    sibling — the subtree never emits, the hyphenated segment is never rendered,
    and the whole filter goes to the caller's post-filter, which handles
    hyphenated keys correctly. Rows are right either way.

    This is asserted **positively**, as a non-raise, on purpose. The tempting
    assertion is the unconditional one — "a hyphenated metadata key always
    raises" — and it would be wrong: it would pin behaviour the compiler does not
    promise and would block the documented future improvement of routing an
    unrenderable segment onto the unsupported path under ``"split"`` instead of
    raising. Its conjunctive sibling
    (:func:`test_unsafe_metadata_segment_raises_the_public_error`) pins the case
    that *is* promised.
    """
    seed = _seeded()
    await _seed_varied(adapter, namespace.id, seed)

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=_filter_ast(
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

    This store's ``updated_at`` is ``TYPE datetime``. Under the ``<`` this
    bound uses, a stringified operand matches no row — so a walk reports itself
    exhausted at the first step and the caller sees an empty namespace rather
    than an error. Direction-specific: the same string under ``>=`` is
    unconditionally true and would pin the bound wide open instead of shut (see
    the module docstring). Measured on an in-memory instance over this corpus: no
    bound = 6 rows, ``datetime`` bind = 6 rows, ``.isoformat()`` bind = 0 rows.

    That is why the assertion below is on *which* rows came back, not merely on
    a count: the string form's 0 rows and a correct 4 are both "narrower than 6",
    and only an exact row set separates a working bound from a broken one.

    ``updated_before`` binds a ``datetime`` object here and in
    :meth:`list_documents`; the string form is measured at 0 rows above and is
    covered directly in ``test_list_documents_updated_before.py``.
    """
    seed = _seeded()
    cutoff = seed.tie_instant + timedelta(hours=1)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await _write(
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
        await _write(
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
        filter_ast=_filter_ast({"source_type": {"$eq": "report"}}),
        status=DocumentStatus.COMPLETED.value,
        updated_before=cutoff,
        after=(cursor_doc.created_at, cursor_doc.id),
        scan_limit=50,
    )

    assert [d.id for d in step.documents] == expected
    assert step.consumed_keys == frozenset({"source_type"})


# --------------------------------------------------------------------------- #
# The CompileError mapping
# --------------------------------------------------------------------------- #


async def test_unsafe_metadata_segment_raises_the_public_error(adapter, namespace) -> None:
    """A hyphenated metadata key is a caller mistake, not an internal fault.

    ``compile_surrealdb``'s injection guard raises the internal ``CompileError``
    on a metadata path segment it cannot render as a SurrealQL identifier — and
    ``metadata.due-date`` is legal JSON and common in the wild, so a caller can
    provoke it with a perfectly well-formed filter. ``CompileError`` is
    documented as "a bug, not a capability gap", so letting it escape would
    present a user-input problem as an internal error. ``scan_documents`` maps it
    to the public ``RecallFilterUnsupportedError``.

    The reported path is the ``metadata`` root rather than the offending dotted
    path, because the guard reports only the segment — asserted, so a later
    change that starts reporting the full path is a visible decision rather than
    a silent one.
    """
    with pytest.raises(RecallFilterUnsupportedError) as excinfo:
        await adapter.scan_documents(
            namespace.id,
            filter_ast=_filter_ast({"metadata.foo-bar": {"$eq": "x"}}),
            scan_limit=10,
        )

    assert not isinstance(excinfo.value, CompileError)
    assert "foo-bar" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, CompileError)


async def test_an_unrelated_compile_error_still_propagates(adapter, namespace, monkeypatch) -> None:
    """The catch is narrow — it maps ONE message, it does not swallow the class.

    ``scan_documents`` discriminates on the guard's message substring because the
    guard has no error subclass of its own. The hazard in that design is the
    obvious widening: an ``except CompileError: raise RecallFilterUnsupportedError``
    with no discriminator would relabel every genuine compiler bug as a user
    input problem and hide it behind a 4xx-shaped error forever. Nothing in the
    filter language can provoke a second ``CompileError`` on this path today —
    ``grep -rn "raise CompileError" src/khora/filter/`` returns exactly one hit —
    so the only way to test the other branch is to inject one through the
    registry, which is also how the real lookup reaches its compiler.

    **This is not hypothetical.** An intermediate revision of ``scan_documents``
    dropped the discriminator and kept only the scope narrowing, on the (true)
    reasoning that the compiler has a single raise site. This test failed against
    it: the injected bug came back as ``RecallFilterUnsupportedError: metadata:
    internal compiler invariant violated…`` — relabelled as a caller input
    problem and attributed to a ``metadata`` path it had nothing to do with. The
    discriminator was restored. Do not re-widen the catch without deleting this
    test on purpose.
    """

    def broken_compiler(ast, ctx):
        raise CompileError("internal compiler invariant violated while emitting a node")

    monkeypatch.setitem(CompilerRegistry._registry, _COMPILER_KEY, broken_compiler)  # noqa: SLF001

    with pytest.raises(CompileError, match="internal compiler invariant"):
        await adapter.scan_documents(
            namespace.id,
            filter_ast=_filter_ast({"source_type": {"$eq": "report"}}),
            scan_limit=10,
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
    await _seed_varied(adapter, namespace.id, _seed_from_ladder(scanned_ids))
    other = await _make_namespace(adapter)
    await _seed_varied(adapter, other.id, _seed_from_ladder(foreign_ids))

    wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
    steps = await walk_scan(adapter.scan_documents, namespace.id, scan_limit=1, filter_ast=_filter_ast(wire))
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

    No compiler emits an ungrouped fragment today (``compile_surrealdb``
    self-groups its boolean nodes), so no compiled-filter test can reach this;
    the registered compiler is therefore replaced with one that emits the bare
    ungrouped shape, and the real ``scan_documents`` is called. Both namespaces'
    rows satisfy the right disjunct, so an absorbed scope predicate yields
    foreign rows deterministically rather than by luck.

    **Mutation-verified, and reverted afterwards.** Removing the parentheses at
    the splice — ``conditions.append(f"({compiled.predicate})")`` becoming
    ``conditions.append(compiled.predicate)`` — fails this test, and the measured
    magnitude is **6 rows becomes 12**: a full cross-tenant read of both
    namespaces, with no error and nothing in the logs.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    await _seed_varied(adapter, namespace.id, _seed_from_ladder(scanned_ids))
    other = await _make_namespace(adapter)
    await _seed_varied(adapter, other.id, _seed_from_ladder(foreign_ids))

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
        filter_ast=_filter_ast({"title": {"$eq": "doc-0"}}),
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

    **Mutation-verified, and reverted afterwards.** Dropping the outer pair from
    the ``conditions`` entry — ``"(created_at < $after_created_at OR (created_at
    = $after_created_at AND id < $after_id))"`` becoming the same string without
    its enclosing parentheses — fails this test: the resumed window returns **8
    rows instead of 4**, the four extra being the other tenant's tie block. The
    same mutant also fails the filtered-walk and namespace-isolation tests above,
    both of which resume across pages.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    seed = _seed_from_ladder(scanned_ids)
    await _seed(adapter, namespace.id, seed)
    other = await _make_namespace(adapter)
    await _seed(adapter, other.id, _seed_from_ladder(foreign_ids))

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
