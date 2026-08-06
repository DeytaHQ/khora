"""``SQLiteLanceRelationalAdapter.scan_documents`` — the bounded keyset scan.

Runs against the real Alembic-migrated SQLite schema (no Docker, no services),
which matters more here than usual: this store keeps ``created_at`` as TEXT and
compares it lexicographically, so the scan's cursor is only correct if it is
bound in the store's own serialization. That property cannot be checked against
a mock, and it is shared with the PostgreSQL leg — both stores build the same
statement through ``build_documents_scan_query`` — so this locally-runnable
module carries most of the semantics and its PostgreSQL sibling
(``tests/integration/storage/test_scan_documents_pg.py``) covers what only a
live server can show.

Seeding goes through ``create_document``, the production write API, so every row
is serialized by the same path production writes take. Timestamps are pinned to
a whole second on purpose; see :mod:`tests.test_helpers.document_scan`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple
from uuid import uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

import sqlalchemy.exc

from khora.core.models import MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentStatus
from khora.filter.compilers.python import compile_python
from tests.test_helpers.document_scan import (
    scan_seed,
    seed_documents,
    seed_varied,
    walk_scan,
    wire_to_ast,
    write_document,
)

# Lives in the unit lane next to the rest of this adapter's tests, and also
# carries the ``embedded`` marker so ``make test-embedded`` — the no-Docker
# command a developer reaches for to check this stack — actually selects it.
pytestmark = [
    pytest.mark.embedded,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]

if _HAS_EMBEDDED:
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )

    # ``_documents_compile_context`` is private, and imported on purpose: the
    # superset test below must use the *same* context the scan itself compiles
    # with, or it would prove a property of some other context.
    from khora.storage.backends.sqlite_lance.relational import (
        SQLiteLanceRelationalAdapter,
        _documents_compile_context,
    )


@pytest.fixture
async def adapter(migrated_sqlite_db, tmp_path):
    handle = EmbeddedStorageHandle(
        EmbeddedStorageHandleConfig(db_path=str(migrated_sqlite_db), lance_path=str(tmp_path / "khora.lance")),
    )
    adapter = SQLiteLanceRelationalAdapter(handle)
    await adapter.connect()
    try:
        yield adapter
    finally:
        await adapter.disconnect()
        await handle.disconnect()


@pytest.fixture
async def namespace(adapter):
    nid = uuid4()
    return await adapter.create_namespace(MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED))


# --------------------------------------------------------------------------- #
# The window bound
# --------------------------------------------------------------------------- #


async def test_scan_limit_bounds_the_window(adapter, namespace) -> None:
    seed = scan_seed(6)
    await seed_documents(adapter, namespace.id, seed)

    step = await adapter.scan_documents(namespace.id, scan_limit=2)

    assert [d.id for d in step.documents] == seed.expected[:2]
    assert step.last_scanned == (step.documents[-1].created_at, step.documents[-1].id)
    assert step.exhausted is False


async def test_a_full_window_is_not_yet_exhausted(adapter, namespace) -> None:
    """``exhausted`` means SQL ran short, not "the caller has seen everything".

    A window filled exactly to the bound cannot distinguish "six rows and no
    more" from "six rows and a seventh waiting", so it must report not-exhausted
    and let the next step find the empty tail. Reporting exhaustion here would
    silently truncate every namespace whose size is a multiple of the bound.
    """
    seed = scan_seed(6)
    await seed_documents(adapter, namespace.id, seed)

    exact = await adapter.scan_documents(namespace.id, scan_limit=6)
    assert len(exact.documents) == 6
    assert exact.exhausted is False

    over = await adapter.scan_documents(namespace.id, scan_limit=7)
    assert len(over.documents) == 6
    assert over.exhausted is True


async def test_scan_limit_below_one_is_rejected(adapter, namespace) -> None:
    """A zero bound would return an empty window that reports neither a resume
    position nor exhaustion — the one pair a walking caller cannot act on."""
    with pytest.raises(ValueError, match="scan_limit"):
        await adapter.scan_documents(namespace.id, scan_limit=0)


# --------------------------------------------------------------------------- #
# The keyset cursor
# --------------------------------------------------------------------------- #


async def test_walk_visits_every_document_exactly_once_in_total_order(adapter, namespace) -> None:
    """One row per step across a tie block, chaining ``last_scanned``.

    ``scan_limit=1`` puts a cursor boundary between every pair of rows,
    including between rows that share a ``created_at`` to the microsecond — so
    every resume in this walk is a mid-tie resume, and the ``id DESC`` leg
    decides all of them.
    """
    seed = scan_seed(6)
    await seed_documents(adapter, namespace.id, seed)

    steps = await walk_scan(adapter.scan_documents, namespace.id, scan_limit=1)
    seen = [d.id for step in steps for d in step.documents]

    assert len(seen) == len(set(seen))  # no document served twice
    assert set(seen) == set(seed.expected)  # every document served
    assert seen == seed.expected  # and in one total order across the concatenation
    assert steps[-1].documents == []
    assert steps[-1].last_scanned is None
    assert steps[-1].exhausted is True


async def test_whole_second_cursor_excludes_its_own_row_and_keeps_its_tie_mates(adapter, namespace) -> None:
    """The cursor round-trips exactly at ``.000000`` microseconds.

    This store writes ``created_at`` as TEXT and orders it lexicographically, so
    a cursor is only correct if it is serialized byte-for-byte the way the
    column was written. The two ways to get that wrong fail in opposite
    directions and neither raises, so both are asserted:

    * A hand-built ISO-8601 string sorts ABOVE every stored value, because
      ``'T'`` outranks the stored space separator. The window then matches the
      cursor's OWN row, and a walk chaining ``last_scanned`` never advances.
    * A space-separated form that omits the six-digit microsecond field — which
      is what ``str(datetime)`` produces at a whole second, and what a bare
      ``datetime`` handed to an untyped bind produces via the driver's
      deprecated adapter — sorts BELOW its tie-mates, silently skipping them.

    At non-zero microseconds the second form is byte-identical to the stored
    value and this test would pass against it, which is why the seed pins a
    whole second rather than sampling the clock.
    """
    seed = scan_seed(6)
    await seed_documents(adapter, namespace.id, seed)

    full = await adapter.scan_documents(namespace.id, scan_limit=10)
    assert [d.id for d in full.documents] == seed.expected

    cursor_doc = next(d for d in full.documents if d.id == seed.tied_ids[0])
    # The whole second survived the write/read round trip, so the divergence
    # between the stored form and a hand-formatted one is live in this corpus.
    assert cursor_doc.created_at.microsecond == 0
    assert cursor_doc.created_at.replace(tzinfo=UTC) == seed.tie_instant

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

    This store's compiler emits positional binds for a qmark driver, which are
    rewritten to ``:kf0`` … ``:kfN`` before the fragment can join a SQLAlchemy
    statement — while the keyset predicate and the namespace scope carry
    SQLAlchemy's own generated names (``param_N``, ``namespace_id_1``). The
    ``kf`` prefix is what keeps the two families apart, and nothing proves it
    unless both appear in the same ``SELECT``.

    The filter is a two-leaf disjunction on purpose: it compiles to two
    positional binds rather than one, so a rewrite that collided or dropped a
    name would show up as a wrong row set rather than as a lucky pass.
    """
    seed = scan_seed(6)
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


async def test_off_type_cursor_operands_are_rejected(adapter, namespace) -> None:
    """A position must be the pair of values a row yielded, not a rendering of it.

    Binding each operand through its ORM column type turns the two ways a caller
    could hand-render a position into an immediate failure instead of a silent
    mis-ordering: the id's type processor wants a ``UUID`` (it reads ``.hex``)
    and the timestamp's wants a ``datetime``. SQLAlchemy raises during bind
    processing, so the error surfaces wrapped rather than as the driver-level
    ``AttributeError`` / ``TypeError`` underneath.

    This has no PostgreSQL counterpart — no bind processor runs on that dialect,
    so the same operands reach asyncpg untouched and a dashed id string decodes
    to the very same bytes. The contract is enforced here and documented there.
    """
    seed = scan_seed(6)
    await seed_documents(adapter, namespace.id, seed)
    row = (await adapter.scan_documents(namespace.id, scan_limit=1)).documents[0]

    with pytest.raises(sqlalchemy.exc.StatementError) as rendered_id:
        await adapter.scan_documents(namespace.id, after=(row.created_at, str(row.id)))
    assert isinstance(rendered_id.value.orig, AttributeError)

    with pytest.raises(sqlalchemy.exc.StatementError) as rendered_ts:
        await adapter.scan_documents(namespace.id, after=(str(row.created_at), row.id))
    assert isinstance(rendered_ts.value.orig, TypeError)


async def test_empty_window_reports_exhausted_without_a_position(adapter, namespace) -> None:
    """Both the never-seeded namespace and the tail past the last row."""
    empty = await adapter.scan_documents(namespace.id, scan_limit=5)
    assert empty.documents == []
    assert empty.last_scanned is None
    assert empty.exhausted is True

    seed = scan_seed(6)
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


async def test_date_system_keys_are_not_pushed_down_by_this_store(adapter, namespace) -> None:
    """``created_at`` reaches the caller's post-filter here, and that is intended.

    This store withholds the date-valued system keys from its pushdown
    whitelist because its stored ``DATETIME`` text does not order against the
    compiler's ISO binds. The leaf therefore compiles to a match-all placeholder,
    stays out of ``consumed_keys``, and narrows nothing — asserted positively,
    because a widened whitelist would show up here as a suddenly-shorter window
    rather than as an error. PostgreSQL reports the same filter as pushed; the
    two answers differing is the intended split, not drift.
    """
    seed = scan_seed(6)
    await seed_documents(adapter, namespace.id, seed)

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast({"created_at": {"$gte": "2999-01-01T00:00:00+00:00"}}),
        scan_limit=10,
    )

    assert step.consumed_keys == frozenset()
    # Every row is older than the bound, so a pushed-down comparison would have
    # returned nothing at all.
    assert [d.id for d in step.documents] == seed.expected


async def test_split_reports_only_the_leaves_sql_enforced(adapter, namespace) -> None:
    """A mixed filter: one pushable leaf, one that must reach the post-filter."""
    seed = scan_seed(6)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await write_document(
            adapter, namespace.id, doc_id, created_at, source_type="report" if i % 2 == 0 else "library"
        )

    step = await adapter.scan_documents(
        namespace.id,
        filter_ast=wire_to_ast({"source_type": {"$eq": "report"}, "created_at": {"$gte": "2026-01-01T00:00:00+00:00"}}),
        scan_limit=10,
    )

    assert step.consumed_keys == frozenset({"source_type"})
    # The pushed leaf really did narrow the window; the unpushed one did not.
    assert {d.source_type for d in step.documents} == {"report"}
    assert len(step.documents) == 3


class _Shape(NamedTuple):
    """One superset parametrization, carrying what makes it non-vacuous.

    ``consumed`` is MEASURED on this store. The raw-SQLite module
    (``tests/unit/test_sqlite_scan_documents.py``) carries a table of the same
    shape and shares ``compile_lance`` with this one, but the numbers are not
    interchangeable — the two stores build different compile contexts — so each
    module measures its own even where the wire form is identical.
    """

    wire: dict[str, Any]
    consumed: frozenset[str]
    """The exact ``consumed_keys`` this shape reports here. Empty means the whole
    filter deferred to the caller's post-filter."""


# The pushdown must never reject a row the full filter would keep. Shapes are
# chosen for the ways a compiler can get that wrong, not for operator coverage:
# the three wrapping an unpushable leaf in a disjunction or a negation are the
# ones that matter, because a match-all placeholder left inside a negation
# inverts into a match-nothing and excludes rows.
#
# ``pushable_exists`` was ``{"source_url": {"$exists": False}}`` until this change,
# measured on THIS store at oracle 0 / window 0: ``source_url`` is a system
# column ``create_document`` always writes, so ``$exists: False`` is
# constant-false and the parametrization could not fail for any seed. (The same
# shape was vacuous in the raw-SQLite and SurrealDB scan modules for the same
# corpus-independent reason; all three were swapped together, since fixing some
# of them and leaving the rest is the worst outcome.) The replacement measures
# 4 oracle / 4 window here and exercises the metadata-presence path rather than
# the constant-false system-key one.
_SUPERSET_SHAPES: dict[str, _Shape] = {
    "pushable_eq": _Shape({"source_type": {"$eq": "report"}}, frozenset({"source_type"})),
    "pushable_ne": _Shape({"source_type": {"$ne": "report"}}, frozenset({"source_type"})),
    "pushable_nin": _Shape({"source_type": {"$nin": ["report"]}}, frozenset({"source_type"})),
    "pushable_exists": _Shape({"metadata.tier": {"$exists": False}}, frozenset({"metadata.tier"})),
    "metadata_eq": _Shape({"metadata.tier": {"$eq": "gold"}}, frozenset({"metadata.tier"})),
    "unpushable_date": _Shape({"created_at": {"$gte": "2026-01-31T12:30:00+00:00"}}, frozenset()),
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
async def test_pushdown_never_rejects_a_row_the_full_filter_would_keep(adapter, namespace, shape) -> None:
    """The superset property the resume contract depends on.

    Resuming past the rows a pushdown rejected is sound only because a rejected
    row could not have satisfied the full filter either. Both ``scan_documents``
    docstrings name that as an assumption about the *compilers*; this checks the
    consequence where it actually lands, by comparing the scan's window against
    the in-process ``compile_python`` evaluation of the same AST over the same
    corpus. If it ever fails, a walk is silently and permanently dropping
    documents — a post-filter can only narrow, never recover a row the window
    never returned.

    **Scope, counted from this module's own instrumentation rather than
    asserted: of the ten shapes, SEVEN can fail the subset assertion and three
    cannot.** The three are ``unpushable_date`` (oracle 5 / window 6),
    ``or_over_unpushable`` (5/6) and ``not_over_unpushable`` (1/6): each reports
    ``consumed_keys == frozenset()``, so nothing reached the ``WHERE``, the window
    IS the whole namespace, and the oracle is computed by filtering that same
    window — which makes ``oracle <= window`` **mathematically incapable of
    failing for any oracle value**. They are structurally unfailable on the subset
    assertion and this docstring is the only place that says so. Do not try to
    detect that state by comparing the window against the corpus: a pushdown bug
    that wrongly *rejected* rows would shrink the window and the subset assertion
    would fire, so an equal-sized window is not evidence of an incapable test. The
    discriminator is ``consumed_keys == frozenset()``.

    Two anti-vacuity assertions therefore ship alongside the subset one:

    * ``assert oracle`` — an empty oracle is a subset of anything. It catches a
      real mutation: hollow out :func:`~tests.test_helpers.document_scan.seed_varied` so no row is
      ``source_type="report"`` and ``pushable_eq`` goes from 3/3 to 0/0 while
      staying green. Unlike the raw-SQLite module, no shape here needs an
      exemption — every one of the ten has a non-empty oracle on this corpus
      (that module carries an ``occurred_at`` shape whose oracle is empty by
      construction; this one does not).
    * ``consumed_keys`` equality per shape — the only thing that gives the three
      deferring shapes any value, and the only thing that fails when a compile
      context or ``field_mapping`` edit silently moves a shape from has-teeth to
      unfailable, or the reverse.

    Measured window/oracle sizes over the 6-row ``seed_varied`` corpus, this
    store, this revision: ``pushable_eq`` 3/3, ``pushable_ne`` 3/3,
    ``pushable_nin`` 3/3, ``pushable_exists`` 4/4, ``metadata_eq`` 2/2,
    ``not_over_pushable`` 3/3, ``and_of_in_and_not`` 5/5, then the three
    deferring shapes above.
    """
    seed = scan_seed(6)
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

    assert oracle, "the oracle is empty, so the subset assertion below cannot fail — re-seed or re-shape"
    assert step.consumed_keys == shape.consumed

    assert oracle <= {d.id for d in step.documents}


# --------------------------------------------------------------------------- #
# What the position means
# --------------------------------------------------------------------------- #


async def test_last_scanned_is_the_final_raw_row_not_the_last_match(adapter, namespace) -> None:
    """Resume from the last row SCANNED, not from the last row that matches.

    The window here deliberately ENDS on a row the caller's post-filter will
    reject: the filter is an ``$or`` mixing a pushable leaf with an unbacked
    one, which the compiler defers as a whole rather than pushing half of a
    disjunction, so SQL narrows nothing and the oldest row — which does not
    satisfy the filter — is the last row of the raw window.

    A walk that resumed from the last *matching* row instead would re-scan the
    rejected gap on every step — and when a whole window is rejected there is no
    matching row to resume from at all, so such a walk cannot advance past a run
    of non-matching rows longer than one window. Taking the position from the
    raw window is what lets ``exhausted`` be the only termination signal.
    """
    newest, middle, oldest = (uuid4() for _ in range(3))
    base = datetime(2026, 1, 31, 12, 30, tzinfo=UTC)
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


# --------------------------------------------------------------------------- #
# The non-filter narrowing legs
# --------------------------------------------------------------------------- #


async def test_status_and_updated_before_narrow_the_window(adapter, namespace) -> None:
    seed = scan_seed(6)
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

    by_updated = await adapter.scan_documents(namespace.id, updated_before=cutoff, scan_limit=10)
    assert {d.id for d in by_updated.documents} == {doc_id for i, (doc_id, _) in enumerate(seed.writes) if i < 4}


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------
#
# Everything above runs in a single-namespace fixture, so no assertion up there
# can notice a scan that ignores its namespace scope — deleting the namespace
# predicate from ``build_documents_scan_query`` left the whole module green
# (found by two reviewers independently, by mutation). These two tests exist to
# make that mutant, and the fragment-grouping mutant, fail.


async def test_scan_never_returns_another_namespaces_rows(adapter, namespace) -> None:
    """A filtered walk over one namespace must not see a byte of the other.

    The second namespace is seeded with the SAME varied corpus, so every row in
    it matches the same filter — if the namespace predicate is dropped (or
    stops AND-composing with the fragment), the foreign rows are not merely
    reachable, they are guaranteed hits. Walked at ``scan_limit=1`` so the
    keyset predicate is exercised across pages too: the cursor is namespace-
    blind on its own, and only the scope predicate keeps a resume inside its
    tenant.
    """
    seed = scan_seed(6)
    await seed_varied(adapter, namespace.id, seed)

    other_id = uuid4()
    other = await adapter.create_namespace(
        MemoryNamespace(id=other_id, namespace_id=other_id, tenancy_mode=TenancyMode.SHARED)
    )
    await seed_varied(adapter, other.id, scan_seed(6))

    wire = {"$or": [{"source_type": {"$eq": "report"}}, {"title": {"$eq": "doc-1"}}]}
    steps = await walk_scan(adapter.scan_documents, namespace.id, scan_limit=1, filter_ast=wire_to_ast(wire))
    seen = [d for step in steps for d in step.documents]

    assert seen, "the filter must match rows in the scanned namespace for this test to bite"
    assert all(d.namespace_id == namespace.id for d in seen)

    unfiltered = await adapter.scan_documents(namespace.id, scan_limit=50)
    assert len(unfiltered.documents) == 6
    assert all(d.namespace_id == namespace.id for d in unfiltered.documents)


async def test_ungrouped_or_fragment_cannot_absorb_the_namespace_scope(adapter, namespace) -> None:
    """The splice's parentheses are load-bearing — proven on rows, not prose.

    No compiler emits an ungrouped fragment today (``compile_lance``
    self-parenthesizes every boolean node), so no compiled-filter test can
    catch ``_lance_fragment_to_text`` losing its grouping. This drives the
    splice directly with a top-level ``OR`` — the shape that, ungrouped,
    absorbs the namespace predicate into its left disjunct
    (``ns = ? AND a = 1 OR b = 2``) and returns the other tenant's rows.
    Removing the parentheses from ``_lance_fragment_to_text`` must fail here.
    """
    from khora.storage.backends._documents_scan import build_documents_scan_query
    from khora.storage.backends.sqlite_lance.relational import _lance_fragment_to_text

    seed = scan_seed(6)
    await seed_varied(adapter, namespace.id, seed)
    other_id = uuid4()
    other = await adapter.create_namespace(
        MemoryNamespace(id=other_id, namespace_id=other_id, tenancy_mode=TenancyMode.SHARED)
    )
    await seed_varied(adapter, other.id, scan_seed(6))

    # Every row in BOTH namespaces satisfies the right disjunct, so an absorbed
    # namespace predicate yields foreign rows deterministically.
    fragment = _lance_fragment_to_text("documents.title = ? OR documents.content = ?", ["doc-0", "scanned content"])
    query = build_documents_scan_query(namespace.id, scan_limit=50).where(fragment)

    async with adapter._get_session() as session:  # noqa: SLF001 — the splice has no public executor yet
        rows = (await session.execute(query)).scalars().all()

    assert rows, "the fragment must match rows in the scanned namespace for this test to bite"
    assert all(row.namespace_id == namespace.id for row in rows)
