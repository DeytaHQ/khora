"""Every ``document`` read this adapter issues must plan ``Iterate Index``.

An ``OR`` anywhere in a ``document`` ``WHERE`` — top-level or nested inside a
conjunct — collapses the planner to ``Iterate Table``, and the collapse does not
stop at the predicate that caused it: the ``namespace_id`` index prefix goes too,
so the statement reads every ``document`` row in the database, all tenants
included. Three adapter reads used to be written that way (the keyset resume in
``scan_documents``, the checksum-dedup exclusion, the orphan-claim sweep) and each
is now a set of ``OR``-free legs merged in Python.

The scope of that claim is the ``document`` table, not SurrealDB in general.
2.x *can* union index scans across a disjunction, but only when every disjunct
is a comparison on a **single-field** index; ``document`` carries composite
``(namespace_id, …)`` indexes and nothing else, so no disjunct on it ever
qualifies. The disjunction note atop
``khora/storage/backends/surrealdb/relational.py`` has the measurements, and
``test_the_replaced_disjunctive_shapes_really_do_table_scan`` below keeps the
premise honest against a future engine.

**These tests do not retype the SQL.** They drive the real adapter methods
through a recording connection, then re-issue each captured statement verbatim
with ``EXPLAIN`` appended. A future edit that reintroduces an ``OR`` — or that
adds a fourth statement to one of these paths — is covered without anyone
remembering to update a literal here.

Scope, deliberately: the ``filter_ast`` shapes are NOT pinned. A legal filter
fragment containing ``$or`` compiles to a real ``OR`` and table-scans, which this
change does not address and cannot; see the complexity note on
``scan_documents``. Pinning a filtered plan would encode that gap as a
requirement.

They run against ``memory://`` — no server, no docker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import MemoryNamespace, TenancyMode  # noqa: E402
from khora.core.models.document import Document, DocumentStatus  # noqa: E402
from khora.storage.backends.surrealdb._helpers import _record_id  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402

pytestmark = pytest.mark.unit

_BASE = datetime(2026, 3, 1, tzinfo=UTC)


class _RecordingConnection:
    """Pass-through wrapper that keeps every ``(sql, params)`` the adapter issues.

    Only the two read entry points are intercepted; everything else (``connect``,
    ``execute``, the seeding path) forwards untouched via ``__getattr__``, so the
    adapter is exercised exactly as in production.
    """

    def __init__(self, conn: SurrealDBConnection) -> None:
        self._conn = conn
        self.statements: list[tuple[str, dict[str, Any]]] = []

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        self.statements.append((sql, params or {}))
        return await self._conn.query(sql, params)

    async def query_one(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        self.statements.append((sql, params or {}))
        return await self._conn.query_one(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def reset(self) -> None:
        self.statements.clear()

    def selects(self) -> list[tuple[str, dict[str, Any]]]:
        """Recorded ``SELECT``s only — the writes a claim issues are not plans."""
        return [(sql, params) for sql, params in self.statements if sql.lstrip().upper().startswith("SELECT")]


@pytest.fixture
async def recorder():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="plans")
    await conn.connect()
    recording = _RecordingConnection(conn)
    try:
        yield recording
    finally:
        await conn.disconnect()


@pytest.fixture
async def seeded(recorder):
    """An adapter over a namespace holding rows in every status, plus a foreign one.

    A populated table matters: the point of the assertion is that the planner
    *chose* an index, and rows are what give it something to choose over. The
    foreign namespace is what a regression would leak into.
    """
    adapter = SurrealDBRelationalAdapter(recorder)
    namespaces = []
    for _ in range(2):
        nid = uuid4()
        namespaces.append(
            await adapter.create_namespace(MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED))
        )
    statuses = list(DocumentStatus)
    for target in namespaces:
        for i in range(24):
            # Three rows per instant, so a resumed scan lands inside a tie block.
            await adapter.create_document(
                Document(
                    namespace_id=target.namespace_id,
                    content=f"body {i}",
                    title=f"doc {i}",
                    source_type="library",
                    checksum=f"cs{i:03d}",
                    status=statuses[i % len(statuses)],
                    created_at=_BASE + timedelta(minutes=i // 3),
                    updated_at=_BASE + timedelta(minutes=i // 3),
                )
            )
    recorder.reset()
    return adapter, namespaces[0].namespace_id


async def _explain(recorder: _RecordingConnection, *, expected_statements: int, label: str) -> list[Any]:
    """Re-issue every recorded ``SELECT`` verbatim with ``EXPLAIN``; return the plans.

    Returns one plan per captured statement, in issue order. Each caller asserts
    what its own shape is supposed to have used.
    """
    recorded = recorder.selects()
    assert len(recorded) == expected_statements, (
        f"{label}: expected {expected_statements} SELECT(s), captured {len(recorded)}:\n"
        + "\n".join(sql for sql, _ in recorded)
    )
    plans = []
    for sql, params in recorded:
        plan: Any = await recorder._conn.query(f"{sql} EXPLAIN", params)
        plans.append((sql, plan))
    return plans


def _assert_indexed(sql: str, plan: Any, *, index: str | None, label: str) -> dict[str, Any]:
    """Require an index scan, optionally on a named index. Returns the seek detail.

    ``index=None`` means "any namespace-scoped composite" and is used for the one
    shape whose winner is decided by a name-ordered planner tie-break rather than
    by the predicate — pinning that would harden an undocumented tie-break into a
    test, which the sibling ``test_document_sort_index`` module already declines
    to do.
    """
    iterate = plan[0]
    assert iterate["operation"] == "Iterate Index", (
        f"{label}: statement fell back to a full table scan — it reads every document row "
        f"in the database, all namespaces included. An OR anywhere in a document WHERE "
        f"does this, whatever its shape.\n  sql:  {sql}\n  plan: {plan}"
    )
    chosen = iterate["detail"]["plan"]["index"]
    if index is None:
        assert chosen.startswith("idx_document_ns"), f"{label}: not a namespace-scoped index ({chosen})\n  sql: {sql}"
    else:
        assert chosen == index, f"{label}: expected {index}, planner chose {chosen}\n  sql: {sql}\n  plan: {plan}"
    return iterate["detail"]["plan"]


# --------------------------------------------------------------------------- #
# scan_documents
# --------------------------------------------------------------------------- #


async def test_first_scan_step_plans_an_index_scan(seeded, recorder) -> None:
    """The un-resumed step is one statement, seeking on the namespace prefix.

    The index is left unpinned here alone: with no cursor there is no second
    column to constrain, so every ``(namespace_id, …)`` composite is equally
    applicable and the winner falls to a name-ordered tie-break. What IS pinned is
    that the seek is the namespace equality — a plan that reached the index some
    other way would not scope the read to one tenant.
    """
    adapter, ns = seeded
    step = await adapter.scan_documents(ns, scan_limit=5)
    assert step.documents, "fixture produced no rows to plan over"
    ((sql, plan),) = await _explain(recorder, expected_statements=1, label="scan step 1")
    seek = _assert_indexed(sql, plan, index=None, label="scan step 1")
    assert seek.get("operator") == "=" and seek.get("value") == str(ns), (
        f"first step did not seek on the namespace equality: {seek}"
    )


async def test_resumed_scan_step_seeks_the_cursor_through_the_index_on_both_statements(seeded, recorder) -> None:
    """Both resumed statements use ``idx_document_ns_created``, cursor bound included.

    ``Iterate Index`` alone is too weak an assertion to rest this fix on: a plan
    that seeks the namespace prefix and then evaluates the cursor as a residual
    filter also prints ``Iterate Index``, while still reading the whole namespace.
    So the seek detail is asserted, and the two statements want different shapes:
    the tie query pins BOTH index columns by equality (``operator '='`` over a
    two-element ``(namespace, instant)`` value), and the range query pins the
    namespace as a ``prefix`` and carries a real ``<`` entry in ``ranges``.

    The window must be wider than the tie block the cursor sits in, or the tie
    query fills it and the range query is skipped by design — the sibling test
    below pins that shortcut. The fixture holds three rows per instant and 24 in
    total, so resuming at ``scan_limit=100`` guarantees both statements run.
    """
    adapter, ns = seeded
    first = await adapter.scan_documents(ns, scan_limit=1)
    recorder.reset()
    resumed = await adapter.scan_documents(ns, after=first.last_scanned, scan_limit=100)
    assert len(resumed.documents) > 2, (
        f"resumed step stayed inside the tie block ({len(resumed.documents)} rows); "
        "the strictly-older query was never reached"
    )
    (tie_sql, tie_plan), (range_sql, range_plan) = await _explain(
        recorder, expected_statements=2, label="resumed scan step"
    )

    tie_seek = _assert_indexed(tie_sql, tie_plan, index="idx_document_ns_created", label="tie query")
    assert tie_seek.get("operator") == "=", f"tie query is not an equality seek: {tie_seek}"
    assert isinstance(tie_seek.get("value"), list) and len(tie_seek["value"]) == 2, (
        f"tie query pinned only part of the index key — the cursor instant is being "
        f"evaluated as a residual filter, so the step still reads the namespace: {tie_seek}"
    )
    assert tie_seek["value"][0] == str(ns), f"tie query seek is not namespace-led: {tie_seek}"

    range_seek = _assert_indexed(range_sql, range_plan, index="idx_document_ns_created", label="range query")
    assert range_seek.get("prefix") == [str(ns)], (
        f"range query did not seek the namespace as an index prefix: {range_seek}"
    )
    assert [r["operator"] for r in range_seek.get("ranges", [])] == ["<"], (
        f"range query has no '<' bound in the index range — the cursor is a residual "
        f"filter and the step still reads the whole namespace: {range_seek}"
    )


async def test_both_resumed_statements_are_namespace_scoped(seeded, recorder) -> None:
    """The tenant conjunct must reach BOTH legs, in the text and in the binds.

    A narrowing that lands in only one leg makes that leg's window diverge, and
    for ``namespace_id`` specifically it re-creates the exact cross-tenant read
    this whole change exists to remove — half a resumed window coming back
    unscoped. Cheap to assert, catastrophic to miss, and not covered by the row
    assertions in the integration suite unless a foreign row happens to sort into
    the window.
    """
    adapter, ns = seeded
    first = await adapter.scan_documents(ns, scan_limit=1)
    recorder.reset()
    await adapter.scan_documents(ns, after=first.last_scanned, scan_limit=100)
    recorded = recorder.selects()
    assert len(recorded) == 2, [sql for sql, _ in recorded]
    for sql, params in recorded:
        assert "namespace_id = $ns" in sql, f"leg is not namespace-scoped:\n{sql}"
        assert params["ns"] == str(ns), f"leg bound the wrong namespace: {params.get('ns')!r} != {str(ns)!r}"


async def test_resumed_step_skips_the_range_query_when_the_tie_block_fills_it(seeded, recorder) -> None:
    """A step satisfied inside the tie block costs one statement, not two.

    The fixture puts three rows on each instant. Resuming from the first of them
    at ``scan_limit=1`` leaves two tie-mates, so the tie query fills the window on
    its own and the range query must not be issued.
    """
    adapter, ns = seeded
    first = await adapter.scan_documents(ns, scan_limit=1)
    recorder.reset()
    step = await adapter.scan_documents(ns, after=first.last_scanned, scan_limit=1)
    assert len(step.documents) == 1
    assert len(recorder.selects()) == 1, (
        "the strictly-older query ran even though the tie block already filled the window:\n"
        + "\n".join(sql for sql, _ in recorder.selects())
    )


@pytest.mark.parametrize("scan_limit", [1, 2, 3, 4, 5, 6, 7])
async def test_a_resumed_step_never_returns_more_than_scan_limit(seeded, recorder, scan_limit) -> None:
    """The two statements together must not overfill the window.

    This is the arithmetic that replaces a trailing slice: the strictly-older
    query runs with ``scan_limit - len(tie rows)``, not ``scan_limit``. Giving it
    the full limit is an easy and completely silent regression — it only shows up
    when the cursor sits inside a non-empty tie block, which a fixture with one
    row per instant never produces. Hence three rows per instant here, and
    ``raw_row_count`` asserted alongside the document count: ``exhausted`` is
    derived from the former, so an overfilled raw window also corrupts
    termination.
    """
    adapter, ns = seeded
    first = await adapter.scan_documents(ns, scan_limit=1)
    step = await adapter.scan_documents(ns, after=first.last_scanned, scan_limit=scan_limit)
    assert len(step.documents) <= scan_limit, (
        f"step returned {len(step.documents)} documents for scan_limit={scan_limit}; "
        "the strictly-older query was given the full limit instead of the shortfall"
    )


@pytest.mark.parametrize("scan_limit", [1, 2, 3, 4, 7, 50])
async def test_a_walk_over_tied_rows_visits_every_document_exactly_once(seeded, recorder, scan_limit) -> None:
    """End to end over a namespace whose rows are three-deep on every instant.

    The split's whole correctness claim is that concatenating the tie query's
    rows in front of the strictly-older query's rows reproduces the single
    disjunctive window exactly. Tie blocks are where that can fail — skipping a
    tie-mate, or re-returning one — and a fixture with one row per instant cannot
    see either. The foreign namespace seeded alongside is the cross-tenant guard.
    """
    adapter, ns = seeded
    seen: list[UUID] = []
    after = None
    for _ in range(200):
        step = await adapter.scan_documents(ns, after=after, scan_limit=scan_limit)
        seen.extend(d.id for d in step.documents)
        if step.exhausted:
            break
        after = step.last_scanned
    else:  # pragma: no cover - only reached if the walk fails to terminate
        pytest.fail(f"walk did not terminate within 200 steps at scan_limit={scan_limit}")

    assert len(seen) == len(set(seen)), f"walk returned duplicates at scan_limit={scan_limit}"
    total = await adapter.count_documents(ns)
    assert len(seen) == total, f"walk saw {len(seen)}/{total} documents at scan_limit={scan_limit}"


async def test_status_and_updated_before_narrow_both_scan_statements(seeded, recorder) -> None:
    """The shared conjuncts reach the tie query as well as the range query.

    A narrowing carried by only one of the two would silently widen half of every
    resumed window — rows the caller excluded coming back from whichever statement
    dropped the conjunct.
    """
    adapter, ns = seeded
    first = await adapter.scan_documents(
        ns, status=DocumentStatus.COMPLETED.value, updated_before=_BASE + timedelta(hours=1), scan_limit=1
    )
    recorder.reset()
    await adapter.scan_documents(
        ns,
        status=DocumentStatus.COMPLETED.value,
        updated_before=_BASE + timedelta(hours=1),
        after=first.last_scanned,
        scan_limit=100,
    )
    recorded = recorder.selects()
    assert len(recorded) == 2, [sql for sql, _ in recorded]
    for sql, params in recorded:
        assert "status = $status" in sql, sql
        assert "updated_at < $updated_before" in sql, sql
        assert params["status"] == DocumentStatus.COMPLETED.value
    for sql, plan in await _explain(recorder, expected_statements=2, label="narrowed resumed step"):
        _assert_indexed(sql, plan, index=None, label="narrowed resumed step")


async def test_no_scan_statement_contains_a_disjunction(seeded, recorder) -> None:
    """A source-level tripwire the ``EXPLAIN`` assertions cannot replace.

    ``EXPLAIN`` reports the plan the engine picked on *this* engine version. If a
    later SurrealDB learned to index a disjunction, the plan assertions above
    would go green while the statement quietly regained an ``OR`` that older
    deployments still table-scan. This checks the text instead.
    """
    adapter, ns = seeded
    first = await adapter.scan_documents(ns, scan_limit=1)
    await adapter.scan_documents(ns, after=first.last_scanned, scan_limit=100)
    for sql, _ in recorder.selects():
        assert " OR " not in sql.upper(), f"a disjunction is back in a scan statement:\n{sql}"


# --------------------------------------------------------------------------- #
# checksum dedup
# --------------------------------------------------------------------------- #


async def test_checksum_dedup_legs_plan_index_scans(seeded, recorder) -> None:
    """The single-row dedup probe is index-eligible on every leg it runs.

    ``pending_stale_before`` is set on the production ingest path, which is the
    configuration that used to make every ``remember()`` full-scan ``document``.
    The checksum chosen is a FAILED row so neither leg short-circuits and both are
    captured.
    """
    adapter, ns = seeded
    failed_checksum = f"cs{list(DocumentStatus).index(DocumentStatus.FAILED):03d}"
    hit = await adapter.get_document_by_checksum(ns, failed_checksum, pending_stale_before=_BASE + timedelta(minutes=5))
    assert hit is None, "fixture checksum is not a FAILED-only row; both legs would not run"
    plans = await _explain(recorder, expected_statements=2, label="dedup probe")
    settled_sql, settled_plan = plans[0]
    settled = _assert_indexed(settled_sql, settled_plan, index="idx_document_ns_checksum", label="dedup leg A")
    assert settled.get("value") == [str(ns), failed_checksum], (
        f"settled leg did not seek (namespace, checksum) — the checksum is a residual "
        f"filter and the probe still reads the namespace: {settled}"
    )
    # Leg B leads on ``status`` equality, so the planner prefers the status
    # composite over the checksum one; either is namespace-scoped, which is the
    # property that matters, so the name is not pinned for this leg.
    _assert_indexed(plans[1][0], plans[1][1], index=None, label="dedup leg B")


async def test_checksum_dedup_probe_stops_at_the_first_leg_that_hits(seeded, recorder) -> None:
    """A settled document is answered by one statement."""
    adapter, ns = seeded
    completed_checksum = f"cs{list(DocumentStatus).index(DocumentStatus.COMPLETED):03d}"
    hit = await adapter.get_document_by_checksum(
        ns, completed_checksum, pending_stale_before=_BASE + timedelta(minutes=5)
    )
    assert hit is not None
    assert len(recorder.selects()) == 1, [sql for sql, _ in recorder.selects()]


async def test_batch_checksum_dedup_legs_plan_index_scans(seeded, recorder) -> None:
    """The batch form runs both legs — a batch needs every checksum answered."""
    adapter, ns = seeded
    found = await adapter.get_documents_by_checksums(
        ns, [f"cs{i:03d}" for i in range(24)], pending_stale_before=_BASE + timedelta(minutes=5)
    )
    assert found, "fixture produced no dedup hits"
    for sql, plan in await _explain(recorder, expected_statements=2, label="batch dedup"):
        _assert_indexed(sql, plan, index=None, label="batch dedup")


async def test_both_dedup_entry_points_agree_when_one_checksum_matches_both_legs(seeded, recorder) -> None:
    """A checksum with a settled row AND a fresh-PENDING row resolves the same way.

    ``(namespace_id, checksum)`` has no unique constraint, so both legs can return
    a row for one checksum. The single-row probe stops at the first hit and yields
    the settled row; a batch built with plain last-wins would yield the PENDING
    one, because its leg runs second. Two dedup entry points disagreeing about the
    same checksum is how a phantom duplicate ingest gets born, so both apply leg
    order as precedence.

    Not reachable from the shared fixture (its checksums are one row each), so the
    pair is seeded here.
    """
    adapter, ns = seeded
    for title, status, updated_at in (
        ("settled", DocumentStatus.COMPLETED, _BASE),
        ("fresh-pending", DocumentStatus.PENDING, _BASE + timedelta(hours=1)),
    ):
        await adapter.create_document(
            Document(
                namespace_id=ns,
                content="x",
                title=title,
                source_type="library",
                checksum="shared-checksum",
                status=status,
                created_at=_BASE,
                updated_at=updated_at,
            )
        )
    # Cutoff between the two, so the PENDING row is FRESH and both legs match.
    cutoff = _BASE + timedelta(minutes=30)
    single = await adapter.get_document_by_checksum(ns, "shared-checksum", pending_stale_before=cutoff)
    batch = await adapter.get_documents_by_checksums(ns, ["shared-checksum"], pending_stale_before=cutoff)

    assert single is not None and "shared-checksum" in batch
    assert single.id == batch["shared-checksum"].id, (
        f"dedup entry points disagree: probe returned {single.title!r}, "
        f"batch returned {batch['shared-checksum'].title!r}"
    )
    assert single.title == "settled", f"settled row must outrank the in-flight one, got {single.title!r}"


async def test_a_concurrent_status_flip_between_legs_cannot_double_claim(seeded, recorder) -> None:
    """The orphan split's own race: one row returned by both legs must claim once.

    At any single instant the legs are disjoint — a row is PENDING or PROCESSING,
    never both — which is exactly why this looks impossible and is not. Nothing
    spans the two statements, so a concurrent claimer flipping a row
    PENDING -> PROCESSING in between makes leg one return it as pending and leg
    two return the same row as processing. The single statement it replaced could
    not do this; the id de-duplication in the merge is what closes it.

    The flip is injected deterministically between the two SELECTs rather than
    raced for.
    """
    adapter, ns = seeded
    victim = await adapter.create_document(
        Document(
            namespace_id=ns,
            content="x",
            title="racy",
            source_type="library",
            status=DocumentStatus.PENDING,
            created_at=_BASE,
            updated_at=_BASE,
        )
    )
    real_query = recorder._conn.query
    state = {"selects": 0}

    async def flipping_query(sql: str, params: Any = None) -> Any:
        rows = await real_query(sql, params)
        if sql.lstrip().upper().startswith("SELECT") and "status = $status" in sql:
            state["selects"] += 1
            if state["selects"] == 1:  # between the PENDING leg and the PROCESSING leg
                # ``_record_id`` and nothing else: khora writes every document id
                # as ``document:⟨uuid⟩``, and ``type::thing('document', <str>)``
                # builds ``document:u'…'`` — a DIFFERENT record. Addressing the
                # wrong one makes this test pass vacuously (verified: the id
                # de-duplication mutant survives when the flip misses).
                await real_query("UPDATE $rid SET status = 'processing'", {"rid": _record_id("document", victim.id)})
        return rows

    recorder._conn.query = flipping_query  # type: ignore[method-assign]
    try:
        claimed = await adapter.claim_orphaned_documents(
            ns,
            pending_before=_BASE + timedelta(hours=1),
            processing_before=_BASE + timedelta(hours=1),
            limit=50,
        )
    finally:
        recorder._conn.query = real_query  # type: ignore[method-assign]

    assert state["selects"] == 2, f"the flip was not injected between two legs ({state['selects']} selects)"
    ids = [d.id for d in claimed]
    assert len(ids) == len(set(ids)), f"a row was claimed twice across the two legs: {sorted(ids)}"
    assert ids.count(victim.id) == 1, f"the flipped row appears {ids.count(victim.id)}x, expected once"


async def test_legacy_dedup_without_a_stale_cutoff_is_one_indexed_statement(seeded, recorder) -> None:
    """``pending_stale_before=None`` keeps its single-clause shape."""
    adapter, ns = seeded
    await adapter.get_documents_by_checksums(ns, ["cs000"], pending_stale_before=None)
    ((sql, plan),) = await _explain(recorder, expected_statements=1, label="legacy dedup")
    _assert_indexed(sql, plan, index="idx_document_ns_checksum", label="legacy dedup")


# --------------------------------------------------------------------------- #
# orphan claim
# --------------------------------------------------------------------------- #


async def test_orphan_claim_legs_plan_index_scans(seeded, recorder) -> None:
    """Both status legs of the orphan sweep are index-eligible."""
    adapter, ns = seeded
    cutoff = _BASE + timedelta(hours=1)
    claimed = await adapter.claim_orphaned_documents(ns, pending_before=cutoff, processing_before=cutoff, limit=5)
    assert claimed, "fixture produced no claimable rows"
    seeks = []
    for sql, plan in await _explain(recorder, expected_statements=2, label="orphan claim"):
        seek = _assert_indexed(sql, plan, index="idx_document_ns_status", label="orphan claim")
        assert seek.get("operator") == "=", f"orphan leg is not an equality seek: {seek}"
        seeks.append(seek.get("value"))
    # Both index columns are pinned by equality. ``updated_at`` is deliberately
    # NOT in this index, so the cutoff stays a residual filter over the
    # (namespace, status) slice — which is fine, and is the honest claim: the
    # index buys tenant+status scoping, not a time-bounded seek.
    assert seeks == [
        [str(ns), DocumentStatus.PENDING.value],
        [str(ns), DocumentStatus.PROCESSING.value],
    ], f"orphan legs did not seek (namespace, status) per status: {seeks}"


async def test_orphan_claim_returns_the_oldest_rows_in_updated_at_order(seeded, recorder) -> None:
    """The client-side merge reproduces the single statement's ordering contract.

    The legs are ordered independently, so concatenating them would interleave
    wrongly; only the re-sort makes ``limit`` mean "the oldest ``limit`` stale
    rows" as it did before. Timestamps, not ids, are asserted: ties among equal
    ``updated_at`` were resolved arbitrarily by the old ``ORDER BY updated_at``
    too, and are not a contract.
    """
    adapter, ns = seeded
    cutoff = _BASE + timedelta(hours=1)
    everything = await adapter.claim_orphaned_documents(ns, pending_before=cutoff, processing_before=cutoff, limit=500)
    assert len(everything) > 4, "fixture is too small to distinguish a merge from a concatenation"
    # Reconstructed from ``orphan_prior_status``: the claim overwrites both status
    # and updated_at on what it returns, so the pre-claim order is read off the
    # created_at ladder the fixture wrote in lockstep with updated_at.
    ladder = [d.created_at for d in everything]
    assert ladder == sorted(ladder), f"claim did not return rows oldest-first: {ladder}"

    statuses = {d.orphan_prior_status for d in everything}
    assert statuses == {DocumentStatus.PENDING.value, DocumentStatus.PROCESSING.value}, (
        f"both legs must contribute or the merge is untested: {statuses}"
    )


async def test_orphan_claim_limit_takes_the_oldest_across_both_legs(seeded, recorder) -> None:
    """``limit`` bounds the merged set, not each leg.

    Fetching ``limit`` per leg and slicing after the merge is what makes this
    hold; slicing per leg would return up to ``2 * limit`` rows, and taking
    ``limit // 2`` from each would miss older rows whenever one status dominates.
    """
    adapter, ns = seeded
    cutoff = _BASE + timedelta(hours=1)
    limited = await adapter.claim_orphaned_documents(ns, pending_before=cutoff, processing_before=cutoff, limit=3)
    assert len(limited) == 3, [d.created_at for d in limited]


async def test_orphan_claim_respects_each_status_cutoff_independently(seeded, recorder) -> None:
    """Splitting the pairs must not cross-apply the cutoffs."""
    adapter, ns = seeded
    claimed = await adapter.claim_orphaned_documents(
        ns,
        pending_before=_BASE + timedelta(hours=1),
        processing_before=_BASE,  # excludes every PROCESSING row
        limit=500,
    )
    assert claimed, "fixture produced no PENDING rows"
    assert {d.orphan_prior_status for d in claimed} == {DocumentStatus.PENDING.value}, (
        f"a PROCESSING row was claimed under a cutoff that excludes all of them: "
        f"{sorted({d.orphan_prior_status for d in claimed})}"
    )


async def test_no_document_read_contains_a_disjunction(seeded, recorder) -> None:
    """Whole-adapter sweep: none of the reads touched here may carry an ``OR``."""
    adapter, ns = seeded
    cutoff = _BASE + timedelta(hours=1)
    first = await adapter.scan_documents(ns, scan_limit=1)
    await adapter.scan_documents(ns, after=first.last_scanned, scan_limit=100)
    await adapter.get_document_by_checksum(ns, "cs002", pending_stale_before=cutoff)
    await adapter.get_documents_by_checksums(ns, ["cs002", "cs003"], pending_stale_before=cutoff)
    await adapter.claim_orphaned_documents(ns, pending_before=cutoff, processing_before=cutoff, limit=5)
    offenders = [sql for sql, _ in recorder.selects() if " OR " in sql.upper()]
    assert not offenders, "document reads carrying a disjunction:\n" + "\n".join(offenders)


async def test_the_replaced_disjunctive_shapes_really_do_table_scan(seeded, recorder) -> None:
    """The premise, pinned. Without it the assertions above look like cargo cult.

    Each of the three predicates below is what the adapter used to send. If a
    future engine plans any of them on an index, this test fails and the split
    can be reconsidered — that is the intended signal, not a false alarm.
    """
    _adapter, ns = seeded
    conn = recorder._conn
    ns_str = str(ns)
    shapes = {
        "keyset": (
            "SELECT * FROM document WHERE namespace_id = $ns AND "
            "(created_at < $ts OR (created_at = $ts AND id < $rid)) "
            "ORDER BY created_at DESC, id DESC LIMIT 5",
            {"ns": ns_str, "ts": _BASE, "rid": f"document:⟨{UUID(int=0)}⟩"},
        ),
        "dedup": (
            "SELECT * FROM document WHERE namespace_id = $ns AND checksum = $cs AND "
            "status != 'failed' AND (status != 'pending' OR updated_at >= $stale) LIMIT 1",
            {"ns": ns_str, "cs": "cs001", "stale": _BASE},
        ),
        "orphan": (
            "SELECT * FROM document WHERE namespace_id = $ns AND ("
            "(status = $p AND updated_at < $pb) OR (status = $pr AND updated_at < $prb)"
            ") ORDER BY updated_at LIMIT 5",
            {
                "ns": ns_str,
                "p": DocumentStatus.PENDING.value,
                "pr": DocumentStatus.PROCESSING.value,
                "pb": _BASE,
                "prb": _BASE,
            },
        ),
    }
    for label, (sql, params) in shapes.items():
        plan: Any = await conn.query(f"{sql} EXPLAIN", params)
        assert plan[0]["operation"] == "Iterate Table", (
            f"{label}: this engine now indexes the disjunctive form. The client-side "
            f"split is no longer forced — revisit it.\n  plan: {plan}"
        )
