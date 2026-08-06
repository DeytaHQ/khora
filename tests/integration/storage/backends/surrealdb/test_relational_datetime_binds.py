"""Datetime binds on ``SurrealDBRelationalAdapter``: checksum dedup + orphan claim.

Runs against an in-memory SurrealDB (``mode="memory"``) — no docker required,
same fixture shape as :mod:`tests.integration.storage.backends.surrealdb.test_relational_scan_documents`.
Skipped when the ``surrealdb`` extra is not installed.

The timestamp columns on this store are ``TYPE datetime`` and SurrealDB compares
across types **value-independently**: ``datetime < string`` is false and
``datetime >= string`` is true for *any* string, the stored value never being
reached. A ``.isoformat()`` bind therefore does not misorder a predicate, it
**pins** it — and neither direction raises. The two sites covered here were
pinned in opposite directions, which is why one test file carries both:

* ``_checksum_reingestable_clause`` returned **too much**. Its stale-PENDING
  exclusion hangs off ``updated_at >= $pending_stale_before``, so the disjunct
  was unconditionally true and the exclusion never fired: a PENDING row left
  behind by a crashed ingest was a permanent dedup hit and its document could
  never re-ingest (#1464 silently inoperative).
* ``claim_orphaned_documents`` returned **nothing**. It filters on
  ``updated_at < $pending_before`` / ``< $processing_before``, both pinned
  false, so no orphan was ever reclaimed. Its UPDATE also stamps ``updated_at``;
  a string there is *rejected* by the engine (``InternalError: ... expected a
  datetime``) rather than coerced, so that write is covered by asserting the
  persisted type rather than only the claim's return value.

Two halves of the dedup contract are asserted together on purpose. "Stale
PENDING re-ingests" alone would also pass for a mutant that made *everything*
re-ingestable; "fresh PENDING still dedups" is the concurrent in-flight guard
that pins the other side. FAILED (always re-ingestable) and COMPLETED (always a
dedup hit) ride along as status controls, so a failure localises to the cutoff
rather than to the clause as a whole.

Seeding goes through ``create_document``, the production write API. Timestamps
are pinned to a whole second and anchored on ``datetime.now(UTC)`` rather than a
fixed calendar instant: ``claim_orphaned_documents`` stamps claimed rows with
its own ``datetime.now(UTC)``, and the re-claim assertion below needs that stamp
to sit above the cutoffs the test passes in.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Document, MemoryNamespace, TenancyMode  # noqa: E402
from khora.core.models.document import DocumentStatus  # noqa: E402
from khora.storage.backends.surrealdb._helpers import _record_id  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402

pytestmark = pytest.mark.integration

STALE = timedelta(days=10)
FRESH = timedelta(minutes=1)
PENDING_CUTOFF = timedelta(hours=1)
PROCESSING_CUTOFF = timedelta(minutes=30)


@pytest.fixture
async def adapter():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="doc_dt_binds")
    await conn.connect()
    adapter = SurrealDBRelationalAdapter(conn)
    try:
        yield adapter
    finally:
        await conn.disconnect()


@pytest.fixture
async def namespace(adapter):
    nid = uuid4()
    return await adapter.create_namespace(MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED))


@pytest.fixture
def anchor() -> datetime:
    """The instant every seeded stamp and every cutoff is measured back from.

    ``now()`` and not a fixed calendar date: the claim path stamps reclaimed
    rows with its own ``datetime.now(UTC)``, so a fixed anchor in the future
    would leave a just-claimed row still below the cutoff and the re-claim
    assertion would pass or fail on the wall clock. Truncated to a whole second
    to match the seeding convention in the sibling scan tests.
    """
    return datetime.now(UTC).replace(microsecond=0)


async def _write(
    adapter: Any,
    namespace_id: UUID,
    *,
    checksum: str,
    status: DocumentStatus,
    updated_at: datetime,
) -> UUID:
    """Insert one document through the production write API."""
    doc_id = uuid4()
    await adapter.create_document(
        Document(
            id=doc_id,
            namespace_id=namespace_id,
            content=f"content for {checksum}",
            checksum=checksum,
            status=status,
            created_at=updated_at,
            updated_at=updated_at,
        )
    )
    return doc_id


async def _raw_row(adapter: Any, doc_id: UUID) -> dict[str, Any]:
    """Read a document row unconverted.

    ``_row_to_document`` runs ``_parse_dt``, which turns an ISO string back into
    a ``datetime`` — so a converted read cannot tell a stored datetime from a
    stored string. The type assertion has to see the row as the engine returns it.

    ``_record_id`` and not a formatted string: a record id binds as a
    ``RecordID`` object, and a string operand simply matches no row (the same
    class of silent cross-type miss this file exists to pin).
    """
    row = await adapter._conn.query_one("SELECT * FROM $rid", {"rid": _record_id("document", doc_id)})
    assert row is not None, f"document {doc_id} not found"
    return row


# --------------------------------------------------------------------------- #
# Checksum dedup — the clause that returned too much
# --------------------------------------------------------------------------- #

CS_STALE_PENDING = "cs-stale-pending"
CS_FRESH_PENDING = "cs-fresh-pending"
CS_FAILED = "cs-failed"
CS_COMPLETED = "cs-completed"


async def _seed_checksum_corpus(adapter: Any, namespace_id: UUID, anchor: datetime) -> dict[str, UUID]:
    """Four rows spanning the clause's whole decision surface.

    Both PENDING rows sit on the ``updated_at`` axis either side of the cutoff;
    the FAILED and COMPLETED rows are both *stale*, so a clause that keyed off
    age alone rather than off ``(status, age)`` would misclassify them.
    """
    return {
        CS_STALE_PENDING: await _write(
            adapter,
            namespace_id,
            checksum=CS_STALE_PENDING,
            status=DocumentStatus.PENDING,
            updated_at=anchor - STALE,
        ),
        CS_FRESH_PENDING: await _write(
            adapter,
            namespace_id,
            checksum=CS_FRESH_PENDING,
            status=DocumentStatus.PENDING,
            updated_at=anchor - FRESH,
        ),
        CS_FAILED: await _write(
            adapter,
            namespace_id,
            checksum=CS_FAILED,
            status=DocumentStatus.FAILED,
            updated_at=anchor - STALE,
        ),
        CS_COMPLETED: await _write(
            adapter,
            namespace_id,
            checksum=CS_COMPLETED,
            status=DocumentStatus.COMPLETED,
            updated_at=anchor - STALE,
        ),
    }


async def test_stale_pending_reingests_while_fresh_pending_still_dedups(adapter, namespace, anchor) -> None:
    """Single-checksum form: the cutoff separates the two PENDING rows.

    The stale row must miss (its half-ingest is abandoned; #1464 lets the
    content back in) and the fresh row must hit (its ingest may still be in
    flight — that hit is the concurrent-duplicate guard).
    """
    await _seed_checksum_corpus(adapter, namespace.id, anchor)
    cutoff = anchor - PENDING_CUTOFF

    stale = await adapter.get_document_by_checksum(namespace.id, CS_STALE_PENDING, pending_stale_before=cutoff)
    fresh = await adapter.get_document_by_checksum(namespace.id, CS_FRESH_PENDING, pending_stale_before=cutoff)

    assert stale is None, "a PENDING row older than the cutoff must be re-ingestable"
    assert fresh is not None, "a PENDING row newer than the cutoff is still in flight and must dedup"
    assert fresh.checksum == CS_FRESH_PENDING


async def test_checksum_status_controls_are_independent_of_the_cutoff(adapter, namespace, anchor) -> None:
    """FAILED always re-ingests, COMPLETED always dedups — at any age.

    Both rows here are as stale as the excluded PENDING one, so this pins that
    the cutoff narrows PENDING only and does not leak into the status test.
    """
    await _seed_checksum_corpus(adapter, namespace.id, anchor)
    cutoff = anchor - PENDING_CUTOFF

    for stale_before in (None, cutoff):
        failed = await adapter.get_document_by_checksum(namespace.id, CS_FAILED, pending_stale_before=stale_before)
        completed = await adapter.get_document_by_checksum(
            namespace.id, CS_COMPLETED, pending_stale_before=stale_before
        )
        assert failed is None, f"FAILED must always be re-ingestable (pending_stale_before={stale_before!r})"
        assert completed is not None, f"COMPLETED must always dedup (pending_stale_before={stale_before!r})"


async def test_omitting_the_cutoff_keeps_the_stale_pending_row_a_dedup_hit(adapter, namespace, anchor) -> None:
    """``pending_stale_before=None`` is the documented legacy behaviour.

    This is the discriminating half of the pair: it shows the exclusion above is
    produced by the *cutoff*, not by something else about the stale row. Under
    the string bind both calls returned the row and this test passed while its
    sibling failed.
    """
    await _seed_checksum_corpus(adapter, namespace.id, anchor)

    stale = await adapter.get_document_by_checksum(namespace.id, CS_STALE_PENDING, pending_stale_before=None)

    assert stale is not None
    assert stale.status is DocumentStatus.PENDING


async def test_batch_checksum_lookup_drops_exactly_the_stale_pending_row(adapter, namespace, anchor) -> None:
    """Batch form: the same clause, reached through ``IN $checksums``.

    Asserted as the difference between the two calls rather than as a count, so
    a failure names *which* checksum moved. Under the string bind the cutoff
    call returned the same three keys as the ``None`` call.
    """
    await _seed_checksum_corpus(adapter, namespace.id, anchor)
    all_checksums = [CS_STALE_PENDING, CS_FRESH_PENDING, CS_FAILED, CS_COMPLETED]

    legacy = await adapter.get_documents_by_checksums(namespace.id, all_checksums, pending_stale_before=None)
    reclaiming = await adapter.get_documents_by_checksums(
        namespace.id, all_checksums, pending_stale_before=anchor - PENDING_CUTOFF
    )

    assert set(legacy) == {CS_STALE_PENDING, CS_FRESH_PENDING, CS_COMPLETED}
    assert set(reclaiming) == {CS_FRESH_PENDING, CS_COMPLETED}
    assert set(legacy) - set(reclaiming) == {CS_STALE_PENDING}
    assert reclaiming[CS_FRESH_PENDING].checksum == CS_FRESH_PENDING


# --------------------------------------------------------------------------- #
# Orphan claim — the query that returned nothing
# --------------------------------------------------------------------------- #

D_STALE_PENDING = "stale-pending"
D_STALE_PROCESSING = "stale-processing"
D_FRESH_PENDING = "fresh-pending"
D_FRESH_PROCESSING = "fresh-processing"
D_STALE_COMPLETED = "stale-completed"


async def _seed_orphan_corpus(adapter: Any, namespace_id: UUID, anchor: datetime) -> dict[str, UUID]:
    """Five rows: two genuinely orphaned, three that must be left alone.

    The two cutoffs differ (1h for PENDING, 30m for PROCESSING) so a swap of the
    two binds would not go unnoticed, and the stale COMPLETED row pins that age
    alone does not make a row claimable.
    """
    return {
        D_STALE_PENDING: await _write(
            adapter,
            namespace_id,
            checksum="orphan-stale-pending",
            status=DocumentStatus.PENDING,
            updated_at=anchor - STALE,
        ),
        D_STALE_PROCESSING: await _write(
            adapter,
            namespace_id,
            checksum="orphan-stale-processing",
            status=DocumentStatus.PROCESSING,
            updated_at=anchor - STALE,
        ),
        D_FRESH_PENDING: await _write(
            adapter,
            namespace_id,
            checksum="orphan-fresh-pending",
            status=DocumentStatus.PENDING,
            updated_at=anchor - FRESH,
        ),
        D_FRESH_PROCESSING: await _write(
            adapter,
            namespace_id,
            checksum="orphan-fresh-processing",
            status=DocumentStatus.PROCESSING,
            updated_at=anchor - FRESH,
        ),
        D_STALE_COMPLETED: await _write(
            adapter,
            namespace_id,
            checksum="orphan-stale-completed",
            status=DocumentStatus.COMPLETED,
            updated_at=anchor - STALE,
        ),
    }


async def _claim(adapter: Any, namespace_id: UUID, anchor: datetime) -> list[Document]:
    return await adapter.claim_orphaned_documents(
        namespace_id,
        pending_before=anchor - PENDING_CUTOFF,
        processing_before=anchor - PROCESSING_CUTOFF,
    )


async def test_claim_takes_the_stale_rows_and_leaves_the_fresh_ones(adapter, namespace, anchor) -> None:
    """Both cutoffs select, and only the two genuinely stale rows come back.

    Under the string bind both ``<`` compares were pinned false and this
    returned zero rows on every call, whatever the corpus.
    """
    ids = await _seed_orphan_corpus(adapter, namespace.id, anchor)

    claimed = await _claim(adapter, namespace.id, anchor)

    assert {d.id for d in claimed} == {ids[D_STALE_PENDING], ids[D_STALE_PROCESSING]}
    assert all(d.status is DocumentStatus.PROCESSING for d in claimed)

    # The prior status is the transient the recovery loop labels its metric
    # with; PROCESSING is overwritten in place, so it is unrecoverable after
    # the claim if the adapter does not capture it here.
    prior = {d.id: d.orphan_prior_status for d in claimed}
    assert prior[ids[D_STALE_PENDING]] == DocumentStatus.PENDING.value
    assert prior[ids[D_STALE_PROCESSING]] == DocumentStatus.PROCESSING.value


async def test_claim_leaves_untouched_rows_byte_for_byte(adapter, namespace, anchor) -> None:
    """The three non-claimed rows keep both their status and their stamp.

    Separate from the selection assertion above: a claim that returned the right
    two documents could still have written over the others.
    """
    ids = await _seed_orphan_corpus(adapter, namespace.id, anchor)
    untouched = {
        D_FRESH_PENDING: (DocumentStatus.PENDING, anchor - FRESH),
        D_FRESH_PROCESSING: (DocumentStatus.PROCESSING, anchor - FRESH),
        D_STALE_COMPLETED: (DocumentStatus.COMPLETED, anchor - STALE),
    }

    claimed = await _claim(adapter, namespace.id, anchor)
    assert len(claimed) == 2, "the two stale rows were not claimed — the read binds are not under test"

    for key, (status, stamp) in untouched.items():
        doc = await adapter.get_document(ids[key], namespace_id=namespace.id)
        assert doc is not None
        assert doc.status is status, f"{key} changed status"
        assert doc.updated_at == stamp, f"{key} was re-stamped"


async def test_claim_persists_a_real_datetime_not_a_string(adapter, namespace, anchor) -> None:
    """The claim's write is the second bind, and it fails differently.

    A string into a ``TYPE datetime`` field is rejected by the engine rather
    than coerced, and the rejection discards the whole multi-field ``UPDATE`` —
    so this is the assertion that catches the write bind. Read raw, because the
    conversion layer would parse a stored string back into a ``datetime`` and
    hide exactly what is being asserted.
    """
    ids = await _seed_orphan_corpus(adapter, namespace.id, anchor)

    claimed = await _claim(adapter, namespace.id, anchor)
    assert claimed, "nothing was claimed — the write bind is not under test"

    for key in (D_STALE_PENDING, D_STALE_PROCESSING):
        row = await _raw_row(adapter, ids[key])
        stored = row["updated_at"]
        assert not isinstance(stored, str), f"{key}: updated_at persisted as a string: {stored!r}"
        assert isinstance(stored, datetime), f"{key}: updated_at is {type(stored).__name__}, expected datetime"
        # The stamp is fresh, not the seeded staleness — proof the SET landed
        # rather than the UPDATE being discarded with the row left as it was.
        assert stored > anchor - PENDING_CUTOFF
        assert row["status"] == DocumentStatus.PROCESSING.value


async def test_second_claim_immediately_after_the_first_returns_nothing(adapter, namespace, anchor) -> None:
    """The refreshed stamp is what stops two workers claiming the same orphan.

    It only works if the write persisted a comparable ``datetime``: a rejected
    write would leave the stale stamp in place and the row would be claimed
    again on every pass.
    """
    await _seed_orphan_corpus(adapter, namespace.id, anchor)

    first = await _claim(adapter, namespace.id, anchor)
    second = await _claim(adapter, namespace.id, anchor)

    assert len(first) == 2
    assert second == []
