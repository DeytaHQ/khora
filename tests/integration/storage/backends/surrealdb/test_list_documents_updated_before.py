"""``SurrealDBRelationalAdapter.list_documents`` — the ``updated_before`` bound
and the namespace scope.

Two conjuncts of one ``WHERE`` clause, neither of which had a single test
anywhere before this module. Both fail as *rows* rather than as errors, which is
why each assertion below is an exact row set and never a count or a
non-emptiness check.

**``updated_before``.** This store's ``updated_at`` is ``TYPE datetime``, and
SurrealDB compares across types without reaching the values: ``datetime <
string`` is false for every string, so an operand bound via ``.isoformat()``
does not narrow the window, it empties it. The caller sees a namespace with no
documents and no error. That is what shipped — measured on a six-document
corpus, ``0`` rows against the ``3`` a ``datetime`` bind returns. A test
asserting ``len(result) > 0`` would still pass against a bind that narrowed by
the wrong operator or the wrong column, so the assertion here is the precise
set of documents, keyed by a seeded id ladder.

**The namespace scope.** Nothing else in this backend's suite notices if
``list_documents`` drops it: neutralising the ``namespace_id`` predicate left
all 78 SurrealDB integration tests green. Both tests below therefore seed a
SECOND namespace with a corpus that would be returned wholesale if the scope
were dropped — same ``created_at``, same ``updated_at``, so every foreign row
ties on the ordering key and satisfies the bound. See
:func:`_two_namespace_ladders` for why the foreign ids are pinned ABOVE the
scanned ones rather than left to chance.

Runs against an in-memory SurrealDB (``mode="memory"``) — no docker required,
same fixture shape as
:mod:`tests.integration.storage.backends.surrealdb.test_list_documents_order`.
Skipped when the ``surrealdb`` extra is not installed. Seeding goes through
``create_document``, the production write API, so every row is serialized by the
path production writes take, and timestamps are pinned to whole seconds; see
:mod:`tests.test_helpers.document_scan`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Document, MemoryNamespace, TenancyMode  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402
from tests.test_helpers.document_order import seed_order  # noqa: E402
from tests.test_helpers.document_scan import WHOLE_SECOND, as_utc  # noqa: E402

pytestmark = pytest.mark.integration

#: The bound every test passes as ``updated_before``. Offset well clear of the
#: shared ``created_at`` so a bound accidentally applied to the wrong column
#: cannot produce the expected row set.
CUTOFF = WHOLE_SECOND + timedelta(hours=12)

#: ``updated_at`` per ladder position, as an offset from :data:`CUTOFF`. Hours
#: apart so no rounding or timezone slip can move a row across the bound, and
#: one row sits EXACTLY on it — the operator is strict, so that row must be
#: excluded, and a ``<=`` would return it.
_UPDATED_AT_OFFSETS = (
    timedelta(hours=-3),
    timedelta(hours=-2),
    timedelta(hours=-1),
    timedelta(0),
    timedelta(hours=1),
    timedelta(hours=2),
)

CORPUS_SIZE = len(_UPDATED_AT_OFFSETS)


@pytest.fixture
async def adapter():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="doc_updated_before")
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


def _two_namespace_ladders(n: int = CORPUS_SIZE) -> tuple[list[UUID], list[UUID]]:
    """Return ``(scanned_ids, foreign_ids)`` where EVERY foreign id sorts ABOVE
    every scanned id.

    Both ladders share one random 23-hex head and differ at the very next
    nibble — ``0`` for the scanned rows, ``1`` for the foreign ones — followed by
    the same 8-hex counter, so the two are 32 hex characters wide like any UUID
    and the first character that can differ is the discriminator. Two
    independent :func:`~tests.test_helpers.document_order.id_ladder` calls would
    decide the relative order by coin flip; the random head still keeps the ids
    unique across runs, which a fixed all-zeros prefix would not.

    The direction is the point. Every row in this module carries the same
    ``created_at``, so ``ORDER BY created_at DESC, id DESC`` degenerates to
    ``id DESC`` across BOTH namespaces at once. With the foreign ids above, a
    scope-less ``list_documents`` sorts all six of them ahead of every scanned
    row, so a ``limit``-bounded read comes back entirely foreign — the leak
    displaces the expected rows instead of merely appending after them. Below,
    the same leak would land past the end of the first page and a bounded read
    would return the right answer for the wrong reason.
    """
    prefix = uuid4().hex[:23]
    return _ladder(prefix, "0", n), _ladder(prefix, "1", n)


def _stamps(ids: list[UUID]) -> dict[UUID, datetime]:
    """Map each ladder position to its seeded ``updated_at``."""
    return {doc_id: CUTOFF + offset for doc_id, offset in zip(ids, _UPDATED_AT_OFFSETS, strict=True)}


async def _seed(adapter: Any, namespace_id: UUID, ids: list[UUID]) -> dict[UUID, datetime]:
    """Write one corpus into ``namespace_id`` through the production write API.

    Every row shares ``created_at``, so the ``id DESC`` leg alone decides the
    order and the expected sequence is ``reversed(ids)`` by construction. Rows
    go in via ``seed_order`` — neither ascending nor descending by id — so
    insertion order cannot coincide with the expected sequence in either
    direction.
    """
    stamps = _stamps(ids)
    for doc_id in seed_order(ids):
        await adapter.create_document(
            Document(
                id=doc_id,
                namespace_id=namespace_id,
                content="bounded content",
                checksum=f"updated-before-{doc_id.hex}",
                created_at=WHOLE_SECOND,
                updated_at=stamps[doc_id],
            )
        )
    return stamps


def _expected_below(ids: list[UUID], stamps: dict[UUID, datetime]) -> list[UUID]:
    """The rows a correct ``updated_before=CUTOFF`` returns, in result order."""
    return [doc_id for doc_id in reversed(ids) if stamps[doc_id] < CUTOFF]


# --------------------------------------------------------------------------- #
# The bound
# --------------------------------------------------------------------------- #


async def test_updated_before_returns_exactly_the_rows_below_the_bound(adapter, namespace) -> None:
    """``updated_before`` narrows to a known subset — not to nothing, not to everything.

    Three separate ways to get this wrong are pinned at once, because all three
    return rows rather than raising. An ISO-string operand matches no row at all
    (``datetime < string`` is false for every string on this store), so the
    result is empty and the caller reads it as an empty namespace. An inverted
    or ``>=`` comparison returns the complement. A ``<=`` returns the boundary
    row as well. Only an exact ordered row set separates the three from a
    working bound; a count, or a ``len(result) > 0``, separates none of them.

    The unbounded read is asserted first so the narrowing is measured against a
    corpus the store demonstrably holds — otherwise an empty result could mean
    the seed never landed.
    """
    ids = _ladder(uuid4().hex[:23], "0", CORPUS_SIZE)
    stamps = await _seed(adapter, namespace.id, ids)

    unbounded = await adapter.list_documents(namespace.id)
    assert [d.id for d in unbounded] == list(reversed(ids))

    expected = _expected_below(ids, stamps)
    # The bound must be doing work in both directions, or the assertion below is
    # vacuous against an implementation that returns everything or nothing.
    assert 0 < len(expected) < CORPUS_SIZE

    bounded = await adapter.list_documents(namespace.id, updated_before=CUTOFF)

    assert [d.id for d in bounded] == expected
    assert all(as_utc(d.updated_at) < CUTOFF for d in bounded)

    # The row sitting exactly on the bound is excluded: the comparison is strict.
    boundary_id = ids[_UPDATED_AT_OFFSETS.index(timedelta(0))]
    assert as_utc(next(d for d in unbounded if d.id == boundary_id).updated_at) == CUTOFF
    assert boundary_id not in {d.id for d in bounded}


# --------------------------------------------------------------------------- #
# The namespace scope
# --------------------------------------------------------------------------- #


async def test_list_documents_never_returns_another_namespaces_rows(adapter, namespace) -> None:
    """The ``namespace_id`` predicate, on rows that tie on the ordering key.

    The second namespace holds an identical corpus — same ``created_at``, same
    ``updated_at`` ladder — so a dropped scope predicate does not merely make
    foreign rows *reachable*, it makes every one of them a guaranteed hit. Both
    the unpaged read and a read bounded to exactly the corpus size are asserted:
    the bounded one is the case a leak would otherwise survive, and it bites
    here only because :func:`_two_namespace_ladders` pins the foreign ids above
    the scanned ones so they displace rather than append.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    await _seed(adapter, namespace.id, scanned_ids)
    other = await _make_namespace(adapter)
    await _seed(adapter, other.id, foreign_ids)

    unpaged = await adapter.list_documents(namespace.id)
    assert [d.id for d in unpaged] == list(reversed(scanned_ids))
    assert all(d.namespace_id == namespace.id for d in unpaged)

    page = await adapter.list_documents(namespace.id, limit=CORPUS_SIZE)
    assert [d.id for d in page] == list(reversed(scanned_ids))
    assert set(foreign_ids).isdisjoint({d.id for d in page})


async def test_updated_before_composes_with_the_namespace_scope(adapter, namespace) -> None:
    """Both conjuncts in one ``WHERE``, each seeded so the other cannot cover for it.

    Every foreign row below the bound satisfies ``updated_at < $updated_before``
    just as well as its scanned counterpart, so the bound does nothing to
    contain a dropped scope; and the scope does nothing to contain a bound that
    stopped narrowing. The expected set is the intersection, and it is a strict
    subset of both single-conjunct answers — asserted, so neither leg can be
    silently absent.
    """
    scanned_ids, foreign_ids = _two_namespace_ladders()
    stamps = await _seed(adapter, namespace.id, scanned_ids)
    other = await _make_namespace(adapter)
    await _seed(adapter, other.id, foreign_ids)

    expected = _expected_below(scanned_ids, stamps)
    assert 0 < len(expected) < CORPUS_SIZE

    bounded = await adapter.list_documents(namespace.id, updated_before=CUTOFF)

    assert [d.id for d in bounded] == expected
    assert all(d.namespace_id == namespace.id for d in bounded)
    assert set(foreign_ids).isdisjoint({d.id for d in bounded})

    # The other tenant's own bounded read is the mirror image — same corpus
    # shape, disjoint result — so the scope is selecting a namespace rather than
    # happening to exclude one.
    foreign_bounded = await adapter.list_documents(other.id, updated_before=CUTOFF)
    assert [d.id for d in foreign_bounded] == _expected_below(foreign_ids, _stamps(foreign_ids))
    assert set(scanned_ids).isdisjoint({d.id for d in foreign_bounded})
