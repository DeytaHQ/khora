"""Seed + paging helpers for the ``list_documents`` tie-break ordering tests.

``list_documents`` sorts on ``(created_at DESC, id DESC)``. Proving the
``id DESC`` leg requires a seed where ``created_at`` alone cannot decide the
order, and an expected sequence that is fixed by construction rather than by
chance - hence :func:`id_ladder` and :func:`seed_order`.

Shared by the four backend test modules (postgresql, raw sqlite, sqlite_lance,
surrealdb) so the "why this seed is not vacuous" reasoning lives in one place.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from uuid import UUID, uuid4


def id_ladder(n: int) -> list[UUID]:
    """Return ``n`` document ids in strictly ASCENDING order.

    Each id is a shared random 96-bit prefix plus an 8-hex-digit counter, so
    ``ids[i] < ids[j]`` exactly when ``i < j``: the expected ``id DESC`` result
    is ``list(reversed(ids))``, known up front rather than sampled.

    Why not plain ``uuid4`` ids: with random ids the expected order is whatever
    the sort happens to produce, so a backend that ignores the ``id`` tie-break
    can still match by luck (with two rows, half the time). Seeding a ladder
    removes luck from the assertion entirely - there is exactly one correct
    answer and an implementation that drops the ``id`` key cannot produce it.

    The random prefix keeps the ids unique across runs, so the tests are safe
    against a persistent database that is not truncated between runs.

    Seed rows through :func:`seed_order` rather than in ladder order - see that
    function for why ladder order is not safe to seed in.
    """
    prefix = uuid4().hex[:24]
    return [UUID(f"{prefix}{i:08x}") for i in range(n)]


def seed_order(ids: Sequence[UUID]) -> list[UUID]:
    """Reorder ``ids`` into the fixed, non-monotonic order rows are written in.

    Interleaves the ladder from both ends (``ids[0], ids[-1], ids[1], ...``), so
    the write order is neither ascending nor descending by id.

    This matters. When every ``created_at`` ties, a backend that ignores the id
    key returns rows in whatever order the scan produced - and the observed
    fallbacks go BOTH ways: raw SQLite sorts stably and yields insertion order,
    while the SQLAlchemy/SQLite path walks the ``(namespace_id, created_at)``
    index backwards and yields REVERSE insertion order. Seeding in ladder order
    therefore makes the second backend return exactly the expected descending
    sequence for the wrong reason - checked, not assumed: an earlier draft of
    these tests seeded in ladder order and passed against the pre-tie-break
    implementation. A non-monotonic write order cannot coincide with descending
    id order in either direction.
    """
    out: list[UUID] = []
    lo, hi = 0, len(ids) - 1
    while lo <= hi:
        out.append(ids[lo])
        if lo != hi:
            out.append(ids[hi])
        lo += 1
        hi -= 1
    return out


async def walk_pages(
    list_documents: Callable[..., Awaitable[Sequence]],
    namespace_id: UUID,
    *,
    page_size: int,
    max_pages: int = 100,
) -> list[list]:
    """Page through ``list_documents`` with ``limit``/``offset`` until drained.

    Returns the pages in request order; the caller asserts on their
    concatenation (order + exhaustiveness) and on their union (no overlap).

    ``max_pages`` is a safety valve: a backend that ignored ``offset`` would
    otherwise loop forever, and a guardrail test should fail rather than hang.
    """
    pages: list[list] = []
    offset = 0
    while len(pages) < max_pages:
        page = list(await list_documents(namespace_id, limit=page_size, offset=offset))
        if not page:
            return pages
        pages.append(page)
        offset += page_size
    raise AssertionError(f"list_documents did not drain within {max_pages} pages of {page_size}")


__all__ = ["id_ladder", "seed_order", "walk_pages"]
