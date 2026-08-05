"""Seed + walk helpers for the bounded ``scan_documents`` tests.

Shared by the two SQLAlchemy-backed relational stores (the embedded sqlite_lance
adapter and PostgreSQL), which run the same keyset scan over the same
``DocumentModel``. The seed reasoning is subtle enough to be worth stating once
here rather than twice in the two test modules.

Two properties of the seed are load-bearing:

**The tie block.** ``total - 2`` rows share one ``created_at``, so only the
``id DESC`` leg can order them and a resume position taken from the middle of
that block is genuinely mid-tie. The other two rows sit outside the block with
timestamp and id in deliberate conflict — the newest row carries the LOWEST id
and the oldest row the HIGHEST — so a scan that sorted on ``id`` first would
park them at exactly the wrong ends and could not reproduce
:attr:`ScanSeed.expected`. This half is borrowed wholesale from the
``list_documents`` ordering seed; see :mod:`tests.test_helpers.document_order`.

**The whole second.** :data:`WHOLE_SECOND` pins ``microsecond=0`` and every
stamp derives from it by whole seconds, which is *not* incidental. The embedded
store holds ``created_at`` as TEXT and compares it lexicographically, and at
non-zero microseconds a cursor formatted with ``str(naive_datetime)`` happens to
be byte-identical to the stored form — so a corpus seeded from
``datetime.now(UTC)`` silently agrees with a broken cursor bind and proves
nothing. At exactly ``.000000`` the two forms diverge (SQLAlchemy writes the
six-digit microsecond field, ``str()`` omits it entirely), which is what makes
the cursor tests able to fail.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from tests.test_helpers.document_order import id_ladder, seed_order

# Whole second, zero microseconds — see the module docstring for why this is
# pinned rather than sampled from the clock.
WHOLE_SECOND = datetime(2026, 1, 31, 12, 30, 0, 0, tzinfo=UTC)


@dataclass(frozen=True)
class ScanSeed:
    """A seed plan: rows to write, and the one order a scan must enumerate them in."""

    writes: list[tuple[UUID, datetime]]
    """``(id, created_at)`` pairs, in the (non-monotonic) order rows are written."""

    expected: list[UUID]
    """The single correct full enumeration, ``(created_at DESC, id DESC)``."""

    tied_ids: list[UUID]
    """The ids sharing :attr:`tie_instant`, already in expected (descending) order."""

    newest_id: UUID
    """Newest ``created_at``, lowest id — must enumerate FIRST."""

    oldest_id: UUID
    """Oldest ``created_at``, highest id — must enumerate LAST."""

    tie_instant: datetime
    """The ``created_at`` every row in :attr:`tied_ids` was written with."""


def scan_seed(total: int = 6, *, instant: datetime = WHOLE_SECOND) -> ScanSeed:
    """Build a tie-heavy seed of ``total`` documents pinning both sort keys."""
    if total < 5:
        raise ValueError(f"total must leave a tie block of at least 3 rows to resume from the middle of, got {total}")

    ids = id_ladder(total)
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


def as_utc(value: datetime) -> datetime:
    """Attach UTC to a naive datetime, leave an aware one alone.

    The two stores disagree about the cursor's tzinfo: PostgreSQL reads an aware
    value off ``timestamptz``, while the embedded store's ``DATETIME`` discards
    the writer's offset at write time and reads back naive. Both are correct for
    their store — the key is store-local — so a shared assertion that wants to
    compare a read-back stamp against a seeded one normalizes here rather than
    pretending the two shapes are the same.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


async def walk_scan(
    scan_documents: Callable[..., Awaitable[Any]],
    namespace_id: UUID,
    *,
    scan_limit: int,
    max_steps: int = 50,
    **scan_kwargs: Any,
) -> list[Any]:
    """Walk a namespace one bounded step at a time, chaining ``last_scanned``.

    Returns every :class:`~khora.storage.backends.base.DocumentScanStep` in
    order; the caller asserts on their concatenation (order, exhaustiveness, no
    repeats) and on the terminal step.

    Two guardrails, because both failures this walk exists to catch are
    non-terminating rather than wrong-answer. A cursor that fails to advance past
    its own row — the exact symptom of a hand-formatted timestamp bind, whose
    ISO ``'T'`` separator sorts above every stored value — would otherwise spin
    forever, so the walk raises the moment ``last_scanned`` repeats. ``max_steps``
    is the backstop for any other form of non-termination. A guardrail test must
    fail, not hang.
    """
    steps: list[Any] = []
    after = None
    while len(steps) < max_steps:
        step = await scan_documents(namespace_id, after=after, scan_limit=scan_limit, **scan_kwargs)
        steps.append(step)
        if step.exhausted:
            return steps
        if step.last_scanned == after:
            raise AssertionError(
                f"scan cursor did not advance: step {len(steps) - 1} resumed from {after!r} and "
                f"reported the same position again, so the walk can never terminate"
            )
        after = step.last_scanned
    raise AssertionError(f"scan did not report exhausted within {max_steps} steps of scan_limit={scan_limit}")


__all__ = ["WHOLE_SECOND", "ScanSeed", "as_utc", "scan_seed", "walk_scan"]
