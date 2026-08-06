"""Seed + walk helpers for the bounded ``scan_documents`` tests.

Shared by all four ``scan_documents`` test modules: the two SQLAlchemy-backed
relational stores (the embedded sqlite_lance adapter and PostgreSQL), which run
the same keyset scan over the same ``DocumentModel``, and the two raw-SQL ones
(``SQLiteRelationalBackend`` and the SurrealDB relational adapter). The seed
reasoning is subtle enough to be worth stating once here rather than four times
over.

**What deliberately stays per store, so nobody "finishes the job" by lifting
it.** Each module's ``_SUPERSET_SHAPES`` table encodes that store's *capability*,
not a shared corpus: SurrealDB pushes ``created_at`` (``pushable_date``) where
the two SQLite-family modules defer it (``unpushable_date``), and the
``occurred_at`` shape whose oracle is empty by construction
(``unpushable_key``) exists on raw SQLite and SurrealDB but not on sqlite_lance
— 11 shapes, 11 shapes and 10. A shared dict would need per-store overrides and
would imply a uniformity that does not exist. Same for the two
``_two_namespace_ladders`` helpers (their fixed id arrangement is load-bearing
on SurrealDB and determinism-only on raw SQLite, and each says so at length) and
for PostgreSQL's ``_seed_varied``, which seeds no ``metadata`` because that
module has no shape that reads one.

Two properties of the seed are load-bearing:

**The tie block.** ``total - 2`` rows share one ``created_at``, so only the
``id DESC`` leg can order them and a resume position taken from the middle of
that block is genuinely mid-tie. The other two rows sit outside the block with
timestamp and id in deliberate conflict — the newest row carries the LOWEST id
and the oldest row the HIGHEST — so a scan that sorted on ``id`` first would
park them at exactly the wrong ends and could not reproduce
:attr:`ScanSeed.expected`. This half is borrowed wholesale from the
``list_documents`` ordering seed; see :mod:`tests.test_helpers.document_order`.

The ladder does a second job here that it was not written for, and it is worth
knowing before anyone "simplifies" it to plain ``uuid4``: its ids share a 24-hex
prefix, and ``str(uuid)`` puts its first dash *inside* that prefix. Since ``-``
(0x2D) sorts below every hex digit, a cursor rendered as a dashed string sorts
below the whole tie block — measured, the tie-mate assertion in the cursor test
drops from four rows to one. With random ids the comparison is decided before
the first dash and that mistake is invisible (it needs two ids agreeing on all
eight leading hex characters). The shared prefix is what makes it catchable.

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

from khora.core.models import Document
from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
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


def scan_seed(total: int = 6, *, instant: datetime = WHOLE_SECOND, ids: list[UUID] | None = None) -> ScanSeed:
    """Build a tie-heavy seed of ``total`` documents pinning both sort keys.

    Pass ``ids`` to supply the ascending id ladder instead of drawing a fresh
    random one (``total`` is then ignored). The two-namespace tests need that:
    they seed both namespaces at the same tie instant from two ladders whose
    relative order is pinned by construction, which two independent
    :func:`~tests.test_helpers.document_order.id_ladder` draws would decide by
    coin flip. Nothing else about the seed changes.
    """
    ids = list(ids) if ids is not None else id_ladder(total)
    if len(ids) < 5:
        raise ValueError(
            f"the seed must leave a tie block of at least 3 rows to resume from the middle of, got {len(ids)}"
        )

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


async def write_document(store: Any, namespace_id: UUID, doc_id: UUID, created_at: datetime, **fields: Any) -> None:
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


async def seed_documents(store: Any, namespace_id: UUID, seed: ScanSeed) -> None:
    for doc_id, created_at in seed.writes:
        await write_document(store, namespace_id, doc_id, created_at)


async def seed_varied(store: Any, namespace_id: UUID, seed: ScanSeed) -> None:
    """Seed the same corpus with attribute variety, so a filter can split it.

    Attributes are assigned by *write* index, which is deliberately not the
    enumeration order — every expectation in the calling module is therefore
    derived from the rows a scan actually returns, never from this loop's
    counter.
    """
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await write_document(
            store,
            namespace_id,
            doc_id,
            created_at,
            title=f"doc-{i}",
            source_type="report" if i % 2 == 0 else "library",
            metadata={"tier": "gold"} if i < 2 else {},
        )


def wire_to_ast(wire: dict[str, Any]) -> Any:
    """Lower a wire-form filter to the AST ``scan_documents`` takes."""
    return parse_to_ast(RecallFilter.model_validate(wire))


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


__all__ = [
    "WHOLE_SECOND",
    "ScanSeed",
    "as_utc",
    "scan_seed",
    "seed_documents",
    "seed_varied",
    "walk_scan",
    "wire_to_ast",
    "write_document",
]
