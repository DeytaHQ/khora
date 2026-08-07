"""Seed, write and walk helpers for the bounded ``scan_documents`` tests.

Shared by all four relational stores' scan modules — the two SQLAlchemy-backed
ones (the embedded sqlite_lance adapter and PostgreSQL, which run the same keyset
scan over the same ``DocumentModel``), the raw-SQL SQLite store, and SurrealDB.
The four scans are genuinely different implementations; what they share is the
*corpus* they must all enumerate identically, and the seed reasoning behind it is
subtle enough to be worth stating once here rather than four times.

Store-specific reasoning deliberately stays in the store's own module: each one
inverts at least one of these traps (the cursor's id is dashed on one tier and
undashed on another; a whole second is the divergent microsecond polarity on one
and the agreeing one on the next), so a helper that tried to explain all four
would be wrong from three directions at once.

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
prefix, and ``str(uuid)`` puts its first dash *inside* that prefix, where ``-``
(0x2D) sorts below every hex digit. The two SQLite tiers store the id in opposite
forms — sqlite_lance holds 32 undashed hex characters, the raw-SQL store holds
the dashed 36 — so on each of them exactly one of the two spellings is the bug,
and on both the wrong spelling mis-resolves the whole tie block. Measured on
sqlite_lance: the tie-mate assertion in the cursor test drops from four rows to
one.

**What the shared prefix buys is the tie-mate half specifically, not
observability as such.** With random ids the comparison against a tie-mate is
decided before index 8 by essentially random bytes, so which rows a wrong
spelling loses (or gains) becomes a coin flip per id — the corpus then proves
whatever it happened to draw. Do not read that as "``uuid4`` hides the defect
entirely": on the raw-SQL store the cursor's own row comes back under any seed,
because ``str(u) < u.hex`` holds for every UUID. Each store module states its own
polarity; this helper only guarantees the prefix.

**The whole second.** :data:`WHOLE_SECOND` pins ``microsecond=0`` and every
stamp derives from it by whole seconds, which is *not* incidental — but which
polarity is the *revealing* one is per store, so read this as "pin it, do not
sample the clock", not as "zero microseconds is the dangerous case". On
sqlite_lance the ORM writes the six-digit microsecond field unconditionally into
a lexicographically-compared TEXT column, so at non-zero microseconds a cursor
formatted with ``str(naive_datetime)`` is byte-identical to the stored form and a
corpus seeded from ``datetime.now(UTC)`` silently agrees with a broken bind; at
exactly ``.000000`` the two diverge (``str()`` omits the field entirely) and the
cursor tests can fail. The raw-SQL store writes ``isoformat()``, which omits the
field at zero and emits it otherwise, so the polarity inverts and that module
seeds both.
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
    """Build a tie-heavy seed pinning both sort keys.

    ``ids`` supersedes ``total``: pass an ascending ladder to build the seed over
    caller-chosen ids instead of a fresh random one. That is what the
    two-namespace isolation tests need — both stores' tripwires seed a second
    namespace and want the two tenants' ids in a pinned relative order, which two
    independent :func:`~tests.test_helpers.document_order.id_ladder` calls decide
    by coin flip. Construction is otherwise identical either way, and the
    properties the module docstring calls load-bearing (the tie block, the two
    rows outside it with timestamp and id in deliberate conflict) hold for any
    ascending ladder — so a caller supplying ids owes only that they ascend and
    share a prefix, not a different seed shape.
    """
    ladder = list(ids) if ids is not None else id_ladder(total)
    if len(ladder) < 5:
        raise ValueError(f"need a tie block of at least 3 rows to resume from the middle of, got {len(ladder)} ids")

    newest_id, oldest_id = ladder[0], ladder[-1]
    tied = ladder[1:-1]

    stamps = dict.fromkeys(tied, instant)
    stamps[newest_id] = instant + timedelta(seconds=1)
    stamps[oldest_id] = instant - timedelta(seconds=1)

    return ScanSeed(
        writes=[(doc_id, stamps[doc_id]) for doc_id in seed_order(ladder)],
        expected=[newest_id, *reversed(tied), oldest_id],
        tied_ids=list(reversed(tied)),
        newest_id=newest_id,
        oldest_id=oldest_id,
        tie_instant=instant,
    )


def to_filter_ast(wire: dict[str, Any]) -> Any:
    """Parse a recall-filter wire dict to the canonical AST the scans take."""
    return parse_to_ast(RecallFilter.model_validate(wire))


async def write_document(store: Any, namespace_id: UUID, doc_id: UUID, created_at: datetime, **fields: Any) -> None:
    """Insert one document through ``create_document``, the production write API.

    Seeding through the production path rather than through raw SQL is what makes
    the cursor tests mean anything: every store serializes ``created_at`` and
    ``id`` on the way in, and a cursor is only correct if it is bound in that same
    serialization. A seed that wrote rows by hand would be comparing the scan
    against a corpus production could never produce.
    """
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
    """Write a seed's rows, in the seed's deliberately non-monotonic write order."""
    for doc_id, created_at in seed.writes:
        await write_document(store, namespace_id, doc_id, created_at)


async def seed_varied(store: Any, namespace_id: UUID, seed: ScanSeed) -> None:
    """Seed the same corpus with attribute variety, so a filter can split it.

    Attributes are assigned by *write* index, which is deliberately not the
    enumeration order — every expectation built on this corpus is therefore
    derived from the rows a scan actually returns, never from this loop's counter.
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


# Filter shapes for the "pushdown never rejects a row the full filter would keep"
# tripwire, restricted to the ones that mean the same thing on every store. Shapes
# are chosen for the ways a compiler can get the superset property wrong, not for
# operator coverage.
#
# **Every shape here must match at least one row of a :func:`seed_varied` corpus**,
# and the tests assert that (``oracle > 0``) rather than trusting it. The
# assertion under test is ``oracle <= window``, which a constant-empty oracle
# satisfies unconditionally — a vacuous shape is not a weak test, it is no test,
# and it looks identical in a green run. ``pushable_exists`` is the cautionary
# case: it read ``{"source_url": {"$exists": False}}`` until khora #1589, and
# ``source_url`` is a system key present on every row, so its oracle was 0 on all
# three stores.
#
# Deliberately NOT lifted: the shapes naming a store's date-key pushability
# (``unpushable_date`` / ``pushable_date`` / ``unpushable_key`` and the ``$or`` /
# ``$not`` wrappers over them). Those differ per store *by design* — PostgreSQL
# pushes ``created_at``, the two SQLite tiers withhold it, SurrealDB pushes it and
# withholds ``occurred_at`` — and a shared dict would have to pick one name and
# make it wrong somewhere. Each module keeps its own, merged onto this base.
SUPERSET_SHAPES: dict[str, dict[str, Any]] = {
    "pushable_eq": {"source_type": {"$eq": "report"}},
    "pushable_ne": {"source_type": {"$ne": "report"}},
    "pushable_nin": {"source_type": {"$nin": ["report"]}},
    "pushable_exists": {"source_url": {"$exists": True}},
    "metadata_eq": {"metadata.tier": {"$eq": "gold"}},
    "not_over_pushable": {"$not": {"source_type": {"$eq": "report"}}},
    "and_of_in_and_not": {
        "$and": [
            {"source_type": {"$in": ["report", "library"]}},
            {"$not": {"title": {"$eq": "doc-0"}}},
        ]
    },
}


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
    "SUPERSET_SHAPES",
    "WHOLE_SECOND",
    "ScanSeed",
    "as_utc",
    "scan_seed",
    "seed_documents",
    "seed_varied",
    "to_filter_ast",
    "walk_scan",
    "write_document",
]
