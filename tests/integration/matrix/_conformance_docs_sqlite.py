"""Raw-``backend: sqlite`` wiring for the documents-target conformance leg.

The store is :class:`~khora.storage.backends.sqlite.SQLiteRelationalBackend` — the
hand-written aiosqlite tier that owns its ``documents`` DDL as a string literal at
``connect()`` time and has no Alembic chain behind it. **This is the first conformance
coverage of that backend at all**: the chunk corpus reaches SQLite only through
``sqlite_lance``, whose ``documents`` table is the shared SQLAlchemy model. The two
diverge in ways the corpus can see, which is why both legs exist rather than one
standing in for the other:

* the physical metadata column is ``metadata_`` here and ``metadata`` there (the
  compile context remaps accordingly);
* timestamps are written by ``dt.isoformat()`` into ``TEXT`` — a ``'T'`` separator with
  the writer's offset preserved — where the ORM tier writes SQLAlchemy's
  space-separated, offset-discarded ``DATETIME``;
* ``source_type`` is nullable here with a ``''`` default, and ``NOT NULL`` there
  (migration 055).

Both stores withhold ``created_at`` / ``source_timestamp`` from pushdown for those
serialization reasons, so the date-key cases land on the post-filter on both legs —
which is exactly the property the residual mode makes load-bearing.

Exposes the two callables the documents test modules inject:

* ``reachable() -> bool`` — the local-dev skip gate. In-process, so always ``True``.
* ``executor_for(case, *, forced_residual) -> DocumentsExecutor`` — seeds the case
  (once per process, cached) and returns a ready executor bound to its namespace.

Kept out of ``conftest.py`` and named with a leading underscore (not a ``test_``
module) so it is a plain helper the test modules import, never collected as tests.
"""

from __future__ import annotations

from functools import lru_cache
from uuid import UUID

from khora.filter.conformance import (
    ConformanceCase,
    DocumentsExecutor,
    _documents_case_namespace_id,
    seed_documents_case,
)
from khora.storage.backends.sqlite import SQLiteRelationalBackend
from khora.storage.coordinator import StorageCoordinator
from tests.integration.matrix._conformance_docs_common import documents_executor, run_async


def reachable() -> bool:
    """Whether the raw-sqlite store is reachable — always ``True`` (in-process)."""
    return True


async def _build_coordinator() -> StorageCoordinator:
    """A connected relational-only coordinator over an in-memory SQLite database.

    ``:memory:`` rather than a tmp file: this store keeps ONE aiosqlite connection for
    its whole lifetime, so the database survives as long as the coordinator does, and
    nothing else needs to read the file. No vector backend — document enumeration
    reads the relational store alone.
    """
    coord = StorageCoordinator(relational=SQLiteRelationalBackend(":memory:"))
    await coord.connect()
    return coord


@lru_cache(maxsize=1)
def _coordinator() -> StorageCoordinator:
    """The process-wide connected coordinator (built exactly once)."""
    return run_async(_build_coordinator())


# ``case.id -> {seed_id: document UUID}``, filled on first use. A plain dict, not an
# ``lru_cache``: :class:`ConformanceCase` is frozen but carries wire-dict filters, so
# it is unhashable and cannot be a cache key.
_SEED_MAPS: dict[str, dict[str, UUID]] = {}


def _seed(case: ConformanceCase) -> dict[str, UUID]:
    """Seed one case's documents (once per process) and return ``seed_id -> document UUID``.

    Seeding is LAZY and per-case rather than a whole-corpus pass, because the embedded
    stores have no cross-process seed-map artifact to build: a leg pays only for the
    cases it actually runs, which keeps the unit smoke subset cheap while the full
    matrix leg still gets every case. Both modes of a case share one seed — the walk
    is read-only.
    """
    if case.id not in _SEED_MAPS:
        _SEED_MAPS[case.id] = run_async(seed_documents_case(_coordinator(), case))
    return _SEED_MAPS[case.id]


def executor_for(case: ConformanceCase, *, forced_residual: bool) -> DocumentsExecutor:
    """Seed ``case`` (in-process, cached) and return a ready :class:`DocumentsExecutor`."""
    id_map = _seed(case)
    return documents_executor(
        _coordinator(),
        _documents_case_namespace_id(case),
        id_map,
        forced_residual=forced_residual,
    )
