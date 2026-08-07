"""Embedded-SurrealDB wiring for the documents-target conformance leg.

The store is :class:`~khora.storage.backends.surrealdb.relational.SurrealDBRelationalAdapter`
over an in-process ``memory://`` connection, with the production schema
(``DEFINE ... IF NOT EXISTS``, applied by ``connect()``) — no docker and no trimmed
test-only table. Unlike the chunk surreal leg, which hand-writes a cut-down
``temporal_chunk`` because the production chunk table needs record links the corpus
never uses, ``document`` has no such obstacle: ``namespace_id`` is a plain string
field, so the production DDL is exactly what the enumeration reads.

**Surreal is not total-exact on this surface.** On the chunk surface it compiles with
``on_unsupported="raise"``, so a leaf it cannot express is a fail-loud gap. Document
enumeration compiles with ``on_unsupported="split"`` and always has the coordinator's
post-filter behind it, so nothing raises and every case is an ordinary oracle-equal
row-set comparison — there is no ``expect_unsupported`` leg here. Two buckets are still
pruned by :func:`~khora.filter.conformance._documents_surreal_excluded`, and both are
storage-representation quirks a post-filter cannot undo (a metadata datetime stored as
a string, and an explicit JSON ``null`` dropped from a FLEXIBLE object on write).

Exposes the two callables the documents test modules inject:

* ``reachable() -> bool`` — the local-dev skip gate. Embedded ``memory://`` is
  in-process, so always ``True``.
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
from khora.storage.coordinator import StorageCoordinator
from tests.integration.matrix._conformance_docs_common import documents_executor, run_async


def reachable() -> bool:
    """Whether the embedded SurrealDB store is importable + reachable.

    ``memory://`` is in-process, so the only way this leg is unavailable is the
    ``surrealdb`` extra not being installed.
    """
    try:
        import surrealdb  # noqa: F401
    except ImportError:
        return False
    return True


async def _build_coordinator() -> StorageCoordinator:
    """A connected relational-only coordinator over an embedded ``memory://`` SurrealDB.

    The adapter's ``connect()`` applies the declarative schema, so the ``document``
    table and its indexes exist before the first seed. No vector/graph adapter is
    wired: document enumeration reads the relational adapter alone, and leaving the
    other roles unset also keeps the coordinator off its unified-backend path.
    """
    from khora.storage.backends.surrealdb.connection import SurrealDBConnection
    from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

    connection = SurrealDBConnection(mode="memory", namespace="khora_test", database="docs_conformance")
    await connection.connect()
    coord = StorageCoordinator(relational=SurrealDBRelationalAdapter(connection))
    await coord.connect()
    return coord


@lru_cache(maxsize=1)
def _coordinator() -> StorageCoordinator:
    """The process-wide connected coordinator (built exactly once)."""
    return run_async(_build_coordinator())


# ``case.id -> {seed_id: document UUID}``, filled on first use. See the sibling
# raw-sqlite module for why this is a plain dict rather than an ``lru_cache``.
_SEED_MAPS: dict[str, dict[str, UUID]] = {}


def _seed(case: ConformanceCase) -> dict[str, UUID]:
    """Seed one case's documents (once per process) and return ``seed_id -> document UUID``."""
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
