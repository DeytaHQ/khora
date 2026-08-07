"""Embedded ``sqlite_lance`` wiring for the documents-target conformance leg.

The store is :class:`~khora.storage.backends.sqlite_lance.SQLiteLanceRelationalAdapter`
over a tmp SQLite file migrated to head — the shared ``DocumentModel`` schema, built by
the real Alembic chain, so ``documents`` here is the same table PostgreSQL has (down to
``source_type`` being ``NOT NULL`` and the metadata column being named ``metadata``).
What differs from its Postgres twin is entirely serialization: SQLAlchemy's SQLite
``DATETIME`` writes a space-separated, offset-DISCARDED string, which is why this
store's ``_documents_compile_context`` withholds ``created_at`` / ``source_timestamp``
from pushdown where Postgres pushes both against real ``timestamptz`` columns.

No LanceDB and no vector backend: document enumeration reads the relational store
alone, so the embedded handle's Lance side is never touched and no embeddings are
generated.

Exposes the two callables the documents test modules inject:

* ``reachable() -> bool`` — the local-dev skip gate. In-process, so always ``True``.
* ``executor_for(case, *, forced_residual) -> DocumentsExecutor`` — seeds the case
  (once per process, cached) and returns a ready executor bound to its namespace.

Kept out of ``conftest.py`` and named with a leading underscore (not a ``test_``
module) so it is a plain helper the test modules import, never collected as tests.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import UUID

from khora.db.session import run_migrations
from khora.filter.conformance import (
    ConformanceCase,
    DocumentsExecutor,
    _documents_case_namespace_id,
    seed_documents_case,
)
from khora.storage.backends.sqlite_lance import SQLiteLanceRelationalAdapter
from khora.storage.backends.sqlite_lance.connection import (
    EmbeddedStorageHandle,
    EmbeddedStorageHandleConfig,
)
from khora.storage.coordinator import StorageCoordinator
from tests.integration.matrix._conformance_docs_common import documents_executor, run_async


def reachable() -> bool:
    """Whether the sqlite_lance store is reachable — always ``True`` (embedded in-process)."""
    return True


async def _build_coordinator() -> StorageCoordinator:
    """Migrate a tmp SQLite file to head and wire a relational-only coordinator over it.

    The full Alembic chain runs (not a hand-written DDL string), so ``documents`` is
    the shared model at head — including migration 055's ``NOT NULL source_type`` and
    056's ``NOT NULL created_at``, both of which the seeder must satisfy.
    """
    tmp_path = Path(tempfile.mkdtemp(prefix="khora-conformance-docs-lance-"))
    db_path = str(tmp_path / "khora.db")
    result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    if not result.success:
        raise RuntimeError(f"migration failed: {result.error}")

    handle = EmbeddedStorageHandle(
        EmbeddedStorageHandleConfig(db_path=db_path, lance_path=str(tmp_path / "khora.lance")),
    )
    await handle.connect()
    coord = StorageCoordinator(relational=SQLiteLanceRelationalAdapter(handle))
    await coord.connect()
    return coord


@lru_cache(maxsize=1)
def _coordinator() -> StorageCoordinator:
    """The process-wide connected coordinator (migrated + built exactly once)."""
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
