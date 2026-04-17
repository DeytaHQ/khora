"""Shared fixture helpers for sqlite_lance integration tests (DYT-2734).

These are plain helpers — NOT pytest fixtures — used by the sqlite_lance
integration modules to spin up a fully-migrated embedded coordinator
in ``tmp_path``.  Kept out of ``conftest.py`` to avoid affecting
existing Postgres/Neo4j integration tests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import aiosqlite

from khora.config.schema import SQLiteLanceConfig
from khora.db.session import run_migrations
from khora.storage.coordinator import StorageCoordinator
from khora.storage.factory import StorageConfig, StorageFactory

EMBED_DIM = 32


# The SQLiteLanceVectorAdapter (DYT-2730) expects two schema details that
# the Alembic migrations (DYT-2727) do not yet provide: a ``metadata_``
# column (migrations use ``metadata``) and an ``embedding TEXT`` column on
# ``chunks``/``entities`` for JSON-encoded vectors.  Reconcile after
# migrations so the adapter's SQL matches the on-disk schema.  Remove this
# shim once the adapter is refactored to track the migration schema
# (tracked under the DYT-2724 parent).
_ADAPTER_SCHEMA_SHIM = """
ALTER TABLE chunks ADD COLUMN metadata_ TEXT DEFAULT '{}';
ALTER TABLE chunks ADD COLUMN embedding TEXT;
ALTER TABLE entities ADD COLUMN metadata_ TEXT DEFAULT '{}';
ALTER TABLE entities ADD COLUMN embedding TEXT;
ALTER TABLE entities ADD COLUMN source_tool TEXT DEFAULT '';
DROP TRIGGER IF EXISTS chunks_au;
DROP TRIGGER IF EXISTS chunks_ad;
DROP TRIGGER IF EXISTS chunks_ai;
DROP TABLE IF EXISTS chunks_fts;
CREATE VIRTUAL TABLE chunks_fts USING fts5(content, chunk_id UNINDEXED, namespace_id UNINDEXED);
"""


async def _apply_adapter_schema_shim(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as conn:
        for stmt in _ADAPTER_SCHEMA_SHIM.strip().split(";"):
            s = stmt.strip()
            if s:
                await conn.execute(s)
        await conn.commit()


async def build_sqlite_lance_coordinator(
    tmp_path: Path,
    *,
    embed_dim: int = EMBED_DIM,
) -> StorageCoordinator:
    """Construct a fully-migrated sqlite_lance coordinator in tmp_path.

    Two runtime quirks the adapters have to work around here:

    1. **Adapter/migration schema drift** — the vector adapter (DYT-2730)
       was written against a schema that uses ``metadata_`` and an
       ``embedding TEXT`` column; the Alembic migrations (DYT-2727)
       produce ``metadata`` and no ``embedding`` column on ``chunks`` /
       ``entities``.  The shim above reconciles them.
    2. **UUID storage format mismatch** — the ORM's ``UUID(as_uuid=True)``
       types serialize to hex-without-dashes on SQLite (32 chars); the
       raw-aiosqlite adapters write dashed UUIDs (36 chars).  This means
       the ``chunks.document_id → documents.id`` FK (written by two
       different stores) can never match in practice.  Disable FKs on
       the shared handle so the adapters can coexist.  Matches the
       ``PRAGMA foreign_keys = OFF`` pattern used by the unit-level
       graph adapter fixture.
    """
    db_path = str(tmp_path / "khora.db")
    lance_path = str(tmp_path / "khora.lance")

    result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    if not result.success:
        raise RuntimeError(f"migration failed: {result.error}")

    await _apply_adapter_schema_shim(db_path)

    storage_config = StorageConfig(
        backend="sqlite_lance",
        sqlite_lance_config=SQLiteLanceConfig(
            db_path=db_path,
            lance_path=lance_path,
            embedding_dimension=embed_dim,
        ),
    )
    coord = StorageFactory(config=storage_config).create_coordinator()
    await coord.connect()
    # Disable FKs on the raw aiosqlite connection — see docstring.
    handle = coord.vector._handle  # type: ignore[union-attr]
    await handle.sqlite.execute("PRAGMA foreign_keys = OFF")
    await handle.sqlite.commit()
    return coord


def fake_embedding(text: str, *, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic, L2-normalized pseudo-embedding derived from SHA-256(text).

    Good enough for ordering assertions: similar text ⇒ identical vector,
    different text ⇒ different hash-derived vector.  Bypasses LiteLLM,
    keeping the integration suite hermetic.
    """
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand the 32-byte digest into ``dim`` floats in [-1, 1].
    raw = [(seed[i % len(seed)] - 128) / 128.0 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5 or 1.0
    return [x / norm for x in raw]
