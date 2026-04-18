"""Shared fixture helpers for sqlite_lance integration tests (DYT-2734).

These are plain helpers — NOT pytest fixtures — used by the sqlite_lance
integration modules to spin up a fully-migrated embedded coordinator
in ``tmp_path``.  Kept out of ``conftest.py`` to avoid affecting
existing Postgres/Neo4j integration tests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from khora.config.schema import SQLiteLanceConfig
from khora.db.session import run_migrations
from khora.storage.coordinator import StorageCoordinator
from khora.storage.factory import StorageConfig, StorageFactory

EMBED_DIM = 32


async def build_sqlite_lance_coordinator(
    tmp_path: Path,
    *,
    embed_dim: int = EMBED_DIM,
) -> StorageCoordinator:
    """Construct a fully-migrated sqlite_lance coordinator in tmp_path.

    Relies on the real Alembic-migrated schema — after DYT-2749 the raw
    adapters align with the migrated column shape (``chunks.metadata``,
    no ``embedding`` column, external-content FTS5 driven by triggers,
    32-char hex UUIDs matching SQLAlchemy ``UUID(as_uuid=True)``), so no
    shim or ``PRAGMA foreign_keys = OFF`` is needed.
    """
    db_path = str(tmp_path / "khora.db")
    lance_path = str(tmp_path / "khora.lance")

    result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    if not result.success:
        raise RuntimeError(f"migration failed: {result.error}")

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
