"""LanceDB-backed storage path for the Chronicle engine.

Chronicle's LanceDB option is implemented by composing the ``sqlite_lance``
unified storage backend (SQLite for relational/graph/FTS5, LanceDB for
vectors). All four storage roles required by chronicle —
``RelationalBackendProtocol``, ``GraphBackendProtocol``,
``VectorBackendProtocol``, and ``EventStoreProtocol`` — are already
implemented there, so this module is a thin assembly point rather than a
parallel storage tree.

Use it via :class:`ChronicleEngine`::

    engine = ChronicleEngine(
        config,
        storage_backend="lancedb",
        lancedb_path="./data/chronicle.db",
    )

Or, if you'd rather construct the coordinator directly:

    coord = await build_lancedb_coordinator(
        db_path="./data/chronicle.db",
        embedding_dimension=1536,
    )

Both paths run the Alembic SQLite migrations against the database before
returning, so the file is ready to accept chunks immediately.

Install: ``pip install 'khora[sqlite-lance]'`` (pulls in ``aiosqlite`` and
``lancedb``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from loguru import logger

if TYPE_CHECKING:
    from khora.storage.coordinator import StorageCoordinator

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False


async def build_lancedb_coordinator(
    *,
    db_path: str = "./chronicle.db",
    lance_path: str | None = None,
    embedding_dimension: int = 1536,
    use_halfvec: bool = False,
    lance_index: Literal["auto", "ivf_pq", "hnsw", "brute"] = "auto",
    run_migrations: bool = True,
) -> StorageCoordinator:
    """Build a connected ``StorageCoordinator`` over SQLite + LanceDB.

    Args:
        db_path: SQLite database file path.
        lance_path: LanceDB directory. Defaults to ``<db_path>.lance``.
        embedding_dimension: Vector dimension for the LanceDB schema.
        use_halfvec: Store embeddings as float16 (smaller index, minor recall hit).
        lance_index: ANN index strategy. ``"auto"`` defers index creation until the
            table is large enough that an ANN scan beats brute force.
        run_migrations: When True, run the bundled Alembic migrations against
            ``db_path`` so the SQLite schema exists before the coordinator
            opens it. Set False if the DB is already migrated out-of-band.

    Returns:
        A ``StorageCoordinator`` with all four roles wired and connected.
    """
    if not _HAS_EMBEDDED:
        raise ImportError(
            "Chronicle's LanceDB backend requires aiosqlite and lancedb. "
            "Install with: pip install 'khora[sqlite-lance]'"
        )

    from khora.config.schema import SQLiteLanceConfig
    from khora.storage.factory import StorageConfig, StorageFactory

    if run_migrations:
        from khora.db.session import run_migrations as _run_migrations

        url = f"sqlite+aiosqlite:///{db_path}"
        result = await _run_migrations(url)
        if not result.success:
            raise RuntimeError(f"sqlite_lance migration failed: {result.error}")

    storage_config = StorageConfig(
        backend="sqlite_lance",
        sqlite_lance_config=SQLiteLanceConfig(
            db_path=db_path,
            lance_path=lance_path,
            embedding_dimension=embedding_dimension,
            use_halfvec=use_halfvec,
            lance_index=lance_index,
        ),
        postgresql_url=None,
    )
    coordinator = StorageFactory(config=storage_config).create_coordinator()
    await coordinator.connect()
    logger.info(
        "Chronicle LanceDB coordinator ready: sqlite={} lance={} dim={}",
        db_path,
        lance_path or f"{db_path}.lance",
        embedding_dimension,
    )
    return coordinator


__all__ = ["build_lancedb_coordinator"]
