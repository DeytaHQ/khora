"""Repro for issue #718 — skeleton engine on SurrealDB embedded mode.

The bug: two independent :class:`SurrealDBConnection` instances pointed
at the same ``surrealkv://`` directory cannot coexist. surrealkv only
supports a single open handle to a given on-disk store; opening a
second one causes the storage layer to read uninitialised revision
metadata, raising::

    surrealdb.errors.InternalError:
      Versioned error: A deserialization error occured:
      Invalid revision `0` for type `Value`

Before the fix, ``SkeletonConstructionEngine.connect()`` constructed a
*new* SurrealDBConnection inside the temporal store (alongside the
coordinator's shared connection), and the second-connection write
tripped the surrealkv internal error on the very first
``Khora.remember()`` call.

The fix mirrors the VectorCypher engine: pull the coordinator's
shared connection out of ``storage.relational._conn`` and hand it to
``SurrealDBTemporalStore(connection=…)``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("surrealdb")

from khora.storage.temporal.surrealdb import SurrealDBTemporalStore  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _reset_surrealdb_schema_init_lock() -> None:
    """Reset the module-level schema-init lock per test.

    The lock is created at import time and bound to whichever event
    loop first awaits it. pytest-asyncio creates a fresh loop per test
    in mode=auto, so re-using the same lock across tests raises
    ``RuntimeError: Lock is bound to a different event loop``. Replacing
    it before each test gives each test its own loop-local lock.
    """
    import asyncio

    from khora.storage.backends.surrealdb import connection as _conn_mod

    _conn_mod._schema_init_lock = asyncio.Lock()


async def test_skeleton_engine_shares_coordinator_surrealdb_connection() -> None:
    """The skeleton engine MUST hand the coordinator's connection to its
    SurrealDBTemporalStore — opening a second connection against the
    same surrealkv:// directory is fatal (see issue #718)."""
    from khora.config import KhoraConfig
    from khora.config.schema import StorageSettings, SurrealDBConfig
    from khora.engines.skeleton.engine import SkeletonConstructionEngine

    with tempfile.TemporaryDirectory(prefix="khora_skeleton_surreal_") as tmp:
        db_path = str(Path(tmp) / "kv")
        config = KhoraConfig(
            storage=StorageSettings(
                backend="surrealdb",
                surrealdb=SurrealDBConfig(
                    mode="embedded",
                    path=db_path,
                    namespace="khora",
                    database="default",
                ),
            ),
        )
        engine = SkeletonConstructionEngine(config)
        await engine.connect()
        try:
            # The temporal store should be using the coordinator's
            # _conn, NOT its own.
            assert engine._temporal_store is not None
            assert engine._storage is not None
            coord_conn = getattr(engine._storage._relational, "_conn", None)
            assert coord_conn is not None, "coordinator must expose a shared SurrealDB connection"
            assert isinstance(engine._temporal_store, SurrealDBTemporalStore)
            assert engine._temporal_store._conn is coord_conn, (
                "skeleton's temporal store must reuse the coordinator's SurrealDB connection (issue #718)"
            )
        finally:
            await engine.disconnect()


async def test_skeleton_remember_does_not_raise_invalid_revision_on_surrealkv() -> None:
    """End-to-end repro: ``Khora.remember()`` on skeleton + embedded
    SurrealDB must succeed on a fresh ``surrealkv://`` directory.

    Before the fix, the first remember() call raised
    ``InternalError: Invalid revision 0 for type Value`` because the
    skeleton engine opened a second SurrealDBConnection alongside the
    coordinator's — see issue #718.
    """
    from unittest.mock import patch

    from khora.config import KhoraConfig
    from khora.config.schema import StorageSettings, SurrealDBConfig
    from khora.engines.skeleton.engine import SkeletonConstructionEngine

    with tempfile.TemporaryDirectory(prefix="khora_skeleton_remember_") as tmp:
        db_path = str(Path(tmp) / "kv")
        config = KhoraConfig(
            storage=StorageSettings(
                backend="surrealdb",
                surrealdb=SurrealDBConfig(
                    mode="embedded",
                    path=db_path,
                    namespace="khora",
                    database="default",
                ),
                embedding_dimension=8,
            ),
        )
        engine = SkeletonConstructionEngine(config)
        await engine.connect()
        try:
            # Stub the embedder so we don't need network access — the
            # fix is about connection topology, not embedding quality.
            async def _fake_embed_batch(texts: list[str]) -> list[list[float]]:
                return [[0.1] * 8 for _ in texts]

            async def _fake_embed(text: str) -> list[float]:
                return [0.1] * 8

            with (
                patch.object(engine._embedder, "embed_batch", _fake_embed_batch),
                patch.object(engine._embedder, "embed", _fake_embed),
            ):
                namespace = await engine.create_namespace()
                result = await engine.remember(
                    "PagerDuty triggered for the payments service.",
                    namespace_id=namespace.namespace_id,
                    entity_types=[],
                    relationship_types=[],
                )
                assert result.chunks_created > 0
                assert result.document_id is not None
        finally:
            await engine.disconnect()
