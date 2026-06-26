"""Regression tests for SurrealDB HNSW index dimension threading (#1386).

The SurrealDB backend used to define its vector indexes from a static DDL
string that hardcoded ``HNSW DIMENSION 1536``. Any embedder whose vectors
aren't 1536-dim (e.g. ``text-embedding-3-large`` = 3072) then had *every*
chunk/entity/episode insert rejected by SurrealDB with "Incorrect vector
dimension". The DDL is now built from the configured embedding dimension and
the ``StorageSettings.hnsw_*`` params.

These tests apply khora's REAL index DDL to an in-memory SurrealDB
(``memory://``, no server / API key) and assert vectors at the configured
dimension are accepted. The 3072 case is RED on the old hardcoded string and
GREEN once the DDL is sized from config.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("surrealdb")

import surrealdb  # noqa: E402

from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.schema import (  # noqa: E402
    build_search_index_definitions,
    ensure_search_indexes,
)

pytestmark = pytest.mark.unit


async def _apply_hnsw_ddl(db: object, ddl: str) -> None:
    """Apply only the HNSW index statements of a search-index DDL block.

    The co-located BM25 full-text index needs the ``khora_fulltext`` analyzer,
    which is irrelevant to the vector-dimension behaviour under test.
    """
    await db.query("DEFINE TABLE chunk SCHEMALESS; DEFINE TABLE entity SCHEMALESS; DEFINE TABLE episode SCHEMALESS;")
    for stmt in (s for s in ddl.split(";") if "HNSW" in s):
        await db.query(stmt + ";")


async def _insert_embedding(db: object, dim: int) -> str | None:
    """Insert one chunk with a ``dim``-length embedding.

    Returns ``None`` on success, otherwise the first line of the raised error.
    """
    try:
        await db.query("CREATE chunk SET content = 'x', embedding = $e;", {"e": [0.01] * dim})
        return None
    except Exception as exc:  # noqa: BLE001 - surface whatever SurrealDB raises
        return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"


async def _fresh_db() -> object:
    db = surrealdb.AsyncSurreal("memory://default")
    await db.connect()
    await db.use("khora", "khora")
    return db


@pytest.mark.parametrize("dimension", [512, 1536, 3072])
async def test_index_accepts_configured_dimension(dimension: int) -> None:
    """An embedding at the configured dimension is accepted by the HNSW index."""
    db = await _fresh_db()
    try:
        await _apply_hnsw_ddl(db, build_search_index_definitions(embedding_dimension=dimension))
        err = await _insert_embedding(db, dimension)
        assert err is None, f"insert of {dimension}-dim vector rejected: {err}"
    finally:
        await db.close()


async def test_large_model_dimension_no_longer_rejected() -> None:
    """text-embedding-3-large (3072) ingests; the 1536-baked index rejected it (#1386)."""
    db = await _fresh_db()
    try:
        await _apply_hnsw_ddl(db, build_search_index_definitions(embedding_dimension=3072))
        # The exact vector size that used to fail on the hardcoded DIMENSION 1536.
        assert await _insert_embedding(db, 3072) is None
        # And a 1536 vector is now the mismatch, proving the index really sized to 3072.
        mismatch = await _insert_embedding(db, 1536)
        assert mismatch is not None and "dimension" in mismatch.lower()
    finally:
        await db.close()


async def test_default_dimension_unchanged_for_1536() -> None:
    """Default rendering keeps the historical 1536 / EFC 128 / M 24 shape."""
    ddl = build_search_index_definitions()
    assert sorted(set(re.findall(r"DIMENSION (\d+)", ddl))) == ["1536"]
    assert sorted(set(re.findall(r"EFC (\d+)", ddl))) == ["128"]
    assert sorted(set(re.findall(r"\bM (\d+)", ddl))) == ["24"]

    db = await _fresh_db()
    try:
        await _apply_hnsw_ddl(db, ddl)
        assert await _insert_embedding(db, 1536) is None
    finally:
        await db.close()


def test_builder_threads_hnsw_params() -> None:
    """HNSW DIMENSION / EFC / M are templated from the builder args."""
    ddl = build_search_index_definitions(embedding_dimension=768, hnsw_m=32, hnsw_ef_construction=200)
    assert sorted(set(re.findall(r"DIMENSION (\d+)", ddl))) == ["768"]
    assert sorted(set(re.findall(r"EFC (\d+)", ddl))) == ["200"]
    assert sorted(set(re.findall(r"\bM (\d+)", ddl))) == ["32"]
    # ef_search is a query-time param, not an index-define slot — never emitted.
    assert "ef_search" not in ddl.lower()


def test_connection_carries_index_params() -> None:
    """The configured dimension / HNSW params are stored on the connection."""
    conn = SurrealDBConnection(
        mode="memory",
        embedding_dimension=3072,
        hnsw_m=32,
        hnsw_ef_construction=200,
    )
    assert conn.embedding_dimension == 3072
    assert conn.hnsw_m == 32
    assert conn.hnsw_ef_construction == 200


async def test_ensure_search_indexes_uses_connection_dimension() -> None:
    """ensure_search_indexes() applies the DDL sized to the connection's dimension."""
    conn = SurrealDBConnection(mode="memory", embedding_dimension=3072)
    await conn.connect()
    try:
        await ensure_search_indexes(conn)
        # The chunk HNSW index now expects 3072; a 3072 insert succeeds.
        await conn.execute("CREATE chunk SET content = 'x', embedding = $e;", {"e": [0.01] * 3072})
    finally:
        await conn.disconnect()
