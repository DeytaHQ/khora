"""Integration tests for configurable embedding dimensions on Postgres (#1260).

The pgvector embedding column and its halfvec HNSW index used to be hardcoded
``vector(1536)``, and a config guard rejected any non-1536 dimension. Now the
dimension flows from ``llm.embedding_dimension`` into the migration, so a model
like ``text-embedding-3-large`` (3072) works end-to-end on Postgres.

Each test provisions its OWN throwaway database (``CREATE DATABASE``) so the
shared dev DB (pinned to 1536) is never touched, and drops it on teardown.

Run locally::

    make dev    # this repo's compose: postgres :5434, neo4j :7688
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \\
    NEO4J_INTEGRATION_TEST=1 KHORA_NEO4J_URL=bolt://localhost:7688 \\
    KHORA_NEO4J_PASSWORD=pleaseletmein \\
        uv run pytest tests/integration/db/test_configurable_embedding_dimension_pg.py \\
        -v -m integration --no-cov
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import asyncpg
import pytest

from khora.db.session import run_migrations

pytestmark = pytest.mark.integration

# This repo's compose puts Postgres on 5434; honor an explicit override.
_DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)


def _plain(url: str) -> str:
    return url.replace("+asyncpg", "")


def _pg_reachable() -> bool:
    parsed = urlparse(_plain(_DATABASE_URL))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


_PG_AVAILABLE = _pg_reachable()


def _with_dbname(url: str, dbname: str) -> str:
    """Return ``url`` pointing at a different database name (async driver form)."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path=f"/{dbname}"))


async def _asyncpg_connect(url: str) -> asyncpg.Connection:
    """Open a raw asyncpg connection to the maintenance database of ``url``."""
    parsed = urlparse(_plain(url))
    return await asyncpg.connect(
        host=parsed.hostname,
        port=parsed.port,
        user=parsed.username,
        password=parsed.password,
        database=(parsed.path or "/").lstrip("/") or "postgres",
    )


@pytest.fixture
async def isolated_db() -> AsyncIterator[Callable[[], Awaitable[str]]]:
    """Factory that creates throwaway databases and drops them on teardown.

    Full DATABASE-level isolation (not a schema wipe) so the shared, 1536-pinned
    dev database is never disturbed.
    """
    if not _PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable (run `make dev` first)")

    created: list[str] = []

    async def make() -> str:
        name = f"khora_dim_{uuid4().hex[:12]}"
        conn = await _asyncpg_connect(_DATABASE_URL)
        try:
            # CREATE DATABASE cannot run in a transaction; asyncpg autocommits.
            await conn.execute(f'CREATE DATABASE "{name}"')
        finally:
            await conn.close()
        created.append(name)
        return _with_dbname(_DATABASE_URL, name)

    yield make

    conn = await _asyncpg_connect(_DATABASE_URL)
    try:
        for name in created:
            await conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
                name,
            )
            await conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
    finally:
        await conn.close()


async def _column_type(url: str, table: str, column: str) -> str | None:
    """Return the fully-qualified column type (e.g. ``vector(3072)``) or None.

    Uses asyncpg directly — this project ships only the asyncpg driver, so a
    SQLAlchemy engine built from a bare ``postgresql://`` URL would resolve to
    the (absent) psycopg2 dialect.
    """
    conn = await _asyncpg_connect(url)
    try:
        return await conn.fetchval(
            "SELECT format_type(a.atttypid, a.atttypmod) "
            "FROM pg_attribute a JOIN pg_class c ON a.attrelid = c.oid "
            "JOIN pg_namespace n ON c.relnamespace = n.oid "
            "WHERE n.nspname = 'public' AND c.relname = $1 AND a.attname = $2",
            table,
            column,
        )
    finally:
        await conn.close()


async def _index_def(url: str, index_name: str) -> str | None:
    conn = await _asyncpg_connect(url)
    try:
        return await conn.fetchval("SELECT indexdef FROM pg_indexes WHERE indexname = $1", index_name)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Migration sizing (PG-only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_migrate_at_3072_sizes_columns_and_halfvec_index(
    isolated_db: Callable[[], Awaitable[str]],
) -> None:
    """A fresh DB migrated at 3072 sizes the vector columns + halfvec index to 3072.

    text-embedding-3-large at full width exceeds the 2000-dim ``vector`` HNSW
    limit, so the full-precision index is skipped and halfvec (limit 4000) is
    used instead.
    """
    url = await isolated_db()
    result = await run_migrations(url, embedding_dimension=3072, use_halfvec=True)
    assert result.success, f"migrations failed: {result.error}"

    assert await _column_type(url, "chunks", "embedding") == "vector(3072)"
    assert await _column_type(url, "entities", "embedding") == "vector(3072)"

    halfvec_def = await _index_def(url, "ix_chunks_embedding_halfvec_hnsw")
    assert halfvec_def is not None, "halfvec HNSW index must exist"
    assert "halfvec(3072)" in halfvec_def, halfvec_def

    # Full-precision vector HNSW is not buildable above 2000 dims — skipped.
    assert await _index_def(url, "ix_chunks_embedding_hnsw") is None


@pytest.mark.asyncio
async def test_fresh_migrate_default_is_1536(
    isolated_db: Callable[[], Awaitable[str]],
) -> None:
    """With no dimension injected, the schema stays at the historical 1536."""
    url = await isolated_db()
    result = await run_migrations(url)
    assert result.success, f"migrations failed: {result.error}"

    assert await _column_type(url, "chunks", "embedding") == "vector(1536)"
    assert await _column_type(url, "entities", "embedding") == "vector(1536)"

    halfvec_def = await _index_def(url, "ix_chunks_embedding_halfvec_hnsw")
    assert halfvec_def is not None and "halfvec(1536)" in halfvec_def, halfvec_def

    # At 1536 the full-precision vector HNSW index is created as before.
    assert await _index_def(url, "ix_chunks_embedding_hnsw") is not None


@pytest.mark.asyncio
async def test_sqlite_migration_chain_is_dimension_agnostic(tmp_path: Any) -> None:
    """Parity: the SQLite chain runs at a non-1536 dimension (LanceDB owns vectors).

    The Postgres-only vector columns are omitted on SQLite, so the injected
    dimension never touches the SQLite schema — the full chain still applies.
    """
    url = f"sqlite+aiosqlite:///{tmp_path / 'khora.db'}"
    result = await run_migrations(url, embedding_dimension=3072, use_halfvec=True)
    assert result.success, f"sqlite migrations failed: {result.error}"


# ---------------------------------------------------------------------------
# remember() end-to-end at 3072 (PG + Neo4j)
# ---------------------------------------------------------------------------

EMBED_DIM_3072 = 3072


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[Any]:
    from khora.extraction.extractors.base import ExtractedEntity, ExtractedRelationship, ExtractionResult

    # Two entities + a relationship so the entity-embedding write path (the ORM
    # EntityModel, which also enforces the pgvector dimension on bind) is
    # exercised, not just the chunk write.
    return [
        ExtractionResult(
            entities=[
                ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.99),
                ExtractedEntity(name="Bob", entity_type="PERSON", confidence=0.99),
            ],
            relationships=[
                ExtractedRelationship(
                    source_entity="Alice", target_entity="Bob", relationship_type="KNOWS", confidence=0.99
                ),
            ],
        )
        for _ in texts
    ]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    """Deterministic 3072-dim unit-vector embedder."""
    unit = [1.0] + [0.0] * (EMBED_DIM_3072 - 1)
    return [unit[:] for _ in texts]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run remember() end-to-end against real Neo4j (requires make dev)",
)
async def test_remember_stores_3072_vector_end_to_end(
    isolated_db: Callable[[], Awaitable[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1260 repro (b): a 3072-dim vector stores without a dimension error.

    Configures the documented large-model dimension on the LLM side (the path
    the embedder docs point users down), migrates a throwaway DB from it, then
    runs remember() and confirms the 3072-dim embedding lands in the column.
    """
    from khora.config import KhoraConfig
    from khora.khora import Khora

    monkeypatch.setattr(
        "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
        _stub_extract_multi,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )

    url = await isolated_db()
    neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7688")
    neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")

    config = KhoraConfig(database_url=url, neo4j_url=neo4j_url)
    config.storage.neo4j_user = neo4j_user
    config.storage.neo4j_password = neo4j_password
    # Dimension set the way the embedder docs describe — on the LLM side.
    config.llm.embedding_model = "text-embedding-3-large"
    config.llm.embedding_dimension = EMBED_DIM_3072
    config.pipeline.chunk_size = 1024
    config.pipeline.extract_entities = True
    config.pipeline.selective_extraction = False

    # Migrations at 3072 flow from get_effective_embedding_dimension().
    assert config.get_effective_embedding_dimension() == EMBED_DIM_3072
    assert await _column_type(url, "chunks", "embedding") is None  # not migrated yet

    kb = Khora(config, run_migrations=True)
    await kb.connect()
    try:
        # The migration-managed chunks table AND the vectorcypher engine's
        # runtime khora_chunks table are both sized to 3072.
        assert await _column_type(url, "chunks", "embedding") == "vector(3072)"
        assert await _column_type(url, "khora_chunks", "embedding") == "vector(3072)"

        ns = await kb.create_namespace()
        result = await kb.remember(
            content="Alice knows Bob very well.",
            namespace=ns.namespace_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        assert result is not None

        # The 3072-dim vectors are persisted (no CharacterNotInRepertoire /
        # dimension-mismatch error at store time). The default vectorcypher
        # engine writes chunks to khora_chunks and entities to the ORM
        # ``entities`` table — both bind through pgvector types that enforce the
        # dimension, so this exercises repro (b) on both write paths.
        conn = await _asyncpg_connect(url)
        try:
            chunk_dims = [
                r["d"]
                for r in await conn.fetch(
                    "SELECT vector_dims(embedding) AS d FROM khora_chunks WHERE embedding IS NOT NULL"
                )
            ]
            entity_dims = [
                r["d"]
                for r in await conn.fetch(
                    "SELECT vector_dims(embedding) AS d FROM entities WHERE embedding IS NOT NULL"
                )
            ]
        finally:
            await conn.close()
        assert chunk_dims, "no chunk embedding was stored"
        assert all(d == EMBED_DIM_3072 for d in chunk_dims), chunk_dims
        assert entity_dims, "no entity embedding was stored"
        assert all(d == EMBED_DIM_3072 for d in entity_dims), entity_dims
    finally:
        await kb.disconnect()
