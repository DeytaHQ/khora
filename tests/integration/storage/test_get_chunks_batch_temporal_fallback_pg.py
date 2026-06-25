"""Live-PG regression guard for #1372 — PPR chunk hydration on vectorcypher.

The vectorcypher / skeleton engines write ingested chunks to the
``khora_chunks`` temporal-store table, NOT the relational ``chunks`` table.
PPR retrieval (``enable_ppr_retrieval=True``) scores chunk ids then hydrates
them via ``storage.get_chunks_batch`` → ``PgVectorBackend.get_chunks_batch``,
which SELECTed only the base ``chunks`` table (``ChunkModel``). The ids lived
in ``khora_chunks``, so the batch returned ``{}`` and the graph channel
silently collapsed to vector-only — zero chunks hydrated.

The fix mirrors the already-shipped sqlite_lance fallback (#905,
``SQLiteLanceVectorAdapter.get_chunks_batch``): when an id is not satisfied by
``chunks``, look it up in ``khora_chunks`` and decode it. A chronicle-only
stack has no ``khora_chunks`` table, so the fallback swallows the missing-table
error and returns what ``chunks`` yielded instead of crashing. The pgvector
backend never received that port; this test is the live-PG guard for it.

This is the Postgres sibling of the embedded
``tests/unit/storage/test_get_chunks_batch_temporal_fallback.py``. It needs a
real PG+Neo4j stack because ``VectorCypherEngine.connect`` verifies Neo4j
connectivity on the PG backend (mirrors
``tests/integration/test_vectorcypher_recency_channel_pg.py``).

How to run locally::

    make dev   # postgres (5434) + neo4j (bolt 7688) via docker compose
    KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora \\
        NEO4J_INTEGRATION_TEST=1 KHORA_NEO4J_URL=bolt://localhost:7688 \\
        uv run pytest tests/integration/storage/test_get_chunks_batch_temporal_fallback_pg.py \\
        -v -m integration --no-cov
"""

from __future__ import annotations

import math
import os
import socket
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from khora.config import KhoraConfig
from khora.db.session import run_migrations
from khora.extraction.extractors.base import ExtractionResult
from khora.khora import Khora

EMBED_DIM = 1536  # matches the khora_chunks.embedding Vector(1536) column

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


def _pg_reachable() -> bool:
    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _neo4j_url() -> str:
    return os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7688")


def _neo4j_reachable() -> bool:
    parsed = urlparse(_neo4j_url())
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _pg_reachable(),
        reason="PostgreSQL not reachable (run `make dev` first)",
    ),
    pytest.mark.skipif(
        not _neo4j_reachable() or not os.environ.get("NEO4J_INTEGRATION_TEST"),
        reason="set NEO4J_INTEGRATION_TEST=1 and run `make dev` (vectorcypher needs Neo4j)",
    ),
]

_CONTENT = "PostgreSQL was chosen for the user database."


async def _remember(kb: Khora, namespace_id: UUID) -> Any:
    return await kb.remember(
        _CONTENT,
        namespace=namespace_id,
        entity_types=[],
        relationship_types=[],
    )


# ---------------------------------------------------------------------------
# Deterministic embedder + empty extractor stubs (no external LLM). Mirrors the
# stub shape in tests/integration/test_vectorcypher_recency_channel_pg.py.
# ---------------------------------------------------------------------------


def _embed_for(text_in: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    vec[EMBED_DIM - 1] = 0.01
    for i, ch in enumerate(text_in.lower()):
        vec[(ord(ch) + i) % (EMBED_DIM - 1)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def _stub_embed_batch(self: Any, texts: list[str]) -> list[list[float]]:
    return [_embed_for(t) for t in texts]


async def _stub_embed(self: Any, text_in: str) -> list[float]:
    return _embed_for(text_in)


async def _stub_extract_multi(self: Any, texts: list[str], **_kwargs: Any) -> list[ExtractionResult]:
    return [ExtractionResult() for _ in texts]


@pytest.fixture
def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed_batch",
        _stub_embed_batch,
    )
    monkeypatch.setattr(
        "khora.extraction.embedders.litellm.LiteLLMEmbedder.embed",
        _stub_embed,
    )
    monkeypatch.setattr(
        "khora.extraction.extractors.llm.LLMEntityExtractor.extract_multi",
        _stub_extract_multi,
    )


@pytest.fixture(scope="module")
async def _migrations_once() -> None:
    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.begin() as conn:
            r = await conn.execute(
                text("SELECT typname FROM pg_type WHERE typnamespace = 'public'::regnamespace AND typtype = 'e'")
            )
            for (typname,) in r.fetchall():
                await conn.execute(text(f"DROP TYPE IF EXISTS public.{typname} CASCADE"))
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    "CREATE TABLE khora_alembic_version ("
                    "  version_num VARCHAR(64) NOT NULL,"
                    "  CONSTRAINT khora_alembic_version_pkc PRIMARY KEY (version_num)"
                    ")"
                )
            )
    finally:
        await eng.dispose()

    result = await run_migrations(DATABASE_URL)
    assert result.success, f"Migrations failed: {result.error}"


@pytest.fixture
async def kb_vc(_migrations_once: None, _patch_llm: None) -> AsyncIterator[Khora]:
    """Connected VectorCypher Khora (live PG + Neo4j)."""
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=_neo4j_url())
    config.storage.neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    config.storage.neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    config.storage.postgresql_url = DATABASE_URL
    config.pipeline.extract_entities = False
    config.pipeline.selective_extraction = False

    instance = Khora(config, engine="vectorcypher", run_migrations=False)
    await instance.connect()
    try:
        yield instance
    finally:
        await instance.disconnect()


async def _khora_chunk_ids(namespace_id: UUID, document_id: UUID) -> list[UUID]:
    """Read the chunk ids the vectorcypher engine wrote to ``khora_chunks``.

    The pgvector ``get_chunks_by_document`` reads only the base ``chunks``
    table (the adjacent #1372 gap, out of scope here), so it cannot be used to
    learn the temporal-store ids on a vectorcypher+PG stack. Query the
    ``khora_chunks`` table directly instead.
    """
    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.connect() as conn:
            rows = await conn.execute(
                text("SELECT id FROM khora_chunks WHERE namespace_id = :ns AND document_id = :doc"),
                {"ns": str(namespace_id), "doc": str(document_id)},
            )
            return [r[0] for r in rows.fetchall()]
    finally:
        await eng.dispose()


async def _plain_chunks_count(namespace_id: UUID, chunk_ids: list[UUID]) -> int:
    """Count how many of ``chunk_ids`` live in the relational ``chunks`` table."""
    eng = create_async_engine(DATABASE_URL)
    try:
        async with eng.connect() as conn:
            rows = await conn.execute(
                text("SELECT count(*) FROM chunks WHERE namespace_id = :ns AND id = ANY(:ids)"),
                {"ns": str(namespace_id), "ids": [str(c) for c in chunk_ids]},
            )
            return int(rows.scalar_one())
    finally:
        await eng.dispose()


async def test_pg_get_chunks_batch_reads_temporal_table(kb_vc: Khora) -> None:
    """vectorcypher+PG writes khora_chunks; get_chunks_batch hydrates them.

    This is the #1372 regression guard: without the ``khora_chunks`` fallback in
    ``PgVectorBackend.get_chunks_batch`` the batch returns ``{}`` for these ids
    (the chunks live ONLY in ``khora_chunks``, never in the base ``chunks``
    table for the temporal engines), so PPR hydration silently collapses to
    vector-only.
    """
    ns = await kb_vc.create_namespace()
    namespace_id: UUID = ns.namespace_id
    result = await _remember(kb_vc, namespace_id)
    assert result.chunks_created >= 1

    resolved = await kb_vc.storage.resolve_namespace(namespace_id)

    chunk_ids = await _khora_chunk_ids(resolved, result.document_id)
    assert chunk_ids, "expected temporal chunks for the ingested doc in khora_chunks"

    # Anti-vacuity guard: these ids live ONLY in khora_chunks, never in the
    # relational ``chunks`` table for the vectorcypher engine, so a
    # ``chunks``-only query would return ``{}``. The fallback is the sole reason
    # the batch resolves them.
    assert await _plain_chunks_count(resolved, chunk_ids) == 0, (
        "expected the relational `chunks` table to hold none of these ids"
    )

    fetched = await kb_vc.storage.get_chunks_batch(chunk_ids, namespace_id=resolved)
    assert set(fetched.keys()) == set(chunk_ids), (
        "get_chunks_batch did not resolve khora_chunks ids via the #1372 fallback; "
        f"got {set(fetched.keys())} vs {set(chunk_ids)}"
    )
    assert all(c.content == _CONTENT for c in fetched.values())
