"""End-to-end occurred_at persistence + recall-filter tests on the real PG/pgvector path.

The PostgreSQL ``chunks`` table carries a distinct ``occurred_at`` TIMESTAMPTZ column
(the real-world event time the chunk's content refers to, separate from both
``created_at`` ingestion time and ``source_timestamp``). These tests drive that column
through the production ``PgVectorBackend`` write/read path against a live, fully-migrated
Postgres — no storage mocks — and prove two halves of the contract:

* **Round-trip** — a chunk written with an ``occurred_at`` distinct from both
  ``created_at`` and ``source_timestamp`` reads back with that exact value. This fails
  loudly if the write path drops ``occurred_at`` (it would read back ``NULL``).
* **Recall-filter regression guard** — a chunk whose ``occurred_at`` is in range but
  whose ``source_timestamp`` is out of range is honored by an ``occurred_at`` recall
  filter. The effective event time is ``COALESCE(occurred_at, source_timestamp)``; if
  the write path dropped ``occurred_at``, that COALESCE would collapse to the
  out-of-range ``source_timestamp`` and the chunk would be (wrongly) filtered out. So
  this proves ``occurred_at`` is genuinely persisted, not silently recovered from
  ``source_timestamp``.

The PG sibling of ``tests/integration/test_chronicle_filter_embedded.py``: same engine,
same filter AST, same recall path — only the storage backend differs. Seeding goes
through the coordinator's own write API (``create_chunks_batch``) with deterministic
fake embeddings; all seed chunks share one embedding so the vector channel returns the
whole seed set and the filter is the only narrowing force.

Tz-aware datetimes throughout (the column is ``DateTime(timezone=True)``); the seeded
bounds and the filter literal are all UTC-anchored so comparisons never mix tz-naive
and tz-aware values.

How to run locally::

    make dev    # only postgres needed (compose.yaml uses port 5434)
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \
        uv run pytest tests/integration/test_chronicle_filter_pgvector.py -v -m integration --no-cov
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from khora.config import KhoraConfig
from khora.config.schema import QuerySettings
from khora.core.models import Chunk, Document, MemoryNamespace
from khora.db.session import run_migrations
from khora.engines.chronicle.engine import ChronicleEngine
from khora.filter import RecallFilter
from khora.filter.ast import parse_to_ast
from khora.query import SearchMode
from khora.storage.backends.pgvector import PgVectorBackend
from khora.storage.backends.postgresql import PostgreSQLBackend
from khora.storage.coordinator import StorageCoordinator

# This repo's compose puts Postgres on 5434 (compose.yaml); honor an explicit
# KHORA_DATABASE_URL override, else default to the compose port.
DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# Matches the chunks.embedding Vector(1536) column hard-coded in migration 000.
EMBED_DIM = 1536

# One shared embedding so every seed chunk matches the query equally — the vector
# channel returns the whole seed set, leaving the filter as the only narrowing force.
_QUERY_TEXT = "shared retrieval anchor"
_SHARED_EMBEDDING = [1.0] + [0.0] * (EMBED_DIM - 1)

# Tz-AWARE bounds (the column is DateTime(timezone=True)). The filter literal is the
# same instant in ISO-8601 Z form so the post-filter compares tz-aware to tz-aware.
_IN_RANGE = datetime(2026, 6, 1, tzinfo=UTC)
_OUT_OF_RANGE = datetime(2020, 1, 1, tzinfo=UTC)
_FILTER_LB = "2026-01-01T00:00:00Z"


def _pg_reachable() -> bool:
    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _pg_reachable(), reason="PostgreSQL not reachable (run `make dev` first)"),
]


async def _reset_public_schema() -> None:
    """Wipe ``public`` and pre-create the wide khora_alembic_version table.

    Mirrors the sibling PG integration modules: alembic creates
    ``khora_alembic_version`` with the default ``VARCHAR(32)`` but several revision
    ids are wider, so pre-create the table with ``VARCHAR(64)`` for the chain to apply
    cleanly. Dropping the public-schema enum types first keeps a half-present enum from
    wedging the re-migrate. A schema-wipe (not a bare ``run_migrations``) so the module
    never inherits a downgraded / partially-dropped shared DB from a preceding test
    file.
    """
    from sqlalchemy import text

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


@pytest.fixture(scope="module")
async def _migrations_once() -> AsyncIterator[None]:
    """Reset and migrate the live PG once for the module."""
    await _reset_public_schema()
    result = await run_migrations(DATABASE_URL)
    assert result.success, f"Migrations failed: {result.error}"
    yield


@pytest.fixture
async def coord(_migrations_once: None) -> AsyncIterator[StorageCoordinator]:
    """A connected coordinator: PostgreSQL relational + pgvector, on one shared engine.

    ``create_namespace`` / ``create_document`` live on the relational backend;
    ``create_chunks_batch`` / ``get_chunk`` / the recall ``search_*`` channels live on
    the pgvector backend. Sharing a single engine keeps both writes landing in the same
    database the recall path later reads.
    """
    engine = create_async_engine(DATABASE_URL)
    relational = PostgreSQLBackend(DATABASE_URL, engine=engine)
    vector = PgVectorBackend(DATABASE_URL, embedding_dimension=EMBED_DIM, engine=engine)
    coordinator = StorageCoordinator(relational=relational, vector=vector)
    await coordinator.connect()
    try:
        yield coordinator
    finally:
        await coordinator.disconnect()
        await engine.dispose()


class _FakeEmbedder:
    async def embed(self, _text: str) -> list[float]:
        return _SHARED_EMBEDDING


def _engine_over(coordinator: StorageCoordinator) -> ChronicleEngine:
    """A ChronicleEngine bound to the live PG coordinator.

    Reranking is disabled (it would lazily pull a cross-encoder on first recall); it
    only reorders candidates, never adds/drops a row, so the filter contract is
    unaffected. The fake embedder returns the shared query embedding so the vector
    channel retrieves the whole seed set.
    """
    engine = ChronicleEngine(KhoraConfig(query=QuerySettings(enable_reranking=False)))
    engine._storage = coordinator
    engine._embedder = _FakeEmbedder()
    engine._connected = True
    return engine


def _filter_ast(wire: dict) -> Any:
    return parse_to_ast(RecallFilter.model_validate(wire))


async def _seed(
    coordinator: StorageCoordinator,
    namespace_id: UUID,
    specs: list[dict[str, Any]],
) -> list[Chunk]:
    """Insert one document + one chunk per spec via the real coordinator write API.

    Each ``spec`` carries ``content`` plus any of ``source_timestamp`` / ``occurred_at``
    / ``created_at``. All chunks share ``_SHARED_EMBEDDING`` so the vector channel
    returns them all.
    """
    chunks: list[Chunk] = []
    for spec in specs:
        doc = Document(
            namespace_id=namespace_id,
            content=spec["content"],
            source="test",
            title=spec["content"][:32],
        )
        await coordinator.create_document(doc)
        chunk_kwargs: dict[str, Any] = {
            "namespace_id": namespace_id,
            "document_id": doc.id,
            "content": spec["content"],
            "chunk_index": 0,
            "embedding": list(_SHARED_EMBEDDING),
            "embedding_model": "fake",
            "metadata": spec.get("metadata", {}),
        }
        for date_key in ("source_timestamp", "occurred_at", "created_at"):
            if date_key in spec:
                chunk_kwargs[date_key] = spec[date_key]
        chunks.append(Chunk(**chunk_kwargs))
    await coordinator.create_chunks_batch(chunks)
    return chunks


async def _recall_ids(engine: ChronicleEngine, namespace_id: UUID, wire: dict) -> set[UUID]:
    result = await engine.recall(
        _QUERY_TEXT,
        namespace_id,
        limit=50,
        mode=SearchMode.VECTOR,
        filter_ast=_filter_ast(wire),
    )
    return {c.id for c in result.chunks}


@pytest.mark.asyncio
async def test_occurred_at_round_trips_through_real_store(coord: StorageCoordinator) -> None:
    # Direct write→read round-trip of the distinct occurred_at column through the real
    # pgvector backend (no filter, no engine). A chunk seeded with an occurred_at that
    # differs from BOTH created_at and source_timestamp must read back with that exact
    # occurred_at — proving the write path persists and the read path restores the
    # column, not silently coalesce it away.
    ns = await coord.create_namespace(MemoryNamespace())
    created_at = datetime(2025, 3, 15, tzinfo=UTC)
    chunks = await _seed(
        coord,
        ns.id,
        [
            {
                "content": "distinct occurred_at",
                "occurred_at": _IN_RANGE,
                "source_timestamp": _OUT_OF_RANGE,
                "created_at": created_at,
            },
        ],
    )
    written = chunks[0]
    # Sanity: all three anchors are genuinely distinct before the round-trip.
    assert written.occurred_at == _IN_RANGE
    assert written.source_timestamp == _OUT_OF_RANGE
    assert written.created_at == created_at
    assert len({written.occurred_at, written.source_timestamp, written.created_at}) == 3

    read_back = await coord.get_chunk(written.id, namespace_id=ns.id)
    assert read_back is not None
    assert read_back.occurred_at == _IN_RANGE, (
        "occurred_at must round-trip through the real store unchanged "
        f"(wrote {written.occurred_at!r}, read back {read_back.occurred_at!r}) — a "
        "regression in the write path would read back NULL"
    )
    # The other anchors stay distinct — occurred_at is not derived from either.
    assert read_back.source_timestamp == _OUT_OF_RANGE
    assert read_back.occurred_at != read_back.source_timestamp
    assert read_back.occurred_at != read_back.created_at


@pytest.mark.asyncio
async def test_occurred_at_filter_honored_over_out_of_range_source_timestamp(
    coord: StorageCoordinator,
) -> None:
    # Recall-filter regression guard. The effective event time is
    # COALESCE(occurred_at, source_timestamp). Seed a chunk whose occurred_at is in
    # range but whose source_timestamp is out of range: an occurred_at recall filter
    # must HONOR it, because the in-range occurred_at — not the out-of-range
    # source_timestamp — is the effective event time.
    #
    # This is the guard for the persist-occurred_at fix: if the write path dropped
    # occurred_at (read back as NULL), the effective event time would fall back to the
    # out-of-range source_timestamp and this chunk would be (wrongly) dropped. That it
    # survives proves occurred_at is genuinely persisted, not silently recovered via
    # COALESCE(occurred_at, source_timestamp).
    #
    # A second chunk with NO occurred_at but an in-range source_timestamp confirms the
    # fallback still recovers (no false-empty); a third chunk with neither anchor in
    # range is the negative case.
    ns = await coord.create_namespace(MemoryNamespace())
    chunks = await _seed(
        coord,
        ns.id,
        [
            # occurred_at in range, source_timestamp out of range → survives ONLY if
            # occurred_at round-trips. This is the regression guard.
            {"content": "occurred honored", "occurred_at": _IN_RANGE, "source_timestamp": _OUT_OF_RANGE},
            # no occurred_at, source_timestamp in range → COALESCE recovers via
            # source_timestamp → survives (proves no false-empty when occurred_at unset).
            {"content": "fallback recover", "source_timestamp": _IN_RANGE},
            # neither anchor in range → dropped.
            {"content": "no anchor in range", "source_timestamp": _OUT_OF_RANGE},
        ],
    )
    honored_id = chunks[0].id
    fallback_id = chunks[1].id

    returned = await _recall_ids(_engine_over(coord), ns.id, {"occurred_at": {"$gte": _FILTER_LB}})

    assert returned == {honored_id, fallback_id}, (
        "occurred_at filter must (1) honor a persisted in-range occurred_at even when "
        "source_timestamp is out of range, and (2) recover event time from "
        "source_timestamp when occurred_at is unset (no false-empty); rows whose "
        "effective event time is out of range are dropped"
    )
    # Explicit regression guard: the honored chunk would be dropped if the write path
    # failed to round-trip occurred_at (its effective event time would fall back to the
    # out-of-range source_timestamp). Assert it survives on its own merits.
    assert honored_id in returned, (
        "chunk with in-range occurred_at + out-of-range source_timestamp must survive — "
        "a regression in occurred_at persistence would drop it"
    )
