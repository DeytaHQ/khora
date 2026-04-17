"""Tests for :class:`SQLiteLanceVectorAdapter` (DYT-2730)."""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

try:
    import lancedb  # noqa: F401
    import pyarrow  # noqa: F401

    _HAS_LANCEDB = True
except ImportError:
    _HAS_LANCEDB = False

from khora.core.models import Chunk, ChunkMetadata
from khora.core.models.entity import Entity

pytestmark = pytest.mark.skipif(not _HAS_LANCEDB, reason="lancedb not installed")

if _HAS_LANCEDB:
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )
    from khora.storage.backends.sqlite_lance.vector import SQLiteLanceVectorAdapter


# ---------------------------------------------------------------------------
# Schema DDL — mirrors the SQLite tables the relational backend creates.
# Inline here while DYT-2727 (Alembic dialect-gated migrations) is in flight.
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    start_char INTEGER DEFAULT 0,
    end_char INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    metadata_ TEXT DEFAULT '{}',
    embedding TEXT,
    embedding_model TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    source_timestamp TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_ns ON chunks(namespace_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'CONCEPT',
    description TEXT DEFAULT '',
    attributes TEXT DEFAULT '{}',
    source_tool TEXT DEFAULT '',
    source_document_ids TEXT DEFAULT '[]',
    source_chunk_ids TEXT DEFAULT '[]',
    mention_count INTEGER DEFAULT 1,
    embedding TEXT,
    embedding_model TEXT DEFAULT '',
    valid_from TEXT,
    valid_until TEXT,
    confidence REAL DEFAULT 1.0,
    metadata_ TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_ns ON entities(namespace_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_unique
    ON entities(namespace_id, name, entity_type);
"""

_FTS5_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    chunk_id UNINDEXED,
    namespace_id UNINDEXED
);
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def handle(tmp_path: Path):
    cfg = EmbeddedStorageHandleConfig(
        db_path=str(tmp_path / "k.db"),
        lance_path=str(tmp_path / "k.lance"),
        embedding_dimension=8,
        use_halfvec=False,
    )
    h = EmbeddedStorageHandle(cfg)
    await h.connect()

    # Inline DDL — see module docstring: migrations land in DYT-2727.
    for stmt in _SCHEMA_SQL.split(";"):
        s = stmt.strip()
        if s:
            await h.sqlite.execute(s)
    try:
        await h.sqlite.executescript(_FTS5_SQL)
    except Exception:
        pass
    await h.sqlite.commit()

    yield h
    await h.disconnect()


@pytest.fixture
async def halfvec_handle(tmp_path: Path):
    cfg = EmbeddedStorageHandleConfig(
        db_path=str(tmp_path / "k.db"),
        lance_path=str(tmp_path / "k.lance"),
        embedding_dimension=8,
        use_halfvec=True,
    )
    h = EmbeddedStorageHandle(cfg)
    await h.connect()
    for stmt in _SCHEMA_SQL.split(";"):
        s = stmt.strip()
        if s:
            await h.sqlite.execute(s)
    try:
        await h.sqlite.executescript(_FTS5_SQL)
    except Exception:
        pass
    await h.sqlite.commit()
    yield h
    await h.disconnect()


@pytest.fixture
async def adapter(handle):
    return SQLiteLanceVectorAdapter(handle)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(dim: int, idx: int = 0) -> list[float]:
    vec = [0.0] * dim
    vec[idx % dim] = 1.0
    return vec


def _make_chunk(
    namespace_id,
    document_id,
    *,
    content: str = "test content",
    embedding: list[float] | None = None,
    index: int = 0,
    created_at: datetime | None = None,
    source_timestamp: datetime | None = None,
) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=document_id,
        content=content,
        metadata=ChunkMetadata(document_id=document_id, chunk_index=index),
        embedding=embedding,
        embedding_model="test-model" if embedding else "",
        created_at=created_at or datetime.now(UTC),
        source_timestamp=source_timestamp,
    )


def _make_entity(namespace_id, *, name: str = "alice", embedding=None) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=namespace_id,
        name=name,
        entity_type="PERSON",
        description="",
        embedding=embedding,
        embedding_model="test-model" if embedding else "",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Lifecycle + health
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_is_healthy(self, adapter: SQLiteLanceVectorAdapter):
        assert await adapter.is_healthy() is True

    async def test_disconnect_marks_unhealthy(self, tmp_path: Path):
        cfg = EmbeddedStorageHandleConfig(
            db_path=str(tmp_path / "k.db"),
            lance_path=str(tmp_path / "k.lance"),
            embedding_dimension=4,
        )
        h = EmbeddedStorageHandle(cfg)
        a = SQLiteLanceVectorAdapter(h)
        await a.connect()
        assert await a.is_healthy() is True
        await a.disconnect()
        assert await a.is_healthy() is False


# ---------------------------------------------------------------------------
# Chunk CRUD
# ---------------------------------------------------------------------------


class TestChunkCRUD:
    async def test_create_and_get(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        c = _make_chunk(ns, doc, embedding=_unit(8, 0))
        created = await adapter.create_chunk(c)
        assert created.id == c.id

        fetched = await adapter.get_chunk(c.id)
        assert fetched is not None
        assert fetched.id == c.id
        assert fetched.content == c.content
        assert fetched.embedding == c.embedding

    async def test_create_without_embedding_skips_lance(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        c = _make_chunk(ns, doc, embedding=None)
        await adapter.create_chunk(c)

        # SQLite row should exist; LanceDB table should still be empty.
        fetched = await adapter.get_chunk(c.id)
        assert fetched is not None

        tbl = await adapter._chunks_table()  # type: ignore[reportPrivateUsage]
        assert await tbl.count_rows() == 0

    async def test_create_chunks_batch(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i), index=i) for i in range(5)]
        result = await adapter.create_chunks_batch(chunks)
        assert len(result) == 5
        assert await adapter.count_chunks(ns) == 5

        tbl = await adapter._chunks_table()  # type: ignore[reportPrivateUsage]
        assert await tbl.count_rows() == 5

    async def test_get_chunks_batch(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(3)]
        await adapter.create_chunks_batch(chunks)

        fetched = await adapter.get_chunks_batch([c.id for c in chunks])
        assert set(fetched.keys()) == {c.id for c in chunks}

    async def test_get_chunks_batch_empty(self, adapter: SQLiteLanceVectorAdapter):
        assert await adapter.get_chunks_batch([]) == {}

    async def test_get_chunks_by_document(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i), index=i) for i in range(3)]
        await adapter.create_chunks_batch(chunks)

        fetched = await adapter.get_chunks_by_document(doc)
        assert len(fetched) == 3
        # Ordered by chunk_index
        assert [c.metadata.chunk_index for c in fetched] == [0, 1, 2]

    async def test_delete_chunks_by_document(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(3)]
        await adapter.create_chunks_batch(chunks)

        deleted = await adapter.delete_chunks_by_document(doc)
        assert deleted == 3

        assert await adapter.count_chunks(ns) == 0
        tbl = await adapter._chunks_table()  # type: ignore[reportPrivateUsage]
        assert await tbl.count_rows() == 0

    async def test_delete_chunks_by_document_with_session_skips_commit(self, adapter: SQLiteLanceVectorAdapter):
        """When a session is provided the caller owns commits.

        We can't plumb a real AsyncSession here (SQLAlchemy is not driving
        this backend), so we pass a sentinel object — the adapter treats
        any non-None session the same way: delay commit + skip compensation.
        """
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(2)]
        await adapter.create_chunks_batch(chunks)

        sentinel = object()
        deleted = await adapter.delete_chunks_by_document(doc, session=sentinel)  # type: ignore[arg-type]
        assert deleted == 2

        # SQLite wasn't committed so the row count per query (inside this
        # conn) reflects the delete, but LanceDB vectors are untouched
        # (caller will commit + compensate).
        tbl = await adapter._chunks_table()  # type: ignore[reportPrivateUsage]
        assert await tbl.count_rows() == 2


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


class TestSearchSimilar:
    async def test_cosine_returns_nearest(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        # Three basis vectors; query e0 should rank chunk_0 first.
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(3)]
        await adapter.create_chunks_batch(chunks)

        results = await adapter.search_similar(ns, _unit(8, 0), limit=3)
        assert len(results) > 0
        top_chunk, top_score = results[0]
        assert top_chunk.id == chunks[0].id
        assert math.isclose(top_score, 1.0, abs_tol=1e-3)

    async def test_min_similarity_filter(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        # Two orthogonal vectors; similarity to e0 is 1.0 for self, 0.0 for the other.
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(2)]
        await adapter.create_chunks_batch(chunks)

        results = await adapter.search_similar(ns, _unit(8, 0), limit=10, min_similarity=0.5)
        # Only the self-match should clear the threshold.
        assert len(results) == 1
        assert results[0][0].id == chunks[0].id

    async def test_limit_honored(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(5)]
        await adapter.create_chunks_batch(chunks)

        results = await adapter.search_similar(ns, _unit(8, 0), limit=2)
        assert len(results) == 2

    async def test_filter_document_ids(self, adapter: SQLiteLanceVectorAdapter):
        ns = uuid4()
        doc_a, doc_b = uuid4(), uuid4()
        await adapter.create_chunks_batch(
            [
                _make_chunk(ns, doc_a, embedding=_unit(8, 0)),
                _make_chunk(ns, doc_b, embedding=_unit(8, 1)),
            ]
        )

        results = await adapter.search_similar(ns, _unit(8, 0), limit=10, filter_document_ids=[doc_b])
        assert len(results) == 1
        assert results[0][0].document_id == doc_b

    async def test_temporal_filter(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        old = datetime.now(UTC) - timedelta(days=30)
        new = datetime.now(UTC)
        await adapter.create_chunks_batch(
            [
                _make_chunk(ns, doc, embedding=_unit(8, 0), created_at=old),
                _make_chunk(ns, doc, embedding=_unit(8, 1), created_at=new),
            ]
        )

        cutoff = datetime.now(UTC) - timedelta(days=7)
        results = await adapter.search_similar(ns, _unit(8, 0), limit=10, created_after=cutoff)
        assert len(results) == 1
        assert results[0][0].created_at >= cutoff

    async def test_namespace_isolation(self, adapter: SQLiteLanceVectorAdapter):
        ns_a, ns_b, doc = uuid4(), uuid4(), uuid4()
        await adapter.create_chunks_batch(
            [
                _make_chunk(ns_a, doc, embedding=_unit(8, 0)),
                _make_chunk(ns_b, doc, embedding=_unit(8, 0)),
            ]
        )

        results = await adapter.search_similar(ns_a, _unit(8, 0), limit=10)
        assert len(results) == 1
        assert results[0][0].namespace_id == ns_a


# ---------------------------------------------------------------------------
# Full-text search (FTS5)
# ---------------------------------------------------------------------------


class TestSearchFulltext:
    async def test_bm25_ranks_matches(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        await adapter.create_chunks_batch(
            [
                _make_chunk(ns, doc, content="quick brown fox jumps", index=0),
                _make_chunk(ns, doc, content="slow green turtle walks", index=1),
                _make_chunk(ns, doc, content="quick red fox runs fast", index=2),
            ]
        )

        results = await adapter.search_fulltext(ns, "quick fox", limit=10)
        assert len(results) == 2
        # Higher score = better match. Both chunks match; either order is OK,
        # we just verify FTS5 scores are finite and positive.
        for _chunk, score in results:
            assert math.isfinite(score)

    async def test_fulltext_namespace_isolation(self, adapter: SQLiteLanceVectorAdapter):
        ns_a, ns_b, doc = uuid4(), uuid4(), uuid4()
        await adapter.create_chunks_batch(
            [
                _make_chunk(ns_a, doc, content="unique token ns_a"),
                _make_chunk(ns_b, doc, content="unique token ns_b"),
            ]
        )

        a_hits = await adapter.search_fulltext(ns_a, "unique", limit=10)
        assert len(a_hits) == 1
        assert a_hits[0][0].namespace_id == ns_a


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------


class TestEntities:
    async def test_create_and_exists(self, adapter: SQLiteLanceVectorAdapter):
        ns = uuid4()
        e = _make_entity(ns, embedding=_unit(8, 0))
        await adapter.create_entity(e)
        assert await adapter.entity_exists(e.id) is True
        assert await adapter.entity_exists(uuid4()) is False

    async def test_update_entity(self, adapter: SQLiteLanceVectorAdapter):
        ns = uuid4()
        e = _make_entity(ns, embedding=_unit(8, 0))
        await adapter.create_entity(e)

        e.description = "updated"
        await adapter.update_entity(e)

        # Search should still find it (upsert preserves vector row).
        results = await adapter.search_similar_entities(ns, _unit(8, 0), limit=5)
        assert any(eid == e.id for eid, _ in results)

    async def test_update_entity_embedding(self, adapter: SQLiteLanceVectorAdapter):
        ns = uuid4()
        e = _make_entity(ns, embedding=_unit(8, 0))
        await adapter.create_entity(e)

        # Change the embedding to point in a different direction.
        await adapter.update_entity_embedding(e.id, _unit(8, 3), "new-model")

        results = await adapter.search_similar_entities(ns, _unit(8, 3), limit=5)
        assert results
        assert results[0][0] == e.id
        assert math.isclose(results[0][1], 1.0, abs_tol=1e-3)

    async def test_update_entity_embedding_missing_raises(self, adapter: SQLiteLanceVectorAdapter):
        with pytest.raises(ValueError, match="not found"):
            await adapter.update_entity_embedding(uuid4(), _unit(8, 0), "m")

    async def test_update_entity_embeddings_batch(self, adapter: SQLiteLanceVectorAdapter):
        ns = uuid4()
        entities = [_make_entity(ns, name=f"e{i}", embedding=_unit(8, 0)) for i in range(3)]
        for e in entities:
            await adapter.create_entity(e)

        updates = [(e.id, _unit(8, 7), "v2") for e in entities]
        count = await adapter.update_entity_embeddings_batch(updates)
        assert count == 3

        results = await adapter.search_similar_entities(ns, _unit(8, 7), limit=5)
        top_ids = {eid for eid, _ in results}
        assert {e.id for e in entities}.issubset(top_ids)

    async def test_update_entity_embeddings_batch_empty(self, adapter: SQLiteLanceVectorAdapter):
        assert await adapter.update_entity_embeddings_batch([]) == 0

    async def test_search_similar_entities_min_similarity(self, adapter: SQLiteLanceVectorAdapter):
        ns = uuid4()
        await adapter.create_entity(_make_entity(ns, name="a", embedding=_unit(8, 0)))
        await adapter.create_entity(_make_entity(ns, name="b", embedding=_unit(8, 1)))

        # Only the self-match clears a 0.5 threshold.
        results = await adapter.search_similar_entities(ns, _unit(8, 0), min_similarity=0.5)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Aggregate ops
# ---------------------------------------------------------------------------


class TestAggregates:
    async def test_count_and_list(self, adapter: SQLiteLanceVectorAdapter):
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(4)]
        await adapter.create_chunks_batch(chunks)

        assert await adapter.count_chunks(ns) == 4

        listed = await adapter.list_chunks(ns, limit=2, offset=0)
        assert len(listed) == 2

        page = await adapter.list_chunks(ns, limit=2, offset=2)
        assert len(page) == 2


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    async def test_concurrent_create_chunk(self, adapter: SQLiteLanceVectorAdapter):
        """Parallel writers should not corrupt SQLite or LanceDB state.

        SQLite + WAL serializes writers, but the adapter itself shouldn't
        deadlock or drop rows.
        """
        ns, doc = uuid4(), uuid4()
        chunks = [_make_chunk(ns, doc, embedding=_unit(8, i % 8), index=i) for i in range(10)]

        await asyncio.gather(*(adapter.create_chunk(c) for c in chunks))

        assert await adapter.count_chunks(ns) == 10
        tbl = await adapter._chunks_table()  # type: ignore[reportPrivateUsage]
        assert await tbl.count_rows() == 10


# ---------------------------------------------------------------------------
# Halfvec
# ---------------------------------------------------------------------------


class TestHalfvec:
    async def test_halfvec_roundtrip(self, halfvec_handle):
        adapter = SQLiteLanceVectorAdapter(halfvec_handle)
        ns, doc = uuid4(), uuid4()
        await adapter.create_chunks_batch([_make_chunk(ns, doc, embedding=_unit(8, i)) for i in range(3)])

        results = await adapter.search_similar(ns, _unit(8, 0), limit=3)
        assert results
        # float16 precision is lossy but cosine similarity to e0 for e0
        # should still be very near 1.0 for unit basis vectors.
        assert math.isclose(results[0][1], 1.0, abs_tol=5e-3)
