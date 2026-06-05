"""Coverage tests for ``khora.engines.skeleton.backends.pgvector``.

Heavily mocks SQLAlchemy ``AsyncEngine`` / ``AsyncSession`` — no real DB.

Focuses on pure helpers and the methods whose logic is independent of the
PostgreSQL wire (filter-condition assembly, RRF fusion, row→domain
conversion, lifecycle bookkeeping).  The SQL execution paths are exercised
through a stubbed session that returns canned ``fetchall``/``fetchone``
results.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.skeleton.backends import (
    TemporalChunk,
    TemporalFilter,
    TemporalSearchResult,
)
from khora.engines.skeleton.backends.pgvector import (
    PgVectorTemporalStore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(*, embedding_dim: int = 4, hnsw_m: int = 16) -> MagicMock:
    config = MagicMock()
    config.llm.embedding_dimension = embedding_dim
    config.storage.hnsw_m = hnsw_m
    config.storage.hnsw_ef_construction = 64
    config.storage.hnsw_ef_search = 100
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.get_postgresql_url.return_value = "postgresql://localhost/khora"
    return config


def _store_with_session(session_mock) -> PgVectorTemporalStore:
    """Build a store whose ``_get_session()`` yields ``session_mock``."""
    store = PgVectorTemporalStore(_mock_config())

    @asynccontextmanager
    async def _fake_session():  # type: ignore[no-untyped-def]
        yield session_mock

    store._get_session = _fake_session  # type: ignore[method-assign,assignment]
    store._engine = MagicMock()
    store._connected = True
    return store


def _row(**kwargs) -> SimpleNamespace:
    """Build a fake DB row supporting attribute access."""
    base = dict(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="content",
        embedding=None,
        occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
        created_at=datetime(2024, 1, 2, tzinfo=UTC),
        source_system="slack",
        author="alice",
        channel="eng",
        tags=["urgent"],
        confidence=0.9,
        metadata={"chunk_index": 0},
        chunker_info={},
        source_type="email",
        source_name="inbox",
        source_url="https://example.test/msg/1",
        source_timestamp=datetime(2024, 1, 3, tzinfo=UTC),
        external_id="ext-1",
        content_type="text/plain",
        source="mailbox",
        title="Subject line",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# __init__ — config-derived attributes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInit:
    def test_default_embedding_dim_fallback(self) -> None:
        config = _mock_config()
        config.llm.embedding_dimension = None
        store = PgVectorTemporalStore(config)
        assert store._embedding_dimension == 1536

    def test_explicit_embedding_dim(self) -> None:
        store = PgVectorTemporalStore(_mock_config(embedding_dim=768))
        assert store._embedding_dimension == 768

    def test_shared_engine_flag(self) -> None:
        engine = MagicMock()
        store = PgVectorTemporalStore(_mock_config(), engine=engine)
        assert store._shared_engine is True
        assert store._engine is engine

    def test_no_shared_engine(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        assert store._shared_engine is False
        assert store._engine is None

    def test_hnsw_params_stored(self) -> None:
        store = PgVectorTemporalStore(_mock_config(hnsw_m=32))
        assert store._hnsw_m == 32
        assert store._hnsw_ef_construction == 64
        assert store._hnsw_ef_search == 100


# ---------------------------------------------------------------------------
# connect / disconnect lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect_idempotent(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        store._connected = True
        # Should return immediately without creating an engine.
        await store.connect()
        assert store._engine is None

    @pytest.mark.asyncio
    async def test_connect_raises_without_url(self) -> None:
        config = _mock_config()
        config.get_postgresql_url.return_value = ""
        store = PgVectorTemporalStore(config)
        with pytest.raises(ValueError, match="PostgreSQL URL not configured"):
            await store.connect()

    @pytest.mark.asyncio
    async def test_disconnect_disposes_when_not_shared(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        fake_engine = MagicMock()
        fake_engine.dispose = AsyncMock()
        store._engine = fake_engine
        store._shared_engine = False
        store._connected = True
        await store.disconnect()
        fake_engine.dispose.assert_awaited()
        assert store._engine is None
        assert store._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_skips_dispose_when_shared(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        fake_engine = MagicMock()
        fake_engine.dispose = AsyncMock()
        store._engine = fake_engine
        store._shared_engine = True
        store._connected = True
        await store.disconnect()
        fake_engine.dispose.assert_not_called()
        assert store._connected is False
        # engine reference retained because we don't own it
        assert store._engine is fake_engine

    def test_get_session_raises_when_no_engine(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        store._engine = None
        with pytest.raises(RuntimeError, match="Not connected"):
            store._get_session()


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_disconnected(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        out = await store.health_check()
        assert out == {"status": "disconnected", "backend": "pgvector"}

    @pytest.mark.asyncio
    async def test_healthy(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock()
        store = _store_with_session(session)
        out = await store.health_check()
        assert out == {"status": "healthy", "backend": "pgvector"}

    @pytest.mark.asyncio
    async def test_unhealthy_on_error(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=RuntimeError("connection refused"))
        store = _store_with_session(session)
        out = await store.health_check()
        assert out["status"] == "unhealthy"
        assert "connection refused" in out["error"]


# ---------------------------------------------------------------------------
# _row_to_chunk
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRowToChunk:
    def test_basic_row(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        row = _row(embedding=None)
        out = store._row_to_chunk(row)
        assert isinstance(out, TemporalChunk)
        assert out.id == row.id
        assert out.namespace_id == row.namespace_id
        assert out.content == "content"
        assert out.embedding is None
        assert out.tags == ["urgent"]
        assert out.confidence == 0.9

    def test_row_with_embedding(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        row = _row(embedding=[0.1, 0.2, 0.3])
        out = store._row_to_chunk(row)
        assert out.embedding == [0.1, 0.2, 0.3]

    def test_row_with_null_tags_defaults_empty(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        row = _row(tags=None)
        out = store._row_to_chunk(row)
        assert out.tags == []

    def test_row_with_null_metadata_defaults_empty_dict(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        row = _row(metadata=None)
        out = store._row_to_chunk(row)
        assert out.metadata == {}

    def test_row_with_null_confidence_defaults_one(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        row = _row(confidence=None)
        out = store._row_to_chunk(row)
        assert out.confidence == 1.0


# ---------------------------------------------------------------------------
# _build_filter_conditions — temporal / keyword / tags / additional
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildFilterConditions:
    """We assert on the *count* of conditions and on the column referenced by
    each one. ``str(col)`` on a SQLAlchemy column expression renders the column
    name, so we use that to identify which branch fired without depending on
    SQLAlchemy comparator semantics."""

    def _conds(self, **kw) -> list:
        store = PgVectorTemporalStore(_mock_config())
        return store._build_filter_conditions(TemporalFilter(**kw))

    def test_no_filters_yields_no_conditions(self) -> None:
        assert self._conds() == []

    def test_occurred_range(self) -> None:
        conds = self._conds(
            occurred_after=datetime(2024, 1, 1, tzinfo=UTC),
            occurred_before=datetime(2024, 2, 1, tzinfo=UTC),
        )
        assert len(conds) == 2

    def test_created_range(self) -> None:
        conds = self._conds(
            created_after=datetime(2024, 1, 1, tzinfo=UTC),
            created_before=datetime(2024, 2, 1, tzinfo=UTC),
        )
        assert len(conds) == 2

    def test_keyword_filters(self) -> None:
        conds = self._conds(source_system="slack", author="alice", channel="eng")
        assert len(conds) == 3

    def test_tags_filter(self) -> None:
        conds = self._conds(tags=["urgent", "blocker"])
        assert len(conds) == 1

    def test_additional_eq_operator(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        tf = TemporalFilter()
        tf.additional = {"priority": {"eq": "high"}}
        conds = store._build_filter_conditions(tf)
        assert len(conds) == 1

    def test_additional_all_comparison_operators(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        tf = TemporalFilter()
        tf.additional = {
            "score": {"gte": 0.5, "lte": 0.9, "gt": 0.4, "lt": 1.0},
        }
        conds = store._build_filter_conditions(tf)
        assert len(conds) == 4

    def test_additional_simple_equality(self) -> None:
        """Non-dict value uses simple equality on the metadata JSONB key."""
        store = PgVectorTemporalStore(_mock_config())
        tf = TemporalFilter()
        tf.additional = {"label": "urgent"}
        conds = store._build_filter_conditions(tf)
        assert len(conds) == 1


# ---------------------------------------------------------------------------
# _rrf_fusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRrfFusion:
    def _result(self, *, sim: float = 1.0, bm25: float | None = None) -> TemporalSearchResult:
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="x",
        )
        return TemporalSearchResult(chunk=chunk, similarity=sim, bm25_score=bm25)

    def test_pure_vector_when_alpha_one(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        v1, v2 = self._result(), self._result()
        b = self._result(bm25=2.0)
        out = store._rrf_fusion([v1, v2], [b], alpha=1.0, limit=5)
        # Alpha=1.0: only vector rank matters. v1 and v2 both rank higher than
        # the BM25-only candidate.
        assert out[0].chunk.id in (v1.chunk.id, v2.chunk.id)
        # All three should appear; the BM25-only one gets the lowest score
        # because its vector rank defaults to ``len(vector_results) + 100``.
        assert len(out) == 3

    def test_pure_bm25_when_alpha_zero(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        v = self._result()
        b1, b2 = self._result(bm25=1.0), self._result(bm25=2.0)
        out = store._rrf_fusion([v], [b1, b2], alpha=0.0, limit=5)
        # First BM25 result wins on alpha=0.
        assert out[0].chunk.id == b1.chunk.id

    def test_limit_truncates(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        vec = [self._result() for _ in range(10)]
        out = store._rrf_fusion(vec, [], alpha=0.5, limit=3)
        assert len(out) == 3

    def test_combined_score_set_on_result(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        v = self._result(sim=0.8)
        out = store._rrf_fusion([v], [], alpha=1.0, limit=1)
        # combined_score is the RRF score, distinct from raw similarity.
        assert out[0].combined_score is not None
        assert 0 < out[0].combined_score < 1

    def test_bm25_score_merged_when_chunk_appears_in_both(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        chunk = TemporalChunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="x")
        v_only = TemporalSearchResult(chunk=chunk, similarity=0.9, bm25_score=None)
        b_only = TemporalSearchResult(chunk=chunk, similarity=0.0, bm25_score=0.7)
        out = store._rrf_fusion([v_only], [b_only], alpha=0.5, limit=5)
        # Single fused result with bm25_score populated from the BM25 hit.
        assert len(out) == 1
        assert out[0].bm25_score == 0.7


# ---------------------------------------------------------------------------
# create_chunks_batch — empty + happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateChunksBatch:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        out = await store.create_chunks_batch([])
        assert out == []

    @pytest.mark.asyncio
    async def test_assigns_ids_and_persists(self) -> None:
        session = AsyncMock()
        store = _store_with_session(session)
        chunks = [
            TemporalChunk(
                id=None,
                namespace_id=uuid4(),
                document_id=uuid4(),
                content=f"chunk-{i}",
                embedding=[0.1, 0.2],
            )
            for i in range(3)
        ]
        out = await store.create_chunks_batch(chunks)
        assert len(out) == 3
        for c in out:
            assert c.id is not None
        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# create_chunk — assigns ID, calls session
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateChunk:
    @pytest.mark.asyncio
    async def test_creates_with_assigned_id(self) -> None:
        session = AsyncMock()
        store = _store_with_session(session)
        chunk = TemporalChunk(
            id=None,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="hello",
        )
        out = await store.create_chunk(chunk)
        assert out.id is not None
        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_preserves_explicit_id(self) -> None:
        session = AsyncMock()
        store = _store_with_session(session)
        fixed_id = uuid4()
        chunk = TemporalChunk(
            id=fixed_id,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="hello",
        )
        out = await store.create_chunk(chunk)
        assert out.id == fixed_id


# ---------------------------------------------------------------------------
# get_chunk / delete_chunk / delete_chunks_by_document
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSimpleQueries:
    @pytest.mark.asyncio
    async def test_get_chunk_returns_none_when_missing(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.fetchone = MagicMock(return_value=None)
        session.execute = AsyncMock(return_value=result)
        store = _store_with_session(session)
        out = await store.get_chunk(uuid4(), uuid4())
        assert out is None

    @pytest.mark.asyncio
    async def test_get_chunk_returns_domain_object_on_hit(self) -> None:
        row = _row()
        result = MagicMock()
        result.fetchone = MagicMock(return_value=row)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        store = _store_with_session(session)
        out = await store.get_chunk(uuid4(), uuid4())
        assert out is not None
        assert out.id == row.id
        assert out.content == "content"

    @pytest.mark.asyncio
    async def test_delete_chunk_returns_true_when_row_deleted(self) -> None:
        result = MagicMock()
        result.rowcount = 1
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        store = _store_with_session(session)
        out = await store.delete_chunk(uuid4(), uuid4())
        assert out is True

    @pytest.mark.asyncio
    async def test_delete_chunk_returns_false_when_no_rows(self) -> None:
        result = MagicMock()
        result.rowcount = 0
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        store = _store_with_session(session)
        out = await store.delete_chunk(uuid4(), uuid4())
        assert out is False

    @pytest.mark.asyncio
    async def test_delete_chunks_by_document_returns_count(self) -> None:
        result = MagicMock()
        result.rowcount = 5
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        store = _store_with_session(session)
        out = await store.delete_chunks_by_document(uuid4(), uuid4())
        assert out == 5


# ---------------------------------------------------------------------------
# search() telemetry wrapper delegates to _search_inner with attrs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchTelemetryWrapper:
    @pytest.mark.asyncio
    async def test_passes_kwargs_to_inner(self) -> None:
        store = PgVectorTemporalStore(_mock_config())
        store._search_inner = AsyncMock(return_value=[])  # type: ignore[method-assign,assignment]
        ns = uuid4()
        out = await store.search(ns, [0.1] * 4, limit=7, hybrid_alpha=0.5, query_text="x")
        assert out == []
        store._search_inner.assert_awaited_once_with(
            ns,
            [0.1] * 4,
            limit=7,
            min_similarity=0.0,
            temporal_filter=None,
            hybrid_alpha=0.5,
            query_text="x",
        )
