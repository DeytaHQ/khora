"""Unit coverage for ``khora.storage.backends.pgvector.PgVectorBackend``.

Complements ``test_pgvector_lock_key.py`` (pure function tests) by exercising
the backend's URL parsing, connect/disconnect lifecycle, health-check, and
lightweight CRUD wrappers using mocked SQLAlchemy ``AsyncEngine`` /
``AsyncSession``. No real PostgreSQL.

The heavyweight paths (similarity search, BM25, batch upserts, bi-temporal
versioning) require real pgvector and are exercised on the integration job.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.db.models import ChunkModel
from khora.storage.backends.pgvector import PgVectorBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(url: str = "postgresql://x/y", *, engine=None) -> PgVectorBackend:
    return PgVectorBackend(url, engine=engine)


def _backend_with_session(session_mock) -> PgVectorBackend:
    """Build a backend whose ``_get_session()`` yields ``session_mock``."""
    backend = _backend()

    @asynccontextmanager
    async def _fake_session():  # type: ignore[no-untyped-def]
        yield session_mock

    backend._get_session = _fake_session  # type: ignore[method-assign,assignment]
    backend._engine = MagicMock()
    backend._session_factory = MagicMock()  # type: ignore[assignment]
    return backend


def _chunk_model(*, embedding=None) -> MagicMock:
    """Build a fake ChunkModel row for ``_chunk_model_to_domain``."""
    m = MagicMock(spec=ChunkModel)
    m.id = uuid4()
    m.namespace_id = uuid4()
    m.document_id = uuid4()
    m.content = "hello"
    m.chunk_index = 1
    m.start_char = 0
    m.end_char = 5
    m.token_count = 2
    m.metadata_ = {"k": "v"}
    m.embedding = embedding
    m.embedding_model = "text-embedding-3-small"
    m.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    m.source_timestamp = None
    m.session_id = None
    return m


# ---------------------------------------------------------------------------
# __init__ — URL parsing and engine shared flag
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInitUrlRewrite:
    def test_postgresql_url_rewritten_to_asyncpg(self) -> None:
        b = _backend("postgresql://localhost/khora")
        assert b._database_url == "postgresql+asyncpg://localhost/khora"

    def test_postgres_url_rewritten_to_asyncpg(self) -> None:
        b = _backend("postgres://localhost/khora")
        assert b._database_url == "postgresql+asyncpg://localhost/khora"

    def test_asyncpg_url_passthrough(self) -> None:
        url = "postgresql+asyncpg://u:p@h:5432/db"
        b = _backend(url)
        assert b._database_url == url

    def test_engine_shared_flag_when_provided(self) -> None:
        engine = MagicMock()
        b = _backend(engine=engine)
        assert b._engine is engine
        assert b._engine_shared is True

    def test_engine_shared_flag_default_false(self) -> None:
        b = _backend()
        assert b._engine_shared is False
        assert b._engine is None

    def test_init_defaults(self) -> None:
        b = _backend()
        assert b._embedding_dimension == 1536
        assert b._pool_size == 10
        assert b._max_overflow == 20
        assert b._hnsw_ef_search == 100
        assert b._use_halfvec is True
        assert b._halfvec_available is None  # detected at connect time
        assert b._session_factory is None

    def test_custom_init_args(self) -> None:
        b = PgVectorBackend(
            "postgresql://x/y",
            embedding_dimension=768,
            pool_size=5,
            max_overflow=15,
            hnsw_ef_search=200,
            use_halfvec=False,
        )
        assert b._embedding_dimension == 768
        assert b._pool_size == 5
        assert b._max_overflow == 15
        assert b._hnsw_ef_search == 200
        assert b._use_halfvec is False


# ---------------------------------------------------------------------------
# halfvec_enabled property
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHalfvecEnabled:
    def test_false_when_use_halfvec_false(self) -> None:
        b = PgVectorBackend("postgresql://x/y", use_halfvec=False)
        assert b.halfvec_enabled is False

    def test_false_when_not_yet_detected(self) -> None:
        b = _backend()
        assert b._halfvec_available is None
        assert b.halfvec_enabled is False  # None is not True

    def test_true_when_both_requested_and_available(self) -> None:
        b = _backend()
        b._halfvec_available = True
        assert b.halfvec_enabled is True

    def test_false_when_detection_negative(self) -> None:
        b = _backend()
        b._halfvec_available = False
        assert b.halfvec_enabled is False


# ---------------------------------------------------------------------------
# is_healthy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsHealthy:
    @pytest.mark.asyncio
    async def test_false_when_engine_none(self) -> None:
        b = _backend()
        assert await b.is_healthy() is False

    @pytest.mark.asyncio
    async def test_false_when_session_factory_none(self) -> None:
        b = _backend()
        b._engine = MagicMock()
        assert await b.is_healthy() is False

    @pytest.mark.asyncio
    async def test_true_on_successful_query(self) -> None:
        b = _backend()
        b._engine = MagicMock()
        session = AsyncMock()
        session.execute = AsyncMock()

        @asynccontextmanager
        async def _ctx():  # type: ignore[no-untyped-def]
            yield session

        b._session_factory = MagicMock(return_value=_ctx())  # type: ignore[assignment]
        assert await b.is_healthy() is True

    @pytest.mark.asyncio
    async def test_false_on_query_error(self) -> None:
        b = _backend()
        b._engine = MagicMock()
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=RuntimeError("connection refused"))

        @asynccontextmanager
        async def _ctx():  # type: ignore[no-untyped-def]
            yield session

        b._session_factory = MagicMock(return_value=_ctx())  # type: ignore[assignment]
        assert await b.is_healthy() is False


# ---------------------------------------------------------------------------
# connect / disconnect lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect_idempotent(self) -> None:
        b = _backend()
        b._session_factory = MagicMock()  # type: ignore[assignment]
        before = b._session_factory
        await b.connect()
        # No new factory was constructed.
        assert b._session_factory is before

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_is_noop(self) -> None:
        b = _backend()
        await b.disconnect()  # no engine -> no-op
        assert b._engine is None

    @pytest.mark.asyncio
    async def test_disconnect_disposes_when_not_shared(self) -> None:
        b = _backend()
        fake_engine = MagicMock()
        fake_engine.dispose = AsyncMock()
        b._engine = fake_engine
        b._engine_shared = False
        b._session_factory = MagicMock()  # type: ignore[assignment]
        await b.disconnect()
        fake_engine.dispose.assert_awaited()
        assert b._engine is None
        assert b._session_factory is None

    @pytest.mark.asyncio
    async def test_disconnect_skips_dispose_when_shared(self) -> None:
        b = _backend()
        fake_engine = MagicMock()
        fake_engine.dispose = AsyncMock()
        b._engine = fake_engine
        b._engine_shared = True
        b._session_factory = MagicMock()  # type: ignore[assignment]
        await b.disconnect()
        fake_engine.dispose.assert_not_called()
        # Engine reference still cleared so subsequent ops can reconnect.
        assert b._engine is None


# ---------------------------------------------------------------------------
# create_tables (deprecated)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateTablesDeprecated:
    @pytest.mark.asyncio
    async def test_raises_when_not_connected(self) -> None:
        b = _backend()
        with pytest.warns(DeprecationWarning):
            with pytest.raises(RuntimeError, match="not connected"):
                await b.create_tables()


# ---------------------------------------------------------------------------
# _chunk_model_to_domain
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkModelToDomain:
    def test_with_null_embedding(self) -> None:
        b = _backend()
        model = _chunk_model(embedding=None)
        chunk = b._chunk_model_to_domain(model)
        assert isinstance(chunk, Chunk)
        assert chunk.id == model.id
        assert chunk.content == "hello"
        assert chunk.embedding is None
        # Post-#748: Chunk.metadata is a flat dict; chunk_index is a
        # top-level field on Chunk itself.
        assert isinstance(chunk.metadata, dict)
        assert chunk.chunk_index == 1

    def test_with_embedding_list(self) -> None:
        b = _backend()
        model = _chunk_model(embedding=[0.1, 0.2, 0.3])
        chunk = b._chunk_model_to_domain(model)
        # NumPy is available in the venv — embedding becomes an ndarray.
        # We assert on iterable content rather than the concrete type.
        assert chunk.embedding is not None
        assert list(chunk.embedding) == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]


# ---------------------------------------------------------------------------
# Mock-session-backed read paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionBackedReads:
    @pytest.mark.asyncio
    async def test_get_chunk_returns_none_when_missing(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.get_chunk(uuid4(), namespace_id=uuid4())
        assert out is None

    @pytest.mark.asyncio
    async def test_get_chunk_returns_domain_chunk(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=_chunk_model())
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.get_chunk(uuid4(), namespace_id=uuid4())
        assert out is not None
        assert out.content == "hello"

    @pytest.mark.asyncio
    async def test_get_chunks_batch_empty(self) -> None:
        b = _backend()
        out = await b.get_chunks_batch([], namespace_id=uuid4())
        assert out == {}

    @pytest.mark.asyncio
    async def test_get_chunks_batch_returns_id_map(self) -> None:
        model = _chunk_model()
        session = AsyncMock()
        result = MagicMock()
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=[model])
        result.scalars = MagicMock(return_value=scalars)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.get_chunks_batch([model.id], namespace_id=uuid4())
        assert model.id in out
        assert out[model.id].content == "hello"

    @pytest.mark.asyncio
    async def test_get_chunks_by_document_returns_ordered_list(self) -> None:
        models = [_chunk_model(), _chunk_model()]
        session = AsyncMock()
        result = MagicMock()
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=models)
        result.scalars = MagicMock(return_value=scalars)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.get_chunks_by_document(uuid4(), namespace_id=uuid4())
        assert len(out) == 2

    @pytest.mark.asyncio
    async def test_count_chunks(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one = MagicMock(return_value=42)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.count_chunks(uuid4())
        assert out == 42

    @pytest.mark.asyncio
    async def test_count_entities(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one = MagicMock(return_value=11)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.count_entities(uuid4())
        assert out == 11

    @pytest.mark.asyncio
    async def test_entity_exists_true(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one = MagicMock(return_value=1)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b.entity_exists(uuid4(), namespace_id=uuid4()) is True

    @pytest.mark.asyncio
    async def test_entity_exists_false(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one = MagicMock(return_value=0)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b.entity_exists(uuid4(), namespace_id=uuid4()) is False

    @pytest.mark.asyncio
    async def test_entity_exists_requires_namespace_kwarg(self) -> None:
        """IDOR — Security: missing ``namespace_id`` must raise TypeError."""
        b = _backend_with_session(AsyncMock())
        with pytest.raises(TypeError):
            await b.entity_exists(uuid4())  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_entity_exists_wrong_namespace_returns_false(self) -> None:
        """Cross-namespace lookups return False — verified by the SQL filter
        producing a zero count."""
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one = MagicMock(return_value=0)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b.entity_exists(uuid4(), namespace_id=uuid4()) is False

    @pytest.mark.asyncio
    async def test_get_entity_requires_namespace_kwarg(self) -> None:
        b = _backend_with_session(AsyncMock())
        with pytest.raises(TypeError):
            await b.get_entity(uuid4())  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_get_entity_wrong_namespace_returns_none(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b.get_entity(uuid4(), namespace_id=uuid4()) is None

    @pytest.mark.asyncio
    async def test_get_entities_batch_requires_namespace_kwarg(self) -> None:
        b = _backend_with_session(AsyncMock())
        with pytest.raises(TypeError):
            await b.get_entities_batch([uuid4()])  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_get_entities_batch_wrong_namespace_returns_empty(self) -> None:
        """SQL filter drops cross-namespace rows — mock returns no models."""
        session = AsyncMock()
        result = MagicMock()
        result.scalars = MagicMock(return_value=[])
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.get_entities_batch([uuid4()], namespace_id=uuid4())
        assert out == {}

    @pytest.mark.asyncio
    async def test_get_embedding_stats(self) -> None:
        # Two calls: chunk_count then entity_count.
        chunk_result = MagicMock()
        chunk_result.scalar_one = MagicMock(return_value=12)
        entity_result = MagicMock()
        entity_result.scalar_one = MagicMock(return_value=3)
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=[chunk_result, entity_result])
        b = _backend_with_session(session)
        out = await b.get_embedding_stats(uuid4())
        assert out == {"chunk_embeddings": 12, "entity_embeddings": 3}


# ---------------------------------------------------------------------------
# Empty-input short-circuits on batch methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmptyShortCircuits:
    @pytest.mark.asyncio
    async def test_create_chunks_batch_empty(self) -> None:
        b = _backend()
        assert await b.create_chunks_batch([]) == []

    @pytest.mark.asyncio
    async def test_delete_entities_batch_empty_returns_zero(self) -> None:
        b = _backend()
        assert await b.delete_entities_batch([], namespace_id=uuid4()) == 0

    @pytest.mark.asyncio
    async def test_delete_relationships_batch_empty_returns_zero(self) -> None:
        b = _backend()
        assert await b.delete_relationships_batch([], namespace_id=uuid4()) == 0

    @pytest.mark.asyncio
    async def test_remove_document_from_entity_sources_empty(self) -> None:
        b = _backend()
        assert await b.remove_document_from_entity_sources([], uuid4()) == 0

    @pytest.mark.asyncio
    async def test_remove_document_from_relationship_sources_empty(self) -> None:
        b = _backend()
        assert await b.remove_document_from_relationship_sources([], uuid4()) == 0

    @pytest.mark.asyncio
    async def test_get_entities_batch_empty(self) -> None:
        b = _backend()
        assert await b.get_entities_batch([], namespace_id=uuid4()) == {}

    @pytest.mark.asyncio
    async def test_get_entities_by_names_batch_empty(self) -> None:
        b = _backend()
        assert await b.get_entities_by_names_batch(uuid4(), []) == {}

    @pytest.mark.asyncio
    async def test_upsert_entities_batch_empty(self) -> None:
        b = _backend()
        assert await b.upsert_entities_batch(uuid4(), []) == []

    @pytest.mark.asyncio
    async def test_update_entity_embeddings_batch_empty(self) -> None:
        b = _backend()
        assert await b.update_entity_embeddings_batch([], namespace_id=uuid4()) == 0

    @pytest.mark.asyncio
    async def test_write_events_empty(self) -> None:
        b = _backend()
        assert await b.write_events([], namespace_id=uuid4()) == []

    @pytest.mark.asyncio
    async def test_write_facts_empty(self) -> None:
        b = _backend()
        assert await b.write_facts([], namespace_id=uuid4()) == []


# ---------------------------------------------------------------------------
# Mock-session-backed write/delete paths returning rowcount
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionBackedDeletes:
    @pytest.mark.asyncio
    async def test_delete_chunks_by_document(self) -> None:
        result = MagicMock()
        result.rowcount = 4
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.delete_chunks_by_document(uuid4(), namespace_id=uuid4())
        assert out == 4
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_entities_batch_returns_rowcount(self) -> None:
        result = MagicMock()
        result.rowcount = 7
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.delete_entities_batch([uuid4(), uuid4()], namespace_id=uuid4())
        assert out == 7

    @pytest.mark.asyncio
    async def test_delete_relationships_batch_returns_rowcount(self) -> None:
        result = MagicMock()
        result.rowcount = 3
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.delete_relationships_batch([uuid4()], namespace_id=uuid4())
        assert out == 3

    @pytest.mark.asyncio
    async def test_remove_document_from_entity_sources(self) -> None:
        result = MagicMock()
        result.rowcount = 2
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.remove_document_from_entity_sources([uuid4()], uuid4())
        assert out == 2

    @pytest.mark.asyncio
    async def test_remove_document_from_relationship_sources(self) -> None:
        result = MagicMock()
        result.rowcount = 1
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.remove_document_from_relationship_sources([uuid4()], uuid4())
        assert out == 1


# ---------------------------------------------------------------------------
# list_chunks / list_entities / list_relationships
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListMethods:
    @pytest.mark.asyncio
    async def test_list_chunks(self) -> None:
        models = [_chunk_model()]
        session = AsyncMock()
        result = MagicMock()
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=models)
        result.scalars = MagicMock(return_value=scalars)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.list_chunks(uuid4(), limit=10, offset=0)
        assert len(out) == 1
        assert out[0].content == "hello"

    @pytest.mark.asyncio
    async def test_list_entities_empty(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        scalars = MagicMock(return_value=iter([]))
        result.scalars = scalars
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.list_entities(uuid4())
        assert out == []

    @pytest.mark.asyncio
    async def test_list_entities_filter_by_type(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        scalars = MagicMock(return_value=iter([]))
        result.scalars = scalars
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.list_entities(uuid4(), entity_type="PERSON")
        assert out == []
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_relationships_empty(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        scalars = MagicMock(return_value=iter([]))
        result.scalars = scalars
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        out = await b.list_relationships(uuid4())
        assert out == []


# ---------------------------------------------------------------------------
# update_entity_embedding — single
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUpdateEntityEmbedding:
    @pytest.mark.asyncio
    async def test_calls_session_with_update(self) -> None:
        session = AsyncMock()
        b = _backend_with_session(session)
        await b.update_entity_embedding(uuid4(), [0.1, 0.2], "model-x", namespace_id=uuid4())
        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# halfvec support detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHalfvecSupportDetection:
    @pytest.mark.asyncio
    async def test_returns_true_for_pgvector_0_7(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.first = MagicMock(return_value=("0.7.0",))
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b._detect_halfvec_support() is True

    @pytest.mark.asyncio
    async def test_returns_true_for_pgvector_0_8(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.first = MagicMock(return_value=("0.8.1",))
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b._detect_halfvec_support() is True

    @pytest.mark.asyncio
    async def test_returns_false_for_pgvector_0_5(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.first = MagicMock(return_value=("0.5.0",))
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b._detect_halfvec_support() is False

    @pytest.mark.asyncio
    async def test_returns_false_when_extension_missing(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.first = MagicMock(return_value=None)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b._detect_halfvec_support() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_query_error(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=RuntimeError("table missing"))
        b = _backend_with_session(session)
        assert await b._detect_halfvec_support() is False


# ---------------------------------------------------------------------------
# halfvec target tables presence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHalfvecTargetTablesExist:
    @pytest.mark.asyncio
    async def test_true_when_both_tables_exist(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=2)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b._halfvec_target_tables_exist() is True

    @pytest.mark.asyncio
    async def test_false_when_one_table_missing(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=1)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b._halfvec_target_tables_exist() is False

    @pytest.mark.asyncio
    async def test_false_when_query_returns_none(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        # None defaults to 0, so 0 != 2 → False.
        assert await b._halfvec_target_tables_exist() is False

    @pytest.mark.asyncio
    async def test_false_on_query_error(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=RuntimeError("fail"))
        b = _backend_with_session(session)
        assert await b._halfvec_target_tables_exist() is False


# ---------------------------------------------------------------------------
# halfvec index validity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckHalfvecIndexes:
    @pytest.mark.asyncio
    async def test_true_when_all_indexes_valid(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.all = MagicMock(
            return_value=[
                ("ix_chunks_embedding_halfvec_hnsw", True),
                ("ix_entities_embedding_halfvec_hnsw", True),
            ]
        )
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b._check_halfvec_indexes() is True

    @pytest.mark.asyncio
    async def test_false_when_one_index_invalid(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.all = MagicMock(
            return_value=[
                ("ix_chunks_embedding_halfvec_hnsw", True),
                ("ix_entities_embedding_halfvec_hnsw", False),
            ]
        )
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b._check_halfvec_indexes() is False

    @pytest.mark.asyncio
    async def test_false_when_indexes_missing(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.all = MagicMock(return_value=[])
        session.execute = AsyncMock(return_value=result)
        b = _backend_with_session(session)
        assert await b._check_halfvec_indexes() is False

    @pytest.mark.asyncio
    async def test_false_on_query_error(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=RuntimeError("fail"))
        b = _backend_with_session(session)
        assert await b._check_halfvec_indexes() is False


# ---------------------------------------------------------------------------
# create_chunk / create_chunks_batch — happy-path session interactions
# ---------------------------------------------------------------------------


def _domain_chunk(*, with_embedding: bool = False) -> Chunk:
    # Post-#748: chunk_index / start_char / end_char / token_count are
    # top-level fields on Chunk; metadata is a flat dict for free-form keys.
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="hello",
        chunk_index=0,
        start_char=0,
        end_char=5,
        token_count=1,
        embedding=[0.1, 0.2] if with_embedding else None,
    )


def _session_with_sync_add() -> AsyncMock:
    """AsyncMock whose ``add`` / ``add_all`` are sync (SQLAlchemy contract)."""
    s = AsyncMock()
    s.add = MagicMock()
    s.add_all = MagicMock()
    return s


@pytest.mark.unit
class TestCreateChunkSession:
    @pytest.mark.asyncio
    async def test_create_chunk_owns_session_commits(self) -> None:
        session = _session_with_sync_add()
        b = _backend_with_session(session)
        chunk = _domain_chunk()
        # PG-side refresh after add: we stub session.refresh + the
        # _chunk_model_to_domain converter to bypass the ORM round-trip.
        b._chunk_model_to_domain = lambda model: chunk  # type: ignore[assignment]
        out = await b.create_chunk(chunk)
        assert out is chunk
        session.commit.assert_awaited_once()
        session.refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_chunk_with_external_session_no_commit(self) -> None:
        session = _session_with_sync_add()
        b = _backend()
        chunk = _domain_chunk()
        b._chunk_model_to_domain = lambda model: chunk  # type: ignore[assignment]
        out = await b.create_chunk(chunk, session=session)
        assert out is chunk
        session.commit.assert_not_called()
        # External-session path uses flush + refresh.
        session.flush.assert_awaited_once()
        session.refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_chunks_batch_owns_session_commits(self) -> None:
        session = _session_with_sync_add()
        b = _backend_with_session(session)
        chunks = [_domain_chunk(), _domain_chunk()]
        out = await b.create_chunks_batch(chunks)
        assert out == chunks
        session.add_all.assert_called_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_chunks_batch_with_external_session_no_commit(self) -> None:
        session = _session_with_sync_add()
        b = _backend()
        chunks = [_domain_chunk()]
        out = await b.create_chunks_batch(chunks, session=session)
        assert out == chunks
        session.add_all.assert_called_once()
        session.commit.assert_not_called()


# ---------------------------------------------------------------------------
# delete_chunks_by_document with external session
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteChunksByDocumentWithExternalSession:
    @pytest.mark.asyncio
    async def test_external_session_no_commit(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.rowcount = 9
        session.execute = AsyncMock(return_value=result)
        b = _backend()
        out = await b.delete_chunks_by_document(uuid4(), namespace_id=uuid4(), session=session)
        assert out == 9
        session.commit.assert_not_called()
