"""Coverage: ``SkeletonConstructionEngine`` happy paths and helpers.

Pre-PR coverage of ``engines/skeleton/engine.py`` was 36%. The big
uncovered surfaces are:

- ``recall()`` — query → embed → temporal_store.search → RecallResult
- ``forget()`` — ns mismatch + happy path
- ``_build_temporal_filter_from_dict`` — every operator branch
- ``_parse_datetime`` — date / iso / Z-suffix / invalid
- engine init backend auto-detection
- ``_get_*`` accessors raising before connect

Tests stub all I/O and never require real backends.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.skeleton.backends import TemporalChunk, TemporalFilter, TemporalSearchResult
from khora.engines.skeleton.engine import SkeletonConstructionEngine
from khora.query import SearchMode


def _mock_config(*, backend: str = "pgvector") -> MagicMock:
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.get_neo4j_url.return_value = None
    config.get_graph_config.return_value = None
    config.get_vector_config.return_value = None
    config.storage.embedding_dimension = 1536
    config.storage.backend = backend
    config.storage.surrealdb = MagicMock()
    config.storage.sqlite_lance = MagicMock()
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.postgresql_pool_pre_ping = False
    config.storage.use_halfvec = True
    config.llm.model = "gpt-4o-mini"
    config.llm.embedding_model = "text-embedding-3-small"
    config.llm.embedding_dimension = 1536
    config.llm.timeout = 30
    config.llm.max_retries = 3
    config.pipeline.chunking_strategy = "recursive"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 200
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


def _connected(backend: str = "pgvector") -> SkeletonConstructionEngine:
    """Build a connected engine with mock storage / embedder / temporal store."""
    eng = SkeletonConstructionEngine(_mock_config(backend=backend), backend=backend)
    eng._connected = True
    eng._storage = AsyncMock()
    eng._embedder = AsyncMock()
    eng._temporal_store = AsyncMock()
    return eng


# ---------------------------------------------------------------------------
# __init__ — backend auto-detection
# ---------------------------------------------------------------------------


class TestInitBackendAutoDetect:
    def test_default_pgvector(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        assert eng._backend_type == "pgvector"

    def test_surrealdb_auto_detected(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config(backend="surrealdb"))
        assert eng._backend_type == "surrealdb"

    def test_sqlite_lance_auto_detected(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config(backend="sqlite_lance"))
        assert eng._backend_type == "sqlite_lance"

    def test_explicit_backend_overrides_auto(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config(backend="surrealdb"), backend="weaviate")
        # explicit "weaviate" stays — auto-detect only triggers when default ("pgvector") is passed
        assert eng._backend_type == "weaviate"


# ---------------------------------------------------------------------------
# Pre-connect guards
# ---------------------------------------------------------------------------


class TestPreConnectGuards:
    def test_get_storage_raises(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        with pytest.raises(RuntimeError, match="not connected"):
            eng._get_storage()

    def test_get_embedder_raises(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        with pytest.raises(RuntimeError, match="not connected"):
            eng._get_embedder()

    def test_get_temporal_store_raises(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        with pytest.raises(RuntimeError, match="not connected"):
            eng._get_temporal_store()


# ---------------------------------------------------------------------------
# disconnect early-return
# ---------------------------------------------------------------------------


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_is_noop(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        # No raises, no AsyncMock setup needed
        await eng.disconnect()
        assert eng._connected is False


# ---------------------------------------------------------------------------
# _parse_datetime
# ---------------------------------------------------------------------------


class TestParseDatetime:
    def test_datetime_passthrough(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        assert eng._parse_datetime(dt) == dt

    def test_naive_datetime_gets_utc(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        dt = datetime(2024, 1, 1)
        result = eng._parse_datetime(dt)
        assert result.tzinfo == UTC

    def test_date_only_string(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        result = eng._parse_datetime("2024-03-15")
        assert result == datetime(2024, 3, 15, tzinfo=UTC)

    def test_iso_with_z_suffix(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        result = eng._parse_datetime("2024-03-15T10:00:00Z")
        assert result == datetime(2024, 3, 15, 10, 0, 0, tzinfo=UTC)

    def test_iso_with_offset(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        result = eng._parse_datetime("2024-03-15T10:00:00+00:00")
        assert result == datetime(2024, 3, 15, 10, 0, 0, tzinfo=UTC)

    def test_invalid_string_raises(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            eng._parse_datetime("not-a-date")

    def test_invalid_type_raises(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            eng._parse_datetime(12345)


# ---------------------------------------------------------------------------
# _build_temporal_filter_from_dict
# ---------------------------------------------------------------------------


class TestBuildTemporalFilter:
    def test_occurred_at_gte_lt(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"occurred_at": {"gte": "2024-01-01", "lt": "2024-02-01"}})
        assert tf.occurred_after == datetime(2024, 1, 1, tzinfo=UTC)
        assert tf.occurred_before == datetime(2024, 2, 1, tzinfo=UTC)

    def test_occurred_at_gt_lte(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"occurred_at": {"gt": "2024-01-01", "lte": "2024-02-01"}})
        assert tf.occurred_after == datetime(2024, 1, 1, tzinfo=UTC)
        assert tf.occurred_before == datetime(2024, 2, 1, tzinfo=UTC)

    def test_created_at_all_operators(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"created_at": {"gte": "2024-01-01", "lte": "2024-12-31"}})
        assert tf.created_after == datetime(2024, 1, 1, tzinfo=UTC)
        assert tf.created_before == datetime(2024, 12, 31, tzinfo=UTC)

    def test_created_at_gt_lt(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"created_at": {"gt": "2024-01-01", "lt": "2024-12-31"}})
        assert tf.created_after == datetime(2024, 1, 1, tzinfo=UTC)
        assert tf.created_before == datetime(2024, 12, 31, tzinfo=UTC)

    def test_source_system_eq(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"source_system": {"eq": "slack"}})
        assert tf.source_system == "slack"

    def test_author_channel_eq(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"author": {"eq": "alice"}, "channel": {"eq": "engineering"}})
        assert tf.author == "alice"
        assert tf.channel == "engineering"

    def test_tags_contains(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"tags": {"contains": ["urgent", "blocker"]}})
        assert tf.tags == ["urgent", "blocker"]

    def test_tags_eq_string(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"tags": {"eq": "important"}})
        assert tf.tags == ["important"]

    def test_tags_eq_list(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"tags": {"eq": ["a", "b"]}})
        assert tf.tags == ["a", "b"]

    def test_unknown_key_goes_to_additional(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"confidence": {"gte": 0.8}})
        assert tf.additional == {"confidence": {"gte": 0.8}}

    def test_scalar_value_wrapped_as_eq(self) -> None:
        """Non-dict value is treated as an ``eq`` operator."""
        eng = SkeletonConstructionEngine(_mock_config())
        tf = eng._build_temporal_filter_from_dict({"author": "bob"})
        assert tf.author == "bob"


# ---------------------------------------------------------------------------
# _adjust_relative_time — placeholder, returns input
# ---------------------------------------------------------------------------


class TestAdjustRelativeTime:
    def test_returns_filter_unchanged(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        tf = TemporalFilter()
        out = eng._adjust_relative_time(tf, datetime.now(UTC))
        assert out is tf


# ---------------------------------------------------------------------------
# recall — happy path
# ---------------------------------------------------------------------------


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_happy_path(self) -> None:
        eng = _connected()
        ns = uuid4()
        doc = uuid4()
        chunk_id = uuid4()
        chunk = TemporalChunk(
            id=chunk_id,
            namespace_id=ns,
            document_id=doc,
            content="hello",
            occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
            created_at=datetime(2024, 1, 2, tzinfo=UTC),
            metadata={"chunk_index": 0},
        )
        eng._embedder.embed = AsyncMock(return_value=[0.1] * 4)
        eng._temporal_store.search = AsyncMock(
            return_value=[TemporalSearchResult(chunk=chunk, similarity=0.85, combined_score=0.9)]
        )

        result = await eng.recall("hello world", ns, limit=3)

        assert len(result.chunks) == 1
        returned_chunk = result.chunks[0]
        assert returned_chunk.content == "hello"
        # Single-chunk recall: min-max normalization collapses to 1.0 (#834).
        assert returned_chunk.score == 1.0
        assert "hello" in returned_chunk.content
        assert result.engine_info["backend"] == "pgvector"
        eng._temporal_store.search.assert_awaited_once()
        # default mode HYBRID → hybrid_alpha 0.7
        _, kwargs = eng._temporal_store.search.call_args
        assert kwargs["hybrid_alpha"] == 0.7

    @pytest.mark.asyncio
    async def test_recall_vector_mode_uses_alpha_1(self) -> None:
        eng = _connected()
        eng._embedder.embed = AsyncMock(return_value=[0.0])
        eng._temporal_store.search = AsyncMock(return_value=[])
        await eng.recall("q", uuid4(), mode=SearchMode.VECTOR)
        _, kwargs = eng._temporal_store.search.call_args
        assert kwargs["hybrid_alpha"] == 1.0

    @pytest.mark.asyncio
    async def test_recall_keyword_mode_uses_alpha_0(self) -> None:
        eng = _connected()
        eng._embedder.embed = AsyncMock(return_value=[0.0])
        eng._temporal_store.search = AsyncMock(return_value=[])
        await eng.recall("q", uuid4(), mode=SearchMode.KEYWORD)
        _, kwargs = eng._temporal_store.search.call_args
        assert kwargs["hybrid_alpha"] == 0.0

    @pytest.mark.asyncio
    async def test_recall_explicit_alpha_overrides_mode(self) -> None:
        eng = _connected()
        eng._embedder.embed = AsyncMock(return_value=[0.0])
        eng._temporal_store.search = AsyncMock(return_value=[])
        await eng.recall("q", uuid4(), mode=SearchMode.HYBRID, hybrid_alpha=0.3)
        _, kwargs = eng._temporal_store.search.call_args
        assert kwargs["hybrid_alpha"] == 0.3

    @pytest.mark.asyncio
    async def test_recall_filters_dict_builds_temporal_filter(self) -> None:
        eng = _connected()
        eng._embedder.embed = AsyncMock(return_value=[0.0])
        eng._temporal_store.search = AsyncMock(return_value=[])
        await eng.recall("q", uuid4(), filters={"author": {"eq": "alice"}})
        _, kwargs = eng._temporal_store.search.call_args
        tf = kwargs["temporal_filter"]
        assert isinstance(tf, TemporalFilter)
        assert tf.author == "alice"

    @pytest.mark.asyncio
    async def test_recall_temporal_reference_with_filter(self) -> None:
        """When temporal_reference + temporal_filter both provided, _adjust is called."""
        eng = _connected()
        eng._embedder.embed = AsyncMock(return_value=[0.0])
        eng._temporal_store.search = AsyncMock(return_value=[])
        tf = TemporalFilter()
        await eng.recall(
            "q",
            uuid4(),
            temporal_filter=tf,
            temporal_reference=datetime(2024, 1, 1, tzinfo=UTC),
        )
        _, kwargs = eng._temporal_store.search.call_args
        # _adjust_relative_time is a no-op placeholder so we get the same filter
        assert kwargs["temporal_filter"] is tf


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    @pytest.mark.asyncio
    async def test_forget_namespace_mismatch_returns_false(self) -> None:
        eng = _connected()
        wrong_ns = uuid4()
        doc_id = uuid4()
        # Security: storage.get_document now filters by namespace at the SQL
        # layer; a cross-namespace lookup just returns None.
        eng._storage.get_document = AsyncMock(return_value=None)
        result = await eng.forget(doc_id, wrong_ns)
        assert result is False
        eng._storage.get_document.assert_awaited_once_with(doc_id, namespace_id=wrong_ns)
        eng._temporal_store.delete_chunks_by_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_uses_namespace_from_document_when_kwarg_none(self) -> None:
        # Security: previously forget() would look up the document with a
        # `namespace_id IS NULL` probe and then trust the doc's namespace.
        # That is an IDOR vector — anyone with a doc_id could delete any
        # document. Forget now bails immediately when namespace_id is None.
        eng = _connected()
        doc_id = uuid4()
        eng._storage.get_document = AsyncMock()
        eng._storage.delete_document = AsyncMock()
        eng._temporal_store.delete_chunks_by_document = AsyncMock()

        result = await eng.forget(doc_id, None)
        assert result is False
        eng._storage.get_document.assert_not_awaited()
        eng._temporal_store.delete_chunks_by_document.assert_not_awaited()
        eng._storage.delete_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forget_no_namespace_no_document_returns_false(self) -> None:
        eng = _connected()
        eng._storage.get_document = AsyncMock(return_value=None)
        result = await eng.forget(uuid4(), None)
        assert result is False

    @pytest.mark.asyncio
    async def test_forget_with_namespace_calls_temporal_delete(self) -> None:
        eng = _connected()
        ns = uuid4()
        doc_id = uuid4()
        document = MagicMock()
        document.namespace_id = ns
        eng._storage.get_document = AsyncMock(return_value=document)
        eng._storage.delete_document = AsyncMock(return_value=True)
        eng._temporal_store.delete_chunks_by_document = AsyncMock()

        result = await eng.forget(doc_id, ns)
        assert result is True
        eng._temporal_store.delete_chunks_by_document.assert_awaited_once_with(doc_id, ns)
        eng._storage.delete_document.assert_awaited_once_with(doc_id, namespace_id=ns)


# ---------------------------------------------------------------------------
# remember_batch — empty / fully-deduped fast paths
# ---------------------------------------------------------------------------


class TestRememberBatchEarlyReturns:
    @pytest.mark.asyncio
    async def test_empty_documents_returns_zero_result(self) -> None:
        eng = _connected()
        result = await eng.remember_batch([], uuid4(), entity_types=[], relationship_types=[])
        assert result.total == 0
        assert result.processed == 0
        assert result.skipped == 0

    @pytest.mark.asyncio
    async def test_all_duplicates_returns_skipped(self) -> None:
        """When every document checksum is found in storage, batch returns 0 processed."""
        eng = _connected()
        # Storage reports every checksum already exists
        eng._storage.get_documents_by_checksums = AsyncMock(return_value={"any-checksum": MagicMock()})

        # Patch the dedup path: by making the lookup return all checksums, every doc is "skipped"
        # Easier path: provide a fake mapping that contains every doc's checksum.
        docs = [{"content": "a"}, {"content": "b"}]
        import hashlib

        checksums = {hashlib.sha256(d["content"].encode()).hexdigest(): MagicMock() for d in docs}
        eng._storage.get_documents_by_checksums = AsyncMock(return_value=checksums)
        on_progress = MagicMock()

        result = await eng.remember_batch(
            docs,
            uuid4(),
            on_progress=on_progress,
            entity_types=[],
            relationship_types=[],
        )
        assert result.total == 2
        assert result.processed == 0
        assert result.skipped == 2
        # #898: on_progress fires once per (skipped) document with an
        # incrementing count, not a single (total, total) summary call.
        assert [c.args for c in on_progress.call_args_list] == [(1, 2), (2, 2)]
