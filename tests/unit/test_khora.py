"""Unit tests for khora.py — Khora primary API."""

from __future__ import annotations

import warnings
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.khora import BatchHandle, BatchResult, DocumentResult, Khora, RecallResult, RememberResult, Stats

from .helpers import RESOLVE_ROW_ID as _RESOLVE_ROW_ID
from .helpers import make_lake as _make_lake
from .helpers import mock_config as _mock_config
from .helpers import mock_engine as _mock_engine

# ---------------------------------------------------------------------------
# RememberResult / RecallResult dataclass tests
# ---------------------------------------------------------------------------


class TestRememberResult:
    """Tests for RememberResult dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        r = RememberResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            chunks_created=5,
            entities_extracted=3,
            relationships_created=2,
        )
        assert r.chunks_created == 5
        assert r.entities_extracted == 3
        assert r.relationships_created == 2
        assert r.metadata == {}

    def test_custom_metadata(self) -> None:
        """Custom metadata can be set."""
        r = RememberResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            chunks_created=0,
            entities_extracted=0,
            relationships_created=0,
            metadata={"duplicate": True},
        )
        assert r.metadata["duplicate"] is True


class TestRecallResult:
    """Tests for RecallResult dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        from khora.core.models.document import Chunk
        from khora.core.models.entity import Entity

        ns_id = uuid4()
        chunk = Chunk(namespace_id=ns_id, document_id=uuid4(), content="hello")
        entity = Entity(namespace_id=ns_id, name="Alice", entity_type="PERSON")
        r = RecallResult(
            query="test query",
            namespace_id=ns_id,
            chunks=[(chunk, 0.9)],
            entities=[(entity, 0.8)],
            context_text="some text",
        )
        assert r.query == "test query"
        assert r.namespace_id == ns_id
        assert len(r.chunks) == 1
        assert len(r.entities) == 1
        assert r.context_text == "some text"

    def test_default_metadata(self) -> None:
        """Default metadata is empty dict."""
        r = RecallResult(
            query="q",
            namespace_id=uuid4(),
            chunks=[],
            entities=[],
            context_text="",
        )
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# Khora initialization
# ---------------------------------------------------------------------------


class TestKhoraInit:
    """Tests for Khora initialization."""

    def test_init_default(self) -> None:
        """Default init loads config from env."""
        lake = _make_lake()
        assert lake._connected is False
        assert lake._engine is None

    def test_init_with_config(self) -> None:
        """Init with explicit config skips load_config."""
        from khora.config import KhoraConfig

        # Create a real KhoraConfig (not a mock) to trigger the isinstance check
        cfg = KhoraConfig(database_url="postgresql://test")
        lake = Khora(cfg)

        assert lake._config is cfg
        assert lake._config.database_url == "postgresql://test"

    def test_init_with_storage_config(self) -> None:
        """Init with explicit storage_config uses it directly."""
        storage_cfg = MagicMock()
        with patch("khora.khora.load_config", return_value=_mock_config()):
            lake = Khora(storage_config=storage_cfg)
        assert lake._storage_config is storage_cfg

    def test_not_connected_properties_raise(self) -> None:
        """Accessing storage before connect raises."""
        lake = _make_lake()

        with pytest.raises(RuntimeError, match="not connected"):
            _ = lake.storage

    def test_connected_properties_return(self) -> None:
        """Accessing storage after connect succeeds."""
        lake = _make_lake(connected=True)
        assert lake.storage is lake._engine._storage


# ---------------------------------------------------------------------------
# connect / disconnect lifecycle
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    """Tests for connect() and disconnect() lifecycle."""

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        """connect() creates engine and sets flag."""
        lake = _make_lake()

        mock_engine = _mock_engine()

        with patch("khora.engines.create_engine", return_value=mock_engine):
            await lake.connect()

        assert lake._connected is True
        assert lake._engine is mock_engine
        mock_engine.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self) -> None:
        """Calling connect() when already connected is a no-op."""
        lake = _make_lake(connected=True)
        original_engine = lake._engine

        await lake.connect()

        assert lake._engine is original_engine

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """disconnect() tears down all components."""
        lake = _make_lake(connected=True)

        await lake.disconnect()

        assert lake._connected is False
        assert lake._engine is None

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self) -> None:
        """Calling disconnect() when not connected is a no-op."""
        lake = _make_lake()
        await lake.disconnect()  # Should not raise
        assert lake._connected is False

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """async with Khora() connects and disconnects."""
        lake = _make_lake()
        lake.connect = AsyncMock()
        lake.disconnect = AsyncMock()

        async with lake as ctx:
            assert ctx is lake
            lake.connect.assert_awaited_once()

        lake.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# _resolve_namespace
# ---------------------------------------------------------------------------


class TestResolveNamespace:
    """Tests for _resolve_namespace helper.

    _resolve_namespace now performs a DB lookup via storage.resolve_namespace()
    to map a stable namespace_id to the active version's row-level id.
    """

    @pytest.mark.asyncio
    async def test_uuid_calls_resolve(self) -> None:
        """UUID is forwarded to storage.resolve_namespace()."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        row_id = uuid4()
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=row_id)

        result = await lake._resolve_namespace(ns_id)
        assert result == row_id
        lake._engine._storage.resolve_namespace.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_uuid_string_parsed_and_resolved(self) -> None:
        """UUID string is parsed then forwarded to storage.resolve_namespace()."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        row_id = uuid4()
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=row_id)

        result = await lake._resolve_namespace(str(ns_id))
        assert result == row_id
        lake._engine._storage.resolve_namespace.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_invalid_string_raises_value_error(self) -> None:
        """Non-UUID string raises ValueError before DB lookup."""
        lake = _make_lake(connected=True)
        with pytest.raises(ValueError, match="Invalid namespace"):
            await lake._resolve_namespace("not-a-uuid")

    @pytest.mark.asyncio
    async def test_no_active_version_raises(self) -> None:
        """ValueError from storage.resolve_namespace propagates."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine._storage.resolve_namespace = AsyncMock(
            side_effect=ValueError(f"No active namespace version found for namespace_id={ns_id}")
        )

        with pytest.raises(ValueError, match="No active namespace version"):
            await lake._resolve_namespace(ns_id)


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


class TestRemember:
    """Tests for remember()."""

    @pytest.mark.asyncio
    async def test_remember_delegates_to_engine(self) -> None:
        """remember() delegates to engine.remember()."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=3,
            entities_extracted=2,
            relationships_created=1,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember(
                "test content",
                namespace=ns_id,
                title="Test",
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        assert result == mock_result
        assert result.llm_usage == []
        lake._engine.remember.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remember_passes_external_id(self) -> None:
        """remember() passes external_id through to engine.remember()."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember(
                "test content",
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="test-123",
            )

        call_kwargs = lake._engine.remember.call_args.kwargs
        assert call_kwargs["external_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_remember_without_external_id(self) -> None:
        """remember() without external_id passes None (backward compat)."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember(
                "test content",
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        call_kwargs = lake._engine.remember.call_args.kwargs
        assert call_kwargs["external_id"] is None

    @pytest.mark.asyncio
    async def test_remember_passes_special_char_external_id(self) -> None:
        """remember() passes external_id with special characters through unchanged."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        special_id = "org/repo#123 — «test» 'quotes' & unicode: café ñ 日本語"

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember(
                "test content",
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id=special_id,
            )

        call_kwargs = lake._engine.remember.call_args.kwargs
        assert call_kwargs["external_id"] == special_id


class TestExternalIdValidation:
    """Tests for Document-level external_id validation."""

    def test_blank_external_id_rejected(self) -> None:
        """Document rejects blank external_id."""
        from khora.core.models import Document

        with pytest.raises(ValueError, match="non-blank"):
            Document(external_id="")

    def test_whitespace_only_external_id_rejected(self) -> None:
        """Document rejects whitespace-only external_id."""
        from khora.core.models import Document

        with pytest.raises(ValueError, match="non-blank"):
            Document(external_id="   ")

    def test_oversized_external_id_rejected(self) -> None:
        """Document rejects external_id exceeding 512 chars."""
        from khora.core.models import Document

        with pytest.raises(ValueError, match="at most 512"):
            Document(external_id="x" * 513)

    def test_max_length_external_id_accepted(self) -> None:
        """Document accepts external_id at exactly 512 chars."""
        from khora.core.models import Document

        doc = Document(external_id="x" * 512)
        assert doc.external_id == "x" * 512

    def test_none_external_id_accepted(self) -> None:
        """Document accepts None external_id (default)."""
        from khora.core.models import Document

        doc = Document()
        assert doc.external_id is None


# ---------------------------------------------------------------------------
# remember_batch
# ---------------------------------------------------------------------------


class TestRememberBatchExternalId:
    """Tests for external_id pass-through in remember_batch()."""

    @pytest.mark.asyncio
    async def test_remember_batch_passes_external_id_in_doc_dicts(self) -> None:
        """remember_batch() forwards doc dicts containing external_id to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        lake._engine.remember_batch = AsyncMock(
            return_value=BatchResult(
                total=1,
                processed=1,
                skipped=0,
                failed=0,
                chunks=2,
                entities=1,
                relationships=0,
            )
        )

        docs = [{"content": "doc one", "external_id": "ext-abc"}]

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember_batch(
                docs,
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        # Doc dicts are the first positional arg to engine.remember_batch
        call_args = lake._engine.remember_batch.call_args
        assert call_args.args, "remember_batch should receive doc list as positional arg"
        passed_docs = call_args.args[0]
        assert passed_docs[0]["external_id"] == "ext-abc"

    @pytest.mark.asyncio
    async def test_remember_batch_mixed_external_ids(self) -> None:
        """remember_batch() with mixed docs (some with external_id, some without)."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        lake._engine.remember_batch = AsyncMock(
            return_value=BatchResult(
                total=3,
                processed=3,
                skipped=0,
                failed=0,
                chunks=6,
                entities=3,
                relationships=1,
            )
        )

        docs = [
            {"content": "doc one", "external_id": "ext-1"},
            {"content": "doc two"},
            {"content": "doc three", "external_id": "ext-3"},
        ]

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember_batch(
                docs,
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        call_args = lake._engine.remember_batch.call_args
        assert call_args.args, "remember_batch should receive doc list as positional arg"
        passed_docs = call_args.args[0]
        assert passed_docs[0]["external_id"] == "ext-1"
        assert "external_id" not in passed_docs[1]
        assert passed_docs[2]["external_id"] == "ext-3"


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


class TestRecall:
    """Tests for recall()."""

    @pytest.mark.asyncio
    async def test_recall_delegates_to_engine(self) -> None:
        """recall() delegates to engine.recall() and returns result."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="search query",
            namespace_id=ns_id,
            chunks=[("chunk", 0.9)],
            entities=[("entity", 0.8)],
            context_text="found content",
            metadata={"mode": "HYBRID"},
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("search query", namespace=ns_id)

        assert isinstance(result, RecallResult)
        assert result.query == "search query"
        lake._engine.recall.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recall_passes_search_mode(self) -> None:
        """recall() passes mode to engine."""
        from khora.query.engine import SearchMode

        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("test", namespace=ns_id, mode=SearchMode.VECTOR)

        call_kwargs = lake._engine.recall.call_args
        assert call_kwargs.kwargs.get("mode") == SearchMode.VECTOR


# ---------------------------------------------------------------------------
# recall — temporal bounds (start_time / end_time)
# ---------------------------------------------------------------------------


class TestRecallTemporalBounds:
    """Tests for start_time/end_time parameters on recall()."""

    # Shared helper: make a minimal RecallResult mock return value
    @staticmethod
    def _mock_result(ns_id: object) -> RecallResult:
        return RecallResult(
            query="q",
            namespace_id=ns_id,  # type: ignore[arg-type]
            chunks=[],
            entities=[],
            context_text="",
        )

    @pytest.mark.asyncio
    async def test_start_time_only_constructs_filter(self) -> None:
        """start_time only → SkeletonTemporalFilter with occurred_after set, occurred_before None."""
        from datetime import UTC, datetime

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        start = datetime(2024, 1, 1, tzinfo=UTC)

        lake._engine.recall = AsyncMock(return_value=self._mock_result(ns_id))

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("q", namespace=ns_id, start_time=start)

        call_kwargs = lake._engine.recall.call_args
        temporal_filter = call_kwargs.kwargs.get("temporal_filter")
        assert temporal_filter is not None
        assert temporal_filter.occurred_after == start
        assert temporal_filter.occurred_before is None

    @pytest.mark.asyncio
    async def test_end_time_only_constructs_filter(self) -> None:
        """end_time only → SkeletonTemporalFilter with occurred_before set, occurred_after None."""
        from datetime import UTC, datetime

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        end = datetime(2024, 12, 31, tzinfo=UTC)

        lake._engine.recall = AsyncMock(return_value=self._mock_result(ns_id))

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("q", namespace=ns_id, end_time=end)

        call_kwargs = lake._engine.recall.call_args
        temporal_filter = call_kwargs.kwargs.get("temporal_filter")
        assert temporal_filter is not None
        assert temporal_filter.occurred_before == end
        assert temporal_filter.occurred_after is None

    @pytest.mark.asyncio
    async def test_both_bounds_valid(self) -> None:
        """Both bounds provided (start < end) → filter constructed correctly."""
        from datetime import UTC, datetime

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)

        lake._engine.recall = AsyncMock(return_value=self._mock_result(ns_id))

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("q", namespace=ns_id, start_time=start, end_time=end)

        call_kwargs = lake._engine.recall.call_args
        temporal_filter = call_kwargs.kwargs.get("temporal_filter")
        assert temporal_filter is not None
        assert temporal_filter.occurred_after == start
        assert temporal_filter.occurred_before == end

    @pytest.mark.asyncio
    async def test_no_bounds_passes_none_filter(self) -> None:
        """Neither bound → temporal_filter=None passed to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        lake._engine.recall = AsyncMock(return_value=self._mock_result(ns_id))

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("q", namespace=ns_id)

        call_kwargs = lake._engine.recall.call_args
        temporal_filter = call_kwargs.kwargs.get("temporal_filter")
        assert temporal_filter is None

    @pytest.mark.asyncio
    async def test_start_after_end_raises_valueerror(self) -> None:
        """start_time > end_time → ValueError before engine is called."""
        from datetime import UTC, datetime

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        start = datetime(2024, 12, 31, tzinfo=UTC)
        end = datetime(2024, 1, 1, tzinfo=UTC)

        with pytest.raises(ValueError, match="start_time must be <= end_time"):
            await lake.recall("q", namespace=ns_id, start_time=start, end_time=end)

        lake._engine.recall.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mixed_timezone_raises_valueerror(self) -> None:
        """naive start_time with aware end_time → ValueError."""
        from datetime import UTC, datetime

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        start = datetime(2024, 1, 1)  # naive
        end = datetime(2024, 12, 31, tzinfo=UTC)  # aware

        with pytest.raises(ValueError, match="timezone-aware or both naive"):
            await lake.recall("q", namespace=ns_id, start_time=start, end_time=end)

        lake._engine.recall.assert_not_awaited()


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    """Tests for forget()."""

    @pytest.mark.asyncio
    async def test_forget_delegates_to_engine(self) -> None:
        """forget() delegates to engine.forget() with resolved namespace."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()
        ns_id = uuid4()

        lake._engine.forget = AsyncMock(return_value=True)

        result = await lake.forget(doc_id, namespace=ns_id)
        assert result is True
        lake._engine.forget.assert_awaited_once_with(doc_id, _RESOLVE_ROW_ID)


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------


class TestEntityOperations:
    """Tests for entity CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_entity(self) -> None:
        """get_entity delegates to engine."""
        lake = _make_lake(connected=True)
        entity_id = uuid4()
        mock_entity = MagicMock()

        lake._engine.get_entity = AsyncMock(return_value=mock_entity)

        result = await lake.get_entity(entity_id)
        assert result is mock_entity
        lake._engine.get_entity.assert_awaited_once_with(entity_id)

    @pytest.mark.asyncio
    async def test_list_entities(self) -> None:
        """list_entities delegates to engine with resolved namespace."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_entities = [MagicMock(), MagicMock()]
        lake._engine.list_entities = AsyncMock(return_value=mock_entities)

        result = await lake.list_entities(namespace=ns_id, entity_type="PERSON", limit=50)
        assert result == mock_entities
        lake._engine.list_entities.assert_awaited_once_with(_RESOLVE_ROW_ID, entity_type="PERSON", limit=50)

    @pytest.mark.asyncio
    async def test_find_related_entities(self) -> None:
        """find_related_entities delegates to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        entity_id = uuid4()

        mock_related = [(MagicMock(), 0.8)]
        lake._engine.find_related_entities = AsyncMock(return_value=mock_related)

        result = await lake.find_related_entities(entity_id, namespace=ns_id, max_depth=3)
        assert result == mock_related


# ---------------------------------------------------------------------------
# Namespace management
# ---------------------------------------------------------------------------


class TestNamespaceManagement:
    """Tests for namespace operations."""

    @pytest.mark.asyncio
    async def test_create_namespace(self) -> None:
        """create_namespace delegates to engine."""
        lake = _make_lake(connected=True)

        mock_ns = MagicMock()
        lake._engine.create_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.create_namespace()
        assert result is mock_ns
        lake._engine.create_namespace.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_namespace(self) -> None:
        """get_namespace delegates to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        mock_ns = MagicMock()

        lake._engine.get_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.get_namespace(ns_id)
        assert result is mock_ns

    @pytest.mark.asyncio
    async def test_get_namespace_by_stable_id(self) -> None:
        """get_namespace_by_stable_id resolves stable id then delegates to engine."""
        lake = _make_lake(connected=True)
        stable_id = uuid4()
        mock_ns = MagicMock()

        lake._engine.get_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.get_namespace_by_stable_id(stable_id)
        assert result is mock_ns
        # Should have resolved the stable id first
        lake._engine._storage.resolve_namespace.assert_awaited_once_with(stable_id)
        # Should pass the resolved row-level id to get_namespace
        lake._engine.get_namespace.assert_awaited_once_with(_RESOLVE_ROW_ID)

    @pytest.mark.asyncio
    async def test_get_namespace_by_stable_id_not_found(self) -> None:
        """get_namespace_by_stable_id raises ValueError when no active version exists."""
        lake = _make_lake(connected=True)
        stable_id = uuid4()
        lake._engine._storage.resolve_namespace = AsyncMock(
            side_effect=ValueError(f"No active namespace version found for namespace_id={stable_id}")
        )

        with pytest.raises(ValueError, match="No active namespace version"):
            await lake.get_namespace_by_stable_id(stable_id)

    @pytest.mark.asyncio
    async def test_get_namespace_by_stable_id_resolved_but_none(self) -> None:
        """get_namespace_by_stable_id returns None when resolved namespace not in engine."""
        lake = _make_lake(connected=True)
        stable_id = uuid4()

        lake._engine.get_namespace = AsyncMock(return_value=None)

        result = await lake.get_namespace_by_stable_id(stable_id)
        assert result is None
        lake._engine._storage.resolve_namespace.assert_awaited_once_with(stable_id)
        lake._engine.get_namespace.assert_awaited_once_with(_RESOLVE_ROW_ID)

    @pytest.mark.asyncio
    async def test_create_namespace_returns_namespace_id(self) -> None:
        """create_namespace returns object with distinct namespace_id."""
        from khora.core.models.tenancy import MemoryNamespace

        lake = _make_lake(connected=True)
        row_id = uuid4()
        stable_id = uuid4()
        mock_ns = MemoryNamespace(id=row_id, namespace_id=stable_id)
        lake._engine.create_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.create_namespace()
        assert result.namespace_id == stable_id
        assert result.id == row_id
        assert result.namespace_id != result.id  # namespace_id is independently generated


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health_check."""

    @pytest.mark.asyncio
    async def test_disconnected(self) -> None:
        """Health check when disconnected."""
        lake = _make_lake()
        result = await lake.health_check()
        assert result["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_healthy(self) -> None:
        """Health check delegates to engine."""
        lake = _make_lake(connected=True)
        lake._engine.health_check = AsyncMock(
            return_value={
                "status": "healthy",
                "storage": {"relational": True, "vector": True},
            }
        )

        result = await lake.health_check()
        assert result["status"] == "healthy"


# ---------------------------------------------------------------------------
# New API: Simplified Constructor
# ---------------------------------------------------------------------------


class TestSimplifiedConstructor:
    """Tests for the simplified Khora constructor."""

    def test_init_with_database_url_string(self) -> None:
        """Init with database URL string creates config."""
        with patch("khora.khora.load_config") as mock_load:
            lake = Khora("postgresql://localhost/mydb")
            mock_load.assert_not_called()

        assert lake._config.database_url == "postgresql://localhost/mydb"

    def test_init_with_database_url_and_graph_url(self) -> None:
        """Init with both database and graph URLs."""
        with patch("khora.khora.load_config"):
            lake = Khora(
                "postgresql://localhost/mydb",
                graph_url="bolt://localhost:7687",
            )

        assert lake._config.database_url == "postgresql://localhost/mydb"
        assert lake._config.neo4j_url == "bolt://localhost:7687"

    def test_init_with_custom_embedding_model(self) -> None:
        """Init with custom embedding model."""
        with patch("khora.khora.load_config"):
            lake = Khora(
                "postgresql://localhost/mydb",
                embedding_model="text-embedding-3-large",
            )

        assert lake._config.llm.embedding_model == "text-embedding-3-large"

    def test_init_with_khora_config(self) -> None:
        """Init with full KhoraConfig object."""
        from khora.config import KhoraConfig

        # Create a real KhoraConfig (not a mock) to trigger the isinstance check
        cfg = KhoraConfig(database_url="postgresql://test")
        lake = Khora(cfg)

        assert lake._config is cfg
        assert lake._config.database_url == "postgresql://test"

    def test_init_with_none_loads_from_env(self) -> None:
        """Init with None loads config from env/file."""
        with patch("khora.khora.load_config", return_value=_mock_config()) as mock_load:
            lake = Khora()
            mock_load.assert_called_once()

        assert lake._config is not None

    def test_init_none_with_graph_override(self) -> None:
        """Init with None but graph_url override."""
        mock_cfg = _mock_config()
        mock_cfg.neo4j_url = None
        with patch("khora.khora.load_config", return_value=mock_cfg):
            lake = Khora(graph_url="bolt://custom:7687")

        assert lake._config.neo4j_url == "bolt://custom:7687"

    def test_init_with_engine_parameter(self) -> None:
        """Init with explicit engine parameter."""
        with patch("khora.khora.load_config", return_value=_mock_config()):
            lake = Khora(engine="chronicle")

        assert lake._engine_name == "chronicle"


# ---------------------------------------------------------------------------
# New API: BatchResult and Stats dataclasses
# ---------------------------------------------------------------------------


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        r = BatchResult(
            total=10,
            processed=8,
            skipped=1,
            failed=1,
            chunks=50,
            entities=20,
            relationships=15,
        )
        assert r.total == 10
        assert r.processed == 8
        assert r.skipped == 1
        assert r.failed == 1
        assert r.chunks == 50
        assert r.entities == 20
        assert r.relationships == 15


class TestStats:
    """Tests for Stats dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        s = Stats(
            documents=100,
            chunks=500,
            entities=200,
            relationships=150,
        )
        assert s.documents == 100
        assert s.chunks == 500
        assert s.entities == 200
        assert s.relationships == 150

    def test_last_activity_at_default_none(self) -> None:
        """last_activity_at defaults to None for backward compatibility."""
        s = Stats(documents=1, chunks=2, entities=3, relationships=4)
        assert s.last_activity_at is None

    def test_last_activity_at_with_value(self) -> None:
        """last_activity_at accepts a datetime value."""
        from datetime import UTC, datetime

        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        s = Stats(
            documents=1,
            chunks=2,
            entities=3,
            relationships=4,
            last_activity_at=ts,
        )
        assert s.last_activity_at == ts

    def test_frozen(self) -> None:
        """Stats is immutable."""
        s = Stats(documents=1, chunks=2, entities=3, relationships=4)
        with pytest.raises(AttributeError):
            s.last_activity_at = datetime.now()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# New API: Storage Property (stable API)
# ---------------------------------------------------------------------------


class TestStorageProperty:
    """Tests for the storage property (promoted to stable API)."""

    def test_storage_no_deprecation_warning(self) -> None:
        """Accessing storage property does NOT emit DeprecationWarning."""
        lake = _make_lake(connected=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = lake.storage
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0

    def test_storage_returns_coordinator(self) -> None:
        """storage property returns the engine's storage coordinator."""
        lake = _make_lake(connected=True)
        assert lake.storage is lake._engine._storage


# ---------------------------------------------------------------------------
# New API: Raw flag in recall
# ---------------------------------------------------------------------------


class TestRecallRawMode:
    """Tests for raw mode in recall()."""

    @pytest.mark.asyncio
    async def test_raw_mode_passed_to_engine(self) -> None:
        """raw=True is passed to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("test query", namespace=ns_id, raw=True)

        call_kwargs = lake._engine.recall.call_args
        assert call_kwargs.kwargs.get("raw") is True


# ---------------------------------------------------------------------------
# New API: Convenience methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    """Tests for convenience methods (get_document, list_documents, etc.)."""

    @pytest.mark.asyncio
    async def test_get_document(self) -> None:
        """get_document delegates to engine."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()
        mock_doc = MagicMock()

        lake._engine.get_document = AsyncMock(return_value=mock_doc)

        result = await lake.get_document(doc_id)
        assert result is mock_doc
        lake._engine.get_document.assert_awaited_once_with(doc_id)

    @pytest.mark.asyncio
    async def test_list_documents(self) -> None:
        """list_documents delegates to engine with resolved namespace."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_docs = [MagicMock(), MagicMock()]
        lake._engine.list_documents = AsyncMock(return_value=mock_docs)

        result = await lake.list_documents(namespace=ns_id, limit=50)
        assert result == mock_docs
        lake._engine.list_documents.assert_awaited_once_with(_RESOLVE_ROW_ID, limit=50)

    @pytest.mark.asyncio
    async def test_search_entities(self) -> None:
        """search_entities delegates to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_entities = [MagicMock()]
        lake._engine.search_entities = AsyncMock(return_value=mock_entities)

        result = await lake.search_entities("test query", namespace=ns_id, limit=5)

        assert len(result) == 1
        lake._engine.search_entities.assert_awaited_once()


# ---------------------------------------------------------------------------
# New API: Enhanced remember_batch
# ---------------------------------------------------------------------------


class TestEnhancedRememberBatch:
    """Tests for enhanced remember_batch() with BatchResult."""

    @pytest.mark.asyncio
    async def test_empty_batch_returns_batch_result(self) -> None:
        """Empty batch returns BatchResult with zeros."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.remember_batch = AsyncMock(
            return_value=BatchResult(
                total=0,
                processed=0,
                skipped=0,
                failed=0,
                chunks=0,
                entities=0,
                relationships=0,
            )
        )

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember_batch(
                [],
                namespace=ns_id,
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        assert isinstance(result, BatchResult)
        assert result.total == 0
        assert result.processed == 0

    @pytest.mark.asyncio
    async def test_batch_returns_batch_result(self) -> None:
        """remember_batch() returns BatchResult with aggregated stats."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.remember_batch = AsyncMock(
            return_value=BatchResult(
                total=3,
                processed=2,
                skipped=1,
                failed=0,
                chunks=10,
                entities=5,
                relationships=5,
            )
        )

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember_batch(
                [
                    {"content": "Doc 1"},
                    {"content": "Doc 2"},
                    {"content": "Doc 3"},
                ],
                namespace=ns_id,
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert result.processed == 2
        assert result.skipped == 1
        assert result.relationships == 5


# ---------------------------------------------------------------------------
# Engine Registry Tests
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    """Tests for engine registry functions."""

    def test_list_engines(self) -> None:
        """list_engines returns available engines."""
        from khora.engines import list_engines

        engines = list_engines()
        assert "vectorcypher" in engines
        assert "chronicle" in engines
        assert "skeleton" in engines
        assert "graphrag" not in engines

    def test_register_engine(self) -> None:
        """register_engine adds new engine to registry."""
        from khora.engines import list_engines, register_engine

        register_engine("test_engine", "test.module", "TestEngine")
        engines = list_engines()
        assert "test_engine" in engines

    def test_create_engine_unknown_raises(self) -> None:
        """create_engine raises for unknown engine."""
        from khora.engines import create_engine

        with pytest.raises(ValueError, match="Unknown engine"):
            create_engine("nonexistent", _mock_config())


# ---------------------------------------------------------------------------
# include_sources feature (DYT-506)
# ---------------------------------------------------------------------------


class TestIncludeSources:
    """Tests for include_sources parameter on read methods."""

    @pytest.mark.asyncio
    async def test_recall_include_sources_false(self) -> None:
        """Default include_sources=False does not call get_document_sources_batch."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id)

        assert isinstance(result, RecallResult)
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recall_include_sources_true(self) -> None:
        """include_sources=True populates source_document on chunks and source_documents on entities."""
        from khora.core.models.document import Chunk, DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id_1 = uuid4()
        doc_id_2 = uuid4()

        chunk = Chunk(namespace_id=ns_id, document_id=doc_id_1, content="hello")
        entity = Entity(
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            source_document_ids=[doc_id_1, doc_id_2],
        )

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[(chunk, 0.9)],
            entities=[(entity, 0.8)],
            context_text="hello",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        src_1 = DocumentSource(id=doc_id_1, title="Doc 1")
        src_2 = DocumentSource(id=doc_id_2, title="Doc 2")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id_1: src_1, doc_id_2: src_2})

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id, include_sources=True)

        # Chunk should have source_document populated
        assert result.chunks[0][0].source_document is src_1

        # Entity should have source_documents populated
        assert result.entities[0][0].source_documents == {doc_id_1: src_1, doc_id_2: src_2}

        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_entities_include_sources(self) -> None:
        """list_entities with include_sources=True populates source_documents on entities."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Bob",
            entity_type="PERSON",
            source_document_ids=[doc_id],
        )
        lake._engine.list_entities = AsyncMock(return_value=[entity])

        src = DocumentSource(id=doc_id, title="Source Doc")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        result = await lake.list_entities(namespace=ns_id, include_sources=True)

        assert len(result) == 1
        assert result[0].source_documents == {doc_id: src}
        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_entities_include_sources(self) -> None:
        """search_entities with include_sources=True populates source_documents."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Acme Corp",
            entity_type="ORGANIZATION",
            source_document_ids=[doc_id],
        )
        lake._engine.search_entities = AsyncMock(return_value=[entity])

        src = DocumentSource(id=doc_id, title="Report")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        result = await lake.search_entities("acme", namespace=ns_id, include_sources=True)

        assert len(result) == 1
        assert result[0].source_documents == {doc_id: src}
        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_find_related_entities_include_sources(self) -> None:
        """find_related_entities with include_sources=True populates source_documents."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        entity_id = uuid4()
        doc_id = uuid4()

        related = Entity(
            namespace_id=ns_id,
            name="Related Entity",
            entity_type="CONCEPT",
            source_document_ids=[doc_id],
        )
        lake._engine.find_related_entities = AsyncMock(return_value=[(related, 0.75)])

        src = DocumentSource(id=doc_id, title="Origin")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        result = await lake.find_related_entities(entity_id, namespace=ns_id, include_sources=True)

        assert len(result) == 1
        assert result[0][0].source_documents == {doc_id: src}
        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_include_sources_empty_results(self) -> None:
        """Empty chunks/entities with include_sources=True does not crash or fetch."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="nothing",
            namespace_id=ns_id,
            chunks=[],
            entities=[],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("nothing", namespace=ns_id, include_sources=True)

        assert result.chunks == []
        assert result.entities == []
        # No doc IDs to fetch, so get_document_sources_batch should not be called
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_entity_include_sources(self) -> None:
        """get_entity with include_sources=True populates source_documents."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            source_document_ids=[doc_id],
        )
        lake._engine.get_entity = AsyncMock(return_value=entity)

        src = DocumentSource(id=doc_id, title="Source Doc")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        result = await lake.get_entity(entity.id, include_sources=True)

        assert result is not None
        assert result.source_documents == {doc_id: src}
        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_entity_include_sources_false(self) -> None:
        """Default include_sources=False does not call get_document_sources_batch."""
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Bob",
            entity_type="PERSON",
        )
        lake._engine.get_entity = AsyncMock(return_value=entity)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        result = await lake.get_entity(entity.id)

        assert result is not None
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_entity_include_sources_not_found(self) -> None:
        """get_entity returns None when entity not found, even with include_sources=True."""
        lake = _make_lake(connected=True)
        lake._engine.get_entity = AsyncMock(return_value=None)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        result = await lake.get_entity(uuid4(), include_sources=True)

        assert result is None
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deleted_document_skipped_on_entities(self) -> None:
        """Entity with partially-deleted source docs only gets found sources."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id_1 = uuid4()
        doc_id_2 = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            source_document_ids=[doc_id_1, doc_id_2],
        )

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[(entity, 0.8)],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        # Only doc_id_1 is returned; doc_id_2 was deleted
        src_1 = DocumentSource(id=doc_id_1, title="Doc 1")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id_1: src_1})

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id, include_sources=True)

        assert result.entities[0][0].source_documents == {doc_id_1: src_1}
        assert doc_id_2 not in result.entities[0][0].source_documents

    @pytest.mark.asyncio
    async def test_chunk_with_missing_document(self) -> None:
        """Chunk whose document_id is not in sources gets source_document=None."""
        from khora.core.models.document import Chunk

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(namespace_id=ns_id, document_id=doc_id, content="orphan chunk")

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[(chunk, 0.9)],
            entities=[],
            context_text="orphan chunk",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        # get_document_sources_batch returns empty dict (document was deleted)
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={})

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id, include_sources=True)

        assert result.chunks[0][0].source_document is None

    @pytest.mark.asyncio
    async def test_storage_exception_propagation(self) -> None:
        """RuntimeError from get_document_sources_batch propagates to caller."""
        from khora.core.models.document import Chunk

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(namespace_id=ns_id, document_id=doc_id, content="test")

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[(chunk, 0.9)],
            entities=[],
            context_text="test",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)
        lake._engine._storage.get_document_sources_batch = AsyncMock(side_effect=RuntimeError("DB error"))

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            with pytest.raises(RuntimeError, match="DB error"):
                await lake.recall("test", namespace=ns_id, include_sources=True)

    @pytest.mark.asyncio
    async def test_entity_empty_source_document_ids(self) -> None:
        """Entity with empty source_document_ids skips fetch and gets source_documents=None."""
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Lonely",
            entity_type="CONCEPT",
            source_document_ids=[],
        )

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[(entity, 0.7)],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id, include_sources=True)

        # No doc IDs to fetch, so get_document_sources_batch should NOT be called
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()
        assert result.entities[0][0].source_documents is None


# ---------------------------------------------------------------------------
# submit_batch
# ---------------------------------------------------------------------------


def _make_staged_doc(ns_id):
    """Build a minimal mock Document as returned by storage.create_document."""
    from khora.core.models.document import Document

    doc = Document(namespace_id=ns_id, content="test content")
    return doc


def _make_lake_with_staged_support(ns_id):
    """Make a lake whose engine exposes process_staged_document, with processor started."""
    lake = _make_lake(connected=True)
    lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

    async def _fake_create_document(doc):
        return doc

    lake._engine._storage.create_document = AsyncMock(side_effect=_fake_create_document)
    lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)
    # No pre-existing docs by default — each test can override as needed.
    lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})

    async def _fake_process_staged(doc, **kwargs):
        return (2, 1, 0)  # chunks, entities, rels

    lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process_staged)
    lake.start_pending_processor()
    return lake


class TestBatchHandleDataclass:
    """Tests for BatchHandle dataclass."""

    def test_initial_state(self) -> None:
        from uuid import uuid4

        handle = BatchHandle(batch_id=uuid4(), total=5)
        assert handle.total == 5
        assert handle.completed == 0
        assert handle.failed == 0
        assert not handle.is_done

    def test_record_result_increments_completed(self) -> None:
        handle = BatchHandle(batch_id=uuid4(), total=2)
        r = DocumentResult(document_id=uuid4(), namespace_id=uuid4(), success=True)
        handle._record_result(r)
        assert handle.completed == 1
        assert handle.failed == 0

    def test_record_result_tracks_failures(self) -> None:
        handle = BatchHandle(batch_id=uuid4(), total=2)
        r = DocumentResult(document_id=uuid4(), namespace_id=uuid4(), success=False, error="oops")
        handle._record_result(r)
        assert handle.completed == 1
        assert handle.failed == 1

    def test_mark_done_sets_is_done(self) -> None:
        handle = BatchHandle(batch_id=uuid4(), total=1)
        assert not handle.is_done
        handle._mark_done()
        assert handle.is_done

    @pytest.mark.asyncio
    async def test_wait_returns_when_done(self) -> None:
        import asyncio

        handle = BatchHandle(batch_id=uuid4(), total=1)

        async def _setter():
            await asyncio.sleep(0)
            handle._mark_done()

        asyncio.create_task(_setter())
        await handle.wait()
        assert handle.is_done


class TestDocumentResultDataclass:
    """Tests for DocumentResult dataclass."""

    def test_success_fields(self) -> None:
        r = DocumentResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            success=True,
            chunks_created=3,
            entities_extracted=2,
            relationships_created=1,
        )
        assert r.success is True
        assert r.error is None
        assert r.chunks_created == 3
        assert r.external_id is None

    def test_failure_fields(self) -> None:
        r = DocumentResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            success=False,
            error="embedding failed",
        )
        assert r.success is False
        assert r.error == "embedding failed"
        assert r.chunks_created == 0
        assert r.external_id is None

    def test_external_id_field(self) -> None:
        r = DocumentResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            success=True,
            external_id="ext-abc",
        )
        assert r.external_id == "ext-abc"


class TestSubmitBatch:
    """Tests for Khora.submit_batch()."""

    @pytest.mark.asyncio
    async def test_empty_documents_returns_done_handle(self) -> None:
        """submit_batch with empty list returns a done handle immediately."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        handle = await lake.submit_batch(
            [],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        assert handle.total == 0
        assert handle.is_done

    @pytest.mark.asyncio
    async def test_returns_handle_before_processing(self) -> None:
        """submit_batch returns handle immediately; create_document called before return."""
        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        docs = [{"content": "hello"}]
        called_before_return = []

        orig_create = lake._engine._storage.create_document.side_effect

        async def _spy_create(doc):
            called_before_return.append(True)
            return await orig_create(doc)

        lake._engine._storage.create_document.side_effect = _spy_create

        handle = await lake.submit_batch(
            docs,
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        # create_document was called synchronously before handle was returned
        assert called_before_return, "storage.create_document must be called before submit_batch returns"
        assert handle.total == 1

    @pytest.mark.asyncio
    async def test_submit_batch_raises_when_processor_not_started(self) -> None:
        """submit_batch raises RuntimeError when pending docs cannot be processed."""
        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        async def _fake_create(doc):
            return doc

        lake._engine._storage.create_document = AsyncMock(side_effect=_fake_create)
        lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})
        lake._engine.process_staged_document = AsyncMock(return_value=(2, 1, 0))

        assert lake._processor_task is None

        with pytest.raises(RuntimeError, match="pending processor is not running"):
            await lake.submit_batch(
                [{"content": "doc without processor"}],
                on_result=lambda c, t, r: None,
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

    @pytest.mark.asyncio
    async def test_on_result_fires_per_document(self) -> None:
        """on_result callback fires once per document with correct args."""

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        docs = [{"content": f"doc {i}"} for i in range(3)]
        results: list[DocumentResult] = []
        calls: list[tuple[int, int]] = []

        def _on_result(completed, total, doc_result):
            results.append(doc_result)
            calls.append((completed, total))

        handle = await lake.submit_batch(
            docs,
            on_result=_on_result,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert len(results) == 3
        assert all(r.success for r in results)
        assert calls[-1] == (3, 3)

    @pytest.mark.asyncio
    async def test_handle_is_done_after_wait(self) -> None:
        """BatchHandle.is_done is True after wait() returns."""

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        handle = await lake.submit_batch(
            [{"content": "x"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert handle.is_done
        assert handle.completed == 1

    @pytest.mark.asyncio
    async def test_failed_document_fires_on_result_with_error(self) -> None:
        """If process_staged_document raises, on_result receives success=False."""

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        async def _fake_create(doc):
            return doc

        lake._engine._storage.create_document = AsyncMock(side_effect=_fake_create)
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)

        async def _failing_process(doc, **kwargs):
            raise RuntimeError("embedding service unavailable")

        lake._engine.process_staged_document = AsyncMock(side_effect=_failing_process)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "will fail"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert len(results) == 1
        assert results[0].success is False
        assert "embedding service unavailable" in results[0].error
        assert handle.failed == 1

    @pytest.mark.asyncio
    async def test_engine_without_process_staged_fires_error(self) -> None:
        """Engine lacking process_staged_document fires error result for each doc."""

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        async def _fake_create(doc):
            return doc

        lake._engine._storage.create_document = AsyncMock(side_effect=_fake_create)
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)
        # Engine has no process_staged_document attribute
        if hasattr(lake._engine, "process_staged_document"):
            del lake._engine.process_staged_document
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "doc"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert len(results) == 1
        assert results[0].success is False
        assert handle.is_done

    @pytest.mark.asyncio
    async def test_multiple_concurrent_batches_dont_interfere(self) -> None:
        """Two concurrent submit_batch calls produce independent handles."""
        import asyncio

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        results_a: list[DocumentResult] = []
        results_b: list[DocumentResult] = []

        handle_a = await lake.submit_batch(
            [{"content": "a1"}, {"content": "a2"}],
            on_result=lambda c, t, r: results_a.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        handle_b = await lake.submit_batch(
            [{"content": "b1"}],
            on_result=lambda c, t, r: results_b.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        await asyncio.gather(handle_a.wait(), handle_b.wait())

        assert handle_a.total == 2
        assert handle_b.total == 1
        assert len(results_a) == 2
        assert len(results_b) == 1
        assert handle_a.batch_id != handle_b.batch_id

    @pytest.mark.asyncio
    async def test_document_result_carries_stats(self) -> None:
        """DocumentResult contains chunks/entities/rels from process_staged_document."""

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        async def _fake_create(doc):
            return doc

        lake._engine._storage.create_document = AsyncMock(side_effect=_fake_create)

        async def _process_with_stats(doc, **kwargs):
            return (5, 3, 2)  # chunks, entities, rels

        lake._engine.process_staged_document = AsyncMock(side_effect=_process_with_stats)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "rich doc"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert results[0].chunks_created == 5
        assert results[0].entities_extracted == 3
        assert results[0].relationships_created == 2

    @pytest.mark.asyncio
    async def test_document_result_carries_llm_usage(self) -> None:
        """DocumentResult.llm_usage is populated from usage recorded during processing."""
        from khora.khora import LLMUsage
        from khora.telemetry.context import record_usage

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)
        lake._engine._storage.create_document = AsyncMock(side_effect=lambda doc: doc)

        async def _process_with_usage(doc, **kwargs):
            record_usage(
                LLMUsage(
                    operation="embedding",
                    model="text-embedding-3-small",
                    prompt_tokens=10,
                    completion_tokens=0,
                    total_tokens=10,
                    latency_ms=5.0,
                )
            )
            record_usage(
                LLMUsage(
                    operation="entity_extraction",
                    model="gpt-4o",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    latency_ms=200.0,
                )
            )
            return (2, 1, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_process_with_usage)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "doc with llm calls"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert results[0].success is True
        assert len(results[0].llm_usage) == 2
        ops = {u.operation for u in results[0].llm_usage}
        assert ops == {"embedding", "entity_extraction"}

    @pytest.mark.asyncio
    async def test_failed_document_result_carries_partial_llm_usage(self) -> None:
        """Failed DocumentResult includes any LLM usage recorded before the exception."""
        from khora.khora import LLMUsage
        from khora.telemetry.context import record_usage

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)
        lake._engine._storage.create_document = AsyncMock(side_effect=lambda doc: doc)
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)

        async def _process_with_usage_then_fail(doc, **kwargs):
            record_usage(
                LLMUsage(
                    operation="embedding",
                    model="text-embedding-3-small",
                    prompt_tokens=10,
                    completion_tokens=0,
                    total_tokens=10,
                    latency_ms=5.0,
                )
            )
            raise RuntimeError("graph write failed")

        lake._engine.process_staged_document = AsyncMock(side_effect=_process_with_usage_then_fail)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "will fail after llm call"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert len(results) == 1
        assert results[0].success is False
        assert len(results[0].llm_usage) == 1
        assert results[0].llm_usage[0].operation == "embedding"

    @pytest.mark.asyncio
    async def test_concurrent_documents_llm_usage_isolation(self) -> None:
        """Each document's llm_usage contains only its own recorded entries."""
        from khora.khora import LLMUsage
        from khora.telemetry.context import record_usage

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)
        lake._engine._storage.create_document = AsyncMock(side_effect=lambda doc: doc)

        call_count = 0

        async def _process_with_distinct_usage(doc, **kwargs):
            nonlocal call_count
            call_count += 1
            op = f"embedding_{call_count}"
            record_usage(
                LLMUsage(
                    operation=op,
                    model="text-embedding-3-small",
                    prompt_tokens=call_count * 10,
                    completion_tokens=0,
                    total_tokens=call_count * 10,
                    latency_ms=float(call_count),
                )
            )
            return (1, 0, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_process_with_distinct_usage)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "doc A"}, {"content": "doc B"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_concurrent=2,
        )
        await handle.wait()

        assert len(results) == 2
        # Each result must have exactly one usage entry, not two
        for r in results:
            assert len(r.llm_usage) == 1
        # The two results must have distinct operation names (no cross-contamination)
        ops = {r.llm_usage[0].operation for r in results}
        assert len(ops) == 2

    @pytest.mark.asyncio
    async def test_failed_document_updates_storage_status(self) -> None:
        """When process_staged_document raises, the document is marked FAILED in storage."""
        from khora.core.models.document import DocumentStatus

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        async def _fake_create(doc):
            return doc

        updated_docs = []

        async def _fake_update(doc):
            updated_docs.append(doc)
            return doc

        lake._engine._storage.create_document = AsyncMock(side_effect=_fake_create)
        lake._engine._storage.update_document = AsyncMock(side_effect=_fake_update)

        async def _failing_process(doc, **kwargs):
            raise RuntimeError("extraction failed")

        lake._engine.process_staged_document = AsyncMock(side_effect=_failing_process)
        lake.start_pending_processor()

        handle = await lake.submit_batch(
            [{"content": "will fail"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert len(updated_docs) == 1
        assert updated_docs[0].status == DocumentStatus.FAILED

    @pytest.mark.asyncio
    async def test_on_result_exception_does_not_hang(self) -> None:
        """If on_result raises, handle.wait() still completes."""
        import asyncio

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        def _bad_callback(completed, total, result):
            raise ValueError("callback exploded")

        handle = await lake.submit_batch(
            [{"content": "doc"}],
            on_result=_bad_callback,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        # Must complete within 5 seconds; if deadlocked, this raises TimeoutError.
        await asyncio.wait_for(handle.wait(), timeout=5.0)
        assert handle.is_done

    @pytest.mark.asyncio
    async def test_create_document_error_fallback_produces_error_result(self) -> None:
        """If get_documents_by_external_ids finds nothing and create_document raises, on_result receives success=False.

        This covers the race-condition path: external_id not found in lookup,
        then a concurrent insert causes the create to fail.
        """
        from sqlalchemy.exc import IntegrityError

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)
        # No existing doc found by external_id lookup
        lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})

        def _raise_integrity(doc):
            raise IntegrityError("INSERT", {}, Exception("unique constraint"))

        lake._engine._storage.create_document = AsyncMock(side_effect=_raise_integrity)

        async def _fake_process(doc, **kwargs):
            return (1, 0, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "duplicate", "external_id": "ext-123"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert len(results) == 1
        assert results[0].success is False
        assert handle.failed == 1

    @pytest.mark.asyncio
    async def test_pending_external_id_requeues_for_processing(self) -> None:
        """PENDING document with same external_id is re-queued, not failed (DYT-3075)."""
        from khora.core.models.document import Document, DocumentStatus

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        existing_doc = Document(namespace_id=ns_id, content="old content", external_id="ext-pending")
        existing_doc.status = DocumentStatus.PENDING

        lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={"ext-pending": existing_doc})
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)

        async def _fake_process(doc, **kwargs):
            return (3, 2, 1)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "new content", "external_id": "ext-pending", "source": "updated-source"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        # create_document must NOT be called — we reuse the existing doc
        lake._engine._storage.create_document.assert_not_called()
        # update_document IS called to reset status + content
        lake._engine._storage.update_document.assert_called_once()
        updated = lake._engine._storage.update_document.call_args[0][0]
        assert updated.status == DocumentStatus.PENDING
        assert updated.content == "new content"
        assert updated.metadata.source == "updated-source"

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].skipped is False
        assert results[0].chunks_created == 3
        assert handle.failed == 0

    @pytest.mark.asyncio
    async def test_completed_external_id_reported_as_skipped(self) -> None:
        """COMPLETED document with same external_id is skipped, not re-processed (DYT-3075)."""
        from khora.core.models.document import Document, DocumentStatus

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        existing_doc = Document(namespace_id=ns_id, content="done", external_id="ext-done")
        existing_doc.status = DocumentStatus.COMPLETED
        existing_doc.chunk_count = 5
        existing_doc.entity_count = 3
        existing_doc.relationship_count = 7

        lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={"ext-done": existing_doc})

        async def _fake_process(doc, **kwargs):
            return (1, 1, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "done", "external_id": "ext-done"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        # Neither create nor update should be called
        lake._engine._storage.create_document.assert_not_called()
        # process_staged_document not called for skipped doc
        lake._engine.process_staged_document.assert_not_called()

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].skipped is True
        assert results[0].chunks_created == 5
        assert results[0].entities_extracted == 3
        assert results[0].relationships_created == 7
        assert results[0].external_id == "ext-done"
        assert handle.failed == 0

    @pytest.mark.asyncio
    async def test_failed_external_id_resets_to_pending_and_reprocesses(self) -> None:
        """FAILED document with same external_id is reset to PENDING and re-processed (DYT-3075)."""
        from khora.core.models.document import Document, DocumentStatus

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        existing_doc = Document(namespace_id=ns_id, content="bad content", external_id="ext-failed")
        existing_doc.status = DocumentStatus.FAILED
        existing_doc.error_message = "previous error"

        lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={"ext-failed": existing_doc})
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)

        async def _fake_process(doc, **kwargs):
            return (2, 1, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "fixed content", "external_id": "ext-failed", "source": "fixed-source"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        # update_document called to reset status and update content
        lake._engine._storage.update_document.assert_called_once()
        updated = lake._engine._storage.update_document.call_args[0][0]
        assert updated.status == DocumentStatus.PENDING
        assert updated.content == "fixed content"
        assert updated.error_message is None

        # Document was re-processed
        lake._engine.process_staged_document.assert_called_once()

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].skipped is False
        assert results[0].external_id == "ext-failed"
        assert handle.failed == 0

    @pytest.mark.asyncio
    async def test_failed_external_id_clears_prior_state_before_reprocess(self) -> None:
        """For a FAILED doc, clear_document_extraction_state is called before re-processing (H1)."""
        from khora.core.models.document import Document, DocumentStatus

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        existing_doc = Document(namespace_id=ns_id, content="bad content", external_id="ext-failed-h1")
        existing_doc.status = DocumentStatus.FAILED
        existing_doc.error_message = "previous error"

        lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={"ext-failed-h1": existing_doc})
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)

        clear_calls: list[tuple] = []

        async def _fake_clear(doc_id, ns_id_arg):
            clear_calls.append((doc_id, ns_id_arg))

        lake._engine.clear_document_extraction_state = AsyncMock(side_effect=_fake_clear)

        async def _fake_process(doc, **kwargs):
            return (2, 1, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)
        lake.start_pending_processor()

        handle = await lake.submit_batch(
            [{"content": "fixed content", "external_id": "ext-failed-h1"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        # Cleanup was called before re-processing
        lake._engine.clear_document_extraction_state.assert_called_once_with(existing_doc.id, ns_id)
        # Document was re-processed
        lake._engine.process_staged_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_processing_external_id_skipped_to_avoid_race(self) -> None:
        """PROCESSING document with same external_id is skipped to avoid race condition (M1)."""
        from khora.core.models.document import Document, DocumentStatus

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        existing_doc = Document(namespace_id=ns_id, content="in progress", external_id="ext-proc")
        existing_doc.status = DocumentStatus.PROCESSING

        lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={"ext-proc": existing_doc})

        async def _fake_process(doc, **kwargs):
            return (1, 0, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "new content", "external_id": "ext-proc"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        # PROCESSING doc skipped — not re-processed and not created
        lake._engine.process_staged_document.assert_not_called()
        lake._engine._storage.create_document.assert_not_called()
        lake._engine._storage.update_document.assert_not_called()

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].skipped is True

    @pytest.mark.asyncio
    async def test_lookup_failure_falls_back_to_create(self) -> None:
        """If get_documents_by_external_ids raises, submit_batch treats all docs as new inserts (M2)."""
        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)
        lake._engine._storage.get_documents_by_external_ids = AsyncMock(side_effect=RuntimeError("DB timeout"))

        async def _fake_create(doc):
            return doc

        lake._engine._storage.create_document = AsyncMock(side_effect=_fake_create)

        async def _fake_process(doc, **kwargs):
            return (2, 1, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "new doc", "external_id": "ext-new-m2"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        # Falls back to create path
        lake._engine._storage.create_document.assert_called_once()
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_duplicate_external_id_in_batch_skips_second(self) -> None:
        """When the same external_id appears twice in a batch, the second is skipped (M4)."""
        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)
        lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})

        async def _fake_create(doc):
            return doc

        lake._engine._storage.create_document = AsyncMock(side_effect=_fake_create)

        async def _fake_process(doc, **kwargs):
            return (2, 1, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [
                {"content": "first content", "external_id": "ext-dup"},
                {"content": "second content", "external_id": "ext-dup"},  # duplicate
            ],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        # Only one document created (second was skipped)
        lake._engine._storage.create_document.assert_called_once()
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_archived_external_id_skipped_by_default(self) -> None:
        """ARCHIVED document with same external_id is skipped by default (DYT-3077).

        ARCHIVED means 'not actively used'. Silently re-activating it on any
        batch submission that includes its external_id violates that semantic.
        By default, submit_batch skips ARCHIVED docs and fires a skipped result.
        """
        from khora.core.models.document import Document, DocumentStatus

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        existing_doc = Document(namespace_id=ns_id, content="archived content", external_id="ext-archived")
        existing_doc.status = DocumentStatus.ARCHIVED
        existing_doc.chunk_count = 4
        existing_doc.entity_count = 2

        lake._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={"ext-archived": existing_doc})

        async def _fake_process(doc, **kwargs):
            return (1, 1, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "new content", "external_id": "ext-archived"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        # ARCHIVED doc skipped — not re-processed, not created
        lake._engine.process_staged_document.assert_not_called()
        lake._engine._storage.create_document.assert_not_called()
        lake._engine._storage.update_document.assert_not_called()

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].skipped is True
        assert results[0].chunks_created == 4
        assert results[0].entities_extracted == 2
        assert handle.failed == 0

    @pytest.mark.asyncio
    async def test_archived_external_id_reprocessed_when_flag_set(self) -> None:
        """ARCHIVED document is reset to PENDING and re-processed when reprocess_archived=True (DYT-3077)."""
        from khora.core.models.document import Document, DocumentStatus

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        existing_doc = Document(namespace_id=ns_id, content="archived content", external_id="ext-archived-reprocess")
        existing_doc.status = DocumentStatus.ARCHIVED

        lake._engine._storage.get_documents_by_external_ids = AsyncMock(
            return_value={"ext-archived-reprocess": existing_doc}
        )
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)
        lake._engine._storage.vector.delete_chunks_by_document = AsyncMock()
        lake._engine.clear_document_extraction_state = AsyncMock()

        async def _fake_process(doc, **kwargs):
            return (3, 2, 1)

        lake._engine.process_staged_document = AsyncMock(side_effect=_fake_process)
        lake.start_pending_processor()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "refreshed content", "external_id": "ext-archived-reprocess", "source": "refresh"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            reprocess_archived=True,
        )
        await handle.wait()

        # update_document called to reset status and update content
        lake._engine._storage.update_document.assert_called_once()
        updated = lake._engine._storage.update_document.call_args[0][0]
        assert updated.status == DocumentStatus.PENDING
        assert updated.content == "refreshed content"
        assert updated.metadata.source == "refresh"
        assert updated.error_message is None

        # Prior extraction state was cleared before re-processing (H1)
        lake._engine._storage.vector.delete_chunks_by_document.assert_called_once_with(existing_doc.id)
        lake._engine.clear_document_extraction_state.assert_called_once_with(existing_doc.id, ns_id)

        # Document was re-processed
        lake._engine.process_staged_document.assert_called_once()

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].skipped is False
        assert results[0].chunks_created == 3
        assert results[0].entities_extracted == 2
        assert handle.failed == 0

    @pytest.mark.asyncio
    async def test_archived_reprocess_update_document_failure_goes_to_failed(self) -> None:
        """When reprocess_archived=True and update_document raises, the doc goes to pre_failed_docs (DYT-3077)."""
        from khora.core.models.document import Document, DocumentStatus

        ns_id = uuid4()
        lake = _make_lake(connected=True)
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

        existing_doc = Document(namespace_id=ns_id, content="archived content", external_id="ext-archived-fail")
        existing_doc.status = DocumentStatus.ARCHIVED

        lake._engine._storage.get_documents_by_external_ids = AsyncMock(
            return_value={"ext-archived-fail": existing_doc}
        )
        lake._engine._storage.update_document = AsyncMock(side_effect=RuntimeError("DB write error"))
        lake._engine.process_staged_document = AsyncMock()

        results: list[DocumentResult] = []

        handle = await lake.submit_batch(
            [{"content": "refreshed content", "external_id": "ext-archived-fail"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            reprocess_archived=True,
        )
        await handle.wait()

        # Document was not re-processed — went to failed path
        lake._engine.process_staged_document.assert_not_called()

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].skipped is False
        assert handle.failed == 1


# ---------------------------------------------------------------------------
# _GlobalChunkSemaphore (DYT-3111)
# ---------------------------------------------------------------------------


class TestGlobalChunkSemaphore:
    """Unit tests for the _GlobalChunkSemaphore counting semaphore."""

    @pytest.mark.asyncio
    async def test_acquire_release_basic(self) -> None:
        """acquire(n) decrements capacity; release(n) restores it."""
        from khora.khora import _GlobalChunkSemaphore

        sem = _GlobalChunkSemaphore(10)
        assert sem.capacity == 10
        await sem.acquire(5)
        await sem.release(5)
        # After release, another acquire of full capacity should succeed immediately.
        await sem.acquire(10)
        await sem.release(10)

    @pytest.mark.asyncio
    async def test_acquire_blocks_until_capacity_available(self) -> None:
        """acquire blocks when in_flight + n > capacity, unblocks on release."""
        import asyncio

        from khora.khora import _GlobalChunkSemaphore

        sem = _GlobalChunkSemaphore(5)
        await sem.acquire(5)  # fills capacity

        unblocked = asyncio.Event()

        async def _waiter():
            await sem.acquire(1)  # must wait
            unblocked.set()
            await sem.release(1)

        task = asyncio.create_task(_waiter())
        # Give waiter a chance to start and block.
        await asyncio.sleep(0)
        assert not unblocked.is_set(), "waiter should still be blocked"

        await sem.release(5)  # unblocks waiter
        await asyncio.wait_for(task, timeout=2.0)
        assert unblocked.is_set()

    @pytest.mark.asyncio
    async def test_multiple_waiters_queue_correctly(self) -> None:
        """Multiple waiters each get capacity in turn."""
        import asyncio

        from khora.khora import _GlobalChunkSemaphore

        sem = _GlobalChunkSemaphore(3)
        order: list[int] = []

        async def _worker(idx: int, n: int) -> None:
            await sem.acquire(n)
            order.append(idx)
            await asyncio.sleep(0)  # yield to allow ordering checks
            await sem.release(n)

        await sem.acquire(3)  # fill semaphore
        # Schedule two waiters
        t1 = asyncio.create_task(_worker(1, 2))
        t2 = asyncio.create_task(_worker(2, 1))
        await asyncio.sleep(0)
        assert order == [], "no worker should have proceeded yet"

        await sem.release(3)
        await asyncio.gather(t1, t2)
        # Both workers completed — order determined by asyncio scheduling.
        assert sorted(order) == [1, 2]

    @pytest.mark.asyncio
    async def test_acquire_clamped_to_capacity(self) -> None:
        """acquire(n) with n > capacity is clamped to capacity (avoids deadlock)."""
        from khora.khora import _GlobalChunkSemaphore

        sem = _GlobalChunkSemaphore(5)
        # n=10 > capacity=5 — should not deadlock; clamped to 5.
        await sem.acquire(10)
        assert sem._in_flight == 5
        await sem.release(5)
        assert sem._in_flight == 0


# ---------------------------------------------------------------------------
# Global semaphore initialization in submit_batch (DYT-3111)
# ---------------------------------------------------------------------------


class TestSubmitBatchGlobalSemaphore:
    """Tests for global chunk semaphore lifecycle and behavior in submit_batch."""

    @pytest.mark.asyncio
    async def test_semaphore_initialized_on_first_call(self) -> None:
        """First submit_batch with max_chunks_in_flight creates _chunk_semaphore."""
        from khora.khora import _GlobalChunkSemaphore

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        assert lake._chunk_semaphore is None

        handle = await lake.submit_batch(
            [{"content": "hello", "external_id": "s1"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=100,
        )
        await handle.wait()

        assert isinstance(lake._chunk_semaphore, _GlobalChunkSemaphore)
        assert lake._chunk_semaphore.capacity == 100

    @pytest.mark.asyncio
    async def test_semaphore_reused_on_second_call_same_value(self) -> None:
        """Second call with same max_chunks_in_flight reuses existing semaphore."""

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        h1 = await lake.submit_batch(
            [{"content": "doc1", "external_id": "r1"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=50,
        )
        await h1.wait()
        first_semaphore = lake._chunk_semaphore

        h2 = await lake.submit_batch(
            [{"content": "doc2", "external_id": "r2"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=50,
        )
        await h2.wait()

        assert lake._chunk_semaphore is first_semaphore, "same semaphore instance reused"

    @pytest.mark.asyncio
    async def test_conflicting_max_chunks_in_flight_logs_warning(self) -> None:
        """Second call with different max_chunks_in_flight logs a warning; first wins."""

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        h1 = await lake.submit_batch(
            [{"content": "doc1", "external_id": "w1"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=100,
        )
        await h1.wait()

        from loguru import logger

        captured: list[str] = []
        handler_id = logger.add(lambda msg: captured.append(msg), level="WARNING")
        try:
            h2 = await lake.submit_batch(
                [{"content": "doc2", "external_id": "w2"}],
                on_result=lambda c, t, r: None,
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                max_chunks_in_flight=200,  # different value
            )
            await h2.wait()
        finally:
            logger.remove(handler_id)

        assert lake._chunk_semaphore is not None
        assert lake._chunk_semaphore.capacity == 100, "first value wins"
        assert any("conflicts" in str(m) or "first value wins" in str(m) for m in captured), (
            f"expected warning about conflicting max_chunks_in_flight; got: {captured}"
        )

    @pytest.mark.asyncio
    async def test_chunk_semaphore_passed_to_process_staged_document(self) -> None:
        """chunk_semaphore kwarg is forwarded to process_staged_document."""

        from khora.khora import _GlobalChunkSemaphore

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        received_kwargs: list[dict] = []

        async def _capturing_process(doc, **kwargs):
            received_kwargs.append(kwargs)
            return (1, 0, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_capturing_process)

        handle = await lake.submit_batch(
            [{"content": "test", "external_id": "cs1"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=10,
        )
        await handle.wait()

        assert len(received_kwargs) == 1
        assert "chunk_semaphore" in received_kwargs[0]
        assert isinstance(received_kwargs[0]["chunk_semaphore"], _GlobalChunkSemaphore)
        assert received_kwargs[0]["chunk_semaphore"].capacity == 10

    @pytest.mark.asyncio
    async def test_no_semaphore_when_max_chunks_in_flight_none(self) -> None:
        """When max_chunks_in_flight=None, no semaphore is created."""
        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        handle = await lake.submit_batch(
            [{"content": "hello", "external_id": "n1"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=None,
        )
        await handle.wait()

        assert lake._chunk_semaphore is None

    @pytest.mark.asyncio
    async def test_concurrent_calls_share_semaphore(self) -> None:
        """Two concurrent submit_batch calls share the same semaphore instance."""
        import asyncio

        from khora.khora import _GlobalChunkSemaphore

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        semaphores_seen: list[object] = []
        processing_events: list[asyncio.Event] = []

        async def _tracking_process(doc, **kwargs):
            semaphores_seen.append(kwargs.get("chunk_semaphore"))
            ev = asyncio.Event()
            processing_events.append(ev)
            return (1, 0, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_tracking_process)

        h1 = await lake.submit_batch(
            [{"content": "doc-a", "external_id": "ca1"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=100,
        )
        h2 = await lake.submit_batch(
            [{"content": "doc-b", "external_id": "ca2"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=100,
        )
        await asyncio.gather(h1.wait(), h2.wait())

        assert len(semaphores_seen) == 2
        assert semaphores_seen[0] is semaphores_seen[1], "both calls share the same semaphore"
        assert isinstance(semaphores_seen[0], _GlobalChunkSemaphore)

    @pytest.mark.asyncio
    async def test_semaphore_released_on_process_failure(self) -> None:
        """Semaphore tokens are released even when process_staged_document raises."""
        from khora.khora import _GlobalChunkSemaphore

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        call_count = 0

        async def _failing_process(doc, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated failure")
            return (1, 0, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_failing_process)

        # First call fails — semaphore should still be released
        h1 = await lake.submit_batch(
            [{"content": "fail-doc", "external_id": "sf1"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=10,
        )
        await h1.wait()

        sem = lake._chunk_semaphore
        assert isinstance(sem, _GlobalChunkSemaphore)

        # NOTE: The semaphore is per-window inside the engine, not per-document
        # in the lake layer. Since the mock doesn't use the semaphore itself,
        # we verify the lake still has a valid semaphore and the second call
        # can proceed (no deadlock from unreleased tokens).
        h2 = await lake.submit_batch(
            [{"content": "ok-doc", "external_id": "sf2"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=10,
        )
        await h2.wait()
        assert h2.failed == 0, "second call should succeed after first failed"

    @pytest.mark.asyncio
    async def test_none_max_chunks_after_prior_semaphore_does_not_inherit(self) -> None:
        """submit_batch(None) after a prior semaphored call passes no semaphore (H-2 fix)."""
        from khora.khora import _GlobalChunkSemaphore

        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        received_semaphores: list[object] = []

        async def _capturing_process(doc, **kwargs):
            received_semaphores.append(kwargs.get("chunk_semaphore"))
            return (1, 0, 0)

        lake._engine.process_staged_document = AsyncMock(side_effect=_capturing_process)

        # First call establishes a semaphore.
        h1 = await lake.submit_batch(
            [{"content": "doc1", "external_id": "h2a"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=50,
        )
        await h1.wait()

        # Second call opts out (None = unbounded) — must NOT inherit the semaphore.
        h2 = await lake.submit_batch(
            [{"content": "doc2", "external_id": "h2b"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            max_chunks_in_flight=None,
        )
        await h2.wait()

        assert isinstance(received_semaphores[0], _GlobalChunkSemaphore), "first call gets semaphore"
        assert received_semaphores[1] is None, "second call with None must receive no semaphore"


# ---------------------------------------------------------------------------
# Tests for acquire/release in _process_document (DYT-3111 M-3/M-4)
# ---------------------------------------------------------------------------


class TestProcessDocumentSemaphore:
    """Tests that exercise the actual acquire/release path in engine._process_document.

    These tests complement TestSubmitBatchGlobalSemaphore, which mocks out
    process_staged_document entirely.  Here we call _process_document directly
    with mocked I/O so that the semaphore acquire/release in engine.py:779-844
    is actually executed.
    """

    @staticmethod
    def _make_minimal_engine():
        """Return a VectorCypherEngine with all I/O mocked — no connect() needed."""
        from unittest.mock import AsyncMock, MagicMock

        from khora.engines.vectorcypher.engine import VectorCypherConfig, VectorCypherEngine

        # Bypass __init__ (requires full config + Neo4j/storage setup).
        engine = object.__new__(VectorCypherEngine)

        cfg = MagicMock()
        cfg.pipeline.chunking_strategy = None
        cfg.pipeline.chunk_size = 512
        cfg.pipeline.chunk_overlap = 50
        cfg.pipeline.extract_entities = False  # skip LLM extraction

        engine._config = cfg
        engine._vc_config = VectorCypherConfig()

        storage = MagicMock()
        storage.update_document = AsyncMock()
        engine._storage = storage

        embedder = MagicMock()
        embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.0] * 8 for _ in texts])
        engine._embedder = embedder

        async def _create_chunks_batch(chunks):
            for c in chunks:
                c.id = uuid4()
            return chunks

        temporal_store = MagicMock()
        temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks_batch)
        engine._temporal_store = temporal_store
        engine._dual_nodes = None

        return engine

    @staticmethod
    def _mock_raw_chunk(content: str = "test chunk"):
        """Return a mock raw chunk with the attributes _process_document reads."""
        from unittest.mock import MagicMock

        chunk = MagicMock()
        chunk.content = content
        chunk.start_char = 0
        chunk.end_char = len(content)
        return chunk

    @pytest.mark.asyncio
    async def test_semaphore_released_on_embed_failure(self) -> None:
        """release() is called in finally even when embed_batch raises (M-3 fix)."""
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock, MagicMock, patch

        from khora.core.models.document import Document
        from khora.khora import _GlobalChunkSemaphore

        engine = self._make_minimal_engine()
        engine._embedder.embed_batch = AsyncMock(side_effect=RuntimeError("embed failure"))

        sem = _GlobalChunkSemaphore(100)
        doc = Document(namespace_id=uuid4(), content="some content")
        mock_chunk = self._mock_raw_chunk()

        with patch("khora.extraction.chunkers.create_chunker") as mock_cc:
            mock_cc.return_value = MagicMock()
            with patch("asyncio.to_thread", new=AsyncMock(return_value=[mock_chunk])):
                with pytest.raises(RuntimeError, match="embed failure"):
                    await engine._process_document(
                        doc,
                        skill_name="default",
                        expertise=None,
                        extraction_model=None,
                        occurred_at=datetime.now(UTC),
                        entity_types=["PERSON"],
                        relationship_types=["KNOWS"],
                        chunk_semaphore=sem,
                    )

        assert sem._in_flight == 0, "semaphore must be fully released after embed_batch failure"

    @pytest.mark.asyncio
    async def test_semaphore_released_after_success(self) -> None:
        """Semaphore tokens return to 0 after a successful window (M-4 coverage)."""
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock, MagicMock, patch

        from khora.core.models.document import Document
        from khora.khora import _GlobalChunkSemaphore

        engine = self._make_minimal_engine()

        sem = _GlobalChunkSemaphore(100)
        doc = Document(namespace_id=uuid4(), content="some content")
        mock_chunk = self._mock_raw_chunk()

        with patch("khora.extraction.chunkers.create_chunker") as mock_cc:
            mock_cc.return_value = MagicMock()
            with patch("asyncio.to_thread", new=AsyncMock(return_value=[mock_chunk])):
                result = await engine._process_document(
                    doc,
                    skill_name="default",
                    expertise=None,
                    extraction_model=None,
                    occurred_at=datetime.now(UTC),
                    entity_types=["PERSON"],
                    relationship_types=["KNOWS"],
                    chunk_semaphore=sem,
                )

        assert sem._in_flight == 0, "semaphore must be fully released after success"
        assert result[0] == 1  # 1 chunk created

    @pytest.mark.asyncio
    async def test_semaphore_clamped_acquire_releases_correctly(self) -> None:
        """When n > capacity, acquire clamps and release uses the clamped value (H-1 fix)."""
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock, MagicMock, patch

        from khora.core.models.document import Document
        from khora.khora import _GlobalChunkSemaphore

        engine = self._make_minimal_engine()

        # Semaphore capacity=3; window will have 1 chunk.
        # With the H-1 fix, acquire(1) returns 1 and release(1) is called — no underflow.
        sem = _GlobalChunkSemaphore(3)
        doc = Document(namespace_id=uuid4(), content="chunk content")
        mock_chunk = self._mock_raw_chunk()

        with patch("khora.extraction.chunkers.create_chunker") as mock_cc:
            mock_cc.return_value = MagicMock()
            with patch("asyncio.to_thread", new=AsyncMock(return_value=[mock_chunk])):
                await engine._process_document(
                    doc,
                    skill_name="default",
                    expertise=None,
                    extraction_model=None,
                    occurred_at=datetime.now(UTC),
                    entity_types=["PERSON"],
                    relationship_types=["KNOWS"],
                    max_chunks_in_flight=10,  # > semaphore capacity
                    chunk_semaphore=sem,
                )

        assert sem._in_flight == 0, "release must use clamped acquire count — no underflow"


# ---------------------------------------------------------------------------
# Unified pending processor (DYT-3305)
# ---------------------------------------------------------------------------


class TestPendingProcessor:
    """Unit tests for the unified pending processor."""

    def _make_lake_with_processor(self) -> Khora:
        """Create a Khora with the pending processor enabled."""
        cfg = _mock_config()
        cfg.pipelines.pending_processor_enabled = True
        cfg.pipelines.pending_processor_max_concurrent = 20
        cfg.pipelines.pending_processor_grace_period_minutes = 5
        cfg.pipelines.entity_types = ["PERSON", "ORGANIZATION"]
        with patch("khora.khora.load_config", return_value=cfg):
            lake = Khora()
        lake._connected = True
        eng = _mock_engine()
        lake._engine = eng
        return lake

    @pytest.mark.asyncio
    async def test_connect_never_starts_processor(self) -> None:
        """connect() never spawns the pending processor regardless of config."""
        lake = _make_lake()
        eng = _mock_engine()
        with patch("khora.engines.create_engine", return_value=eng):
            await lake.connect()
        assert lake._processor_task is None

    @pytest.mark.asyncio
    async def test_start_pending_processor_starts_task(self) -> None:
        """start_pending_processor() spawns the background task."""
        lake = self._make_lake_with_processor()
        assert lake._processor_task is None
        lake.start_pending_processor()
        assert lake._processor_task is not None
        assert not lake._processor_task.done()
        lake._processor_task.cancel()

    @pytest.mark.asyncio
    async def test_start_pending_processor_idempotent(self) -> None:
        """Calling start_pending_processor() twice does not spawn two tasks."""
        lake = self._make_lake_with_processor()
        lake.start_pending_processor()
        first_task = lake._processor_task
        lake.start_pending_processor()
        assert lake._processor_task is first_task
        lake._processor_task.cancel()

    @pytest.mark.asyncio
    async def test_start_pending_processor_requires_connected(self) -> None:
        """start_pending_processor() raises if the lake is not connected."""
        with patch("khora.khora.load_config", return_value=_mock_config()):
            lake = Khora()
        with pytest.raises(RuntimeError, match="not connected"):
            lake.start_pending_processor()

    @pytest.mark.asyncio
    async def test_stop_pending_processor_cancels_task(self) -> None:
        """stop_pending_processor() cancels the running task."""
        lake = self._make_lake_with_processor()
        lake.start_pending_processor()
        assert lake._processor_task is not None
        await lake.stop_pending_processor()
        assert lake._processor_task is None

    @pytest.mark.asyncio
    async def test_stop_pending_processor_noop_when_not_started(self) -> None:
        """stop_pending_processor() is a no-op if the processor was never started."""
        lake = self._make_lake_with_processor()
        await lake.stop_pending_processor()  # Should not raise
        assert lake._processor_task is None

    @pytest.mark.asyncio
    async def test_start_after_stop_restarts_processor(self) -> None:
        """start_pending_processor() after stop_pending_processor() starts a new task."""
        lake = self._make_lake_with_processor()
        lake.start_pending_processor()
        first_task = lake._processor_task
        await lake.stop_pending_processor()
        lake.start_pending_processor()
        assert lake._processor_task is not None
        assert lake._processor_task is not first_task
        lake._processor_task.cancel()

    @pytest.mark.asyncio
    async def test_orphan_recovery_skipped_when_no_process_fn(self) -> None:
        """Orphan recovery exits silently if engine has no process_staged_document."""
        lake = self._make_lake_with_processor()
        del lake._engine.process_staged_document

        await lake._enqueue_orphaned_pending_docs()  # Should not raise

    @pytest.mark.asyncio
    async def test_orphan_recovery_skipped_when_no_storage(self) -> None:
        """Orphan recovery exits silently if engine exposes no _storage."""
        lake = self._make_lake_with_processor()
        lake._engine._storage = None

        await lake._enqueue_orphaned_pending_docs()  # Should not raise

    @pytest.mark.asyncio
    async def test_orphan_recovery_enqueues_stale_docs(self) -> None:
        """Stale PENDING documents are enqueued and processed by the processor."""
        from datetime import UTC, timedelta

        from khora.core.models import MemoryNamespace
        from khora.core.models.document import Document
        from khora.storage.backends.base import PaginatedResult

        lake = self._make_lake_with_processor()

        ns_id = uuid4()
        ns = MemoryNamespace(id=ns_id, namespace_id=ns_id)
        stale_doc = Document(namespace_id=ns_id, content="stale content")

        lake._engine._storage.list_namespaces = AsyncMock(
            side_effect=[
                PaginatedResult(items=[ns], total=1, limit=100, offset=0),
                PaginatedResult(items=[], total=0, limit=100, offset=100),
            ]
        )
        lake._engine._storage.list_documents = AsyncMock(
            side_effect=[
                [stale_doc],
                [],
            ]
        )

        await lake._enqueue_orphaned_pending_docs()

        # Verify the grace-period filter is applied correctly.
        list_docs_call = lake._engine._storage.list_documents.call_args_list[0]
        assert list_docs_call.kwargs["status"] == "pending"
        assert list_docs_call.kwargs["updated_before"] <= datetime.now(UTC) - timedelta(minutes=5)

        # Verify doc was enqueued.
        assert lake._processor_queue.qsize() == 1
        item = lake._processor_queue.get_nowait()
        assert item.doc is stale_doc
        assert item.batch_reg is None  # orphan — no batch registration

    @pytest.mark.asyncio
    async def test_orphan_recovery_processes_with_stored_params(self) -> None:
        """Orphaned docs use their stored extraction_params for processing."""
        from khora.core.models.document import Document
        from khora.khora import _ProcessorItem

        lake = self._make_lake_with_processor()
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)

        ns_id = uuid4()
        doc = Document(
            namespace_id=ns_id,
            content="content",
            extraction_config_hash="abc123",
            extraction_params={
                "skill_name": "custom_skill",
                "entity_types": ["PERSON"],
                "relationship_types": ["KNOWS"],
                "expertise": None,
                "chunk_strategy": "fixed",
                "max_chunks_in_flight": None,
            },
        )

        process_fn = AsyncMock(return_value=(1, 0, 0))
        lake._engine.process_staged_document = process_fn

        await lake._process_pending_item(_ProcessorItem(doc=doc, doc_data=None, batch_reg=None))

        process_fn.assert_awaited_once()
        _, call_kwargs = process_fn.call_args
        assert call_kwargs["skill_name"] == "custom_skill"
        assert call_kwargs["entity_types"] == ["PERSON"]
        assert call_kwargs["relationship_types"] == ["KNOWS"]
        assert call_kwargs["extraction_config_hash"] == "abc123"
        assert call_kwargs["chunk_strategy"] == "fixed"

    @pytest.mark.asyncio
    async def test_orphan_recovery_falls_back_to_defaults(self) -> None:
        """Orphaned docs without extraction_params fall back to config defaults."""
        from khora.core.models.document import Document
        from khora.khora import _ProcessorItem

        lake = self._make_lake_with_processor()
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)

        doc = Document(namespace_id=uuid4(), content="content")

        process_fn = AsyncMock(return_value=(1, 0, 0))
        lake._engine.process_staged_document = process_fn

        await lake._process_pending_item(_ProcessorItem(doc=doc, doc_data=None, batch_reg=None))

        _, call_kwargs = process_fn.call_args
        assert call_kwargs["skill_name"] == "general_entities"
        assert call_kwargs["entity_types"] == ["PERSON", "ORGANIZATION"]

    @pytest.mark.asyncio
    async def test_processor_handles_per_doc_failure(self) -> None:
        """Per-document failures in the processor are handled gracefully."""
        from khora.core.models.document import Document, DocumentStatus
        from khora.khora import _ProcessorItem

        lake = self._make_lake_with_processor()
        lake._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)

        doc = Document(namespace_id=uuid4(), content="will fail")
        process_fn = AsyncMock(side_effect=RuntimeError("boom"))
        lake._engine.process_staged_document = process_fn

        await lake._process_pending_item(_ProcessorItem(doc=doc, doc_data=None, batch_reg=None))

        # Doc should be marked FAILED.
        assert doc.status == DocumentStatus.FAILED
        assert "boom" in doc.error_message

    @pytest.mark.asyncio
    async def test_submit_batch_stores_extraction_params(self) -> None:
        """submit_batch stores extraction params on created documents."""
        ns_id = uuid4()
        lake = _make_lake_with_staged_support(ns_id)

        created_docs = []
        orig_create = lake._engine._storage.create_document.side_effect

        async def _spy_create(doc):
            created_docs.append(doc)
            return await orig_create(doc)

        lake._engine._storage.create_document.side_effect = _spy_create

        handle = await lake.submit_batch(
            [{"content": "test doc"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            skill_name="custom_skill",
        )
        await handle.wait()

        assert len(created_docs) == 1
        params = created_docs[0].extraction_params
        assert params is not None
        assert params["skill_name"] == "custom_skill"
        assert params["entity_types"] == ["PERSON"]
        assert params["relationship_types"] == ["KNOWS"]


# ---------------------------------------------------------------------------
# DYT-3787: undefined-table detection for fresh-DB orphan recovery
# ---------------------------------------------------------------------------


class TestIsUndefinedTableError:
    """Tests for `_is_undefined_table_error` — used by `_run_pending_processor`
    to silence the "memory_namespaces does not exist" ERROR on fresh ephemeral
    DBs (DYT-3787)."""

    def test_detects_undefined_table_via_sqlstate_attribute(self) -> None:
        from khora.khora import _is_undefined_table_error

        class _FakeAsyncpgError(Exception):
            sqlstate = "42P01"

        assert _is_undefined_table_error(_FakeAsyncpgError("relation does not exist")) is True

    def test_detects_when_wrapped_via_orig(self) -> None:
        """SQLAlchemy wraps the asyncpg exception under `.orig` — the helper
        must look through the wrapper, not just the top-level exception."""
        from khora.khora import _is_undefined_table_error

        class _AsyncpgError(Exception):
            sqlstate = "42P01"

        class _SQLAlchemyError(Exception):
            def __init__(self, orig: Exception) -> None:
                super().__init__(str(orig))
                self.orig = orig

        wrapped = _SQLAlchemyError(_AsyncpgError("table missing"))
        assert _is_undefined_table_error(wrapped) is True

    def test_returns_false_for_other_sqlstate(self) -> None:
        """Other postgres errors (constraint violation, etc.) must not match."""
        from khora.khora import _is_undefined_table_error

        class _OtherError(Exception):
            sqlstate = "23505"  # unique_violation

        assert _is_undefined_table_error(_OtherError("dup key")) is False

    def test_returns_false_for_plain_exception(self) -> None:
        from khora.khora import _is_undefined_table_error

        assert _is_undefined_table_error(RuntimeError("not a db error")) is False
