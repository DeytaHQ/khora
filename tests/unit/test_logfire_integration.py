"""Tests for optional Logfire/OTEL integration.

Validates that:
- logfire_span() works as a no-op when logfire is absent
- logfire_span() emits real spans when logfire is present (mocked)
- Decorated functions (instrument_llm, instrument_storage, pipeline_stage)
  emit logfire spans with correct names and attributes
- MemoryLake.remember/recall/forget/remember_batch create top-level spans
- No content strings leak into span attributes
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.memory_lake import BatchResult, MemoryLake, RecallResult, RememberResult
from khora.telemetry import NoOpCollector
from khora.telemetry.instrument import instrument_llm, instrument_storage, pipeline_stage
from khora.telemetry.logfire_integration import logfire_span

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingCollector(NoOpCollector):
    """Collector that records calls for assertions."""

    def __init__(self):
        self.llm_calls: list[dict] = []
        self.storage_calls: list[dict] = []
        self.pipeline_calls: list[dict] = []

    def record_llm_call(self, **kwargs):
        self.llm_calls.append(kwargs)

    def record_storage_op(self, **kwargs):
        self.storage_calls.append(kwargs)

    def record_pipeline_stage(self, **kwargs):
        self.pipeline_calls.append(kwargs)


@pytest.fixture
def recording_collector():
    collector = _RecordingCollector()
    with patch("khora.telemetry._collector", collector):
        yield collector


def _mock_config() -> MagicMock:
    mock_config = MagicMock()
    mock_config.get_postgresql_url.return_value = "postgresql://test"
    mock_config.get_graph_config.return_value = None
    mock_config.get_vector_config.return_value = None
    mock_config.get_neo4j_url.return_value = None
    mock_config.get_neo4j_user.return_value = None
    mock_config.get_neo4j_password.return_value = None
    mock_config.get_neo4j_database.return_value = None
    mock_config.storage.embedding_dimension = 1536
    mock_config.llm.model = "gpt-4o-mini"
    mock_config.llm.embedding_model = "text-embedding-3-small"
    mock_config.llm.embedding_dimension = 1536
    mock_config.llm.extraction_model = None
    mock_config.llm.timeout = 30
    mock_config.llm.max_retries = 3
    mock_config.telemetry_database_url = None
    mock_config.telemetry_service_name = "khora-test"
    return mock_config


def _mock_engine() -> MagicMock:
    mock_eng = MagicMock()
    mock_eng._storage = MagicMock()
    mock_eng._embedder = MagicMock()
    mock_eng._default_namespace_id = None
    mock_eng.connect = AsyncMock()
    mock_eng.disconnect = AsyncMock()
    mock_eng.health_check = AsyncMock(return_value={"status": "healthy"})
    mock_eng.remember = AsyncMock()
    mock_eng.recall = AsyncMock()
    mock_eng.forget = AsyncMock()
    mock_eng.remember_batch = AsyncMock()
    mock_eng.get_or_create_default_namespace = AsyncMock(return_value=uuid4())
    mock_eng.create_namespace = AsyncMock()
    mock_eng.get_namespace = AsyncMock()
    mock_eng.ensure_namespace = AsyncMock()
    mock_eng.get_entity = AsyncMock()
    mock_eng.list_entities = AsyncMock(return_value=[])
    mock_eng.find_related_entities = AsyncMock(return_value=[])
    mock_eng.get_document = AsyncMock()
    mock_eng.list_documents = AsyncMock(return_value=[])
    mock_eng.search_entities = AsyncMock(return_value=[])
    mock_eng.stats = AsyncMock()
    return mock_eng


def _make_lake(*, connected: bool = False) -> MemoryLake:
    with patch("khora.memory_lake.load_config", return_value=_mock_config()):
        lake = MemoryLake()
    if connected:
        lake._connected = True
        lake._engine = _mock_engine()
    return lake


def _make_mock_logfire_span():
    """Create a mock logfire.span context manager that tracks set_attribute calls."""
    mock_span = MagicMock()
    mock_span.set_attribute = MagicMock()

    mock_logfire = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_span)
    cm.__exit__ = MagicMock(return_value=False)
    mock_logfire.span.return_value = cm

    return mock_logfire, mock_span


# =========================================================================
# 1. logfire_span helper tests
# =========================================================================


class TestLogfireSpanNoLogfire:
    """Tests for logfire_span when logfire is not installed."""

    def test_yields_none_when_absent(self):
        """logfire_span yields None when _HAS_LOGFIRE is False."""
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with logfire_span("test.span", key="value") as span:
                assert span is None

    def test_context_manager_works_when_absent(self):
        """logfire_span context manager enters and exits cleanly."""
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            result = "untouched"
            with logfire_span("test.span") as span:
                result = "executed"
            assert result == "executed"
            assert span is None

    def test_no_errors_with_attributes_when_absent(self):
        """Passing attributes when logfire is absent causes no errors."""
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with logfire_span("test.span", foo="bar", count=42) as span:
                assert span is None


class TestLogfireSpanWithLogfire:
    """Tests for logfire_span when logfire is installed (mocked)."""

    def test_creates_span_with_correct_name(self):
        """logfire_span creates a span with the given name."""
        mock_logfire, mock_span = _make_mock_logfire_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            with logfire_span("khora.test.operation") as span:
                assert span is mock_span

        mock_logfire.span.assert_called_once_with("khora.test.operation")

    def test_passes_attributes_to_span(self):
        """logfire_span passes keyword attributes to logfire.span()."""
        mock_logfire, _ = _make_mock_logfire_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            with logfire_span("khora.test.op", backend="neo4j", count=5):
                pass

        mock_logfire.span.assert_called_once_with("khora.test.op", backend="neo4j", count=5)

    def test_span_set_attribute_works(self):
        """set_attribute on the yielded span works when logfire is present."""
        mock_logfire, mock_span = _make_mock_logfire_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            with logfire_span("khora.test.op") as span:
                span.set_attribute("latency_ms", 42.5)
                span.set_attribute("status", "success")

        mock_span.set_attribute.assert_any_call("latency_ms", 42.5)
        mock_span.set_attribute.assert_any_call("status", "success")


# =========================================================================
# 2. Decorator bridging tests
# =========================================================================


class TestInstrumentLLMLogfire:
    """Tests for @instrument_llm logfire span emission."""

    @pytest.mark.asyncio
    async def test_emits_logfire_span_when_present(self, recording_collector):
        """instrument_llm emits a logfire span with correct name."""
        mock_logfire, mock_span = _make_mock_logfire_span()

        @instrument_llm("entity_extraction")
        async def my_llm_call():
            return MagicMock(usage=MagicMock(total_tokens=100, prompt_tokens=50, completion_tokens=50), model="gpt-4o")

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            result = await my_llm_call()

        assert result is not None
        mock_logfire.span.assert_called_once_with("khora.llm.entity_extraction")
        mock_span.set_attribute.assert_any_call("model", "gpt-4o")
        mock_span.set_attribute.assert_any_call("total_tokens", 100)
        # latency_ms should be set
        latency_calls = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == "latency_ms"]
        assert len(latency_calls) == 1
        assert latency_calls[0][0][1] > 0

    @pytest.mark.asyncio
    async def test_works_without_logfire(self, recording_collector):
        """instrument_llm works normally when logfire is absent."""

        @instrument_llm("test_op")
        async def my_llm_call():
            return "result"

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            result = await my_llm_call()

        assert result == "result"
        assert len(recording_collector.llm_calls) == 1
        assert recording_collector.llm_calls[0]["operation"] == "test_op"
        assert recording_collector.llm_calls[0]["status"] == "success"


class TestInstrumentStorageLogfire:
    """Tests for @instrument_storage logfire span emission."""

    @pytest.mark.asyncio
    async def test_emits_logfire_span_when_present(self, recording_collector):
        """instrument_storage emits a logfire span with correct name and attributes."""
        mock_logfire, mock_span = _make_mock_logfire_span()

        @instrument_storage("pgvector", "search_similar")
        async def search():
            return [1, 2, 3]

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            result = await search()

        assert result == [1, 2, 3]
        mock_logfire.span.assert_called_once_with("khora.storage.search_similar", backend="pgvector")
        mock_span.set_attribute.assert_any_call("status", "success")
        mock_span.set_attribute.assert_any_call("record_count", 3)
        latency_calls = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == "latency_ms"]
        assert len(latency_calls) == 1
        assert latency_calls[0][0][1] > 0

    @pytest.mark.asyncio
    async def test_works_without_logfire(self, recording_collector):
        """instrument_storage works normally when logfire is absent."""

        @instrument_storage("postgresql", "create_document")
        async def create():
            return {"id": 1}

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            await create()

        assert len(recording_collector.storage_calls) == 1
        assert recording_collector.storage_calls[0]["backend"] == "postgresql"
        assert recording_collector.storage_calls[0]["operation"] == "create_document"

    @pytest.mark.asyncio
    async def test_custom_telemetry_fires_regardless(self, recording_collector):
        """Custom telemetry (collector.record_storage_op) fires whether or not logfire is present."""
        mock_logfire, _ = _make_mock_logfire_span()

        @instrument_storage("neo4j", "create_entity")
        async def create():
            return {"id": 1}

        # With logfire
        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            await create()

        assert len(recording_collector.storage_calls) == 1
        assert recording_collector.storage_calls[0]["operation"] == "create_entity"

        # Without logfire
        recording_collector.storage_calls.clear()
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            await create()

        assert len(recording_collector.storage_calls) == 1
        assert recording_collector.storage_calls[0]["operation"] == "create_entity"


class TestPipelineStageLogfire:
    """Tests for pipeline_stage logfire span emission."""

    @pytest.mark.asyncio
    async def test_emits_logfire_span_when_present(self, recording_collector):
        """pipeline_stage emits a logfire span with correct name and attributes."""
        mock_logfire, mock_span = _make_mock_logfire_span()
        run_id = uuid4()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            async with pipeline_stage("ingestion", "chunking", run_id, input_count=5) as ctx:
                ctx["output_count"] = 20

        mock_logfire.span.assert_called_once_with("khora.ingestion.chunking", input_count=5)
        mock_span.set_attribute.assert_any_call("output_count", 20)
        mock_span.set_attribute.assert_any_call("status", "success")
        latency_calls = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == "latency_ms"]
        assert len(latency_calls) == 1

    @pytest.mark.asyncio
    async def test_works_without_logfire(self, recording_collector):
        """pipeline_stage works normally when logfire is absent."""
        run_id = uuid4()

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            async with pipeline_stage("query", "understanding", run_id, input_count=1) as ctx:
                ctx["output_count"] = 3

        assert len(recording_collector.pipeline_calls) == 1
        call = recording_collector.pipeline_calls[0]
        assert call["pipeline"] == "query"
        assert call["stage"] == "understanding"
        assert call["input_count"] == 1
        assert call["output_count"] == 3
        assert call["status"] == "success"


# =========================================================================
# 3. No-op behavior tests
# =========================================================================


class TestNoOpBehavior:
    """Tests that logfire-absent path works cleanly."""

    def test_no_import_errors_when_logfire_absent(self):
        """Module imports cleanly when logfire is not installed."""
        # This test verifies the module itself handles ImportError.
        # The module is already imported; verify the flag mechanism works.
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with logfire_span("test") as span:
                assert span is None

    @pytest.mark.asyncio
    async def test_all_decorated_functions_work_without_logfire(self, recording_collector):
        """All three decorator types work normally when logfire is absent."""

        @instrument_llm("test_llm")
        async def llm_fn():
            return "llm_result"

        @instrument_storage("pg", "test_storage")
        async def storage_fn():
            return [1, 2]

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            llm_result = await llm_fn()
            storage_result = await storage_fn()

            run_id = uuid4()
            async with pipeline_stage("test_pipeline", "test_stage", run_id) as ctx:
                ctx["output_count"] = 10

        assert llm_result == "llm_result"
        assert storage_result == [1, 2]
        assert len(recording_collector.llm_calls) == 1
        assert len(recording_collector.storage_calls) == 1
        assert len(recording_collector.pipeline_calls) == 1


# =========================================================================
# 4. Memory lake span tests
# =========================================================================


class TestMemoryLakeSpans:
    """Tests that MemoryLake methods create correct top-level logfire spans."""

    @pytest.mark.asyncio
    async def test_remember_creates_span(self):
        """remember() creates a logfire span with namespace_id and content_length."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)
        lake._engine.remember = AsyncMock(
            return_value=RememberResult(
                document_id=uuid4(),
                namespace_id=ns_id,
                chunks_created=3,
                entities_extracted=2,
                relationships_created=1,
            )
        )

        mock_logfire, _ = _make_mock_logfire_span()

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.memory_lake.logfire_span") as mock_span_fn,
        ):
            # Use a real context manager that yields None so the code runs
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                mock_span_fn._last_name = name
                mock_span_fn._last_attrs = attrs
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.remember("Hello, this is test content", title="Test")

        mock_span_fn.assert_called_once()
        call_args = mock_span_fn.call_args
        assert call_args[0][0] == "khora.remember"
        assert call_args[1]["namespace_id"] == str(ns_id)
        assert call_args[1]["content_length"] == len("Hello, this is test content")

    @pytest.mark.asyncio
    async def test_recall_creates_span(self):
        """recall() creates a logfire span with namespace_id and query_length."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)
        lake._engine.recall = AsyncMock(
            return_value=RecallResult(
                query="test query",
                namespace_id=ns_id,
                chunks=[],
                entities=[],
                context_text="",
            )
        )

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.memory_lake.logfire_span") as mock_span_fn,
        ):
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.recall("test query")

        mock_span_fn.assert_called_once()
        call_args = mock_span_fn.call_args
        assert call_args[0][0] == "khora.recall"
        assert call_args[1]["namespace_id"] == str(ns_id)
        assert call_args[1]["query_length"] == len("test query")

    @pytest.mark.asyncio
    async def test_forget_creates_span(self):
        """forget() creates a logfire span with document_id."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()
        lake._engine.forget = AsyncMock(return_value=True)

        with patch("khora.memory_lake.logfire_span") as mock_span_fn:
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.forget(doc_id)

        mock_span_fn.assert_called_once()
        call_args = mock_span_fn.call_args
        assert call_args[0][0] == "khora.forget"
        assert call_args[1]["document_id"] == str(doc_id)

    @pytest.mark.asyncio
    async def test_remember_batch_creates_span(self):
        """remember_batch() creates a logfire span with namespace_id and batch_size."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)
        lake._engine.remember_batch = AsyncMock(
            return_value=BatchResult(
                total=3,
                processed=3,
                skipped=0,
                failed=0,
                chunks=15,
                entities=9,
                relationships=6,
            )
        )

        docs = [{"content": "Doc 1"}, {"content": "Doc 2"}, {"content": "Doc 3"}]

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.memory_lake.logfire_span") as mock_span_fn,
        ):
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.remember_batch(docs)

        mock_span_fn.assert_called_once()
        call_args = mock_span_fn.call_args
        assert call_args[0][0] == "khora.remember_batch"
        assert call_args[1]["namespace_id"] == str(ns_id)
        assert call_args[1]["batch_size"] == 3


class TestSpanAttributeWhitelist:
    """Tests that no raw content strings leak into span attributes."""

    @pytest.mark.asyncio
    async def test_remember_does_not_leak_content(self):
        """remember() span attributes contain content_length, not the content itself."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)
        lake._engine.remember = AsyncMock(
            return_value=RememberResult(
                document_id=uuid4(),
                namespace_id=ns_id,
                chunks_created=1,
                entities_extracted=0,
                relationships_created=0,
            )
        )

        secret_content = "This is secret private user data that should not appear in spans"

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.memory_lake.logfire_span") as mock_span_fn,
        ):
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.remember(secret_content)

        call_args = mock_span_fn.call_args
        # The attributes should contain content_length (integer), not the content itself
        all_attr_values = list(call_args[1].values())
        for val in all_attr_values:
            if isinstance(val, str):
                assert secret_content not in val, "Raw content leaked into span attributes"

    @pytest.mark.asyncio
    async def test_recall_does_not_leak_query(self):
        """recall() span attributes contain query_length, not the query text."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)
        lake._engine.recall = AsyncMock(
            return_value=RecallResult(
                query="secret query",
                namespace_id=ns_id,
                chunks=[],
                entities=[],
                context_text="",
            )
        )

        secret_query = "What are the admin credentials for the production database?"

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.memory_lake.logfire_span") as mock_span_fn,
        ):
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.recall(secret_query)

        call_args = mock_span_fn.call_args
        all_attr_values = list(call_args[1].values())
        for val in all_attr_values:
            if isinstance(val, str):
                assert secret_query not in val, "Raw query leaked into span attributes"


# =========================================================================
# 5. Coordinator _record_storage_op logfire bridging
# =========================================================================


class TestCoordinatorLogfireBridging:
    """Tests that _record_storage_op in coordinator emits logfire spans."""

    @pytest.mark.asyncio
    async def test_record_storage_op_emits_logfire_span(self):
        """_record_storage_op decorator emits a logfire span when logfire is present."""
        from khora.storage.coordinator import _record_storage_op

        mock_logfire, mock_span = _make_mock_logfire_span()

        @_record_storage_op("test_create", "postgresql")
        async def create_thing():
            return "created"

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            result = await create_thing()

        assert result == "created"
        mock_logfire.span.assert_called_once_with("khora.storage.test_create", backend="postgresql")
        mock_span.set_attribute.assert_any_call("status", "success")
        latency_calls = [c for c in mock_span.set_attribute.call_args_list if c[0][0] == "latency_ms"]
        assert len(latency_calls) == 1
        assert latency_calls[0][0][1] > 0

    @pytest.mark.asyncio
    async def test_record_storage_op_works_without_logfire(self):
        """_record_storage_op works normally when logfire is absent."""
        from khora.storage.coordinator import _record_storage_op

        @_record_storage_op("test_op", "pgvector")
        async def do_thing():
            return [1, 2, 3]

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            result = await do_thing()

        assert result == [1, 2, 3]
