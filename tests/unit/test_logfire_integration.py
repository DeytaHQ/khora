"""Tests for optional Logfire/OTEL integration.

Validates that:
- trace_span() works as a no-op when logfire is absent
- trace_span() emits real spans when logfire is present (mocked)
- Decorated functions (instrument_llm, instrument_storage, pipeline_stage)
  emit logfire spans with correct names and attributes
- Khora.remember/recall/forget/remember_batch create top-level spans
- No content strings leak into span attributes
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.khora import BatchResult, Khora, RecallResult, RememberResult
from khora.telemetry import NoOpCollector
from khora.telemetry.instrument import instrument_llm, instrument_storage, pipeline_stage
from khora.telemetry.logfire_integration import LogfireSpan, NoOpSpan, trace_span

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


_RESOLVE_ROW_ID = uuid4()


def _mock_engine() -> MagicMock:
    mock_eng = MagicMock()
    mock_eng._storage = MagicMock()
    mock_eng._storage.resolve_namespace = AsyncMock(return_value=_RESOLVE_ROW_ID)
    mock_eng._embedder = MagicMock()
    mock_eng.connect = AsyncMock()
    mock_eng.disconnect = AsyncMock()
    mock_eng.health_check = AsyncMock(return_value={"status": "healthy"})
    mock_eng.remember = AsyncMock()
    mock_eng.recall = AsyncMock()
    mock_eng.forget = AsyncMock()
    mock_eng.remember_batch = AsyncMock()
    mock_eng.create_namespace = AsyncMock()
    mock_eng.get_namespace = AsyncMock()
    mock_eng.get_entity = AsyncMock()
    mock_eng.list_entities = AsyncMock(return_value=[])
    mock_eng.find_related_entities = AsyncMock(return_value=[])
    mock_eng.get_document = AsyncMock()
    mock_eng.list_documents = AsyncMock(return_value=[])
    mock_eng.search_entities = AsyncMock(return_value=[])
    mock_eng.stats = AsyncMock()
    return mock_eng


def _make_lake(*, connected: bool = False) -> Khora:
    with patch("khora.khora.load_config", return_value=_mock_config()):
        lake = Khora()
    if connected:
        lake._connected = True
        lake._engine = _mock_engine()
    return lake


def _make_mock_trace_span():
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
# 1. trace_span helper tests
# =========================================================================


class TestLogfireSpanNoLogfire:
    """Tests for trace_span when logfire is not installed."""

    def test_yields_noop_span_when_absent(self):
        """trace_span yields a NoOpSpan when _HAS_LOGFIRE is False."""
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with trace_span("test.span", key="value") as span:
                assert isinstance(span, NoOpSpan)

    def test_context_manager_works_when_absent(self):
        """trace_span context manager enters and exits cleanly."""
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            result = "untouched"
            with trace_span("test.span") as span:
                result = "executed"
            assert result == "executed"
            assert isinstance(span, NoOpSpan)

    def test_no_errors_with_attributes_when_absent(self):
        """Passing attributes when logfire is absent causes no errors."""
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with trace_span("test.span", foo="bar", count=42) as span:
                assert isinstance(span, NoOpSpan)

    def test_noop_span_set_attribute_is_silent(self):
        """NoOpSpan.set_attribute does nothing without raising."""
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with trace_span("test.span") as span:
                span.set_attribute("key", "value")
                span.set_attribute("count", 42)

    def test_noop_span_set_attributes_is_silent(self):
        """NoOpSpan.set_attributes does nothing without raising."""
        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with trace_span("test.span") as span:
                span.set_attributes({"key": "value", "count": "42"})


class TestLogfireSpanWithLogfire:
    """Tests for trace_span when logfire is installed (mocked)."""

    def test_creates_span_with_correct_name(self):
        """trace_span creates a LogfireSpan wrapping the real span."""
        mock_logfire, mock_span = _make_mock_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            with trace_span("khora.test.operation") as span:
                assert isinstance(span, LogfireSpan)
                assert span._inner is mock_span

        mock_logfire.span.assert_called_once_with("khora.test.operation")

    def test_passes_attributes_to_span(self):
        """trace_span passes keyword attributes to logfire.span()."""
        mock_logfire, _ = _make_mock_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            with trace_span("khora.test.op", backend="neo4j", count=5):
                pass

        mock_logfire.span.assert_called_once_with("khora.test.op", backend="neo4j", count=5)

    def test_span_set_attribute_works(self):
        """set_attribute on the yielded span works when logfire is present."""
        mock_logfire, mock_span = _make_mock_trace_span()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            with trace_span("khora.test.op") as span:
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
    async def test_emits_trace_span_when_present(self, recording_collector):
        """instrument_llm emits a logfire span with correct name."""
        mock_logfire, mock_span = _make_mock_trace_span()

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
    async def test_emits_trace_span_when_present(self, recording_collector):
        """instrument_storage emits a logfire span with correct name and attributes."""
        mock_logfire, mock_span = _make_mock_trace_span()

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
        mock_logfire, _ = _make_mock_trace_span()

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
    async def test_emits_trace_span_when_present(self, recording_collector):
        """pipeline_stage emits a logfire span with correct name and attributes."""
        mock_logfire, mock_span = _make_mock_trace_span()
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
            with trace_span("test") as span:
                assert isinstance(span, NoOpSpan)

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


class TestKhoraSpans:
    """Tests that Khora methods create correct top-level logfire spans."""

    @pytest.mark.asyncio
    async def test_remember_creates_span(self):
        """remember() creates a logfire span with namespace_id and content_length."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.remember = AsyncMock(
            return_value=RememberResult(
                document_id=uuid4(),
                namespace_id=ns_id,
                chunks_created=3,
                entities_extracted=2,
                relationships_created=1,
            )
        )

        mock_logfire, _ = _make_mock_trace_span()

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.khora.trace_span") as mock_span_fn,
        ):
            # Use a real context manager that yields None so the code runs
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                mock_span_fn._last_name = name
                mock_span_fn._last_attrs = attrs
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.remember(
                "Hello, this is test content",
                namespace=ns_id,
                title="Test",
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        mock_span_fn.assert_called_once()
        call_args = mock_span_fn.call_args
        assert call_args[0][0] == "khora.remember"
        assert call_args[1]["namespace_id"] == str(_RESOLVE_ROW_ID)
        assert call_args[1]["content_length"] == len("Hello, this is test content")

    @pytest.mark.asyncio
    async def test_recall_creates_span(self):
        """recall() creates a bounded logfire span — query_hash + query_length, no raw query."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
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
            patch("khora.khora.trace_span") as mock_span_fn,
        ):
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.recall("test query", namespace=ns_id)

        mock_span_fn.assert_called_once()
        call_args = mock_span_fn.call_args
        assert call_args[0][0] == "khora.recall"
        assert call_args[1]["namespace_id"] == str(_RESOLVE_ROW_ID)
        assert "query" not in call_args[1], "raw query must not be a span attribute (cardinality bomb)"
        assert call_args[1]["query_length"] == len("test query")
        assert isinstance(call_args[1]["query_hash"], str)
        assert len(call_args[1]["query_hash"]) == 8

    @pytest.mark.asyncio
    async def test_forget_creates_span(self):
        """forget() creates a logfire span with document_id."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()
        ns_id = uuid4()
        lake._engine.forget = AsyncMock(return_value=True)

        with patch("khora.khora.trace_span") as mock_span_fn:
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.forget(doc_id, namespace=ns_id)

        mock_span_fn.assert_called_once()
        call_args = mock_span_fn.call_args
        assert call_args[0][0] == "khora.forget"
        assert call_args[1]["document_id"] == str(doc_id)

    @pytest.mark.asyncio
    async def test_remember_batch_creates_span(self):
        """remember_batch() creates a logfire span with namespace_id and batch_size."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
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
            patch("khora.khora.trace_span") as mock_span_fn,
        ):
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.remember_batch(
                docs,
                namespace=ns_id,
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        mock_span_fn.assert_called_once()
        call_args = mock_span_fn.call_args
        assert call_args[0][0] == "khora.remember_batch"
        assert call_args[1]["namespace_id"] == str(_RESOLVE_ROW_ID)
        assert call_args[1]["batch_size"] == 3


class TestSpanAttributeWhitelist:
    """Tests that no raw content strings leak into span attributes."""

    @pytest.mark.asyncio
    async def test_remember_does_not_leak_content(self):
        """remember() span attributes contain content_length, not the content itself."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
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
            patch("khora.khora.trace_span") as mock_span_fn,
        ):
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.remember(
                secret_content,
                namespace=ns_id,
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        call_args = mock_span_fn.call_args
        # The attributes should contain content_length (integer), not the content itself
        all_attr_values = list(call_args[1].values())
        for val in all_attr_values:
            if isinstance(val, str):
                assert secret_content not in val, "Raw content leaked into span attributes"

    @pytest.mark.asyncio
    async def test_recall_does_not_leak_query(self):
        """recall() span attributes record query_hash + query_length, never the raw query."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        secret_query = "what was the password the CEO mentioned in slack yesterday"
        lake._engine.recall = AsyncMock(
            return_value=RecallResult(
                query=secret_query,
                namespace_id=ns_id,
                chunks=[],
                entities=[],
                context_text="",
            )
        )

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.khora.trace_span") as mock_span_fn,
        ):
            from contextlib import contextmanager

            @contextmanager
            def tracking_span(name, **attrs):
                yield None

            mock_span_fn.side_effect = tracking_span

            await lake.recall(secret_query, namespace=ns_id)

        call_args = mock_span_fn.call_args
        for val in call_args[1].values():
            if isinstance(val, str):
                assert secret_query not in val, "raw query leaked into span attributes"
        assert call_args[1]["query_length"] == len(secret_query)
        assert len(call_args[1]["query_hash"]) == 8


# =========================================================================
# 5. Coordinator _record_storage_op logfire bridging
# =========================================================================


class TestCoordinatorLogfireBridging:
    """Tests that _record_storage_op in coordinator emits logfire spans."""

    @pytest.mark.asyncio
    async def test_record_storage_op_emits_trace_span(self):
        """_record_storage_op decorator emits a logfire span when logfire is present."""
        from khora.storage.coordinator import _record_storage_op

        mock_logfire, mock_span = _make_mock_trace_span()

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


# =========================================================================
# 6. Exception path tests
# =========================================================================


class TestInstrumentLLMExceptionPath:
    """Tests that instrument_llm handles exceptions correctly."""

    @pytest.mark.asyncio
    async def test_exception_is_reraised(self, recording_collector):
        """instrument_llm re-raises exceptions from the decorated function."""

        @instrument_llm("failing_op")
        async def failing_llm_call():
            raise ValueError("LLM call failed")

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with pytest.raises(ValueError, match="LLM call failed"):
                await failing_llm_call()

    @pytest.mark.asyncio
    async def test_collector_records_error_status(self, recording_collector):
        """Custom telemetry collector records the call with status='error'."""

        @instrument_llm("failing_op")
        async def failing_llm_call():
            raise ValueError("LLM call failed")

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with pytest.raises(ValueError):
                await failing_llm_call()

        assert len(recording_collector.llm_calls) == 1
        assert recording_collector.llm_calls[0]["status"] == "error"
        assert recording_collector.llm_calls[0]["operation"] == "failing_op"
        assert "LLM call failed" in recording_collector.llm_calls[0]["error_message"]

    @pytest.mark.asyncio
    async def test_trace_span_created_on_exception(self, recording_collector):
        """When logfire is present, a span is still created even if function raises."""
        mock_logfire, mock_span = _make_mock_trace_span()

        @instrument_llm("failing_op")
        async def failing_llm_call():
            raise ValueError("LLM call failed")

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            with pytest.raises(ValueError):
                await failing_llm_call()

        mock_logfire.span.assert_called_once_with("khora.llm.failing_op")


class TestInstrumentStorageExceptionPath:
    """Tests that instrument_storage handles exceptions correctly."""

    @pytest.mark.asyncio
    async def test_exception_is_reraised(self, recording_collector):
        """instrument_storage re-raises exceptions from the decorated function."""

        @instrument_storage("neo4j", "create_entity")
        async def failing_storage_op():
            raise RuntimeError("Connection lost")

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with pytest.raises(RuntimeError, match="Connection lost"):
                await failing_storage_op()

    @pytest.mark.asyncio
    async def test_collector_records_error_status(self, recording_collector):
        """collector.record_storage_op is called with status='error'."""

        @instrument_storage("neo4j", "create_entity")
        async def failing_storage_op():
            raise RuntimeError("Connection lost")

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with pytest.raises(RuntimeError):
                await failing_storage_op()

        assert len(recording_collector.storage_calls) == 1
        assert recording_collector.storage_calls[0]["status"] == "error"
        assert recording_collector.storage_calls[0]["backend"] == "neo4j"
        assert recording_collector.storage_calls[0]["operation"] == "create_entity"
        assert "Connection lost" in recording_collector.storage_calls[0]["error_message"]

    @pytest.mark.asyncio
    async def test_trace_span_created_on_exception(self, recording_collector):
        """When logfire is present, a span is still created even if function raises."""
        mock_logfire, mock_span = _make_mock_trace_span()

        @instrument_storage("pgvector", "search_similar")
        async def failing_storage_op():
            raise RuntimeError("Connection lost")

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            with pytest.raises(RuntimeError):
                await failing_storage_op()

        mock_logfire.span.assert_called_once_with("khora.storage.search_similar", backend="pgvector")


class TestPipelineStageExceptionPath:
    """Tests that pipeline_stage handles exceptions correctly."""

    @pytest.mark.asyncio
    async def test_exception_is_reraised(self, recording_collector):
        """pipeline_stage re-raises exceptions from the context body."""
        run_id = uuid4()

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with pytest.raises(ValueError, match="Pipeline failed"):
                async with pipeline_stage("ingestion", "chunking", run_id, input_count=5):
                    raise ValueError("Pipeline failed")

    @pytest.mark.asyncio
    async def test_collector_records_error_status(self, recording_collector):
        """collector.record_pipeline_stage is called with status='error'."""
        run_id = uuid4()

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with pytest.raises(ValueError):
                async with pipeline_stage("ingestion", "chunking", run_id, input_count=5):
                    raise ValueError("Pipeline failed")

        assert len(recording_collector.pipeline_calls) == 1
        assert recording_collector.pipeline_calls[0]["status"] == "error"
        assert recording_collector.pipeline_calls[0]["pipeline"] == "ingestion"
        assert recording_collector.pipeline_calls[0]["stage"] == "chunking"
        assert "Pipeline failed" in recording_collector.pipeline_calls[0]["error_message"]

    @pytest.mark.asyncio
    async def test_trace_span_created_on_exception(self, recording_collector):
        """When logfire is present, a span is still created even if body raises."""
        mock_logfire, mock_span = _make_mock_trace_span()
        run_id = uuid4()

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            with pytest.raises(ValueError):
                async with pipeline_stage("ingestion", "chunking", run_id, input_count=5):
                    raise ValueError("Pipeline failed")

        mock_logfire.span.assert_called_once_with("khora.ingestion.chunking", input_count=5)


class TestCoordinatorRecordStorageOpExceptionPath:
    """Tests that _record_storage_op handles exceptions correctly."""

    @pytest.mark.asyncio
    async def test_exception_is_reraised(self):
        """_record_storage_op re-raises exceptions from the decorated function."""
        from khora.storage.coordinator import _record_storage_op

        @_record_storage_op("test_create", "postgresql")
        async def failing_op():
            raise RuntimeError("DB error")

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with pytest.raises(RuntimeError, match="DB error"):
                await failing_op()

    @pytest.mark.asyncio
    async def test_collector_records_error_status(self, recording_collector):
        """collector.record_storage_op is called with status='error'."""
        from khora.storage.coordinator import _record_storage_op

        @_record_storage_op("test_create", "postgresql")
        async def failing_op():
            raise RuntimeError("DB error")

        with patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", False):
            with pytest.raises(RuntimeError):
                await failing_op()

        assert len(recording_collector.storage_calls) == 1
        assert recording_collector.storage_calls[0]["status"] == "error"
        assert recording_collector.storage_calls[0]["operation"] == "test_create"
        assert recording_collector.storage_calls[0]["backend"] == "postgresql"


# =========================================================================
# 7. Deep tracing span tests (query pipeline, engines, neo4j)
# =========================================================================


class TestQueryEngineDeepTracing:
    """Tests that query engine methods create correct trace spans."""

    @pytest.mark.asyncio
    async def test_vector_search_creates_span(self):
        """_vector_search creates a khora.query.vector_search span."""
        mock_logfire, mock_span = _make_mock_trace_span()
        ns_id = uuid4()

        from khora.query.engine import HybridQueryEngine

        engine = object.__new__(HybridQueryEngine)
        engine._storage = MagicMock()
        engine._storage.search_similar_chunks = AsyncMock(return_value=[])
        engine._storage.get_entities_batch = AsyncMock(return_value={})
        engine._entity_search_cache = {}

        async def _cached(ns, emb, *, limit, min_similarity):
            return []

        engine._cached_entity_search = _cached

        from khora.query import QueryConfig

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            result = await engine._vector_search(ns_id, [0.1] * 10, QueryConfig())

        assert result["source"] == "vector"
        mock_logfire.span.assert_called_once_with("khora.query.vector_search", namespace_id=str(ns_id))
        mock_span.set_attribute.assert_any_call("chunk_count", 0)
        mock_span.set_attribute.assert_any_call("entity_count", 0)

    @pytest.mark.asyncio
    async def test_keyword_search_bm25_creates_span(self):
        """_keyword_search_bm25 creates a khora.query.keyword_search_bm25 span."""
        mock_logfire, mock_span = _make_mock_trace_span()
        ns_id = uuid4()

        from khora.query.engine import HybridQueryEngine

        engine = object.__new__(HybridQueryEngine)
        engine._keyword_searchers = {}
        engine._storage = MagicMock()
        engine._storage.list_chunks = AsyncMock(return_value=[])

        from khora.query import QueryConfig

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            result = await engine._keyword_search_bm25(ns_id, "test query", QueryConfig())

        assert result["source"] == "keyword"
        mock_logfire.span.assert_called_once_with("khora.query.keyword_search_bm25", namespace_id=str(ns_id))
        mock_span.set_attribute.assert_any_call("result_count", 0)

    @pytest.mark.asyncio
    async def test_find_related_entities_creates_span(self):
        """find_related_entities creates a khora.query.find_related_entities span."""
        mock_logfire, mock_span = _make_mock_trace_span()
        entity_id = uuid4()
        ns_id = uuid4()

        from khora.query.engine import HybridQueryEngine

        engine = object.__new__(HybridQueryEngine)
        engine._storage = MagicMock()
        engine._storage.get_neighborhood = AsyncMock(return_value={"entities": [], "relationships": []})

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            result = await engine.find_related_entities(entity_id, ns_id)

        assert result == []
        mock_logfire.span.assert_called_once_with(
            "khora.query.find_related_entities",
            entity_id=str(entity_id),
            max_depth=2,
        )
        mock_span.set_attribute.assert_any_call("result_count", 0)

    @pytest.mark.asyncio
    async def test_stage4_rerank_creates_span(self):
        """_stage4_rerank creates a khora.query.rerank span."""
        mock_logfire, mock_span = _make_mock_trace_span()

        from khora.query.engine import HybridQueryEngine

        engine = object.__new__(HybridQueryEngine)
        engine._llm_config = MagicMock()

        mock_chunk = MagicMock()
        mock_chunk.content = "test"
        mock_chunk.metadata = {}
        chunks = [(mock_chunk, 0.9), (mock_chunk, 0.8), (mock_chunk, 0.7)]

        mock_reranker = MagicMock()
        mock_reranked = [MagicMock(item=mock_chunk, final_score=0.95)]
        mock_reranker.rerank = AsyncMock(return_value=mock_reranked)
        engine._rerankers = {"cross_encoder": mock_reranker}

        from khora.query import QueryConfig

        config = QueryConfig()
        config.enable_reranking = True

        with (
            patch("khora.telemetry.logfire_integration._HAS_LOGFIRE", True),
            patch("khora.telemetry.logfire_integration._logfire", mock_logfire),
        ):
            result = await engine._stage4_rerank(chunks, "test query", config)

        assert len(result) == 1
        mock_logfire.span.assert_called_once_with("khora.query.rerank", candidate_count=3)
        mock_span.set_attribute.assert_any_call("reranked_count", 1)
