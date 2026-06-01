"""Tests for telemetry collector and initialization."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.telemetry import NoOpCollector, TelemetryCollector, get_collector, init_telemetry, shutdown_telemetry
from khora.telemetry.config import TelemetryConfig
from khora.telemetry.models import LLMEvent, PipelineEvent, StorageEvent

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_llm_event_defaults(self):
        event = LLMEvent()
        assert event.service_name == "khora"
        assert event.status == "success"
        assert event.prompt_tokens == 0

    def test_storage_event_defaults(self):
        event = StorageEvent()
        assert event.backend == ""
        assert event.record_count == 0

    def test_pipeline_event_defaults(self):
        event = PipelineEvent()
        assert event.pipeline == ""
        assert event.run_id is None

    def test_llm_event_with_values(self):
        event = LLMEvent(
            operation="embedding",
            model="text-embedding-3-small",
            prompt_tokens=100,
            total_tokens=100,
            latency_ms=50.0,
        )
        assert event.model == "text-embedding-3-small"
        assert event.prompt_tokens == 100


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestTelemetryConfig:
    def test_from_env_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            cfg = TelemetryConfig.from_env()
            assert cfg.database_url is None
            assert cfg.service_name == "khora"

    def test_from_env_with_url(self):
        with patch.dict("os.environ", {"KHORA_TELEMETRY_DATABASE_URL": "postgresql://localhost/telemetry"}):
            cfg = TelemetryConfig.from_env()
            # database_url is SecretStr — unwrap to compare plaintext.
            assert cfg.database_url.get_secret_value() == "postgresql://localhost/telemetry"

    def test_from_env_custom_service(self):
        with patch.dict("os.environ", {"KHORA_TELEMETRY_SERVICE_NAME": "test-service"}):
            cfg = TelemetryConfig.from_env()
            assert cfg.service_name == "test-service"


# ---------------------------------------------------------------------------
# NoOpCollector
# ---------------------------------------------------------------------------


class TestNoOpCollector:
    @pytest.mark.asyncio
    async def test_noop_start_shutdown(self):
        collector = NoOpCollector()
        await collector.start()
        await collector.shutdown()

    def test_noop_record_methods(self):
        collector = NoOpCollector()
        collector.record_llm_call(operation="test", model="gpt-4o")
        collector.record_storage_op(backend="postgresql", operation="create_document")
        collector.record_pipeline_stage(pipeline="ingestion", stage="chunking")
        # Should not raise


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------


class TestTelemetryCollector:
    def test_record_llm_call_buffers(self):
        engine = MagicMock()
        collector = TelemetryCollector(engine, service_name="test")
        collector.record_llm_call(operation="embedding", model="gpt-4o", prompt_tokens=100)
        assert len(collector._buffer) == 1
        kind, data = collector._buffer[0]
        assert kind == "llm"
        assert data["operation"] == "embedding"
        assert data["model"] == "gpt-4o"

    def test_record_storage_op_buffers(self):
        engine = MagicMock()
        collector = TelemetryCollector(engine, service_name="test")
        collector.record_storage_op(backend="pgvector", operation="search_similar_chunks", record_count=10)
        assert len(collector._buffer) == 1
        kind, data = collector._buffer[0]
        assert kind == "storage"
        assert data["backend"] == "pgvector"
        assert data["record_count"] == 10

    def test_record_pipeline_stage_buffers(self):
        engine = MagicMock()
        collector = TelemetryCollector(engine, service_name="test")
        run_id = uuid4()
        collector.record_pipeline_stage(pipeline="ingestion", stage="chunking", run_id=run_id)
        assert len(collector._buffer) == 1
        kind, data = collector._buffer[0]
        assert kind == "pipeline"
        assert data["pipeline"] == "ingestion"

    def test_service_name_applied(self):
        engine = MagicMock()
        collector = TelemetryCollector(engine, service_name="test-service")
        collector.record_llm_call(operation="chat")
        _, data = collector._buffer[0]
        assert data["service_name"] == "test-service"

    @pytest.mark.asyncio
    async def test_flush_drains_buffer(self):
        engine = MagicMock()
        conn = AsyncMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        engine.begin.return_value = cm

        collector = TelemetryCollector(engine, service_name="test")
        collector.record_llm_call(operation="test")
        collector.record_storage_op(backend="pg", operation="test")
        collector.record_pipeline_stage(pipeline="test", stage="test")

        assert len(collector._buffer) == 3
        await collector._flush()
        assert len(collector._buffer) == 0

    @pytest.mark.asyncio
    async def test_flush_handles_errors_gracefully(self):
        engine = AsyncMock()
        engine.begin.side_effect = Exception("connection failed")

        collector = TelemetryCollector(engine, service_name="test")
        collector.record_llm_call(operation="test")

        # Should not raise
        await collector._flush()
        # Events are retained (re-queued) on a transient error, not dropped (#924).
        assert len(collector._buffer) == 1

    @pytest.mark.asyncio
    async def test_flush_failure_requeues_and_retries(self):
        # First flush fails (transient error), second succeeds. Events recorded
        # before the failed flush must survive and be written on the retry (#924).
        engine = MagicMock()
        engine.begin.side_effect = RuntimeError("transient DB error")

        collector = TelemetryCollector(engine, service_name="test")
        collector.record_storage_op(backend="pg", operation="upsert", record_count=3)
        collector.record_llm_call(operation="extract", model="gpt-4o", prompt_tokens=10)

        before = len(collector._buffer)
        assert before == 2

        from loguru import logger

        warnings: list[str] = []
        sink_id = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
        try:
            await collector._flush()
        finally:
            logger.remove(sink_id)

        # Not lost: the failed batch is back on the buffer for the next tick.
        assert len(collector._buffer) == before
        assert any("re-queued" in w for w in warnings)

        # Next flush succeeds and drains the retained events.
        conn = AsyncMock()
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        engine.begin.side_effect = None
        engine.begin.return_value = cm
        await collector._flush()
        assert len(collector._buffer) == 0

    @pytest.mark.asyncio
    async def test_flush_requeue_drops_oldest_when_buffer_full(self):
        engine = AsyncMock()
        engine.begin.side_effect = RuntimeError("transient DB error")

        collector = TelemetryCollector(engine, service_name="test", max_buffer_size=2)
        collector.record_storage_op(backend="pg", operation="op0")
        collector.record_storage_op(backend="pg", operation="op1")
        collector.record_storage_op(backend="pg", operation="op2")
        assert len(collector._buffer) == 3

        from loguru import logger

        warnings: list[str] = []
        sink_id = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
        try:
            await collector._flush()
        finally:
            logger.remove(sink_id)

        # Capped at max_buffer_size; oldest dropped.
        assert len(collector._buffer) == 2
        ops = [data["operation"] for _, data in collector._buffer]
        assert ops == ["op1", "op2"]
        assert any("buffer full" in w for w in warnings)


# ---------------------------------------------------------------------------
# init_telemetry / shutdown_telemetry
# ---------------------------------------------------------------------------


class TestInitTelemetry:
    @pytest.mark.asyncio
    async def test_init_returns_noop_when_no_url(self):
        cfg = TelemetryConfig(database_url=None)
        collector = await init_telemetry(cfg)
        assert isinstance(collector, NoOpCollector)
        assert isinstance(get_collector(), NoOpCollector)

    @pytest.mark.asyncio
    async def test_init_returns_collector_when_url_set(self):
        cfg = TelemetryConfig(database_url="postgresql://localhost/telemetry")

        with patch("khora.telemetry.session.create_telemetry_engine") as mock_engine_factory:
            mock_engine = MagicMock()
            # Make begin() return an async context manager
            mock_conn = AsyncMock()
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=mock_conn)
            cm.__aexit__ = AsyncMock(return_value=False)
            mock_engine.begin.return_value = cm
            mock_engine.dispose = AsyncMock()
            mock_engine_factory.return_value = mock_engine

            collector = await init_telemetry(cfg)
            assert isinstance(collector, TelemetryCollector)
            assert isinstance(get_collector(), TelemetryCollector)

            # Clean up
            await shutdown_telemetry()
            assert isinstance(get_collector(), NoOpCollector)

    @pytest.mark.asyncio
    async def test_init_from_env_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            collector = await init_telemetry()
            assert isinstance(collector, NoOpCollector)
