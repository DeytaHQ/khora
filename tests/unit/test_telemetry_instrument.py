"""Tests for telemetry instrumentation helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import patch
from uuid import uuid4

import pytest

from khora.telemetry import NoOpCollector
from khora.telemetry.instrument import instrument_llm, instrument_storage, pipeline_stage


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


# ---------------------------------------------------------------------------
# @instrument_llm
# ---------------------------------------------------------------------------


class TestInstrumentLLM:
    @pytest.mark.asyncio
    async def test_records_success(self, recording_collector):
        @instrument_llm("test_op")
        async def my_llm_call():
            return "result"

        result = await my_llm_call()
        assert result == "result"
        assert len(recording_collector.llm_calls) == 1
        call = recording_collector.llm_calls[0]
        assert call["operation"] == "test_op"
        assert call["status"] == "success"
        assert call["latency_ms"] > 0
        assert call["error_message"] is None

    @pytest.mark.asyncio
    async def test_records_error(self, recording_collector):
        @instrument_llm("failing_op")
        async def my_llm_call():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await my_llm_call()

        assert len(recording_collector.llm_calls) == 1
        call = recording_collector.llm_calls[0]
        assert call["status"] == "error"
        assert "boom" in call["error_message"]


# ---------------------------------------------------------------------------
# @instrument_storage
# ---------------------------------------------------------------------------


class TestInstrumentStorage:
    @pytest.mark.asyncio
    async def test_records_success_with_list_result(self, recording_collector):
        @instrument_storage("pgvector", "search_similar")
        async def search():
            return [1, 2, 3]

        result = await search()
        assert result == [1, 2, 3]
        assert len(recording_collector.storage_calls) == 1
        call = recording_collector.storage_calls[0]
        assert call["backend"] == "pgvector"
        assert call["operation"] == "search_similar"
        assert call["record_count"] == 3
        assert call["status"] == "success"

    @pytest.mark.asyncio
    async def test_records_single_result(self, recording_collector):
        @instrument_storage("postgresql", "create_document")
        async def create():
            return {"id": 1}

        await create()
        call = recording_collector.storage_calls[0]
        assert call["record_count"] == 1

    @pytest.mark.asyncio
    async def test_records_error(self, recording_collector):
        @instrument_storage("neo4j", "create_entity")
        async def failing():
            raise ConnectionError("db down")

        with pytest.raises(ConnectionError):
            await failing()

        call = recording_collector.storage_calls[0]
        assert call["status"] == "error"
        assert "db down" in call["error_message"]


# ---------------------------------------------------------------------------
# pipeline_stage context manager
# ---------------------------------------------------------------------------


class TestPipelineStage:
    @pytest.mark.asyncio
    async def test_records_success(self, recording_collector):
        run_id = uuid4()
        ns_id = uuid4()
        async with pipeline_stage("ingestion", "chunking", run_id, namespace_id=ns_id):
            await asyncio.sleep(0.001)

        assert len(recording_collector.pipeline_calls) == 1
        call = recording_collector.pipeline_calls[0]
        assert call["pipeline"] == "ingestion"
        assert call["stage"] == "chunking"
        assert call["run_id"] == run_id
        assert call["namespace_id"] == ns_id
        assert call["status"] == "success"
        assert call["latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_records_error(self, recording_collector):
        with pytest.raises(RuntimeError, match="stage failed"):
            async with pipeline_stage("query", "understanding", uuid4()):
                raise RuntimeError("stage failed")

        call = recording_collector.pipeline_calls[0]
        assert call["status"] == "error"
        assert "stage failed" in call["error_message"]

    @pytest.mark.asyncio
    async def test_extra_metadata(self, recording_collector):
        async with pipeline_stage("ingestion", "embedding", uuid4(), extra_metadata={"chunk_count": 42}):
            pass

        call = recording_collector.pipeline_calls[0]
        assert call["metadata"] == {"chunk_count": 42}
