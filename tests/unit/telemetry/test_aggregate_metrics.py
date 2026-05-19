"""Tests for the Phase-3 aggregate OTel metrics.

Covers the five service-wide metrics added by ``aggregate_metrics.py``:
    - khora.memory.recall.duration
    - khora.memory.ingest.duration
    - khora.llm.tokens
    - khora.llm.cost_usd
    - khora.log.queue.depth

Hermetic — no infra dependencies. Patches ``aggregate_metrics`` module-
level instruments with a recording fake so we can assert observations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.khora import Khora, RecallResult
from khora.telemetry import aggregate_metrics


class _RecordingHistogram:
    def __init__(self) -> None:
        self.observations: list[tuple[float, dict[str, Any]]] = []

    def record(self, value: float, attributes: Any = None) -> None:
        self.observations.append((value, dict(attributes or {})))


class _RecordingCounter:
    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


@pytest.fixture
def recall_histogram() -> _RecordingHistogram:
    h = _RecordingHistogram()
    aggregate_metrics._recall_histogram = h
    yield h
    aggregate_metrics._recall_histogram = None


@pytest.fixture
def ingest_histogram() -> _RecordingHistogram:
    h = _RecordingHistogram()
    aggregate_metrics._ingest_histogram = h
    yield h
    aggregate_metrics._ingest_histogram = None


@pytest.fixture
def llm_tokens_counter() -> _RecordingCounter:
    c = _RecordingCounter()
    aggregate_metrics._llm_tokens_counter = c
    yield c
    aggregate_metrics._llm_tokens_counter = None


@pytest.fixture
def llm_cost_counter() -> _RecordingCounter:
    c = _RecordingCounter()
    aggregate_metrics._llm_cost_counter = c
    yield c
    aggregate_metrics._llm_cost_counter = None


# ---------------------------------------------------------------------------
# Helpers cribbed from test_logfire_integration.py to build a minimal kb
# ---------------------------------------------------------------------------


_RESOLVE_ROW_ID = uuid4()


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
    eng = MagicMock()
    eng._storage = MagicMock()
    eng._storage.resolve_namespace = AsyncMock(return_value=_RESOLVE_ROW_ID)
    eng.recall = AsyncMock()
    eng.remember = AsyncMock()
    eng.remember_batch = AsyncMock()
    return eng


def _make_kb() -> Khora:
    with patch("khora.khora.load_config", return_value=_mock_config()):
        kb = Khora()
    kb._connected = True
    kb._engine = _mock_engine()
    return kb


# ---------------------------------------------------------------------------
# recall duration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_emits_duration_histogram(recall_histogram: _RecordingHistogram) -> None:
    """recall() emits one observation on the recall histogram on success."""
    kb = _make_kb()
    ns_id = uuid4()
    kb._engine.recall = AsyncMock(
        return_value=RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[],
            chunks=[],
            entities=[],
            relationships=[],
        )
    )

    await kb.recall("q", namespace=ns_id)

    assert len(recall_histogram.observations) == 1
    value, attrs = recall_histogram.observations[0]
    assert value >= 0.0
    assert attrs["status"] == "success"
    assert attrs["engine"] == kb._engine_name
    assert "mode" in attrs


@pytest.mark.asyncio
async def test_recall_error_path_emits_status_error(
    recall_histogram: _RecordingHistogram,
) -> None:
    """Engine error must still emit the histogram with status=error."""
    kb = _make_kb()
    ns_id = uuid4()
    kb._engine.recall = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await kb.recall("q", namespace=ns_id)

    assert len(recall_histogram.observations) == 1
    _, attrs = recall_histogram.observations[0]
    assert attrs["status"] == "error"


# ---------------------------------------------------------------------------
# LLM tokens / cost via record_llm_call
# ---------------------------------------------------------------------------


def test_record_llm_call_increments_tokens_split_by_kind(
    llm_tokens_counter: _RecordingCounter,
    llm_cost_counter: _RecordingCounter,
) -> None:
    """record_llm_call(prompt=10, completion=20) → two adds totalling 30."""
    from khora.telemetry.noop import NoOpCollector

    collector = NoOpCollector()
    collector.record_llm_call(
        operation="entity_extraction",
        model="gpt-4o-mini",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        latency_ms=5.0,
    )

    kinds = sorted(a[1]["kind"] for a in llm_tokens_counter.adds)
    assert kinds == ["completion", "prompt"]
    by_kind = {a[1]["kind"]: a[0] for a in llm_tokens_counter.adds}
    assert by_kind["prompt"] == 10
    assert by_kind["completion"] == 20
    assert sum(a[0] for a in llm_tokens_counter.adds) == 30
    # No cost emitted because cost_usd was not provided.
    assert llm_cost_counter.adds == []


def test_record_llm_call_emits_cost_when_provided(
    llm_tokens_counter: _RecordingCounter,
    llm_cost_counter: _RecordingCounter,
) -> None:
    from khora.telemetry.noop import NoOpCollector

    NoOpCollector().record_llm_call(
        operation="completion",
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.0125,
    )

    assert len(llm_cost_counter.adds) == 1
    value, attrs = llm_cost_counter.adds[0]
    assert value == pytest.approx(0.0125)
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["operation"] == "completion"
    assert attrs["status"] == "success"


# ---------------------------------------------------------------------------
# Log queue depth gauge
# ---------------------------------------------------------------------------


def test_log_handler_error_counter_increments() -> None:
    start = aggregate_metrics._log_handler_error_count
    aggregate_metrics._increment_log_handler_errors()
    aggregate_metrics._increment_log_handler_errors()
    assert aggregate_metrics._log_handler_error_count == start + 2


def test_register_log_queue_depth_gauge_idempotent() -> None:
    """Calling twice must not double-register."""
    aggregate_metrics._log_queue_gauge_registered = False
    aggregate_metrics.register_log_queue_depth_gauge()
    first = aggregate_metrics._log_queue_gauge_registered
    aggregate_metrics.register_log_queue_depth_gauge()
    assert first is True
    assert aggregate_metrics._log_queue_gauge_registered is True
