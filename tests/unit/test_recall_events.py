"""Tests for RECALL_* event emission from Khora.recall() (Issue #576 Phase 1)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Entity, RecallChunk, RecallEntity
from khora.core.models.event import EventType, MemoryEvent
from khora.khora import RecallResult

from .helpers import make_kb as _make_kb


def _make_kb_for_hooks():
    """make_kb() plus hooks=None on the mock config so the dispatcher gets defaults.

    The shared mock_config returns a MagicMock for every attribute including
    ``hooks``; ``_get_hook_dispatcher()`` then passes that MagicMock through to
    ``HookDispatcher(max_concurrent=<MagicMock>)``, which trips an int compare
    inside ``asyncio.Semaphore``. Setting hooks to None forces the defaults.
    """
    kb = _make_kb(connected=True)
    kb._config.hooks = None
    return kb


def _make_result(ns_id) -> RecallResult:
    """Build a RecallResult stub with one chunk and one entity."""
    chunk = Chunk(namespace_id=ns_id, document_id=uuid4(), content="hello")
    entity = Entity(namespace_id=ns_id, name="Alice", entity_type="PERSON")
    return RecallResult(
        query="search query",
        namespace_id=ns_id,
        documents=[],
        chunks=[
            RecallChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=0.87,
                created_at=chunk.created_at,
            )
        ],
        entities=[
            RecallEntity(
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                description="",
                score=0.75,
                attributes={},
                mention_count=0,
                source_document_ids=[],
                source_chunk_ids=[],
            )
        ],
        relationships=[],
        engine_info={
            "abstention_signals": {"should_abstain": False},
            "llm_usage": {"total_tokens": 42},
        },
    )


@pytest.mark.asyncio
async def test_recall_no_subscribers_no_error() -> None:
    """recall() with no subscribers — no error, dispatcher subscription_count stays 0."""
    kb = _make_kb_for_hooks()
    ns_id = uuid4()
    kb._engine.recall = AsyncMock(return_value=_make_result(ns_id))

    with (
        patch("khora.telemetry.context.ensure_trace_id"),
        patch("khora.telemetry.context.clear_trace_id"),
    ):
        result = await kb.recall("search query", namespace=ns_id)

    assert isinstance(result, RecallResult)
    assert kb._get_hook_dispatcher().subscription_count == 0


@pytest.mark.asyncio
async def test_recall_requested_emitted() -> None:
    """RECALL_REQUESTED fires with query, resource_id, and namespace_id."""
    kb = _make_kb_for_hooks()
    ns_id = uuid4()
    kb._engine.recall = AsyncMock(return_value=_make_result(ns_id))

    received: list[MemoryEvent] = []

    async def on_requested(event: MemoryEvent) -> None:
        received.append(event)

    kb.subscribe(EventType.RECALL_REQUESTED, on_requested)

    with (
        patch("khora.telemetry.context.ensure_trace_id"),
        patch("khora.telemetry.context.clear_trace_id"),
    ):
        await kb.recall("search query", namespace=ns_id, limit=7)

    assert len(received) == 1
    ev = received[0]
    assert ev.event_type == EventType.RECALL_REQUESTED
    assert ev.resource_type == "recall"
    assert isinstance(ev.resource_id, type(uuid4()))
    assert ev.data["query"] == "search query"
    assert ev.data["k"] == 7
    # mode default is SearchMode.HYBRID — Enum(auto()) so .value is its int
    from khora.query import SearchMode

    assert ev.data["mode"] == SearchMode.HYBRID.value
    assert ev.data["namespace_id"] == str(ev.namespace_id)


@pytest.mark.asyncio
async def test_recall_results_ready_payload() -> None:
    """RECALL_RESULTS_READY contains result_count, chunk_ids, entity_ids, abstention_signals."""
    kb = _make_kb_for_hooks()
    ns_id = uuid4()
    stub = _make_result(ns_id)
    kb._engine.recall = AsyncMock(return_value=stub)

    received: list[MemoryEvent] = []

    async def on_ready(event: MemoryEvent) -> None:
        received.append(event)

    kb.subscribe(EventType.RECALL_RESULTS_READY, on_ready)

    with (
        patch("khora.telemetry.context.ensure_trace_id"),
        patch("khora.telemetry.context.clear_trace_id"),
    ):
        await kb.recall("search query", namespace=ns_id)

    assert len(received) == 1
    ev = received[0]
    assert ev.event_type == EventType.RECALL_RESULTS_READY
    assert ev.data["result_count"] == 1
    assert ev.data["top_score"] == 0.87
    assert len(ev.data["chunk_ids"]) == 1
    assert ev.data["chunk_ids"][0] == str(stub.chunks[0].id)
    assert len(ev.data["entity_ids"]) == 1
    assert ev.data["entity_ids"][0] == str(stub.entities[0].id)
    assert ev.data["abstention_signals"] == {"should_abstain": False}


@pytest.mark.asyncio
async def test_recall_completed_has_latency() -> None:
    """RECALL_COMPLETED includes positive latency_ms."""
    kb = _make_kb_for_hooks()
    ns_id = uuid4()
    kb._engine.recall = AsyncMock(return_value=_make_result(ns_id))

    received: list[MemoryEvent] = []

    async def on_completed(event: MemoryEvent) -> None:
        received.append(event)

    kb.subscribe(EventType.RECALL_COMPLETED, on_completed)

    with (
        patch("khora.telemetry.context.ensure_trace_id"),
        patch("khora.telemetry.context.clear_trace_id"),
    ):
        await kb.recall("search query", namespace=ns_id)

    assert len(received) == 1
    ev = received[0]
    assert ev.event_type == EventType.RECALL_COMPLETED
    assert ev.data["query"] == "search query"
    assert ev.data["result_count"] == 1
    assert ev.data["latency_ms"] > 0
    assert ev.data["llm_usage"] == {"total_tokens": 42}
