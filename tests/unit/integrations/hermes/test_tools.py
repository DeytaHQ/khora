"""Unit tests for the Hermes ``_tools`` module and ``format_memory_context``.

These tests run without any real Khora, database, or hermes-agent
plumbing — the dispatch path is exercised with a mocked ``Khora.recall``
returning a ``RecallResult``-shaped stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from khora.integrations.hermes._mapping import format_memory_context
from khora.integrations.hermes._tools import (
    dispatch_memory_recall,
    dispatch_memory_search,
    memory_recall_schema,
    memory_search_schema,
)

# ---------------------------------------------------------------------------
# Lightweight shapes — duck-typed against RecallChunk / RecallEntity /
# RecallResult so we don't depend on the real dataclasses' field ordering.
# ---------------------------------------------------------------------------


@dataclass
class _Chunk:
    content: str
    score: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime(2026, 3, 14, tzinfo=UTC))


@dataclass
class _Entity:
    name: str
    entity_type: str
    mention_count: int


@dataclass
class _RecallResultStub:
    chunks: list[Any] = field(default_factory=list)
    entities: list[Any] = field(default_factory=list)
    documents: list[Any] = field(default_factory=list)
    relationships: list[Any] = field(default_factory=list)
    query: str = ""
    namespace_id: UUID = field(default_factory=uuid4)
    engine_info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_memory_search_schema_shape() -> None:
    schema = memory_search_schema()
    assert schema["name"] == "memory_search"
    assert isinstance(schema["description"], str) and schema["description"]
    props = schema["input_schema"]["properties"]
    assert {"query", "top_k", "min_similarity"} <= set(props)
    for name in props:
        assert isinstance(props[name]["description"], str) and props[name]["description"]
    # `required` must reference real properties.
    for required_name in schema["input_schema"]["required"]:
        assert required_name in props


@pytest.mark.unit
def test_memory_recall_schema_shape() -> None:
    schema = memory_recall_schema()
    assert schema["name"] == "memory_recall"
    props = schema["input_schema"]["properties"]
    # Inherits base props and adds the temporal ones.
    assert {"query", "top_k", "min_similarity", "before", "after"} <= set(props)
    for required_name in schema["input_schema"]["required"]:
        assert required_name in props
    # before/after carry per-arg descriptions optimised for LLM selection.
    assert "ISO 8601" in props["before"]["description"]
    assert "ISO 8601" in props["after"]["description"]


# ---------------------------------------------------------------------------
# dispatch_memory_search — validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_memory_search_rejects_empty_query() -> None:
    kb = object()  # never touched — validation fires first
    with pytest.raises(ValueError, match="query"):
        await dispatch_memory_search(kb, uuid4(), {"query": "   "})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_memory_search_rejects_top_k_above_max() -> None:
    kb = object()
    with pytest.raises(ValueError, match="top_k"):
        await dispatch_memory_search(kb, uuid4(), {"query": "hi", "top_k": 51})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_memory_search_rejects_min_similarity_out_of_range() -> None:
    kb = object()
    with pytest.raises(ValueError, match="min_similarity"):
        await dispatch_memory_search(kb, uuid4(), {"query": "hi", "min_similarity": 1.5})


# ---------------------------------------------------------------------------
# dispatch_memory_search — happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_memory_search_formats_chunks() -> None:
    chunks = [
        _Chunk(content="user mentioned Alice", score=0.87, created_at=datetime(2026, 3, 14, tzinfo=UTC)),
        _Chunk(content="user asked about Phoenix", score=0.81, created_at=datetime(2026, 3, 12, tzinfo=UTC)),
    ]
    kb = type("KhoraStub", (), {})()
    kb.recall = AsyncMock(return_value=_RecallResultStub(chunks=chunks))

    out = await dispatch_memory_search(kb, uuid4(), {"query": "Alice"})

    kb.recall.assert_awaited_once()
    call_kwargs = kb.recall.await_args.kwargs
    assert call_kwargs["limit"] == 10  # default top_k
    assert call_kwargs["min_similarity"] == pytest.approx(0.1)
    assert "start_time" not in call_kwargs  # search is non-temporal

    assert out.startswith("<memory-context>")
    assert out.endswith("</memory-context>")
    assert "[score: 0.87, 2026-03-14]" in out
    assert "user mentioned Alice" in out
    assert "user asked about Phoenix" in out


# ---------------------------------------------------------------------------
# dispatch_memory_recall — temporal bounds
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_memory_recall_passes_temporal_bounds() -> None:
    kb = type("KhoraStub", (), {})()
    kb.recall = AsyncMock(return_value=_RecallResultStub(chunks=[]))

    await dispatch_memory_recall(
        kb,
        uuid4(),
        {"query": "what happened?", "after": "2026-03-01T00:00:00Z", "before": "2026-03-31T23:59:59Z"},
    )

    call_kwargs = kb.recall.await_args.kwargs
    assert call_kwargs["start_time"] == datetime(2026, 3, 1, tzinfo=UTC)
    assert call_kwargs["end_time"] == datetime(2026, 3, 31, 23, 59, 59, tzinfo=UTC)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_memory_recall_rejects_inverted_window() -> None:
    kb = object()
    with pytest.raises(ValueError, match="after"):
        await dispatch_memory_recall(
            kb,
            uuid4(),
            {"query": "x", "after": "2026-03-31T00:00:00Z", "before": "2026-03-01T00:00:00Z"},
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dispatch_memory_recall_rejects_bad_iso() -> None:
    kb = object()
    with pytest.raises(ValueError, match="after"):
        await dispatch_memory_recall(kb, uuid4(), {"query": "x", "after": "yesterday"})


# ---------------------------------------------------------------------------
# format_memory_context — direct
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_format_memory_context_empty_returns_abstention_message() -> None:
    out = format_memory_context([])
    assert out == "<memory-context>\nNo prior memories found.\n</memory-context>"


@pytest.mark.unit
def test_format_memory_context_renders_score_date_and_content() -> None:
    chunks = [
        _Chunk(content="we talked about Project Phoenix", score=0.87, created_at=datetime(2026, 3, 14, tzinfo=UTC)),
    ]
    out = format_memory_context(chunks)
    assert "[score: 0.87, 2026-03-14] we talked about Project Phoenix" in out
    assert "Relevant memories" in out
    assert "Key entities mentioned" not in out  # no entities → section omitted


@pytest.mark.unit
def test_format_memory_context_truncates_overlong_content() -> None:
    long_text = "x" * 800
    chunks = [_Chunk(content=long_text, score=0.5, created_at=datetime(2026, 3, 14, tzinfo=UTC))]
    out = format_memory_context(chunks)
    assert "..." in out
    # 500 chars + "..." plus the prefix line — the raw 800-char block must not appear verbatim.
    assert long_text not in out


@pytest.mark.unit
def test_format_memory_context_includes_entity_section_when_present() -> None:
    chunks = [_Chunk(content="hi", score=0.5, created_at=datetime(2026, 3, 14, tzinfo=UTC))]
    entities = [
        _Entity(name="Alice", entity_type="PERSON", mention_count=12),
        _Entity(name="Project Phoenix", entity_type="PROJECT", mention_count=5),
    ]
    out = format_memory_context(chunks, entities)
    assert "Key entities mentioned:" in out
    assert "- Alice (PERSON, mentions: 12)" in out
    assert "- Project Phoenix (PROJECT, mentions: 5)" in out


@pytest.mark.unit
def test_format_memory_context_caps_at_five_chunks() -> None:
    chunks = [_Chunk(content=f"chunk-{i}", score=0.5, created_at=datetime(2026, 3, 14, tzinfo=UTC)) for i in range(10)]
    out = format_memory_context(chunks)
    assert "chunk-0" in out
    assert "chunk-4" in out
    assert "chunk-5" not in out
