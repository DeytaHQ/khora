"""Cardinality budget for telemetry span attributes.

Span attributes (Logfire / OTel exporter → Prometheus) bill per distinct
attribute *value*. A raw user query as a span attr is therefore an
unbounded cost bomb: every keystroke variant mints a new label.

These tests enforce the contract that:

1. ``khora.recall`` spans carry only ``query_hash`` (8 hex chars) and
   ``query_length`` (int) — never the raw query.
2. The hash is deterministic across calls.
3. Every ``khora.recall`` span attribute fits inside a 64-char budget,
   so future additions cannot silently smuggle in unbounded text.

This is a pure unit test — no infra, no logfire dependency.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.khora import Khora, RecallResult
from khora.telemetry import bounded_text_hash

ATTR_BUDGET_CHARS = 64


def _mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.get_postgresql_url.return_value = "postgresql://test"
    cfg.get_graph_config.return_value = None
    cfg.get_vector_config.return_value = None
    cfg.get_neo4j_url.return_value = None
    cfg.get_neo4j_user.return_value = None
    cfg.get_neo4j_password.return_value = None
    cfg.get_neo4j_database.return_value = None
    cfg.storage.embedding_dimension = 1536
    cfg.llm.model = "gpt-4o-mini"
    cfg.llm.embedding_model = "text-embedding-3-small"
    cfg.llm.embedding_dimension = 1536
    cfg.llm.extraction_model = None
    cfg.llm.timeout = 30
    cfg.llm.max_retries = 3
    cfg.telemetry_database_url = None
    cfg.telemetry_service_name = "khora-test"
    return cfg


_RESOLVED_NS = uuid4()


def _mock_engine() -> MagicMock:
    eng = MagicMock()
    eng._storage = MagicMock()
    eng._storage.resolve_namespace = AsyncMock(return_value=_RESOLVED_NS)
    eng.recall = AsyncMock()
    return eng


def _make_kb() -> Khora:
    with patch("khora.khora.load_config", return_value=_mock_config()):
        kb = Khora()
    kb._connected = True
    kb._engine = _mock_engine()
    return kb


def _capturing_span(captured: list[dict]):
    """Return a stub trace_span(name, **attrs) that records kwargs."""

    @contextmanager
    def _stub(name, **attrs):
        captured.append({"name": name, "attrs": attrs})
        yield None

    return _stub


async def _run_recall(kb: Khora, query: str, captured: list[dict]) -> None:
    ns_id = uuid4()
    kb._engine.recall = AsyncMock(
        return_value=RecallResult(
            query=query,
            namespace_id=ns_id,
            chunks=[],
            entities=[],
            context_text="",
        )
    )
    with (
        patch("khora.telemetry.context.ensure_trace_id"),
        patch("khora.telemetry.context.clear_trace_id"),
        patch("khora.khora.trace_span", side_effect=_capturing_span(captured)),
    ):
        await kb.recall(query, namespace=ns_id)


@pytest.mark.asyncio
async def test_recall_span_replaces_raw_query_with_hash_and_length():
    captured: list[dict] = []
    kb = _make_kb()
    long_query = "tell me everything you know about " + "x" * 500

    await _run_recall(kb, long_query, captured)

    assert len(captured) == 1
    attrs = captured[0]["attrs"]
    assert captured[0]["name"] == "khora.recall"
    assert "query" not in attrs, "raw query must not appear as a span attribute"
    assert attrs["query_length"] == len(long_query)
    assert isinstance(attrs["query_hash"], str)
    assert len(attrs["query_hash"]) == 8
    assert all(c in "0123456789abcdef" for c in attrs["query_hash"])


@pytest.mark.asyncio
async def test_identical_queries_produce_identical_hashes():
    captured: list[dict] = []
    kb = _make_kb()
    query = "who knows about project orion"

    await _run_recall(kb, query, captured)
    await _run_recall(kb, query, captured)

    assert len(captured) == 2
    assert captured[0]["attrs"]["query_hash"] == captured[1]["attrs"]["query_hash"]
    # And independently match the helper's output.
    assert captured[0]["attrs"]["query_hash"] == bounded_text_hash(query)


@pytest.mark.asyncio
async def test_recall_span_attributes_respect_cardinality_budget():
    """No span attribute on khora.recall may exceed the per-attr budget.

    The budget protects against silently re-introducing an unbounded text
    attribute (e.g. someone adds ``error_message=str(exc)`` to the span).
    """
    captured: list[dict] = []
    kb = _make_kb()
    # Pathological query: long enough to blow the budget if it leaked through.
    pathological = "A" * 4096

    await _run_recall(kb, pathological, captured)

    attrs = captured[0]["attrs"]
    for key, value in attrs.items():
        rendered = str(value)
        assert len(rendered) <= ATTR_BUDGET_CHARS, (
            f"span attr {key!r}={rendered[:32]!r}... is {len(rendered)} chars, exceeds {ATTR_BUDGET_CHARS}-char budget"
        )


def test_bounded_text_hash_is_deterministic_and_short():
    assert bounded_text_hash("hello world") == bounded_text_hash("hello world")
    assert len(bounded_text_hash("anything")) == 8
    # Hex.
    assert all(c in "0123456789abcdef" for c in bounded_text_hash("anything"))
    # Distinct inputs → distinct hashes (with overwhelming probability).
    assert bounded_text_hash("foo") != bounded_text_hash("bar")
