"""Unit tests for the deterministic recall-filter telemetry — ``@internal``.

Covers the two OTel counters in :mod:`khora.filter.telemetry` and the
``filter.canonical_hash`` / ``filter.metadata_leaf_count`` attributes set on the
``khora.recall`` span:

1. Both counters are constructible via their ``_get_*`` helpers (and memoized —
   a second call returns the same instrument).
2. The two V1-only counters (``under_filled`` / ``graph_channel_empty``) are
   pre-declared but have no ``.add()`` call site yet, so they stay quiet on the
   compile + recall paths exercised here.
3. ``filter.canonical_hash`` and ``filter.metadata_leaf_count`` are present on
   the ``khora.recall`` span exactly when a ``filter=`` is supplied, and absent
   on an unfiltered recall.

Hermetic — no Docker / DB. The counter tests monkeypatch the module-level
singletons in ``khora.filter.telemetry`` with recording fakes (mirroring
``tests/unit/telemetry/test_aggregate_metrics.py``). The span test installs an
in-memory OTel exporter (mirroring ``tests/unit/telemetry/test_otel.py``) and
drives the real ``Khora.recall`` facade over a mocked engine (the ``_make_kb``
shape from ``test_aggregate_metrics.py``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from opentelemetry import trace as _otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from khora.filter import RecallFilter, canonical_hash, metadata_leaf_count
from khora.filter import telemetry as filter_telemetry
from khora.filter.ast import parse_to_ast
from khora.filter.compilers.postgres import compile_postgres
from khora.filter.context import CompileContext

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Recording fakes + helpers.
# ---------------------------------------------------------------------------


class _RecordingCounter:
    """Captures ``.add(value, attributes=...)`` calls for assertions."""

    def __init__(self) -> None:
        self.adds: list[tuple[float, dict[str, Any]]] = []

    def add(self, value: float, attributes: Any = None) -> None:
        self.adds.append((value, dict(attributes or {})))


_CTX = CompileContext(backend_target="khora_chunks")


def _ast(wire: dict) -> Any:
    """Validate a wire-form filter and lower it to the canonical AST."""
    return parse_to_ast(RecallFilter.model_validate(wire))


@pytest.fixture
def recording_counters(monkeypatch: pytest.MonkeyPatch) -> dict[str, _RecordingCounter]:
    """Replace the two module-level counter singletons with recording fakes.

    The ``_get_*`` helpers return the singleton if already set, so pre-seeding
    each module global makes ``record_*`` / any ``.add()`` site land on the fake.
    """
    counters = {
        "under_filled": _RecordingCounter(),
        "graph_channel_empty": _RecordingCounter(),
    }
    monkeypatch.setattr(filter_telemetry, "_under_filled_counter", counters["under_filled"])
    monkeypatch.setattr(filter_telemetry, "_graph_channel_empty_counter", counters["graph_channel_empty"])
    return counters


# ===========================================================================
# (1) Both counters are constructible via their _get_* helpers.
# ===========================================================================


@pytest.fixture
def _reset_counter_singletons() -> Any:
    """Null out the two singletons so the _get_* helpers build fresh ones."""
    saved = (
        filter_telemetry._under_filled_counter,
        filter_telemetry._graph_channel_empty_counter,
    )
    filter_telemetry._under_filled_counter = None
    filter_telemetry._graph_channel_empty_counter = None
    yield
    (
        filter_telemetry._under_filled_counter,
        filter_telemetry._graph_channel_empty_counter,
    ) = saved


def test_both_counter_helpers_construct(_reset_counter_singletons: Any) -> None:
    """Each ``_get_*`` helper returns a non-None instrument (no real provider needed)."""
    assert filter_telemetry._get_under_filled_counter() is not None
    assert filter_telemetry._get_graph_channel_empty_counter() is not None


def test_counter_helpers_are_memoized(_reset_counter_singletons: Any) -> None:
    """A second call returns the same instrument (lazy double-checked singleton)."""
    for getter in (
        filter_telemetry._get_under_filled_counter,
        filter_telemetry._get_graph_channel_empty_counter,
    ):
        first = getter()
        assert getter() is first


# ===========================================================================
# (2) under_filled + graph_channel_empty stay quiet on V1 paths.
# ===========================================================================


def test_v1_counters_quiet_on_metadata_compile(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    """Compiling a metadata predicate fires neither V1 counter."""
    compile_postgres(_ast({"metadata.tier": "gold"}), _CTX)

    assert recording_counters["under_filled"].adds == []
    assert recording_counters["graph_channel_empty"].adds == []


def test_v1_counters_quiet_on_system_key_compile(
    recording_counters: dict[str, _RecordingCounter],
) -> None:
    """A system-key-only compile emits neither V1 counter."""
    compile_postgres(_ast({"source_name": "linear"}), _CTX)

    assert recording_counters["under_filled"].adds == []
    assert recording_counters["graph_channel_empty"].adds == []


# ===========================================================================
# (3) filter.canonical_hash + filter.metadata_leaf_count on the khora.recall span.
# ===========================================================================
#
# Drive the real ``Khora.recall`` facade over a mocked engine so the actual
# ``trace_span("khora.recall")`` block runs, and read the emitted span back from
# an in-memory exporter. The mock shape mirrors test_aggregate_metrics.py.


from khora.khora import Khora, RecallResult  # noqa: E402 - after the OTel imports above

_RESOLVE_ROW_ID = uuid4()


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


def _make_kb(ns_id: Any) -> Khora:
    with patch("khora.khora.load_config", return_value=_mock_config()):
        kb = Khora()
    kb._connected = True
    engine = MagicMock()
    engine._storage = MagicMock()
    engine._storage.resolve_namespace = AsyncMock(return_value=_RESOLVE_ROW_ID)
    engine.recall = AsyncMock(
        return_value=RecallResult(
            query="q",
            namespace_id=ns_id,
            documents=[],
            chunks=[],
            entities=[],
            relationships=[],
        )
    )
    kb._engine = engine
    return kb


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install an in-memory OTel exporter and rebind khora's cached tracer.

    Resets the process-wide tracer provider so a real ``TracerProvider`` can be
    set, then restores the proxy provider on teardown (so the recording span
    machinery does not leak into later tests).
    """
    import opentelemetry.trace as _t
    from opentelemetry.trace import ProxyTracerProvider

    from khora.telemetry import _otel as _otel_module

    # Fresh provider slot.
    _t._TRACER_PROVIDER_SET_ONCE = _t.Once()
    _t._TRACER_PROVIDER = None
    _t._PROXY_TRACER_PROVIDER = ProxyTracerProvider()

    tp = TracerProvider()
    exporter = InMemorySpanExporter()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    _otel_trace.set_tracer_provider(tp)
    monkeypatch.setattr(
        _otel_module,
        "_TRACER",
        _otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION),
    )

    yield exporter

    exporter.shutdown()
    # Restore proxy provider + cached tracer so other tests start clean.
    _t._TRACER_PROVIDER_SET_ONCE = _t.Once()
    _t._TRACER_PROVIDER = None
    _t._PROXY_TRACER_PROVIDER = ProxyTracerProvider()
    monkeypatch.setattr(
        _otel_module,
        "_TRACER",
        _otel_trace.get_tracer("khora", _otel_module._KHORA_VERSION),
    )


def _recall_span(exporter: InMemorySpanExporter) -> Any:
    spans = [s for s in exporter.get_finished_spans() if s.name == "khora.recall"]
    assert len(spans) == 1, f"expected exactly one khora.recall span, got {len(spans)}"
    return spans[0]


@pytest.mark.asyncio
async def test_recall_span_has_canonical_hash_when_filter_given(span_exporter: Any) -> None:
    """A ``filter=`` recall tags the span with the AST's canonical hash."""
    ns_id = uuid4()
    kb = _make_kb(ns_id)
    wire = {"source_name": "linear"}

    await kb.recall("q", namespace=ns_id, filter=wire)

    span = _recall_span(span_exporter)
    expected = canonical_hash(parse_to_ast(RecallFilter.model_validate(wire)))
    assert span.attributes.get("filter.canonical_hash") == expected


@pytest.mark.asyncio
async def test_recall_span_omits_canonical_hash_without_filter(span_exporter: Any) -> None:
    """An unfiltered recall does not carry the ``filter.canonical_hash`` attribute."""
    ns_id = uuid4()
    kb = _make_kb(ns_id)

    await kb.recall("q", namespace=ns_id)

    span = _recall_span(span_exporter)
    assert "filter.canonical_hash" not in span.attributes


@pytest.mark.asyncio
async def test_recall_span_has_metadata_leaf_count_when_filter_given(span_exporter: Any) -> None:
    """A ``filter=`` recall tags the span with the AST's metadata-leaf count."""
    ns_id = uuid4()
    kb = _make_kb(ns_id)
    # Two metadata leaves plus a system key (which does not count).
    wire = {"source_name": "linear", "metadata.tier": "gold", "metadata.score": {"$gt": 5}}

    await kb.recall("q", namespace=ns_id, filter=wire)

    span = _recall_span(span_exporter)
    expected = metadata_leaf_count(parse_to_ast(RecallFilter.model_validate(wire)))
    assert expected == 2
    assert span.attributes.get("filter.metadata_leaf_count") == expected


@pytest.mark.asyncio
async def test_recall_span_metadata_leaf_count_zero_for_system_key_only(span_exporter: Any) -> None:
    """A system-key-only ``filter=`` recall tags the span with a zero leaf count."""
    ns_id = uuid4()
    kb = _make_kb(ns_id)
    wire = {"source_name": "linear"}

    await kb.recall("q", namespace=ns_id, filter=wire)

    span = _recall_span(span_exporter)
    assert span.attributes.get("filter.metadata_leaf_count") == 0


@pytest.mark.asyncio
async def test_recall_span_omits_metadata_leaf_count_without_filter(span_exporter: Any) -> None:
    """An unfiltered recall does not carry the ``filter.metadata_leaf_count`` attribute."""
    ns_id = uuid4()
    kb = _make_kb(ns_id)

    await kb.recall("q", namespace=ns_id)

    span = _recall_span(span_exporter)
    assert "filter.metadata_leaf_count" not in span.attributes
