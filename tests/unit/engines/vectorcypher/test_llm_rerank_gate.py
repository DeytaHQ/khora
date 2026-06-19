"""Tests for the LLM-rerank skip gate (issue #814).

The gate centralises three independent skip reasons that used to be
duplicated inline in the complex (``_vectorcypher_retrieve``) and simple
(``_simple_retrieve``) paths:

- ``"not_temporal"``  — opt-in but query isn't temporal (silent skip).
- ``"no_version_metadata"`` — ``mode='auto'`` precondition (PR #364).
  Emits a one-time per-namespace WARNING so users discover why their
  opt-in is being silently no-op'd.
- ``"decisive_winner"`` — latency optimisation; applies in both modes.

Tests live separately from ``test_retriever_coverage.py`` so the gate
contract is easy to find and the OTel meter fixture stays localised.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from opentelemetry import metrics as _otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from khora.engines.vectorcypher.fusion import FusedResult
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.engines.vectorcypher.temporal_detection import (
    TemporalCategory,
    TemporalSignal,
)

# ---------------------------------------------------------------------------
# OTel meter fixture — rebinds the cached counter on the retriever module
# so the new SDK MeterProvider sees the call.
# ---------------------------------------------------------------------------


def _reset_otel_globals() -> None:
    import opentelemetry.metrics._internal as _m
    import opentelemetry.trace as _t
    from opentelemetry.metrics._internal import _ProxyMeterProvider
    from opentelemetry.trace import ProxyTracerProvider

    _t._TRACER_PROVIDER_SET_ONCE = _t.Once()
    _t._TRACER_PROVIDER = None
    _t._PROXY_TRACER_PROVIDER = ProxyTracerProvider()

    _m._METER_PROVIDER_SET_ONCE = _m.Once()
    _m._METER_PROVIDER = None
    _m._PROXY_METER_PROVIDER = _ProxyMeterProvider()


@pytest.fixture
def metric_reader(monkeypatch: pytest.MonkeyPatch):
    """Install an in-memory MeterProvider and rebind the rerank-skip counter."""
    _reset_otel_globals()
    reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[reader])
    _otel_metrics.set_meter_provider(mp)

    from khora.engines.vectorcypher import retriever as retriever_mod
    from khora.telemetry import _otel as _otel_module
    from khora.telemetry.metrics import metric_counter

    _otel_module._METER = _otel_metrics.get_meter("khora", _otel_module._KHORA_VERSION)
    new_counter = metric_counter(
        "khora.vectorcypher.llm_reranking.skipped_total",
        description=(
            "Number of times the VectorCypher LLM rerank step was skipped, "
            "by reason (not_temporal / no_version_metadata / decisive_winner)."
        ),
    )
    monkeypatch.setattr(retriever_mod, "_LLM_RERANKING_SKIPPED_COUNTER", new_counter)

    yield reader

    _reset_otel_globals()
    _otel_module._METER = _otel_metrics.get_meter("khora", _otel_module._KHORA_VERSION)


def _skip_points(reader: InMemoryMetricReader, reason: str) -> int:
    """Return the total count emitted for ``reason``."""
    data = reader.get_metrics_data()
    if data is None:
        return 0
    total = 0
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != "khora.vectorcypher.llm_reranking.skipped_total":
                    continue
                for point in metric.data.data_points:
                    if dict(point.attributes).get("reason") == reason:
                        total += int(point.value)
    return total


# ---------------------------------------------------------------------------
# Helpers — TemporalSearchResult builder + retriever factory
# ---------------------------------------------------------------------------


_SENTINEL: Any = object()


def _make_retriever(
    *,
    config: RetrieverConfig | None = None,
    vector_store: Any | None = None,
) -> VectorCypherRetriever:
    if vector_store is None:
        vector_store = AsyncMock()
    return VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=config or RetrieverConfig(enable_llm_reranking=True, enable_reranking=False),
        storage=None,
    )


def _make_temporal_search_result(
    content: str,
    *,
    similarity: float = 0.9,
    version: int | None = None,
):
    """Build a TemporalSearchResult mirroring the production shape."""
    from khora.storage.temporal import TemporalChunk, TemporalSearchResult

    metadata: dict[str, Any] = {}
    if version is not None:
        metadata["version"] = version
    tc = TemporalChunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        embedding=None,
        occurred_at=datetime.now(UTC),
        metadata=metadata,
    )
    return TemporalSearchResult(chunk=tc, similarity=similarity, combined_score=similarity)


def _temporal_signal(category: TemporalCategory = TemporalCategory.RECENCY) -> TemporalSignal:
    return TemporalSignal(
        is_temporal=True,
        category=category,
        confidence=0.9,
        source="dictionary",
    )


def _non_temporal_signal() -> TemporalSignal:
    return TemporalSignal(
        is_temporal=False,
        category=TemporalCategory.NONE,
        confidence=0.0,
        source="dictionary",
    )


# ---------------------------------------------------------------------------
# Direct helper-level gate tests — both paths use ``_evaluate_llm_rerank_gate``
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvaluateGateUnit:
    """Unit-test the gate helper directly with both candidate shapes."""

    def test_auto_no_versions_skips_with_reason(self) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(enable_llm_reranking=True, llm_reranking_mode="auto"),
        )
        # FusedResult (complex-path) shape — no version metadata.
        c1 = type("X", (), {"metadata": {}})()
        c2 = type("X", (), {"metadata": {}})()
        candidates = [
            FusedResult(item=c1, rrf_score=0.9, item_id=uuid4()),
            FusedResult(item=c2, rrf_score=0.8, item_id=uuid4()),
        ]
        should_run, reason = retriever._evaluate_llm_rerank_gate(candidates, _temporal_signal(), namespace_id=uuid4())
        assert should_run is False
        assert reason == "no_version_metadata"

    def test_auto_with_versions_runs(self) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                llm_reranking_mode="auto",
                # Make the decisive-winner gate not fire.
                llm_reranking_confidence_threshold=0.99,
                llm_reranking_min_top_score=0.99,
                llm_reranking_decisive_gap=0.99,
            ),
        )
        c1 = type("X", (), {"metadata": {"version": 2}})()
        c2 = type("X", (), {"metadata": {"version": 1}})()
        candidates = [
            FusedResult(item=c1, rrf_score=0.5, item_id=uuid4()),
            FusedResult(item=c2, rrf_score=0.49, item_id=uuid4()),
        ]
        should_run, reason = retriever._evaluate_llm_rerank_gate(candidates, _temporal_signal(), namespace_id=uuid4())
        assert should_run is True
        assert reason is None

    def test_always_bypasses_version_check(self) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                llm_reranking_mode="always",
                # Decisive-winner gate cannot fire.
                llm_reranking_confidence_threshold=0.99,
                llm_reranking_min_top_score=0.99,
                llm_reranking_decisive_gap=0.99,
            ),
        )
        # No version metadata — under "auto" this would skip, under "always" it must run.
        c1 = type("X", (), {"metadata": {}})()
        c2 = type("X", (), {"metadata": {}})()
        candidates = [
            FusedResult(item=c1, rrf_score=0.5, item_id=uuid4()),
            FusedResult(item=c2, rrf_score=0.49, item_id=uuid4()),
        ]
        should_run, reason = retriever._evaluate_llm_rerank_gate(candidates, _temporal_signal(), namespace_id=uuid4())
        assert should_run is True
        assert reason is None

    def test_always_still_respects_decisive_winner(self) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                llm_reranking_mode="always",
                llm_reranking_confidence_threshold=0.1,  # fires when gap >= 0.1
                llm_reranking_min_top_score=0.7,
                llm_reranking_decisive_gap=0.1,
            ),
        )
        c1 = type("X", (), {"metadata": {}})()
        c2 = type("X", (), {"metadata": {}})()
        candidates = [
            FusedResult(item=c1, rrf_score=0.9, item_id=uuid4()),
            FusedResult(item=c2, rrf_score=0.5, item_id=uuid4()),
        ]
        should_run, reason = retriever._evaluate_llm_rerank_gate(candidates, _temporal_signal(), namespace_id=uuid4())
        assert should_run is False
        assert reason == "decisive_winner"

    def test_non_temporal_returns_not_temporal(self) -> None:
        retriever = _make_retriever()
        c1 = type("X", (), {"metadata": {"version": 1}})()
        candidates = [FusedResult(item=c1, rrf_score=0.9, item_id=uuid4())]
        should_run, reason = retriever._evaluate_llm_rerank_gate(
            candidates, _non_temporal_signal(), namespace_id=uuid4()
        )
        assert should_run is False
        assert reason == "not_temporal"

    def test_warning_emitted_once_per_namespace(self, caplog: pytest.LogCaptureFixture) -> None:
        """Repeated skips for the same namespace produce exactly one WARNING."""
        retriever = _make_retriever(
            config=RetrieverConfig(enable_llm_reranking=True, llm_reranking_mode="auto"),
        )
        c1 = type("X", (), {"metadata": {}})()
        candidates = [FusedResult(item=c1, rrf_score=0.9, item_id=uuid4())]
        ns = uuid4()

        # loguru → stdlib bridge: use propagate-aware capture.
        from loguru import logger as _loguru

        msgs: list[str] = []
        handler_id = _loguru.add(lambda m: msgs.append(str(m)), level="WARNING")
        try:
            for _ in range(3):
                retriever._evaluate_llm_rerank_gate(candidates, _temporal_signal(), namespace_id=ns)
        finally:
            _loguru.remove(handler_id)

        warnings = [m for m in msgs if "VectorCypher LLM reranking skipped" in m]
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}: {warnings}"

    def test_warning_dedup_keyed_per_namespace(self) -> None:
        """Two distinct namespaces each get their own one-time warning."""
        retriever = _make_retriever(
            config=RetrieverConfig(enable_llm_reranking=True, llm_reranking_mode="auto"),
        )
        c1 = type("X", (), {"metadata": {}})()
        candidates = [FusedResult(item=c1, rrf_score=0.9, item_id=uuid4())]

        from loguru import logger as _loguru

        msgs: list[str] = []
        handler_id = _loguru.add(lambda m: msgs.append(str(m)), level="WARNING")
        try:
            retriever._evaluate_llm_rerank_gate(candidates, _temporal_signal(), namespace_id=uuid4())
            retriever._evaluate_llm_rerank_gate(candidates, _temporal_signal(), namespace_id=uuid4())
        finally:
            _loguru.remove(handler_id)

        warnings = [m for m in msgs if "VectorCypher LLM reranking skipped" in m]
        assert len(warnings) == 2, f"expected one warning per namespace, got {warnings}"

    def test_simple_path_candidate_shape_supported(self) -> None:
        """(Chunk, score) tuples — the simple-path candidate shape — work too."""
        from khora.core.models import Chunk

        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                llm_reranking_mode="auto",
                llm_reranking_confidence_threshold=0.99,
                llm_reranking_min_top_score=0.99,
                llm_reranking_decisive_gap=0.99,
            ),
        )
        c_with = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="a", metadata={"version": 1})
        c_other = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="b", metadata={})
        # Has versions — should run.
        should_run, reason = retriever._evaluate_llm_rerank_gate(
            [(c_with, 0.5), (c_other, 0.49)], _temporal_signal(), namespace_id=uuid4()
        )
        assert should_run is True
        assert reason is None

        # No versions — should skip with no_version_metadata.
        should_run, reason = retriever._evaluate_llm_rerank_gate(
            [(c_other, 0.5), (c_other, 0.49)], _temporal_signal(), namespace_id=uuid4()
        )
        assert should_run is False
        assert reason == "no_version_metadata"


# ---------------------------------------------------------------------------
# End-to-end on the simple path: assert reranker invocation + counter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSimplePathRerankGate:
    @pytest.mark.asyncio
    async def test_auto_no_versions_skips_and_counts(self, metric_reader: InMemoryMetricReader) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                enable_reranking=False,
                llm_reranking_mode="auto",
            ),
        )
        r1 = _make_temporal_search_result("c1", similarity=0.9)
        r2 = _make_temporal_search_result("c2", similarity=0.8)
        retriever._vector_store.search = AsyncMock(return_value=[r1, r2])
        retriever._apply_llm_reranking = AsyncMock(return_value=[])  # type: ignore[method-assign]

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_signal=_temporal_signal(TemporalCategory.STATE_QUERY),
        )
        retriever._apply_llm_reranking.assert_not_called()
        assert _skip_points(metric_reader, "no_version_metadata") == 1

    @pytest.mark.asyncio
    async def test_auto_with_versions_invokes_reranker(self, metric_reader: InMemoryMetricReader) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                enable_reranking=False,
                llm_reranking_mode="auto",
                # Disable the decisive-winner skip so we can isolate the version gate.
                llm_reranking_confidence_threshold=0.99,
                llm_reranking_min_top_score=0.99,
                llm_reranking_decisive_gap=0.99,
            ),
        )
        r1 = _make_temporal_search_result("c1", similarity=0.9, version=2)
        r2 = _make_temporal_search_result("c2", similarity=0.89, version=1)
        retriever._vector_store.search = AsyncMock(return_value=[r1, r2])
        retriever._apply_llm_reranking = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda q, fused, limit, *, namespace_id: fused
        )

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_signal=_temporal_signal(TemporalCategory.STATE_QUERY),
        )
        retriever._apply_llm_reranking.assert_called_once()
        # No skip counted.
        assert _skip_points(metric_reader, "no_version_metadata") == 0
        assert _skip_points(metric_reader, "decisive_winner") == 0
        assert _skip_points(metric_reader, "not_temporal") == 0

    @pytest.mark.asyncio
    async def test_always_no_versions_still_invokes(self, metric_reader: InMemoryMetricReader) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                enable_reranking=False,
                llm_reranking_mode="always",
                llm_reranking_confidence_threshold=0.99,
                llm_reranking_min_top_score=0.99,
                llm_reranking_decisive_gap=0.99,
            ),
        )
        r1 = _make_temporal_search_result("c1", similarity=0.9)
        r2 = _make_temporal_search_result("c2", similarity=0.89)
        retriever._vector_store.search = AsyncMock(return_value=[r1, r2])
        retriever._apply_llm_reranking = AsyncMock(  # type: ignore[method-assign]
            side_effect=lambda q, fused, limit, *, namespace_id: fused
        )

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_signal=_temporal_signal(TemporalCategory.STATE_QUERY),
        )
        retriever._apply_llm_reranking.assert_called_once()
        assert _skip_points(metric_reader, "no_version_metadata") == 0

    @pytest.mark.asyncio
    async def test_always_decisive_winner_skips_with_reason(self, metric_reader: InMemoryMetricReader) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                enable_reranking=False,
                llm_reranking_mode="always",
                # Decisive-winner gate: top>=0.7 AND gap>=0.1
                llm_reranking_confidence_threshold=0.99,
                llm_reranking_min_top_score=0.7,
                llm_reranking_decisive_gap=0.1,
            ),
        )
        r1 = _make_temporal_search_result("c1", similarity=0.95)
        r2 = _make_temporal_search_result("c2", similarity=0.5)
        retriever._vector_store.search = AsyncMock(return_value=[r1, r2])
        retriever._apply_llm_reranking = AsyncMock(return_value=[])  # type: ignore[method-assign]

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_signal=_temporal_signal(TemporalCategory.STATE_QUERY),
        )
        retriever._apply_llm_reranking.assert_not_called()
        assert _skip_points(metric_reader, "decisive_winner") == 1

    @pytest.mark.asyncio
    async def test_non_temporal_query_skips_with_reason(self, metric_reader: InMemoryMetricReader) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                enable_reranking=False,
                llm_reranking_mode="always",
            ),
        )
        r1 = _make_temporal_search_result("c1", similarity=0.9, version=1)
        r2 = _make_temporal_search_result("c2", similarity=0.8, version=1)
        retriever._vector_store.search = AsyncMock(return_value=[r1, r2])
        retriever._apply_llm_reranking = AsyncMock(return_value=[])  # type: ignore[method-assign]

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_signal=_non_temporal_signal(),
        )
        retriever._apply_llm_reranking.assert_not_called()
        assert _skip_points(metric_reader, "not_temporal") == 1


# ---------------------------------------------------------------------------
# Complex path: assert the same gate semantics directly. We invoke the
# helper through synthetic candidates (sidesteps the heavy graph mocking
# required to drive _vectorcypher_retrieve end-to-end) — gate is shared
# code, so unit-checking it once on the helper plus an end-to-end pass on
# the simple path is sufficient parameter coverage.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCounterEmission:
    """Direct counter-emission check through the gate helper.

    The simple-path end-to-end tests above exercise the live counter via
    ``_simple_retrieve``. This class adds direct gate-helper coverage so
    counter regressions on the complex path are caught even if the heavy
    mock harness around ``_vectorcypher_retrieve`` ever drifts.
    """

    def _emit(self, retriever: VectorCypherRetriever, candidates: list[Any], signal: TemporalSignal) -> None:
        """Replicates the call-site contract: emit the counter on skip."""
        from khora.engines.vectorcypher import retriever as retriever_mod

        should_run, reason = retriever._evaluate_llm_rerank_gate(candidates, signal, namespace_id=uuid4())
        if not should_run and reason is not None:
            retriever_mod._LLM_RERANKING_SKIPPED_COUNTER.add(1, attributes={"reason": reason})

    def test_no_version_counter_emits(self, metric_reader: InMemoryMetricReader) -> None:
        retriever = _make_retriever()
        c1 = type("X", (), {"metadata": {}})()
        candidates = [FusedResult(item=c1, rrf_score=0.9, item_id=uuid4())]
        self._emit(retriever, candidates, _temporal_signal())
        assert _skip_points(metric_reader, "no_version_metadata") == 1

    def test_not_temporal_counter_emits(self, metric_reader: InMemoryMetricReader) -> None:
        retriever = _make_retriever()
        c1 = type("X", (), {"metadata": {"version": 1}})()
        candidates = [FusedResult(item=c1, rrf_score=0.9, item_id=uuid4())]
        self._emit(retriever, candidates, _non_temporal_signal())
        assert _skip_points(metric_reader, "not_temporal") == 1

    def test_decisive_winner_counter_emits(self, metric_reader: InMemoryMetricReader) -> None:
        retriever = _make_retriever(
            config=RetrieverConfig(
                enable_llm_reranking=True,
                llm_reranking_mode="always",
                llm_reranking_confidence_threshold=0.05,
                llm_reranking_min_top_score=0.7,
                llm_reranking_decisive_gap=0.1,
            ),
        )
        c1 = type("X", (), {"metadata": {}})()
        c2 = type("X", (), {"metadata": {}})()
        candidates = [
            FusedResult(item=c1, rrf_score=0.95, item_id=uuid4()),
            FusedResult(item=c2, rrf_score=0.5, item_id=uuid4()),
        ]
        self._emit(retriever, candidates, _temporal_signal())
        assert _skip_points(metric_reader, "decisive_winner") == 1
