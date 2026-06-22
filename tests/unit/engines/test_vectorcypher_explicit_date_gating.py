"""A date-constrained caller filter is treated as explicit temporal intent.

When a caller passes a ``filter`` whose AST constrains a date system key
(``occurred_at`` / ``created_at``), the engine synthesizes an EXPLICIT temporal
signal — exactly as it does for an API-supplied ``temporal_filter``. That
EXPLICIT signal drives downstream recency behavior (the version filter, the
restrictive-fallback skip, recency weighting). A NON-date caller filter (e.g. a
pure ``metadata.channel`` constraint) must NOT trigger EXPLICIT: it runs the
normal query-string temporal detector and rides alongside as ``filter_ast``.

These run at the engine layer with a mock-connected retriever; the engine's
``recall`` builds the temporal signal and forwards it to ``retriever.retrieve``.
We spy on the ``temporal_signal`` the retriever receives — the observable
contract — rather than re-deriving the synthesis logic. No database.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.engines.vectorcypher.retriever import VectorCypherResult
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.filter import RecallFilter, parse_to_ast
from khora.query.temporal_detection import TemporalCategory

pytestmark = pytest.mark.unit


def _ast(doc: dict[str, Any]) -> object:
    """Lower a wire-form filter document to its canonical AST (as the facade does)."""
    return parse_to_ast(RecallFilter.model_validate(doc))


def _connected_engine() -> VectorCypherEngine:
    """A mock-connected engine whose retriever.retrieve is an inspectable spy.

    Mirrors ``TestVectorCypherEngineRecall.connected_engine`` in
    ``test_vectorcypher_engine.py``: the retriever returns an empty
    ``VectorCypherResult`` so ``recall`` completes, and the engine forwards the
    temporal_signal it built into ``retriever.retrieve``.
    """
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.get_neo4j_url.return_value = "bolt://localhost:7687"
    config.get_neo4j_user.return_value = "neo4j"
    config.get_neo4j_password.return_value = "password"
    config.get_neo4j_database.return_value = "neo4j"
    config.get_graph_config.return_value = MagicMock()
    config.get_vector_config.return_value = MagicMock()
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.embedding_dimension = 1536
    # Abstention knobs (#1331) — the engine reads these off config.query now.
    config.query.abstention_min_chunks = 1
    config.query.abstention_min_top_score = 0.3
    config.query.abstention_combined_threshold = 0.5
    config.query.abstention_weight_entities_empty = 0.3
    config.query.abstention_weight_chunks_below_min = 0.4
    config.query.abstention_weight_top_score_low = 0.3
    config.query.abstention_mode = "cosine_floor"
    config.query.abstention_confidence_target_cosine = 0.5
    config.query.abstention_confidence_target_gap = 0.1

    engine = VectorCypherEngine(config)
    engine._connected = True
    engine._storage = AsyncMock()
    engine._temporal_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._dual_nodes = AsyncMock()
    engine._neo4j_driver = AsyncMock()

    routing = RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        use_graph=False,
        graph_depth=0,
        confidence=0.8,
        reasoning="test",
    )
    empty_result = VectorCypherResult(
        chunks=[],
        entities=[],
        routing_decision=routing,
        metadata={},
    )
    engine._retriever = AsyncMock()
    engine._retriever.retrieve = AsyncMock(return_value=empty_result)
    engine._router = MagicMock()
    # ``hybrid_alpha`` is read/restored around the retrieve call; give it a real
    # float so the save/restore around ``retrieve`` doesn't choke on a MagicMock.
    engine._retriever._config = MagicMock()
    engine._retriever._config.hybrid_alpha = 0.7
    return engine


def _forwarded_signal(engine: VectorCypherEngine) -> Any:
    """Pull the temporal_signal the engine forwarded into retriever.retrieve."""
    assert engine._retriever.retrieve.await_count >= 1, "retriever.retrieve was never called"
    return engine._retriever.retrieve.await_args.kwargs["temporal_signal"]


class TestDateFilterSynthesizesExplicit:
    """A date-key caller filter yields an EXPLICIT, high-confidence signal."""

    @pytest.mark.parametrize(
        "doc",
        [
            {"occurred_at": {"$gte": "2026-04-05"}},
            {"created_at": {"$gte": "2026-04-05"}},
            {"occurred_at": {"$gte": "2026-04-05", "$lte": "2026-04-30"}},
        ],
        ids=["occurred_at_gte", "created_at_gte", "occurred_at_range"],
    )
    async def test_date_filter_is_explicit(self, doc: dict[str, Any]) -> None:
        """A filter that constrains occurred_at / created_at -> EXPLICIT signal.

        EXPLICIT is the engine's "API-asserted temporal intent" category: it
        carries confidence 1.0 and source 'api', and downstream code applies the
        recency-weighting / version-filter path that a high-confidence temporal
        query gets — the same treatment an API ``temporal_filter`` would receive.
        """
        engine = _connected_engine()
        await engine.recall("any phrasing at all", uuid4(), filter_ast=_ast(doc))

        signal = _forwarded_signal(engine)
        assert signal is not None
        assert signal.category == TemporalCategory.EXPLICIT, (
            f"date-key filter did not synthesize EXPLICIT; got {signal.category}"
        )
        assert signal.is_temporal is True
        # API-asserted: confidence 1.0, source 'api' (disambiguates from the
        # dictionary / semantic / none detector sources in traces).
        assert signal.confidence == 1.0
        assert signal.source == "api"

    async def test_date_filter_without_temporal_filter_keeps_signal_filter_none(self) -> None:
        """When only the caller filter carries the date, the synthesized signal's
        own ``temporal_filter`` stays None.

        The date predicate is enforced via ``filter_ast`` on the channels; the
        version-filter block no-ops on a None signal filter, so the date is not
        double-applied through two different mechanisms.
        """
        engine = _connected_engine()
        await engine.recall("any phrasing", uuid4(), filter_ast=_ast({"occurred_at": {"$gte": "2026-04-05"}}))

        signal = _forwarded_signal(engine)
        assert signal.category == TemporalCategory.EXPLICIT
        assert signal.temporal_filter is None


class TestNonDateFilterDoesNotForceExplicit:
    """A non-date caller filter runs the normal detector, not EXPLICIT-from-API."""

    async def test_metadata_channel_filter_is_not_api_explicit(self) -> None:
        """A pure ``metadata.channel`` filter on a time-blind query is NOT EXPLICIT.

        It must run the normal query-string temporal detector (source != 'api')
        and ride alongside as ``filter_ast`` — it does not earn the
        API-asserted high-confidence recency treatment a date predicate gets.
        """
        engine = _connected_engine()
        await engine.recall(
            "what is the architecture of the system",  # no temporal cue
            uuid4(),
            filter_ast=_ast({"metadata.channel": "alpha"}),
        )

        signal = _forwarded_signal(engine)
        # The normal detector ran (source is one of dictionary / semantic / none),
        # NOT the api-asserted EXPLICIT synthesis.
        assert signal.source != "api", "a non-date metadata filter wrongly triggered the API EXPLICIT synthesis"
        assert not (signal.category == TemporalCategory.EXPLICIT and signal.confidence == 1.0), (
            "metadata.channel filter must not be promoted to API-confidence EXPLICIT"
        )

    async def test_no_filter_time_blind_query_runs_normal_detector(self) -> None:
        """Control: no filter + a time-blind query -> detector source != 'api'.

        Proves the EXPLICIT synthesis is gated on the date predicate, not a
        side-effect of the recall path itself.
        """
        engine = _connected_engine()
        await engine.recall("what is the architecture of the system", uuid4(), filter_ast=None)

        signal = _forwarded_signal(engine)
        assert signal.source != "api"


class TestUnderFilledCounter:
    """A filtered recall returning fewer than the requested limit emits the counter."""

    async def test_under_filled_fires_when_filtered_recall_below_limit(self, monkeypatch: Any) -> None:
        """The mock retriever returns zero chunks; a filtered recall under k records it.

        Wires the declared ``khora.recall.filter.under_filled`` counter. The
        engine imports the helper locally inside ``recall``, so patching the
        source module attribute is picked up at call time.
        """
        import khora.filter.telemetry as tel

        calls: list[int] = []
        monkeypatch.setattr(tel, "record_under_filled", lambda: calls.append(1))

        engine = _connected_engine()  # retriever.retrieve returns 0 chunks
        await engine.recall("any query", uuid4(), limit=10, filter_ast=_ast({"metadata.channel": "alpha"}))

        assert calls, "under_filled not recorded for a filtered recall that returned fewer than the limit"

    async def test_under_filled_not_fired_without_filter(self, monkeypatch: Any) -> None:
        """Control: an unfiltered recall never emits the filter under-filled counter."""
        import khora.filter.telemetry as tel

        calls: list[int] = []
        monkeypatch.setattr(tel, "record_under_filled", lambda: calls.append(1))

        engine = _connected_engine()
        await engine.recall("any query", uuid4(), limit=10, filter_ast=None)

        assert not calls, "under_filled fired for an unfiltered recall"
