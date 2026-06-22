"""Correctness regressions for VectorCypherEngine (#1116, #1117, #1156).

#1116 (shared-state race): recall() must NOT mutate the shared
       ``retriever._config.hybrid_alpha`` to implement per-call hybrid_alpha /
       SearchMode.ALL. The effective alpha must be threaded as an explicit
       ``hybrid_alpha_override`` parameter so concurrent recalls cannot corrupt
       each other's blend factor.

#1117 (race-retry resets provenance): the ``(namespace_id, external_id)``
       IntegrityError race-retry path must forward the caller-supplied
       ``source_type`` / ``source_name`` / ``source_url`` to
       ``_remember_via_replace`` (matching the primary dispatch), instead of
       silently falling back to the defaults and corrupting the winner's
       provenance.

#1156 (protocol violation): recall() must accept ``recency_bias`` per
       MemoryEngineProtocol.recall so callers coding against the protocol do not
       crash with TypeError on the default engine.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Document
from khora.engines.protocol import MemoryEngineProtocol
from khora.engines.vectorcypher.engine import VectorCypherEngine
from khora.khora import RememberResult
from khora.query import SearchMode


def _make_config() -> MagicMock:
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
    return config


def _make_recall_result():
    from khora.engines.vectorcypher.retriever import VectorCypherResult
    from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

    routing = RoutingDecision(
        complexity=QueryComplexity.SIMPLE,
        use_graph=False,
        graph_depth=0,
        confidence=0.8,
        reasoning="test",
    )
    chunk = Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="A chunk with enough content to pass the validation threshold for recall.",
    )
    return VectorCypherResult(
        chunks=[(chunk, 0.9)],
        entities=[],
        routing_decision=routing,
        metadata={"search_mode": "simple_vector"},
    )


def _connected_engine() -> VectorCypherEngine:
    engine = VectorCypherEngine(_make_config())
    engine._connected = True
    engine._storage = AsyncMock()
    engine._temporal_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._dual_nodes = AsyncMock()
    engine._neo4j_driver = AsyncMock()

    engine._retriever = MagicMock()
    engine._retriever._config = MagicMock()
    engine._retriever._config.hybrid_alpha = 0.7
    engine._retriever.retrieve = AsyncMock(return_value=_make_recall_result())
    return engine


# ---------------------------------------------------------------------------
# #1116: shared-state race on hybrid_alpha
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecallHybridAlphaNoSharedMutation:
    @pytest.mark.asyncio
    async def test_explicit_hybrid_alpha_threaded_as_override_not_mutated(self) -> None:
        """An explicit hybrid_alpha kwarg must be passed to retrieve() as an
        explicit ``hybrid_alpha_override`` and must NOT mutate the shared
        ``retriever._config.hybrid_alpha``."""
        engine = _connected_engine()
        await engine.recall("test query", uuid4(), hybrid_alpha=0.2)

        # Shared config untouched.
        assert engine._retriever._config.hybrid_alpha == 0.7

        engine._retriever.retrieve.assert_awaited_once()
        kwargs = engine._retriever.retrieve.await_args.kwargs
        assert kwargs["hybrid_alpha_override"] == 0.2

    @pytest.mark.asyncio
    async def test_all_mode_threads_half_override_not_mutated(self) -> None:
        """SearchMode.ALL must thread hybrid_alpha_override=0.5 explicitly and
        leave the shared config untouched."""
        engine = _connected_engine()
        await engine.recall("test query", uuid4(), mode=SearchMode.ALL)

        assert engine._retriever._config.hybrid_alpha == 0.7

        kwargs = engine._retriever.retrieve.await_args.kwargs
        assert kwargs["hybrid_alpha_override"] == 0.5

    @pytest.mark.asyncio
    async def test_default_mode_threads_none_override(self) -> None:
        """Default HYBRID mode without explicit alpha threads no override
        (None) so the retriever uses its configured behaviour."""
        engine = _connected_engine()
        await engine.recall("test query", uuid4())

        assert engine._retriever._config.hybrid_alpha == 0.7

        kwargs = engine._retriever.retrieve.await_args.kwargs
        assert kwargs["hybrid_alpha_override"] is None

    @pytest.mark.asyncio
    async def test_concurrent_recalls_do_not_corrupt_each_other(self) -> None:
        """Two overlapping recalls with different alpha must each observe their
        own blend factor and never leave the shared config corrupted.

        The retriever is made to await an event mid-retrieve so the two recalls
        interleave across the awaits, reproducing the #1116 race window.
        """
        engine = _connected_engine()

        gate = asyncio.Event()
        captured: list[float | None] = []

        async def fake_retrieve(*args, **kwargs):
            captured.append(kwargs.get("hybrid_alpha_override"))
            # Hold both calls inside retrieve() simultaneously so any mutation
            # of shared config by one would be observed by the other.
            await gate.wait()
            return _make_recall_result()

        engine._retriever.retrieve = AsyncMock(side_effect=fake_retrieve)

        task_a = asyncio.create_task(engine.recall("query a", uuid4(), hybrid_alpha=0.1))
        task_b = asyncio.create_task(engine.recall("query b", uuid4(), mode=SearchMode.ALL))

        # Let both reach the gate (both inside retrieve concurrently).
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(task_a, task_b)

        # Each call threaded its own override; no cross-contamination.
        assert sorted(v for v in captured if v is not None) == [0.1, 0.5]
        # Shared config is never written, so it cannot be left at the wrong value.
        assert engine._retriever._config.hybrid_alpha == 0.7


# ---------------------------------------------------------------------------
# #1117: race-retry preserves provenance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberRaceRetryPreservesProvenance:
    @pytest.mark.asyncio
    async def test_integrity_error_retry_forwards_provenance(self) -> None:
        """When create_document hits an IntegrityError on the
        (namespace_id, external_id) race, the retry routes to
        _remember_via_replace WITH the caller-supplied source_type /
        source_name / source_url, not the defaults."""
        from sqlalchemy.exc import IntegrityError

        engine = VectorCypherEngine(_make_config())
        engine._connected = True

        ns_id = uuid4()
        winner = Document(id=uuid4(), namespace_id=ns_id, content="winner", external_id="ext-1")

        storage = AsyncMock()
        # Pre-check finds nothing -> proceed to create.
        # Post-race retry finds the winner.
        storage.get_document_by_external_id = AsyncMock(side_effect=[None, winner])
        storage.get_document_by_checksum = AsyncMock(return_value=None)
        storage.create_document = AsyncMock(
            side_effect=IntegrityError("INSERT documents", {}, Exception("duplicate external_id"))
        )
        engine._storage = storage

        captured: dict = {}

        async def fake_replace(**kwargs):
            captured.update(kwargs)
            return RememberResult(
                document_id=winner.id,
                namespace_id=ns_id,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                metadata={},
            )

        engine._remember_via_replace = AsyncMock(side_effect=fake_replace)

        await engine.remember(
            content="loser content",
            namespace_id=ns_id,
            title="My Title",
            source="my-source",
            source_type="email",
            source_name="inbox-connector",
            source_url="https://example.com/msg/1",
            source_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            metadata={"k": "v"},
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
            external_id="ext-1",
        )

        engine._remember_via_replace.assert_awaited_once()
        assert captured["source_type"] == "email"
        assert captured["source_name"] == "inbox-connector"
        assert captured["source_url"] == "https://example.com/msg/1"
        # Sanity: the provenance fields the path already carried still survive.
        assert captured["source_timestamp"] == datetime(2025, 1, 1, tzinfo=UTC)
        assert captured["metadata"] == {"k": "v"}
        assert captured["title"] == "My Title"


# ---------------------------------------------------------------------------
# #1156: recall accepts recency_bias per protocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecallRecencyBiasProtocolParity:
    def test_recall_signature_has_recency_bias(self) -> None:
        """VectorCypherEngine.recall must declare recency_bias to match
        MemoryEngineProtocol.recall."""
        sig = inspect.signature(VectorCypherEngine.recall)
        assert "recency_bias" in sig.parameters

        proto_params = set(inspect.signature(MemoryEngineProtocol.recall).parameters)
        engine_params = set(sig.parameters)
        missing = proto_params - engine_params
        assert not missing, f"VectorCypherEngine.recall missing protocol params: {missing}"

    @pytest.mark.asyncio
    async def test_recall_accepts_recency_bias_none(self) -> None:
        """recency_bias=None (the protocol default) must be accepted without
        raising TypeError."""
        engine = _connected_engine()
        result = await engine.recall("test query", uuid4(), recency_bias=None)
        assert result is not None

    @pytest.mark.asyncio
    async def test_recall_threads_recency_bias_to_retriever(self) -> None:
        """A concrete recency_bias must be threaded into retriever.retrieve so
        it actually influences recency weighting (wired, not silently ignored)."""
        engine = _connected_engine()
        await engine.recall("test query", uuid4(), recency_bias=0.8)

        kwargs = engine._retriever.retrieve.await_args.kwargs
        assert kwargs["recency_bias"] == 0.8
