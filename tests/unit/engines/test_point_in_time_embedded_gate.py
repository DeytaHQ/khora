"""Occurred-bounds temporal recall on the embedded sqlite_lance backend.

The embedded layer has no ``version_valid_from/to`` columns, so the
``_version_filter_entities`` path cannot do point-in-time *entity-version*
narrowing. Previously the retriever fail-fasted with ``NotImplementedError``
for any target_date on sqlite_lance. That blanket gate was replaced by a
call-site guard inside ``_vectorcypher_retrieve``: it skips only the
entity-version filtering (recording a structured degradation) while the
occurred-bounds chunk filter (start_time/end_time) still pushes down to
``khora_chunks.occurred_at``. So an occurred-bounds recall now falls through
to the normal retrieval path instead of raising; the production stack
(PostgreSQL+Neo4j) keeps working unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.retriever import VectorCypherRetriever
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.engines.vectorcypher.temporal_detection import TemporalCategory, TemporalSignal
from khora.storage.temporal import TemporalFilter as SkeletonTemporalFilter


def _make_retriever(backend: str) -> VectorCypherRetriever:
    """Build a retriever wired with mocks; only the gate path is exercised."""
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 16)
    embedder.model_name = "mock"
    embedder.dimension = 16

    retriever = VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=AsyncMock() if backend == "postgres" else None,
        embedder=embedder,
        backend=backend,
    )
    # Force a deterministic SIMPLE route so we can verify the non-gated path
    # falls through to ``_simple_retrieve`` (which we stub) without touching
    # any real storage.
    retriever._router.route = AsyncMock(  # type: ignore[method-assign]
        return_value=RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="test",
        )
    )
    retriever._simple_retrieve = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
    return retriever


def _explicit_signal(target: datetime) -> TemporalSignal:
    return TemporalSignal(
        is_temporal=True,
        category=TemporalCategory.EXPLICIT,
        confidence=0.9,
        source="dictionary",
        temporal_filter=SkeletonTemporalFilter(occurred_before=target),
    )


@pytest.mark.unit
class TestEmbeddedPointInTimeGate:
    @pytest.mark.asyncio
    async def test_embedded_falls_through_with_target_date_in_signal(self) -> None:
        """sqlite_lance + EXPLICIT temporal signal carrying a date no longer raises.

        The blanket gate is gone; an occurred-bounds recall falls through to the
        normal retrieval path (here the stubbed SIMPLE route → ``_simple_retrieve``).
        """
        retriever = _make_retriever("sqlite_lance")
        signal = _explicit_signal(datetime(2024, 1, 1, tzinfo=UTC))

        await retriever.retrieve("what was X in 2024", uuid4(), temporal_signal=signal)

        retriever._simple_retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedded_falls_through_with_target_date_in_filter(self) -> None:
        """sqlite_lance + user-passed occurred-bounds temporal_filter no longer raises."""
        retriever = _make_retriever("sqlite_lance")
        tf = SkeletonTemporalFilter(occurred_after=datetime(2024, 1, 1, tzinfo=UTC))

        await retriever.retrieve("changes since Jan", uuid4(), temporal_filter=tf)

        retriever._simple_retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_pgvector_does_not_raise_with_target_date(self) -> None:
        """Production backend (postgres) lets target_date queries through."""
        retriever = _make_retriever("postgres")
        signal = _explicit_signal(datetime(2024, 1, 1, tzinfo=UTC))

        await retriever.retrieve("what was X in 2024", uuid4(), temporal_signal=signal)

        retriever._simple_retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedded_without_target_date_does_not_raise(self) -> None:
        """sqlite_lance recall without a date works normally (no gate fires)."""
        retriever = _make_retriever("sqlite_lance")

        await retriever.retrieve("plain query", uuid4())

        retriever._simple_retrieve.assert_called_once()
