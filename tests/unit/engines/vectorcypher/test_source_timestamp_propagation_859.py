"""``VectorCypherEngine.remember`` honors ``source_timestamp`` (#859).

The VectorCypher engine used to resolve ``occurred_at`` from
``occurred_at`` kwarg → ``metadata["occurred_at"]`` → ``datetime.now(UTC)``,
silently dropping ``source_timestamp`` even though the kwarg was in scope
and persisted on ``Document``. The fix mirrors the Skeleton resolution
chain (#856): explicit ``occurred_at`` > ``metadata["occurred_at"]`` >
``source_timestamp`` > ``now(UTC)``. These tests pin the symmetry so the
bug can't regress.

Slice B (the recall-side dropout in ``retriever.py``) is covered in
``test_source_timestamp_recall_859.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.engine import VectorCypherEngine


def _mock_config() -> MagicMock:
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
    config.llm.model = "gpt-4o-mini"
    config.llm.timeout = 30
    config.pipeline.extract_entities = True
    config.pipeline.chunking_strategy = "recursive"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 200
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


def _connected_engine() -> VectorCypherEngine:
    """Build a VectorCypher engine with mocked storage / embedder / dual nodes."""
    engine = VectorCypherEngine(_mock_config())
    engine._connected = True
    engine._storage = AsyncMock()
    engine._temporal_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._dual_nodes = AsyncMock()
    engine._retriever = AsyncMock()
    engine._router = MagicMock()
    engine._neo4j_driver = AsyncMock()

    # remember() looks up duplicates and external_id rows first; both miss.
    engine._storage.get_document_by_checksum = AsyncMock(return_value=None)
    engine._storage.get_document_by_external_id = AsyncMock(return_value=None)

    async def _create_document(doc):
        doc.id = uuid4()
        return doc

    engine._storage.create_document = AsyncMock(side_effect=_create_document)
    return engine


@pytest.mark.unit
class TestRememberSourceTimestampPropagation859:
    """``source_timestamp`` must flow into ``occurred_at`` (#859).

    Resolution order: explicit ``occurred_at`` > ``metadata['occurred_at']`` >
    ``source_timestamp`` > ``datetime.now(UTC)``. Mirrors the Skeleton
    contract pinned by #856.
    """

    @pytest.mark.asyncio
    async def test_source_timestamp_propagates_to_occurred_at(self) -> None:
        """``source_timestamp`` becomes ``occurred_at`` when nothing else is set."""
        engine = _connected_engine()
        namespace_id = uuid4()
        intended = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            await engine.remember(
                "hello world",
                namespace_id,
                source_timestamp=intended,
                entity_types=[],
                relationship_types=[],
            )

        assert proc.await_args.kwargs["occurred_at"] == intended

    @pytest.mark.asyncio
    async def test_explicit_occurred_at_wins_over_source_timestamp(self) -> None:
        """``occurred_at=`` kwarg beats ``source_timestamp``."""
        engine = _connected_engine()
        namespace_id = uuid4()
        explicit = datetime(2026, 1, 1, tzinfo=UTC)
        source_ts = datetime(2024, 6, 15, tzinfo=UTC)

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            await engine.remember(
                "hello world",
                namespace_id,
                occurred_at=explicit,
                source_timestamp=source_ts,
                entity_types=[],
                relationship_types=[],
            )

        assert proc.await_args.kwargs["occurred_at"] == explicit

    @pytest.mark.asyncio
    async def test_metadata_occurred_at_wins_over_source_timestamp(self) -> None:
        """``metadata['occurred_at']`` beats ``source_timestamp``."""
        engine = _connected_engine()
        namespace_id = uuid4()
        meta_value = datetime(2025, 3, 10, 9, 0, 0, tzinfo=UTC)
        source_ts = datetime(2024, 6, 15, tzinfo=UTC)

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            await engine.remember(
                "hello world",
                namespace_id,
                metadata={"occurred_at": "2025-03-10T09:00:00Z"},
                source_timestamp=source_ts,
                entity_types=[],
                relationship_types=[],
            )

        assert proc.await_args.kwargs["occurred_at"] == meta_value

    @pytest.mark.asyncio
    async def test_fallback_to_now_when_nothing_supplied(self) -> None:
        """No kwargs, no metadata, no source_timestamp - fall back to now(UTC)."""
        engine = _connected_engine()
        namespace_id = uuid4()

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            before = datetime.now(UTC)
            await engine.remember(
                "hello world",
                namespace_id,
                entity_types=[],
                relationship_types=[],
            )
            after = datetime.now(UTC)

        passed = proc.await_args.kwargs["occurred_at"]
        assert before - timedelta(seconds=1) <= passed <= after + timedelta(seconds=1)

    @pytest.mark.asyncio
    async def test_reporter_invariant_chunk_occurred_at_within_60s(self) -> None:
        """Reporter's repro: |occurred_at - source_timestamp| < 60s (#859)."""
        engine = _connected_engine()
        namespace_id = uuid4()
        intended = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            await engine.remember(
                "hello world",
                namespace_id,
                source_timestamp=intended,
                entity_types=[],
                relationship_types=[],
            )

        passed = proc.await_args.kwargs["occurred_at"]
        assert abs((passed - intended).total_seconds()) < 60
