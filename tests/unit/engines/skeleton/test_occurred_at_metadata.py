"""``Skeleton.remember`` honors ``metadata['occurred_at']``.

Previously the single-doc ``remember()`` path silently dropped
``metadata['occurred_at']`` and stamped chunks with ``datetime.now(UTC)``,
while ``remember_batch`` (lines 746-752) parsed the same key. These tests
pin the symmetry so the bug can't regress.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.engines.skeleton.engine import SkeletonConstructionEngine


def _mock_config() -> MagicMock:
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.get_neo4j_url.return_value = None
    config.get_neo4j_user.return_value = None
    config.get_neo4j_password.return_value = None
    config.get_neo4j_database.return_value = None
    config.get_graph_config.return_value = None
    config.get_vector_config.return_value = None
    config.storage.embedding_dimension = 1536
    config.storage.backend = "pgvector"
    config.llm.model = "gpt-4o-mini"
    config.llm.embedding_model = "text-embedding-3-small"
    config.llm.embedding_dimension = 1536
    config.llm.timeout = 30
    config.llm.max_retries = 3
    config.pipeline.chunking_strategy = "recursive"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 200
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


def _connected_engine() -> SkeletonConstructionEngine:
    """Build an engine with mocked storage/embedder/temporal store."""
    engine = SkeletonConstructionEngine(_mock_config(), backend="pgvector")
    engine._connected = True
    engine._storage = AsyncMock()
    engine._embedder = AsyncMock()
    engine._temporal_store = AsyncMock()

    # remember() looks up duplicates via checksum first; ensure it's a miss.
    engine._storage.get_document_by_checksum = AsyncMock(return_value=None)

    # create_document echoes back the document with an id assigned.
    async def _create_document(doc):
        doc.id = uuid4()
        return doc

    engine._storage.create_document = AsyncMock(side_effect=_create_document)
    return engine


@pytest.mark.unit
class TestRememberOccurredAtMetadata:
    """``remember()`` must read ``metadata['occurred_at']`` like ``remember_batch``."""

    @pytest.mark.asyncio
    async def test_metadata_occurred_at_is_honored(self) -> None:
        """ISO timestamp in metadata becomes the chunk's occurred_at."""
        engine = _connected_engine()
        namespace_id = uuid4()
        expected = datetime(2026, 4, 25, 10, 0, 0, tzinfo=UTC)

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            await engine.remember(
                "hello world",
                namespace_id,
                metadata={"occurred_at": "2026-04-25T10:00:00Z"},
                entity_types=[],
                relationship_types=[],
            )

        assert proc.await_count == 1
        assert proc.await_args.kwargs["occurred_at"] == expected

    @pytest.mark.asyncio
    async def test_invalid_occurred_at_falls_back_to_now(self) -> None:
        """Unparseable values fall back to ``now(UTC)``, matching remember_batch."""
        engine = _connected_engine()
        namespace_id = uuid4()
        before = datetime.now(UTC)

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            await engine.remember(
                "hello world",
                namespace_id,
                metadata={"occurred_at": "not-a-date"},
                entity_types=[],
                relationship_types=[],
            )

        after = datetime.now(UTC)
        passed = proc.await_args.kwargs["occurred_at"]
        assert before <= passed <= after

    @pytest.mark.asyncio
    async def test_missing_occurred_at_defaults_to_now(self) -> None:
        """No metadata key, no kwarg → still defaults to now(UTC) (no regression)."""
        engine = _connected_engine()
        namespace_id = uuid4()
        before = datetime.now(UTC)

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            await engine.remember(
                "hello world",
                namespace_id,
                entity_types=[],
                relationship_types=[],
            )

        after = datetime.now(UTC)
        passed = proc.await_args.kwargs["occurred_at"]
        assert before <= passed <= after

    @pytest.mark.asyncio
    async def test_explicit_kwarg_wins_over_metadata(self) -> None:
        """When both are provided, the explicit ``occurred_at=`` kwarg wins."""
        engine = _connected_engine()
        namespace_id = uuid4()
        explicit = datetime(2026, 1, 1, tzinfo=UTC)

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            await engine.remember(
                "hello world",
                namespace_id,
                metadata={"occurred_at": "2099-12-31T00:00:00Z"},
                occurred_at=explicit,
                entity_types=[],
                relationship_types=[],
            )

        assert proc.await_args.kwargs["occurred_at"] == explicit


@pytest.mark.unit
class TestRememberRememberBatchParity:
    """Parity check: same metadata input → same ``occurred_at`` resolution."""

    @pytest.mark.asyncio
    async def test_same_metadata_same_occurred_at(self) -> None:
        """``remember`` and ``remember_batch`` must agree on ``metadata['occurred_at']``."""
        # Reference value: what remember_batch's inline parsing produces.
        engine = SkeletonConstructionEngine(_mock_config(), backend="pgvector")
        reference = engine._parse_datetime("2026-04-25T10:00:00Z")

        # Single-doc path goes through the new resolution logic.
        single_engine = _connected_engine()
        namespace_id = uuid4()

        with patch.object(single_engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            await single_engine.remember(
                "hello world",
                namespace_id,
                metadata={"occurred_at": "2026-04-25T10:00:00Z"},
                entity_types=[],
                relationship_types=[],
            )

        single_resolved = proc.await_args.kwargs["occurred_at"]
        assert single_resolved == reference

    @pytest.mark.asyncio
    async def test_invalid_value_parity(self) -> None:
        """Both paths fall back to ``now(UTC)`` for unparseable inputs."""
        engine = _connected_engine()
        namespace_id = uuid4()

        with patch.object(engine, "_process_document", new_callable=AsyncMock, return_value=(1, 0, 0)) as proc:
            t0 = datetime.now(UTC)
            await engine.remember(
                "hello world",
                namespace_id,
                metadata={"occurred_at": "garbage"},
                entity_types=[],
                relationship_types=[],
            )
            t1 = datetime.now(UTC)

        passed = proc.await_args.kwargs["occurred_at"]
        # remember_batch's same-shaped fallback yields a value within [t0, t1].
        # Allow a tiny buffer for clock granularity.
        assert t0 - timedelta(seconds=1) <= passed <= t1 + timedelta(seconds=1)


@pytest.mark.unit
class TestRememberSourceTimestampPropagation:
    """``source_timestamp`` must flow into the chunk's ``occurred_at`` (#856).

    Resolution order: explicit ``occurred_at`` > ``metadata['occurred_at']`` >
    ``source_timestamp`` > ``datetime.now(UTC)``.
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
        """Mirror reporter's repro: |occurred_at - source_timestamp| < 60s (#856)."""
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
