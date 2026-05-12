"""``VectorCypherEngine.remember`` honors ``metadata['occurred_at']``.

Pre-DYT-3581 the single-doc ``remember()`` path silently dropped
``metadata['occurred_at']`` and stamped chunks with ``datetime.now(UTC)``,
while ``remember_batch`` (engine.py:2118-2120) parsed the same key. These
tests pin the symmetry so the bug can't regress. Mirrors PR #484
for the Skeleton engine.
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
    config.get_graph_config.return_value = None
    config.get_vector_config.return_value = None
    config.storage.embedding_dimension = 1536
    config.storage.backend = "postgres"
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


def _connected_engine() -> VectorCypherEngine:
    """Build an engine with mocked storage/embedder so remember() can run."""
    engine = VectorCypherEngine(_mock_config())
    engine._connected = True
    engine._storage = AsyncMock()
    engine._embedder = AsyncMock()

    # remember() looks up by external_id only when external_id is given;
    # always miss on checksum so we go down the create_document path.
    engine._storage.get_document_by_checksum = AsyncMock(return_value=None)
    engine._storage.get_document_by_external_id = AsyncMock(return_value=None)

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
        engine = VectorCypherEngine(_mock_config())
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
