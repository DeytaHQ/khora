"""Additional coverage for ``khora.engines.skeleton.engine``.

Complements ``test_engine_paths.py`` by exercising the remaining surfaces:

- ``remember()`` duplicate-detection short-circuit
- ``_process_document()`` empty-chunks + happy path
- ``remember_batch()`` empty-list and all-skipped paths
- ``stats()`` aggregation with all-success and partial-failure modes
- ``health_check()`` connected/disconnected branches
- Trivial delegating methods (``create_namespace``, ``get_namespace``,
  ``get_entity``, ``list_entities``, ``get_document``, ``list_documents``)
- ``find_related_entities`` / ``search_entities`` NotImplementedError surfaces

All tests stub I/O — no real DB, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.engines.skeleton.engine import SkeletonConstructionEngine


def _mock_config(*, backend: str = "pgvector") -> MagicMock:
    config = MagicMock()
    config.storage.backend = backend
    config.storage.surrealdb = MagicMock()
    config.storage.sqlite_lance = MagicMock()
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.use_halfvec = True
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


def _connected(backend: str = "pgvector") -> SkeletonConstructionEngine:
    eng = SkeletonConstructionEngine(_mock_config(backend=backend), backend=backend)
    eng._connected = True
    eng._storage = AsyncMock()
    eng._embedder = AsyncMock()
    eng._temporal_store = AsyncMock()
    return eng


# ---------------------------------------------------------------------------
# remember() — duplicate detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberDuplicate:
    @pytest.mark.asyncio
    async def test_returns_existing_doc_for_duplicate_checksum(self) -> None:
        eng = _connected()
        existing = MagicMock()
        existing.id = uuid4()
        existing.chunk_count = 5
        existing.entity_count = 3
        existing.relationship_count = 2
        existing.status = "COMPLETED"
        eng._storage.get_document_by_checksum = AsyncMock(return_value=existing)

        ns = uuid4()
        result = await eng.remember(
            "hello world",
            ns,
            entity_types=[],
            relationship_types=[],
        )

        assert result.document_id == existing.id
        assert result.chunks_created == 5
        assert result.metadata["duplicate"] is True
        # Should not have created a new document
        eng._storage.create_document.assert_not_called()


# ---------------------------------------------------------------------------
# _process_document() — empty chunks short circuit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessDocument:
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_zeros(self) -> None:
        eng = _connected()

        # Build a document with metadata.custom dict
        document = MagicMock()
        document.id = uuid4()
        document.namespace_id = uuid4()
        document.content = ""
        document.metadata.custom = {}
        document.mark_completed = MagicMock()

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []  # empty chunk list

        with patch(
            "khora.extraction.chunkers.create_chunker",
            return_value=mock_chunker,
        ):
            chunks, entities, rels = await eng._process_document(
                document,
                skill_name="general",
                occurred_at=datetime.now(UTC),
            )

        assert (chunks, entities, rels) == (0, 0, 0)
        document.mark_completed.assert_called_once_with(0, 0)
        eng._storage.update_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_happy_path_chunks_embedded_and_stored(self) -> None:
        eng = _connected()
        document = MagicMock()
        document.id = uuid4()
        document.namespace_id = uuid4()
        document.content = "lorem ipsum"
        document.metadata.custom = {"source_system": "slack", "author": "alice"}
        document.mark_completed = MagicMock()

        raw = SimpleNamespace(content="chunk text", start_char=0, end_char=10)
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [raw]

        eng._embedder.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])
        eng._temporal_store.create_chunks_batch = AsyncMock(return_value=[object()])

        with patch(
            "khora.extraction.chunkers.create_chunker",
            return_value=mock_chunker,
        ):
            chunks, entities, rels = await eng._process_document(
                document,
                skill_name="general",
                occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
            )

        assert chunks == 1
        assert entities == 0
        assert rels == 0
        eng._temporal_store.create_chunks_batch.assert_awaited_once()
        document.mark_completed.assert_called_once_with(1, 0)


# ---------------------------------------------------------------------------
# remember_batch() — empty + all-skipped
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberBatch:
    @pytest.mark.asyncio
    async def test_empty_documents_returns_zero_batch_result(self) -> None:
        eng = _connected()
        result = await eng.remember_batch(
            [],
            uuid4(),
            entity_types=[],
            relationship_types=[],
        )
        assert result.total == 0
        assert result.processed == 0
        assert result.skipped == 0
        assert result.failed == 0
        assert result.chunks == 0

    @pytest.mark.asyncio
    async def test_all_duplicates_returns_skipped(self) -> None:
        """Every checksum is in existing_docs -> all skipped, early return."""
        eng = _connected()
        eng._storage.get_documents_by_checksums = AsyncMock(
            return_value={"abc": MagicMock()},  # any non-empty mapping shape
        )
        # Patch hashlib so every checksum collides to the same key the mock returns.
        with patch(
            "khora.engines.skeleton.engine.hashlib.sha256",
        ) as fake_sha:
            fake_digest = MagicMock()
            fake_digest.hexdigest.return_value = "abc"
            fake_sha.return_value = fake_digest

            result = await eng.remember_batch(
                [{"content": "a"}, {"content": "b"}],
                uuid4(),
                entity_types=[],
                relationship_types=[],
            )

        # Both should be skipped: one because the checksum exists, the other
        # because it's a duplicate within the batch.
        assert result.total == 2
        assert result.processed == 0
        assert result.skipped == 2

    @pytest.mark.asyncio
    async def test_progress_callback_fired_on_early_return(self) -> None:
        eng = _connected()
        eng._storage.get_documents_by_checksums = AsyncMock(return_value={"abc": MagicMock()})
        with patch(
            "khora.engines.skeleton.engine.hashlib.sha256",
        ) as fake_sha:
            fake_sha.return_value.hexdigest.return_value = "abc"

            progress: list[tuple[int, int]] = []
            await eng.remember_batch(
                [{"content": "a"}],
                uuid4(),
                entity_types=[],
                relationship_types=[],
                on_progress=lambda c, t: progress.append((c, t)),
            )
        assert progress == [(1, 1)]

    @pytest.mark.asyncio
    async def test_progress_fires_per_document_not_once_at_end(self) -> None:
        """#898: on_progress fires once per document with incrementing count,
        not a single (total, total) call at the end."""
        eng = _connected()
        eng._storage.get_documents_by_checksums = AsyncMock(return_value={})

        def _make_doc(content: str) -> MagicMock:
            doc = MagicMock()
            doc.id = uuid4()
            doc.namespace_id = uuid4()
            doc.content = content
            doc.mark_completed = MagicMock()
            return doc

        eng._storage.create_document = AsyncMock(side_effect=lambda d: _make_doc(d.content))
        eng._storage.update_document = AsyncMock()

        raw = SimpleNamespace(content="chunk text", start_char=0, end_char=10)
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [raw]
        eng._embedder.embed_batch = AsyncMock(return_value=[[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
        eng._temporal_store.create_chunks_batch = AsyncMock(return_value=[object(), object(), object()])

        progress: list[tuple[int, int]] = []
        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker):
            await eng.remember_batch(
                [{"content": "a"}, {"content": "b"}, {"content": "c"}],
                uuid4(),
                entity_types=[],
                relationship_types=[],
                on_progress=lambda c, t: progress.append((c, t)),
            )

        assert progress == [(1, 3), (2, 3), (3, 3)]


# ---------------------------------------------------------------------------
# stats() — partial failures degrade gracefully
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStats:
    @pytest.mark.asyncio
    async def test_all_metrics_collected(self) -> None:
        eng = _connected()
        ns = uuid4()
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        eng._storage.get_document_stats = AsyncMock(return_value=(10, ts))
        eng._storage.count_chunks = AsyncMock(return_value=42)
        eng._storage.count_entities = AsyncMock(return_value=5)
        eng._storage.count_relationships = AsyncMock(return_value=7)

        out = await eng.stats(ns)

        assert out.documents == 10
        assert out.chunks == 42
        assert out.entities == 5
        assert out.relationships == 7
        assert out.last_activity_at == ts

    @pytest.mark.asyncio
    async def test_missing_methods_degrade_to_zero(self) -> None:
        """Each metric branch independently swallows AttributeError/NotImplementedError."""
        eng = _connected()
        eng._storage.get_document_stats = AsyncMock(side_effect=NotImplementedError())
        eng._storage.count_chunks = AsyncMock(side_effect=AttributeError())
        eng._storage.count_entities = AsyncMock(side_effect=NotImplementedError())
        eng._storage.count_relationships = AsyncMock(side_effect=AttributeError())

        out = await eng.stats(uuid4())
        assert out.documents == 0
        assert out.chunks == 0
        assert out.entities == 0
        assert out.relationships == 0
        assert out.last_activity_at is None


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_disconnected_returns_status_disconnected(self) -> None:
        eng = SkeletonConstructionEngine(_mock_config())
        out = await eng.health_check()
        assert out == {"status": "disconnected"}

    @pytest.mark.asyncio
    async def test_healthy_when_all_components_healthy(self) -> None:
        eng = _connected()
        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {"db": "ok"}
        eng._storage.health_check = AsyncMock(return_value=storage_health)
        eng._temporal_store.health_check = AsyncMock(return_value={"status": "healthy"})

        out = await eng.health_check()
        assert out["status"] == "healthy"
        assert out["backend"] == "pgvector"
        assert out["temporal_store"] == {"status": "healthy"}

    @pytest.mark.asyncio
    async def test_degraded_when_storage_unhealthy(self) -> None:
        eng = _connected()
        storage_health = MagicMock()
        storage_health.is_healthy = False
        storage_health.summary = {}
        eng._storage.health_check = AsyncMock(return_value=storage_health)
        eng._temporal_store.health_check = AsyncMock(return_value={"status": "healthy"})

        out = await eng.health_check()
        assert out["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_degraded_when_temporal_unhealthy(self) -> None:
        eng = _connected()
        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {}
        eng._storage.health_check = AsyncMock(return_value=storage_health)
        eng._temporal_store.health_check = AsyncMock(return_value={"status": "down"})

        out = await eng.health_check()
        assert out["status"] == "degraded"


# ---------------------------------------------------------------------------
# Namespace / Entity / Document delegating methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDelegatingMethods:
    @pytest.mark.asyncio
    async def test_create_namespace_no_overrides(self) -> None:
        eng = _connected()
        sentinel = MagicMock()
        eng._storage.create_namespace = AsyncMock(return_value=sentinel)
        out = await eng.create_namespace()
        assert out is sentinel
        eng._storage.create_namespace.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_namespace_with_overrides(self) -> None:
        eng = _connected()
        sentinel = MagicMock()
        eng._storage.create_namespace = AsyncMock(return_value=sentinel)
        overrides = {"foo": "bar"}
        out = await eng.create_namespace(config_overrides=overrides)
        assert out is sentinel
        call_arg = eng._storage.create_namespace.call_args.args[0]
        assert call_arg.config_overrides == overrides

    @pytest.mark.asyncio
    async def test_get_namespace_delegates(self) -> None:
        eng = _connected()
        sentinel = MagicMock()
        eng._storage.get_namespace = AsyncMock(return_value=sentinel)
        ns_id = uuid4()
        out = await eng.get_namespace(ns_id)
        assert out is sentinel
        eng._storage.get_namespace.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_get_entity_delegates(self) -> None:
        eng = _connected()
        sentinel = MagicMock()
        eng._storage.get_entity = AsyncMock(return_value=sentinel)
        eid, nsid = uuid4(), uuid4()
        out = await eng.get_entity(eid, namespace_id=nsid)
        assert out is sentinel
        eng._storage.get_entity.assert_awaited_once_with(eid, namespace_id=nsid)

    @pytest.mark.asyncio
    async def test_list_entities_delegates(self) -> None:
        eng = _connected()
        eng._storage.list_entities = AsyncMock(return_value=[])
        ns = uuid4()
        out = await eng.list_entities(ns, entity_type="PERSON", limit=50)
        assert out == []
        eng._storage.list_entities.assert_awaited_once_with(ns, entity_type="PERSON", limit=50)

    @pytest.mark.asyncio
    async def test_find_related_entities_not_implemented(self) -> None:
        eng = _connected()
        with pytest.raises(NotImplementedError, match="VectorCypher"):
            await eng.find_related_entities(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_get_document_delegates(self) -> None:
        eng = _connected()
        sentinel = MagicMock()
        eng._storage.get_document = AsyncMock(return_value=sentinel)
        did = uuid4()
        nsid = uuid4()
        out = await eng.get_document(did, namespace_id=nsid)
        assert out is sentinel
        eng._storage.get_document.assert_awaited_once_with(did, namespace_id=nsid)

    @pytest.mark.asyncio
    async def test_list_documents_delegates(self) -> None:
        eng = _connected()
        eng._storage.list_documents = AsyncMock(return_value=[])
        ns = uuid4()
        out = await eng.list_documents(ns, limit=25)
        assert out == []
        eng._storage.list_documents.assert_awaited_once_with(ns, limit=25)

    @pytest.mark.asyncio
    async def test_search_entities_not_implemented(self) -> None:
        eng = _connected()
        with pytest.raises(NotImplementedError, match="VectorCypher"):
            await eng.search_entities("q", uuid4())
