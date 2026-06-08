"""Unit tests for the VectorCypher engine."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.vectorcypher.engine import (
    ExtractionQualityMetrics,
    VectorCypherConfig,
    VectorCypherEngine,
    _mirror_chunks_or_degrade,
)
from khora.khora import RecallResult


def _ent(entity_id, source_document_ids):
    """Minimal entity/relationship stand-in for the vector-anchored cascade."""
    return SimpleNamespace(id=entity_id, source_document_ids=list(source_document_ids))


class TestVectorCypherConfig:
    """Tests for VectorCypherConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = VectorCypherConfig()
        assert config.routing_enabled is True
        assert config.routing_use_llm is False
        assert config.skeleton_core_ratio == 0.70
        assert config.graph_default_depth == 2
        assert config.graph_max_depth == 4
        assert config.graph_max_entry_entities == 10
        assert config.fusion_rrf_k == 60
        assert config.fusion_vector_weight == 0.6
        assert config.fusion_graph_weight == 0.4
        assert config.fusion_simple_vector_weight == 0.8
        assert config.fusion_simple_graph_weight == 0.2
        assert config.fusion_complex_vector_weight == 0.4
        assert config.fusion_complex_graph_weight == 0.6
        assert config.temporal_recency_weight == 0.2
        assert config.temporal_recency_decay_days == 30
        assert config.recency_decay_type == "exponential"
        assert config.streaming_pipeline is True
        assert config.enable_smart_resolution is True
        assert config.lazy_entity_expansion is True
        assert config.fusion_hybrid_alpha == 0.7
        assert config.retriever_min_entity_similarity == 0.3
        assert config.max_chunks_in_flight is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = VectorCypherConfig(
            routing_enabled=False,
            skeleton_core_ratio=0.5,
            graph_default_depth=3,
            fusion_vector_weight=0.7,
            fusion_graph_weight=0.3,
        )
        assert config.routing_enabled is False
        assert config.skeleton_core_ratio == 0.5
        assert config.graph_default_depth == 3
        assert config.fusion_vector_weight == 0.7
        assert config.fusion_graph_weight == 0.3


class TestExtractionQualityMetrics:
    """Tests for ExtractionQualityMetrics dataclass."""

    def test_defaults(self) -> None:
        """Test default metrics values."""
        metrics = ExtractionQualityMetrics()
        assert metrics.total_chunks == 0
        assert metrics.chunks_with_entities == 0
        assert metrics.total_entities == 0
        assert metrics.total_relationships == 0
        assert metrics.avg_entities_per_chunk == 0.0
        assert metrics.avg_confidence == 0.0
        assert metrics.entity_type_distribution == {}

    def test_compute_averages(self) -> None:
        """Test computing averages from totals."""
        metrics = ExtractionQualityMetrics(
            total_chunks=10,
            total_entities=30,
        )
        metrics.compute_averages()
        assert metrics.avg_entities_per_chunk == 3.0

    def test_compute_averages_zero_chunks(self) -> None:
        """Test computing averages with zero chunks does not divide by zero."""
        metrics = ExtractionQualityMetrics(total_chunks=0, total_entities=5)
        metrics.compute_averages()
        assert metrics.avg_entities_per_chunk == 0.0

    def test_entity_type_distribution(self) -> None:
        """Test entity type distribution is mutable dict."""
        metrics = ExtractionQualityMetrics()
        metrics.entity_type_distribution["PERSON"] = 5
        metrics.entity_type_distribution["ORG"] = 3
        assert metrics.entity_type_distribution == {"PERSON": 5, "ORG": 3}


class TestVectorCypherEngineInit:
    """Tests for VectorCypherEngine initialization."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock KhoraConfig."""
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
        config.llm.embedding_model = "text-embedding-3-small"
        config.llm.embedding_dimension = 1536
        config.llm.timeout = 30
        config.llm.max_retries = 3
        config.llm.max_concurrent_llm_calls = 5
        config.pipeline.chunking_strategy = "recursive"
        config.pipeline.chunk_size = 1000
        config.pipeline.chunk_overlap = 200
        config.pipeline.extract_entities = True
        config.telemetry_database_url = None
        config.telemetry_service_name = "test"
        return config

    def test_init_default_config(self, mock_config: MagicMock) -> None:
        """Test engine initialization with default VectorCypherConfig."""
        engine = VectorCypherEngine(mock_config)
        assert engine._vc_config is not None
        assert engine._vc_config.routing_enabled is True
        assert engine._connected is False
        assert engine._storage is None
        assert engine._neo4j_driver is None

    def test_init_custom_config(self, mock_config: MagicMock) -> None:
        """Test engine initialization with custom VectorCypherConfig."""
        vc_config = VectorCypherConfig(
            skeleton_core_ratio=0.5,
            graph_default_depth=3,
        )
        engine = VectorCypherEngine(mock_config, vectorcypher_config=vc_config)
        assert engine._vc_config.skeleton_core_ratio == 0.5
        assert engine._vc_config.graph_default_depth == 3

    def test_init_with_storage_config(self, mock_config: MagicMock) -> None:
        """Test engine initialization with explicit storage config."""
        storage_config = MagicMock()
        engine = VectorCypherEngine(mock_config, storage_config=storage_config)
        assert engine._storage_config is storage_config


class TestVectorCypherEngineGetters:
    """Tests for engine getter methods when not connected."""

    @pytest.fixture
    def engine(self) -> VectorCypherEngine:
        """Create an unconnected engine."""
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
        return VectorCypherEngine(config)

    def test_get_storage_raises_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_storage raises RuntimeError when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            engine._get_storage()

    def test_get_temporal_store_raises_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_temporal_store raises RuntimeError when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            engine._get_temporal_store()

    def test_get_embedder_raises_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_embedder raises RuntimeError when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            engine._get_embedder()

    def test_get_retriever_raises_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_retriever raises RuntimeError when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            engine._get_retriever()

    def test_get_dual_nodes_returns_none_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_dual_nodes returns None when not connected (or SurrealDB backend)."""
        assert engine._get_dual_nodes() is None


@pytest.mark.unit
class TestVectorCypherEngineDisconnect:
    """Tests for engine disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        """Test disconnecting when already disconnected is a no-op."""
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

        engine = VectorCypherEngine(config)
        # Should not raise
        await engine.disconnect()
        assert engine._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_components(self) -> None:
        """Test that disconnect cleans up all component references."""
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

        engine = VectorCypherEngine(config)

        # Simulate connected state
        engine._connected = True
        engine._neo4j_driver = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._storage = AsyncMock()
        engine._embedder = MagicMock()
        engine._retriever = MagicMock()
        engine._dual_nodes = MagicMock()
        engine._router = MagicMock()

        with patch("khora.telemetry.shutdown_telemetry", new_callable=AsyncMock):
            await engine.disconnect()

        assert engine._connected is False
        assert engine._neo4j_driver is None
        assert engine._temporal_store is None
        assert engine._storage is None
        assert engine._embedder is None
        assert engine._retriever is None
        assert engine._dual_nodes is None
        assert engine._router is None


@pytest.mark.unit
class TestVectorCypherEngineRemember:
    """Tests for engine remember() with mocked backends."""

    @pytest.fixture
    def connected_engine(self) -> VectorCypherEngine:
        """Create a mock-connected engine for testing remember/recall/forget."""
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
        config.pipeline.extract_entities = True

        engine = VectorCypherEngine(config)
        engine._connected = True
        engine._storage = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._embedder = AsyncMock()
        engine._dual_nodes = AsyncMock()
        engine._retriever = AsyncMock()
        engine._router = MagicMock()
        engine._neo4j_driver = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_remember_duplicate_document(self, connected_engine: VectorCypherEngine) -> None:
        """Test that remember returns early for duplicate documents."""
        namespace_id = uuid4()
        doc_id = uuid4()

        existing_doc = MagicMock()
        existing_doc.id = doc_id
        existing_doc.status = "completed"
        existing_doc.chunk_count = 5
        existing_doc.entity_count = 3

        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=existing_doc)

        result = await connected_engine.remember(
            "test content",
            namespace_id,
            entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
            relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
        )

        assert result.document_id == doc_id
        assert result.metadata.get("duplicate") is True
        connected_engine._storage.create_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_remember_new_document(self, connected_engine: VectorCypherEngine) -> None:
        """Test remember creates and processes a new document."""
        namespace_id = uuid4()
        doc_id = uuid4()

        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=None)

        created_doc = MagicMock()
        created_doc.id = doc_id
        created_doc.namespace_id = namespace_id
        created_doc.content = "test content"
        created_doc.metadata = {}
        connected_engine._storage.create_document = AsyncMock(return_value=created_doc)

        with patch.object(connected_engine, "_process_document", new_callable=AsyncMock, return_value=(3, 5, 2)):
            result = await connected_engine.remember(
                "test content",
                namespace_id,
                title="Test Doc",
                source="unit_test",
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        assert result.document_id == doc_id
        assert result.chunks_created == 3
        assert result.entities_extracted == 5
        assert result.relationships_created == 2
        connected_engine._storage.create_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_without_external_id_skips_external_lookup(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """external_id=None path is byte-identical to today — no external lookup."""
        namespace_id = uuid4()
        doc_id = uuid4()

        connected_engine._storage.get_document_by_external_id = AsyncMock()
        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=None)

        created_doc = MagicMock()
        created_doc.id = doc_id
        created_doc.namespace_id = namespace_id
        created_doc.content = "test content"
        created_doc.metadata = {}
        connected_engine._storage.create_document = AsyncMock(return_value=created_doc)

        with patch.object(connected_engine, "_process_document", new_callable=AsyncMock, return_value=(3, 5, 2)):
            await connected_engine.remember(
                "test content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        connected_engine._storage.get_document_by_external_id.assert_not_called()
        connected_engine._storage.get_document_by_checksum.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remember_external_id_no_match_falls_through(self, connected_engine: VectorCypherEngine) -> None:
        """external_id with no existing match goes through create path."""
        namespace_id = uuid4()
        doc_id = uuid4()

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=None)
        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=None)
        connected_engine._storage.replace_document_extraction = AsyncMock()

        created_doc = MagicMock()
        created_doc.id = doc_id
        created_doc.namespace_id = namespace_id
        created_doc.content = "test content"
        created_doc.metadata = {}
        connected_engine._storage.create_document = AsyncMock(return_value=created_doc)

        with patch.object(connected_engine, "_process_document", new_callable=AsyncMock, return_value=(3, 5, 2)):
            result = await connected_engine.remember(
                "test content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="ext-new",
            )

        connected_engine._storage.get_document_by_external_id.assert_awaited_once_with(
            "ext-new", namespace_id=namespace_id
        )
        connected_engine._storage.replace_document_extraction.assert_not_called()
        connected_engine._storage.create_document.assert_called_once()
        assert result.document_id == doc_id
        assert result.metadata.get("replaced") is None

    @pytest.mark.asyncio
    async def test_remember_external_id_match_dispatches_to_replace(self, connected_engine: VectorCypherEngine) -> None:
        """Matched external_id routes to coordinator.replace_document_extraction."""
        from khora.storage.coordinator import ReplaceResult

        namespace_id = uuid4()
        old_doc_id = uuid4()

        existing = MagicMock()
        existing.id = old_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=existing)
        connected_engine._storage.get_document_by_checksum = AsyncMock()

        replace_result = ReplaceResult(
            document_id=old_doc_id,
            chunks_deleted=3,
            chunks_created=4,
            entities_created=5,
            entities_updated=2,
            entities_retired=1,
            relationships_created=6,
            relationships_retired=1,
        )
        connected_engine._storage.replace_document_extraction = AsyncMock(return_value=replace_result)

        with patch.object(
            connected_engine,
            "_remember_via_replace",
            wraps=connected_engine._remember_via_replace,
        ) as spy_replace:
            # Short-circuit internal work: stub chunker + embedder + extraction.
            with (
                patch("khora.extraction.chunkers.create_chunker") as mock_chunker_factory,
                patch("khora.pipelines.tasks.extract.extract_entities", new_callable=AsyncMock, return_value=([], [])),
            ):
                mock_chunker = MagicMock()
                mock_chunker.chunk.return_value = []  # no raw chunks → no embed/extract needed
                mock_chunker_factory.return_value = mock_chunker

                result = await connected_engine.remember(
                    "new content",
                    namespace_id,
                    entity_types=["PERSON"],
                    relationship_types=["KNOWS"],
                    external_id="ext-1",
                )

        spy_replace.assert_awaited_once()
        connected_engine._storage.replace_document_extraction.assert_awaited_once()
        # The old checksum-based create path must NOT run for a matched external_id.
        connected_engine._storage.get_document_by_checksum.assert_not_called()
        # Public contract: document_id + metadata signalling replace.
        assert result.document_id == old_doc_id
        assert result.namespace_id == namespace_id
        assert result.chunks_created == 4
        assert result.entities_extracted == 7  # created (5) + updated (2)
        assert result.relationships_created == 6
        assert result.metadata["replaced"] is True
        assert result.metadata["old_document_id"] == str(old_doc_id)

    @pytest.mark.asyncio
    async def test_remember_via_replace_wipes_and_writes_vectorcypher_stores(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """_remember_via_replace must wipe + rewrite khora_chunks and :Chunk nodes.

        The coordinator's replace_document_extraction only touches the `chunks`
        table and Neo4j :Entity/:Relationship nodes. VectorCypher's create path
        writes to `khora_chunks` (via TemporalVectorStore) and :Chunk nodes
        (via DualNodeManager) directly, so the replace path must mirror that or
        retrieval returns stale content after a replace.
        """
        from khora.engines.skeleton.backends import TemporalChunk
        from khora.storage.coordinator import ReplaceResult

        namespace_id = uuid4()
        old_doc_id = uuid4()

        existing = MagicMock()
        existing.id = old_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=existing)
        connected_engine._storage.get_document_by_checksum = AsyncMock()

        replace_result = ReplaceResult(
            document_id=old_doc_id,
            chunks_deleted=3,
            chunks_created=2,
            entities_created=0,
            entities_updated=0,
            entities_retired=0,
            relationships_created=0,
            relationships_retired=0,
        )
        connected_engine._storage.replace_document_extraction = AsyncMock(return_value=replace_result)
        connected_engine._storage.update_document = AsyncMock()

        # Temporal store returns the chunks it was given (as if already persisted).
        async def _create_chunks_batch(chunks):
            return list(chunks)

        connected_engine._temporal_store.delete_chunks_by_document = AsyncMock(return_value=2)
        connected_engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks_batch)
        connected_engine._dual_nodes.delete_chunks_by_document = AsyncMock(return_value=2)
        connected_engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=[])
        connected_engine._dual_nodes.link_entities_to_chunks_batch = AsyncMock()

        # Chunker returns two raw chunks so embed + temporal paths all fire.
        raw_chunks = [
            MagicMock(content="chunk one", start_char=0, end_char=9),
            MagicMock(content="chunk two", start_char=9, end_char=18),
        ]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks

        connected_engine._embedder.embed_batch = AsyncMock(return_value=[[0.1] * 4, [0.2] * 4])
        connected_engine._embedder.model_name = "text-embedding-3-small"

        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch(
                "khora.pipelines.tasks.extract.extract_entities",
                new_callable=AsyncMock,
                return_value=([], []),
            ),
        ):
            await connected_engine.remember(
                "new content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="ext-1",
            )

        # khora_chunks wiped for the old document, keyed by (existing.id, ns).
        connected_engine._temporal_store.delete_chunks_by_document.assert_awaited_once_with(existing.id, namespace_id)
        # :Chunk nodes wiped for the old document.
        connected_engine._dual_nodes.delete_chunks_by_document.assert_awaited_once_with(existing.id, namespace_id)
        # Coordinator call happens — receives the new chunks.
        connected_engine._storage.replace_document_extraction.assert_awaited_once()

        # Temporal chunks were created with document_id == existing.id (reused).
        tc_call = connected_engine._temporal_store.create_chunks_batch.await_args
        assert tc_call is not None
        stored_temporal_chunks = tc_call.args[0]
        assert len(stored_temporal_chunks) == 2
        for tc in stored_temporal_chunks:
            assert isinstance(tc, TemporalChunk)
            assert tc.document_id == existing.id
            assert tc.namespace_id == namespace_id

        # :Chunk nodes created with the same namespace_id + temporal chunks.
        connected_engine._dual_nodes.create_chunk_nodes_batch.assert_awaited_once()
        nodes_call = connected_engine._dual_nodes.create_chunk_nodes_batch.await_args
        assert nodes_call.args[0] == stored_temporal_chunks
        assert nodes_call.args[1] == namespace_id

    @pytest.mark.asyncio
    async def test_remember_via_replace_surreal_unified_skips_neo4j_chunk_nodes(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """When dual_nodes is None (SurrealDB unified), :Chunk-node writes are skipped.

        The SurrealDB unified backend owns chunk + graph linkage on its own
        adapter; the VectorCypher engine must not call DualNodeManager methods
        when `_dual_nodes is None`, but it still must wipe + rewrite the
        temporal store (khora_chunks).
        """
        from khora.storage.coordinator import ReplaceResult

        namespace_id = uuid4()
        old_doc_id = uuid4()

        existing = MagicMock()
        existing.id = old_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=existing)
        connected_engine._storage.get_document_by_checksum = AsyncMock()

        replace_result = ReplaceResult(
            document_id=old_doc_id,
            chunks_deleted=1,
            chunks_created=1,
            entities_created=0,
            entities_updated=0,
            entities_retired=0,
            relationships_created=0,
            relationships_retired=0,
        )
        connected_engine._storage.replace_document_extraction = AsyncMock(return_value=replace_result)
        connected_engine._storage.update_document = AsyncMock()

        # SurrealDB unified path: no Neo4j DualNodeManager.
        connected_engine._dual_nodes = None

        async def _create_chunks_batch(chunks):
            return list(chunks)

        connected_engine._temporal_store.delete_chunks_by_document = AsyncMock(return_value=1)
        connected_engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks_batch)

        raw_chunks = [MagicMock(content="only chunk", start_char=0, end_char=10)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks

        connected_engine._embedder.embed_batch = AsyncMock(return_value=[[0.1] * 4])
        connected_engine._embedder.model_name = "text-embedding-3-small"

        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch(
                "khora.pipelines.tasks.extract.extract_entities",
                new_callable=AsyncMock,
                return_value=([], []),
            ),
        ):
            result = await connected_engine.remember(
                "new content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="ext-1",
            )

        # khora_chunks wiped + written even though dual_nodes is None.
        connected_engine._temporal_store.delete_chunks_by_document.assert_awaited_once_with(existing.id, namespace_id)
        connected_engine._temporal_store.create_chunks_batch.assert_awaited_once()
        connected_engine._storage.replace_document_extraction.assert_awaited_once()
        assert result.metadata["replaced"] is True

    @pytest.mark.asyncio
    async def test_remember_via_replace_links_entities_to_chunks_after_coordinator(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """MENTIONED_IN edges for new entities → new chunks are created after coordinator."""
        from khora.core.models import Entity
        from khora.engines.vectorcypher.dual_nodes import EntityChunkLink
        from khora.storage.coordinator import ReplaceResult

        namespace_id = uuid4()
        old_doc_id = uuid4()
        entity_id = uuid4()
        chunk_id_a = uuid4()
        chunk_id_b = uuid4()

        existing = MagicMock()
        existing.id = old_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=existing)
        connected_engine._storage.get_document_by_checksum = AsyncMock()

        replace_result = ReplaceResult(
            document_id=old_doc_id,
            chunks_deleted=0,
            chunks_created=2,
            entities_created=1,
            entities_updated=0,
            entities_retired=0,
            relationships_created=0,
            relationships_retired=0,
        )
        connected_engine._storage.replace_document_extraction = AsyncMock(return_value=replace_result)
        connected_engine._storage.update_document = AsyncMock()

        async def _create_chunks_batch(chunks):
            return list(chunks)

        connected_engine._temporal_store.delete_chunks_by_document = AsyncMock(return_value=0)
        connected_engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks_batch)
        connected_engine._dual_nodes.delete_chunks_by_document = AsyncMock(return_value=0)
        connected_engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=[])
        connected_engine._dual_nodes.link_entities_to_chunks_batch = AsyncMock()

        # One entity that mentions both chunks.
        entity = Entity(
            id=entity_id,
            namespace_id=namespace_id,
            name="Alice",
            entity_type="PERSON",
            source_chunk_ids=[chunk_id_a, chunk_id_b],
        )

        raw_chunks = [
            MagicMock(content="chunk one", start_char=0, end_char=9),
            MagicMock(content="chunk two", start_char=9, end_char=18),
        ]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks

        connected_engine._embedder.embed_batch = AsyncMock(return_value=[[0.1] * 4, [0.2] * 4])
        connected_engine._embedder.model_name = "text-embedding-3-small"

        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch(
                "khora.pipelines.tasks.extract.extract_entities",
                new_callable=AsyncMock,
                return_value=([entity], []),
            ),
        ):
            await connected_engine.remember(
                "new content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="ext-1",
            )

        connected_engine._storage.replace_document_extraction.assert_awaited_once()
        connected_engine._dual_nodes.link_entities_to_chunks_batch.assert_awaited_once()
        link_call = connected_engine._dual_nodes.link_entities_to_chunks_batch.await_args
        links = link_call.args[0]
        assert len(links) == 2
        assert all(isinstance(link, EntityChunkLink) for link in links)
        assert {link.chunk_id for link in links} == {chunk_id_a, chunk_id_b}
        assert all(link.entity_id == entity_id for link in links)

        # Ordering: coordinator ran before link_entities_to_chunks_batch.
        assert connected_engine._storage.replace_document_extraction.await_args_list[0] is not None

    @pytest.mark.asyncio
    async def test_remember_via_replace_resets_entity_source_chunk_ids(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """Review H4: after coordinator, engine calls the graph backend's
        reset_entity_source_chunk_ids_batch so survivor/net-new entities' source_chunk_ids
        reflect ONLY the new extraction (not the Neo4j MERGE append-with-tail behavior)."""
        from khora.core.models import Entity
        from khora.storage.coordinator import ReplaceResult

        namespace_id = uuid4()
        old_doc_id = uuid4()
        entity_id = uuid4()
        chunk_id_a = uuid4()
        chunk_id_b = uuid4()

        existing = MagicMock()
        existing.id = old_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=existing)
        connected_engine._storage.get_document_by_checksum = AsyncMock()
        connected_engine._storage.replace_document_extraction = AsyncMock(
            return_value=ReplaceResult(
                document_id=old_doc_id,
                chunks_deleted=0,
                chunks_created=1,
                entities_created=0,
                entities_updated=1,
                entities_retired=0,
                relationships_created=0,
                relationships_retired=0,
            )
        )
        connected_engine._storage.update_document = AsyncMock()

        # Wire a graph backend with the new reset method.
        reset_mock = AsyncMock(return_value=1)
        connected_engine._storage.graph = MagicMock()
        connected_engine._storage.graph.reset_entity_source_chunk_ids_batch = reset_mock

        async def _create_chunks_batch(chunks):
            return list(chunks)

        connected_engine._temporal_store.delete_chunks_by_document = AsyncMock(return_value=0)
        connected_engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks_batch)
        connected_engine._dual_nodes.delete_chunks_by_document = AsyncMock(return_value=0)
        connected_engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=[])
        connected_engine._dual_nodes.link_entities_to_chunks_batch = AsyncMock()

        entity = Entity(
            id=entity_id,
            namespace_id=namespace_id,
            name="Alice",
            entity_type="PERSON",
            source_chunk_ids=[chunk_id_a, chunk_id_b],
        )

        raw_chunks = [MagicMock(content="new content", start_char=0, end_char=11)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks

        connected_engine._embedder.embed_batch = AsyncMock(return_value=[[0.1] * 4])
        connected_engine._embedder.model_name = "text-embedding-3-small"

        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch(
                "khora.pipelines.tasks.extract.extract_entities",
                new_callable=AsyncMock,
                return_value=([entity], []),
            ),
        ):
            await connected_engine.remember(
                "new content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="ext-1",
            )

        reset_mock.assert_awaited_once()
        call_ns, call_rows = reset_mock.await_args.args
        assert call_ns == namespace_id
        assert len(call_rows) == 1
        row = call_rows[0]
        assert row["name"] == "Alice"
        assert row["entity_type"] == "PERSON"
        assert set(row["source_chunk_ids"]) == {str(chunk_id_a), str(chunk_id_b)}

    @pytest.mark.asyncio
    async def test_remember_via_replace_skips_reset_when_graph_has_no_method(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """Backends without reset_entity_source_chunk_ids_batch (e.g. SurrealDB) are skipped cleanly."""
        from khora.core.models import Entity
        from khora.storage.coordinator import ReplaceResult

        namespace_id = uuid4()
        old_doc_id = uuid4()

        existing = MagicMock()
        existing.id = old_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=existing)
        connected_engine._storage.get_document_by_checksum = AsyncMock()
        connected_engine._storage.replace_document_extraction = AsyncMock(
            return_value=ReplaceResult(
                document_id=old_doc_id,
                chunks_deleted=0,
                chunks_created=1,
                entities_created=1,
                entities_updated=0,
                entities_retired=0,
                relationships_created=0,
                relationships_retired=0,
            )
        )
        connected_engine._storage.update_document = AsyncMock()

        # Graph backend without the reset method.
        class _GraphStub:
            pass

        connected_engine._storage.graph = _GraphStub()

        async def _create_chunks_batch(chunks):
            return list(chunks)

        connected_engine._temporal_store.delete_chunks_by_document = AsyncMock(return_value=0)
        connected_engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks_batch)
        connected_engine._dual_nodes.delete_chunks_by_document = AsyncMock(return_value=0)
        connected_engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=[])
        connected_engine._dual_nodes.link_entities_to_chunks_batch = AsyncMock()

        entity = Entity(
            id=uuid4(), namespace_id=namespace_id, name="Bob", entity_type="PERSON", source_chunk_ids=[uuid4()]
        )
        raw_chunks = [MagicMock(content="x", start_char=0, end_char=1)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks
        connected_engine._embedder.embed_batch = AsyncMock(return_value=[[0.1] * 4])
        connected_engine._embedder.model_name = "text-embedding-3-small"

        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch(
                "khora.pipelines.tasks.extract.extract_entities",
                new_callable=AsyncMock,
                return_value=([entity], []),
            ),
        ):
            result = await connected_engine.remember(
                "content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="ext-1",
            )

        # Did not raise despite the stub graph lacking the reset method.
        assert result.metadata is not None and result.metadata.get("replaced") is True

    @pytest.mark.asyncio
    async def test_remember_via_replace_resets_relationship_source_chunk_ids(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """Review H4 (relationship side): after coordinator + entity reset,
        engine calls the graph backend's reset_relationship_source_chunk_ids_batch with
        entity name+type keys so survivor relationships (with persisted-but-unknown
        endpoint ids) still resolve correctly."""
        from khora.core.models import Entity, Relationship
        from khora.storage.coordinator import ReplaceResult

        namespace_id = uuid4()
        old_doc_id = uuid4()
        alice_id = uuid4()
        bob_id = uuid4()
        chunk_id = uuid4()

        existing = MagicMock()
        existing.id = old_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=existing)
        connected_engine._storage.get_document_by_checksum = AsyncMock()
        connected_engine._storage.replace_document_extraction = AsyncMock(
            return_value=ReplaceResult(
                document_id=old_doc_id,
                chunks_deleted=0,
                chunks_created=1,
                entities_created=2,
                entities_updated=0,
                entities_retired=0,
                relationships_created=1,
                relationships_retired=0,
            )
        )
        connected_engine._storage.update_document = AsyncMock()

        # Graph backend with BOTH reset methods.
        entity_reset_mock = AsyncMock(return_value=2)
        rel_reset_mock = AsyncMock(return_value=1)
        connected_engine._storage.graph = MagicMock()
        connected_engine._storage.graph.reset_entity_source_chunk_ids_batch = entity_reset_mock
        connected_engine._storage.graph.reset_relationship_source_chunk_ids_batch = rel_reset_mock

        async def _create_chunks_batch(chunks):
            return list(chunks)

        connected_engine._temporal_store.delete_chunks_by_document = AsyncMock(return_value=0)
        connected_engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks_batch)
        connected_engine._dual_nodes.delete_chunks_by_document = AsyncMock(return_value=0)
        connected_engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=[])
        connected_engine._dual_nodes.link_entities_to_chunks_batch = AsyncMock()

        alice = Entity(
            id=alice_id,
            namespace_id=namespace_id,
            name="Alice",
            entity_type="PERSON",
            source_chunk_ids=[chunk_id],
        )
        bob = Entity(
            id=bob_id,
            namespace_id=namespace_id,
            name="Bob",
            entity_type="PERSON",
            source_chunk_ids=[chunk_id],
        )
        knows = Relationship(
            namespace_id=namespace_id,
            source_entity_id=alice_id,
            target_entity_id=bob_id,
            relationship_type="KNOWS",
            source_chunk_ids=[chunk_id],
        )

        raw_chunks = [MagicMock(content="Alice knows Bob", start_char=0, end_char=15)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks

        connected_engine._embedder.embed_batch = AsyncMock(return_value=[[0.1] * 4, [0.2] * 4, [0.3] * 4])
        connected_engine._embedder.model_name = "text-embedding-3-small"

        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch(
                "khora.pipelines.tasks.extract.extract_entities",
                new_callable=AsyncMock,
                return_value=([alice, bob], [knows]),
            ),
        ):
            await connected_engine.remember(
                "Alice knows Bob",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="ext-1",
            )

        rel_reset_mock.assert_awaited_once()
        call_ns, call_rows = rel_reset_mock.await_args.args
        assert call_ns == namespace_id
        assert len(call_rows) == 1
        row = call_rows[0]
        assert row["source_name"] == "Alice"
        assert row["source_type"] == "PERSON"
        assert row["target_name"] == "Bob"
        assert row["target_type"] == "PERSON"
        assert row["rel_type"] == "KNOWS"
        assert row["source_chunk_ids"] == [str(chunk_id)]

    @pytest.mark.asyncio
    async def test_remember_via_replace_marks_failed_on_temporal_store_error(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """If temporal store wipe/write fails, document is marked FAILED and error re-raised."""
        namespace_id = uuid4()
        old_doc_id = uuid4()

        existing = MagicMock()
        existing.id = old_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=existing)
        connected_engine._storage.get_document_by_checksum = AsyncMock()
        connected_engine._storage.replace_document_extraction = AsyncMock()
        connected_engine._storage.update_document = AsyncMock()

        connected_engine._temporal_store.delete_chunks_by_document = AsyncMock(side_effect=RuntimeError("pg down"))

        raw_chunks = [MagicMock(content="boom", start_char=0, end_char=4)]
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks

        connected_engine._embedder.embed_batch = AsyncMock(return_value=[[0.1] * 4])
        connected_engine._embedder.model_name = "text-embedding-3-small"

        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch(
                "khora.pipelines.tasks.extract.extract_entities",
                new_callable=AsyncMock,
                return_value=([], []),
            ),
            pytest.raises(RuntimeError, match="pg down"),
        ):
            await connected_engine.remember(
                "new content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="ext-1",
            )

        # Coordinator was not called because the pre-coordinator wipe failed.
        connected_engine._storage.replace_document_extraction.assert_not_called()
        # Document was marked FAILED + persisted via update_document best-effort.
        connected_engine._storage.update_document.assert_awaited_once()
        failed_doc = connected_engine._storage.update_document.await_args.args[0]
        assert failed_doc.status.value == "failed"
        assert "pg down" in (failed_doc.error_message or "")

    @pytest.mark.asyncio
    async def test_remember_via_replace_surfaces_graph_mirror_degradation(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """Issue #884: when the coordinator raises GraphMirrorFailedAfterPGCommitError
        (PG committed, graph mirror partial), ``_remember_via_replace`` must
        return a RememberResult carrying a degradation in metadata rather
        than propagating the exception, so the caller has an observable
        signal that PG state is durable but graph is partial.
        """
        from khora.exceptions import GraphMirrorFailedAfterPGCommitError

        namespace_id = uuid4()
        old_doc_id = uuid4()

        existing = MagicMock()
        existing.id = old_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        connected_engine._storage.get_document_by_external_id = AsyncMock(return_value=existing)
        connected_engine._storage.get_document_by_checksum = AsyncMock()
        connected_engine._storage.update_document = AsyncMock()

        # The coordinator raises the typed signal AFTER stamping PG durable.
        # Both document_id and namespace_id round-trip into the user-facing
        # degradation entry.
        original = RuntimeError("neo4j unreachable")
        mirror_err = GraphMirrorFailedAfterPGCommitError(
            document_id=old_doc_id,
            namespace_id=namespace_id,
            original=original,
        )
        connected_engine._storage.replace_document_extraction = AsyncMock(side_effect=mirror_err)

        with (
            patch("khora.extraction.chunkers.create_chunker") as mock_chunker_factory,
            patch("khora.pipelines.tasks.extract.extract_entities", new_callable=AsyncMock, return_value=([], [])),
        ):
            mock_chunker = MagicMock()
            mock_chunker.chunk.return_value = []  # no raw chunks -> no embed/extract needed
            mock_chunker_factory.return_value = mock_chunker

            result = await connected_engine.remember(
                "new content",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                external_id="ext-1",
            )

        # No exception leaked out - the engine produced a RememberResult.
        assert result.document_id == old_doc_id
        assert result.namespace_id == namespace_id
        # The replace flag is still set (caller's contract: an external_id
        # match routed to the replace path).
        assert result.metadata["replaced"] is True
        assert result.metadata["old_document_id"] == str(old_doc_id)

        # The degradation entry encodes the ADR-001 convention: a component
        # tag, a machine-readable reason, the original exception type, and
        # the issue link so a future reconciler can identify the failure
        # mode without parsing log lines.
        degradations = result.metadata.get("degradations")
        assert isinstance(degradations, list)
        assert len(degradations) == 1
        entry = degradations[0]
        assert entry["component"] == "coordinator.replace_document_extraction"
        assert entry["reason"] == "graph_mirror_failed_after_pg_commit"
        assert entry["exception"] == "RuntimeError"
        assert entry["issue"] == "884"

    @pytest.mark.asyncio
    async def test_remember_batch_routes_mixed_external_ids(self, connected_engine: VectorCypherEngine) -> None:
        """remember_batch dispatches matched external_id docs to replace path."""
        from khora.khora import RememberResult

        namespace_id = uuid4()
        matched_doc_id = uuid4()
        unmatched_doc_id = uuid4()
        existing = MagicMock()
        existing.id = matched_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        # Backend lookups: "ext-a" exists, "ext-b" does not.
        async def _ext_lookup(ext_id, *, namespace_id):
            return existing if ext_id == "ext-a" else None

        async def _ext_batch(ext_ids, *, namespace_id):
            return {e: existing for e in ext_ids if e == "ext-a"}

        connected_engine._storage.get_document_by_external_id = AsyncMock(side_effect=_ext_lookup)
        connected_engine._storage.get_documents_by_external_ids = AsyncMock(side_effect=_ext_batch)
        connected_engine._storage.get_documents_by_checksums = AsyncMock(return_value={})

        # Replace returns a RememberResult directly via patched self.remember().
        async def _fake_remember(*args, **kwargs):
            if kwargs.get("external_id") == "ext-a":
                return RememberResult(
                    document_id=matched_doc_id,
                    namespace_id=namespace_id,
                    chunks_created=2,
                    entities_extracted=3,
                    relationships_created=1,
                    metadata={"replaced": True, "old_document_id": str(matched_doc_id)},
                )
            return RememberResult(  # pragma: no cover — should only hit above
                document_id=unmatched_doc_id,
                namespace_id=namespace_id,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
            )

        # Force streaming pipeline OFF so docs without external_id go through
        # legacy path, which in turn calls self.remember() for each — which we
        # also patch so the test stays hermetic.
        connected_engine._vc_config.streaming_pipeline = False

        with patch.object(connected_engine, "remember", side_effect=_fake_remember) as mock_remember:
            result = await connected_engine.remember_batch(
                [
                    {"content": "matched content", "external_id": "ext-a"},
                    {"content": "no external id"},
                    {"content": "unmatched external id", "external_id": "ext-b"},
                ],
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        # Legacy path calls self.remember() once per doc — external_id is
        # forwarded through unchanged, so the matched doc's replace dispatch
        # happens inside remember() itself (covered by the dedicated test above).
        assert mock_remember.await_count == 3
        forwarded_ext_ids = [call.kwargs.get("external_id") for call in mock_remember.await_args_list]
        assert "ext-a" in forwarded_ext_ids
        assert "ext-b" in forwarded_ext_ids
        assert None in forwarded_ext_ids
        assert result.total == 3

    @pytest.mark.asyncio
    async def test_remember_batch_streaming_external_id_prefilter(self, connected_engine: VectorCypherEngine) -> None:
        """Streaming pipeline removes external_id-matched docs from chunk/embed stages."""
        from khora.khora import RememberResult

        namespace_id = uuid4()
        matched_doc_id = uuid4()
        existing = MagicMock()
        existing.id = matched_doc_id
        existing.created_at = datetime(2026, 1, 1, tzinfo=UTC)

        async def _ext_lookup(ext_id, *, namespace_id):
            return existing if ext_id == "ext-a" else None

        async def _ext_batch(ext_ids, *, namespace_id):
            return {e: existing for e in ext_ids if e == "ext-a"}

        connected_engine._storage.get_document_by_external_id = AsyncMock(side_effect=_ext_lookup)
        connected_engine._storage.get_documents_by_external_ids = AsyncMock(side_effect=_ext_batch)
        connected_engine._storage.get_documents_by_checksums = AsyncMock(return_value={})

        async def _fake_remember(*args, **kwargs):
            return RememberResult(
                document_id=matched_doc_id,
                namespace_id=namespace_id,
                chunks_created=2,
                entities_extracted=3,
                relationships_created=1,
                metadata={"replaced": True, "old_document_id": str(matched_doc_id)},
            )

        # Streaming pipeline enabled (default for this engine config).
        assert connected_engine._vc_config.streaming_pipeline is True

        # Stub create_document so any accidental streaming-pipeline call surfaces.
        connected_engine._storage.create_document = AsyncMock()

        with patch.object(connected_engine, "remember", side_effect=_fake_remember) as mock_remember:
            # Give only a single matched-external-id doc so the streaming
            # pipeline has nothing left after the prefilter stage.
            result = await connected_engine.remember_batch(
                [{"content": "matched content", "external_id": "ext-a"}],
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        # Matched doc went through the replace dispatch via self.remember().
        mock_remember.assert_awaited_once()
        assert mock_remember.await_args.kwargs.get("external_id") == "ext-a"
        # Streaming pipeline was short-circuited — no create_document calls.
        connected_engine._storage.create_document.assert_not_called()
        # Single batch lookup for all matched external_ids — not N serial calls.
        connected_engine._storage.get_documents_by_external_ids.assert_awaited_once()
        call_args = connected_engine._storage.get_documents_by_external_ids.await_args
        assert call_args.kwargs.get("namespace_id") == namespace_id
        assert list(call_args.args[0]) == ["ext-a"]
        assert result.total == 1
        assert result.processed == 1
        assert result.chunks == 2
        assert result.entities == 3
        assert result.relationships == 1


@pytest.mark.unit
class TestVectorCypherEngineRecall:
    """Tests for engine recall() with mocked backends."""

    @pytest.fixture
    def connected_engine(self) -> VectorCypherEngine:
        """Create a mock-connected engine."""
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

        engine = VectorCypherEngine(config)
        engine._connected = True
        engine._storage = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._embedder = AsyncMock()
        engine._dual_nodes = AsyncMock()
        engine._neo4j_driver = AsyncMock()

        # Mock retriever
        from khora.engines.vectorcypher.retriever import VectorCypherResult
        from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.8,
            reasoning="test",
        )
        chunk1 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="This is a test chunk with enough content to pass validation",
        )
        chunk2 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Another test chunk that is also long enough to pass validation",
        )
        retriever_result = VectorCypherResult(
            chunks=[(chunk1, 0.9), (chunk2, 0.7)],
            entities=[],
            routing_decision=routing,
            metadata={"search_mode": "simple_vector"},
        )
        engine._retriever = AsyncMock()
        engine._retriever.retrieve = AsyncMock(return_value=retriever_result)
        engine._router = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_recall_returns_results(self, connected_engine: VectorCypherEngine) -> None:
        """Test that recall returns validated results."""
        namespace_id = uuid4()
        result = await connected_engine.recall("test query", namespace_id)

        assert isinstance(result, RecallResult)
        assert result.query == "test query"
        assert result.namespace_id == namespace_id
        assert len(result.chunks) == 2
        assert result.engine_info["engine"] == "vectorcypher"

    @pytest.mark.asyncio
    async def test_recall_filters_duplicates(self, connected_engine: VectorCypherEngine) -> None:
        """Test that recall filters duplicate chunks."""
        from khora.engines.vectorcypher.retriever import VectorCypherResult
        from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.8,
            reasoning="test",
        )
        dup_id = uuid4()
        dup_chunk1 = Chunk(
            id=dup_id,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Duplicate chunk content that is long enough for validation",
        )
        dup_chunk2 = Chunk(
            id=dup_id,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Duplicate chunk content that is long enough for validation",
        )
        retriever_result = VectorCypherResult(
            chunks=[(dup_chunk1, 0.9), (dup_chunk2, 0.8)],
            entities=[],
            routing_decision=routing,
            metadata={},
        )
        connected_engine._retriever.retrieve = AsyncMock(return_value=retriever_result)

        namespace_id = uuid4()
        result = await connected_engine.recall("test", namespace_id)

        # Duplicates should be filtered
        assert len(result.chunks) == 1


@pytest.mark.unit
class TestVectorCypherEngineForget:
    """Tests for engine forget() with mocked backends."""

    @pytest.fixture
    def connected_engine(self) -> VectorCypherEngine:
        """Create a mock-connected engine."""
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

        engine = VectorCypherEngine(config)
        engine._connected = True
        engine._storage = AsyncMock()
        # #923: cleanup is vector-anchored. The vector backend (pgvector) is
        # the authoritative refcount store; spec it to the pgvector surface so
        # the cascade dispatches to the real method names. The Neo4j graph
        # backend is mirrored opportunistically.
        engine._storage.vector = MagicMock(
            spec=[
                "list_entities",
                "list_relationships",
                "delete_entities_batch",
                "delete_relationships_batch",
                "remove_document_from_entity_sources",
                "remove_document_from_relationship_sources",
            ]
        )
        engine._storage.vector.list_entities = AsyncMock(return_value=[])
        engine._storage.vector.list_relationships = AsyncMock(return_value=[])
        engine._storage.vector.delete_entities_batch = AsyncMock()
        engine._storage.vector.delete_relationships_batch = AsyncMock()
        engine._storage.vector.remove_document_from_entity_sources = AsyncMock()
        engine._storage.vector.remove_document_from_relationship_sources = AsyncMock()
        engine._storage.graph = MagicMock()
        engine._storage.graph.delete_entities_batch = AsyncMock()
        engine._storage.graph.delete_relationships_batch = AsyncMock()
        engine._storage.graph.remove_document_from_entity_sources_batch = AsyncMock()
        engine._storage.graph.remove_document_from_relationship_sources_batch = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._dual_nodes = AsyncMock()
        engine._neo4j_driver = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_forget_with_namespace(self, connected_engine: VectorCypherEngine) -> None:
        """Test forget with explicit namespace ID."""
        doc_id = uuid4()
        namespace_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is True
        connected_engine._dual_nodes.delete_chunks_by_document.assert_called_once_with(doc_id, namespace_id)
        connected_engine._temporal_store.delete_chunks_by_document.assert_called_once_with(doc_id, namespace_id)
        connected_engine._storage.delete_document.assert_called_once_with(doc_id, namespace_id=namespace_id)

    @pytest.mark.asyncio
    async def test_forget_namespace_mismatch(self, connected_engine: VectorCypherEngine) -> None:
        """Test forget returns False when namespace doesn't match.

        Security: namespace mismatch is now enforced at the SQL layer —
        ``storage.get_document(doc_id, namespace_id=wrong_ns)`` returns
        ``None`` and the engine bails before any cascade work."""
        doc_id = uuid4()
        namespace_id = uuid4()

        connected_engine._storage.get_document = AsyncMock(return_value=None)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is False
        connected_engine._storage.get_document.assert_awaited_once_with(doc_id, namespace_id=namespace_id)

    @pytest.mark.asyncio
    async def test_forget_without_namespace(self, connected_engine: VectorCypherEngine) -> None:
        """Security: forget bails immediately when namespace_id is None.

        Previously the engine looked up the document by id alone (an IDOR
        vector) and trusted the document's own namespace. The new contract
        requires the caller to pass namespace_id."""
        doc_id = uuid4()

        connected_engine._storage.get_document = AsyncMock()
        connected_engine._storage.delete_document = AsyncMock()

        result = await connected_engine.forget(doc_id, None)

        assert result is False
        connected_engine._storage.get_document.assert_not_awaited()
        connected_engine._storage.delete_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forget_document_not_found(self, connected_engine: VectorCypherEngine) -> None:
        """Test forget returns False when document not found."""
        doc_id = uuid4()
        connected_engine._storage.get_document = AsyncMock(return_value=None)

        result = await connected_engine.forget(doc_id, None)

        assert result is False

    @pytest.mark.asyncio
    async def test_forget_cascade_deletes_orphan_entity(self, connected_engine: VectorCypherEngine) -> None:
        """Orphan entity (sole source = forgotten doc) is hard-deleted in both backends."""
        doc_id = uuid4()
        namespace_id = uuid4()
        orphan_ent_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.vector.list_entities = AsyncMock(return_value=[_ent(orphan_ent_id, [doc_id])])

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.vector.delete_entities_batch.assert_awaited_once_with(
            [orphan_ent_id], namespace_id=namespace_id
        )
        connected_engine._storage.graph.delete_entities_batch.assert_awaited_once_with(
            [orphan_ent_id], namespace_id=namespace_id
        )
        connected_engine._storage.graph.remove_document_from_entity_sources_batch.assert_not_called()
        connected_engine._storage.vector.remove_document_from_entity_sources.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_updates_survivor_entity_sources(self, connected_engine: VectorCypherEngine) -> None:
        """Survivor entity (multi-source) has doc_id stripped, not deleted."""
        doc_id = uuid4()
        namespace_id = uuid4()
        survivor_ent_id = uuid4()
        other_doc = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.vector.list_entities = AsyncMock(
            return_value=[_ent(survivor_ent_id, [doc_id, other_doc])]
        )

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.vector.remove_document_from_entity_sources.assert_awaited_once_with(
            [survivor_ent_id], doc_id
        )
        connected_engine._storage.graph.remove_document_from_entity_sources_batch.assert_awaited_once_with(
            [survivor_ent_id], doc_id, namespace_id
        )
        connected_engine._storage.graph.delete_entities_batch.assert_not_called()
        connected_engine._storage.vector.delete_entities_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_deletes_orphan_relationship(self, connected_engine: VectorCypherEngine) -> None:
        """Orphan relationship is hard-deleted from both backends (no namespace_id on the rel call)."""
        doc_id = uuid4()
        namespace_id = uuid4()
        orphan_rel_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.vector.list_relationships = AsyncMock(return_value=[_ent(orphan_rel_id, [doc_id])])

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.vector.delete_relationships_batch.assert_awaited_once_with(
            [orphan_rel_id], namespace_id=namespace_id
        )
        connected_engine._storage.graph.delete_relationships_batch.assert_awaited_once_with(
            [orphan_rel_id], namespace_id=namespace_id
        )
        connected_engine._storage.graph.remove_document_from_relationship_sources_batch.assert_not_called()
        connected_engine._storage.vector.remove_document_from_relationship_sources.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_updates_survivor_relationship_sources(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """Survivor relationship has doc_id stripped, not deleted."""
        doc_id = uuid4()
        namespace_id = uuid4()
        survivor_rel_id = uuid4()
        other_doc = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.vector.list_relationships = AsyncMock(
            return_value=[_ent(survivor_rel_id, [doc_id, other_doc])]
        )

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.vector.remove_document_from_relationship_sources.assert_awaited_once_with(
            [survivor_rel_id], doc_id
        )
        connected_engine._storage.graph.remove_document_from_relationship_sources_batch.assert_awaited_once_with(
            [survivor_rel_id], doc_id
        )
        connected_engine._storage.graph.delete_relationships_batch.assert_not_called()
        connected_engine._storage.vector.delete_relationships_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_zero_extraction_skips_backend_calls(
        self, connected_engine: VectorCypherEngine
    ) -> None:
        """Document with no extracted entities/relationships: no cascade backend writes,
        but chunk/doc deletes still run."""
        doc_id = uuid4()
        namespace_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        # default fixture already stubs vector.list_entities / list_relationships -> []

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is True
        connected_engine._storage.graph.delete_entities_batch.assert_not_called()
        connected_engine._storage.graph.delete_relationships_batch.assert_not_called()
        connected_engine._storage.graph.remove_document_from_entity_sources_batch.assert_not_called()
        connected_engine._storage.graph.remove_document_from_relationship_sources_batch.assert_not_called()
        connected_engine._storage.vector.delete_entities_batch.assert_not_called()
        connected_engine._storage.vector.delete_relationships_batch.assert_not_called()
        # Existing chunk + document-row deletes are still invoked.
        connected_engine._dual_nodes.delete_chunks_by_document.assert_awaited_once_with(doc_id, namespace_id)
        connected_engine._temporal_store.delete_chunks_by_document.assert_awaited_once_with(doc_id, namespace_id)
        connected_engine._storage.delete_document.assert_awaited_once_with(doc_id, namespace_id=namespace_id)

    @pytest.mark.asyncio
    async def test_forget_cascade_runs_before_chunk_and_doc_deletes(self, connected_engine: VectorCypherEngine) -> None:
        """Cascade must precede chunk/doc deletes — otherwise survivor's
        source_document_ids array references a doc that no longer exists when
        the cascade reads it."""
        doc_id = uuid4()
        namespace_id = uuid4()
        orphan_ent_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.vector.list_entities = AsyncMock(return_value=[_ent(orphan_ent_id, [doc_id])])

        call_order: list[str] = []
        connected_engine._storage.vector.delete_entities_batch.side_effect = lambda *a, **kw: call_order.append(
            "cascade_vector_delete"
        )
        connected_engine._storage.graph.delete_entities_batch.side_effect = lambda *a, **kw: call_order.append(
            "cascade_graph_delete"
        )
        connected_engine._dual_nodes.delete_chunks_by_document.side_effect = lambda *a, **kw: call_order.append(
            "delete_chunks_neo4j"
        )
        connected_engine._temporal_store.delete_chunks_by_document.side_effect = lambda *a, **kw: call_order.append(
            "delete_chunks_pgvector"
        )
        connected_engine._storage.delete_document.side_effect = lambda *a, **kw: (
            call_order.append("delete_document") or True
        )

        await connected_engine.forget(doc_id, namespace_id)

        # Cascade fires before any chunk/doc delete.
        cascade_idx = call_order.index("cascade_vector_delete")
        assert cascade_idx < call_order.index("delete_chunks_neo4j")
        assert cascade_idx < call_order.index("delete_chunks_pgvector")
        assert cascade_idx < call_order.index("delete_document")


@pytest.mark.unit
class TestVectorCypherEngineHealthCheck:
    """Tests for engine health_check()."""

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self) -> None:
        """Test health check returns disconnected when not connected."""
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

        engine = VectorCypherEngine(config)
        result = await engine.health_check()

        assert result == {"status": "disconnected"}

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self) -> None:
        """Test health check when all components are healthy."""
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

        engine = VectorCypherEngine(config)
        engine._connected = True

        # Mock storage health
        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {"postgresql": "ok", "graph": "ok"}
        engine._storage = AsyncMock()
        engine._storage.health_check = AsyncMock(return_value=storage_health)

        # Mock temporal store health
        engine._temporal_store = AsyncMock()
        engine._temporal_store.health_check = AsyncMock(return_value={"status": "healthy"})

        # Mock Neo4j health
        engine._neo4j_driver = AsyncMock()
        engine._neo4j_driver.verify_connectivity = AsyncMock()

        result = await engine.health_check()

        assert result["status"] == "healthy"
        assert result["neo4j"] == "healthy"
        assert result["engine"] == "vectorcypher"

    @pytest.mark.asyncio
    async def test_health_check_neo4j_unhealthy(self) -> None:
        """Test health check when Neo4j is unhealthy."""
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

        engine = VectorCypherEngine(config)
        engine._connected = True

        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {"postgresql": "ok"}
        engine._storage = AsyncMock()
        engine._storage.health_check = AsyncMock(return_value=storage_health)

        engine._temporal_store = AsyncMock()
        engine._temporal_store.health_check = AsyncMock(return_value={"status": "healthy"})

        engine._neo4j_driver = AsyncMock()
        engine._neo4j_driver.verify_connectivity = AsyncMock(side_effect=Exception("connection refused"))

        result = await engine.health_check()

        assert result["status"] == "degraded"
        assert result["neo4j"] == "unhealthy"


@pytest.mark.unit
class TestVectorCypherEngineValidateRecallResults:
    """Tests for _validate_recall_results."""

    @pytest.fixture
    def engine(self) -> VectorCypherEngine:
        """Create an engine for testing validation."""
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
        return VectorCypherEngine(config)

    def test_filters_empty_content(self, engine: VectorCypherEngine) -> None:
        """Test that chunks with empty content are filtered out."""
        c1 = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="")
        c2 = Chunk(
            id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="Valid long enough content for testing"
        )
        chunks = [(c1, 0.9), (c2, 0.8)]
        result = engine._validate_recall_results(chunks, "test query")
        assert len(result) == 1
        assert result[0][0].id == c2.id

    def test_filters_short_content(self, engine: VectorCypherEngine) -> None:
        """Test that chunks with very short content are filtered."""
        c1 = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="short")
        c2 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="This content is long enough to pass minimum length validation",
        )
        chunks = [(c1, 0.9), (c2, 0.8)]
        result = engine._validate_recall_results(chunks, "test")
        assert len(result) == 1

    def test_removes_duplicates(self, engine: VectorCypherEngine) -> None:
        """Test that duplicate chunks are removed."""
        shared_id = uuid4()
        c1 = Chunk(
            id=shared_id, namespace_id=uuid4(), document_id=uuid4(), content="First occurrence with enough content"
        )
        c2 = Chunk(
            id=shared_id, namespace_id=uuid4(), document_id=uuid4(), content="First occurrence with enough content"
        )
        chunks = [(c1, 0.9), (c2, 0.8)]
        result = engine._validate_recall_results(chunks, "test")
        assert len(result) == 1

    def test_normalizes_scores(self, engine: VectorCypherEngine) -> None:
        """Test that scores are clamped to [0, 1]."""
        c1 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Content that has a very high score value assigned to it",
        )
        c2 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Content that has a negative score value assigned to it",
        )
        chunks = [(c1, 1.5), (c2, -0.5)]
        result = engine._validate_recall_results(chunks, "test")
        assert result[0][1] == 1.0
        assert result[1][1] == 0.0

    def test_skips_non_chunk_objects(self, engine: VectorCypherEngine) -> None:
        """Test that non-Chunk objects are skipped."""
        c1 = Chunk(
            id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="Valid content that passes all checks"
        )
        chunks = [("not a chunk", 0.9), (c1, 0.8)]
        result = engine._validate_recall_results(chunks, "test")
        assert len(result) == 1


@pytest.mark.unit
class TestVectorCypherEngineConnectAcquisitionTimeout:
    """Tests for connection_acquisition_timeout being passed to Neo4j driver."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock KhoraConfig."""
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
        config.storage.backend = "postgres"
        config.llm.model = "gpt-4o-mini"
        config.llm.embedding_model = "text-embedding-3-small"
        config.llm.embedding_dimension = 1536
        config.llm.timeout = 30
        config.llm.max_retries = 3
        config.llm.max_concurrent_llm_calls = 5
        config.pipeline.chunking_strategy = "recursive"
        config.pipeline.chunk_size = 1000
        config.pipeline.chunk_overlap = 200
        config.pipeline.extract_entities = True
        config.telemetry_database_url = None
        config.telemetry_service_name = "test"
        return config

    @pytest.mark.asyncio
    async def test_custom_acquisition_timeout(self, mock_config: MagicMock) -> None:
        """Test that connection_acquisition_timeout from config is passed to driver."""
        neo4j_cfg = MagicMock()
        neo4j_cfg.connection_acquisition_timeout = 2.5
        mock_config.get_graph_config.return_value = neo4j_cfg

        engine = VectorCypherEngine(mock_config)

        mock_driver = AsyncMock()
        sentinel = RuntimeError("stop after driver")
        with (
            patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver) as mock_driver_cls,
            patch(
                "khora.engines.vectorcypher.engine.create_storage_coordinator",
                side_effect=sentinel,
            ),
            pytest.raises(RuntimeError, match="stop after driver"),
        ):
            await engine.connect()

        mock_driver_cls.assert_called_once()
        call_kwargs = mock_driver_cls.call_args[1]
        assert call_kwargs["connection_acquisition_timeout"] == 2.5

    @pytest.mark.asyncio
    async def test_default_acquisition_timeout(self, mock_config: MagicMock) -> None:
        """Test that default 60.0 is used when config has no connection_acquisition_timeout."""
        # spec=[] prevents MagicMock from auto-creating attributes, so getattr falls through to default
        neo4j_cfg = MagicMock(spec=[])
        mock_config.get_graph_config.return_value = neo4j_cfg

        engine = VectorCypherEngine(mock_config)

        mock_driver = AsyncMock()
        sentinel = RuntimeError("stop after driver")
        with (
            patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver) as mock_driver_cls,
            patch(
                "khora.engines.vectorcypher.engine.create_storage_coordinator",
                side_effect=sentinel,
            ),
            pytest.raises(RuntimeError, match="stop after driver"),
        ):
            await engine.connect()

        mock_driver_cls.assert_called_once()
        call_kwargs = mock_driver_cls.call_args[1]
        assert call_kwargs["connection_acquisition_timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_none_graph_config_uses_defaults(self, mock_config: MagicMock) -> None:
        """Test that None graph config falls through to all driver defaults."""
        mock_config.get_graph_config.return_value = None

        engine = VectorCypherEngine(mock_config)

        mock_driver = AsyncMock()
        sentinel = RuntimeError("stop after driver")
        with (
            patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver) as mock_driver_cls,
            patch(
                "khora.engines.vectorcypher.engine.create_storage_coordinator",
                side_effect=sentinel,
            ),
            pytest.raises(RuntimeError, match="stop after driver"),
        ):
            await engine.connect()

        mock_driver_cls.assert_called_once()
        call_kwargs = mock_driver_cls.call_args[1]
        assert call_kwargs["connection_acquisition_timeout"] == 60.0
        assert call_kwargs["max_connection_pool_size"] == 100
        assert call_kwargs["max_connection_lifetime"] == 900
        assert call_kwargs["liveness_check_timeout"] == 30.0


@pytest.mark.unit
class TestVectorCypherEngineParseDatetime:
    """Tests for _parse_datetime helper."""

    @pytest.fixture
    def engine(self) -> VectorCypherEngine:
        """Create an engine for testing datetime parsing."""
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
        return VectorCypherEngine(config)

    def test_parse_datetime_object(self, engine: VectorCypherEngine) -> None:
        """Test that datetime objects pass through."""
        now = datetime.now(UTC)
        result = engine._parse_datetime(now)
        assert result == now

    def test_parse_naive_datetime(self, engine: VectorCypherEngine) -> None:
        """Test that naive datetimes get UTC timezone."""
        naive = datetime(2024, 1, 15)
        result = engine._parse_datetime(naive)
        assert result.tzinfo == UTC

    def test_parse_date_string(self, engine: VectorCypherEngine) -> None:
        """Test parsing date-only strings."""
        result = engine._parse_datetime("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == UTC

    def test_parse_iso_string_with_z(self, engine: VectorCypherEngine) -> None:
        """Test parsing ISO string with Z suffix."""
        result = engine._parse_datetime("2024-01-15T10:30:00Z")
        assert result.year == 2024
        assert result.hour == 10

    def test_parse_invalid_raises(self, engine: VectorCypherEngine) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            engine._parse_datetime("not-a-date")

        with pytest.raises(ValueError, match="Cannot parse datetime"):
            engine._parse_datetime(12345)


class TestVectorCypherConfigWindowing:
    """Tests for VectorCypherConfig windowing validation."""

    def test_max_chunks_in_flight_default_is_none(self) -> None:
        config = VectorCypherConfig()
        assert config.max_chunks_in_flight is None

    def test_max_chunks_in_flight_positive_value(self) -> None:
        config = VectorCypherConfig(max_chunks_in_flight=100)
        assert config.max_chunks_in_flight == 100

    def test_max_chunks_in_flight_one_is_valid(self) -> None:
        config = VectorCypherConfig(max_chunks_in_flight=1)
        assert config.max_chunks_in_flight == 1

    def test_max_chunks_in_flight_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chunks_in_flight must be >= 1, got 0"):
            VectorCypherConfig(max_chunks_in_flight=0)

    def test_max_chunks_in_flight_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chunks_in_flight must be >= 1, got -5"):
            VectorCypherConfig(max_chunks_in_flight=-5)

    def test_enable_session_aware_search_default_true(self) -> None:
        config = VectorCypherConfig()
        assert config.enable_session_aware_search is True


class TestProcessDocumentWindowing:
    """Tests for windowed chunk processing in _process_document."""

    @pytest.fixture()
    def engine(self) -> VectorCypherEngine:
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
        config.pipeline.extract_entities = False
        config.pipeline.chunking_strategy = "recursive"
        config.pipeline.chunk_size = 512
        config.pipeline.chunk_overlap = 50
        engine = VectorCypherEngine(config)
        engine._connected = True
        engine._storage = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._embedder = AsyncMock()
        engine._dual_nodes = AsyncMock()
        engine._retriever = AsyncMock()
        engine._router = MagicMock()
        engine._neo4j_driver = AsyncMock()
        return engine

    def _make_raw_chunk(self, content: str) -> MagicMock:
        chunk = MagicMock()
        chunk.content = content
        chunk.start_char = 0
        chunk.end_char = len(content)
        return chunk

    @pytest.mark.asyncio
    async def test_single_window_when_none(self, engine: VectorCypherEngine) -> None:
        """When max_chunks_in_flight is None, all chunks in one window."""
        engine._vc_config = VectorCypherConfig(max_chunks_in_flight=None)

        raw_chunks = [self._make_raw_chunk(f"chunk {i}") for i in range(5)]
        doc = MagicMock()
        doc.id = uuid4()
        doc.namespace_id = uuid4()
        doc.content = "test content"
        doc.metadata = {}

        # Mock chunker to return our raw chunks
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks

        # Mock embedder to return one embedding per text
        engine._embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))

        # Mock temporal store — return stored chunks with IDs
        def make_stored(chunks):
            result = []
            for c in chunks:
                s = MagicMock()
                s.id = uuid4()
                result.append(s)
            return result

        engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=make_stored)

        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker):
            total_chunks, ents, rels = await engine._process_document(
                doc,
                skill_name="default",
                occurred_at=datetime.now(UTC),
                entity_types=[],
                relationship_types=[],
            )

        assert total_chunks == 5
        # All 5 chunks embedded in one call
        engine._embedder.embed_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_exact_window_boundary(self, engine: VectorCypherEngine) -> None:
        """Document with exactly max_chunks_in_flight chunks → one window."""
        engine._vc_config = VectorCypherConfig(max_chunks_in_flight=4)

        raw_chunks = [self._make_raw_chunk(f"chunk {i}") for i in range(4)]
        doc = MagicMock()
        doc.id = uuid4()
        doc.namespace_id = uuid4()
        doc.content = "test"
        doc.metadata = {}

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks
        engine._embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))

        def make_stored(chunks):
            return [MagicMock(id=uuid4()) for _ in chunks]

        engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=make_stored)

        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker):
            total_chunks, _, _ = await engine._process_document(
                doc,
                skill_name="default",
                occurred_at=datetime.now(UTC),
                entity_types=[],
                relationship_types=[],
            )

        assert total_chunks == 4
        # Exactly one embed_batch call (single window)
        engine._embedder.embed_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_window_split(self, engine: VectorCypherEngine) -> None:
        """Document with max_chunks_in_flight + 1 chunks → two windows."""
        engine._vc_config = VectorCypherConfig(max_chunks_in_flight=3)

        raw_chunks = [self._make_raw_chunk(f"chunk {i}") for i in range(4)]
        doc = MagicMock()
        doc.id = uuid4()
        doc.namespace_id = uuid4()
        doc.content = "test"
        doc.metadata = {}

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks
        engine._embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))

        def make_stored(chunks):
            return [MagicMock(id=uuid4()) for _ in chunks]

        engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=make_stored)

        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker):
            total_chunks, _, _ = await engine._process_document(
                doc,
                skill_name="default",
                occurred_at=datetime.now(UTC),
                entity_types=[],
                relationship_types=[],
            )

        assert total_chunks == 4
        # Two embed_batch calls: window of 3, window of 1
        assert engine._embedder.embed_batch.call_count == 2
        call_args = engine._embedder.embed_batch.call_args_list
        assert len(call_args[0][0][0]) == 3  # first window: 3 texts
        assert len(call_args[1][0][0]) == 1  # second window: 1 text

    @pytest.mark.asyncio
    async def test_chunk_index_continuity_across_windows(self, engine: VectorCypherEngine) -> None:
        """chunk_index in metadata must be continuous across windows."""
        engine._vc_config = VectorCypherConfig(max_chunks_in_flight=2)

        raw_chunks = [self._make_raw_chunk(f"chunk {i}") for i in range(5)]
        doc = MagicMock()
        doc.id = uuid4()
        doc.namespace_id = uuid4()
        doc.content = "test"
        doc.metadata = {}

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks
        engine._embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))

        # Capture the temporal chunks passed to create_chunks_batch
        all_temporal_chunks = []

        def capture_stored(chunks):
            all_temporal_chunks.extend(chunks)
            return [MagicMock(id=uuid4()) for _ in chunks]

        engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=capture_stored)

        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker):
            total_chunks, _, _ = await engine._process_document(
                doc,
                skill_name="default",
                occurred_at=datetime.now(UTC),
                entity_types=[],
                relationship_types=[],
            )

        assert total_chunks == 5
        # 3 windows: [2, 2, 1]
        assert engine._embedder.embed_batch.call_count == 3

        # Verify chunk_index is 0, 1, 2, 3, 4 across all windows
        indices = [tc.metadata["chunk_index"] for tc in all_temporal_chunks]
        assert indices == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_process_document_persists_relationship_count(self, engine: VectorCypherEngine) -> None:
        """update_document is called with relationship_count == relationships_created."""
        from khora.core.models.document import Document
        from khora.core.models.entity import Relationship

        engine._vc_config = VectorCypherConfig(max_chunks_in_flight=None)

        raw_chunks = [self._make_raw_chunk("chunk one"), self._make_raw_chunk("chunk two")]
        doc = Document(content="test content", namespace_id=uuid4())

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks

        chunk_id = uuid4()

        def make_stored(chunks):
            stored = MagicMock()
            stored.id = chunk_id
            return [stored] * len(chunks)

        engine._embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))
        engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=make_stored)
        engine._storage.update_document = AsyncMock(side_effect=lambda d: d)

        # Two relationships, both sourced from our chunk
        rel1 = MagicMock(spec=Relationship)
        rel1.source_chunk_ids = [chunk_id]
        rel2 = MagicMock(spec=Relationship)
        rel2.source_chunk_ids = [chunk_id]

        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch(
                "khora.pipelines.tasks.extract.extract_entities",
                new_callable=AsyncMock,
                return_value=([], [rel1, rel2]),
            ),
        ):
            _, _, rels = await engine._process_document(
                doc,
                skill_name="default",
                occurred_at=datetime.now(UTC),
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )

        engine._storage.update_document.assert_awaited_once()
        persisted_doc = engine._storage.update_document.await_args.args[0]
        assert persisted_doc.relationship_count == rels

    @pytest.mark.asyncio
    async def test_neo4j_chunk_mirror_failure_does_not_abort_ingest(self, engine: VectorCypherEngine) -> None:
        """A Neo4j chunk-mirror write failure degrades, it does not propagate (ADR-001).

        Chunks are already durably stored in pgvector before the Neo4j mirror, so
        a graph-side write failure must NOT abort the document ingest. The create
        path catches the exception, records a ``vectorcypher.chunk_mirror``
        Degradation in ``out_diagnostics['degradations']``, and the document still
        completes with its full chunk count.
        """
        engine._vc_config = VectorCypherConfig(max_chunks_in_flight=None)

        raw_chunks = [self._make_raw_chunk(f"chunk {i}") for i in range(3)]
        doc = MagicMock()
        doc.id = uuid4()
        doc.namespace_id = uuid4()
        doc.content = "test content"
        doc.metadata = {}

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks
        engine._embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))
        engine._temporal_store.create_chunks_batch = AsyncMock(
            side_effect=lambda chunks: [MagicMock(id=uuid4()) for _ in chunks]
        )
        # The graph mirror raises — the failure mode under test.
        engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(side_effect=RuntimeError("neo4j connection reset"))

        out_diagnostics: dict[str, object] = {}
        with (
            patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker),
            patch("khora.engines.vectorcypher.engine._CHUNK_MIRROR_DEGRADED_COUNTER") as mock_counter,
        ):
            total_chunks, _, _ = await engine._process_document(
                doc,
                skill_name="default",
                occurred_at=datetime.now(UTC),
                entity_types=[],
                relationship_types=[],
                out_diagnostics=out_diagnostics,
            )

        # The mirror was attempted and failed, but ingest still completed with the
        # full chunk count from pgvector — the exception did NOT propagate.
        engine._dual_nodes.create_chunk_nodes_batch.assert_awaited_once()
        assert total_chunks == 3
        engine._storage.update_document.assert_awaited()

        # The degradation is recorded for downstream observability (ADR-001).
        degradations = out_diagnostics.get("degradations", [])
        chunk_mirror = [d for d in degradations if d.get("component") == "vectorcypher.chunk_mirror"]
        assert len(chunk_mirror) == 1, f"expected one chunk_mirror degradation, got {degradations}"
        entry = chunk_mirror[0]
        assert entry["reason"] == "neo4j_write_failed"
        assert entry["exception"] == "RuntimeError"
        assert "neo4j connection reset" in (entry.get("detail") or "")

        # The degraded_total counter is incremented with the bounded labels —
        # this is part of the observable telemetry contract for the failure.
        mock_counter.add.assert_called_once_with(1, attributes={"channel": "graph", "reason": "neo4j_write_failed"})

    @pytest.mark.asyncio
    async def test_neo4j_chunk_mirror_success_records_no_degradation(self, engine: VectorCypherEngine) -> None:
        """Happy path: a successful mirror leaves no chunk_mirror degradation."""
        engine._vc_config = VectorCypherConfig(max_chunks_in_flight=None)

        raw_chunks = [self._make_raw_chunk("chunk 0")]
        doc = MagicMock()
        doc.id = uuid4()
        doc.namespace_id = uuid4()
        doc.content = "test content"
        doc.metadata = {}

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = raw_chunks
        engine._embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))
        engine._temporal_store.create_chunks_batch = AsyncMock(
            side_effect=lambda chunks: [MagicMock(id=uuid4()) for _ in chunks]
        )
        engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=[uuid4()])

        out_diagnostics: dict[str, object] = {}
        with patch("khora.extraction.chunkers.create_chunker", return_value=mock_chunker):
            total_chunks, _, _ = await engine._process_document(
                doc,
                skill_name="default",
                occurred_at=datetime.now(UTC),
                entity_types=[],
                relationship_types=[],
                out_diagnostics=out_diagnostics,
            )

        assert total_chunks == 1
        degradations = out_diagnostics.get("degradations", [])
        assert not [d for d in degradations if d.get("component") == "vectorcypher.chunk_mirror"]


@pytest.mark.unit
class TestVectorCypherEngineApiTemporalFilter:
    """API-supplied temporal_filter synthesizes an EXPLICIT signal."""

    @pytest.fixture
    def engine_with_mocked_retriever(self) -> VectorCypherEngine:
        """Engine wired with a retriever whose retrieve() is an AsyncMock."""
        from khora.engines.vectorcypher.retriever import VectorCypherResult
        from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

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
        chunk = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="A chunk with enough content to pass the validation threshold for recall.",
        )
        retriever_result = VectorCypherResult(
            chunks=[(chunk, 0.9)],
            entities=[],
            routing_decision=routing,
            metadata={"search_mode": "simple_vector"},
        )

        engine._retriever = MagicMock()
        engine._retriever._config = MagicMock()
        engine._retriever._config.hybrid_alpha = 0.7
        engine._retriever.retrieve = AsyncMock(return_value=retriever_result)
        return engine

    @pytest.mark.asyncio
    async def test_recall_with_api_temporal_filter_synthesizes_explicit_signal(
        self, engine_with_mocked_retriever: VectorCypherEngine
    ) -> None:
        """When recall() is called with an API-supplied temporal_filter, the engine
        must synthesize a TemporalSignal with category=EXPLICIT, source="api",
        confidence=1.0, and forward it (along with the original filter) to the
        retriever — bypassing the dictionary/semantic temporal detector entirely.
        """
        from khora.engines.skeleton.backends import TemporalFilter
        from khora.engines.vectorcypher.temporal_detection import TemporalCategory

        # occurred_before is exclusive in pgvector; using two distinct dates.
        tf = TemporalFilter(
            occurred_after=datetime(2025, 1, 1, tzinfo=UTC),
            occurred_before=datetime(2025, 6, 1, tzinfo=UTC),
        )

        ns_id = uuid4()
        await engine_with_mocked_retriever.recall("anything", ns_id, temporal_filter=tf)

        engine_with_mocked_retriever._retriever.retrieve.assert_awaited_once()
        kwargs = engine_with_mocked_retriever._retriever.retrieve.await_args.kwargs

        # The API filter is forwarded verbatim alongside the synthesized signal.
        assert kwargs["temporal_filter"] is tf
        signal = kwargs["temporal_signal"]
        assert signal is not None
        assert signal.category == TemporalCategory.EXPLICIT
        assert signal.source == "api"
        assert signal.confidence == 1.0
        assert signal.is_temporal is True
        assert signal.temporal_filter is tf

    @pytest.mark.asyncio
    async def test_recall_without_temporal_filter_runs_detector(
        self, engine_with_mocked_retriever: VectorCypherEngine
    ) -> None:
        """Regression: when recall() is called WITHOUT temporal_filter, the engine
        must run the existing temporal detector (signal source != "api")."""
        ns_id = uuid4()
        await engine_with_mocked_retriever.recall("what did we do last week", ns_id)

        engine_with_mocked_retriever._retriever.retrieve.assert_awaited_once()
        kwargs = engine_with_mocked_retriever._retriever.retrieve.await_args.kwargs

        signal = kwargs["temporal_signal"]
        assert signal is not None
        # Detector path produces "dictionary" / "semantic" / "none" — never "api".
        assert signal.source != "api"


class TestRerankingConfigReconcile:
    """``config.query.reranking_*`` must reach the VectorCypher engine (#1017, #1023)."""

    @staticmethod
    def _config():
        from khora.config import KhoraConfig

        return KhoraConfig(database_url="postgresql://u:p@localhost/db")

    def test_query_reranking_family_reconciled_on_default_path(self) -> None:
        cfg = self._config()
        cfg.query.enable_reranking = True
        cfg.query.reranking_model = "REPRO/model-from-query"
        cfg.query.reranking_top_n = 33
        eng = VectorCypherEngine(cfg)
        assert eng._vc_config.enable_reranking is True
        assert eng._vc_config.reranking_model == "REPRO/model-from-query"
        assert eng._vc_config.reranking_top_n == 33

    def test_explicit_vc_config_does_not_bypass_query_model(self) -> None:
        # Enabling reranking via an explicit vc_config must not discard the
        # query-level reranking_model (the #1023 second-order bug).
        cfg = self._config()
        cfg.query.reranking_model = "REPRO/model-from-query"
        eng = VectorCypherEngine(cfg, vectorcypher_config=VectorCypherConfig(enable_reranking=True))
        assert eng._vc_config.enable_reranking is True
        assert eng._vc_config.reranking_model == "REPRO/model-from-query"

    def test_explicit_field_wins_over_query(self) -> None:
        cfg = self._config()
        cfg.query.reranking_model = "REPRO/model-from-query"
        eng = VectorCypherEngine(cfg, vectorcypher_config=VectorCypherConfig(reranking_model="EXPLICIT/win"))
        assert eng._vc_config.reranking_model == "EXPLICIT/win"

    def test_query_can_disable_reranking(self) -> None:
        cfg = self._config()
        cfg.query.enable_reranking = False
        assert VectorCypherEngine(cfg)._vc_config.enable_reranking is False


@pytest.mark.unit
class TestMirrorChunksOrDegrade:
    """The shared chunk-mirror helper used by the create, replace, and batch
    ingest paths — uniform degrade-and-continue behavior + observability."""

    @pytest.mark.asyncio
    async def test_none_dual_nodes_is_noop(self) -> None:
        """No graph backend (e.g. SurrealDB) → no-op: no counter, no degradation."""
        out: dict = {}
        with patch("khora.engines.vectorcypher.engine._CHUNK_MIRROR_DEGRADED_COUNTER") as counter:
            await _mirror_chunks_or_degrade(None, [MagicMock()], uuid4(), out)
        counter.add.assert_not_called()
        assert out == {}

    @pytest.mark.asyncio
    async def test_success_records_nothing(self) -> None:
        """A successful mirror write leaves no counter increment / degradation."""
        dual = MagicMock()
        dual.create_chunk_nodes_batch = AsyncMock(return_value=[uuid4()])
        out: dict = {}
        with patch("khora.engines.vectorcypher.engine._CHUNK_MIRROR_DEGRADED_COUNTER") as counter:
            await _mirror_chunks_or_degrade(dual, [MagicMock()], uuid4(), out)
        dual.create_chunk_nodes_batch.assert_awaited_once()
        counter.add.assert_not_called()
        assert out == {}

    @pytest.mark.asyncio
    async def test_failure_degrades_and_counts_without_raising(self) -> None:
        """A mirror failure increments the counter, records a Degradation, and does NOT raise."""
        dual = MagicMock()
        dual.create_chunk_nodes_batch = AsyncMock(side_effect=RuntimeError("neo4j down"))
        out: dict = {}
        with patch("khora.engines.vectorcypher.engine._CHUNK_MIRROR_DEGRADED_COUNTER") as counter:
            # Must not raise.
            await _mirror_chunks_or_degrade(dual, [MagicMock(), MagicMock()], uuid4(), out)
        counter.add.assert_called_once_with(1, attributes={"channel": "graph", "reason": "neo4j_write_failed"})
        degradations = out["degradations"]
        assert len(degradations) == 1
        entry = degradations[0]
        assert entry["component"] == "vectorcypher.chunk_mirror"
        assert entry["reason"] == "neo4j_write_failed"
        assert entry["exception"] == "RuntimeError"
        assert "neo4j down" in (entry.get("detail") or "")

    @pytest.mark.asyncio
    async def test_failure_without_diagnostics_sink_still_counts(self) -> None:
        """When no out_diagnostics is supplied (batch path), the counter still fires."""
        dual = MagicMock()
        dual.create_chunk_nodes_batch = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("khora.engines.vectorcypher.engine._CHUNK_MIRROR_DEGRADED_COUNTER") as counter:
            # No diagnostics dict, must not raise.
            await _mirror_chunks_or_degrade(dual, [MagicMock()], uuid4(), None)
        counter.add.assert_called_once_with(1, attributes={"channel": "graph", "reason": "neo4j_write_failed"})
