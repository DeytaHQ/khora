"""Unit tests for the Skeleton Construction engine."""

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from khora.engines.skeleton.backends import TemporalChunk, TemporalFilter, TemporalSearchResult


class TestTemporalFilter:
    """Tests for TemporalFilter dataclass."""

    def test_empty_filter(self):
        """Test creating an empty filter."""
        f = TemporalFilter()
        assert f.occurred_after is None
        assert f.occurred_before is None
        assert f.source_system is None
        assert f.tags is None
        assert f.additional == {}

    def test_filter_with_time_range(self):
        """Test filter with time range."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 2, 1, tzinfo=UTC)

        f = TemporalFilter(occurred_after=start, occurred_before=end)

        assert f.occurred_after == start
        assert f.occurred_before == end

    def test_filter_with_metadata(self):
        """Test filter with metadata fields."""
        f = TemporalFilter(
            source_system="slack",
            author="alice",
            channel="general",
            tags=["important", "followup"],
        )

        assert f.source_system == "slack"
        assert f.author == "alice"
        assert f.channel == "general"
        assert f.tags == ["important", "followup"]

    def test_filter_with_additional(self):
        """Test filter with additional structured filters."""
        f = TemporalFilter(
            additional={
                "confidence": {"gte": 0.8},
                "metadata.priority": {"eq": "high"},
            }
        )

        assert f.additional["confidence"] == {"gte": 0.8}
        assert f.additional["metadata.priority"] == {"eq": "high"}


class TestTemporalChunk:
    """Tests for TemporalChunk dataclass."""

    def test_create_chunk(self):
        """Test creating a temporal chunk."""
        chunk_id = uuid4()
        namespace_id = uuid4()
        document_id = uuid4()
        now = datetime.now(UTC)

        chunk = TemporalChunk(
            id=chunk_id,
            namespace_id=namespace_id,
            document_id=document_id,
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            occurred_at=now,
            created_at=now,
            source_system="slack",
            author="alice",
            channel="general",
            tags=["test"],
            confidence=0.95,
            metadata={"key": "value"},
        )

        assert chunk.id == chunk_id
        assert chunk.namespace_id == namespace_id
        assert chunk.document_id == document_id
        assert chunk.content == "Test content"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.occurred_at == now
        assert chunk.source_system == "slack"
        assert chunk.author == "alice"
        assert chunk.tags == ["test"]
        assert chunk.confidence == 0.95

    def test_chunk_defaults(self):
        """Test chunk default values."""
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Test",
        )

        assert chunk.embedding is None
        assert chunk.occurred_at is None
        assert chunk.source_system is None
        assert chunk.tags == []
        assert chunk.confidence == 1.0
        assert chunk.metadata == {}


class TestTemporalSearchResult:
    """Tests for TemporalSearchResult dataclass."""

    def test_search_result(self):
        """Test creating a search result."""
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Test",
        )

        result = TemporalSearchResult(
            chunk=chunk,
            similarity=0.85,
            bm25_score=0.6,
            combined_score=0.75,
        )

        assert result.chunk == chunk
        assert result.similarity == 0.85
        assert result.bm25_score == 0.6
        assert result.combined_score == 0.75

    def test_search_result_vector_only(self):
        """Test search result with vector search only."""
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Test",
        )

        result = TemporalSearchResult(
            chunk=chunk,
            similarity=0.85,
        )

        assert result.similarity == 0.85
        assert result.bm25_score is None
        assert result.combined_score is None


class TestSkeletonConstructionEngineFilterBuilding:
    """Tests for SkeletonConstructionEngine filter building."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = None
        config.get_neo4j_user.return_value = None
        config.get_neo4j_password.return_value = None
        config.get_neo4j_database.return_value = None
        config.get_graph_config.return_value = None
        config.get_vector_config.return_value = None
        config.storage.embedding_dimension = 1536
        config.llm.model = "gpt-4o-mini"
        config.llm.embedding_model = "text-embedding-3-small"
        config.llm.embedding_dimension = 1536
        config.llm.timeout = 30
        config.llm.max_retries = 3
        config.llm.extraction_model = None
        config.pipeline.chunking_strategy = "recursive"
        config.pipeline.chunk_size = 1000
        config.pipeline.chunk_overlap = 200
        config.telemetry_database_url = None
        config.telemetry_service_name = "test"
        return config

    def test_build_temporal_filter_from_dict(self, mock_config):
        """Test building TemporalFilter from dict."""
        from khora.engines.skeleton.engine import SkeletonConstructionEngine

        engine = SkeletonConstructionEngine(mock_config, backend="pgvector")

        filters = {
            "occurred_at": {"gte": "2024-01-01", "lt": "2024-02-01"},
            "author": {"eq": "alice"},
            "source_system": {"eq": "slack"},
            "tags": {"contains": ["important"]},
        }

        tf = engine._build_temporal_filter_from_dict(filters)

        assert tf.occurred_after == datetime(2024, 1, 1, tzinfo=UTC)
        assert tf.occurred_before == datetime(2024, 2, 1, tzinfo=UTC)
        assert tf.author == "alice"
        assert tf.source_system == "slack"
        assert tf.tags == ["important"]

    def test_build_temporal_filter_simple_values(self, mock_config):
        """Test building TemporalFilter with simple values (not dicts)."""
        from khora.engines.skeleton.engine import SkeletonConstructionEngine

        engine = SkeletonConstructionEngine(mock_config, backend="pgvector")

        filters = {
            "author": "alice",
            "channel": "general",
        }

        tf = engine._build_temporal_filter_from_dict(filters)

        # Simple values are converted to {"eq": value}
        assert tf.author == "alice"
        assert tf.channel == "general"

    def test_parse_datetime_iso(self, mock_config):
        """Test parsing ISO datetime strings."""
        from khora.engines.skeleton.engine import SkeletonConstructionEngine

        engine = SkeletonConstructionEngine(mock_config, backend="pgvector")

        # ISO format with timezone
        dt = engine._parse_datetime("2024-01-15T10:30:00+00:00")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 10

        # ISO format with Z
        dt = engine._parse_datetime("2024-01-15T10:30:00Z")
        assert dt.year == 2024

        # Date only
        dt = engine._parse_datetime("2024-01-15")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_datetime_object(self, mock_config):
        """Test that datetime objects pass through."""
        from khora.engines.skeleton.engine import SkeletonConstructionEngine

        engine = SkeletonConstructionEngine(mock_config, backend="pgvector")

        now = datetime.now(UTC)
        result = engine._parse_datetime(now)
        assert result == now


class TestCreateTemporalStore:
    """Tests for create_temporal_store factory."""

    def test_create_pgvector_store(self):
        """Test creating pgvector store."""
        from khora.engines.skeleton.backends import create_temporal_store
        from khora.engines.skeleton.backends.pgvector import PgVectorTemporalStore

        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.llm.embedding_dimension = 1536

        store = create_temporal_store("pgvector", config)
        assert isinstance(store, PgVectorTemporalStore)

    def test_create_weaviate_store(self):
        """Test creating weaviate store."""
        from khora.engines.skeleton.backends import create_temporal_store
        from khora.engines.skeleton.backends.weaviate import WeaviateTemporalStore

        config = MagicMock()
        config.llm.embedding_dimension = 1536

        store = create_temporal_store("weaviate", config, weaviate_url="http://localhost:8080")
        assert isinstance(store, WeaviateTemporalStore)

    def test_create_weaviate_store_requires_url(self):
        """Test that weaviate store requires URL."""
        from khora.engines.skeleton.backends import create_temporal_store

        config = MagicMock()

        with pytest.raises(ValueError, match="weaviate_url is required"):
            create_temporal_store("weaviate", config)

    def test_create_unknown_backend_raises(self):
        """Test that unknown backend raises error."""
        from khora.engines.skeleton.backends import create_temporal_store

        config = MagicMock()

        with pytest.raises(ValueError, match="Unknown backend"):
            create_temporal_store("unknown", config)


@pytest.mark.unit
class TestEngineRegistration:
    """Tests for engine registration."""

    def test_skeleton_engine_registered(self):
        """Test that skeleton engine is registered."""
        from khora.engines import list_engines

        engines = list_engines()
        assert "skeleton" in engines
        assert "graphrag" in engines

    def test_create_skeleton_engine(self):
        """Test creating Skeleton Construction engine via factory."""
        from khora.engines import create_engine

        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = None
        config.get_neo4j_user.return_value = None
        config.get_neo4j_password.return_value = None
        config.get_neo4j_database.return_value = None
        config.get_graph_config.return_value = None
        config.get_vector_config.return_value = None
        config.storage.embedding_dimension = 1536
        config.llm.model = "gpt-4o-mini"
        config.llm.embedding_model = "text-embedding-3-small"
        config.llm.embedding_dimension = 1536
        config.llm.timeout = 30
        config.llm.max_retries = 3

        engine = create_engine("skeleton", config)

        from khora.engines.skeleton.engine import SkeletonConstructionEngine

        assert isinstance(engine, SkeletonConstructionEngine)
