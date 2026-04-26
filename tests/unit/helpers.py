"""Shared test helpers for MemoryLake unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from khora.memory_lake import MemoryLake

# Stable row-level ID returned by resolve_namespace across all tests
RESOLVE_ROW_ID = uuid4()


def mock_config() -> MagicMock:
    """Create a mock KhoraConfig with all required methods."""
    mock_config = MagicMock()
    mock_config.get_postgresql_url.return_value = "postgresql://test"
    mock_config.get_graph_config.return_value = None
    mock_config.get_vector_config.return_value = None
    mock_config.get_neo4j_url.return_value = None
    mock_config.get_neo4j_user.return_value = None
    mock_config.get_neo4j_password.return_value = None
    mock_config.get_neo4j_database.return_value = None
    mock_config.storage.embedding_dimension = 1536
    mock_config.llm.model = "gpt-4o-mini"
    mock_config.llm.embedding_model = "text-embedding-3-small"
    mock_config.llm.embedding_dimension = 1536
    mock_config.llm.extraction_model = None
    mock_config.llm.timeout = 30
    mock_config.llm.max_retries = 3
    mock_config.telemetry_database_url = None
    mock_config.telemetry_service_name = "khora-test"
    # Disable pending recovery in unit tests to avoid background task noise.
    mock_config.pipelines.pending_recovery_enabled = False
    return mock_config


def mock_engine() -> MagicMock:
    """Create a mock engine with all required methods."""
    mock_eng = MagicMock()

    # Storage and embedder — resolve_namespace returns a distinct row-level ID
    mock_eng._storage = MagicMock()
    mock_eng._storage.resolve_namespace = AsyncMock(return_value=RESOLVE_ROW_ID)
    mock_eng._embedder = MagicMock()

    # Lifecycle
    mock_eng.connect = AsyncMock()
    mock_eng.disconnect = AsyncMock()
    mock_eng.health_check = AsyncMock(return_value={"status": "healthy"})

    # Core operations
    mock_eng.remember = AsyncMock()
    mock_eng.recall = AsyncMock()
    mock_eng.forget = AsyncMock()
    mock_eng.remember_batch = AsyncMock()

    # Namespace operations
    mock_eng.create_namespace = AsyncMock()
    mock_eng.get_namespace = AsyncMock()

    # Entity operations
    mock_eng.get_entity = AsyncMock()
    mock_eng.list_entities = AsyncMock(return_value=[])
    mock_eng.find_related_entities = AsyncMock(return_value=[])

    # Document operations
    mock_eng.get_document = AsyncMock()
    mock_eng.list_documents = AsyncMock(return_value=[])
    mock_eng.search_entities = AsyncMock(return_value=[])

    # Stats
    mock_eng.stats = AsyncMock()

    return mock_eng


def make_lake(*, connected: bool = False) -> MemoryLake:
    """Create a MemoryLake with mocked config, optionally pre-connected."""
    with patch("khora.memory_lake.load_config", return_value=mock_config()):
        lake = MemoryLake()

    if connected:
        lake._connected = True
        lake._engine = mock_engine()

    return lake
