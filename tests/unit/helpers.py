"""Shared test helpers for Khora unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from khora.khora import Khora

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
    # Disable pending processor in unit tests to avoid background task noise.
    mock_config.pipelines.pending_processor_enabled = False
    mock_config.pipelines.pending_processor_max_concurrent = 20
    mock_config.pipelines.pending_processor_grace_period_minutes = 5
    # Deprecated fields — keep None to avoid backwards-compat fallback.
    mock_config.pipelines.pending_recovery_enabled = None
    mock_config.pipelines.pending_recovery_grace_period_minutes = None
    return mock_config


def mock_engine() -> MagicMock:
    """Create a mock engine with all required methods."""
    mock_eng = MagicMock()

    # Storage and embedder — resolve_namespace returns a distinct row-level ID
    mock_eng._storage = MagicMock()
    mock_eng._storage.resolve_namespace = AsyncMock(return_value=RESOLVE_ROW_ID)
    _empty_ns_page = MagicMock()
    _empty_ns_page.items = []
    mock_eng._storage.list_namespaces = AsyncMock(return_value=_empty_ns_page)
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


def make_kb(*, connected: bool = False) -> Khora:
    """Create a Khora with mocked config, optionally pre-connected."""
    with patch("khora.khora.load_config", return_value=mock_config()):
        kb = Khora()

    if connected:
        kb._connected = True
        kb._engine = mock_engine()

    return kb
