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
    # Default to empty maps — tests that exercise the recall document
    # upgrade pass can override these per-test.
    mock_eng._storage.get_document_sources_batch = AsyncMock(return_value={})
    mock_eng._storage.get_document_projections_batch = AsyncMock(return_value={})

    # #932: the pending processor re-loads the full Document by id at dequeue
    # time. Default create_document/update_document to a tiny in-memory store
    # keyed by doc id, and default get_document to answer from it - falling
    # back to the recorded create_document/update_document call args so the
    # worker's re-load still finds the row even when a test overrides
    # create_document/update_document with its own AsyncMock. Tests can still
    # override get_document explicitly.
    _doc_store: dict = {}

    async def _store_doc(doc):
        _doc_store[doc.id] = doc
        return doc

    async def _load_doc(doc_id, *, namespace_id):
        if doc_id in _doc_store:
            return _doc_store[doc_id]
        # Fall back to whatever was last passed to create/update_document,
        # even if a test reassigned those mocks (call_args_list is read live).
        for mock in (mock_eng._storage.create_document, mock_eng._storage.update_document):
            call_args_list = getattr(mock, "call_args_list", [])
            for call in reversed(call_args_list):
                passed = call.args[0] if call.args else None
                if passed is not None and getattr(passed, "id", None) == doc_id:
                    return passed
        return None

    mock_eng._storage.create_document = AsyncMock(side_effect=_store_doc)
    mock_eng._storage.update_document = AsyncMock(side_effect=_store_doc)
    mock_eng._storage.get_document = AsyncMock(side_effect=_load_doc)
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
