"""Unit tests for _populate_sources with relationships."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models.document import Chunk, DocumentSource
from khora.core.models.entity import Entity, Relationship
from khora.khora import Khora


def _mock_config() -> MagicMock:
    """Create a mock KhoraConfig."""
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
    return mock_config


_RESOLVE_ROW_ID = uuid4()


def _mock_engine() -> MagicMock:
    """Create a mock engine."""
    mock_eng = MagicMock()
    mock_eng._storage = MagicMock()
    mock_eng._storage.resolve_namespace = AsyncMock(return_value=_RESOLVE_ROW_ID)
    mock_eng._embedder = MagicMock()
    mock_eng.connect = AsyncMock()
    mock_eng.disconnect = AsyncMock()
    mock_eng.recall = AsyncMock()
    return mock_eng


def _make_kb() -> Khora:
    """Create a connected Khora with mocked internals."""
    with patch("khora.khora.load_config", return_value=_mock_config()):
        kb = Khora()
    kb._connected = True
    kb._engine = _mock_engine()
    return kb


@pytest.mark.unit
class TestPopulateSourcesRelationships:
    """Tests for _populate_sources with relationships."""

    @pytest.mark.asyncio
    async def test_populate_sources_with_relationships(self) -> None:
        """Relationships with source_document_ids get source_documents populated."""
        kb = _make_kb()
        ns_id = uuid4()
        doc_id = uuid4()

        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="WORKS_AT",
            source_document_ids=[doc_id],
        )

        src = DocumentSource(id=doc_id, title="Contract")
        kb._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        await kb._populate_sources([], [], [(rel, 0.8)], namespace_id=ns_id)

        assert rel.source_documents == {doc_id: src}
        kb._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_populate_sources_relationships_empty_doc_ids(self) -> None:
        """Relationships with no source_document_ids get source_documents=None."""
        kb = _make_kb()

        rel = Relationship(
            namespace_id=uuid4(),
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="KNOWS",
            source_document_ids=[],
        )

        # No doc IDs to fetch → get_document_sources_batch should not be called
        kb._engine._storage.get_document_sources_batch = AsyncMock(return_value={})

        await kb._populate_sources([], [], [(rel, 0.5)], namespace_id=uuid4())

        assert rel.source_documents is None
        kb._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_populate_sources_empty_relationships_noop(self) -> None:
        """Empty relationships list doesn't cause errors."""
        kb = _make_kb()
        kb._engine._storage.get_document_sources_batch = AsyncMock(return_value={})

        # Should not raise
        await kb._populate_sources([], [], [], namespace_id=uuid4())

        kb._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_populate_sources_mixed(self) -> None:
        """Chunks, entities, and relationships all populated in one call."""
        kb = _make_kb()
        ns_id = uuid4()
        doc_id_1 = uuid4()
        doc_id_2 = uuid4()
        doc_id_3 = uuid4()

        chunk = Chunk(namespace_id=ns_id, document_id=doc_id_1, content="hello")
        entity = Entity(
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            source_document_ids=[doc_id_2],
        )
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="WORKS_AT",
            source_document_ids=[doc_id_3],
        )

        src_1 = DocumentSource(id=doc_id_1, title="Doc 1")
        src_2 = DocumentSource(id=doc_id_2, title="Doc 2")
        src_3 = DocumentSource(id=doc_id_3, title="Doc 3")
        kb._engine._storage.get_document_sources_batch = AsyncMock(
            return_value={doc_id_1: src_1, doc_id_2: src_2, doc_id_3: src_3}
        )

        await kb._populate_sources([(chunk, 0.9)], [(entity, 0.8)], [(rel, 0.7)], namespace_id=ns_id)

        assert chunk.source_document is src_1
        assert entity.source_documents == {doc_id_2: src_2}
        assert rel.source_documents == {doc_id_3: src_3}
        kb._engine._storage.get_document_sources_batch.assert_awaited_once()
