"""Unit tests for citation source features (DYT-199).

Tests Source dataclass, chunk denormalization, entity source resolution,
source_tool persistence, and remember() source_tool parameter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models.document import Chunk, Document, DocumentMetadata
from khora.core.models.entity import Entity, EntityType
from khora.core.models.source import Source
from khora.memory_lake import BatchResult, MemoryLake, RecallResult, RememberResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config() -> MagicMock:
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


def _mock_engine() -> MagicMock:
    mock_eng = MagicMock()
    mock_eng._storage = MagicMock()
    mock_eng._embedder = MagicMock()
    mock_eng._default_namespace_id = None
    mock_eng.connect = AsyncMock()
    mock_eng.disconnect = AsyncMock()
    mock_eng.remember = AsyncMock()
    mock_eng.recall = AsyncMock()
    mock_eng.forget = AsyncMock()
    mock_eng.remember_batch = AsyncMock()
    mock_eng.get_or_create_default_namespace = AsyncMock(return_value=uuid4())
    mock_eng.get_entity = AsyncMock()
    mock_eng.list_entities = AsyncMock(return_value=[])
    mock_eng.find_related_entities = AsyncMock(return_value=[])
    mock_eng.get_document = AsyncMock()
    mock_eng.list_documents = AsyncMock(return_value=[])
    mock_eng.search_entities = AsyncMock(return_value=[])
    mock_eng.stats = AsyncMock()
    return mock_eng


def _make_lake(*, connected: bool = False) -> MemoryLake:
    with patch("khora.memory_lake.load_config", return_value=_mock_config()):
        lake = MemoryLake()
    if connected:
        lake._connected = True
        lake._engine = _mock_engine()
    return lake


# ---------------------------------------------------------------------------
# Source dataclass
# ---------------------------------------------------------------------------


class TestSource:
    """Tests for the Source dataclass."""

    def test_source_creation(self) -> None:
        """Source can be created with all fields."""
        doc_id = uuid4()
        s = Source(
            document_id=doc_id,
            title="Test Doc",
            url="https://example.com/doc",
            source_type="url",
            source_tool="slack",
        )
        assert s.document_id == doc_id
        assert s.title == "Test Doc"
        assert s.url == "https://example.com/doc"
        assert s.source_type == "url"
        assert s.source_tool == "slack"

    def test_source_defaults(self) -> None:
        """Source fields default to empty strings."""
        doc_id = uuid4()
        s = Source(document_id=doc_id)
        assert s.title == ""
        assert s.url == ""
        assert s.source_type == ""
        assert s.source_tool == ""

    def test_source_is_frozen(self) -> None:
        """Source is immutable (frozen dataclass)."""
        s = Source(document_id=uuid4(), title="Test")
        with pytest.raises(AttributeError):
            s.title = "Changed"  # type: ignore[misc]

    def test_source_equality(self) -> None:
        """Two Sources with same values are equal."""
        doc_id = uuid4()
        s1 = Source(document_id=doc_id, title="A")
        s2 = Source(document_id=doc_id, title="A")
        assert s1 == s2

    def test_source_hashable(self) -> None:
        """Source is hashable (frozen + slots)."""
        s = Source(document_id=uuid4())
        assert hash(s) is not None
        # Can be used in sets
        assert len({s, s}) == 1


# ---------------------------------------------------------------------------
# Chunk source field
# ---------------------------------------------------------------------------


class TestChunkSource:
    """Tests for the source field on Chunk."""

    def test_chunk_source_default_none(self) -> None:
        """Chunk.source defaults to None."""
        chunk = Chunk(content="test")
        assert chunk.source is None

    def test_chunk_source_set(self) -> None:
        """Chunk.source can be set to a Source object."""
        doc_id = uuid4()
        source = Source(document_id=doc_id, title="My Doc", source_tool="linear")
        chunk = Chunk(content="test", source=source)
        assert chunk.source is source
        assert chunk.source.title == "My Doc"
        assert chunk.source.source_tool == "linear"


# ---------------------------------------------------------------------------
# Entity source_documents field
# ---------------------------------------------------------------------------


class TestEntitySourceDocuments:
    """Tests for the source_documents field on Entity."""

    def test_entity_source_documents_default_empty(self) -> None:
        """Entity.source_documents defaults to empty list."""
        entity = Entity(name="test", entity_type=EntityType.CONCEPT)
        assert entity.source_documents == []

    def test_entity_source_documents_mutable(self) -> None:
        """Entity.source_documents can be mutated in-place."""
        entity = Entity(name="test", entity_type=EntityType.CONCEPT)
        doc_id = uuid4()
        entity.source_documents = [Source(document_id=doc_id, title="Doc")]
        assert len(entity.source_documents) == 1
        assert entity.source_documents[0].title == "Doc"


# ---------------------------------------------------------------------------
# Chunk ingest — Source from document
# ---------------------------------------------------------------------------


class TestChunkIngest:
    """Tests for Source population during document chunking."""

    @pytest.mark.asyncio
    async def test_chunk_document_populates_source(self) -> None:
        """chunk_document() builds Source from parent document metadata."""
        from khora.pipelines.tasks.chunk import chunk_document

        doc = Document(
            content="This is a test document with enough content to create at least one chunk.",
            metadata=DocumentMetadata(
                title="Slack Message",
                source="https://slack.com/msg/123",
                source_type="url",
                source_tool="slack",
            ),
        )

        chunks = await chunk_document(doc, strategy="fixed", chunk_size=512, chunk_overlap=0)
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.source is not None
            assert chunk.source.document_id == doc.id
            assert chunk.source.title == "Slack Message"
            assert chunk.source.url == "https://slack.com/msg/123"
            assert chunk.source.source_type == "url"
            assert chunk.source.source_tool == "slack"

    @pytest.mark.asyncio
    async def test_chunk_document_empty_metadata(self) -> None:
        """chunk_document() creates Source even with empty metadata."""
        from khora.pipelines.tasks.chunk import chunk_document

        doc = Document(content="Some content for chunking test.")

        chunks = await chunk_document(doc, strategy="fixed", chunk_size=512, chunk_overlap=0)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.source is not None
            assert chunk.source.document_id == doc.id
            assert chunk.source.title == ""
            assert chunk.source.url == ""


# ---------------------------------------------------------------------------
# Entity source resolution in MemoryLake
# ---------------------------------------------------------------------------


class TestEntitySourceResolution:
    """Tests for _resolve_entity_sources in MemoryLake."""

    @pytest.mark.asyncio
    async def test_resolve_entity_sources(self) -> None:
        """_resolve_entity_sources populates source_documents from docs."""
        lake = _make_lake(connected=True)

        doc_id_1 = uuid4()
        doc_id_2 = uuid4()
        ns_id = uuid4()

        doc1 = Document(
            id=doc_id_1,
            namespace_id=ns_id,
            content="doc1",
            metadata=DocumentMetadata(
                title="Doc 1",
                source="https://example.com/1",
                source_type="url",
                source_tool="slack",
            ),
        )
        doc2 = Document(
            id=doc_id_2,
            namespace_id=ns_id,
            content="doc2",
            metadata=DocumentMetadata(
                title="Doc 2",
                source="https://example.com/2",
                source_type="api",
                source_tool="linear",
            ),
        )

        lake._engine._storage.get_documents_batch = AsyncMock(return_value={doc_id_1: doc1, doc_id_2: doc2})

        entity = Entity(
            name="test entity",
            entity_type=EntityType.CONCEPT,
            namespace_id=ns_id,
            source_document_ids=[doc_id_1, doc_id_2],
        )

        await lake._resolve_entity_sources([entity])

        assert len(entity.source_documents) == 2
        sources_by_id = {s.document_id: s for s in entity.source_documents}
        assert sources_by_id[doc_id_1].title == "Doc 1"
        assert sources_by_id[doc_id_1].source_tool == "slack"
        assert sources_by_id[doc_id_2].title == "Doc 2"
        assert sources_by_id[doc_id_2].source_tool == "linear"

    @pytest.mark.asyncio
    async def test_resolve_entity_sources_deleted_document_stub(self) -> None:
        """Deleted documents produce stub Source with empty metadata."""
        lake = _make_lake(connected=True)

        existing_doc_id = uuid4()
        deleted_doc_id = uuid4()
        ns_id = uuid4()

        doc = Document(
            id=existing_doc_id,
            namespace_id=ns_id,
            content="exists",
            metadata=DocumentMetadata(title="Exists"),
        )

        # Only return one doc — the other is "deleted"
        lake._engine._storage.get_documents_batch = AsyncMock(return_value={existing_doc_id: doc})

        entity = Entity(
            name="test",
            entity_type=EntityType.CONCEPT,
            namespace_id=ns_id,
            source_document_ids=[existing_doc_id, deleted_doc_id],
        )

        await lake._resolve_entity_sources([entity])

        assert len(entity.source_documents) == 2
        sources_by_id = {s.document_id: s for s in entity.source_documents}
        assert sources_by_id[existing_doc_id].title == "Exists"
        # Deleted doc gets stub with empty fields
        assert sources_by_id[deleted_doc_id].title == ""
        assert sources_by_id[deleted_doc_id].url == ""

    @pytest.mark.asyncio
    async def test_resolve_entity_sources_no_doc_ids_skips(self) -> None:
        """Entities with no source_document_ids skip resolution."""
        lake = _make_lake(connected=True)
        lake._engine._storage.get_documents_batch = AsyncMock(return_value={})

        entity = Entity(name="orphan", entity_type=EntityType.CONCEPT)

        await lake._resolve_entity_sources([entity])

        assert entity.source_documents == []
        # get_documents_batch should not be called
        lake._engine._storage.get_documents_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recall_resolves_entity_sources(self) -> None:
        """recall() resolves entity sources before returning."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        entity = Entity(
            name="test",
            entity_type=EntityType.CONCEPT,
            namespace_id=ns_id,
            source_document_ids=[doc_id],
        )

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[(entity, 0.9)],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        doc = Document(
            id=doc_id,
            namespace_id=ns_id,
            content="doc",
            metadata=DocumentMetadata(title="Recalled Doc", source_tool="slack"),
        )
        lake._engine._storage.get_documents_batch = AsyncMock(return_value={doc_id: doc})

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test")

        # Entity should have source_documents populated
        resolved_entity = result.entities[0][0]
        assert len(resolved_entity.source_documents) == 1
        assert resolved_entity.source_documents[0].title == "Recalled Doc"
        assert resolved_entity.source_documents[0].source_tool == "slack"

    @pytest.mark.asyncio
    async def test_search_entities_resolves_sources(self) -> None:
        """search_entities() resolves entity sources."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        entity = Entity(
            name="test",
            entity_type=EntityType.CONCEPT,
            namespace_id=ns_id,
            source_document_ids=[doc_id],
        )
        lake._engine.search_entities = AsyncMock(return_value=[entity])

        doc = Document(
            id=doc_id,
            namespace_id=ns_id,
            content="doc",
            metadata=DocumentMetadata(title="Search Doc"),
        )
        lake._engine._storage.get_documents_batch = AsyncMock(return_value={doc_id: doc})

        entities = await lake.search_entities("test")

        assert len(entities) == 1
        assert len(entities[0].source_documents) == 1
        assert entities[0].source_documents[0].title == "Search Doc"

    @pytest.mark.asyncio
    async def test_find_related_resolves_sources(self) -> None:
        """find_related_entities() resolves entity sources."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()
        entity_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        entity = Entity(
            id=entity_id,
            name="related",
            entity_type=EntityType.CONCEPT,
            namespace_id=ns_id,
            source_document_ids=[doc_id],
        )
        lake._engine.find_related_entities = AsyncMock(return_value=[(entity, 0.8)])

        doc = Document(
            id=doc_id,
            namespace_id=ns_id,
            content="doc",
            metadata=DocumentMetadata(title="Related Doc"),
        )
        lake._engine._storage.get_documents_batch = AsyncMock(return_value={doc_id: doc})

        results = await lake.find_related_entities(entity_id)

        assert len(results) == 1
        assert len(results[0][0].source_documents) == 1
        assert results[0][0].source_documents[0].title == "Related Doc"


# ---------------------------------------------------------------------------
# remember() source_tool parameter
# ---------------------------------------------------------------------------


class TestRememberSourceTool:
    """Tests for source_tool parameter on remember()."""

    @pytest.mark.asyncio
    async def test_remember_passes_source_tool(self) -> None:
        """remember() passes source_tool to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember("content", source_tool="slack")

        call_kwargs = lake._engine.remember.call_args
        assert call_kwargs.kwargs.get("source_tool") == "slack"

    @pytest.mark.asyncio
    async def test_remember_source_tool_default_empty(self) -> None:
        """remember() defaults source_tool to empty string."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember("content")

        call_kwargs = lake._engine.remember.call_args
        assert call_kwargs.kwargs.get("source_tool") == ""


# ---------------------------------------------------------------------------
# Source export from core.models
# ---------------------------------------------------------------------------


class TestSourceExport:
    """Tests that Source is properly exported."""

    def test_source_importable_from_models(self) -> None:
        """Source can be imported from khora.core.models."""
        from khora.core.models import Source as ImportedSource

        assert ImportedSource is Source

    def test_source_in_models_all(self) -> None:
        """Source is listed in core.models.__all__."""
        import khora.core.models

        assert "Source" in khora.core.models.__all__


# ---------------------------------------------------------------------------
# source_tool persistence on DocumentModel
# ---------------------------------------------------------------------------


class TestDocumentModelSourceTool:
    """Tests that source_tool is on DocumentModel."""

    def test_document_model_has_source_tool_column(self) -> None:
        """DocumentModel has a source_tool mapped column."""
        from khora.db.models import DocumentModel

        assert hasattr(DocumentModel, "source_tool")

    def test_chunk_model_has_source_columns(self) -> None:
        """ChunkModel has denormalized source columns."""
        from khora.db.models import ChunkModel

        for col in ("source_title", "source_url", "source_type", "source_tool"):
            assert hasattr(ChunkModel, col), f"ChunkModel missing {col}"


# ---------------------------------------------------------------------------
# Neo4j source_tool in entity params
# ---------------------------------------------------------------------------


class TestNeo4jSourceTool:
    """Tests for source_tool in Neo4j entity serialization."""

    def test_entity_to_cypher_params_includes_source_tool(self) -> None:
        """_entity_to_cypher_params includes source_tool."""
        from khora.storage.backends.neo4j import _entity_to_cypher_params

        entity = Entity(
            name="test",
            entity_type=EntityType.CONCEPT,
            source_tool="slack",
        )
        params = _entity_to_cypher_params(entity)
        assert params["source_tool"] == "slack"

    def test_entity_to_cypher_params_empty_source_tool(self) -> None:
        """_entity_to_cypher_params handles empty source_tool."""
        from khora.storage.backends.neo4j import _entity_to_cypher_params

        entity = Entity(name="test", entity_type=EntityType.CONCEPT)
        params = _entity_to_cypher_params(entity)
        assert params["source_tool"] == ""


# ---------------------------------------------------------------------------
# M3: remember_batch() source_tool passthrough
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberBatchSourceTool:
    """Tests for source_tool passthrough in remember_batch()."""

    @pytest.mark.asyncio
    async def test_remember_batch_passes_source_tool_in_documents(self) -> None:
        """remember_batch() forwards source_tool from document dicts to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        lake._engine.remember_batch = AsyncMock(
            return_value=BatchResult(
                total=2,
                processed=2,
                skipped=0,
                failed=0,
                chunks=4,
                entities=2,
                relationships=1,
            )
        )

        docs = [
            {"content": "Slack message about project", "source_tool": "slack"},
            {"content": "Linear issue description", "source_tool": "linear"},
        ]

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember_batch(docs)

        assert isinstance(result, BatchResult)
        assert result.total == 2
        assert result.processed == 2

        # Verify the engine received the documents with source_tool intact
        call_args = lake._engine.remember_batch.call_args
        forwarded_docs = call_args.args[0]
        assert forwarded_docs[0]["source_tool"] == "slack"
        assert forwarded_docs[1]["source_tool"] == "linear"

    @pytest.mark.asyncio
    async def test_remember_batch_source_tool_absent_defaults_gracefully(self) -> None:
        """remember_batch() handles documents without source_tool."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        lake._engine.remember_batch = AsyncMock(
            return_value=BatchResult(
                total=1,
                processed=1,
                skipped=0,
                failed=0,
                chunks=2,
                entities=1,
                relationships=0,
            )
        )

        docs = [{"content": "Plain document without source_tool"}]

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember_batch(docs)

        assert isinstance(result, BatchResult)
        assert result.processed == 1

        # Document should be forwarded as-is (no source_tool key injected by MemoryLake)
        call_args = lake._engine.remember_batch.call_args
        forwarded_docs = call_args.args[0]
        assert "source_tool" not in forwarded_docs[0] or forwarded_docs[0].get("source_tool", "") == ""


# ---------------------------------------------------------------------------
# M4: pgvector chunk write/read round-trip with Source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPgVectorChunkSourceRoundTrip:
    """Tests for Source data surviving pgvector _chunk_model_to_domain round-trip."""

    def test_chunk_model_to_domain_with_source(self) -> None:
        """_chunk_model_to_domain reconstructs Source from denormalized columns."""
        from datetime import UTC, datetime
        from types import SimpleNamespace

        from khora.storage.backends.pgvector import PgVectorBackend

        doc_id = uuid4()
        chunk_id = uuid4()
        ns_id = uuid4()

        # Simulate a ChunkModel row with source columns populated
        model = SimpleNamespace(
            id=chunk_id,
            namespace_id=ns_id,
            document_id=doc_id,
            content="Test chunk content",
            chunk_index=0,
            start_char=0,
            end_char=18,
            token_count=4,
            metadata_={},
            source_title="Slack Message",
            source_url="https://slack.com/msg/123",
            source_type="url",
            source_tool="slack",
            embedding=None,
            embedding_model="text-embedding-3-small",
            created_at=datetime(2024, 6, 15, tzinfo=UTC),
            source_timestamp=None,
        )

        backend = PgVectorBackend.__new__(PgVectorBackend)
        chunk = backend._chunk_model_to_domain(model)

        assert chunk.source is not None
        assert chunk.source.document_id == doc_id
        assert chunk.source.title == "Slack Message"
        assert chunk.source.url == "https://slack.com/msg/123"
        assert chunk.source.source_type == "url"
        assert chunk.source.source_tool == "slack"

    def test_chunk_model_to_domain_empty_source_columns_preserves_source(self) -> None:
        """_chunk_model_to_domain creates Source even when all source columns are empty (H1 fix).

        This is the H1 fix: chunk_document() always creates a Source from the
        parent document, so the round-trip must preserve it even if all metadata
        fields are empty strings. Source is created whenever document_id is present.
        """
        from datetime import UTC, datetime
        from types import SimpleNamespace

        from khora.storage.backends.pgvector import PgVectorBackend

        doc_id = uuid4()

        # Simulate a ChunkModel with all source columns as empty strings
        model = SimpleNamespace(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=doc_id,
            content="Chunk with empty metadata",
            chunk_index=0,
            start_char=0,
            end_char=24,
            token_count=5,
            metadata_={},
            source_title="",
            source_url="",
            source_type="",
            source_tool="",
            embedding=None,
            embedding_model="",
            created_at=datetime(2024, 6, 15, tzinfo=UTC),
            source_timestamp=None,
        )

        backend = PgVectorBackend.__new__(PgVectorBackend)
        chunk = backend._chunk_model_to_domain(model)

        # H1 fix: Source is always created when document_id is present,
        # even with empty metadata fields — round-trip preservation
        assert chunk.source is not None
        assert chunk.source.document_id == doc_id
        assert chunk.source.title == ""
        assert chunk.source.url == ""
        assert chunk.source.source_type == ""
        assert chunk.source.source_tool == ""

    def test_chunk_model_to_domain_null_document_id_no_source(self) -> None:
        """_chunk_model_to_domain returns source=None when document_id is None."""
        from datetime import UTC, datetime
        from types import SimpleNamespace

        from khora.storage.backends.pgvector import PgVectorBackend

        model = SimpleNamespace(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=None,
            content="Orphan chunk",
            chunk_index=0,
            start_char=0,
            end_char=12,
            token_count=2,
            metadata_={},
            source_title="",
            source_url="",
            source_type="",
            source_tool="",
            embedding=None,
            embedding_model="",
            created_at=datetime(2024, 6, 15, tzinfo=UTC),
            source_timestamp=None,
        )

        backend = PgVectorBackend.__new__(PgVectorBackend)
        chunk = backend._chunk_model_to_domain(model)

        # No document_id => no Source
        assert chunk.source is None

    def test_chunk_model_to_domain_partial_source(self) -> None:
        """_chunk_model_to_domain creates Source when at least one source column is non-empty."""
        from datetime import UTC, datetime
        from types import SimpleNamespace

        from khora.storage.backends.pgvector import PgVectorBackend

        doc_id = uuid4()

        # Only source_tool is set (e.g., document with source_tool but no title/url)
        model = SimpleNamespace(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=doc_id,
            content="Chunk with partial source",
            chunk_index=0,
            start_char=0,
            end_char=25,
            token_count=4,
            metadata_={},
            source_title="",
            source_url="",
            source_type="",
            source_tool="linear",
            embedding=None,
            embedding_model="",
            created_at=datetime(2024, 6, 15, tzinfo=UTC),
            source_timestamp=None,
        )

        backend = PgVectorBackend.__new__(PgVectorBackend)
        chunk = backend._chunk_model_to_domain(model)

        # At least one field non-empty => Source created
        assert chunk.source is not None
        assert chunk.source.document_id == doc_id
        assert chunk.source.source_tool == "linear"
        assert chunk.source.title == ""
        assert chunk.source.url == ""
