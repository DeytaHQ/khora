"""Unit tests for core domain models."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from khora.core.models import Chunk, Document, Entity, Relationship
from khora.core.models.document import ChunkMetadata, DocumentMetadata, DocumentStatus
from khora.core.models.entity import EntityType, Episode, RelationshipType
from khora.core.models.event import EventType, MemoryEvent
from khora.core.models.tenancy import MemoryNamespace, TenancyMode


class TestDocument:
    """Tests for Document model."""

    def test_create_document(self) -> None:
        """Test basic document creation."""
        doc = Document(content="This is test content.")
        assert doc.content == "This is test content."
        assert doc.status == DocumentStatus.PENDING
        assert doc.id is not None

    def test_document_with_metadata(self) -> None:
        """Test document with metadata."""
        metadata = DocumentMetadata(source="test", author="user", title="Test Doc")
        doc = Document(content="Content", metadata=metadata)
        assert doc.metadata.source == "test"
        assert doc.metadata.author == "user"
        assert doc.metadata.title == "Test Doc"

    def test_document_timestamps(self) -> None:
        """Test document timestamp handling."""
        now = datetime.now(timezone.utc)
        doc = Document(content="Content", created_at=now, updated_at=now)
        assert doc.created_at == now
        assert doc.updated_at == now

    def test_document_mark_processing(self) -> None:
        """Test marking document as processing."""
        doc = Document(content="Content")
        doc.mark_processing()
        assert doc.status == DocumentStatus.PROCESSING

    def test_document_mark_completed(self) -> None:
        """Test marking document as completed."""
        doc = Document(content="Content")
        doc.mark_completed(chunk_count=5, entity_count=3)
        assert doc.status == DocumentStatus.COMPLETED
        assert doc.chunk_count == 5
        assert doc.entity_count == 3
        assert doc.is_processed

    def test_document_mark_failed(self) -> None:
        """Test marking document as failed."""
        doc = Document(content="Content")
        doc.mark_failed("Processing error")
        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message == "Processing error"


class TestChunk:
    """Tests for Chunk model."""

    def test_create_chunk(self) -> None:
        """Test basic chunk creation."""
        chunk = Chunk(content="Chunk content")
        assert chunk.content == "Chunk content"
        assert chunk.id is not None

    def test_chunk_with_embedding(self) -> None:
        """Test chunk with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk = Chunk(
            content="Content",
            embedding=embedding,
            embedding_model="text-embedding-3-small",
        )
        assert chunk.embedding == embedding
        assert chunk.embedding_model == "text-embedding-3-small"
        assert chunk.has_embedding

    def test_chunk_without_embedding(self) -> None:
        """Test chunk without embedding."""
        chunk = Chunk(content="Content")
        assert not chunk.has_embedding

    def test_chunk_with_metadata(self) -> None:
        """Test chunk with metadata."""
        doc_id = uuid4()
        metadata = ChunkMetadata(
            document_id=doc_id,
            chunk_index=1,
            start_char=100,
            end_char=200,
            token_count=25,
        )
        chunk = Chunk(content="Content", metadata=metadata)
        assert chunk.metadata.document_id == doc_id
        assert chunk.metadata.chunk_index == 1
        assert chunk.metadata.start_char == 100
        assert chunk.metadata.end_char == 200
        assert chunk.metadata.token_count == 25


class TestEntity:
    """Tests for Entity model."""

    def test_create_entity(self) -> None:
        """Test basic entity creation."""
        entity = Entity(name="John Smith", entity_type=EntityType.PERSON)
        assert entity.name == "John Smith"
        assert entity.entity_type == EntityType.PERSON

    def test_entity_with_attributes(self) -> None:
        """Test entity with attributes."""
        entity = Entity(
            name="Acme Corp",
            entity_type=EntityType.ORGANIZATION,
            attributes={"industry": "Technology", "employees": 500},
        )
        assert entity.attributes["industry"] == "Technology"
        assert entity.attributes["employees"] == 500

    def test_entity_with_description(self) -> None:
        """Test entity with description."""
        entity = Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            description="A programming language",
        )
        assert entity.description == "A programming language"

    def test_entity_confidence(self) -> None:
        """Test entity confidence score."""
        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
            confidence=0.85,
        )
        assert entity.confidence == 0.85

    def test_entity_source_tracking(self) -> None:
        """Test entity source document/chunk tracking."""
        doc_id = uuid4()
        chunk_id = uuid4()
        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
            source_document_ids=[doc_id],
            source_chunk_ids=[chunk_id],
        )
        assert doc_id in entity.source_document_ids
        assert chunk_id in entity.source_chunk_ids

    def test_entity_mention_count(self) -> None:
        """Test entity mention counting."""
        entity = Entity(
            name="Test",
            entity_type=EntityType.CONCEPT,
            mention_count=5,
        )
        assert entity.mention_count == 5

    def test_entity_temporal_validity(self) -> None:
        """Test entity temporal validity range."""
        now = datetime.now(timezone.utc)
        entity = Entity(
            name="Test",
            entity_type=EntityType.EVENT,
            valid_from=now,
            valid_until=now,
        )
        assert entity.valid_from == now
        assert entity.valid_until == now

    def test_entity_merge(self) -> None:
        """Test merging two entities."""
        doc_id1 = uuid4()
        doc_id2 = uuid4()
        entity1 = Entity(
            name="Test",
            entity_type=EntityType.PERSON,
            source_document_ids=[doc_id1],
            mention_count=2,
            confidence=0.8,
        )
        entity2 = Entity(
            name="Test",
            entity_type=EntityType.PERSON,
            source_document_ids=[doc_id2],
            mention_count=3,
            confidence=0.9,
            description="A person",
        )
        entity1.merge_with(entity2)
        assert doc_id1 in entity1.source_document_ids
        assert doc_id2 in entity1.source_document_ids
        assert entity1.mention_count == 5
        assert entity1.confidence == 0.9
        assert entity1.description == "A person"


class TestRelationship:
    """Tests for Relationship model."""

    def test_create_relationship(self) -> None:
        """Test basic relationship creation."""
        source_id = uuid4()
        target_id = uuid4()
        rel = Relationship(
            source_entity_id=source_id,
            target_entity_id=target_id,
            relationship_type=RelationshipType.WORKS_FOR,
        )
        assert rel.source_entity_id == source_id
        assert rel.target_entity_id == target_id
        assert rel.relationship_type == RelationshipType.WORKS_FOR

    def test_relationship_with_properties(self) -> None:
        """Test relationship with properties."""
        rel = Relationship(
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type=RelationshipType.WORKS_FOR,
            properties={"since": "2020", "role": "Engineer"},
        )
        assert rel.properties["since"] == "2020"
        assert rel.properties["role"] == "Engineer"

    def test_relationship_confidence(self) -> None:
        """Test relationship confidence score."""
        rel = Relationship(
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type=RelationshipType.KNOWS,
            confidence=0.75,
        )
        assert rel.confidence == 0.75

    def test_relationship_description(self) -> None:
        """Test relationship with description."""
        rel = Relationship(
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type=RelationshipType.COLLABORATES_WITH,
            description="Worked together on Project X",
        )
        assert rel.description == "Worked together on Project X"

    def test_relationship_weight(self) -> None:
        """Test relationship weight."""
        rel = Relationship(
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type=RelationshipType.RELATES_TO,
            weight=0.5,
        )
        assert rel.weight == 0.5


class TestEpisode:
    """Tests for Episode model."""

    def test_create_episode(self) -> None:
        """Test basic episode creation."""
        episode = Episode(name="Meeting with client")
        assert episode.name == "Meeting with client"
        assert episode.id is not None

    def test_episode_with_entities(self) -> None:
        """Test episode with associated entities."""
        entity_ids = [uuid4(), uuid4()]
        episode = Episode(name="Team standup", entity_ids=entity_ids)
        assert len(episode.entity_ids) == 2

    def test_episode_temporal(self) -> None:
        """Test episode with temporal information."""
        now = datetime.now(timezone.utc)
        episode = Episode(name="Conference", occurred_at=now)
        assert episode.occurred_at == now

    def test_episode_duration(self) -> None:
        """Test episode duration and end_time property."""
        episode = Episode(name="Meeting", duration_seconds=3600)
        assert episode.duration_seconds == 3600
        assert episode.end_time is not None

    def test_episode_no_duration(self) -> None:
        """Test episode without duration has no end_time."""
        episode = Episode(name="Event")
        assert episode.end_time is None


class TestMemoryEvent:
    """Tests for MemoryEvent model."""

    def test_create_event(self) -> None:
        """Test basic event creation."""
        event = MemoryEvent(
            event_type=EventType.DOCUMENT_CREATED,
            resource_id=uuid4(),
            data={"title": "New doc"},
        )
        assert event.event_type == EventType.DOCUMENT_CREATED
        assert event.data["title"] == "New doc"
        assert event.resource_type == "document"

    def test_event_types(self) -> None:
        """Test different event types."""
        ns_id = uuid4()
        resource_id = uuid4()

        created = MemoryEvent(
            namespace_id=ns_id,
            event_type=EventType.DOCUMENT_CREATED,
            resource_id=resource_id,
        )
        assert created.event_type == EventType.DOCUMENT_CREATED

        updated = MemoryEvent(
            namespace_id=ns_id,
            event_type=EventType.DOCUMENT_UPDATED,
            resource_id=resource_id,
        )
        assert updated.event_type == EventType.DOCUMENT_UPDATED

        deleted = MemoryEvent(
            namespace_id=ns_id,
            event_type=EventType.DOCUMENT_DELETED,
            resource_id=resource_id,
        )
        assert deleted.event_type == EventType.DOCUMENT_DELETED

    def test_event_timestamp(self) -> None:
        """Test event timestamp."""
        event = MemoryEvent(event_type=EventType.ENTITY_CREATED)
        assert event.timestamp is not None

    def test_event_factory_methods(self) -> None:
        """Test event factory methods."""
        ns_id = uuid4()
        doc_id = uuid4()

        event = MemoryEvent.document_created(
            namespace_id=ns_id,
            document_id=doc_id,
            data={"content": "test"},
        )
        assert event.event_type == EventType.DOCUMENT_CREATED
        assert event.resource_id == doc_id
        assert event.resource_type == "document"

    def test_event_entity_created_factory(self) -> None:
        """Test entity_created factory method."""
        ns_id = uuid4()
        entity_id = uuid4()

        event = MemoryEvent.entity_created(
            namespace_id=ns_id,
            entity_id=entity_id,
            data={"name": "Test"},
        )
        assert event.event_type == EventType.ENTITY_CREATED
        assert event.resource_id == entity_id

    def test_event_resource_type_auto_extraction(self) -> None:
        """Test that resource_type is auto-extracted from event_type."""
        event = MemoryEvent(event_type=EventType.CHUNK_EMBEDDED)
        assert event.resource_type == "chunk"

        event2 = MemoryEvent(event_type=EventType.RELATIONSHIP_CREATED)
        assert event2.resource_type == "relationship"


class TestTenancyModels:
    """Tests for tenancy models (MemoryNamespace)."""

    def test_create_namespace(self) -> None:
        """Test namespace creation without workspace_id."""
        ns = MemoryNamespace(name="Project Alpha", slug="project-alpha")
        assert ns.name == "Project Alpha"
        assert ns.slug == "project-alpha"
        assert ns.id is not None

    def test_namespace_auto_slug(self) -> None:
        """Test namespace auto-generates slug from name."""
        ns = MemoryNamespace(name="My Project")
        assert ns.slug == "my-project"

    def test_namespace_with_config(self) -> None:
        """Test namespace with configuration overrides."""
        ns = MemoryNamespace(
            name="Test",
            slug="test",
            config_overrides={"extraction_skill": "technical_docs"},
        )
        assert ns.config_overrides["extraction_skill"] == "technical_docs"

    def test_namespace_sync_checkpoints(self) -> None:
        """Test namespace sync checkpoints."""
        ns = MemoryNamespace(
            name="Test",
            sync_checkpoints={"source1": "checkpoint123"},
        )
        assert ns.sync_checkpoints["source1"] == "checkpoint123"

    def test_namespace_tenancy_mode_defaults_to_shared(self) -> None:
        """Test that MemoryNamespace.tenancy_mode defaults to SHARED."""
        ns = MemoryNamespace(name="Test")
        assert ns.tenancy_mode == TenancyMode.SHARED

    def test_namespace_tenancy_mode_isolated(self) -> None:
        """Test that MemoryNamespace.tenancy_mode can be set to ISOLATED."""
        ns = MemoryNamespace(name="Test", tenancy_mode=TenancyMode.ISOLATED)
        assert ns.tenancy_mode == TenancyMode.ISOLATED

    def test_namespace_no_workspace_id_attribute(self) -> None:
        """Test that MemoryNamespace no longer has workspace_id."""
        ns = MemoryNamespace(name="Test")
        assert not hasattr(ns, "workspace_id")

    def test_namespace_no_full_path_attribute(self) -> None:
        """Test that MemoryNamespace no longer has full_path."""
        ns = MemoryNamespace(name="Test")
        assert not hasattr(ns, "full_path")
