"""Unit tests for VectorCypher dual-node manager."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.dual_nodes import (
    ChunkNode,
    DualNodeManager,
    EntityChunkLink,
)


def _make_neo4j_driver() -> tuple[MagicMock, AsyncMock]:
    """Create a mock Neo4j driver with properly mocked session context manager.

    Returns:
        Tuple of (driver, session) where session is the mock session object.
    """
    driver = MagicMock()
    session = AsyncMock()

    # Neo4j driver.session() returns a sync object that supports `async with`
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx

    return driver, session


class TestChunkNode:
    """Tests for ChunkNode dataclass."""

    def test_create_chunk_node(self) -> None:
        """Test creating a ChunkNode with all fields."""
        chunk_id = uuid4()
        namespace_id = uuid4()
        document_id = uuid4()
        now = datetime.now(UTC)

        node = ChunkNode(
            id=chunk_id,
            namespace_id=namespace_id,
            document_id=document_id,
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            occurred_at=now,
            created_at=now,
            metadata={"key": "value"},
        )

        assert node.id == chunk_id
        assert node.namespace_id == namespace_id
        assert node.document_id == document_id
        assert node.content == "test content"
        assert node.embedding == [0.1, 0.2, 0.3]
        assert node.occurred_at == now

    def test_chunk_node_defaults(self) -> None:
        """Test ChunkNode default values."""
        node = ChunkNode(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="test",
        )
        assert node.embedding is None
        assert node.occurred_at is None
        assert node.created_at is None
        assert node.metadata is None


class TestEntityChunkLink:
    """Tests for EntityChunkLink dataclass."""

    def test_create_link(self) -> None:
        """Test creating an EntityChunkLink."""
        entity_id = uuid4()
        chunk_id = uuid4()
        link = EntityChunkLink(
            entity_id=entity_id,
            chunk_id=chunk_id,
            mention_count=3,
            context="mentioned in paragraph 2",
        )

        assert link.entity_id == entity_id
        assert link.chunk_id == chunk_id
        assert link.mention_count == 3
        assert link.context == "mentioned in paragraph 2"

    def test_link_defaults(self) -> None:
        """Test EntityChunkLink default values."""
        link = EntityChunkLink(entity_id=uuid4(), chunk_id=uuid4())
        assert link.mention_count == 1
        assert link.context == ""


class TestDualNodeManagerInit:
    """Tests for DualNodeManager initialization."""

    def test_init_default_database(self) -> None:
        """Test initialization with default database."""
        driver = MagicMock()
        manager = DualNodeManager(driver)
        assert manager._driver is driver
        assert manager._database == "neo4j"

    def test_init_custom_database(self) -> None:
        """Test initialization with custom database name."""
        driver = MagicMock()
        manager = DualNodeManager(driver, database="custom_db")
        assert manager._database == "custom_db"


@pytest.mark.unit
class TestDualNodeManagerEnsureIndexes:
    """Tests for ensure_indexes method."""

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_all(self) -> None:
        """Test that ensure_indexes runs all index creation queries."""
        driver, session = _make_neo4j_driver()
        session.run = AsyncMock()

        manager = DualNodeManager(driver)
        await manager.ensure_indexes()

        # Should run multiple index creation statements
        assert session.run.call_count >= 9  # At least 9 indexes defined

    @pytest.mark.asyncio
    async def test_ensure_indexes_handles_errors(self) -> None:
        """Test that index creation errors are handled gracefully."""
        driver, session = _make_neo4j_driver()
        session.run = AsyncMock(side_effect=Exception("index already exists"))

        manager = DualNodeManager(driver)
        # Should not raise
        await manager.ensure_indexes()


@pytest.mark.unit
class TestDualNodeManagerCreateChunkNode:
    """Tests for creating chunk nodes in Neo4j."""

    @pytest.fixture
    def manager(self) -> DualNodeManager:
        """Create a manager with mocked driver."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock()
        return DualNodeManager(driver)

    @pytest.mark.asyncio
    async def test_create_chunk_node(self, manager: DualNodeManager) -> None:
        """Test creating a single chunk node."""
        from khora.engines.skeleton.backends import TemporalChunk

        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="test chunk content",
            occurred_at=datetime.now(UTC),
        )

        result_id = await manager.create_chunk_node(chunk)
        assert result_id == chunk.id

    @pytest.mark.asyncio
    async def test_create_chunk_node_generates_id(self, manager: DualNodeManager) -> None:
        """Test that a UUID is generated if chunk has no ID."""
        from khora.engines.skeleton.backends import TemporalChunk

        chunk = TemporalChunk(
            id=None,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="test",
        )

        result_id = await manager.create_chunk_node(chunk)
        assert result_id is not None


@pytest.mark.unit
class TestDualNodeManagerBatchOperations:
    """Tests for batch chunk node operations."""

    @pytest.fixture
    def manager(self) -> DualNodeManager:
        """Create a manager with mocked driver."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock()
        return DualNodeManager(driver)

    @pytest.mark.asyncio
    async def test_create_chunk_nodes_batch(self, manager: DualNodeManager) -> None:
        """Test creating chunk nodes in batch."""
        from khora.engines.skeleton.backends import TemporalChunk

        namespace_id = uuid4()
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=namespace_id,
                document_id=uuid4(),
                content=f"chunk {i}",
                occurred_at=datetime.now(UTC),
            )
            for i in range(5)
        ]

        ids = await manager.create_chunk_nodes_batch(chunks, namespace_id)
        assert len(ids) == 5

    @pytest.mark.asyncio
    async def test_create_chunk_nodes_batch_empty(self, manager: DualNodeManager) -> None:
        """Test batch creation with empty list."""
        ids = await manager.create_chunk_nodes_batch([], uuid4())
        assert ids == []


@pytest.mark.unit
class TestDualNodeManagerLinkOperations:
    """Tests for entity-chunk linking operations."""

    @pytest.fixture
    def manager(self) -> DualNodeManager:
        """Create a manager with mocked driver."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock()
        return DualNodeManager(driver)

    @pytest.mark.asyncio
    async def test_link_entity_to_chunk(self, manager: DualNodeManager) -> None:
        """Test creating a single MENTIONED_IN relationship."""
        entity_id = uuid4()
        chunk_id = uuid4()

        await manager.link_entity_to_chunk(entity_id, chunk_id, mention_count=2)
        # Should not raise

    @pytest.mark.asyncio
    async def test_link_entities_to_chunks_batch(self, manager: DualNodeManager) -> None:
        """Test batch MENTIONED_IN relationship creation."""
        links = [
            EntityChunkLink(entity_id=uuid4(), chunk_id=uuid4(), mention_count=1),
            EntityChunkLink(entity_id=uuid4(), chunk_id=uuid4(), mention_count=3),
        ]

        await manager.link_entities_to_chunks_batch(links)
        # Should not raise

    @pytest.mark.asyncio
    async def test_link_entities_to_chunks_batch_empty(self) -> None:
        """Test batch linking with empty list is a no-op."""
        driver = MagicMock()
        manager = DualNodeManager(driver)
        await manager.link_entities_to_chunks_batch([])
        driver.session.assert_not_called()


@pytest.mark.unit
class TestDualNodeManagerTimeLinks:
    """Tests for chunk-to-time linking operations."""

    @pytest.fixture
    def manager(self) -> DualNodeManager:
        """Create a manager with mocked driver."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock()
        return DualNodeManager(driver)

    @pytest.mark.asyncio
    async def test_link_chunk_to_time(self, manager: DualNodeManager) -> None:
        """Test creating a single AT_TIME relationship."""
        chunk_id = uuid4()
        time_node_id = uuid4()

        await manager.link_chunk_to_time(chunk_id, time_node_id)
        # Should not raise

    @pytest.mark.asyncio
    async def test_link_chunks_to_time_batch(self, manager: DualNodeManager) -> None:
        """Test batch AT_TIME relationship creation."""
        links = [(uuid4(), uuid4()), (uuid4(), uuid4())]

        await manager.link_chunks_to_time_batch(links)
        # Should not raise

    @pytest.mark.asyncio
    async def test_link_chunks_to_time_batch_empty(self) -> None:
        """Test batch time linking with empty list is a no-op."""
        driver = MagicMock()
        manager = DualNodeManager(driver)
        await manager.link_chunks_to_time_batch([])
        driver.session.assert_not_called()


@pytest.mark.unit
class TestDualNodeManagerGetChunksByEntities:
    """Tests for get_chunks_by_entities method."""

    @pytest.mark.asyncio
    async def test_get_chunks_empty_entity_ids(self) -> None:
        """Test that empty entity IDs returns empty list."""
        driver = MagicMock()
        manager = DualNodeManager(driver)

        result = await manager.get_chunks_by_entities([], uuid4())
        assert result == []
        driver.session.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_chunks_by_entities(self) -> None:
        """Test fetching chunks connected to entities."""
        driver, session = _make_neo4j_driver()

        mock_records = [
            {
                "chunk_id": str(uuid4()),
                "content": "test content",
                "document_id": str(uuid4()),
                "occurred_at": None,
                "metadata": {},
                "entity_ids": [str(uuid4())],
                "total_mentions": 2,
            }
        ]
        session.execute_read = AsyncMock(return_value=mock_records)

        manager = DualNodeManager(driver)
        result = await manager.get_chunks_by_entities(
            [uuid4()],
            uuid4(),
            limit=10,
        )

        assert len(result) == 1
        assert result[0]["content"] == "test content"


@pytest.mark.unit
class TestDualNodeManagerGetEntityNeighborhoods:
    """Tests for get_entity_neighborhoods method."""

    @pytest.mark.asyncio
    async def test_get_neighborhoods_empty_ids(self) -> None:
        """Test that empty entity IDs returns empty dict."""
        driver = MagicMock()
        manager = DualNodeManager(driver)

        result = await manager.get_entity_neighborhoods([], uuid4())
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_neighborhoods_depth_clamped(self) -> None:
        """Test that depth is clamped to 1-4."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=[])

        manager = DualNodeManager(driver)

        # Depth 0 should be clamped to 1
        await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=0)
        # Depth 10 should be clamped to 4
        await manager.get_entity_neighborhoods([uuid4()], uuid4(), depth=10)


@pytest.mark.unit
class TestDualNodeManagerDeleteOperations:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_chunks_by_document(self) -> None:
        """Test deleting chunks by document ID."""
        driver, session = _make_neo4j_driver()
        session.execute_write = AsyncMock(return_value=5)

        manager = DualNodeManager(driver)
        deleted = await manager.delete_chunks_by_document(uuid4(), uuid4())

        assert deleted == 5


@pytest.mark.unit
class TestDualNodeManagerCountChunks:
    """Tests for count_chunks method."""

    @pytest.mark.asyncio
    async def test_count_chunks(self) -> None:
        """Test counting chunks in a namespace."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=42)

        manager = DualNodeManager(driver)
        count = await manager.count_chunks(uuid4())

        assert count == 42


@pytest.mark.unit
class TestDualNodeManagerGetEntityChannels:
    """Tests for get_entity_channels method."""

    @pytest.mark.asyncio
    async def test_get_entity_channels_returns_channels(self) -> None:
        """Test that get_entity_channels returns distinct channel strings."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=["session-1", "session-2", "session-3"])

        manager = DualNodeManager(driver)
        channels = await manager.get_entity_channels(
            entity_ids=[str(uuid4()), str(uuid4())],
            namespace_id=str(uuid4()),
        )

        assert channels == ["session-1", "session-2", "session-3"]
        session.execute_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entity_channels_empty_entity_ids(self) -> None:
        """Test that get_entity_channels returns empty list for no entities."""
        driver, session = _make_neo4j_driver()
        manager = DualNodeManager(driver)

        channels = await manager.get_entity_channels(
            entity_ids=[],
            namespace_id=str(uuid4()),
        )

        assert channels == []
        session.execute_read.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_entity_channels_single_channel(self) -> None:
        """Test that a single channel is returned correctly."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=["only-session"])

        manager = DualNodeManager(driver)
        channels = await manager.get_entity_channels(
            entity_ids=[str(uuid4())],
            namespace_id=str(uuid4()),
        )

        assert channels == ["only-session"]
