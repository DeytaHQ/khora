"""Unit tests for DualNodeManager.get_relationships_between()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.vectorcypher.dual_nodes import DualNodeManager


def _make_neo4j_driver() -> tuple[MagicMock, AsyncMock]:
    """Create a mock Neo4j driver with properly mocked session context manager."""
    driver = MagicMock()
    session = AsyncMock()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx

    return driver, session


@pytest.mark.unit
class TestGetRelationshipsBetween:
    """Tests for get_relationships_between method."""

    @pytest.mark.asyncio
    async def test_empty_entity_ids_returns_empty(self) -> None:
        """Empty entity_ids list returns [] without hitting Neo4j."""
        driver = MagicMock()
        manager = DualNodeManager(driver)

        result = await manager.get_relationships_between([], str(uuid4()))

        assert result == []
        driver.session.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_entity_returns_empty(self) -> None:
        """A single entity_id (len < 2) returns [] without hitting Neo4j."""
        driver = MagicMock()
        manager = DualNodeManager(driver)

        result = await manager.get_relationships_between([str(uuid4())], str(uuid4()))

        assert result == []
        driver.session.assert_not_called()

    @pytest.mark.asyncio
    async def test_basic_relationship_fetch(self) -> None:
        """Verify correct Cypher params are passed to the driver."""
        driver, session = _make_neo4j_driver()

        eid_1 = str(uuid4())
        eid_2 = str(uuid4())
        ns_id = str(uuid4())

        session.execute_read = AsyncMock(return_value=[])

        manager = DualNodeManager(driver)
        await manager.get_relationships_between([eid_1, eid_2], ns_id, limit=50)

        session.execute_read.assert_awaited_once()
        # Verify the work function was called — extract the callable
        work_fn = session.execute_read.call_args[0][0]
        # Create a mock transaction to inspect params
        mock_tx = AsyncMock()
        mock_result = AsyncMock()
        mock_result.__aiter__ = lambda self: self
        mock_result.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_tx.run = AsyncMock(return_value=mock_result)

        await work_fn(mock_tx)

        _, kwargs = mock_tx.run.call_args
        assert kwargs["entity_ids"] == [eid_1, eid_2]
        assert kwargs["namespace_id"] == ns_id
        assert kwargs["limit"] == 50

    @pytest.mark.asyncio
    async def test_result_parsing(self) -> None:
        """Mock Neo4j returns relationship records, verify dict structure."""
        driver, session = _make_neo4j_driver()

        rel_id = str(uuid4())
        src_id = str(uuid4())
        tgt_id = str(uuid4())
        doc_id = str(uuid4())
        chunk_id = str(uuid4())

        mock_records = [
            {
                "id": rel_id,
                "source_entity_id": src_id,
                "target_entity_id": tgt_id,
                "relationship_type": "WORKS_AT",
                "description": "Alice works at Acme",
                "confidence": 0.9,
                "weight": 1.0,
                "source_document_ids": [doc_id],
                "source_chunk_ids": [chunk_id],
            }
        ]
        session.execute_read = AsyncMock(return_value=mock_records)

        manager = DualNodeManager(driver)
        result = await manager.get_relationships_between([src_id, tgt_id], str(uuid4()))

        assert len(result) == 1
        assert result[0]["id"] == rel_id
        assert result[0]["source_entity_id"] == src_id
        assert result[0]["target_entity_id"] == tgt_id
        assert result[0]["relationship_type"] == "WORKS_AT"
        assert result[0]["description"] == "Alice works at Acme"
        assert result[0]["confidence"] == 0.9
        assert result[0]["source_document_ids"] == [doc_id]
        assert result[0]["source_chunk_ids"] == [chunk_id]

    @pytest.mark.asyncio
    async def test_default_limit(self) -> None:
        """Verify default limit=90 is passed to the query."""
        driver, session = _make_neo4j_driver()
        session.execute_read = AsyncMock(return_value=[])

        manager = DualNodeManager(driver)
        await manager.get_relationships_between([str(uuid4()), str(uuid4())], str(uuid4()))

        # Extract the work function and inspect the limit param
        work_fn = session.execute_read.call_args[0][0]
        mock_tx = AsyncMock()
        mock_result = AsyncMock()
        mock_result.__aiter__ = lambda self: self
        mock_result.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_tx.run = AsyncMock(return_value=mock_result)

        await work_fn(mock_tx)

        _, kwargs = mock_tx.run.call_args
        assert kwargs["limit"] == 90
