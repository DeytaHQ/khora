"""Unit tests for the typed-entity recency fast-path retriever (issue #569)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from khora.engines.vectorcypher.retriever import (
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.query.router import QueryComplexity, RoutingDecision


def _routing_decision() -> RoutingDecision:
    return RoutingDecision(
        complexity=QueryComplexity.TYPED_ENTITY_RECENT,
        use_graph=True,
        graph_depth=1,
        confidence=0.95,
        reasoning="test",
        suggested_entry_limit=10,
    )


def _make_retriever_with_mock_session(rows: list[dict]) -> tuple[VectorCypherRetriever, MagicMock]:
    """Build a minimal retriever with a mocked dual_nodes session.

    Bypasses ``__init__`` to avoid constructing the full pgvector / Neo4j
    infrastructure: the fast-path method only touches ``self._dual_nodes``
    and ``self._vectorcypher_retrieve`` (for fallback).
    """
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)

    # Build a dual_nodes mock whose ``_session()`` async-context yields a
    # session whose ``execute_read`` returns the canned rows.
    session = AsyncMock()
    session.execute_read = AsyncMock(return_value=rows)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    dual_nodes = MagicMock()
    dual_nodes._session = MagicMock(return_value=session_ctx)

    retriever._dual_nodes = dual_nodes  # type: ignore[attr-defined]
    # Spy for fallback assertions.
    retriever._vectorcypher_retrieve = AsyncMock(  # type: ignore[attr-defined]
        return_value=VectorCypherResult(
            chunks=[],
            entities=[],
            routing_decision=_routing_decision(),
            metadata={"from_fallback": True},
        )
    )
    return retriever, session


def _make_row(
    entity_name: str,
    last_mention_iso: str,
    chunk_content: str = "evidence",
) -> dict:
    return {
        "entity": {
            "id": str(uuid4()),
            "name": entity_name,
            "entity_type": "ACTION_ITEM",
            "description": "",
            "source_tool": "",
            "valid_from": None,
            "valid_until": None,
            "confidence": 1.0,
        },
        "last_mention": last_mention_iso,
        "evidence_chunk": {
            "id": str(uuid4()),
            "document_id": str(uuid4()),
            "content": chunk_content,
            "occurred_at": last_mention_iso,
        },
    }


class TestTypedEntityRecentRetrieve:
    async def test_returns_entities_ordered_by_last_mention_desc(self) -> None:
        # Neo4j returned three ACTION_ITEM entities, already ordered DESC by
        # last_mention (server side). Verify the retriever preserves order.
        rows = [
            _make_row("Ship login fix", "2026-05-12T10:00:00+00:00", "newest"),
            _make_row("Add audit logs", "2026-05-10T10:00:00+00:00", "mid"),
            _make_row("Rotate secrets", "2026-05-01T10:00:00+00:00", "oldest"),
        ]
        retriever, _session = _make_retriever_with_mock_session(rows)

        ns = uuid4()
        result = await retriever._typed_entity_recent_retrieve(
            query="latest action items",
            query_embedding=[0.0] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=None,
            limit=10,
            routing=_routing_decision(),
        )

        assert len(result.entities) == 3
        names = [e.name for e, _ in result.entities]
        assert names == ["Ship login fix", "Add audit logs", "Rotate secrets"]
        # First chunk should correspond to the newest entity.
        assert result.chunks[0][0].content == "newest"
        # Scores are monotonically non-increasing.
        scores = [s for _, s in result.entities]
        assert scores == sorted(scores, reverse=True)
        assert result.metadata["typed_entity_fast_path"] is True
        assert result.metadata["typed_entity_type"] == "ACTION_ITEM"

    async def test_zero_rows_falls_back_to_vectorcypher(self) -> None:
        rows: list[dict] = []
        retriever, _session = _make_retriever_with_mock_session(rows)

        ns = uuid4()
        result = await retriever._typed_entity_recent_retrieve(
            query="latest action items",
            query_embedding=[0.0] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=None,
            limit=10,
            routing=_routing_decision(),
        )

        # The fallback spy was called exactly once.
        retriever._vectorcypher_retrieve.assert_awaited_once()  # type: ignore[attr-defined]
        # The returned result carries the fallback metadata flag.
        assert result.metadata.get("from_fallback") is True
        assert result.metadata.get("typed_entity_fast_path_fallback") is True
        assert result.metadata.get("typed_entity_type") == "ACTION_ITEM"

    async def test_cypher_query_string_contains_required_clauses(self) -> None:
        # Capture the cypher string passed to tx.run by intercepting
        # execute_read's worker callable.
        captured: dict = {}

        async def _capture_execute_read(work_fn):
            tx = AsyncMock()
            tx_result = AsyncMock()

            async def _aiter():
                if False:  # pragma: no cover — empty iterator
                    yield None

            tx_result.__aiter__ = lambda self: _aiter()

            async def _run(cypher: str, **params):
                captured["cypher"] = cypher
                captured["params"] = params
                return tx_result

            tx.run = _run
            return await work_fn(tx)

        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        session = AsyncMock()
        session.execute_read = _capture_execute_read

        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        dual_nodes = MagicMock()
        dual_nodes._session = MagicMock(return_value=session_ctx)
        retriever._dual_nodes = dual_nodes  # type: ignore[attr-defined]
        retriever._vectorcypher_retrieve = AsyncMock(  # type: ignore[attr-defined]
            return_value=VectorCypherResult(chunks=[], entities=[], routing_decision=_routing_decision(), metadata={})
        )

        ns = uuid4()
        await retriever._typed_entity_recent_retrieve(
            query="latest action items",
            query_embedding=[0.0] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=None,
            limit=10,
            routing=_routing_decision(),
        )

        cypher = captured["cypher"]
        assert "entity_type: $entity_type" in cypher
        assert "ORDER BY last_mention DESC" in cypher
        assert "MENTIONED_IN" in cypher
        assert "max(c.occurred_at) AS last_mention" in cypher
        # status filter is applied for ACTION_ITEM
        assert "a.status IS NULL OR NOT" in cypher
        # params include entity_type / namespace_id / limit
        assert captured["params"]["entity_type"] == "ACTION_ITEM"
        assert captured["params"]["namespace_id"] == str(ns)
        assert captured["params"]["limit"] == 10

    async def test_no_status_filter_for_decision(self) -> None:
        """DECISION/RISK/BLOCKER don't carry a status — no filter clause."""
        captured: dict = {}

        async def _capture_execute_read(work_fn):
            tx = AsyncMock()
            tx_result = AsyncMock()

            async def _aiter():
                if False:  # pragma: no cover
                    yield None

            tx_result.__aiter__ = lambda self: _aiter()

            async def _run(cypher: str, **params):
                captured["cypher"] = cypher
                return tx_result

            tx.run = _run
            return await work_fn(tx)

        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        session = AsyncMock()
        session.execute_read = _capture_execute_read

        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        dual_nodes = MagicMock()
        dual_nodes._session = MagicMock(return_value=session_ctx)
        retriever._dual_nodes = dual_nodes  # type: ignore[attr-defined]
        retriever._vectorcypher_retrieve = AsyncMock(  # type: ignore[attr-defined]
            return_value=VectorCypherResult(chunks=[], entities=[], routing_decision=_routing_decision(), metadata={})
        )

        await retriever._typed_entity_recent_retrieve(
            query="latest decisions",
            query_embedding=[0.0] * 4,
            namespace_id=uuid4(),
            temporal_filter=None,
            graph_depth=None,
            limit=10,
            routing=_routing_decision(),
        )

        assert "a.status" not in captured["cypher"]

    async def test_no_dual_nodes_falls_back(self) -> None:
        """If neo4j is unavailable (no DualNodeManager), fall back."""
        retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
        retriever._dual_nodes = None  # type: ignore[attr-defined]
        retriever._vectorcypher_retrieve = AsyncMock(  # type: ignore[attr-defined]
            return_value=VectorCypherResult(chunks=[], entities=[], routing_decision=_routing_decision(), metadata={})
        )

        await retriever._typed_entity_recent_retrieve(
            query="latest action items",
            query_embedding=[0.0] * 4,
            namespace_id=uuid4(),
            temporal_filter=None,
            graph_depth=None,
            limit=10,
            routing=_routing_decision(),
        )

        retriever._vectorcypher_retrieve.assert_awaited_once()  # type: ignore[attr-defined]
