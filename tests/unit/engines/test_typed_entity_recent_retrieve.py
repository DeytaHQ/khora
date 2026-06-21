"""Unit tests for the typed-entity recency fast-path retriever (issue #569)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.filter import RecallFilter, parse_to_ast
from khora.query import SearchMode
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


def _ast(spec: dict | None):
    """Build the canonical recall-filter AST exactly as the public facade does."""
    if spec is None:
        return None
    return parse_to_ast(RecallFilter.model_validate(spec))


def _make_dispatch_retriever(complexity: QueryComplexity) -> VectorCypherRetriever:
    """A retriever wired so ``retrieve()`` runs to the route-dispatch gate.

    Bypasses ``__init__`` and stubs only what ``retrieve()`` touches before the
    dispatch (``_config``, ``_embedder``, ``_router``). Both downstream paths are
    spies so the test asserts WHICH one the gate selected:

    * ``_typed_entity_recent_retrieve`` — the #569 fast path.
    * ``_vectorcypher_retrieve`` — the filtered / fallback path.

    The router is mocked to return ``complexity`` (and ``use_graph``), so the
    routing decision is fixed regardless of the query text.
    """
    retriever = VectorCypherRetriever.__new__(VectorCypherRetriever)
    retriever._config = RetrieverConfig(  # type: ignore[attr-defined]
        temporal_recency_floor_enabled=False,
        temporal_llm_disambiguation_enabled=False,
        # Isolate the dispatch-gate assertions from the HyDE embed step (#1018):
        # these queries route non-SIMPLE, so default "auto" would fire HyDE.
        enable_hyde="never",
    )
    retriever._hyde_expander = None  # type: ignore[attr-defined]

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.0] * 4)
    embedder.model_name = "test-model"
    embedder.dimension = 4
    embedder.cache_stats = {"hits": 0}
    retriever._embedder = embedder  # type: ignore[attr-defined]

    router = MagicMock()
    router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=complexity,
            use_graph=complexity is not QueryComplexity.SIMPLE,
            graph_depth=1,
            confidence=0.95,
            reasoning="test",
            suggested_entry_limit=10,
        )
    )
    router.compute_adaptive_depth = MagicMock(return_value=1)
    retriever._router = router  # type: ignore[attr-defined]

    # Both downstream paths are spies returning a benign result.
    def _result(meta: dict) -> VectorCypherResult:
        return VectorCypherResult(
            chunks=[],
            entities=[],
            routing_decision=router.route.return_value,
            metadata=meta,
        )

    retriever._typed_entity_recent_retrieve = AsyncMock(  # type: ignore[attr-defined]
        return_value=_result({"typed_entity_fast_path": True})
    )
    retriever._vectorcypher_retrieve = AsyncMock(  # type: ignore[attr-defined]
        return_value=_result({"from_fallback": True})
    )
    retriever._simple_retrieve = AsyncMock(  # type: ignore[attr-defined]
        return_value=_result({"from_simple": True})
    )
    return retriever


class TestTypedEntityFastPathDispatchGate:
    """The route-dispatch gate enters the fast path ONLY when ``filter_ast is None``."""

    async def test_filtered_typed_recent_skips_fast_path(self) -> None:
        """TYPED_ENTITY_RECENT + a non-empty filter -> fast path NOT entered.

        A filtered typed-recent recall falls through to ``_vectorcypher_retrieve``
        (which enforces + reports the filter per channel); the fast path's Cypher
        cannot enforce caller filters, so the gate must skip it. The SAME
        ``filter_ast`` instance is forwarded to ``_vectorcypher_retrieve``, and the
        result carries no ``typed_entity_fast_path`` marker.
        """
        ns = uuid4()
        ast = _ast({"source_name": "linear", "metadata.tier": "gold"})
        retriever = _make_dispatch_retriever(QueryComplexity.TYPED_ENTITY_RECENT)

        result = await retriever.retrieve("latest action items", ns, limit=10, filter_ast=ast)

        # Fast path was NOT entered.
        retriever._typed_entity_recent_retrieve.assert_not_awaited()  # type: ignore[attr-defined]
        # The filtered path ran and received the SAME filter_ast.
        retriever._vectorcypher_retrieve.assert_awaited_once()  # type: ignore[attr-defined]
        assert retriever._vectorcypher_retrieve.await_args.kwargs["filter_ast"] is ast  # type: ignore[attr-defined]
        assert "typed_entity_fast_path" not in result.metadata

    async def test_unfiltered_typed_recent_takes_fast_path(self) -> None:
        """TYPED_ENTITY_RECENT + ``filter_ast=None`` -> fast path STILL taken (#569).

        Guards the latency fast path against regression from the gate: with no
        caller filter the gate condition holds and the fast path runs, stamping
        ``typed_entity_fast_path=True``.
        """
        ns = uuid4()
        retriever = _make_dispatch_retriever(QueryComplexity.TYPED_ENTITY_RECENT)

        result = await retriever.retrieve("latest action items", ns, limit=10, filter_ast=None)

        retriever._typed_entity_recent_retrieve.assert_awaited_once()  # type: ignore[attr-defined]
        retriever._vectorcypher_retrieve.assert_not_awaited()  # type: ignore[attr-defined]
        assert result.metadata["typed_entity_fast_path"] is True

    async def test_simple_mode_unaffected_by_filter(self) -> None:
        """``mode=VECTOR``/``KEYWORD`` (force_simple) routes to ``_simple_retrieve``.

        The new ``filter_ast is None`` gate condition lives on the TYPED-recent
        branch only; ``force_simple`` short-circuits ahead of it, so a filtered
        OR unfiltered VECTOR/KEYWORD recall routes to ``_simple_retrieve`` exactly
        as before, never the typed fast path.
        """
        ns = uuid4()
        ast = _ast({"source_name": "linear"})
        for mode in (SearchMode.VECTOR, SearchMode.KEYWORD):
            for fast in (ast, None):
                retriever = _make_dispatch_retriever(QueryComplexity.TYPED_ENTITY_RECENT)

                await retriever.retrieve("latest action items", ns, limit=10, mode=mode, filter_ast=fast)

                retriever._simple_retrieve.assert_awaited_once()  # type: ignore[attr-defined]
                retriever._typed_entity_recent_retrieve.assert_not_awaited()  # type: ignore[attr-defined]
                retriever._vectorcypher_retrieve.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_graph_mode_unaffected_by_filter(self) -> None:
        """``mode=GRAPH`` (force_graph) routes to ``_vectorcypher_retrieve`` regardless.

        ``force_graph`` forces the entity-expansion path and is evaluated ahead of
        the TYPED-recent branch, so a GRAPH recall — filtered or not — bypasses
        both the fast path and the simple path. The new gate condition does not
        perturb this branch.
        """
        ns = uuid4()
        ast = _ast({"source_name": "linear"})
        for fast in (ast, None):
            retriever = _make_dispatch_retriever(QueryComplexity.TYPED_ENTITY_RECENT)

            await retriever.retrieve("latest action items", ns, limit=10, mode=SearchMode.GRAPH, filter_ast=fast)

            retriever._vectorcypher_retrieve.assert_awaited_once()  # type: ignore[attr-defined]
            retriever._typed_entity_recent_retrieve.assert_not_awaited()  # type: ignore[attr-defined]
            retriever._simple_retrieve.assert_not_awaited()  # type: ignore[attr-defined]
