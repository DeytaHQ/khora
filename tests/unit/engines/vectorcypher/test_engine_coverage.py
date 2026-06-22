"""Coverage push for ``khora.engines.vectorcypher.engine``.

These tests target still-uncovered helpers + simple-orchestration branches in
VectorCypherEngine (issue #695 coverage ladder). They mock at the storage /
temporal-store / embedder / dual-nodes boundary so no live services are
required.

Blocks targeted:
    - ``_ensure_tags`` module helper (lines 65-77)
    - ``_coerce_session_id_from_metadata`` module helper (lines 94-99)
    - ``_bfs_distances_from`` extended coverage (lines 115-143)
    - ``_build_cooccurrence_relationships`` co-occurrence cap + dedup
    - ``VectorCypherConfig.__post_init__`` validation (lines 265-267)
    - ``VectorCypherEngine._neo4j_driver_kwargs`` static helper
    - ``_build_conversation_context`` (lines 1980-2000)
    - ``_detect_temporal_filter`` regex path (lines 2848-2882)
    - ``_parse_datetime`` LongMemEval/dateparser fallback (lines 2884-2922)
    - ``find_related_entities`` graph-only + dual-nodes paths (lines 2962-3015)
    - ``search_entities`` empty + populated paths (lines 3018-3046)
    - ``list_entities``, ``get_entity``, ``list_documents``, ``get_document``,
      ``create_namespace``, ``get_namespace`` (passthroughs)
    - ``stats`` degradation when count_* raises (lines 3065-3102)
    - ``health_check`` all-healthy / neo4j-unhealthy branches
    - ``clear_document_extraction_state`` with errors swallowed
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk, Entity, Relationship
from khora.engines.vectorcypher.engine import (
    VectorCypherConfig,
    VectorCypherEngine,
    _bfs_distances_from,
    _build_cooccurrence_relationships,
    _coerce_session_id_from_metadata,
    _ensure_tags,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnsureTags:
    def test_list_passthrough(self) -> None:
        assert _ensure_tags(["a", "b"]) == ["a", "b"]

    def test_json_list_string_parsed(self) -> None:
        # JSON-encoded list (Postgres jsonb column round-trip)
        assert _ensure_tags('["x", "y"]') == ["x", "y"]

    def test_non_json_string_wrapped(self) -> None:
        # Bare string is wrapped into a single-element list
        assert _ensure_tags("solo") == ["solo"]

    def test_empty_string_returns_empty(self) -> None:
        assert _ensure_tags("") == []

    def test_invalid_json_string_wrapped(self) -> None:
        # Looks like JSON but malformed — falls back to wrap
        assert _ensure_tags("[not, json") == ["[not, json"]

    def test_json_string_parsed_not_list_wrapped(self) -> None:
        # JSON object (not list) → falls back to wrap behavior
        assert _ensure_tags('{"k": 1}') == ['{"k": 1}']

    def test_none_returns_empty(self) -> None:
        assert _ensure_tags(None) == []

    def test_int_returns_empty(self) -> None:
        assert _ensure_tags(42) == []


@pytest.mark.unit
class TestCoerceSessionId:
    def test_none_input(self) -> None:
        assert _coerce_session_id_from_metadata(None) is None

    def test_empty_dict(self) -> None:
        assert _coerce_session_id_from_metadata({}) is None

    def test_missing_key(self) -> None:
        assert _coerce_session_id_from_metadata({"other": "value"}) is None

    def test_none_value(self) -> None:
        assert _coerce_session_id_from_metadata({"session_id": None}) is None

    def test_empty_string(self) -> None:
        assert _coerce_session_id_from_metadata({"session_id": ""}) is None

    def test_uuid_passthrough(self) -> None:
        sid = uuid4()
        assert _coerce_session_id_from_metadata({"session_id": sid}) == sid

    def test_uuid_string(self) -> None:
        sid = uuid4()
        assert _coerce_session_id_from_metadata({"session_id": str(sid)}) == sid

    def test_invalid_string_returns_none(self) -> None:
        assert _coerce_session_id_from_metadata({"session_id": "not-a-uuid"}) is None

    def test_unrelated_type_returns_none(self) -> None:
        # int → str(int) → UUID(...) raises ValueError → None
        assert _coerce_session_id_from_metadata({"session_id": 42}) is None


# ---------------------------------------------------------------------------
# _bfs_distances_from (covered partly by test_vectorcypher_find_related_distance)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBfsDistancesFromExtra:
    def test_dict_shaped_relationship_in_out(self) -> None:
        seed = uuid4()
        b = uuid4()
        c = uuid4()
        # SurrealDB-shaped dict using "in"/"out" keys
        rels = [
            {"in": seed, "out": b},
            {"in": b, "out": c},
        ]
        distances = _bfs_distances_from(seed, rels)
        assert distances[seed] == 0
        assert distances[b] == 1
        assert distances[c] == 2

    def test_dict_shaped_relationship_from_to(self) -> None:
        seed = uuid4()
        b = uuid4()
        rels = [{"from": seed, "to": b}]
        distances = _bfs_distances_from(seed, rels)
        assert distances == {seed: 0, b: 1}

    def test_dict_rel_with_missing_endpoints_skipped(self) -> None:
        seed = uuid4()
        b = uuid4()
        rels = [{"unrelated": "stuff"}, {"in": seed, "out": b}]
        distances = _bfs_distances_from(seed, rels)
        assert distances == {seed: 0, b: 1}


# ---------------------------------------------------------------------------
# _build_cooccurrence_relationships
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildCooccurrenceRelationships:
    def test_no_pairs_when_single_entity_per_chunk(self) -> None:
        ns = uuid4()
        doc = uuid4()
        chunk = uuid4()
        e1 = Entity(name="A", entity_type="PERSON", source_chunk_ids=[chunk])
        chunks = [Chunk(id=chunk, namespace_id=ns, document_id=doc)]
        rels = _build_cooccurrence_relationships([e1], chunks, ns, [])
        assert rels == []

    def test_pairs_created_for_shared_chunk(self) -> None:
        ns = uuid4()
        doc = uuid4()
        chunk = uuid4()
        e1 = Entity(name="A", entity_type="PERSON", source_chunk_ids=[chunk])
        e2 = Entity(name="B", entity_type="PERSON", source_chunk_ids=[chunk])
        chunks = [Chunk(id=chunk, namespace_id=ns, document_id=doc)]
        rels = _build_cooccurrence_relationships([e1, e2], chunks, ns, [])
        assert len(rels) == 1
        assert rels[0].relationship_type == "ASSOCIATED_WITH"
        assert rels[0].namespace_id == ns
        # Pair (min, max) ordering is internal but should be normalized.
        pair = {rels[0].source_entity_id, rels[0].target_entity_id}
        assert pair == {e1.id, e2.id}

    def test_existing_pair_skipped(self) -> None:
        ns = uuid4()
        doc = uuid4()
        chunk = uuid4()
        e1 = Entity(name="A", entity_type="PERSON", source_chunk_ids=[chunk])
        e2 = Entity(name="B", entity_type="PERSON", source_chunk_ids=[chunk])
        # Pre-existing edge with same pair (in different orientation)
        existing = Relationship(
            source_entity_id=e2.id,
            target_entity_id=e1.id,
            relationship_type="KNOWS",
            namespace_id=ns,
        )
        chunks = [Chunk(id=chunk, namespace_id=ns, document_id=doc)]
        rels = _build_cooccurrence_relationships([e1, e2], chunks, ns, [existing])
        assert rels == []

    def test_per_chunk_cap(self) -> None:
        """Co-occurrence is capped at 15 pairs per chunk to prevent quadratic explosion."""
        ns = uuid4()
        doc = uuid4()
        chunk = uuid4()
        # 7 entities in a single chunk → C(7,2)=21 raw pairs, capped at 15
        entities = [Entity(name=f"E{i}", entity_type="X", source_chunk_ids=[chunk]) for i in range(7)]
        chunks = [Chunk(id=chunk, namespace_id=ns, document_id=doc)]
        rels = _build_cooccurrence_relationships(entities, chunks, ns, [])
        assert len(rels) == 15


# ---------------------------------------------------------------------------
# VectorCypherConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorCypherConfigPostInit:
    def test_post_init_validates_max_chunks_in_flight(self) -> None:
        with pytest.raises(ValueError, match="max_chunks_in_flight must be >= 1"):
            VectorCypherConfig(max_chunks_in_flight=0)

    def test_post_init_validates_negative(self) -> None:
        with pytest.raises(ValueError, match="max_chunks_in_flight"):
            VectorCypherConfig(max_chunks_in_flight=-5)

    def test_post_init_accepts_none(self) -> None:
        # None means "process all chunks at once" (backward-compat)
        cfg = VectorCypherConfig(max_chunks_in_flight=None)
        assert cfg.max_chunks_in_flight is None

    def test_post_init_accepts_positive(self) -> None:
        cfg = VectorCypherConfig(max_chunks_in_flight=10)
        assert cfg.max_chunks_in_flight == 10


# ---------------------------------------------------------------------------
# VectorCypherEngine._neo4j_driver_kwargs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNeo4jDriverKwargs:
    def test_none_config_uses_defaults(self) -> None:
        kwargs = VectorCypherEngine._neo4j_driver_kwargs(None)
        assert kwargs["max_connection_pool_size"] == 100
        assert kwargs["max_connection_lifetime"] == 900
        assert kwargs["liveness_check_timeout"] == 30.0
        assert kwargs["connection_acquisition_timeout"] == 60.0

    def test_custom_config_overrides(self) -> None:
        neo4j_cfg = MagicMock(
            spec=[
                "max_connection_pool_size",
                "max_connection_lifetime",
                "liveness_check_timeout",
                "connection_acquisition_timeout",
            ]
        )
        neo4j_cfg.max_connection_pool_size = 50
        neo4j_cfg.max_connection_lifetime = 600
        neo4j_cfg.liveness_check_timeout = 15.0
        neo4j_cfg.connection_acquisition_timeout = 30.0
        kwargs = VectorCypherEngine._neo4j_driver_kwargs(neo4j_cfg)
        assert kwargs["max_connection_pool_size"] == 50
        assert kwargs["max_connection_lifetime"] == 600
        assert kwargs["liveness_check_timeout"] == 15.0
        assert kwargs["connection_acquisition_timeout"] == 30.0


# ---------------------------------------------------------------------------
# _build_conversation_context (static method)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildConversationContext:
    def test_empty_list(self) -> None:
        assert VectorCypherEngine._build_conversation_context([]) == {}

    def test_single_message_no_context(self) -> None:
        docs = [{"content": "Hello", "metadata": {"author": "Alice"}}]
        out = VectorCypherEngine._build_conversation_context(docs)
        # Single doc: parts empty → context_map[0] == content unchanged
        assert out == {0: "Hello"}

    def test_neighbor_window(self) -> None:
        docs = [
            {"content": "first", "metadata": {"author": "Alice"}},
            {"content": "second", "metadata": {"author": "Bob"}},
            {"content": "third", "metadata": {"author": "Carol"}},
        ]
        out = VectorCypherEngine._build_conversation_context(docs)
        # Index 1 should see both prev (Alice) and next (Carol)
        ctx_1 = out[1]
        assert "Alice" in ctx_1
        assert "Carol" in ctx_1
        assert "second" in ctx_1
        # Prefixes
        assert "prev:" in ctx_1
        assert "next:" in ctx_1

    def test_truncates_content_to_100_chars(self) -> None:
        long = "x" * 200
        docs = [
            {"content": "anchor", "metadata": {"author": "A"}},
            {"content": long, "metadata": {"author": "B"}},
        ]
        out = VectorCypherEngine._build_conversation_context(docs)
        # The 200-char message should appear truncated in the context window
        # of its neighbor
        assert "x" * 200 not in out[0]  # truncated to 100 chars in context

    def test_missing_metadata_defaults(self) -> None:
        docs = [
            {"content": "first"},
            {"content": "second"},
        ]
        out = VectorCypherEngine._build_conversation_context(docs)
        # Author defaults to empty string; content still appears
        assert "first" in out[1]
        assert "second" in out[0]


# ---------------------------------------------------------------------------
# Fixtures for connected engine
# ---------------------------------------------------------------------------


def _make_config() -> MagicMock:
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.get_neo4j_url.return_value = "bolt://localhost:7687"
    config.get_neo4j_user.return_value = "neo4j"
    config.get_neo4j_password.return_value = "password"
    config.get_neo4j_database.return_value = "neo4j"
    config.get_graph_config.return_value = MagicMock()
    config.get_vector_config.return_value = MagicMock()
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.embedding_dimension = 1536
    config.llm.model = "gpt-4o-mini"
    config.llm.timeout = 30
    config.pipeline.extract_entities = True
    config.pipeline.chunking_strategy = "recursive"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 200
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    # Abstention knobs (#1331) — the engine reads these off config.query now.
    config.query.abstention_min_chunks = 1
    config.query.abstention_min_top_score = 0.3
    config.query.abstention_combined_threshold = 0.5
    config.query.abstention_weight_entities_empty = 0.3
    config.query.abstention_weight_chunks_below_min = 0.4
    config.query.abstention_weight_top_score_low = 0.3
    config.query.abstention_mode = "cosine_floor"
    config.query.abstention_confidence_target_cosine = 0.5
    config.query.abstention_confidence_target_gap = 0.1
    return config


def _make_connected_engine() -> VectorCypherEngine:
    engine = VectorCypherEngine(_make_config())
    engine._connected = True
    engine._storage = AsyncMock()
    engine._temporal_store = AsyncMock()
    engine._embedder = AsyncMock()
    engine._dual_nodes = AsyncMock()
    engine._retriever = AsyncMock()
    engine._router = MagicMock()
    engine._neo4j_driver = AsyncMock()
    return engine


# ---------------------------------------------------------------------------
# _detect_temporal_filter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectTemporalFilter:
    def test_no_temporal_keywords_returns_none(self) -> None:
        engine = _make_connected_engine()
        assert engine._detect_temporal_filter("What is python?") is None

    def test_temporal_keyword_without_date_returns_none(self) -> None:
        engine = _make_connected_engine()
        # Trigger temporal keyword but no parseable date → falls through to None
        assert engine._detect_temporal_filter("What happened recently?") is None

    def test_explicit_date_before_kw(self) -> None:
        engine = _make_connected_engine()
        tf = engine._detect_temporal_filter("What happened before 2024-01-15?")
        assert tf is not None
        assert tf.occurred_before is not None
        assert tf.occurred_after is None

    def test_explicit_date_after_kw(self) -> None:
        engine = _make_connected_engine()
        tf = engine._detect_temporal_filter("Events after 2024-01-15?")
        assert tf is not None
        assert tf.occurred_after is not None
        assert tf.occurred_before is None

    def test_explicit_date_since_kw_maps_to_after(self) -> None:
        engine = _make_connected_engine()
        tf = engine._detect_temporal_filter("Items since 2024-01-15?")
        assert tf is not None
        # "since" should map to occurred_after (same branch as "after")
        assert tf.occurred_after is not None

    def test_explicit_date_without_before_after_uses_window(self) -> None:
        engine = _make_connected_engine()
        # "when did X happen on 2024-01-15" → defaults to ±30 days window
        tf = engine._detect_temporal_filter("When did this happen on 2024-01-15?")
        assert tf is not None
        assert tf.occurred_after is not None
        assert tf.occurred_before is not None
        # ±30 days window
        delta = tf.occurred_before - tf.occurred_after
        assert delta == timedelta(days=60)


# ---------------------------------------------------------------------------
# _parse_datetime
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseDatetime:
    def test_datetime_passthrough_tz_aware(self) -> None:
        engine = _make_connected_engine()
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        assert engine._parse_datetime(dt) is dt

    def test_datetime_naive_gets_utc(self) -> None:
        engine = _make_connected_engine()
        dt = datetime(2024, 1, 1)
        out = engine._parse_datetime(dt)
        assert out.tzinfo == UTC

    def test_iso_string_with_z(self) -> None:
        engine = _make_connected_engine()
        out = engine._parse_datetime("2024-01-15T12:30:00Z")
        assert out.year == 2024
        assert out.tzinfo is not None

    def test_iso_string_date_only(self) -> None:
        engine = _make_connected_engine()
        out = engine._parse_datetime("2024-01-15")
        assert out.year == 2024
        assert out.month == 1
        assert out.tzinfo == UTC

    def test_longmemeval_format(self) -> None:
        engine = _make_connected_engine()
        # "%Y/%m/%d (%a) %H:%M" format
        out = engine._parse_datetime("2023/04/10 (Mon) 17:50")
        assert out.year == 2023
        assert out.tzinfo == UTC

    def test_b_d_y_format(self) -> None:
        engine = _make_connected_engine()
        out = engine._parse_datetime("January 15, 2024")
        assert out.year == 2024
        assert out.month == 1

    def test_unparseable_raises_value_error(self) -> None:
        engine = _make_connected_engine()
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            engine._parse_datetime("xxx not a date xxx zzz")


# ---------------------------------------------------------------------------
# Namespace / Entity / Document passthroughs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPassthroughOperations:
    @pytest.mark.asyncio
    async def test_create_namespace(self) -> None:
        engine = _make_connected_engine()
        engine._storage.create_namespace = AsyncMock(side_effect=lambda ns: ns)
        ns = await engine.create_namespace(config_overrides={"k": "v"})
        assert ns.config_overrides == {"k": "v"}
        engine._storage.create_namespace.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_namespace_no_overrides(self) -> None:
        engine = _make_connected_engine()
        engine._storage.create_namespace = AsyncMock(side_effect=lambda ns: ns)
        ns = await engine.create_namespace()
        assert ns.config_overrides == {}

    @pytest.mark.asyncio
    async def test_get_namespace(self) -> None:
        engine = _make_connected_engine()
        ns_id = uuid4()
        sentinel = MagicMock()
        engine._storage.get_namespace = AsyncMock(return_value=sentinel)
        result = await engine.get_namespace(ns_id)
        assert result is sentinel
        engine._storage.get_namespace.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_get_entity(self) -> None:
        engine = _make_connected_engine()
        eid = uuid4()
        ns = uuid4()
        sentinel = MagicMock()
        engine._storage.get_entity = AsyncMock(return_value=sentinel)
        result = await engine.get_entity(eid, namespace_id=ns)
        assert result is sentinel
        engine._storage.get_entity.assert_awaited_once_with(eid, namespace_id=ns)

    @pytest.mark.asyncio
    async def test_list_entities(self) -> None:
        engine = _make_connected_engine()
        ns = uuid4()
        engine._storage.list_entities = AsyncMock(return_value=[])
        out = await engine.list_entities(ns, entity_type="PERSON", limit=5)
        assert out == []
        engine._storage.list_entities.assert_awaited_once_with(ns, entity_type="PERSON", limit=5)

    @pytest.mark.asyncio
    async def test_get_document(self) -> None:
        engine = _make_connected_engine()
        doc_id = uuid4()
        ns_id = uuid4()
        engine._storage.get_document = AsyncMock(return_value=None)
        result = await engine.get_document(doc_id, namespace_id=ns_id)
        assert result is None
        engine._storage.get_document.assert_awaited_once_with(doc_id, namespace_id=ns_id)

    @pytest.mark.asyncio
    async def test_list_documents(self) -> None:
        engine = _make_connected_engine()
        ns = uuid4()
        engine._storage.list_documents = AsyncMock(return_value=[])
        out = await engine.list_documents(ns, limit=10)
        assert out == []
        engine._storage.list_documents.assert_awaited_once_with(ns, limit=10)


# ---------------------------------------------------------------------------
# find_related_entities
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindRelatedEntities:
    @pytest.mark.asyncio
    async def test_graph_only_path_no_graph_backend(self) -> None:
        """Graph-less storage (no .graph backend) returns []."""
        engine = _make_connected_engine()
        engine._dual_nodes = None  # Force graph-only path
        engine._storage.graph = None
        eid = uuid4()
        ns = uuid4()
        out = await engine.find_related_entities(eid, ns)
        assert out == []

    @pytest.mark.asyncio
    async def test_graph_only_path_with_neighborhood(self) -> None:
        """Graph-only path scores entities by BFS hop distance."""
        engine = _make_connected_engine()
        engine._dual_nodes = None
        seed_id = uuid4()
        b = Entity(id=uuid4(), name="B", entity_type="PERSON")
        c = Entity(id=uuid4(), name="C", entity_type="PERSON")
        rels = [
            Relationship(source_entity_id=seed_id, target_entity_id=b.id, relationship_type="KNOWS"),
            Relationship(source_entity_id=b.id, target_entity_id=c.id, relationship_type="KNOWS"),
        ]
        seed_entity = Entity(id=seed_id, name="Seed", entity_type="PERSON")
        engine._storage.graph = AsyncMock()
        engine._storage.graph.get_neighborhood = AsyncMock(
            return_value={"entities": [seed_entity, b, c], "relationships": rels}
        )

        out = await engine.find_related_entities(seed_id, uuid4(), max_depth=2, limit=10)
        # Seed entity itself filtered out; b at distance 1 → 0.5; c at distance 2 → 0.33
        assert len(out) == 2
        names = {e.name for e, _ in out}
        assert names == {"B", "C"}
        # Sorted by score desc → B comes first
        assert out[0][0].name == "B"
        assert out[0][1] == pytest.approx(0.5)
        assert out[1][1] == pytest.approx(1.0 / 3.0)

    @pytest.mark.asyncio
    async def test_dual_nodes_path(self) -> None:
        """Dual-node path queries DualNodeManager and resolves entities by ID."""
        engine = _make_connected_engine()
        seed_id = uuid4()
        ns_id = uuid4()
        # dual_nodes returns infos with id strings + distances
        b_id = uuid4()
        engine._dual_nodes.get_entity_neighborhoods = AsyncMock(
            return_value={str(seed_id): [{"id": str(b_id), "distance": 1}]}
        )
        b_entity = Entity(id=b_id, name="B", entity_type="PERSON")
        engine._storage.get_entity = AsyncMock(return_value=b_entity)

        out = await engine.find_related_entities(seed_id, ns_id, max_depth=2, limit=10)
        assert len(out) == 1
        assert out[0][0].name == "B"
        assert out[0][1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# search_entities
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSearchEntities:
    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        engine = _make_connected_engine()
        ns = uuid4()
        engine._embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        engine._storage.search_similar_entities = AsyncMock(return_value=[])
        out = await engine.search_entities("query", ns)
        assert out == []
        # When empty, no need to call get_entities_batch
        engine._storage.get_entities_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_populated_results_preserve_score_order(self) -> None:
        engine = _make_connected_engine()
        ns = uuid4()
        e1 = Entity(id=uuid4(), name="E1", entity_type="PERSON")
        e2 = Entity(id=uuid4(), name="E2", entity_type="PERSON")
        # search returns score-ordered ids
        engine._embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        engine._storage.search_similar_entities = AsyncMock(return_value=[(e1.id, 0.9), (e2.id, 0.7)])
        engine._storage.get_entities_batch = AsyncMock(return_value={e1.id: e1, e2.id: e2})

        out = await engine.search_entities("query", ns, limit=5)
        assert [e.name for e in out] == ["E1", "E2"]

    @pytest.mark.asyncio
    async def test_filters_missing_entities(self) -> None:
        """Entities not returned by get_entities_batch are silently dropped."""
        engine = _make_connected_engine()
        ns = uuid4()
        e1 = Entity(id=uuid4(), name="E1", entity_type="PERSON")
        missing_id = uuid4()
        engine._embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        engine._storage.search_similar_entities = AsyncMock(return_value=[(e1.id, 0.9), (missing_id, 0.5)])
        engine._storage.get_entities_batch = AsyncMock(return_value={e1.id: e1})

        out = await engine.search_entities("q", ns)
        assert [e.name for e in out] == ["E1"]


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStats:
    @pytest.mark.asyncio
    async def test_all_counts_succeed(self) -> None:
        engine = _make_connected_engine()
        ns = uuid4()
        engine._storage.get_document_stats = AsyncMock(return_value=(7, datetime(2024, 1, 1, tzinfo=UTC)))
        engine._storage.count_chunks = AsyncMock(return_value=20)
        engine._storage.count_entities = AsyncMock(return_value=15)
        engine._storage.count_relationships = AsyncMock(return_value=10)
        out = await engine.stats(ns)
        assert out.documents == 7
        assert out.chunks == 20
        assert out.entities == 15
        assert out.relationships == 10
        assert out.last_activity_at is not None

    @pytest.mark.asyncio
    async def test_document_stats_not_implemented_degrades(self) -> None:
        engine = _make_connected_engine()
        ns = uuid4()
        engine._storage.get_document_stats = AsyncMock(side_effect=NotImplementedError())
        engine._storage.count_chunks = AsyncMock(return_value=5)
        engine._storage.count_entities = AsyncMock(return_value=3)
        engine._storage.count_relationships = AsyncMock(return_value=2)
        out = await engine.stats(ns)
        # doc_count + last_activity stay at defaults (0, None)
        assert out.documents == 0
        assert out.last_activity_at is None
        assert out.chunks == 5

    @pytest.mark.asyncio
    async def test_count_chunks_raises_degrades_to_zero(self) -> None:
        """Failed counter degrades to 0 — never blocks the rest."""
        engine = _make_connected_engine()
        ns = uuid4()
        engine._storage.get_document_stats = AsyncMock(return_value=(1, None))
        engine._storage.count_chunks = AsyncMock(side_effect=RuntimeError("boom"))
        engine._storage.count_entities = AsyncMock(return_value=3)
        engine._storage.count_relationships = AsyncMock(return_value=2)
        out = await engine.stats(ns)
        assert out.chunks == 0
        assert out.entities == 3
        assert out.relationships == 2


# ---------------------------------------------------------------------------
# clear_document_extraction_state
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClearDocumentExtractionState:
    @pytest.mark.asyncio
    async def test_clears_both_backends(self) -> None:
        engine = _make_connected_engine()
        doc_id = uuid4()
        ns = uuid4()
        await engine.clear_document_extraction_state(doc_id, ns)
        engine._temporal_store.delete_chunks_by_document.assert_awaited_once_with(doc_id, ns)
        engine._dual_nodes.delete_chunks_by_document.assert_awaited_once_with(doc_id, ns)

    @pytest.mark.asyncio
    async def test_swallows_temporal_store_error(self) -> None:
        """Best-effort cleanup: temporal_store errors are logged, never raised."""
        engine = _make_connected_engine()
        engine._temporal_store.delete_chunks_by_document = AsyncMock(side_effect=RuntimeError("pgvector down"))
        # Should not raise
        await engine.clear_document_extraction_state(uuid4(), uuid4())
        # dual_nodes still attempted
        engine._dual_nodes.delete_chunks_by_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_swallows_dual_nodes_error(self) -> None:
        engine = _make_connected_engine()
        engine._dual_nodes.delete_chunks_by_document = AsyncMock(side_effect=RuntimeError("neo4j down"))
        # Should not raise
        await engine.clear_document_extraction_state(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_no_dual_nodes_only_clears_temporal(self) -> None:
        """When dual_nodes is None (SurrealDB), only clears temporal store."""
        engine = _make_connected_engine()
        engine._dual_nodes = None
        await engine.clear_document_extraction_state(uuid4(), uuid4())
        engine._temporal_store.delete_chunks_by_document.assert_awaited_once()


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthCheckBranches:
    @pytest.mark.asyncio
    async def test_all_healthy(self) -> None:
        engine = _make_connected_engine()
        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {"pg": "ok"}
        engine._storage.health_check = AsyncMock(return_value=storage_health)
        engine._temporal_store.health_check = AsyncMock(return_value={"status": "healthy"})
        engine._neo4j_driver.verify_connectivity = AsyncMock(return_value=None)
        out = await engine.health_check()
        assert out["status"] == "healthy"
        assert out["neo4j"] == "healthy"
        assert out["engine"] == "vectorcypher"

    @pytest.mark.asyncio
    async def test_neo4j_unhealthy_degrades(self) -> None:
        engine = _make_connected_engine()
        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {}
        engine._storage.health_check = AsyncMock(return_value=storage_health)
        engine._temporal_store.health_check = AsyncMock(return_value={"status": "healthy"})
        engine._neo4j_driver.verify_connectivity = AsyncMock(side_effect=RuntimeError("down"))
        out = await engine.health_check()
        assert out["status"] == "degraded"
        assert out["neo4j"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_no_neo4j_driver_unhealthy_label(self) -> None:
        """When self._neo4j_driver is None (SurrealDB), neo4j label is 'unhealthy'."""
        engine = _make_connected_engine()
        engine._neo4j_driver = None
        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {}
        engine._storage.health_check = AsyncMock(return_value=storage_health)
        engine._temporal_store.health_check = AsyncMock(return_value={"status": "healthy"})
        out = await engine.health_check()
        # neo4j_healthy stays False because driver is None
        assert out["status"] == "degraded"
        assert out["neo4j"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_temporal_store_degraded(self) -> None:
        engine = _make_connected_engine()
        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {}
        engine._storage.health_check = AsyncMock(return_value=storage_health)
        engine._temporal_store.health_check = AsyncMock(return_value={"status": "degraded"})
        engine._neo4j_driver.verify_connectivity = AsyncMock(return_value=None)
        out = await engine.health_check()
        assert out["status"] == "degraded"


# ---------------------------------------------------------------------------
# Disconnect re-entrancy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDisconnectReentrancy:
    @pytest.mark.asyncio
    async def test_disconnect_handles_double_call(self) -> None:
        """Calling disconnect twice is safe (second call is a no-op)."""
        engine = _make_connected_engine()
        from unittest.mock import patch

        with patch("khora.telemetry.shutdown_telemetry", new_callable=AsyncMock):
            await engine.disconnect()
            assert engine._connected is False
            # Second call should not re-enter (no error)
            await engine.disconnect()
            assert engine._connected is False


@pytest.mark.unit
class TestDisconnectOrdering:
    """disconnect() must stop the backend's pool task before closing the shared driver.

    The Neo4jBackend wrapped via ``from_driver`` runs a pool sampler against the
    shared driver's pool. Closing the driver first would tear the pool out from
    under a still-running sampler tick. ``disconnect()`` therefore calls
    ``self._storage.disconnect()`` (which stops the sampler) BEFORE
    ``self._neo4j_driver.close()``.
    """

    @pytest.mark.asyncio
    async def test_storage_disconnect_runs_before_driver_close(self) -> None:
        from unittest.mock import patch

        engine = _make_connected_engine()

        # Attach both call sites to one parent spy so mock_calls records their
        # relative order across the two distinct objects.
        order = MagicMock()
        order.storage_disconnect = AsyncMock()
        order.driver_close = AsyncMock()
        engine._storage.disconnect = order.storage_disconnect
        engine._neo4j_driver.close = order.driver_close

        with patch("khora.telemetry.shutdown_telemetry", new_callable=AsyncMock):
            await engine.disconnect()

        # Both must actually be awaited (not merely called) — guards against a
        # called-but-not-awaited regression that mock_calls alone would miss.
        order.storage_disconnect.assert_awaited_once()
        order.driver_close.assert_awaited_once()

        names = [c[0] for c in order.mock_calls]
        assert "storage_disconnect" in names
        assert "driver_close" in names
        assert names.index("storage_disconnect") < names.index("driver_close"), (
            f"storage.disconnect() must run before driver.close(); saw order {names}"
        )

    @pytest.mark.asyncio
    async def test_both_storage_and_driver_are_torn_down(self) -> None:
        from unittest.mock import patch

        engine = _make_connected_engine()
        storage = engine._storage
        driver = engine._neo4j_driver

        with patch("khora.telemetry.shutdown_telemetry", new_callable=AsyncMock):
            await engine.disconnect()

        storage.disconnect.assert_awaited_once()
        driver.close.assert_awaited_once()
        assert engine._storage is None
        assert engine._neo4j_driver is None


# ---------------------------------------------------------------------------
# Recall — entity/relationship section formatting + hybrid_alpha override
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecallContextFormatting:
    def _make_recall_engine(self) -> VectorCypherEngine:
        engine = _make_connected_engine()

        from khora.core.models import Chunk
        from khora.engines.vectorcypher.retriever import VectorCypherResult
        from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.8,
            reasoning="t",
        )
        c = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Long enough content for the validation pass to keep this chunk in results",
        )
        e = Entity(name="Alice", entity_type="PERSON", description="An engineer")
        r = Relationship(
            relationship_type="KNOWS",
            source_entity_name="Alice",
            target_entity_name="Bob",
            description="works with",
        )
        retriever_result = VectorCypherResult(
            chunks=[(c, 0.9)],
            entities=[(e, 0.9)],
            relationships=[(r, 0.7)],
            routing_decision=routing,
            metadata={},
        )
        engine._retriever.retrieve = AsyncMock(return_value=retriever_result)
        return engine

    @pytest.mark.asyncio
    async def test_recall_includes_entity_in_projection(self) -> None:
        engine = self._make_recall_engine()
        result = await engine.recall("q", uuid4())
        # Entities are surfaced as typed projections (no context_text rendering).
        assert any(e.name == "Alice" for e in result.entities)

    @pytest.mark.asyncio
    async def test_recall_includes_relationship_in_projection(self) -> None:
        engine = self._make_recall_engine()
        result = await engine.recall("q", uuid4())
        # Relationships are surfaced as typed projections.
        assert any(r.relationship_type == "KNOWS" for r in result.relationships)

    @pytest.mark.asyncio
    async def test_recall_hybrid_alpha_override_restored(self) -> None:
        """hybrid_alpha kwarg must NOT mutate the shared retriever config (#1116):
        it is threaded as an explicit hybrid_alpha_override instead."""
        engine = self._make_recall_engine()
        original = engine._retriever._config.hybrid_alpha = 0.7
        await engine.recall("q", uuid4(), hybrid_alpha=0.2)
        # The shared config is never written.
        assert engine._retriever._config.hybrid_alpha == original

    @pytest.mark.asyncio
    async def test_recall_search_mode_all_sets_alpha(self) -> None:
        """SearchMode.ALL threads hybrid_alpha_override=0.5 without mutating the
        shared retriever config (#1116)."""
        from khora.query import SearchMode

        engine = self._make_recall_engine()
        engine._retriever._config.hybrid_alpha = 0.9
        # The override is threaded into retrieve(); the shared config is untouched.
        await engine.recall("q", uuid4(), mode=SearchMode.ALL)
        assert engine._retriever._config.hybrid_alpha == 0.9

    @pytest.mark.asyncio
    async def test_recall_explicit_temporal_filter_synthesizes_signal(self) -> None:
        """When the caller passes temporal_filter, the engine synthesizes an
        EXPLICIT TemporalSignal (source='api')."""
        from khora.storage.temporal import TemporalFilter

        engine = self._make_recall_engine()
        tf = TemporalFilter(occurred_after=datetime(2024, 1, 1, tzinfo=UTC))
        await engine.recall("q", uuid4(), temporal_filter=tf)
        # The retriever was invoked with a TemporalSignal carrying source='api'
        call = engine._retriever.retrieve.call_args
        signal = call.kwargs["temporal_signal"]
        assert signal is not None
        assert signal.source == "api"
        assert signal.is_temporal is True

    @pytest.mark.asyncio
    async def test_recall_forwards_min_similarity_to_retriever(self) -> None:
        """#830 regression: ``recall(min_similarity=T)`` must be plumbed through to
        ``retriever.retrieve(min_similarity=T)``. Prior to v0.17.1 the engine
        declared the kwarg but did not forward it, so the floor was a no-op."""
        engine = self._make_recall_engine()
        await engine.recall("q", uuid4(), min_similarity=0.5)
        call = engine._retriever.retrieve.call_args
        assert call.kwargs["min_similarity"] == 0.5

    @pytest.mark.asyncio
    async def test_recall_single_chunk_score_signals(self) -> None:
        """Single-chunk result still computes mean_score, variance=0, gap=0."""
        engine = self._make_recall_engine()
        result = await engine.recall("q", uuid4())
        meta = result.engine_info
        assert meta["retrieval_mean_score"] >= 0.0
        # With only one chunk after validation, variance == 0 and gap == 0
        assert meta["retrieval_score_variance"] == 0.0
        assert meta["retrieval_top_score_gap"] == 0.0

    @pytest.mark.asyncio
    async def test_recall_empty_results_metadata(self) -> None:
        """Empty chunk list produces zero score signals."""
        from khora.engines.vectorcypher.retriever import VectorCypherResult
        from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

        engine = _make_connected_engine()
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.5,
            reasoning="t",
        )
        engine._retriever.retrieve = AsyncMock(
            return_value=VectorCypherResult(chunks=[], entities=[], routing_decision=routing, metadata={})
        )
        result = await engine.recall("nothing", uuid4())
        assert result.chunks == []
        assert result.engine_info["retrieval_mean_score"] == 0.0
        assert result.engine_info["retrieval_score_variance"] == 0.0
        assert result.engine_info["retrieval_top_score_gap"] == 0.0


# ---------------------------------------------------------------------------
# process_staged_document — passthrough delegation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessStagedDocument:
    @pytest.mark.asyncio
    async def test_passes_extraction_config_hash_when_changed(self) -> None:
        """The hash is mutated on the document when caller supplies a new value."""
        from khora.core.models import Document

        engine = _make_connected_engine()
        # Patch _process_document so we can inspect inputs without going
        # through the full chunk + embed pipeline.
        called: dict = {}

        async def fake_process(document, **kwargs):
            called["doc_hash"] = document.extraction_config_hash
            called["kwargs"] = kwargs
            return (0, 0, 0)

        engine._process_document = fake_process  # type: ignore[method-assign]

        doc = Document(
            namespace_id=uuid4(),
            content="x",
            title="t",
            checksum="x" * 64,
            source_type="api",
            extraction_config_hash="old-hash",
        )

        result = await engine.process_staged_document(
            doc,
            skill_name="general_entities",
            occurred_at=datetime.now(UTC),
            entity_types=[],
            relationship_types=[],
            extraction_config_hash="new-hash",
        )
        assert result == (0, 0, 0)
        # The hash on the doc was updated in place before delegating.
        assert called["doc_hash"] == "new-hash"

    @pytest.mark.asyncio
    async def test_no_hash_update_when_same(self) -> None:
        """No update when the hash matches the document's existing value."""
        from khora.core.models import Document

        engine = _make_connected_engine()
        called: dict = {}

        async def fake_process(document, **kwargs):
            called["doc_hash"] = document.extraction_config_hash
            return (0, 0, 0)

        engine._process_document = fake_process  # type: ignore[method-assign]

        doc = Document(
            namespace_id=uuid4(),
            content="x",
            title="t",
            checksum="x" * 64,
            source_type="api",
            extraction_config_hash="same",
        )

        await engine.process_staged_document(
            doc,
            skill_name="general_entities",
            occurred_at=datetime.now(UTC),
            entity_types=[],
            relationship_types=[],
            extraction_config_hash="same",
        )
        # No mutation when hashes already match.
        assert called["doc_hash"] == "same"


# ---------------------------------------------------------------------------
# _run_skeleton_extraction_deferred (returns triples; no storage writes)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunSkeletonExtractionDeferred:
    @pytest.mark.asyncio
    async def test_empty_chunks_short_circuit(self) -> None:
        engine = _make_connected_engine()
        ents, rels, links = await engine._run_skeleton_extraction_deferred(
            [], uuid4(), entity_types=[], relationship_types=[]
        )
        assert ents == [] and rels == [] and links == []

    @pytest.mark.asyncio
    async def test_all_chunks_below_token_threshold_skipped(self) -> None:
        """If every chunk has ≤ min_extraction_tokens tokens, extraction is skipped."""
        from khora.storage.temporal import TemporalChunk

        engine = _make_connected_engine()
        # Use a tiny token budget so the short chunks below trip the threshold
        engine._vc_config.min_extraction_tokens = 5
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="one two three",  # 3 tokens
                embedding=[0.1] * 4,
            )
        ]
        ents, rels, links = await engine._run_skeleton_extraction_deferred(
            chunks, uuid4(), entity_types=[], relationship_types=[]
        )
        assert (ents, rels, links) == ([], [], [])

    @pytest.mark.asyncio
    async def test_returns_entities_with_embeddings_and_links(self, monkeypatch) -> None:
        """When extraction yields entities, they get embeddings + chunk links."""
        from khora.storage.temporal import TemporalChunk

        engine = _make_connected_engine()
        engine._vc_config.min_extraction_tokens = 0  # Don't skip on token count

        chunk_id = uuid4()
        ns_id = uuid4()
        chunks = [
            TemporalChunk(
                id=chunk_id,
                namespace_id=ns_id,
                document_id=uuid4(),
                content="Alice met Bob in Berlin to discuss the new contract",
                embedding=[0.1] * 4,
            )
        ]
        e_alice = Entity(name="Alice", entity_type="PERSON", source_chunk_ids=[chunk_id])
        e_bob = Entity(name="Bob", entity_type="PERSON", source_chunk_ids=[chunk_id])

        async def fake_extract(*args, **kwargs):
            return [e_alice, e_bob], []

        monkeypatch.setattr("khora.pipelines.tasks.extract.extract_entities", fake_extract)

        engine._embedder.model_name = "mock-embed"
        engine._embedder.embed_batch = AsyncMock(return_value=[[0.0] * 4, [0.0] * 4])

        ents, rels, links = await engine._run_skeleton_extraction_deferred(
            chunks, ns_id, entity_types=["PERSON"], relationship_types=[]
        )
        assert {e.name for e in ents} == {"Alice", "Bob"}
        # Embeddings were attached
        assert all(e.embedding is not None for e in ents)
        assert all(e.embedding_model == "mock-embed" for e in ents)
        # Both entities share chunk_id → co-occurrence relationship added
        assert len(rels) == 1
        assert rels[0].relationship_type == "ASSOCIATED_WITH"
        # Two entity → chunk links (one per entity)
        assert len(links) == 2

    @pytest.mark.asyncio
    async def test_no_entities_extracted_returns_empty_triple(self, monkeypatch) -> None:
        from khora.storage.temporal import TemporalChunk

        engine = _make_connected_engine()
        engine._vc_config.min_extraction_tokens = 0
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="This is a sentence with plenty of words for the threshold filter",
                embedding=[0.1] * 4,
            )
        ]

        async def fake_extract(*args, **kwargs):
            return [], []

        monkeypatch.setattr("khora.pipelines.tasks.extract.extract_entities", fake_extract)

        ents, rels, links = await engine._run_skeleton_extraction_deferred(
            chunks, uuid4(), entity_types=[], relationship_types=[]
        )
        assert ents == [] and rels == [] and links == []


# ---------------------------------------------------------------------------
# _run_skeleton_extraction (writes to storage; mock the LLM + storage layer)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunSkeletonExtraction:
    @pytest.mark.asyncio
    async def test_empty_chunks_short_circuit(self) -> None:
        engine = _make_connected_engine()
        ents, rels = await engine._run_skeleton_extraction([], uuid4(), entity_types=[], relationship_types=[])
        assert (ents, rels) == (0, 0)

    @pytest.mark.asyncio
    async def test_few_chunks_skip_skeleton(self, monkeypatch) -> None:
        """≤2 chunks bypass the skeleton indexer (all chunks are core)."""
        from khora.storage.temporal import TemporalChunk

        engine = _make_connected_engine()
        ns_id = uuid4()
        chunk_id = uuid4()
        chunks = [
            TemporalChunk(
                id=chunk_id,
                namespace_id=ns_id,
                document_id=uuid4(),
                content="A chunk for skeleton extraction with a fair amount of words to extract",
                embedding=[0.1] * 4,
            )
        ]

        # No entities extracted → short-circuits before storage writes.
        async def fake_extract(*a, **k):
            return [], []

        monkeypatch.setattr("khora.pipelines.tasks.extract.extract_entities", fake_extract)

        ents_count, rels_count = await engine._run_skeleton_extraction(
            chunks, ns_id, entity_types=["PERSON"], relationship_types=[]
        )
        assert (ents_count, rels_count) == (0, 0)
        # Storage upsert never triggered when no entities returned
        engine._storage.upsert_entities_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_entities_extracted_writes_to_storage(self, monkeypatch) -> None:
        from khora.storage.temporal import TemporalChunk

        engine = _make_connected_engine()
        ns_id = uuid4()
        chunk_id = uuid4()
        chunks = [
            TemporalChunk(
                id=chunk_id,
                namespace_id=ns_id,
                document_id=uuid4(),
                content="some content",
                embedding=[0.1] * 4,
            )
        ]
        alice = Entity(name="Alice", entity_type="PERSON", source_chunk_ids=[chunk_id])

        async def fake_extract(*a, **k):
            return [alice], []

        monkeypatch.setattr("khora.pipelines.tasks.extract.extract_entities", fake_extract)
        engine._embedder.model_name = "m"
        engine._embedder.embed_batch = AsyncMock(return_value=[[0.0] * 4])
        engine._storage.upsert_entities_batch = AsyncMock()
        engine._storage.create_relationships_batch = AsyncMock(return_value=0)

        ents_count, rels_count = await engine._run_skeleton_extraction(
            chunks, ns_id, entity_types=["PERSON"], relationship_types=[]
        )
        assert ents_count == 1
        assert rels_count == 0
        # Upsert called with namespace + list of entities
        engine._storage.upsert_entities_batch.assert_awaited_once()
        # MENTIONED_IN link batch called via dual_nodes
        engine._dual_nodes.link_entities_to_chunks_batch.assert_awaited_once()


# ---------------------------------------------------------------------------
# remember_batch — early short-circuit on empty list
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberBatchEmpty:
    @pytest.mark.asyncio
    async def test_empty_documents_returns_zero_batch_result(self) -> None:
        engine = _make_connected_engine()
        result = await engine.remember_batch([], uuid4(), entity_types=["PERSON"], relationship_types=["KNOWS"])
        assert result.total == 0
        assert result.processed == 0
        assert result.chunks == 0
        # Underlying impl was never invoked
        engine._storage.get_documents_by_checksums.assert_not_called()


# ---------------------------------------------------------------------------
# _validate_recall_results — additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateRecallResultsExtra:
    def test_empty_input(self) -> None:
        engine = _make_connected_engine()
        assert engine._validate_recall_results([], "q") == []

    def test_whitespace_only_content_filtered(self) -> None:
        from khora.core.models import Chunk

        engine = _make_connected_engine()
        c1 = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="    \n\t  ")
        out = engine._validate_recall_results([(c1, 0.9)], "q")
        assert out == []

    def test_custom_min_content_length(self) -> None:
        from khora.core.models import Chunk

        engine = _make_connected_engine()
        c1 = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="x" * 30)
        # Custom min_content_length above the content length filters it out
        out = engine._validate_recall_results([(c1, 0.5)], "q", min_content_length=50)
        assert out == []


# ---------------------------------------------------------------------------
# _remember_batch_impl — early-exit paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberBatchImplShortCircuits:
    @pytest.mark.asyncio
    async def test_all_duplicates_returns_early(self) -> None:
        """When every doc's checksum already exists, no extraction is run."""
        import hashlib

        engine = _make_connected_engine()
        ns = uuid4()
        docs = [
            {"content": "duplicate content alpha"},
            {"content": "duplicate content beta"},
        ]
        checksums = [hashlib.sha256(d["content"].encode()).hexdigest() for d in docs]
        # Every checksum is already in DB
        engine._storage.get_documents_by_checksums = AsyncMock(
            return_value={cs: MagicMock(id=uuid4()) for cs in checksums}
        )

        result = await engine.remember_batch(docs, ns, entity_types=["PERSON"], relationship_types=["KNOWS"])
        assert result.total == 2
        assert result.skipped == 2
        assert result.processed == 0
        # No document was created (we short-circuited before stage 1)
        engine._storage.create_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_external_id_matches_routes_to_replace(self) -> None:
        """external_id docs that match existing rows are routed to remember()."""
        engine = _make_connected_engine()
        ns = uuid4()
        existing_doc = MagicMock(id=uuid4(), status="completed", chunk_count=2, entity_count=1, relationship_count=0)

        # Both docs have external_id and match
        engine._storage.get_documents_by_external_ids = AsyncMock(
            return_value={"ext-1": existing_doc, "ext-2": existing_doc}
        )
        # Inside remember(), get_document_by_external_id is queried again — return
        # the same existing doc to route to _remember_via_replace.
        engine._storage.get_document_by_external_id = AsyncMock(return_value=existing_doc)
        # Make checksum lookup empty so dedup doesn't double-trigger
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={})
        # When _remember_via_replace runs, it ends up calling lots of things —
        # short-circuit by making it raise so this stage marks it failed.
        engine._remember_via_replace = AsyncMock(side_effect=RuntimeError("stop"))  # type: ignore[method-assign]

        result = await engine.remember_batch(
            [
                {"content": "a", "external_id": "ext-1"},
                {"content": "b", "external_id": "ext-2"},
            ],
            ns,
            entity_types=[],
            relationship_types=[],
        )
        assert result.total == 2
        assert result.failed == 2

    @pytest.mark.asyncio
    async def test_legacy_path_when_streaming_disabled(self) -> None:
        """streaming_pipeline=False routes to _remember_batch_legacy."""
        engine = _make_connected_engine()
        engine._vc_config.streaming_pipeline = False
        engine._remember_batch_legacy = AsyncMock(
            return_value=MagicMock(total=0, processed=0, skipped=0, failed=0, chunks=0, entities=0, relationships=0)
        )  # type: ignore[method-assign]

        await engine.remember_batch([{"content": "x"}], uuid4(), entity_types=[], relationship_types=[])
        engine._remember_batch_legacy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_progress_callback_invoked_for_skipped(self) -> None:
        """on_progress fires for each document, including skipped ones."""
        import hashlib

        engine = _make_connected_engine()
        ns = uuid4()
        docs = [{"content": "c1"}, {"content": "c2"}]
        checksums = [hashlib.sha256(d["content"].encode()).hexdigest() for d in docs]
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={cs: MagicMock() for cs in checksums})
        seen: list[tuple[int, int]] = []

        def cb(done: int, total: int) -> None:
            seen.append((done, total))

        await engine.remember_batch(docs, ns, entity_types=[], relationship_types=[], on_progress=cb)
        # Both skipped → 2 progress reports
        assert len(seen) == 2
        assert seen[-1] == (2, 2)


# ---------------------------------------------------------------------------
# Connect path — SurrealDB skip-neo4j branch (lightweight smoke)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConnectSkipNeo4j:
    @pytest.mark.asyncio
    async def test_disconnect_idempotent_with_partial_state(self) -> None:
        """Disconnect leaves all members None and doesn't crash with partial state."""
        engine = VectorCypherEngine(_make_config())
        engine._connected = True
        engine._neo4j_driver = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._storage = AsyncMock()
        # Some components set, others None — disconnect must tolerate.
        engine._embedder = None
        engine._retriever = None
        engine._dual_nodes = None
        engine._router = None

        from unittest.mock import patch

        with patch("khora.telemetry.shutdown_telemetry", new_callable=AsyncMock):
            await engine.disconnect()

        assert engine._connected is False
        assert engine._neo4j_driver is None
        assert engine._temporal_store is None
        assert engine._storage is None


# ---------------------------------------------------------------------------
# _remember_batch_legacy — direct test of the per-doc fallback path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberBatchLegacy:
    @pytest.mark.asyncio
    async def test_legacy_all_duplicates(self) -> None:
        """Every checksum matches existing → all docs skipped."""
        import hashlib

        engine = _make_connected_engine()
        ns = uuid4()
        docs = [{"content": "a"}, {"content": "b"}]
        checksums = [hashlib.sha256(d["content"].encode()).hexdigest() for d in docs]
        engine._storage.get_documents_by_checksums = AsyncMock(
            return_value={cs: MagicMock(id=uuid4()) for cs in checksums}
        )

        result = await engine._remember_batch_legacy(docs, ns, entity_types=[], relationship_types=[])
        assert result.total == 2
        assert result.skipped == 2
        assert result.processed == 0

    @pytest.mark.asyncio
    async def test_legacy_in_flight_dedup(self) -> None:
        """If the same checksum appears twice in the batch, only one is processed."""
        engine = _make_connected_engine()
        ns = uuid4()
        # Two docs with IDENTICAL content → same checksum
        docs = [{"content": "same"}, {"content": "same"}]
        # No existing duplicates in DB
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={})
        # Make remember() return a non-duplicate result (success)
        from khora.khora import RememberResult

        engine.remember = AsyncMock(  # type: ignore[method-assign]
            return_value=RememberResult(
                document_id=uuid4(),
                namespace_id=ns,
                chunks_created=1,
                entities_extracted=0,
                relationships_created=0,
            )
        )

        result = await engine._remember_batch_legacy(docs, ns, entity_types=[], relationship_types=[])
        assert result.total == 2
        # First wins, second is skipped (in-flight dedup)
        # OR both processed and one short-circuits in checksums_in_flight check
        assert result.processed + result.skipped == 2
        # We expect at least one to be skipped via the in-flight check
        assert result.skipped >= 1

    @pytest.mark.asyncio
    async def test_legacy_remember_raises_marks_failed(self) -> None:
        """If remember() raises, the doc is marked failed (not propagated)."""
        engine = _make_connected_engine()
        ns = uuid4()
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={})
        engine.remember = AsyncMock(side_effect=RuntimeError("LLM broke"))  # type: ignore[method-assign]

        result = await engine._remember_batch_legacy([{"content": "x"}], ns, entity_types=[], relationship_types=[])
        assert result.total == 1
        assert result.failed == 1
        assert result.processed == 0

    @pytest.mark.asyncio
    async def test_legacy_duplicate_remember_response(self) -> None:
        """When remember() reports a duplicate, the legacy path increments skipped."""
        from khora.khora import RememberResult

        engine = _make_connected_engine()
        ns = uuid4()
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={})
        engine.remember = AsyncMock(  # type: ignore[method-assign]
            return_value=RememberResult(
                document_id=uuid4(),
                namespace_id=ns,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
                metadata={"duplicate": True},
            )
        )

        result = await engine._remember_batch_legacy([{"content": "x"}], ns, entity_types=[], relationship_types=[])
        assert result.skipped == 1
        assert result.processed == 0

    @pytest.mark.asyncio
    async def test_legacy_on_progress_invoked(self) -> None:
        """on_progress callback fires for both duplicate-skip and success paths."""
        import hashlib

        engine = _make_connected_engine()
        ns = uuid4()
        docs = [{"content": "dup"}]
        cs = hashlib.sha256(b"dup").hexdigest()
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={cs: MagicMock()})

        calls: list[tuple[int, int]] = []

        def cb(done: int, total: int) -> None:
            calls.append((done, total))

        await engine._remember_batch_legacy(docs, ns, entity_types=[], relationship_types=[], on_progress=cb)
        assert calls == [(1, 1)]

    @pytest.mark.asyncio
    async def test_legacy_parses_occurred_at_from_metadata(self) -> None:
        """When metadata has occurred_at, _parse_datetime is invoked and forwarded."""
        from khora.khora import RememberResult

        engine = _make_connected_engine()
        ns = uuid4()
        engine._storage.get_documents_by_checksums = AsyncMock(return_value={})
        engine.remember = AsyncMock(  # type: ignore[method-assign]
            return_value=RememberResult(
                document_id=uuid4(),
                namespace_id=ns,
                chunks_created=1,
                entities_extracted=0,
                relationships_created=0,
            )
        )

        await engine._remember_batch_legacy(
            [{"content": "z", "metadata": {"occurred_at": "2024-01-15"}}],
            ns,
            entity_types=[],
            relationship_types=[],
        )
        # remember() was called with the parsed occurred_at
        call_kwargs = engine.remember.call_args.kwargs
        assert call_kwargs["occurred_at"] is not None
        assert call_kwargs["occurred_at"].year == 2024


# ---------------------------------------------------------------------------
# _remember_batch_impl — streaming pipeline (full path) tests
# ---------------------------------------------------------------------------


def _make_streaming_engine() -> VectorCypherEngine:
    """Engine wired up for streaming pipeline tests with all storage mocks."""
    engine = _make_connected_engine()
    # Force a low chunk size + extract_entities disabled by default so the
    # pipeline runs fast without needing LLM mocks.
    engine._config.pipeline.extract_entities = False
    engine._config.pipeline.chunking_strategy = "fixed"
    engine._config.pipeline.chunk_size = 1000
    engine._config.pipeline.chunk_overlap = 0

    # Storage scaffolding for the streaming pipeline.
    engine._storage.get_documents_by_checksums = AsyncMock(return_value={})
    engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})

    async def _create_document(doc):
        # Simulate the DB returning the document with an id assigned.
        return doc

    engine._storage.create_document = AsyncMock(side_effect=_create_document)
    engine._storage.update_document = AsyncMock(return_value=None)

    # Temporal store returns the chunks with assigned ids.
    async def _create_chunks(chunks):
        for c in chunks:
            if c.id is None:
                c.id = uuid4()
        return list(chunks)

    engine._temporal_store.create_chunks_batch = AsyncMock(side_effect=_create_chunks)
    engine._dual_nodes.create_chunk_nodes_batch = AsyncMock(return_value=None)
    engine._dual_nodes.link_entities_to_chunks_batch = AsyncMock(return_value=None)

    # Embedder: return a list of small embeddings, one per text.
    async def _embed_batch(texts):
        return [[0.0] * 4 for _ in texts]

    engine._embedder.embed_batch = AsyncMock(side_effect=_embed_batch)
    engine._embedder.model_name = "mock-embed"
    return engine


@pytest.mark.unit
class TestRememberBatchImplStreaming:
    @pytest.mark.asyncio
    async def test_single_doc_no_extraction(self) -> None:
        """Streaming path with extract_entities=False stores chunks but no entities."""
        engine = _make_streaming_engine()
        ns = uuid4()
        result = await engine.remember_batch(
            [{"content": "short body for the chunker"}],
            ns,
            entity_types=[],
            relationship_types=[],
        )
        assert result.total == 1
        assert result.processed == 1
        assert result.failed == 0
        assert result.chunks >= 1
        # No extraction → no entities/relationships
        assert result.entities == 0
        assert result.relationships == 0
        engine._temporal_store.create_chunks_batch.assert_awaited()
        engine._dual_nodes.create_chunk_nodes_batch.assert_awaited()

    @pytest.mark.asyncio
    async def test_empty_content_doc_marked_completed(self) -> None:
        """Docs that yield zero chunks are marked completed (not failed)."""
        engine = _make_streaming_engine()
        result = await engine.remember_batch([{"content": ""}], uuid4(), entity_types=[], relationship_types=[])
        # Empty content → zero chunks → marked completed via state branch
        assert result.total == 1
        # processed reflects the "no chunks" path
        assert result.processed == 1
        assert result.chunks == 0

    @pytest.mark.asyncio
    async def test_with_extraction_writes_entities(self, monkeypatch) -> None:
        """Streaming pipeline calls extract_entities + embeds + persists when entities returned."""
        engine = _make_streaming_engine()
        engine._config.pipeline.extract_entities = True
        # Default min_extraction_tokens=50 skips short content; lower the gate.
        engine._vc_config.min_extraction_tokens = 0

        # Patch extract_entities to return one entity + zero relationships.
        # The pipeline imports it from khora.pipelines.tasks.extract — patch at
        # the original module so the local import inside the function resolves
        # to our mock.
        captured: list = []

        async def fake_extract(chunks, **kw):
            ent = Entity(
                name="Alice",
                entity_type="PERSON",
                source_chunk_ids=[c.id for c in chunks],
            )
            captured.append(chunks)
            return [ent], []

        monkeypatch.setattr("khora.pipelines.tasks.extract.extract_entities", fake_extract)
        engine._storage.upsert_entities_batch = AsyncMock(return_value=None)
        engine._storage.create_relationships_batch = AsyncMock(return_value=0)

        ns = uuid4()
        result = await engine.remember_batch(
            [{"content": "alice and bob were here in the room"}],
            ns,
            entity_types=[],
            relationship_types=[],
        )
        assert result.processed == 1
        assert result.entities == 1
        # upsert called once with the entity batch
        engine._storage.upsert_entities_batch.assert_awaited_once()
        # entity-chunk links registered via dual_nodes
        engine._dual_nodes.link_entities_to_chunks_batch.assert_awaited()

    @pytest.mark.asyncio
    async def test_chunker_error_marks_failed(self, monkeypatch) -> None:
        """If chunker raises for a doc, it is marked failed (not propagated)."""
        engine = _make_streaming_engine()

        # Replace create_chunker so chunk() raises
        bad_chunker = MagicMock()
        bad_chunker.chunk.side_effect = RuntimeError("chunker broke")

        def fake_create_chunker(*args, **kwargs):
            return bad_chunker

        monkeypatch.setattr("khora.extraction.chunkers.create_chunker", fake_create_chunker)

        result = await engine.remember_batch([{"content": "doc"}], uuid4(), entity_types=[], relationship_types=[])
        assert result.total == 1
        assert result.failed == 1
        assert result.processed == 0


# ---------------------------------------------------------------------------
# Use of UUID type imports — make ruff happy about unused
# ---------------------------------------------------------------------------


def test_uuid_import_used() -> None:
    """Module-level imports anchor."""
    assert isinstance(uuid4(), UUID)
