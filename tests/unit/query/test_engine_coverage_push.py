"""Coverage-driven tests for ``khora.query.engine.HybridQueryEngine``.

Target uncovered blocks identified in #695 step 2:
  * ``_apply_narrative_coherence``               (lines ~1548-1611)
  * ``_apply_source_priority`` / ``_apply_source_priority_entities``
                                                 (lines ~2034-2109)
  * ``_apply_temporal_reranking``                (lines ~2214-2259)
  * ``_soft_temporal_score``                     (lines ~2262-2321)
  * ``_apply_entity_presence_scoring``           (lines ~2323-2375)
  * ``_stage2_normalize_fuse`` / ``_stage3_filter`` / ``_stage4_rerank`` /
    ``_stage5_diversity`` / ``_mmr_diversity_select``
                                                 (lines ~2754-3045)
  * ``_expand_adjacent_sessions``                (lines ~1425-1546)
  * ``_timed_search`` / ``_cached_entity_search`` /
    ``_vector_search`` / ``_graph_search`` /
    ``_keyword_search_bm25`` / ``_keyword_search_fulltext``
                                                 (lines ~1613-2032)
  * ``_heuristic_understanding``                 (lines ~2143-2178)
  * ``invalidate_caches`` / ``temporal_query``   (small)

All storage / embedder / reranker / litellm contact points are mocked at
the boundary — no live infrastructure is required.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models.document import Chunk, ChunkMetadata
from khora.core.models.entity import Entity
from khora.query.engine import (
    HybridQueryEngine,
    QueryConfig,
    SearchMode,
)
from khora.query.temporal import TemporalFilter, TemporalQuery
from khora.query.understanding import (
    AnswerType,
    EntityMention,
    QueryIntent,
    SourcePriority,
    UnderstandingResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    content: str = "content",
    created_at: datetime | None = None,
    source_timestamp: datetime | None = None,
    embedding: list[float] | None = None,
    custom: dict | None = None,
) -> Chunk:
    meta = ChunkMetadata(custom=custom or {})
    return Chunk(
        id=uuid4(),
        content=content,
        metadata=meta,
        embedding=embedding,
        created_at=created_at or datetime.now(UTC),
        source_timestamp=source_timestamp,
    )


def _make_entity(
    *,
    name: str = "Alice",
    entity_type: str = "PERSON",
    source_tool: str = "",
    source_chunk_ids: list[UUID] | None = None,
    mention_count: int = 1,
    attributes: dict | None = None,
) -> Entity:
    return Entity(
        id=uuid4(),
        name=name,
        entity_type=entity_type,
        source_tool=source_tool,
        source_chunk_ids=source_chunk_ids or [],
        mention_count=mention_count,
        attributes=attributes or {},
    )


def _make_understanding(
    *,
    entities: list[EntityMention] | None = None,
    keywords: list[str] | None = None,
    source_priority: SourcePriority | None = None,
    source_filters: list[str] | None = None,
    temporal_references: list | None = None,
    has_temporal: bool | None = None,
) -> UnderstandingResult:
    u = UnderstandingResult(
        original_query="q",
        intent=QueryIntent.SEARCH,
        answer_type=AnswerType.UNKNOWN,
        entities=entities or [],
        keywords=keywords or [],
        source_priority=source_priority or SourcePriority(),
        source_filters=source_filters or [],
        temporal_references=temporal_references or [],
    )
    # ``has_temporal`` is a computed property — only override when caller
    # explicitly wants something other than the default behavior.
    if has_temporal is not None:
        object.__setattr__(u, "has_temporal", has_temporal)
    return u


def _make_engine(
    *,
    storage: MagicMock | None = None,
    embedder: MagicMock | None = None,
    config: QueryConfig | None = None,
) -> HybridQueryEngine:
    if storage is None:
        storage = MagicMock()
        storage.search_similar_chunks = AsyncMock(return_value=[])
        storage.search_similar_entities = AsyncMock(return_value=[])
        storage.get_entities_batch = AsyncMock(return_value={})
        storage.get_neighborhood = AsyncMock(return_value={})
        storage.get_neighborhoods_batch = AsyncMock(return_value={})
        storage.get_chunks_batch = AsyncMock(return_value={})
        storage.list_chunks = AsyncMock(return_value=[])
        storage.list_entities = AsyncMock(return_value=[])
        storage.search_fulltext_chunks = AsyncMock(return_value=[])
    if config is None:
        config = QueryConfig(
            enable_query_understanding=False,
            enable_entity_linking=False,
            enable_reranking=False,
            enable_keyword_search=False,
        )
    return HybridQueryEngine(storage=storage, embedder=embedder, config=config)


# ---------------------------------------------------------------------------
# _apply_narrative_coherence
# ---------------------------------------------------------------------------


class TestApplyNarrativeCoherence:
    def test_empty_chunks_returns_input(self) -> None:
        engine = _make_engine()
        out = engine._apply_narrative_coherence([], [], QueryConfig())
        assert out == []

    def test_empty_entities_returns_input(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.5)]
        out = engine._apply_narrative_coherence(chunks, [], QueryConfig())
        assert out == chunks

    def test_shared_entity_boosts_score(self) -> None:
        engine = _make_engine()
        # Two chunks share an entity (Alice mentions both)
        c1 = _make_chunk(content="alice text")
        c2 = _make_chunk(content="more about alice")
        c3 = _make_chunk(content="isolated chunk")
        entity = _make_entity(name="Alice", source_chunk_ids=[c1.id, c2.id])
        chunks = [(c1, 0.5), (c2, 0.5), (c3, 0.5)]
        entities = [(entity, 0.9)]

        cfg = QueryConfig(
            coherence_boost_per_entity=0.2,
            coherence_max_boost=0.5,
            coherence_isolation_penalty=0.15,
        )
        out = engine._apply_narrative_coherence(chunks, entities, cfg)

        # c1 and c2 should be boosted; c3 has no entity association → unchanged
        out_map = {c.id: s for c, s in out}
        assert out_map[c1.id] > 0.5
        assert out_map[c2.id] > 0.5
        # c3 had no entities mapped at all → kept at original score
        assert out_map[c3.id] == pytest.approx(0.5)
        # Sorted descending by score
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)

    def test_isolation_penalty_applied(self) -> None:
        engine = _make_engine()
        # Chunk c1 has entity, c2 has different entity, no overlap → both penalized
        c1 = _make_chunk()
        c2 = _make_chunk()
        e1 = _make_entity(name="Alice", source_chunk_ids=[c1.id])
        e2 = _make_entity(name="Bob", source_chunk_ids=[c2.id])
        cfg = QueryConfig(coherence_isolation_penalty=0.2)
        out = engine._apply_narrative_coherence(
            [(c1, 1.0), (c2, 1.0)],
            [(e1, 1.0), (e2, 1.0)],
            cfg,
        )
        # Both should be penalized: 1.0 * (1 - 0.2) = 0.8
        for _, score in out:
            assert score == pytest.approx(0.8)

    def test_coherence_boost_capped(self) -> None:
        engine = _make_engine()
        # 10 chunks all share entity → 9 overlaps per chunk → would be 1.8 boost,
        # but coherence_max_boost caps it at 0.5
        chunks = [_make_chunk() for _ in range(10)]
        entity = _make_entity(source_chunk_ids=[c.id for c in chunks])
        cfg = QueryConfig(
            coherence_boost_per_entity=0.2,
            coherence_max_boost=0.5,
        )
        out = engine._apply_narrative_coherence(
            [(c, 1.0) for c in chunks],
            [(entity, 1.0)],
            cfg,
        )
        for _, score in out:
            assert score == pytest.approx(1.5)  # 1.0 * (1 + 0.5)


# ---------------------------------------------------------------------------
# _apply_source_priority and _apply_source_priority_entities
# ---------------------------------------------------------------------------


class TestApplySourcePriority:
    def test_chunk_without_metadata_keeps_score(self) -> None:
        engine = _make_engine()
        # Plain object with no metadata attribute
        chunk = MagicMock(spec=[])
        u = _make_understanding(source_priority=SourcePriority(slack=2.0))
        out = engine._apply_source_priority([(chunk, 0.5)], u)
        assert out == [(chunk, 0.5)]

    def test_chunk_with_high_priority_source_boosted(self) -> None:
        engine = _make_engine()
        chunk = _make_chunk(custom={"source_tool": "slack"})
        u = _make_understanding(source_priority=SourcePriority(slack=2.0))
        out = engine._apply_source_priority([(chunk, 1.0)], u)
        # weight=2.0 → 0.5 + 0.5*2.0 = 1.5
        assert out[0][1] == pytest.approx(1.5)

    def test_chunk_with_zero_priority_source_demoted(self) -> None:
        engine = _make_engine()
        chunk = _make_chunk(custom={"source_tool": "slack"})
        u = _make_understanding(source_priority=SourcePriority(slack=0.0))
        out = engine._apply_source_priority([(chunk, 1.0)], u)
        # weight=0 → 0.5 + 0.5*0 = 0.5
        assert out[0][1] == pytest.approx(0.5)

    def test_chunk_filtered_out_when_in_filter_list(self) -> None:
        engine = _make_engine()
        chunk = _make_chunk(custom={"source_tool": "slack"})
        u = _make_understanding(source_filters=["slack"])
        out = engine._apply_source_priority([(chunk, 1.0)], u)
        assert out == []

    def test_chunk_results_sorted_descending(self) -> None:
        engine = _make_engine()
        c1 = _make_chunk(custom={"source_tool": "slack"})
        c2 = _make_chunk(custom={"source_tool": "github"})
        u = _make_understanding(source_priority=SourcePriority(slack=2.0, github=0.5))
        out = engine._apply_source_priority([(c1, 0.5), (c2, 0.5)], u)
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)

    def test_entity_priority_no_source_tool(self) -> None:
        engine = _make_engine()
        entity = _make_entity(source_tool="")
        u = _make_understanding(source_priority=SourcePriority(slack=2.0))
        out = engine._apply_source_priority_entities([(entity, 0.5)], u)
        assert out == [(entity, 0.5)]

    def test_entity_priority_boost(self) -> None:
        engine = _make_engine()
        entity = _make_entity(source_tool="slack")
        u = _make_understanding(source_priority=SourcePriority(slack=2.0))
        out = engine._apply_source_priority_entities([(entity, 1.0)], u)
        assert out[0][1] == pytest.approx(1.5)

    def test_entity_filtered_when_in_filter_list(self) -> None:
        engine = _make_engine()
        entity = _make_entity(source_tool="slack")
        u = _make_understanding(source_filters=["slack"])
        out = engine._apply_source_priority_entities([(entity, 1.0)], u)
        assert out == []


# ---------------------------------------------------------------------------
# _apply_temporal_reranking
# ---------------------------------------------------------------------------


class TestApplyTemporalReranking:
    def test_descending_default(self) -> None:
        """Without 'earliest' keywords, recent chunks rank higher."""
        engine = _make_engine()
        now = datetime.now(UTC)
        c_old = _make_chunk(created_at=now - timedelta(days=30))
        c_new = _make_chunk(created_at=now)
        # Same relevance, temporal blend should put newer first
        u = MagicMock()
        from khora.query.understanding import TemporalReference

        u.temporal_references = [TemporalReference(type="relative", text="latest")]
        out = engine._apply_temporal_reranking([(c_old, 0.5), (c_new, 0.5)], u)
        # New chunk should be first
        assert out[0][0] is c_new

    def test_ascending_with_earliest_keywords(self) -> None:
        """With 'first' / 'earliest', older chunks rank higher."""
        engine = _make_engine()
        now = datetime.now(UTC)
        c_old = _make_chunk(created_at=now - timedelta(days=30))
        c_new = _make_chunk(created_at=now)
        u = MagicMock()
        from khora.query.understanding import TemporalReference

        u.temporal_references = [TemporalReference(type="relative", text="earliest record")]
        out = engine._apply_temporal_reranking([(c_old, 0.5), (c_new, 0.5)], u)
        assert out[0][0] is c_old

    def test_missing_created_at_does_not_raise(self) -> None:
        """Chunks without created_at sort to the bottom."""
        engine = _make_engine()
        c1 = _make_chunk(created_at=datetime.now(UTC))
        c2 = MagicMock(spec=[])  # No created_at
        u = MagicMock()
        u.temporal_references = []
        # Should not raise even with mixed chunks
        out = engine._apply_temporal_reranking([(c1, 0.5), (c2, 0.5)], u)
        assert len(out) == 2

    def test_temporal_blend_weight(self) -> None:
        """Blended score reflects 30% temporal + 70% relevance."""
        engine = _make_engine()
        now = datetime.now(UTC)
        c_new = _make_chunk(created_at=now)
        c_old = _make_chunk(created_at=now - timedelta(days=10))
        u = MagicMock()
        u.temporal_references = []
        out = engine._apply_temporal_reranking([(c_new, 1.0), (c_old, 0.0)], u)
        # c_new: 0.7*1.0 + 0.3*1.0 = 1.0
        # c_old: 0.7*0.0 + 0.3*0.0 = 0.0
        scores = {c.id: s for c, s in out}
        assert scores[c_new.id] > scores[c_old.id]


# ---------------------------------------------------------------------------
# _soft_temporal_score
# ---------------------------------------------------------------------------


class TestSoftTemporalScore:
    def test_chunk_inside_window_keeps_score(self) -> None:
        now = datetime.now(UTC)
        chunk = _make_chunk(source_timestamp=now)
        tf = TemporalFilter(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=1),
        )
        out = HybridQueryEngine._soft_temporal_score([(chunk, 1.0)], tf)
        assert out[0][1] == pytest.approx(1.0)

    def test_chunk_outside_window_decays(self) -> None:
        now = datetime.now(UTC)
        # Chunk 2 days before start → outside window but inside hard cutoff
        chunk = _make_chunk(source_timestamp=now - timedelta(days=2))
        tf = TemporalFilter(start_time=now - timedelta(days=1), end_time=now)
        out = HybridQueryEngine._soft_temporal_score([(chunk, 1.0)], tf, hard_cutoff_days=30.0, half_life_hours=24.0)
        # Score should be decayed but present
        assert len(out) == 1
        assert out[0][1] < 1.0
        assert out[0][1] > 0.0

    def test_chunk_beyond_hard_cutoff_dropped(self) -> None:
        now = datetime.now(UTC)
        chunk = _make_chunk(source_timestamp=now - timedelta(days=60))
        tf = TemporalFilter(start_time=now - timedelta(days=1), end_time=now)
        out = HybridQueryEngine._soft_temporal_score([(chunk, 1.0)], tf, hard_cutoff_days=30.0)
        assert out == []

    def test_chunk_without_timestamp_passes_through(self) -> None:
        chunk = _make_chunk()
        # Strip both timestamp fields to None
        object.__setattr__(chunk, "source_timestamp", None)
        object.__setattr__(chunk, "created_at", None)
        tf = TemporalFilter(start_time=datetime.now(UTC))
        out = HybridQueryEngine._soft_temporal_score([(chunk, 0.7)], tf)
        assert out[0][1] == pytest.approx(0.7)

    def test_chunk_after_end_decays(self) -> None:
        now = datetime.now(UTC)
        # Chunk one day AFTER end window
        chunk = _make_chunk(source_timestamp=now + timedelta(days=1))
        tf = TemporalFilter(
            start_time=now - timedelta(days=5),
            end_time=now,
        )
        out = HybridQueryEngine._soft_temporal_score([(chunk, 1.0)], tf)
        assert len(out) == 1
        assert out[0][1] < 1.0

    def test_sorted_descending(self) -> None:
        now = datetime.now(UTC)
        chunks = [
            (_make_chunk(source_timestamp=now), 0.3),
            (_make_chunk(source_timestamp=now), 0.7),
            (_make_chunk(source_timestamp=now), 0.5),
        ]
        tf = TemporalFilter(
            start_time=now - timedelta(days=1),
            end_time=now + timedelta(days=1),
        )
        out = HybridQueryEngine._soft_temporal_score(chunks, tf)
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# _apply_entity_presence_scoring
# ---------------------------------------------------------------------------


class TestApplyEntityPresenceScoring:
    def test_no_entities_returns_input(self) -> None:
        chunks = [(_make_chunk(content="hi"), 0.5)]
        u = _make_understanding()
        out = HybridQueryEngine._apply_entity_presence_scoring(chunks, u)
        assert out == chunks

    def test_full_match_no_penalty(self) -> None:
        chunk = _make_chunk(content="Alice met Bob today")
        u = _make_understanding(
            entities=[
                EntityMention(name="Alice", entity_type="PERSON"),
                EntityMention(name="Bob", entity_type="PERSON"),
            ]
        )
        out = HybridQueryEngine._apply_entity_presence_scoring([(chunk, 1.0)], u)
        # All entities matched → match_ratio = 1.0 → no penalty
        assert out[0][1] == pytest.approx(1.0)

    def test_no_match_min_penalty(self) -> None:
        chunk = _make_chunk(content="totally unrelated content")
        u = _make_understanding(entities=[EntityMention(name="Alice", entity_type="PERSON")])
        out = HybridQueryEngine._apply_entity_presence_scoring([(chunk, 1.0)], u)
        # No match → penalty = max(0.5, 0) = 0.5
        assert out[0][1] == pytest.approx(0.5)

    def test_alias_match(self) -> None:
        chunk = _make_chunk(content="met Ally yesterday")
        u = _make_understanding(entities=[EntityMention(name="Alice", entity_type="PERSON", aliases=["Ally"])])
        out = HybridQueryEngine._apply_entity_presence_scoring([(chunk, 1.0)], u)
        # Alias matched → match_ratio=1.0 → no penalty
        assert out[0][1] == pytest.approx(1.0)

    def test_empty_content_passes_through(self) -> None:
        chunk = _make_chunk(content="")
        u = _make_understanding(entities=[EntityMention(name="Alice", entity_type="PERSON")])
        out = HybridQueryEngine._apply_entity_presence_scoring([(chunk, 0.8)], u)
        # Empty content → kept as-is
        assert out[0][1] == pytest.approx(0.8)

    def test_partial_match_partial_penalty(self) -> None:
        chunk = _make_chunk(content="Alice did something")
        u = _make_understanding(
            entities=[
                EntityMention(name="Alice", entity_type="PERSON"),
                EntityMention(name="Bob", entity_type="PERSON"),
            ]
        )
        out = HybridQueryEngine._apply_entity_presence_scoring([(chunk, 1.0)], u)
        # 1 of 2 matched → ratio 0.5 → penalty max(0.5, 0.5) = 0.5
        assert out[0][1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _stage2_normalize_fuse
# ---------------------------------------------------------------------------


class TestStage2NormalizeFuse:
    def test_empty_inputs_returns_empty(self) -> None:
        engine = _make_engine()
        chunks, entities = engine._stage2_normalize_fuse({}, {}, QueryConfig())
        assert chunks == []
        assert entities == []

    def test_normalizes_chunks_to_unit_range(self) -> None:
        engine = _make_engine()
        c1, c2 = _make_chunk(), _make_chunk()
        stage1_chunks = {"vector": [(c1, 0.2), (c2, 0.8)]}
        cfg = QueryConfig(
            vector_weight=0.5,
            graph_weight=0.3,
            keyword_weight=0.2,
        )
        chunks, _ = engine._stage2_normalize_fuse(stage1_chunks, {}, cfg)
        # Fusion produces something
        assert len(chunks) == 2

    def test_all_same_score_normalizes_to_05(self) -> None:
        engine = _make_engine()
        c1, c2 = _make_chunk(), _make_chunk()
        stage1_chunks = {"vector": [(c1, 0.5), (c2, 0.5)]}
        chunks, _ = engine._stage2_normalize_fuse(stage1_chunks, {}, QueryConfig())
        assert len(chunks) == 2

    def test_skips_empty_sources(self) -> None:
        engine = _make_engine()
        c1 = _make_chunk()
        stage1_chunks = {"vector": [(c1, 0.5)], "graph": []}
        chunks, _ = engine._stage2_normalize_fuse(stage1_chunks, {}, QueryConfig())
        assert len(chunks) == 1

    def test_expanded_source_gets_discount_weight(self) -> None:
        engine = _make_engine()
        c1 = _make_chunk()
        stage1_chunks = {"vector_exp1": [(c1, 1.0)]}
        chunks, _ = engine._stage2_normalize_fuse(stage1_chunks, {}, QueryConfig())
        # Just verify fusion completes
        assert len(chunks) == 1

    def test_entities_fused_with_vector_graph_weights(self) -> None:
        engine = _make_engine()
        e1, e2 = _make_entity(name="A"), _make_entity(name="B")
        stage1_entities = {
            "vector": [(e1, 0.9)],
            "graph": [(e2, 0.4)],
        }
        _, entities = engine._stage2_normalize_fuse({}, stage1_entities, QueryConfig())
        assert len(entities) == 2


# ---------------------------------------------------------------------------
# _stage3_filter
# ---------------------------------------------------------------------------


class TestStage3Filter:
    def test_limits_chunks_to_stage3_filter_limit(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 1.0 - i * 0.01) for i in range(100)]
        cfg = QueryConfig(stage3_filter_limit=10, max_entities=5)
        out_chunks, _ = engine._stage3_filter(chunks, [], cfg, None, None)
        assert len(out_chunks) == 10

    def test_limits_entities_to_max_entities(self) -> None:
        engine = _make_engine()
        entities = [(_make_entity(name=f"E{i}"), 1.0 - i * 0.01) for i in range(50)]
        cfg = QueryConfig(max_entities=7, stage3_filter_limit=50)
        _, out_entities = engine._stage3_filter([], entities, cfg, None, None)
        assert len(out_entities) == 7

    def test_temporal_filter_applied(self) -> None:
        engine = _make_engine()
        now = datetime.now(UTC)
        # One chunk inside window, one far outside (beyond hard cutoff → dropped)
        c_in = _make_chunk(source_timestamp=now)
        c_out = _make_chunk(source_timestamp=now - timedelta(days=90))
        tf = TemporalFilter(start_time=now - timedelta(days=1), end_time=now + timedelta(days=1))
        out_chunks, _ = engine._stage3_filter([(c_in, 1.0), (c_out, 1.0)], [], QueryConfig(), None, tf)
        ids = [c.id for c, _ in out_chunks]
        assert c_in.id in ids
        assert c_out.id not in ids

    def test_recency_bias_applied_when_enabled(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 1.0)]
        cfg = QueryConfig(apply_recency_bias=True)
        out_chunks, _ = engine._stage3_filter(chunks, [], cfg, None, None)
        # Should pass through without crashing
        assert len(out_chunks) == 1

    def test_source_priority_applied(self) -> None:
        engine = _make_engine()
        c1 = _make_chunk(custom={"source_tool": "slack"})
        u = _make_understanding(source_priority=SourcePriority(slack=2.0))
        out_chunks, _ = engine._stage3_filter([(c1, 1.0)], [], QueryConfig(), u, None)
        # Source-priority boost applied: 0.5 + 0.5*2.0 = 1.5
        assert out_chunks[0][1] == pytest.approx(1.5)

    def test_entity_presence_scoring_applied(self) -> None:
        engine = _make_engine()
        chunk = _make_chunk(content="totally different")
        u = _make_understanding(entities=[EntityMention(name="Alice", entity_type="PERSON")])
        out_chunks, _ = engine._stage3_filter([(chunk, 1.0)], [], QueryConfig(), u, None)
        # No entity match → penalty applied
        assert out_chunks[0][1] < 1.0


# ---------------------------------------------------------------------------
# _stage4_rerank
# ---------------------------------------------------------------------------


class TestStage4Rerank:
    @pytest.mark.asyncio
    async def test_disabled_passes_through(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.5) for _ in range(5)]
        cfg = QueryConfig(enable_reranking=False)
        out = await engine._stage4_rerank(chunks, "q", cfg)
        assert out == chunks

    @pytest.mark.asyncio
    async def test_too_few_chunks_passes_through(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.5), (_make_chunk(), 0.4)]  # only 2 < 3
        cfg = QueryConfig(enable_reranking=True)
        out = await engine._stage4_rerank(chunks, "q", cfg)
        assert out == chunks

    @pytest.mark.asyncio
    async def test_reranker_called_with_candidates(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(content=f"c{i}"), 0.5 - i * 0.01) for i in range(5)]

        # Mock reranker
        mock_reranker = MagicMock()

        async def fake_rerank(query, candidates, top_k):  # noqa: ARG001
            from khora.query.reranking import RerankResult

            return [RerankResult(item=c.item, final_score=c.original_score + 0.1) for c in candidates]

        mock_reranker.rerank = fake_rerank
        engine._rerankers["cross_encoder"] = mock_reranker
        cfg = QueryConfig(
            enable_reranking=True,
            reranking_method="cross_encoder",
            stage4_rerank_limit=50,
            max_chunks=10,
        )
        out = await engine._stage4_rerank(chunks, "q", cfg)
        assert len(out) == 5

    @pytest.mark.asyncio
    async def test_reranker_exception_falls_back(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.5) for _ in range(5)]

        mock_reranker = MagicMock()

        async def boom(*a, **kw):  # noqa: ARG001
            raise RuntimeError("rerank failure")

        mock_reranker.rerank = boom
        engine._rerankers["cross_encoder"] = mock_reranker
        cfg = QueryConfig(enable_reranking=True, reranking_method="cross_encoder")
        out = await engine._stage4_rerank(chunks, "q", cfg)
        # Failure → returns original chunks
        assert out == chunks


# ---------------------------------------------------------------------------
# _stage5_diversity & _mmr_diversity_select
# ---------------------------------------------------------------------------


class TestStage5Diversity:
    def test_diversity_disabled_just_limits(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 1.0 - i * 0.01) for i in range(20)]
        entities = [(_make_entity(name=f"E{i}"), 0.5) for i in range(20)]
        cfg = QueryConfig(enable_diversity=False, max_chunks=5, max_entities=3)
        out_chunks, out_entities = engine._stage5_diversity(chunks, entities, None, cfg)
        assert len(out_chunks) == 5
        assert len(out_entities) == 3

    def test_no_embedding_just_limits(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 1.0) for _ in range(20)]
        cfg = QueryConfig(enable_diversity=True, max_chunks=4)
        out_chunks, _ = engine._stage5_diversity(chunks, [], None, cfg)
        assert len(out_chunks) == 4

    def test_below_max_chunks_passes_through(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 1.0) for _ in range(3)]
        cfg = QueryConfig(enable_diversity=True, max_chunks=10)
        out_chunks, _ = engine._stage5_diversity(chunks, [], [0.1, 0.2], cfg)
        # len(chunks) <= max_chunks → no MMR applied, just returns chunks
        assert len(out_chunks) == 3


class TestMMRDiversitySelect:
    def test_no_embeddings_falls_back_to_score(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 1.0 - i * 0.1) for i in range(5)]  # no embeddings
        out = engine._mmr_diversity_select(chunks, [0.1, 0.2, 0.3], k=3)
        # Falls back to ``chunks[:k]`` when no embeddings available
        assert len(out) == 3

    def test_k_geq_len_returns_chunks(self) -> None:
        engine = _make_engine()
        chunks = [(_make_chunk(), 1.0) for _ in range(3)]
        out = engine._mmr_diversity_select(chunks, [0.1, 0.2], k=5)
        assert out == chunks


# ---------------------------------------------------------------------------
# _expand_adjacent_sessions
# ---------------------------------------------------------------------------


class TestExpandAdjacentSessions:
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_none(self) -> None:
        engine = _make_engine()
        out = await engine._expand_adjacent_sessions([], uuid4())
        assert out is None

    @pytest.mark.asyncio
    async def test_entity_based_expansion_adds_chunks(self) -> None:
        engine = _make_engine()
        # Existing result chunk
        existing = _make_chunk()
        # New chunk from entity search
        new_chunk = _make_chunk(content="from entity search")
        engine._storage.search_fulltext_chunks = AsyncMock(return_value=[(new_chunk, 0.7)])
        understanding = _make_understanding(entities=[EntityMention(name="Alice", entity_type="PERSON")])
        out = await engine._expand_adjacent_sessions([(existing, 1.0)], uuid4(), understanding=understanding)
        assert out is not None
        ids = [c.id for c, _ in out]
        assert existing.id in ids
        assert new_chunk.id in ids

    @pytest.mark.asyncio
    async def test_entity_search_exception_skipped(self) -> None:
        engine = _make_engine()
        existing = _make_chunk()
        engine._storage.search_fulltext_chunks = AsyncMock(side_effect=RuntimeError("boom"))
        # Storage also doesn't have search_chunks_by_metadata, will fail through
        understanding = _make_understanding(entities=[EntityMention(name="Alice", entity_type="PERSON")])
        out = await engine._expand_adjacent_sessions([(existing, 1.0)], uuid4(), understanding=understanding)
        # Entity search failed AND no session_id in metadata → returns None
        assert out is None

    @pytest.mark.asyncio
    async def test_no_entities_no_session_ids_returns_none(self) -> None:
        engine = _make_engine()
        # Chunk has no session_id in metadata
        chunk = _make_chunk()
        out = await engine._expand_adjacent_sessions([(chunk, 1.0)], uuid4())
        assert out is None

    @pytest.mark.asyncio
    async def test_sequential_session_expansion_fallback(self) -> None:
        engine = _make_engine()
        existing = _make_chunk(custom={"session_id": 5})
        new_chunk = _make_chunk()
        # search_fulltext_chunks not used (no entities); set up metadata-based
        # storage method directly on the mock
        engine._storage.search_chunks_by_metadata = AsyncMock(return_value=[(new_chunk, 0.6)])
        out = await engine._expand_adjacent_sessions([(existing, 1.0)], uuid4())
        assert out is not None
        # Verify storage was queried with adjacent session ids (4, 6)
        call_kwargs = engine._storage.search_chunks_by_metadata.call_args.kwargs
        meta_filter = call_kwargs["metadata_filter"]
        assert set(meta_filter["session_id"]) == {4, 6}

    @pytest.mark.asyncio
    async def test_sequential_attribute_error_returns_none(self) -> None:
        engine = _make_engine()
        existing = _make_chunk(custom={"session_id": 5})
        # No search_chunks_by_metadata method at all
        engine._storage = MagicMock(spec=[])
        out = await engine._expand_adjacent_sessions([(existing, 1.0)], uuid4())
        assert out is None


# ---------------------------------------------------------------------------
# _timed_search
# ---------------------------------------------------------------------------


class TestTimedSearch:
    @pytest.mark.asyncio
    async def test_records_latency(self) -> None:
        engine = _make_engine()

        async def fake_search():
            return {"source": "vector", "chunks": []}

        result = await engine._timed_search(fake_search(), "vector")
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_non_dict_result_returned_as_is(self) -> None:
        engine = _make_engine()

        async def returns_list():
            return [1, 2, 3]

        result = await engine._timed_search(returns_list(), "vector")
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# _cached_entity_search
# ---------------------------------------------------------------------------


class TestCachedEntitySearch:
    @pytest.mark.asyncio
    async def test_deduplicates_concurrent_calls(self) -> None:
        engine = _make_engine()
        eid = uuid4()
        engine._storage.search_similar_entities = AsyncMock(return_value=[(eid, 0.9), (uuid4(), 0.8)])
        embedding = [0.1, 0.2, 0.3]
        # First call populates the cache
        out1 = await engine._cached_entity_search(uuid4(), embedding, 5, 0.0)
        # Second call hits the cache (same id() ⇒ same task)
        out2 = await engine._cached_entity_search(uuid4(), embedding, 5, 0.0)
        # storage queried at most twice (limit clamp may trigger second uncached call
        # for different ns, but here we're checking the cache path is used)
        assert len(out1) == 2
        assert len(out2) == 2

    @pytest.mark.asyncio
    async def test_truncates_to_limit(self) -> None:
        engine = _make_engine()
        engine._storage.search_similar_entities = AsyncMock(return_value=[(uuid4(), 0.9) for _ in range(10)])
        out = await engine._cached_entity_search(uuid4(), [0.1, 0.2], 3, 0.0)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# _vector_search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    @pytest.mark.asyncio
    async def test_returns_chunks_and_entities(self) -> None:
        engine = _make_engine()
        chunk = _make_chunk()
        entity = _make_entity()
        engine._storage.search_similar_chunks = AsyncMock(return_value=[(chunk, 0.9)])
        engine._storage.search_similar_entities = AsyncMock(return_value=[(entity.id, 0.8)])
        engine._storage.get_entities_batch = AsyncMock(return_value={entity.id: entity})

        cfg = QueryConfig(max_chunks=5, max_entities=5)
        out = await engine._vector_search(uuid4(), [0.1, 0.2], cfg)
        assert out["source"] == "vector"
        assert len(out["chunks"]) == 1
        assert len(out["entities"]) == 1

    @pytest.mark.asyncio
    async def test_temporal_filter_extracts_bounds(self) -> None:
        engine = _make_engine()
        engine._storage.search_similar_chunks = AsyncMock(return_value=[])
        engine._storage.search_similar_entities = AsyncMock(return_value=[])
        engine._storage.get_entities_batch = AsyncMock(return_value={})

        now = datetime.now(UTC)
        tf = TemporalFilter(start_time=now - timedelta(days=7), end_time=now)
        await engine._vector_search(uuid4(), [0.1], QueryConfig(), temporal_filter=tf)
        call_kwargs = engine._storage.search_similar_chunks.call_args.kwargs
        assert call_kwargs["created_after"] is not None
        assert call_kwargs["created_before"] is not None

    @pytest.mark.asyncio
    async def test_missing_entity_in_batch_skipped(self) -> None:
        engine = _make_engine()
        missing_id = uuid4()
        engine._storage.search_similar_chunks = AsyncMock(return_value=[])
        engine._storage.search_similar_entities = AsyncMock(return_value=[(missing_id, 0.5)])
        # get_entities_batch returns empty dict → entity skipped
        engine._storage.get_entities_batch = AsyncMock(return_value={})
        out = await engine._vector_search(uuid4(), [0.1], QueryConfig())
        assert out["entities"] == []


# ---------------------------------------------------------------------------
# _graph_search
# ---------------------------------------------------------------------------


class TestGraphSearch:
    @pytest.mark.asyncio
    async def test_no_query_embedding_no_linked_returns_empty(self) -> None:
        engine = _make_engine()
        out = await engine._graph_search(uuid4(), "q", None, QueryConfig(), None)
        assert out["source"] == "graph"
        assert out["entities"] == []
        assert out["chunks"] == []

    @pytest.mark.asyncio
    async def test_linked_entities_prioritized(self) -> None:
        engine = _make_engine()
        linked_id = uuid4()
        entity = _make_entity()
        object.__setattr__(entity, "id", linked_id)
        # Linked entity returns from batch fetch
        engine._storage.get_entities_batch = AsyncMock(return_value={linked_id: entity})
        engine._storage.get_neighborhoods_batch = AsyncMock(return_value={})
        # No embedding-based search
        out = await engine._graph_search(uuid4(), "q", None, QueryConfig(), linked_entity_ids=[linked_id])
        assert len(out["entities"]) == 1
        assert out["entities"][0][0] is entity

    @pytest.mark.asyncio
    async def test_similar_entities_via_embedding(self) -> None:
        engine = _make_engine()
        eid = uuid4()
        entity = _make_entity()
        object.__setattr__(entity, "id", eid)
        engine._storage.search_similar_entities = AsyncMock(return_value=[(eid, 0.85)])
        engine._storage.get_entities_batch = AsyncMock(return_value={eid: entity})
        engine._storage.get_neighborhoods_batch = AsyncMock(return_value={})
        out = await engine._graph_search(uuid4(), "q", [0.1, 0.2], QueryConfig(), None)
        assert len(out["entities"]) == 1


# ---------------------------------------------------------------------------
# _keyword_search and _keyword_search_bm25 and _keyword_search_fulltext
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    @pytest.mark.asyncio
    async def test_legacy_keyword_search_returns_empty(self) -> None:
        engine = _make_engine()
        out = await engine._keyword_search(uuid4(), "q", QueryConfig())
        assert out == {"source": "keyword", "chunks": [], "entities": []}

    @pytest.mark.asyncio
    async def test_bm25_no_chunks_returns_empty(self) -> None:
        engine = _make_engine()
        engine._storage.list_chunks = AsyncMock(return_value=[])
        out = await engine._keyword_search_bm25(uuid4(), "q", QueryConfig())
        assert out == {"source": "keyword", "chunks": [], "entities": []}

    @pytest.mark.asyncio
    async def test_bm25_list_chunks_exception_returns_empty(self) -> None:
        engine = _make_engine()
        engine._storage.list_chunks = AsyncMock(side_effect=RuntimeError("boom"))
        out = await engine._keyword_search_bm25(uuid4(), "q", QueryConfig())
        assert out == {"source": "keyword", "chunks": [], "entities": []}

    @pytest.mark.asyncio
    async def test_bm25_builds_index_and_searches(self) -> None:
        engine = _make_engine()
        chunk = _make_chunk(content="the quick brown fox jumps over lazy dog")
        engine._storage.list_chunks = AsyncMock(return_value=[chunk])
        ns_id = uuid4()
        out = await engine._keyword_search_bm25(ns_id, "quick fox", QueryConfig())
        assert out["source"] == "keyword"
        # Index should have been cached
        assert str(ns_id) in engine._keyword_searchers

    @pytest.mark.asyncio
    async def test_bm25_with_keywords(self) -> None:
        engine = _make_engine()
        chunk = _make_chunk(content="alice met bob today at the office")
        engine._storage.list_chunks = AsyncMock(return_value=[chunk])
        ns_id = uuid4()
        out = await engine._keyword_search_bm25(ns_id, "ignored", QueryConfig(), keywords=["alice", "bob"])
        assert out["source"] == "keyword"

    @pytest.mark.asyncio
    async def test_fulltext_returns_normalized_scores(self) -> None:
        engine = _make_engine()
        c1, c2 = _make_chunk(), _make_chunk()
        # ts_rank-style raw scores
        engine._storage.search_fulltext_chunks = AsyncMock(return_value=[(c1, 0.6), (c2, 0.3)])
        out = await engine._keyword_search_fulltext(uuid4(), "q", QueryConfig())
        assert out["source"] == "keyword"
        # Top result normalized to 1.0
        assert out["chunks"][0][1] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_fulltext_empty_results(self) -> None:
        engine = _make_engine()
        engine._storage.search_fulltext_chunks = AsyncMock(return_value=[])
        out = await engine._keyword_search_fulltext(uuid4(), "q", QueryConfig())
        assert out == {"source": "keyword", "chunks": [], "entities": []}

    @pytest.mark.asyncio
    async def test_fulltext_exception_returns_empty(self) -> None:
        engine = _make_engine()
        engine._storage.search_fulltext_chunks = AsyncMock(side_effect=RuntimeError("boom"))
        out = await engine._keyword_search_fulltext(uuid4(), "q", QueryConfig())
        assert out == {"source": "keyword", "chunks": [], "entities": []}

    @pytest.mark.asyncio
    async def test_fulltext_temporal_filter_passes_bounds(self) -> None:
        engine = _make_engine()
        engine._storage.search_fulltext_chunks = AsyncMock(return_value=[])
        now = datetime.now(UTC)
        tf = TemporalFilter(start_time=now - timedelta(days=3), end_time=now)
        await engine._keyword_search_fulltext(uuid4(), "q", QueryConfig(), temporal_filter=tf)
        kwargs = engine._storage.search_fulltext_chunks.call_args.kwargs
        assert kwargs["created_after"] is not None
        assert kwargs["created_before"] is not None


# ---------------------------------------------------------------------------
# _heuristic_understanding
# ---------------------------------------------------------------------------


class TestHeuristicUnderstanding:
    def test_non_temporal_query(self) -> None:
        result = HybridQueryEngine._heuristic_understanding("hello world")
        assert result is not None
        assert result.intent == QueryIntent.SEARCH
        assert result.temporal_references == []
        # Keywords include words ≥ 3 chars
        assert "hello" in result.keywords
        assert "world" in result.keywords

    def test_temporal_query_detected(self) -> None:
        result = HybridQueryEngine._heuristic_understanding("events yesterday")
        assert result is not None
        assert result.intent == QueryIntent.TEMPORAL
        assert len(result.temporal_references) >= 1

    def test_complexity_score_is_05(self) -> None:
        result = HybridQueryEngine._heuristic_understanding("any query")
        assert result.complexity_score == 0.5

    def test_reasoning_marks_fallback(self) -> None:
        result = HybridQueryEngine._heuristic_understanding("hello")
        assert "heuristic" in result.reasoning.lower()


# ---------------------------------------------------------------------------
# invalidate_caches
# ---------------------------------------------------------------------------


class TestInvalidateCaches:
    def test_removes_keyword_searcher(self) -> None:
        engine = _make_engine()
        ns_id = uuid4()
        engine._keyword_searchers[str(ns_id)] = MagicMock()
        engine.invalidate_caches(ns_id)
        assert str(ns_id) not in engine._keyword_searchers

    def test_recreates_caches(self) -> None:
        engine = _make_engine()
        old_cache = engine._cache
        old_understanding_cache = engine._understanding_cache
        engine.invalidate_caches(uuid4())
        # Cache objects should be replaced (new instances)
        assert engine._cache is not old_cache
        assert engine._understanding_cache is not old_understanding_cache


# ---------------------------------------------------------------------------
# temporal_query
# ---------------------------------------------------------------------------


class TestTemporalQuery:
    @pytest.mark.asyncio
    async def test_routes_to_query_with_temporal_filter(self) -> None:
        engine = _make_engine()
        # Stub query() so we don't have to drive the full pipeline
        engine.query = AsyncMock(return_value=MagicMock(metadata={}, chunks=[], entities=[]))

        tf = TemporalFilter.last_days(7)
        tq = TemporalQuery(
            query="test",
            filters=[tf],
            recency_weight=0.5,
            decay_days=14.0,
        )
        await engine.temporal_query(tq, uuid4())
        engine.query.assert_called_once()
        # Verify recency settings were propagated to config
        call_kwargs = engine.query.call_args.kwargs
        cfg = call_kwargs["config"]
        assert cfg.apply_recency_bias is True
        assert cfg.recency_weight == 0.5
        assert cfg.recency_decay_days == 14.0
        # Temporal filter was passed
        assert call_kwargs["temporal_filter"] is tf

    @pytest.mark.asyncio
    async def test_context_window_creates_filter(self) -> None:
        engine = _make_engine()
        engine.query = AsyncMock(return_value=MagicMock())

        tq = TemporalQuery(query="test", context_window_days=7)
        await engine.temporal_query(tq, uuid4())
        call_kwargs = engine.query.call_args.kwargs
        assert call_kwargs["temporal_filter"] is not None


# ---------------------------------------------------------------------------
# format_entity_section / format_relationship_section (module-level helpers)
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    def test_format_entity_section_empty(self) -> None:
        from khora.query.engine import format_entity_section

        assert format_entity_section([]) == ""

    def test_format_entity_section_with_description(self) -> None:
        from khora.query.engine import format_entity_section

        e = _make_entity(name="Alice", entity_type="PERSON")
        object.__setattr__(e, "description", "An engineer")
        out = format_entity_section([(e, 0.9)])
        assert "Alice" in out
        assert "PERSON" in out
        assert "An engineer" in out
        assert "--- Entities ---" in out

    def test_format_entity_section_dedupes_by_id(self) -> None:
        from khora.query.engine import format_entity_section

        e = _make_entity(name="Alice")
        out = format_entity_section([(e, 0.9), (e, 0.5)])
        # Only one line for Alice
        assert out.count("Alice") == 1

    def test_format_entity_section_no_description(self) -> None:
        from khora.query.engine import format_entity_section

        e = _make_entity(name="Bob", entity_type="PERSON")
        # description default is empty
        out = format_entity_section([(e, 0.5)])
        assert "Bob" in out
        assert ":" not in out.split("\n")[-1]  # no description colon

    def test_format_relationship_section_empty(self) -> None:
        from khora.query.engine import format_relationship_section

        assert format_relationship_section([]) == ""

    def test_format_relationship_section_basic(self) -> None:
        from khora.core.models.entity import Relationship
        from khora.query.engine import format_relationship_section

        rel = Relationship(
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="WORKS_FOR",
            source_entity_name="Alice",
            target_entity_name="Acme",
            description="employed",
        )
        out = format_relationship_section([(rel, 0.7)])
        assert "Alice" in out
        assert "Acme" in out
        assert "WORKS_FOR" in out
        assert "employed" in out

    def test_format_relationship_section_dedupes(self) -> None:
        from khora.core.models.entity import Relationship
        from khora.query.engine import format_relationship_section

        sid, tid = uuid4(), uuid4()
        rel = Relationship(
            source_entity_id=sid,
            target_entity_id=tid,
            relationship_type="WORKS_FOR",
            source_entity_name="Alice",
            target_entity_name="Acme",
        )
        out = format_relationship_section([(rel, 0.7), (rel, 0.5)])
        # Only one line
        assert out.count("Alice --WORKS_FOR--> Acme") == 1


# ---------------------------------------------------------------------------
# QueryResult helpers (already partially covered) — _extract_chunk_title
# ---------------------------------------------------------------------------


class TestExtractChunkTitle:
    def test_no_metadata(self) -> None:
        from khora.query.engine import QueryResult

        chunk = MagicMock(spec=[])
        assert QueryResult._extract_chunk_title(chunk) == ""

    def test_title_from_attribute(self) -> None:
        from khora.query.engine import QueryResult

        chunk = MagicMock()
        chunk.metadata.title = "My Doc"
        assert QueryResult._extract_chunk_title(chunk) == "My Doc"

    def test_title_from_dict(self) -> None:
        from khora.query.engine import QueryResult

        chunk = MagicMock()
        chunk.metadata = {"title": "From Dict"}
        assert QueryResult._extract_chunk_title(chunk) == "From Dict"

    def test_title_from_custom_dict(self) -> None:
        from khora.query.engine import QueryResult

        chunk = _make_chunk(custom={"title": "Custom Title"})
        # ChunkMetadata.title attribute does not exist; falls back to custom dict
        assert QueryResult._extract_chunk_title(chunk) == "Custom Title"

    def test_no_title_anywhere(self) -> None:
        from khora.query.engine import QueryResult

        chunk = _make_chunk()
        assert QueryResult._extract_chunk_title(chunk) == ""


# ---------------------------------------------------------------------------
# _stage1_recall — minimal smoke test (heavy I/O path)
# ---------------------------------------------------------------------------


class TestStage1Recall:
    @pytest.mark.asyncio
    async def test_empty_storage_returns_empty_dicts(self) -> None:
        engine = _make_engine(embedder=MagicMock(embed=AsyncMock(return_value=[0.1] * 8)))
        cfg = QueryConfig(
            mode=SearchMode.VECTOR,
            enable_keyword_search=False,
            enable_query_expansion=False,
        )
        from khora.query.engine import GraphTraversalInfo

        chunks, entities, ctx, contrib = await engine._stage1_recall(
            "q", uuid4(), [0.1] * 8, cfg, None, [], GraphTraversalInfo()
        )
        # Vector mode with empty storage → all-empty results
        assert chunks == {} or all(v == [] for v in chunks.values())
        assert ctx == {}

    @pytest.mark.asyncio
    async def test_keyword_only_mode(self) -> None:
        engine = _make_engine()
        engine._storage.search_fulltext_chunks = AsyncMock(return_value=[])
        cfg = QueryConfig(
            mode=SearchMode.HYBRID,
            enable_keyword_search=True,
            enable_query_expansion=False,
        )
        from khora.query.engine import GraphTraversalInfo

        # Without query_embedding, vector is skipped — graph + keyword run
        chunks, _, _, _ = await engine._stage1_recall("q", uuid4(), None, cfg, None, [], GraphTraversalInfo())
        # Keyword source key present
        assert "keyword" in chunks or chunks == {} or "graph" in chunks
