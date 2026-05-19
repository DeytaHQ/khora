"""Coverage-driven tests for ``khora.query.understanding``.

Tests focus on the response parser ``_parse_comprehensive_response`` and
the public ``understand`` entry point with ``acompletion`` mocked at the
boundary. Pure helpers (date parsing, keyword extraction, source
priority sorting) are exercised directly.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from khora.query.understanding import (
    AnswerType,
    QueryIntent,
    QueryUnderstanding,
    SourcePriority,
)


def _full_payload(**overrides) -> dict:
    payload = {
        "intent": "question",
        "answer_type": "summary",
        "entities": [
            {
                "name": "Alice",
                "type": "PERSON",
                "confidence": 0.9,
                "aliases": ["Al"],
                "context_hint": "engineer",
            }
        ],
        "relationships": [
            {
                "from_entity": "Alice",
                "relationship_type": "WORKS_ON",
                "to_entity": "Phoenix",
                "importance": 0.8,
            }
        ],
        "temporal": [
            {
                "type": "relative",
                "text": "last week",
                "start_date": "2026-05-11",
                "end_date": "2026-05-18",
            }
        ],
        "expanded_queries": ["alternative phrasing"],
        "keywords": ["alice", "phoenix"],
        "source_priority": {
            "slack": 0.9,
            "linear": 0.8,
            "notion": 0.05,  # below threshold → filtered out
            "attio": 0.2,
            "gong": 0.1,
            "github": 0.7,
            "bamboohr": 0.5,
        },
        "search_strategy": {
            "use_vector": True,
            "use_graph": True,
            "use_keyword": False,
            "vector_weight": 0.5,
            "graph_weight": 0.3,
            "keyword_weight": 0.2,
            "graph_depth": 3,
            "explore_neighborhoods": False,
            "reasoning": "entity-heavy",
        },
        "follow_up_queries": [
            {
                "query": "deeper dive",
                "reasoning": "explore Phoenix",
                "target_sources": ["linear"],
                "priority": 0.9,
            }
        ],
        "requires_multi_step": True,
        "complexity_score": 0.75,
        "confidence": 0.85,
        "reasoning": "complex multi-source",
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
class TestSourcePriority:
    def test_get_top_sources_sorts_descending(self) -> None:
        # All 7 sources explicitly set (defaults are 1.0, which would
        # produce a tied top-3 keyed by definition order).
        sp = SourcePriority(
            slack=0.9,
            linear=0.5,
            notion=0.7,
            attio=0.1,
            gong=0.0,
            github=0.0,
            bamboohr=0.0,
        )
        top = sp.get_top_sources(n=3)
        # gong/github/bamboohr at 0.0 must be filtered (>0 check)
        assert top == ["slack", "notion", "linear"]

    def test_get_top_sources_filters_zero(self) -> None:
        # All 7 sources at 0.0 → empty list (the implementation drops
        # zero-scored entries before truncating to n).
        sp = SourcePriority(
            slack=0.0,
            linear=0.0,
            notion=0.0,
            attio=0.0,
            gong=0.0,
            github=0.0,
            bamboohr=0.0,
        )
        assert sp.get_top_sources() == []

    def test_get_top_sources_default_n(self) -> None:
        sp = SourcePriority()  # all 1.0
        # Default n=3 returns first 3
        assert len(sp.get_top_sources()) == 3


@pytest.mark.unit
class TestParseComprehensiveResponse:
    def test_parses_full_payload(self) -> None:
        qu = QueryUnderstanding()
        response = json.dumps(_full_payload())
        result = qu._parse_comprehensive_response(response, "original q")

        assert result.original_query == "original q"
        assert result.intent == QueryIntent.QUESTION
        assert result.answer_type == AnswerType.SUMMARY
        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"
        assert result.entities[0].aliases == ["Al"]
        assert len(result.relationships) == 1
        assert result.relationships[0].relationship_type == "WORKS_ON"
        assert len(result.temporal_references) == 1
        assert result.temporal_references[0].start_date is not None
        assert "notion" in result.source_filters  # 0.05 < 0.1
        assert "slack" not in result.source_filters
        assert result.search_strategy.graph_depth == 3
        assert result.search_strategy.use_keyword is False
        assert result.complexity_score == 0.75
        assert result.requires_multi_step is True
        assert len(result.follow_up_queries) == 1

    def test_unwraps_json_code_fence(self) -> None:
        qu = QueryUnderstanding()
        body = json.dumps(_full_payload())
        wrapped = f"```json\n{body}\n```"
        result = qu._parse_comprehensive_response(wrapped, "q")
        assert result.intent == QueryIntent.QUESTION

    def test_unwraps_plain_code_fence(self) -> None:
        qu = QueryUnderstanding()
        body = json.dumps(_full_payload())
        wrapped = f"```\n{body}\n```"
        result = qu._parse_comprehensive_response(wrapped, "q")
        assert result.intent == QueryIntent.QUESTION

    def test_invalid_json_returns_fallback(self) -> None:
        qu = QueryUnderstanding()
        result = qu._parse_comprehensive_response("not json at all", "q")
        assert result.original_query == "q"
        assert result.intent == QueryIntent.SEARCH
        assert result.confidence == 0.3

    def test_unknown_intent_returns_unknown(self) -> None:
        qu = QueryUnderstanding()
        payload = _full_payload(intent="weird_intent_xyz")
        result = qu._parse_comprehensive_response(json.dumps(payload), "q")
        assert result.intent == QueryIntent.UNKNOWN

    def test_unknown_answer_type_returns_unknown(self) -> None:
        qu = QueryUnderstanding()
        payload = _full_payload(answer_type="weird_answer_type")
        result = qu._parse_comprehensive_response(json.dumps(payload), "q")
        assert result.answer_type == AnswerType.UNKNOWN

    def test_missing_optional_fields_uses_defaults(self) -> None:
        qu = QueryUnderstanding()
        # Minimum viable payload — everything optional missing
        response = json.dumps({"intent": "search"})
        result = qu._parse_comprehensive_response(response, "q")
        assert result.intent == QueryIntent.SEARCH
        assert result.entities == []
        assert result.relationships == []
        assert result.temporal_references == []
        assert result.complexity_score == 0.5
        # All-default SourcePriority → no source_filters
        assert result.source_filters == []


@pytest.mark.unit
class TestParseIsoDate:
    def test_parses_iso_with_z(self) -> None:
        qu = QueryUnderstanding()
        out = qu._parse_iso_date("2024-06-15T10:00:00Z")
        assert out is not None
        assert out.year == 2024 and out.month == 6 and out.day == 15

    def test_parses_plain_date(self) -> None:
        qu = QueryUnderstanding()
        out = qu._parse_iso_date("2024-06-15")
        assert out is not None

    def test_parses_microseconds(self) -> None:
        qu = QueryUnderstanding()
        out = qu._parse_iso_date("2024-06-15T10:00:00.123456")
        assert out is not None

    def test_none_input_returns_none(self) -> None:
        qu = QueryUnderstanding()
        assert qu._parse_iso_date(None) is None

    def test_string_null_returns_none(self) -> None:
        qu = QueryUnderstanding()
        assert qu._parse_iso_date("null") is None

    def test_invalid_format_returns_none(self) -> None:
        qu = QueryUnderstanding()
        assert qu._parse_iso_date("not a date") is None

    def test_rejects_far_future_date(self) -> None:
        qu = QueryUnderstanding()
        future = (datetime.now() + timedelta(days=5000)).strftime("%Y-%m-%d")
        assert qu._parse_iso_date(future) is None

    def test_rejects_ancient_date(self) -> None:
        qu = QueryUnderstanding()
        assert qu._parse_iso_date("1990-01-01") is None


@pytest.mark.unit
class TestExtractKeywordsSimple:
    def test_strips_stopwords(self) -> None:
        qu = QueryUnderstanding()
        keywords = qu._extract_keywords_simple("What is the project status for Phoenix")
        # 'project' filtered? no — 'project' isn't a stopword. 'what','is','the','for' are.
        assert "project" in keywords
        assert "status" in keywords
        assert "phoenix" in keywords
        assert "is" not in keywords
        assert "the" not in keywords

    def test_strips_short_tokens(self) -> None:
        qu = QueryUnderstanding()
        # words length <= 2 dropped
        keywords = qu._extract_keywords_simple("a bb ccc dddd")
        assert "ccc" in keywords
        assert "dddd" in keywords
        assert "a" not in keywords
        assert "bb" not in keywords

    def test_strips_punctuation(self) -> None:
        qu = QueryUnderstanding()
        keywords = qu._extract_keywords_simple("alice, where is bob?")
        assert "alice" in keywords
        assert "bob" in keywords


@pytest.mark.unit
class TestUnderstandFlow:
    async def test_understand_round_trips_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = _full_payload()

        async def fake_acomp(prompt, config, **kwargs) -> str:
            return json.dumps(payload)

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        qu = QueryUnderstanding()
        result = await qu.understand("test query")
        assert result.intent == QueryIntent.QUESTION
        assert len(result.entities) == 1

    async def test_understand_disables_expansions_when_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(prompt, config, **kwargs) -> str:
            return json.dumps(_full_payload())

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        qu = QueryUnderstanding()
        result = await qu.understand(
            "test",
            expand_query=False,
            extract_entities=False,
            detect_temporal=False,
        )
        assert result.expanded_queries == []
        assert result.entities == []
        assert result.temporal_references == []

    async def test_understand_lightweight_uses_lightweight_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen_prompts: list[str] = []

        async def fake_acomp(prompt, config, **kwargs) -> str:
            seen_prompts.append(prompt)
            return json.dumps(_full_payload())

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        qu = QueryUnderstanding()
        await qu.understand("q", lightweight=True)
        # Lightweight prompt has fewer sections, shorter overall
        assert "follow_up_queries" not in seen_prompts[0]

    async def test_understand_falls_back_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(prompt, config, **kwargs) -> str:
            raise RuntimeError("llm boom")

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        qu = QueryUnderstanding()
        result = await qu.understand("the project status for phoenix")
        # Fallback result: intent=SEARCH, confidence=0.3, keywords populated
        assert result.intent == QueryIntent.SEARCH
        assert result.confidence == 0.3
        assert "phoenix" in result.keywords


@pytest.mark.unit
class TestUnderstandingResultProperties:
    def test_has_temporal_true_when_refs_present(self) -> None:
        qu = QueryUnderstanding()
        result = qu._parse_comprehensive_response(json.dumps(_full_payload()), "q")
        assert result.has_temporal is True
        assert result.has_entities is True
        all_queries = result.get_all_queries()
        assert "q" in all_queries
        assert "alternative phrasing" in all_queries
        names = result.get_entity_names()
        assert "Alice" in names
        assert "Al" in names  # aliases included
