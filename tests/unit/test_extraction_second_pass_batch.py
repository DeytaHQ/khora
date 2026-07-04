"""#1409: two-pass relationship extraction on the batch path.

The second pass ("catches 30-40% more connections") previously ran only in
single-text ``extract()`` while the production ingest path uses
``extract_multi`` -> ``_extract_multi_batch``. These tests pin:

- the second pass fires on the batch path for under-connected sections
  (relationships < entities - 1) and merges its relationships;
- a second-pass failure records an ADR-001 Degradation and never zeroes
  out first-pass results (single and batch paths);
- the second pass requests a relationships-only response_format instead of
  the full entities+relationships+events strict schema.

#1420: the batch second pass is an explicit cost opt-in - it only runs when
the extractor is built with ``second_pass=True`` (wired from
``pipeline.extraction_second_pass`` / KHORA_PIPELINES_EXTRACTION_SECOND_PASS).
The default-off case makes no extra LLM call.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.extraction.extractors.base import ExtractedEntity
from khora.extraction.extractors.llm import LLMEntityExtractor

FIRST_PASS_SECTIONS = {
    "sections": [
        {
            # Under-connected: 3 entities, 0 relationships (< entities - 1)
            "entities": [
                {"name": "Alice", "entity_type": "PERSON"},
                {"name": "Bob", "entity_type": "PERSON"},
                {"name": "Acme", "entity_type": "ORGANIZATION"},
            ],
            "relationships": [],
            "events": [],
        },
        {
            # Well-connected enough: 1 entity, no second pass needed
            "entities": [{"name": "Solo", "entity_type": "PERSON"}],
            "relationships": [],
            "events": [],
        },
    ]
}

SECOND_PASS_SECTIONS = {
    "sections": [
        {
            "relationships": [
                {
                    "source_entity": "Alice",
                    "target_entity": "Acme",
                    "relationship_type": "WORKS_FOR",
                    "description": "Alice works at Acme",
                },
                {
                    "source_entity": "Bob",
                    "target_entity": "Alice",
                    "relationship_type": "KNOWS",
                    "description": "",
                },
                {
                    # Unknown entity: must be filtered out
                    "source_entity": "Outsider",
                    "target_entity": "Alice",
                    "relationship_type": "KNOWS",
                    "description": "",
                },
            ]
        }
    ]
}


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return resp


@pytest.mark.unit
class TestBatchSecondPass:
    @pytest.mark.asyncio
    async def test_second_pass_fires_on_batch_path(self) -> None:
        """Under-connected sections from _extract_multi_batch get a batched second pass."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, second_pass=True)

        calls: list[dict] = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _response(FIRST_PASS_SECTIONS)
            return _response(SECOND_PASS_SECTIONS)

        with (
            patch("litellm.acompletion", side_effect=fake_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["Alice and Bob discussed Acme.", "Solo update."],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
                tiered_extraction=False,
            )

        # One first-pass batch call + one batched second-pass call
        assert len(calls) == 2
        # Second-pass prompt carries the under-connected section's entities
        second_prompt = calls[1]["messages"][1]["content"]
        assert "Alice (PERSON)" in second_prompt
        assert "Acme (ORGANIZATION)" in second_prompt
        # Well-connected section is not resent
        assert "Solo" not in second_prompt

        assert len(results) == 2
        # Merged second-pass relationships, unknown-entity one filtered
        rel_keys = {(r.source_entity, r.target_entity, r.relationship_type) for r in results[0].relationships}
        assert rel_keys == {("Alice", "Acme", "WORKS_FOR"), ("Bob", "Alice", "KNOWS")}
        assert results[0].metadata["second_pass_relationships"] == 2
        # Untouched section keeps its first-pass output
        assert results[1].entities[0].name == "Solo"
        assert results[1].relationships == []

    @pytest.mark.asyncio
    async def test_second_pass_off_by_default(self) -> None:
        """#1420: default extractor makes NO second-pass call even for
        under-connected sections - the extra LLM cost is an explicit opt-in."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        calls: list[dict] = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return _response(FIRST_PASS_SECTIONS)

        with (
            patch("litellm.acompletion", side_effect=fake_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["Alice and Bob discussed Acme.", "Solo update."],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
                tiered_extraction=False,
            )

        # Only the first-pass call - the under-connected section did NOT
        # trigger a second pass.
        assert len(calls) == 1
        assert len(results) == 2
        # First-pass output is untouched.
        assert [e.name for e in results[0].entities] == ["Alice", "Bob", "Acme"]
        assert results[0].relationships == []
        assert "second_pass_relationships" not in results[0].metadata

    @pytest.mark.asyncio
    async def test_second_pass_not_triggered_when_connected(self) -> None:
        """No extra LLM call when every section meets the density trigger."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, second_pass=True)

        payload = {
            "sections": [
                {
                    "entities": [
                        {"name": "Alice", "entity_type": "PERSON"},
                        {"name": "Acme", "entity_type": "ORGANIZATION"},
                    ],
                    "relationships": [
                        {
                            "source_entity": "Alice",
                            "target_entity": "Acme",
                            "relationship_type": "WORKS_FOR",
                            "description": "",
                        }
                    ],
                    "events": [],
                }
            ]
        }
        mock_acompletion = AsyncMock(return_value=_response(payload))
        with (
            patch("litellm.acompletion", mock_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["Alice works at Acme."],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
                tiered_extraction=False,
            )

        assert mock_acompletion.call_count == 1
        assert len(results[0].relationships) == 1

    @pytest.mark.asyncio
    async def test_second_pass_failure_records_degradation_keeps_first_pass(self) -> None:
        """A failed batched second pass degrades loudly and preserves first-pass output."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, second_pass=True)

        first_pass = {
            "sections": [
                {
                    "entities": [
                        {"name": "Alice", "entity_type": "PERSON"},
                        {"name": "Bob", "entity_type": "PERSON"},
                        {"name": "Acme", "entity_type": "ORGANIZATION"},
                    ],
                    "relationships": [
                        {
                            "source_entity": "Alice",
                            "target_entity": "Bob",
                            "relationship_type": "KNOWS",
                            "description": "",
                        }
                    ],
                    "events": [],
                }
            ]
        }

        calls = 0

        async def fake_acompletion(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _response(first_pass)
            raise Exception("second pass boom")

        with (
            patch("litellm.acompletion", side_effect=fake_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["Alice, Bob and Acme."],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["KNOWS"],
                tiered_extraction=False,
            )

        assert calls == 2
        result = results[0]
        # First-pass output is intact - not zeroed out
        assert [e.name for e in result.entities] == ["Alice", "Bob", "Acme"]
        assert len(result.relationships) == 1
        # Not marked as a failed extraction
        assert "error" not in result.metadata
        # ADR-001: degradation recorded
        degradations = result.metadata["degradations"]
        assert degradations[0]["component"] == "extraction.llm.second_pass"
        assert degradations[0]["reason"] == "second_pass_failed"
        assert "second pass boom" in degradations[0]["detail"]

    @pytest.mark.asyncio
    async def test_batch_second_pass_requests_relationships_only_schema(self) -> None:
        """The batched second pass asks for a relationships-only strict schema."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1, second_pass=True)

        calls: list[dict] = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _response(FIRST_PASS_SECTIONS)
            return _response(SECOND_PASS_SECTIONS)

        with (
            patch("litellm.acompletion", side_effect=fake_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            await extractor.extract_multi(
                ["Alice and Bob discussed Acme.", "Solo update."],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
                tiered_extraction=False,
            )

        assert len(calls) == 2
        fmt = calls[1]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "relationship_multi_extraction_result"
        section_schema = fmt["json_schema"]["schema"]["properties"]["sections"]["items"]
        assert set(section_schema["properties"].keys()) == {"relationships"}


@pytest.mark.unit
class TestSinglePathSecondPass:
    @pytest.mark.asyncio
    async def test_single_second_pass_requests_relationships_only_schema(self) -> None:
        """The single-doc second pass no longer reuses the full extraction schema."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)

        first_pass = {
            "entities": [
                {"name": "Alice", "entity_type": "PERSON"},
                {"name": "Bob", "entity_type": "PERSON"},
                {"name": "Acme", "entity_type": "ORGANIZATION"},
            ],
            "relationships": [],
            "events": [],
        }
        second_pass = {
            "relationships": [
                {
                    "source_entity": "Alice",
                    "target_entity": "Acme",
                    "relationship_type": "WORKS_FOR",
                    "description": "",
                }
            ]
        }

        calls: list[dict] = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _response(first_pass)
            return _response(second_pass)

        with (
            patch("litellm.acompletion", side_effect=fake_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract(
                "Alice and Bob discussed Acme.",
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
            )

        assert len(calls) == 2
        fmt = calls[1]["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "relationship_extraction_result"
        assert set(fmt["json_schema"]["schema"]["properties"].keys()) == {"relationships"}
        assert len(result.relationships) == 1

    @pytest.mark.asyncio
    async def test_single_second_pass_failure_records_degradation(self) -> None:
        """extract(): second-pass failure is recorded on result.metadata (ADR-001)."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        first_pass = {
            "entities": [
                {"name": "Alice", "entity_type": "PERSON"},
                {"name": "Bob", "entity_type": "PERSON"},
                {"name": "Acme", "entity_type": "ORGANIZATION"},
            ],
            "relationships": [],
            "events": [],
        }

        calls = 0

        async def fake_acompletion(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _response(first_pass)
            raise Exception("second pass boom")

        with (
            patch("litellm.acompletion", side_effect=fake_acompletion),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract(
                "Alice and Bob discussed Acme.",
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
            )

        assert calls == 2
        assert [e.name for e in result.entities] == ["Alice", "Bob", "Acme"]
        degradations = result.metadata["degradations"]
        assert degradations[0]["component"] == "extraction.llm.second_pass"
        assert degradations[0]["reason"] == "second_pass_failed"


@pytest.mark.unit
class TestRelationshipResponseFormats:
    def test_single_format_off_allowlist_is_json_object(self) -> None:
        ex = LLMEntityExtractor(model="custom-llama")
        assert ex._get_relationship_response_format() == {"type": "json_object"}
        assert ex._get_relationship_multi_response_format() == {"type": "json_object"}

    def test_single_format_allowlist_is_relationships_only(self) -> None:
        ex = LLMEntityExtractor(model="gpt-4o-mini")
        fmt = ex._get_relationship_response_format()
        assert fmt["type"] == "json_schema"
        schema = fmt["json_schema"]["schema"]
        assert set(schema["properties"].keys()) == {"relationships"}
        assert schema["additionalProperties"] is False

    @pytest.mark.asyncio
    async def test_batch_helper_empty_items(self) -> None:
        ex = LLMEntityExtractor(model="custom-llama")
        assert await ex._extract_additional_relationships_batch([]) == []

    @pytest.mark.asyncio
    async def test_batch_helper_section_count_mismatch_is_additive(self) -> None:
        """Fewer returned sections than inputs: unmatched items get no extras, no crash."""
        ex = LLMEntityExtractor(model="custom-llama", max_retries=1)
        entities = [
            ExtractedEntity(name="Alice", entity_type="PERSON"),
            ExtractedEntity(name="Acme", entity_type="ORGANIZATION"),
        ]
        payload = {
            "sections": [
                {
                    "relationships": [
                        {
                            "source_entity": "Alice",
                            "target_entity": "Acme",
                            "relationship_type": "WORKS_FOR",
                            "description": "",
                        }
                    ]
                }
            ]
        }
        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=_response(payload)),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            out = await ex._extract_additional_relationships_batch(
                [(entities, "text one"), (entities, "text two")],
                relationship_types=["WORKS_FOR"],
            )
        assert len(out) == 2
        assert len(out[0]) == 1
        assert out[1] == []
