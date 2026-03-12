"""Unit tests for parameterized extraction (DYT-262).

Tests that entity_types and relationship_types parameters are correctly
threaded through the extraction pipeline: MemoryLake -> Engine -> Extractor -> Prompt.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.extraction.extractors.base import ExtractionResult
from khora.extraction.extractors.llm import LLMEntityExtractor
from khora.extraction.skills.base import EntityTypeConfig, ExpertiseConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(entities: list | None = None, relationships: list | None = None) -> MagicMock:
    """Create a mock LLM response with valid JSON content."""
    payload = {
        "entities": entities or [],
        "relationships": relationships or [],
        "events": [],
    }
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(payload)
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    return mock_response


def _make_multi_response(num_sections: int) -> MagicMock:
    """Create a mock LLM response for extract_multi with sections."""
    sections = []
    for _ in range(num_sections):
        sections.append({"entities": [], "relationships": [], "events": []})
    payload = {"sections": sections}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(payload)
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    return mock_response


def _mock_config() -> MagicMock:
    """Create a mock KhoraConfig with all required methods."""
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


_RESOLVE_ROW_ID = uuid4()


def _mock_engine() -> MagicMock:
    """Create a mock engine with all required methods."""
    mock_eng = MagicMock()
    mock_eng._storage = MagicMock()
    mock_eng._storage.resolve_namespace = AsyncMock(return_value=_RESOLVE_ROW_ID)
    mock_eng._embedder = MagicMock()
    mock_eng.connect = AsyncMock()
    mock_eng.disconnect = AsyncMock()
    mock_eng.health_check = AsyncMock(return_value={"status": "healthy"})
    mock_eng.remember = AsyncMock()
    mock_eng.recall = AsyncMock()
    mock_eng.forget = AsyncMock()
    mock_eng.remember_batch = AsyncMock()
    mock_eng.create_namespace = AsyncMock()
    mock_eng.get_namespace = AsyncMock()
    mock_eng.get_entity = AsyncMock()
    mock_eng.list_entities = AsyncMock(return_value=[])
    mock_eng.find_related_entities = AsyncMock(return_value=[])
    mock_eng.get_document = AsyncMock()
    mock_eng.list_documents = AsyncMock(return_value=[])
    mock_eng.search_entities = AsyncMock(return_value=[])
    mock_eng.stats = AsyncMock()
    return mock_eng


def _make_lake(*, connected: bool = False):
    """Create a MemoryLake with mocked config, optionally pre-connected."""
    from khora.memory_lake import MemoryLake

    with patch("khora.memory_lake.load_config", return_value=_mock_config()):
        lake = MemoryLake()

    if connected:
        lake._connected = True
        lake._engine = _mock_engine()

    return lake


# ---------------------------------------------------------------------------
# 1. No defaults injected when None — extract() errors without types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractNoDefaultsWhenNone:
    """Calling extract() without entity_types/relationship_types no longer injects defaults."""

    async def test_extract_none_types_raises_without_expertise(self) -> None:
        """When entity_types=None, relationship_types=None, and no expertise, extract errors."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)

        with patch("litellm.acompletion", new_callable=AsyncMock):
            # Without types or expertise, the prompt builder gets None and fails
            with pytest.raises(TypeError):
                await extractor.extract("Alice works at Acme Corp in New York.")

    async def test_extract_with_explicit_types_works(self) -> None:
        """When entity_types and relationship_types are provided, extract succeeds."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)
        captured_messages: list = []

        async def _capture_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _make_llm_response()

        with patch("litellm.acompletion", side_effect=_capture_acompletion):
            await extractor.extract(
                "Alice works at Acme Corp in New York.",
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        assert len(captured_messages) == 1
        user_prompt = captured_messages[0][1]["content"]
        assert "PERSON" in user_prompt
        assert "ORGANIZATION" in user_prompt
        assert "WORKS_FOR" in user_prompt


# ---------------------------------------------------------------------------
# 2. Custom entity types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractCustomEntityTypes:
    """Passing explicit entity_types overrides defaults."""

    async def test_extract_custom_entity_types(self) -> None:
        """Prompt contains custom types, not defaults like PERSON."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)
        captured_messages: list = []

        async def _capture_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _make_llm_response()

        with patch("litellm.acompletion", side_effect=_capture_acompletion):
            await extractor.extract(
                "BRCA1 gene is targeted by Olaparib.",
                entity_types=["DRUG", "GENE"],
                relationship_types=["TARGETS"],
            )

        user_prompt = captured_messages[0][1]["content"]

        assert "DRUG" in user_prompt
        assert "GENE" in user_prompt
        # Default type PERSON should NOT be in the entity types line
        # (it may appear in the template boilerplate, so check the specific line)
        entity_line = [line for line in user_prompt.split("\n") if "Entity types to extract:" in line]
        assert len(entity_line) == 1
        assert "PERSON" not in entity_line[0]


# ---------------------------------------------------------------------------
# 3. Custom relationship types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractCustomRelationshipTypes:
    """Passing explicit relationship_types overrides defaults."""

    async def test_extract_custom_relationship_types(self) -> None:
        """Prompt contains custom relationship types, not defaults like WORKS_FOR."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)
        captured_messages: list = []

        async def _capture_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _make_llm_response()

        with patch("litellm.acompletion", side_effect=_capture_acompletion):
            await extractor.extract(
                "BRCA1 is targeted by Olaparib which inhibits PARP.",
                entity_types=["DRUG", "GENE", "PROTEIN"],
                relationship_types=["TARGETS", "INHIBITS"],
            )

        user_prompt = captured_messages[0][1]["content"]

        assert "TARGETS" in user_prompt
        assert "INHIBITS" in user_prompt
        # Check the specific relationship types line
        rel_line = [line for line in user_prompt.split("\n") if "Relationship types to use:" in line]
        assert len(rel_line) == 1
        assert "WORKS_FOR" not in rel_line[0]


# ---------------------------------------------------------------------------
# 4. Explicit overrides expertise
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractExplicitOverridesExpertise:
    """Explicit params take precedence over expertise config."""

    async def test_extract_explicit_overrides_expertise(self) -> None:
        """Prompt contains explicit DRUG, not expertise CONCEPT."""
        expertise = ExpertiseConfig(
            name="tech_expert",
            entity_types=[
                EntityTypeConfig(name="CONCEPT", description="A concept"),
                EntityTypeConfig(name="TECHNOLOGY", description="A technology"),
            ],
        )

        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)
        captured_messages: list = []

        async def _capture_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _make_llm_response()

        with patch("litellm.acompletion", side_effect=_capture_acompletion):
            await extractor.extract(
                "Olaparib targets BRCA1.",
                entity_types=["DRUG"],
                relationship_types=["TARGETS"],
                expertise=expertise,
            )

        user_prompt = captured_messages[0][1]["content"]
        entity_line = [line for line in user_prompt.split("\n") if "Entity types to extract:" in line]
        assert len(entity_line) == 1
        assert "DRUG" in entity_line[0]
        assert "CONCEPT" not in entity_line[0]


# ---------------------------------------------------------------------------
# 5. Empty lists pass through as-is (no defaults injected)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractEmptyListNoDefaults:
    """Empty lists [] are NOT normalized to defaults anymore."""

    async def test_extract_empty_list_no_defaults_injected(self) -> None:
        """Empty entity_types=[] and relationship_types=[] do not inject defaults."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)
        captured_messages: list = []

        async def _capture_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _make_llm_response()

        with patch("litellm.acompletion", side_effect=_capture_acompletion):
            await extractor.extract(
                "Alice works at Acme.",
                entity_types=[],
                relationship_types=[],
            )

        user_prompt = captured_messages[0][1]["content"]

        # Old defaults should NOT appear since empty lists are no longer
        # normalized to defaults
        entity_line = [line for line in user_prompt.split("\n") if "Entity types to extract:" in line]
        if entity_line:
            assert "PERSON" not in entity_line[0], "Old default PERSON should not be injected for empty list"
            assert "ORGANIZATION" not in entity_line[0], "Old default ORGANIZATION should not be injected"


# ---------------------------------------------------------------------------
# 6. extract_multi threads custom types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractMultiCustomTypes:
    """extract_multi passes custom types to the batch prompt."""

    async def test_extract_multi_custom_types(self) -> None:
        """Batch prompt contains custom entity and relationship types."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)
        captured_messages: list = []

        async def _capture_acompletion(**kwargs):
            captured_messages.append(kwargs["messages"])
            return _make_multi_response(2)

        with patch("litellm.acompletion", side_effect=_capture_acompletion):
            results = await extractor.extract_multi(
                ["Text one about drugs.", "Text two about genes."],
                entity_types=["DRUG"],
                relationship_types=["TARGETS"],
                tiered_extraction=False,
            )

        assert len(results) == 2
        # At least one LLM call should have been made
        assert len(captured_messages) >= 1
        user_prompt = captured_messages[0][1]["content"]

        assert "DRUG" in user_prompt
        assert "TARGETS" in user_prompt


# ---------------------------------------------------------------------------
# 7. extract_entities task threads types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractEntitiesTaskThreadsTypes:
    """The extract_entities pipeline task passes custom types to the extractor."""

    async def test_extract_entities_task_threads_types(self) -> None:
        """extract_entities() passes entity_types to extractor.extract_multi()."""
        from khora.pipelines.tasks.extract import extract_entities

        mock_chunk = MagicMock()
        mock_chunk.content = "Olaparib targets BRCA1 gene."
        mock_chunk.namespace_id = uuid4()
        mock_chunk.document_id = uuid4()
        mock_chunk.id = uuid4()
        mock_chunk.created_at = None

        mock_result = ExtractionResult(entities=[], relationships=[], events=[])

        with patch(
            "khora.extraction.extractors.LLMEntityExtractor.extract_multi",
            new_callable=AsyncMock,
            return_value=[mock_result],
        ) as mock_extract_multi:
            await extract_entities(
                [mock_chunk],
                entity_types=["DRUG", "GENE"],
                relationship_types=["TARGETS"],
            )

        mock_extract_multi.assert_awaited_once()
        call_kwargs = mock_extract_multi.call_args
        assert call_kwargs.kwargs["entity_types"] == ["DRUG", "GENE"]
        assert call_kwargs.kwargs["relationship_types"] == ["TARGETS"]


# ---------------------------------------------------------------------------
# 8. MemoryLake.remember() threads types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryLakeRememberThreadsTypes:
    """MemoryLake.remember() passes entity_types to the engine."""

    async def test_memory_lake_remember_threads_types(self) -> None:
        """engine.remember() is called with entity_types and relationship_types."""
        from khora.memory_lake import RememberResult

        lake = _make_lake(connected=True)
        ns_id = uuid4()

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
            await lake.remember(
                "Olaparib targets BRCA1.",
                namespace=ns_id,
                entity_types=["DRUG", "GENE"],
                relationship_types=["TARGETS"],
            )

        lake._engine.remember.assert_awaited_once()
        call_kwargs = lake._engine.remember.call_args
        assert call_kwargs.kwargs["entity_types"] == ["DRUG", "GENE"]
        assert call_kwargs.kwargs["relationship_types"] == ["TARGETS"]


# ---------------------------------------------------------------------------
# 9. MemoryLake.remember_batch() threads types
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryLakeRememberBatchThreadsTypes:
    """MemoryLake.remember_batch() passes entity_types to the engine."""

    async def test_memory_lake_remember_batch_threads_types(self) -> None:
        """engine.remember_batch() is called with entity_types and relationship_types."""
        from khora.memory_lake import BatchResult

        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = BatchResult(
            total=1,
            processed=1,
            skipped=0,
            failed=0,
            chunks=1,
            entities=0,
            relationships=0,
        )
        lake._engine.remember_batch = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.remember_batch(
                [{"content": "Olaparib targets BRCA1."}],
                namespace=ns_id,
                entity_types=["DRUG"],
                relationship_types=["TARGETS"],
            )

        lake._engine.remember_batch.assert_awaited_once()
        call_kwargs = lake._engine.remember_batch.call_args
        assert call_kwargs.kwargs["entity_types"] == ["DRUG"]
        assert call_kwargs.kwargs["relationship_types"] == ["TARGETS"]


# ---------------------------------------------------------------------------
# 10. remember() requires ontology params
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberRequiresOntologyParams:
    """Verify that calling lake.remember() without entity_types or relationship_types raises TypeError."""

    async def test_remember_missing_entity_types_raises(self) -> None:
        """remember() without entity_types raises TypeError."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        with pytest.raises(TypeError):
            await lake.remember("some text", namespace=ns_id, relationship_types=["KNOWS"])

    async def test_remember_missing_relationship_types_raises(self) -> None:
        """remember() without relationship_types raises TypeError."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        with pytest.raises(TypeError):
            await lake.remember("some text", namespace=ns_id, entity_types=["PERSON"])

    async def test_remember_missing_both_raises(self) -> None:
        """remember() without either ontology param raises TypeError."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        with pytest.raises(TypeError):
            await lake.remember("some text", namespace=ns_id)


# ---------------------------------------------------------------------------
# 11. remember_batch() requires ontology params
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberBatchRequiresOntologyParams:
    """Verify that calling lake.remember_batch() without entity_types or relationship_types raises TypeError."""

    async def test_remember_batch_missing_entity_types_raises(self) -> None:
        """remember_batch() without entity_types raises TypeError."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        with pytest.raises(TypeError):
            await lake.remember_batch(
                [{"content": "text"}],
                namespace=ns_id,
                relationship_types=["KNOWS"],
            )

    async def test_remember_batch_missing_relationship_types_raises(self) -> None:
        """remember_batch() without relationship_types raises TypeError."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        with pytest.raises(TypeError):
            await lake.remember_batch(
                [{"content": "text"}],
                namespace=ns_id,
                entity_types=["PERSON"],
            )

    async def test_remember_batch_missing_both_raises(self) -> None:
        """remember_batch() without either ontology param raises TypeError."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        with pytest.raises(TypeError):
            await lake.remember_batch([{"content": "text"}], namespace=ns_id)
