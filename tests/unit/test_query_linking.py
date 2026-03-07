"""Unit tests for query/linking.py — Entity linking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity
from khora.query.linking import (
    EntityLinker,
    LinkedEntity,
    LinkingResult,
    link_query_entities,
)
from khora.query.understanding import EntityMention


def _make_entity(name: str, entity_type: str = "PERSON") -> Entity:
    """Helper to create an Entity."""
    return Entity(
        namespace_id=uuid4(),
        name=name,
        entity_type=entity_type,
    )


def _make_mention(name: str, entity_type: str = "PERSON") -> EntityMention:
    """Helper to create an EntityMention."""
    return EntityMention(name=name, entity_type=entity_type)


class TestLinkedEntity:
    """Tests for LinkedEntity dataclass."""

    def test_is_linked_true(self) -> None:
        """is_linked returns True when entity is set."""
        entity = _make_entity("Alice")
        le = LinkedEntity(
            mention=_make_mention("Alice"),
            entity=entity,
            match_method="exact",
            match_score=1.0,
        )
        assert le.is_linked is True

    def test_is_linked_false(self) -> None:
        """is_linked returns False when entity is None."""
        le = LinkedEntity(mention=_make_mention("Unknown"))
        assert le.is_linked is False


class TestLinkingResult:
    """Tests for LinkingResult dataclass."""

    def test_linked_count(self) -> None:
        """linked_count counts successfully linked entities."""
        result = LinkingResult(
            linked_entities=[
                LinkedEntity(mention=_make_mention("A"), entity=_make_entity("A")),
                LinkedEntity(mention=_make_mention("B"), entity=None),
                LinkedEntity(mention=_make_mention("C"), entity=_make_entity("C")),
            ],
            total_mentions=3,
        )
        assert result.linked_count == 2

    def test_success_rate(self) -> None:
        """success_rate is linked_count / total_mentions."""
        result = LinkingResult(
            linked_entities=[
                LinkedEntity(mention=_make_mention("A"), entity=_make_entity("A")),
                LinkedEntity(mention=_make_mention("B"), entity=None),
            ],
            total_mentions=2,
        )
        assert result.success_rate == 0.5

    def test_success_rate_zero_mentions(self) -> None:
        """success_rate with zero mentions returns 0.0."""
        result = LinkingResult(total_mentions=0)
        assert result.success_rate == 0.0

    def test_get_linked_entity_ids(self) -> None:
        """get_linked_entity_ids returns IDs of linked entities."""
        entity1 = _make_entity("A")
        entity2 = _make_entity("B")
        result = LinkingResult(
            linked_entities=[
                LinkedEntity(mention=_make_mention("A"), entity=entity1),
                LinkedEntity(mention=_make_mention("B"), entity=None),
                LinkedEntity(mention=_make_mention("C"), entity=entity2),
            ],
            total_mentions=3,
        )
        ids = result.get_linked_entity_ids()
        assert len(ids) == 2
        assert entity1.id in ids
        assert entity2.id in ids


class TestEntityLinker:
    """Tests for EntityLinker."""

    def _make_linker(
        self,
        entities: list[Entity] | None = None,
        embedder: MagicMock | None = None,
    ) -> tuple[EntityLinker, MagicMock]:
        """Create a linker with mock storage."""
        storage = MagicMock()
        storage.get_entity_by_name = AsyncMock(return_value=None)
        storage.list_entities = AsyncMock(return_value=entities or [])
        linker = EntityLinker(storage, embedder=embedder)
        return linker, storage

    @pytest.mark.asyncio
    async def test_empty_mentions(self) -> None:
        """Empty mentions returns empty result."""
        linker, _ = self._make_linker()
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_pipeline_stage = MagicMock()
            result = await linker.link([], uuid4())
        assert result.total_mentions == 0

    @pytest.mark.asyncio
    async def test_exact_match_early_exit(self) -> None:
        """Exact name match returns immediately."""
        entity = _make_entity("Alice")
        storage = MagicMock()
        storage.get_entity_by_name = AsyncMock(return_value=entity)
        storage.list_entities = AsyncMock(return_value=[entity])
        linker = EntityLinker(storage)

        mention = _make_mention("Alice")
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_pipeline_stage = MagicMock()
            result = await linker.link([mention], uuid4())

        assert result.linked_count == 1
        linked = result.linked_entities[0]
        assert linked.match_method == "exact"
        assert linked.match_score == 1.0

    @pytest.mark.asyncio
    async def test_fuzzy_match(self) -> None:
        """Fuzzy matching finds near-identical names."""
        entity = _make_entity("Alexander Hamilton")
        storage = MagicMock()
        storage.get_entity_by_name = AsyncMock(return_value=None)
        storage.list_entities = AsyncMock(return_value=[entity])
        linker = EntityLinker(storage, fuzzy_threshold=0.7)

        mention = _make_mention("Alexandr Hamilton")
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_pipeline_stage = MagicMock()
            result = await linker.link([mention], uuid4())

        assert result.linked_count == 1
        assert result.linked_entities[0].match_method == "fuzzy"

    @pytest.mark.asyncio
    async def test_no_match(self) -> None:
        """No match returns unlinked entity."""
        storage = MagicMock()
        storage.get_entity_by_name = AsyncMock(return_value=None)
        storage.list_entities = AsyncMock(return_value=[])
        linker = EntityLinker(storage)

        mention = _make_mention("Nonexistent")
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_pipeline_stage = MagicMock()
            result = await linker.link([mention], uuid4())

        assert result.linked_count == 0
        assert result.unlinked_count == 1

    @pytest.mark.asyncio
    async def test_batch_linking(self) -> None:
        """Multiple mentions are linked in parallel."""
        entity_a = _make_entity("Alice")
        entity_b = _make_entity("Bob")

        storage = MagicMock()
        storage.get_entity_by_name = AsyncMock(
            side_effect=lambda ns, name, et: {
                "Alice": entity_a,
                "Bob": entity_b,
            }.get(name)
        )
        storage.list_entities = AsyncMock(return_value=[entity_a, entity_b])
        linker = EntityLinker(storage)

        mentions = [_make_mention("Alice"), _make_mention("Bob")]
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_pipeline_stage = MagicMock()
            result = await linker.link(mentions, uuid4())

        assert result.linked_count == 2
        assert result.total_mentions == 2

    def test_type_compatibility_exact(self) -> None:
        """Exact type match is compatible."""
        linker, _ = self._make_linker()
        assert linker._types_compatible("PERSON", "PERSON") is True

    def test_type_compatibility_concept_wildcard(self) -> None:
        """CONCEPT is compatible with any type."""
        linker, _ = self._make_linker()
        assert linker._types_compatible("CONCEPT", "PERSON") is True
        assert linker._types_compatible("PERSON", "CONCEPT") is True

    def test_type_compatibility_custom_wildcard(self) -> None:
        """CUSTOM is compatible with any type."""
        linker, _ = self._make_linker()
        assert linker._types_compatible("CUSTOM", "PERSON") is True

    def test_type_incompatible(self) -> None:
        """Incompatible types return False."""
        linker, _ = self._make_linker()
        assert linker._types_compatible("PERSON", "ORGANIZATION") is False

    def test_type_penalty_exact_match(self) -> None:
        """Exact type match has no penalty."""
        linker, _ = self._make_linker()
        assert linker._type_penalty("PERSON", "PERSON") == 1.0

    def test_type_penalty_no_type_hint(self) -> None:
        """No type hint means no penalty."""
        linker, _ = self._make_linker()
        assert linker._type_penalty(None, "PERSON") == 1.0

    def test_type_penalty_wildcard(self) -> None:
        """Wildcard types get minimal penalty."""
        linker, _ = self._make_linker()
        assert linker._type_penalty("CONCEPT", "PERSON") == 0.9

    def test_type_penalty_wrong_type(self) -> None:
        """Wrong type gets heavy penalty."""
        linker, _ = self._make_linker()
        assert linker._type_penalty("PERSON", "ORGANIZATION") == 0.3


class TestLinkQueryEntities:
    """Tests for the link_query_entities convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function(self) -> None:
        """Convenience function creates linker and delegates."""
        entity = _make_entity("Alice")
        storage = MagicMock()
        storage.get_entity_by_name = AsyncMock(return_value=entity)
        storage.list_entities = AsyncMock(return_value=[entity])

        mentions = [_make_mention("Alice")]
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_pipeline_stage = MagicMock()
            result = await link_query_entities(mentions, uuid4(), storage)

        assert result.linked_count == 1
