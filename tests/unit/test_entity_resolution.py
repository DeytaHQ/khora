"""Unit tests for extraction/entity_resolution.py — Entity resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity, EntityType
from khora.extraction.entity_resolution import (
    EntityResolver,
    ResolutionCandidate,
    ResolutionResult,
    resolve_and_merge_entity,
)


def _make_entity(name: str = "Test", entity_type: EntityType = EntityType.PERSON, **kwargs) -> Entity:
    """Helper to create an Entity with sensible defaults."""
    return Entity(
        namespace_id=kwargs.get("namespace_id", uuid4()),
        name=name,
        entity_type=entity_type,
        description=kwargs.get("description", ""),
        attributes=kwargs.get("attributes", {}),
        metadata=kwargs.get("metadata", {}),
    )


class TestResolutionCandidate:
    """Tests for ResolutionCandidate dataclass."""

    def test_create(self) -> None:
        """Basic creation."""
        entity = _make_entity()
        c = ResolutionCandidate(entity=entity, match_type="exact", score=1.0)
        assert c.entity is entity
        assert c.match_type == "exact"
        assert c.score == 1.0


class TestResolutionResult:
    """Tests for ResolutionResult dataclass."""

    def test_no_match(self) -> None:
        """No match defaults."""
        r = ResolutionResult(is_duplicate=False)
        assert r.is_duplicate is False
        assert r.existing_entity is None
        assert r.should_merge is False

    def test_match(self) -> None:
        """Match with existing entity."""
        entity = _make_entity()
        r = ResolutionResult(
            is_duplicate=True,
            existing_entity=entity,
            match_type="exact",
            match_score=1.0,
            should_merge=True,
        )
        assert r.is_duplicate is True
        assert r.should_merge is True


class TestEntityResolver:
    """Tests for EntityResolver."""

    def _make_resolver(self, entities: list[Entity] | None = None) -> tuple[EntityResolver, MagicMock]:
        """Create resolver with mock storage."""
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=entities or [])
        resolver = EntityResolver(storage, embedder=None)
        return resolver, storage

    @pytest.mark.asyncio
    async def test_exact_match(self) -> None:
        """Exact name match returns duplicate."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve("Alice", "PERSON", ns_id)
        assert result.is_duplicate is True
        assert result.match_type == "exact"
        assert result.match_score == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_case_insensitive(self) -> None:
        """Exact match is case-insensitive."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve("alice", "PERSON", ns_id)
        assert result.is_duplicate is True
        assert result.match_type == "exact"

    @pytest.mark.asyncio
    async def test_alias_match(self) -> None:
        """Alias matching finds entities by their aliases."""
        ns_id = uuid4()
        existing = _make_entity("Robert", namespace_id=ns_id, metadata={"aliases": ["Bob"]})
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve("Bob", "PERSON", ns_id)
        assert result.is_duplicate is True
        assert result.match_type == "alias"

    @pytest.mark.asyncio
    async def test_alias_match_reverse(self) -> None:
        """New entity alias matches existing entity name."""
        ns_id = uuid4()
        existing = _make_entity("Bob", namespace_id=ns_id)
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve("Robert", "PERSON", ns_id, aliases=["Bob"])
        assert result.is_duplicate is True
        assert result.match_type == "alias"

    @pytest.mark.asyncio
    async def test_fuzzy_match(self) -> None:
        """Fuzzy match catches near-identical names."""
        ns_id = uuid4()
        existing = _make_entity("Alexander Hamilton", namespace_id=ns_id)
        resolver, _ = self._make_resolver([existing])
        resolver._fuzzy_threshold = 0.8

        result = await resolver.resolve("Alexandr Hamilton", "PERSON", ns_id)
        assert result.is_duplicate is True
        assert result.match_type == "fuzzy"

    @pytest.mark.asyncio
    async def test_no_match(self) -> None:
        """No match returns non-duplicate."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve("Completely Different", "PERSON", ns_id)
        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_exact_match_prioritized(self) -> None:
        """Exact match short-circuits before fuzzy/embedding."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve("Alice", "PERSON", ns_id)
        # Should have returned immediately with exact match
        assert result.match_type == "exact"

    @pytest.mark.asyncio
    async def test_disabled_strategies(self) -> None:
        """Disabling strategies skips them."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        resolver = EntityResolver(
            storage,
            exact_match=False,
            alias_match=False,
            fuzzy_match=False,
            embedding_match=False,
        )
        result = await resolver.resolve("Alice", "PERSON", ns_id)
        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_cache_populated(self) -> None:
        """Entity cache is populated on first resolve call."""
        ns_id = uuid4()
        resolver, storage = self._make_resolver([])

        await resolver.resolve("Test", "PERSON", ns_id)
        # Cache should have been populated
        cache_key = f"{ns_id}:PERSON"
        assert cache_key in resolver._entity_cache

    def test_invalidate_cache_all(self) -> None:
        """invalidate_cache with no namespace clears everything."""
        resolver, _ = self._make_resolver()
        resolver._entity_cache["key1"] = []
        resolver._entity_cache["key2"] = []
        resolver.invalidate_cache()
        assert len(resolver._entity_cache) == 0

    def test_invalidate_cache_namespace(self) -> None:
        """invalidate_cache with namespace only clears matching entries."""
        ns_id = uuid4()
        resolver, _ = self._make_resolver()
        resolver._entity_cache[f"{ns_id}:PERSON"] = []
        resolver._entity_cache["other:PERSON"] = []
        resolver.invalidate_cache(namespace_id=ns_id)
        assert f"{ns_id}:PERSON" not in resolver._entity_cache
        assert "other:PERSON" in resolver._entity_cache


class TestResolveAndMergeEntity:
    """Tests for the resolve_and_merge_entity convenience function."""

    @pytest.mark.asyncio
    async def test_high_score_merge(self) -> None:
        """High-score match merges into existing entity."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)
        existing.mention_count = 1

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        entity, is_new = await resolve_and_merge_entity(
            "Alice",
            "PERSON",
            ns_id,
            storage,
            description="A person named Alice",
            attributes={"role": "engineer"},
        )
        assert is_new is False
        assert entity is existing
        assert entity.mention_count == 2
        assert entity.attributes.get("role") == "engineer"

    @pytest.mark.asyncio
    async def test_no_match_creates_new(self) -> None:
        """No match returns a new entity."""
        ns_id = uuid4()
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])

        entity, is_new = await resolve_and_merge_entity(
            "NewPerson",
            "PERSON",
            ns_id,
            storage,
            description="Brand new person",
        )
        assert is_new is True
        assert entity.name == "NewPerson"
        assert entity.entity_type == EntityType.PERSON
        assert entity.description == "Brand new person"

    @pytest.mark.asyncio
    async def test_merge_aliases(self) -> None:
        """Merging accumulates aliases."""
        ns_id = uuid4()
        existing = _make_entity("Robert", namespace_id=ns_id, metadata={"aliases": ["Bob"]})
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        entity, is_new = await resolve_and_merge_entity(
            "Robert",
            "PERSON",
            ns_id,
            storage,
            aliases=["Rob", "Bobby"],
        )
        assert is_new is False
        aliases = entity.metadata.get("aliases", [])
        assert "Bob" in aliases
        assert "Rob" in aliases
        assert "Bobby" in aliases

    @pytest.mark.asyncio
    async def test_merge_description_fill(self) -> None:
        """Merge fills empty description."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)
        existing.description = ""
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        entity, is_new = await resolve_and_merge_entity(
            "Alice",
            "PERSON",
            ns_id,
            storage,
            description="An engineer",
        )
        assert entity.description == "An engineer"

    @pytest.mark.asyncio
    async def test_unknown_entity_type_fallback(self) -> None:
        """Unknown entity type falls back to CONCEPT."""
        ns_id = uuid4()
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])

        entity, is_new = await resolve_and_merge_entity(
            "Something",
            "UNKNOWN_TYPE",
            ns_id,
            storage,
        )
        assert is_new is True
        assert entity.entity_type == EntityType.CONCEPT
