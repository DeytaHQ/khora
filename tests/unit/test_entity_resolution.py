"""Unit tests for extraction/entity_resolution.py — Entity resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity
from khora.extraction.entity_resolution import (
    DEFAULT_MERGE_THRESHOLDS,
    EntityResolver,
    MergeSource,
    ResolutionCandidate,
    ResolutionMetrics,
    ResolutionResult,
    _coordinates_distance,
    _normalize_domain,
    _normalize_email,
    _parse_coordinates,
    resolve_and_merge_entity,
)


def _make_entity(name: str = "Test", entity_type: str = "PERSON", **kwargs) -> Entity:
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
        assert entity.entity_type == "PERSON"
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
    async def test_unknown_entity_type_preserved(self) -> None:
        """Unknown entity type is preserved as a string (not mapped to CONCEPT)."""
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
        assert entity.entity_type == "UNKNOWN_TYPE"

    @pytest.mark.asyncio
    async def test_shared_resolver_reused(self) -> None:
        """Passing a shared resolver reuses its cache."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        # Create resolver and populate cache
        resolver = EntityResolver(storage, embedder=None)
        await resolver.resolve("Alice", "PERSON", ns_id)

        # Reset mock to verify it's not called again
        storage.list_entities.reset_mock()

        # Use shared resolver
        entity, is_new = await resolve_and_merge_entity(
            "Alice",
            "PERSON",
            ns_id,
            storage,
            resolver=resolver,
        )
        assert is_new is False
        # Cache should have been used, no new storage call
        storage.list_entities.assert_not_called()


class TestEntityResolverPreloadedCache:
    """Tests for EntityResolver.with_preloaded_cache()."""

    @pytest.mark.asyncio
    async def test_preloaded_cache_loads_entities(self) -> None:
        """with_preloaded_cache loads entities for all default types."""
        ns_id = uuid4()
        alice = _make_entity("Alice", namespace_id=ns_id)
        acme = _make_entity("Acme Corp", "ORGANIZATION", namespace_id=ns_id)

        storage = MagicMock()

        async def mock_list_entities(namespace_id, entity_type=None, limit=5000):
            if entity_type == "PERSON":
                return [alice]
            if entity_type == "ORGANIZATION":
                return [acme]
            return []

        storage.list_entities = AsyncMock(side_effect=mock_list_entities)

        resolver = await EntityResolver.with_preloaded_cache(
            storage,
            namespace_id=ns_id,
            entity_types=["PERSON", "ORGANIZATION"],
        )

        # Cache should be populated
        assert f"{ns_id}:PERSON" in resolver._entity_cache
        assert f"{ns_id}:ORGANIZATION" in resolver._entity_cache
        assert len(resolver._entity_cache[f"{ns_id}:PERSON"]) == 1
        assert len(resolver._entity_cache[f"{ns_id}:ORGANIZATION"]) == 1

    @pytest.mark.asyncio
    async def test_preloaded_cache_resolve_uses_cache(self) -> None:
        """Resolving after preload uses cached entities."""
        ns_id = uuid4()
        alice = _make_entity("Alice", namespace_id=ns_id)

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[alice])

        resolver = await EntityResolver.with_preloaded_cache(
            storage,
            namespace_id=ns_id,
            entity_types=["PERSON"],
        )

        # Reset mock to verify cache is used
        storage.list_entities.reset_mock()

        result = await resolver.resolve("Alice", "PERSON", ns_id)
        assert result.is_duplicate is True
        assert result.match_type == "exact"
        # Should not have called storage again
        storage.list_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_preloaded_cache_no_namespace(self) -> None:
        """with_preloaded_cache without namespace returns empty cache."""
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])

        resolver = await EntityResolver.with_preloaded_cache(
            storage,
            namespace_id=None,
        )

        assert len(resolver._entity_cache) == 0

    def test_share_cache_from(self) -> None:
        """share_cache_from copies cache from another resolver."""
        storage = MagicMock()
        resolver1 = EntityResolver(storage, embedder=None)
        resolver2 = EntityResolver(storage, embedder=None)

        # Add to resolver1's cache
        resolver1._entity_cache["key1"] = [_make_entity("Alice")]

        # Share cache
        resolver2.share_cache_from(resolver1)

        # resolver2 should have the same cache
        assert "key1" in resolver2._entity_cache
        assert resolver2._entity_cache is resolver1._entity_cache

    def test_add_to_cache(self) -> None:
        """add_to_cache adds new entity to cache."""
        ns_id = uuid4()
        storage = MagicMock()
        resolver = EntityResolver(storage, embedder=None)

        # Initialize cache
        resolver._entity_cache[f"{ns_id}:PERSON"] = []

        # Add entity
        alice = _make_entity("Alice", namespace_id=ns_id)
        resolver.add_to_cache(ns_id, alice)

        assert len(resolver._entity_cache[f"{ns_id}:PERSON"]) == 1
        assert resolver._entity_cache[f"{ns_id}:PERSON"][0].name == "Alice"

    def test_add_to_cache_no_duplicate(self) -> None:
        """add_to_cache doesn't add duplicate entities."""
        ns_id = uuid4()
        storage = MagicMock()
        resolver = EntityResolver(storage, embedder=None)

        alice = _make_entity("Alice", namespace_id=ns_id)
        resolver._entity_cache[f"{ns_id}:PERSON"] = [alice]

        # Try to add same entity again
        resolver.add_to_cache(ns_id, alice)

        # Should still be just one
        assert len(resolver._entity_cache[f"{ns_id}:PERSON"]) == 1

    def test_add_to_cache_creates_key(self) -> None:
        """add_to_cache creates cache key if not exists."""
        ns_id = uuid4()
        storage = MagicMock()
        resolver = EntityResolver(storage, embedder=None)

        alice = _make_entity("Alice", namespace_id=ns_id)
        resolver.add_to_cache(ns_id, alice)

        assert f"{ns_id}:PERSON" in resolver._entity_cache
        assert len(resolver._entity_cache[f"{ns_id}:PERSON"]) == 1


class TestAttributeNormalization:
    """Tests for attribute normalization helper functions."""

    def test_normalize_email(self) -> None:
        """Email normalization handles various formats."""
        assert _normalize_email("Alice@Example.COM") == "alice@example.com"
        assert _normalize_email("  bob@test.org  ") == "bob@test.org"
        assert _normalize_email(None) is None
        assert _normalize_email("") is None

    def test_normalize_domain(self) -> None:
        """Domain normalization strips protocol and www."""
        assert _normalize_domain("https://www.example.com/") == "example.com"
        assert _normalize_domain("http://example.com") == "example.com"
        assert _normalize_domain("www.example.com") == "example.com"
        assert _normalize_domain("example.com") == "example.com"
        assert _normalize_domain(None) is None
        assert _normalize_domain("") is None

    def test_parse_coordinates(self) -> None:
        """Coordinate parsing handles various formats."""
        assert _parse_coordinates("40.7128, -74.0060") == (40.7128, -74.0060)
        assert _parse_coordinates("40.7128,-74.0060") == (40.7128, -74.0060)
        assert _parse_coordinates("(40.7128, -74.0060)") == (40.7128, -74.0060)
        assert _parse_coordinates(None) is None
        assert _parse_coordinates("") is None
        assert _parse_coordinates("invalid") is None

    def test_coordinates_distance(self) -> None:
        """Coordinate distance calculation is reasonably accurate."""
        # NYC to LA is roughly 3940 km
        nyc = (40.7128, -74.0060)
        la = (34.0522, -118.2437)
        distance = _coordinates_distance(nyc, la)
        assert 3900 < distance < 4000

        # Same point should be 0
        assert _coordinates_distance(nyc, nyc) == 0.0


class TestPerTypeMergeThresholds:
    """Tests for per-type merge thresholds."""

    def test_default_thresholds_exist(self) -> None:
        """Default thresholds are defined for common types."""
        assert "PERSON" in DEFAULT_MERGE_THRESHOLDS
        assert "ORGANIZATION" in DEFAULT_MERGE_THRESHOLDS
        assert "LOCATION" in DEFAULT_MERGE_THRESHOLDS
        assert "CONCEPT" in DEFAULT_MERGE_THRESHOLDS

    def test_person_threshold_highest(self) -> None:
        """PERSON has highest threshold to avoid false merges."""
        assert DEFAULT_MERGE_THRESHOLDS["PERSON"] >= DEFAULT_MERGE_THRESHOLDS["CONCEPT"]
        assert DEFAULT_MERGE_THRESHOLDS["PERSON"] >= DEFAULT_MERGE_THRESHOLDS["LOCATION"]

    def test_resolver_uses_per_type_thresholds(self) -> None:
        """Resolver uses per-type thresholds."""
        storage = MagicMock()
        resolver = EntityResolver(storage, embedder=None)

        # Check threshold retrieval
        assert resolver._get_merge_threshold("PERSON") == DEFAULT_MERGE_THRESHOLDS["PERSON"]
        assert resolver._get_merge_threshold("CONCEPT") == DEFAULT_MERGE_THRESHOLDS["CONCEPT"]
        assert resolver._get_merge_threshold("UNKNOWN") == 0.85  # Default

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override defaults."""
        storage = MagicMock()
        custom = {"PERSON": 0.99, "CUSTOM_TYPE": 0.50}
        resolver = EntityResolver(storage, embedder=None, merge_thresholds=custom)

        assert resolver._get_merge_threshold("PERSON") == 0.99
        assert resolver._get_merge_threshold("CUSTOM_TYPE") == 0.50
        assert resolver._get_merge_threshold("UNKNOWN") == 0.85

    @pytest.mark.asyncio
    async def test_per_type_threshold_affects_merge_decision(self) -> None:
        """Per-type threshold affects should_merge decision."""
        ns_id = uuid4()
        # Create entity with fuzzy-matchable name
        existing = _make_entity("Alexander Hamilton", namespace_id=ns_id)

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        # With low threshold (CONCEPT), fuzzy match should merge
        resolver = EntityResolver(
            storage,
            merge_thresholds={"CONCEPT": 0.75},
            fuzzy_threshold=0.80,
        )
        result = await resolver.resolve("Alexandr Hamilton", "CONCEPT", ns_id)
        assert result.is_duplicate is True
        assert result.should_merge is True

        # With very high threshold (0.99), same fuzzy match should not merge
        # "Alexandr Hamilton" vs "Alexander Hamilton" has ~0.97 score
        resolver2 = EntityResolver(
            storage,
            merge_thresholds={"PERSON": 0.99},
            fuzzy_threshold=0.80,
        )
        result2 = await resolver2.resolve("Alexandr Hamilton", "PERSON", ns_id)
        assert result2.is_duplicate is True
        assert result2.should_merge is False  # Score (~0.97) too low for 0.99 threshold


class TestAttributeMatching:
    """Tests for attribute-level matching."""

    def _make_resolver(self, entities: list[Entity] | None = None) -> tuple[EntityResolver, MagicMock]:
        """Create resolver with mock storage."""
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=entities or [])
        resolver = EntityResolver(storage, embedder=None, attribute_match=True)
        return resolver, storage

    @pytest.mark.asyncio
    async def test_person_email_match(self) -> None:
        """PERSON entities match on email."""
        ns_id = uuid4()
        existing = _make_entity(
            "Alice Smith",
            namespace_id=ns_id,
            attributes={"email": "alice@example.com"},
        )
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve(
            "A. Smith",  # Different name
            "PERSON",
            ns_id,
            attributes={"email": "alice@example.com"},  # Same email
        )
        assert result.is_duplicate is True
        assert result.match_type == "attribute"
        assert result.match_score == 1.0

    @pytest.mark.asyncio
    async def test_person_different_email_no_match(self) -> None:
        """PERSON entities with different emails don't match well."""
        ns_id = uuid4()
        existing = _make_entity(
            "Alice Smith",
            namespace_id=ns_id,
            attributes={"email": "alice@example.com"},
        )
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve(
            "Alice Smith",  # Same name
            "PERSON",
            ns_id,
            attributes={"email": "alice.smith@different.com"},  # Different email
        )
        # Different email for same name suggests different person
        # Exact name match should still work
        assert result.is_duplicate is True
        assert result.match_type == "exact"

    @pytest.mark.asyncio
    async def test_organization_domain_match(self) -> None:
        """ORGANIZATION entities match on domain."""
        ns_id = uuid4()
        existing = _make_entity(
            "Acme Corporation",
            "ORGANIZATION",
            namespace_id=ns_id,
            attributes={"website": "https://www.acme.com"},
        )
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve(
            "ACME Corp",  # Different name variant
            "ORGANIZATION",
            ns_id,
            attributes={"domain": "acme.com"},  # Same domain
        )
        assert result.is_duplicate is True
        assert result.match_type == "attribute"
        assert result.match_score == 1.0

    @pytest.mark.asyncio
    async def test_location_coordinates_match(self) -> None:
        """LOCATION entities match on nearby coordinates."""
        ns_id = uuid4()
        existing = _make_entity(
            "Times Square",
            "LOCATION",
            namespace_id=ns_id,
            attributes={"coordinates": "40.7580, -73.9855"},
        )
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve(
            "Times Sq",
            "LOCATION",
            ns_id,
            attributes={"coordinates": "40.7581, -73.9856"},  # Very close
        )
        assert result.is_duplicate is True
        assert result.match_type == "attribute"
        assert result.match_score == 1.0

    @pytest.mark.asyncio
    async def test_technology_vendor_match(self) -> None:
        """TECHNOLOGY entities match on vendor."""
        ns_id = uuid4()
        existing = _make_entity(
            "TensorFlow",
            "TECHNOLOGY",
            namespace_id=ns_id,
            attributes={"vendor": "Google", "type": "framework"},
        )
        resolver, _ = self._make_resolver([existing])

        result = await resolver.resolve(
            "Tensorflow",  # Different case
            "TECHNOLOGY",
            ns_id,
            attributes={"vendor": "Google", "type": "framework"},
        )
        # Exact name match should take precedence
        assert result.is_duplicate is True

    @pytest.mark.asyncio
    async def test_attribute_match_disabled(self) -> None:
        """Attribute matching can be disabled."""
        ns_id = uuid4()
        existing = _make_entity(
            "Different Name",
            namespace_id=ns_id,
            attributes={"email": "same@example.com"},
        )

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        resolver = EntityResolver(
            storage,
            attribute_match=False,
            exact_match=False,
            alias_match=False,
            fuzzy_match=False,
        )
        result = await resolver.resolve(
            "Another Name",
            "PERSON",
            ns_id,
            attributes={"email": "same@example.com"},
        )
        # Should not match without attribute matching
        assert result.is_duplicate is False


class TestResolutionMetrics:
    """Tests for resolution quality metrics."""

    def test_metrics_creation(self) -> None:
        """Metrics can be created and have correct defaults."""
        metrics = ResolutionMetrics()
        assert metrics.total_resolutions == 0
        assert metrics.exact_matches == 0
        assert metrics.new_entities == 0
        assert metrics.average_merge_confidence == 0.0

    def test_record_resolution(self) -> None:
        """Recording resolutions updates counters correctly."""
        metrics = ResolutionMetrics()

        metrics.record_resolution("PERSON", "exact", 1.0, True)
        assert metrics.total_resolutions == 1
        assert metrics.exact_matches == 1
        assert metrics.matches_by_type["PERSON"] == 1
        assert metrics.merges_by_type["PERSON"] == 1

        metrics.record_resolution("ORGANIZATION", "fuzzy", 0.88, True)
        assert metrics.total_resolutions == 2
        assert metrics.fuzzy_matches == 1
        assert metrics.low_confidence_merges == 1  # 0.88 < 0.9

    def test_record_new_entity(self) -> None:
        """Recording new entity (no match) updates correctly."""
        metrics = ResolutionMetrics()

        metrics.record_resolution("PERSON", "", 0.0, False)
        assert metrics.new_entities == 1
        assert metrics.total_resolutions == 1

    def test_average_merge_confidence(self) -> None:
        """Average merge confidence is calculated correctly."""
        metrics = ResolutionMetrics()

        metrics.record_resolution("PERSON", "exact", 1.0, True)
        metrics.record_resolution("PERSON", "fuzzy", 0.9, True)

        assert metrics.average_merge_confidence == 0.95

    def test_match_rate_by_type(self) -> None:
        """Match rate by type is calculated correctly."""
        metrics = ResolutionMetrics()

        # PERSON: 2 resolutions, 1 merge = 50%
        metrics.record_resolution("PERSON", "exact", 1.0, True)
        metrics.record_resolution("PERSON", "", 0.0, False)

        # ORG: 1 resolution, 1 merge = 100%
        metrics.record_resolution("ORGANIZATION", "alias", 0.95, True)

        rates = metrics.match_rate_by_type
        assert rates["PERSON"] == 0.5
        assert rates["ORGANIZATION"] == 1.0

    def test_resolver_tracks_metrics(self) -> None:
        """Resolver tracks metrics when enabled."""
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])

        resolver = EntityResolver(storage, embedder=None, track_metrics=True)
        assert resolver.metrics is not None
        assert isinstance(resolver.metrics, ResolutionMetrics)

    def test_resolver_no_metrics_by_default(self) -> None:
        """Resolver does not track metrics by default."""
        storage = MagicMock()
        resolver = EntityResolver(storage, embedder=None)
        assert resolver.metrics is None

    @pytest.mark.asyncio
    async def test_resolver_records_metrics(self) -> None:
        """Resolver records metrics on resolution."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        resolver = EntityResolver(storage, embedder=None, track_metrics=True)

        await resolver.resolve("Alice", "PERSON", ns_id)
        await resolver.resolve("Bob", "PERSON", ns_id)

        assert resolver.metrics is not None
        assert resolver.metrics.total_resolutions == 2
        assert resolver.metrics.exact_matches == 1
        assert resolver.metrics.new_entities == 1


class TestMergeProvenance:
    """Tests for merge provenance tracking."""

    def test_merge_source_creation(self) -> None:
        """MergeSource can be created and converted to dict."""
        source = MergeSource(
            entity_id="abc-123",
            score=0.95,
            match_type="fuzzy",
            source_tool="slack",
        )

        d = source.to_dict()
        assert d["entity_id"] == "abc-123"
        assert d["score"] == 0.95
        assert d["match_type"] == "fuzzy"
        assert d["source_tool"] == "slack"

    @pytest.mark.asyncio
    async def test_merge_tracks_provenance(self) -> None:
        """Merging entities tracks provenance in metadata."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        entity, is_new = await resolve_and_merge_entity(
            "Alice",
            "PERSON",
            ns_id,
            storage,
            source_tool="slack",
            track_provenance=True,
        )

        assert is_new is False
        assert "merge_sources" in entity.metadata
        sources = entity.metadata["merge_sources"]
        assert len(sources) == 1
        assert sources[0]["source_tool"] == "slack"
        assert sources[0]["match_type"] == "exact"
        assert sources[0]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_merge_accumulates_provenance(self) -> None:
        """Multiple merges accumulate provenance."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)
        existing.metadata["merge_sources"] = [
            {"entity_id": "prev-1", "score": 0.95, "match_type": "fuzzy", "source_tool": "gmail"}
        ]

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        entity, is_new = await resolve_and_merge_entity(
            "Alice",
            "PERSON",
            ns_id,
            storage,
            source_tool="slack",
            track_provenance=True,
        )

        assert is_new is False
        sources = entity.metadata["merge_sources"]
        assert len(sources) == 2
        assert sources[0]["source_tool"] == "gmail"
        assert sources[1]["source_tool"] == "slack"

    @pytest.mark.asyncio
    async def test_provenance_disabled(self) -> None:
        """Provenance tracking can be disabled."""
        ns_id = uuid4()
        existing = _make_entity("Alice", namespace_id=ns_id)

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        entity, is_new = await resolve_and_merge_entity(
            "Alice",
            "PERSON",
            ns_id,
            storage,
            source_tool="slack",
            track_provenance=False,
        )

        assert is_new is False
        assert "merge_sources" not in entity.metadata

    @pytest.mark.asyncio
    async def test_new_entity_has_source_tool(self) -> None:
        """New entities have source_tool set."""
        ns_id = uuid4()

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])

        entity, is_new = await resolve_and_merge_entity(
            "NewPerson",
            "PERSON",
            ns_id,
            storage,
            source_tool="confluence",
        )

        assert is_new is True
        assert entity.source_tool == "confluence"
