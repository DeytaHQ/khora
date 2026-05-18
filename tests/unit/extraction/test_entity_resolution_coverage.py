"""Coverage tests for ``khora.extraction.entity_resolution``.

Targets uncovered branches:
- ``ResolutionMetrics.log_summary`` (lines 137-158)
- ``MergeSource.to_dict`` (lines 182-189)
- ``_parse_coordinates`` malformed inputs (lines 237-238)
- ``_compute_person_attribute_similarity`` title/org scores (421-440)
- ``_compute_organization_attribute_similarity`` (461-483)
- ``_compute_location_attribute_similarity`` (504-533)
- ``_compute_technology_attribute_similarity`` (544-583)
- ``_compute_product_attribute_similarity`` (594-617)
- ``EntityResolver.with_preloaded_cache`` exception path (700-702)
- ``EntityResolver.resolve`` storage failure (791-793)
- Alias matching: new aliases match existing aliases (831-833)
- Embedding match path (865-887)
- ``resolve_and_merge_entity`` provenance + attribute merge (1008-1010)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity
from khora.extraction.entity_resolution import (
    EntityResolver,
    MergeSource,
    ResolutionMetrics,
    _coordinates_distance,
    _normalize_domain,
    _parse_coordinates,
    resolve_and_merge_entity,
)


def _make_entity(name: str = "Test", entity_type: str = "PERSON", **kwargs) -> Entity:
    return Entity(
        namespace_id=kwargs.pop("namespace_id", uuid4()),
        name=name,
        entity_type=entity_type,
        description=kwargs.pop("description", ""),
        attributes=kwargs.pop("attributes", {}),
        metadata=kwargs.pop("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHelperFunctions:
    def test_parse_coordinates_with_parens(self) -> None:
        assert _parse_coordinates("(40.7128, -74.0060)") == (40.7128, -74.0060)

    def test_parse_coordinates_brackets(self) -> None:
        assert _parse_coordinates("[40.7128,-74.0060]") == (40.7128, -74.0060)

    def test_parse_coordinates_malformed_returns_none(self) -> None:
        assert _parse_coordinates("not coords") is None
        assert _parse_coordinates("1,2,3") is None
        assert _parse_coordinates("") is None
        assert _parse_coordinates(None) is None

    def test_parse_coordinates_valueerror(self) -> None:
        # Triggers the except ValueError branch
        assert _parse_coordinates("abc,def") is None

    def test_normalize_domain_handles_protocol_www(self) -> None:
        assert _normalize_domain("https://www.example.com/") == "example.com"
        assert _normalize_domain("http://Example.com/") == "example.com"
        assert _normalize_domain(None) is None
        assert _normalize_domain("") is None

    def test_coordinates_distance_zero_for_same_point(self) -> None:
        d = _coordinates_distance((40.0, -74.0), (40.0, -74.0))
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_coordinates_distance_far_apart(self) -> None:
        # NYC to LA ~3940km
        d = _coordinates_distance((40.7128, -74.0060), (34.0522, -118.2437))
        assert 3900 < d < 4000


# ---------------------------------------------------------------------------
# ResolutionMetrics — log_summary and rare paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolutionMetricsExtras:
    def test_log_summary_zero_resolutions_is_noop(self) -> None:
        m = ResolutionMetrics()
        m.log_summary()  # Should not raise

    def test_log_summary_with_data(self) -> None:
        m = ResolutionMetrics()
        m.record_resolution("PERSON", "exact", 1.0, True)
        m.record_resolution("PERSON", "fuzzy", 0.85, True)
        m.record_resolution("ORGANIZATION", "", 0.0, False)
        m.log_summary()  # Should not raise

    def test_record_resolution_low_confidence_merge(self) -> None:
        m = ResolutionMetrics()
        m.record_resolution("PERSON", "fuzzy", 0.85, merged=True)
        assert m.low_confidence_merges == 1
        assert m.merges_by_type["PERSON"] == 1


@pytest.mark.unit
class TestMergeSource:
    def test_to_dict(self) -> None:
        eid = str(uuid4())
        src = MergeSource(entity_id=eid, score=0.95, match_type="exact", source_tool="slack")
        assert src.to_dict() == {
            "entity_id": eid,
            "score": 0.95,
            "match_type": "exact",
            "source_tool": "slack",
        }


# ---------------------------------------------------------------------------
# Attribute similarity branches
# ---------------------------------------------------------------------------


def _make_resolver(entities: list[Entity] | None = None) -> EntityResolver:
    storage = MagicMock()
    storage.list_entities = AsyncMock(return_value=entities or [])
    return EntityResolver(storage, embedder=None)


@pytest.mark.unit
class TestPersonAttributeSimilarity:
    def test_title_only(self) -> None:
        r = _make_resolver()
        score = r._compute_person_attribute_similarity(
            {"title": "Software Engineer"},
            {"title": "Software Engineer"},
        )
        assert score is not None and score > 0.5

    def test_organization_only(self) -> None:
        r = _make_resolver()
        score = r._compute_person_attribute_similarity(
            {"organization": "Acme Corp"},
            {"organization": "Acme Corp"},
        )
        assert score is not None and score > 0.5

    def test_no_comparable_attrs(self) -> None:
        r = _make_resolver()
        assert r._compute_person_attribute_similarity({}, {}) is None


@pytest.mark.unit
class TestOrganizationAttributeSimilarity:
    def test_exact_domain_match_returns_one(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"domain": "acme.com"},
            {"domain": "acme.com"},
            "ORGANIZATION",
        )
        assert score == 1.0

    def test_partial_domain_match(self) -> None:
        r = _make_resolver()
        # google.com is contained in cloud.google.com -> partial-match path
        score = r._compute_attribute_similarity(
            {"website": "google.com"},
            {"website": "cloud.google.com"},
            "ORGANIZATION",
        )
        assert score is not None and score > 0.5

    def test_different_domains_low_score(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"domain": "acme.com"},
            {"domain": "foobar.io"},
            "ORGANIZATION",
        )
        assert score == 0.3

    def test_industry_match(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"industry": "Technology"},
            {"industry": "Technology"},
            "ORGANIZATION",
        )
        assert score is not None and score > 0.0

    def test_type_match(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"type": "company"},
            {"type": "Company"},
            "ORGANIZATION",
        )
        assert score is not None and score > 0.0

    def test_no_attrs_returns_none(self) -> None:
        r = _make_resolver()
        assert r._compute_attribute_similarity({}, {}, "ORGANIZATION") is None


@pytest.mark.unit
class TestLocationAttributeSimilarity:
    def test_coordinates_within_1km_returns_one(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"coordinates": "40.7128, -74.0060"},
            {"coordinates": "40.7129, -74.0061"},
            "LOCATION",
        )
        assert score == 1.0

    def test_coordinates_within_5km(self) -> None:
        r = _make_resolver()
        # ~3km apart
        score = r._compute_attribute_similarity(
            {"coordinates": "40.7128, -74.0060"},
            {"coordinates": "40.7400, -74.0060"},
            "LOCATION",
        )
        assert score is not None and 0.5 < score < 1.0

    def test_coordinates_far_apart(self) -> None:
        r = _make_resolver()
        # NYC to LA
        score = r._compute_attribute_similarity(
            {"coordinates": "40.7128, -74.0060"},
            {"coordinates": "34.0522, -118.2437"},
            "LOCATION",
        )
        assert score == 0.2

    def test_coordinates_within_20km(self) -> None:
        r = _make_resolver()
        # ~12km apart in lat
        score = r._compute_attribute_similarity(
            {"coordinates": "40.7128, -74.0060"},
            {"coordinates": "40.8200, -74.0060"},
            "LOCATION",
        )
        assert score is not None

    def test_coordinates_within_100km(self) -> None:
        r = _make_resolver()
        # ~50km apart
        score = r._compute_attribute_similarity(
            {"coordinates": "40.7128, -74.0060"},
            {"coordinates": "41.1500, -74.0060"},
            "LOCATION",
        )
        assert score is not None

    def test_address_similarity(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"address": "123 Main St"},
            {"address": "123 Main St"},
            "LOCATION",
        )
        assert score is not None and score > 0.5

    def test_country_match(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"country": "USA"},
            {"country": "usa"},
            "LOCATION",
        )
        assert score is not None and score > 0.0

    def test_country_mismatch_low_signal(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"country": "USA"},
            {"country": "UK"},
            "LOCATION",
        )
        assert score == 0.2

    def test_no_attrs_returns_none(self) -> None:
        r = _make_resolver()
        assert r._compute_attribute_similarity({}, {}, "LOCATION") is None


@pytest.mark.unit
class TestTechnologyAttributeSimilarity:
    def test_vendor_strong_match(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"vendor": "Google"},
            {"vendor": "Google"},
            "TECHNOLOGY",
        )
        assert score is not None and score > 0.5

    def test_vendor_moderate_match(self) -> None:
        r = _make_resolver()
        # Two similar names — should land in 0.7-0.9 range
        score = r._compute_attribute_similarity(
            {"vendor": "Google Inc"},
            {"vendor": "Google"},
            "TECHNOLOGY",
        )
        assert score is not None

    def test_vendor_mismatch_low_signal(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"vendor": "Microsoft"},
            {"vendor": "Google"},
            "TECHNOLOGY",
        )
        assert score == 0.3

    def test_type_match(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"type": "framework"},
            {"type": "framework"},
            "TECHNOLOGY",
        )
        assert score is not None and score > 0.5

    def test_type_mismatch(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"type": "framework"},
            {"type": "library"},
            "TECHNOLOGY",
        )
        assert score is not None

    def test_version_same_major(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"version": "3.10.1"},
            {"version": "3.8.0"},
            "TECHNOLOGY",
        )
        assert score is not None and score > 0.5

    def test_version_different_major(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"version": "3.10.1"},
            {"version": "2.7.0"},
            "TECHNOLOGY",
        )
        assert score is not None

    def test_no_attrs_returns_none(self) -> None:
        r = _make_resolver()
        assert r._compute_attribute_similarity({}, {}, "TECHNOLOGY") is None


@pytest.mark.unit
class TestProductAttributeSimilarity:
    def test_vendor_strong_match(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"vendor": "Apple"},
            {"vendor": "Apple"},
            "PRODUCT",
        )
        assert score is not None and score > 0.5

    def test_vendor_weak_match(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"vendor": "Apple"},
            {"vendor": "Microsoft"},
            "PRODUCT",
        )
        assert score is not None

    def test_category_match(self) -> None:
        r = _make_resolver()
        score = r._compute_attribute_similarity(
            {"category": "smartphone"},
            {"category": "Smartphone"},
            "PRODUCT",
        )
        assert score is not None and score > 0.0

    def test_no_attrs_returns_none(self) -> None:
        r = _make_resolver()
        assert r._compute_attribute_similarity({}, {}, "PRODUCT") is None


@pytest.mark.unit
class TestComputeAttributeSimilarityUnknownType:
    def test_unknown_type_returns_none(self) -> None:
        r = _make_resolver()
        assert r._compute_attribute_similarity({"foo": "bar"}, {"foo": "bar"}, "UNKNOWN_TYPE") is None


# ---------------------------------------------------------------------------
# resolve() error path: storage failure populates empty cache
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveErrorPaths:
    @pytest.mark.asyncio
    async def test_storage_failure_returns_no_duplicate(self) -> None:
        ns_id = uuid4()
        storage = MagicMock()
        storage.list_entities = AsyncMock(side_effect=Exception("boom"))
        resolver = EntityResolver(storage, embedder=None)

        result = await resolver.resolve("Alice", "PERSON", ns_id)
        assert result.is_duplicate is False
        # Cache populated with empty list to avoid retry
        assert resolver._entity_cache[f"{ns_id}:PERSON"] == []

    @pytest.mark.asyncio
    async def test_alias_matches_other_alias(self) -> None:
        """New alias matches an existing alias (line 831-833)."""
        ns_id = uuid4()
        existing = _make_entity(
            "Robert",
            namespace_id=ns_id,
            metadata={"aliases": ["Bob", "Rob"]},
        )
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])
        resolver = EntityResolver(storage, embedder=None)

        result = await resolver.resolve(
            "Someone Else",
            "PERSON",
            ns_id,
            aliases=["Rob"],
        )
        assert result.is_duplicate is True
        assert result.match_type == "alias"


# ---------------------------------------------------------------------------
# Embedding match path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingMatch:
    @pytest.mark.asyncio
    async def test_embedding_match_finds_similar(self) -> None:
        ns_id = uuid4()
        target = _make_entity("Alice Wonderland", "PERSON", namespace_id=ns_id)

        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])
        storage.search_similar_entities = AsyncMock(return_value=[(target.id, 0.92)])
        storage.get_entity = AsyncMock(return_value=target)

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        resolver = EntityResolver(storage, embedder=embedder)
        result = await resolver.resolve(
            "A Wholly Different Name",
            "PERSON",
            ns_id,
            description="A person from a story",
        )
        assert result.is_duplicate is True
        assert result.match_type == "embedding"
        assert result.match_score == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_embedding_match_handles_exception(self) -> None:
        ns_id = uuid4()
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])
        storage.search_similar_entities = AsyncMock(side_effect=Exception("embed-fail"))

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1, 0.2])
        resolver = EntityResolver(storage, embedder=embedder)

        result = await resolver.resolve(
            "Bob",
            "PERSON",
            ns_id,
            description="someone",
        )
        # No candidates returned → not a duplicate
        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_embedding_match_skips_wrong_type(self) -> None:
        ns_id = uuid4()
        other_type = _make_entity("Acme", "ORGANIZATION", namespace_id=ns_id)
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[])
        storage.search_similar_entities = AsyncMock(return_value=[(other_type.id, 0.9)])
        storage.get_entity = AsyncMock(return_value=other_type)

        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1])
        resolver = EntityResolver(storage, embedder=embedder)

        result = await resolver.resolve(
            "Alice",
            "PERSON",
            ns_id,
            description="someone",
        )
        # Type mismatch — not selected as candidate
        assert result.is_duplicate is False


# ---------------------------------------------------------------------------
# with_preloaded_cache failure path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPreloadedCacheFailures:
    @pytest.mark.asyncio
    async def test_preload_with_failing_storage(self) -> None:
        """Storage failure during preload yields empty list, not raise."""
        storage = MagicMock()
        storage.list_entities = AsyncMock(side_effect=Exception("nope"))
        resolver = await EntityResolver.with_preloaded_cache(
            storage,
            embedder=None,
            namespace_id=uuid4(),
            entity_types=["PERSON", "ORGANIZATION"],
        )
        # All types have empty lists in cache
        assert all(v == [] for v in resolver._entity_cache.values())


# ---------------------------------------------------------------------------
# resolve_and_merge_entity provenance and attribute paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveAndMergeProvenance:
    @pytest.mark.asyncio
    async def test_merges_attributes_not_overwriting(self) -> None:
        ns_id = uuid4()
        existing = _make_entity(
            "Alice",
            "PERSON",
            namespace_id=ns_id,
            attributes={"email": "alice@old.com"},
        )
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])

        entity, is_new = await resolve_and_merge_entity(
            "Alice",
            "PERSON",
            ns_id,
            storage,
            attributes={"email": "alice@new.com", "phone": "555-0100"},
        )
        assert is_new is False
        # Existing email is preserved
        assert entity.attributes["email"] == "alice@old.com"
        # New attribute added
        assert entity.attributes["phone"] == "555-0100"

    @pytest.mark.asyncio
    async def test_increments_mention_count(self) -> None:
        ns_id = uuid4()
        existing = _make_entity("Alice", "PERSON", namespace_id=ns_id)
        initial = existing.mention_count
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])
        entity, is_new = await resolve_and_merge_entity("Alice", "PERSON", ns_id, storage)
        assert is_new is False
        assert entity.mention_count == initial + 1

    @pytest.mark.asyncio
    async def test_tracks_source_docs_and_chunks(self) -> None:
        ns_id = uuid4()
        existing = _make_entity("Alice", "PERSON", namespace_id=ns_id)
        doc_id = uuid4()
        chunk_id = uuid4()
        storage = MagicMock()
        storage.list_entities = AsyncMock(return_value=[existing])
        entity, _ = await resolve_and_merge_entity(
            "Alice",
            "PERSON",
            ns_id,
            storage,
            source_document_id=doc_id,
            source_chunk_id=chunk_id,
            source_tool="slack",
        )
        assert doc_id in entity.source_document_ids
        assert chunk_id in entity.source_chunk_ids
        # Provenance recorded
        assert "merge_sources" in entity.metadata
        assert entity.metadata["merge_sources"][-1]["source_tool"] == "slack"
