"""Entity resolution for deduplication during extraction.

Resolves and merges entities to avoid duplicates in the knowledge graph.
Uses multiple strategies:
- Exact name match
- Alias matching
- Attribute matching (email, domain, coordinates, etc.)
- Embedding similarity
- Fuzzy name matching
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora._accel import (
    levenshtein_similarity,
    sequence_match_ratio,
)
from khora.core.models.entity import entity_type_str

if TYPE_CHECKING:
    from khora.core.models import Entity
    from khora.extraction.embedders import Embedder
    from khora.storage import StorageCoordinator


# ---------------------------------------------------------------------------
# Per-type merge thresholds
# ---------------------------------------------------------------------------

DEFAULT_MERGE_THRESHOLDS: dict[str, float] = {
    "PERSON": 0.92,  # High - avoid merging different people
    "ORGANIZATION": 0.88,  # Medium-high - companies have unique names
    "LOCATION": 0.85,  # Medium - locations can have aliases
    "CONCEPT": 0.82,  # Moderate — prevents over-merging distinct concepts while allowing fuzzy matches
    "EVENT": 0.80,  # Medium - events have specific names
    "TECHNOLOGY": 0.85,  # Medium - tech names are specific
    "PRODUCT": 0.85,  # Medium - product names are specific
    "DATE": 0.95,  # Very high - dates are precise
}

DEFAULT_THRESHOLD = 0.85  # Fallback for unknown types


# ---------------------------------------------------------------------------
# Resolution metrics for quality tracking
# ---------------------------------------------------------------------------


@dataclass
class ResolutionMetrics:
    """Metrics for entity resolution quality tracking.

    Tracks resolution outcomes to monitor deduplication quality:
    - Match rates by type and strategy
    - Confidence distribution
    - Potential false positive detection (low-confidence merges)
    """

    total_resolutions: int = 0
    exact_matches: int = 0
    alias_matches: int = 0
    attribute_matches: int = 0
    fuzzy_matches: int = 0
    embedding_matches: int = 0
    new_entities: int = 0

    # Per-type stats
    matches_by_type: dict[str, int] = field(default_factory=dict)
    merges_by_type: dict[str, int] = field(default_factory=dict)

    # Confidence tracking
    total_confidence: float = 0.0
    low_confidence_merges: int = 0  # Merges below 0.9 confidence

    def record_resolution(
        self,
        entity_type: str,
        match_type: str,
        score: float,
        merged: bool,
    ) -> None:
        """Record a resolution event for metrics.

        Args:
            entity_type: Type of entity being resolved
            match_type: Strategy that matched (exact, alias, attribute, fuzzy, embedding)
            score: Match confidence score
            merged: Whether the entity was merged
        """
        self.total_resolutions += 1
        self.matches_by_type[entity_type] = self.matches_by_type.get(entity_type, 0) + 1

        if match_type == "exact":
            self.exact_matches += 1
        elif match_type == "alias":
            self.alias_matches += 1
        elif match_type == "attribute":
            self.attribute_matches += 1
        elif match_type == "fuzzy":
            self.fuzzy_matches += 1
        elif match_type == "embedding":
            self.embedding_matches += 1
        elif match_type == "":
            self.new_entities += 1

        if merged:
            self.merges_by_type[entity_type] = self.merges_by_type.get(entity_type, 0) + 1
            self.total_confidence += score
            if score < 0.9:
                self.low_confidence_merges += 1

    @property
    def average_merge_confidence(self) -> float:
        """Average confidence of merged entities."""
        total_merges = sum(self.merges_by_type.values())
        if total_merges == 0:
            return 0.0
        return self.total_confidence / total_merges

    @property
    def match_rate_by_type(self) -> dict[str, float]:
        """Match rate (merges / total) by entity type."""
        rates = {}
        for entity_type, total in self.matches_by_type.items():
            merges = self.merges_by_type.get(entity_type, 0)
            rates[entity_type] = merges / total if total > 0 else 0.0
        return rates

    def log_summary(self) -> None:
        """Log a summary of resolution metrics."""
        if self.total_resolutions == 0:
            return

        total_matches = (
            self.exact_matches
            + self.alias_matches
            + self.attribute_matches
            + self.fuzzy_matches
            + self.embedding_matches
        )
        logger.info(
            f"Entity resolution summary: {self.total_resolutions} resolutions, "
            f"{total_matches} matches ({total_matches / self.total_resolutions * 100:.1f}%), "
            f"{self.new_entities} new entities"
        )
        logger.debug(
            f"Match types: exact={self.exact_matches}, alias={self.alias_matches}, "
            f"attribute={self.attribute_matches}, fuzzy={self.fuzzy_matches}, "
            f"embedding={self.embedding_matches}"
        )
        if self.average_merge_confidence > 0:
            logger.debug(
                f"Average merge confidence: {self.average_merge_confidence:.2f}, "
                f"low-confidence merges: {self.low_confidence_merges}"
            )


# ---------------------------------------------------------------------------
# Merge provenance tracking
# ---------------------------------------------------------------------------


@dataclass
class MergeSource:
    """A source that contributed to a merged entity.

    Tracks provenance information for merged entities to understand
    which sources contributed to the final entity.
    """

    entity_id: str
    score: float
    match_type: str
    source_tool: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage in metadata."""
        return {
            "entity_id": self.entity_id,
            "score": self.score,
            "match_type": self.match_type,
            "source_tool": self.source_tool,
        }


# ---------------------------------------------------------------------------
# Attribute matching helpers
# ---------------------------------------------------------------------------


def _normalize_email(email: str | None) -> str | None:
    """Normalize email address for comparison."""
    if not email:
        return None
    return email.lower().strip()


def _normalize_domain(domain: str | None) -> str | None:
    """Normalize domain/website for comparison.

    Strips protocol, www prefix, and trailing slashes.
    """
    if not domain:
        return None
    domain = domain.lower().strip()
    # Remove protocol
    domain = re.sub(r"^https?://", "", domain)
    # Remove www prefix
    domain = re.sub(r"^www\.", "", domain)
    # Remove trailing slash
    domain = domain.rstrip("/")
    return domain


def _parse_coordinates(coords: str | None) -> tuple[float, float] | None:
    """Parse coordinate string into (lat, lon) tuple.

    Accepts formats like:
    - "40.7128, -74.0060"
    - "40.7128,-74.0060"
    - "(40.7128, -74.0060)"
    """
    if not coords:
        return None
    try:
        # Remove parentheses and extra whitespace
        coords = coords.strip("()[] ").replace(" ", "")
        parts = coords.split(",")
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    except (ValueError, IndexError):
        pass
    return None


def _coordinates_distance(
    coords1: tuple[float, float],
    coords2: tuple[float, float],
) -> float:
    """Compute approximate distance between coordinates in kilometers.

    Uses Haversine formula for spherical Earth approximation.
    """
    import math

    lat1, lon1 = coords1
    lat2, lon2 = coords2

    # Earth radius in km
    R = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


@dataclass
class ResolutionCandidate:
    """A candidate entity for resolution."""

    entity: Entity
    match_type: str  # exact, alias, fuzzy, embedding
    score: float


@dataclass
class ResolutionResult:
    """Result of entity resolution."""

    is_duplicate: bool
    existing_entity: Entity | None = None
    match_type: str = ""
    match_score: float = 0.0
    should_merge: bool = False


class EntityResolver:
    """Resolves new entities against existing ones.

    Before creating a new entity, use this resolver to check
    if an equivalent entity already exists.

    Features:
    - Multiple matching strategies (exact, alias, attribute, fuzzy, embedding)
    - Per-type merge thresholds for improved accuracy
    - Attribute-level matching (email, domain, coordinates, etc.)
    - Resolution metrics for quality tracking
    - Merge provenance tracking
    """

    # Default entity types to pre-load when using with_preloaded_cache
    DEFAULT_ENTITY_TYPES: list[str] = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "CONCEPT",
        "EVENT",
        "PRODUCT",
        "TECHNOLOGY",
    ]

    def __init__(
        self,
        storage: StorageCoordinator,
        embedder: Embedder | None = None,
        *,
        exact_match: bool = True,
        alias_match: bool = True,
        attribute_match: bool = True,
        fuzzy_match: bool = True,
        embedding_match: bool = True,
        fuzzy_threshold: float = 0.85,
        embedding_threshold: float = 0.85,
        merge_thresholds: dict[str, float] | None = None,
        track_metrics: bool = False,
    ) -> None:
        """Initialize the entity resolver.

        Args:
            storage: Storage coordinator for entity access
            embedder: Embedder for semantic matching
            exact_match: Enable exact name matching
            alias_match: Enable alias matching
            attribute_match: Enable attribute-level matching
            fuzzy_match: Enable fuzzy string matching
            embedding_match: Enable embedding-based matching
            fuzzy_threshold: Minimum fuzzy match ratio
            embedding_threshold: Minimum embedding similarity
            merge_thresholds: Per-type merge thresholds (defaults to DEFAULT_MERGE_THRESHOLDS)
            track_metrics: Enable resolution metrics collection
        """
        self._storage = storage
        self._embedder = embedder
        self._exact_match = exact_match
        self._alias_match = alias_match
        self._attribute_match = attribute_match
        self._fuzzy_match = fuzzy_match
        self._embedding_match = embedding_match
        self._fuzzy_threshold = fuzzy_threshold
        self._embedding_threshold = embedding_threshold

        # Per-type merge thresholds
        self._merge_thresholds = merge_thresholds or DEFAULT_MERGE_THRESHOLDS.copy()
        self._default_threshold = DEFAULT_THRESHOLD

        # Metrics tracking
        self._track_metrics = track_metrics
        self._metrics = ResolutionMetrics() if track_metrics else None

        # Cache for performance
        self._entity_cache: dict[str, list[Entity]] = {}

    def _get_merge_threshold(self, entity_type: str) -> float:
        """Get merge threshold for entity type.

        Args:
            entity_type: Entity type string (e.g., "PERSON", "ORGANIZATION")

        Returns:
            Merge threshold for the type, or default if not configured
        """
        return self._merge_thresholds.get(
            entity_type.upper(),
            self._default_threshold,
        )

    def _compute_attribute_similarity(
        self,
        entity1_attrs: dict[str, Any],
        entity2_attrs: dict[str, Any],
        entity_type: str,
    ) -> float | None:
        """Compute similarity based on entity attributes.

        Matching rules by entity type:
        - PERSON: email, phone, title similarity
        - ORGANIZATION: domain, industry similarity
        - LOCATION: coordinates proximity, address similarity
        - TECHNOLOGY: version compatibility, vendor match

        Args:
            entity1_attrs: Attributes of first entity
            entity2_attrs: Attributes of second entity
            entity_type: Type of entities being compared

        Returns:
            Score in [0, 1] or None if no comparable attributes
        """
        entity_type_upper = entity_type.upper()

        if entity_type_upper == "PERSON":
            return self._compute_person_attribute_similarity(entity1_attrs, entity2_attrs)
        elif entity_type_upper == "ORGANIZATION":
            return self._compute_organization_attribute_similarity(entity1_attrs, entity2_attrs)
        elif entity_type_upper == "LOCATION":
            return self._compute_location_attribute_similarity(entity1_attrs, entity2_attrs)
        elif entity_type_upper == "TECHNOLOGY":
            return self._compute_technology_attribute_similarity(entity1_attrs, entity2_attrs)
        elif entity_type_upper == "PRODUCT":
            return self._compute_product_attribute_similarity(entity1_attrs, entity2_attrs)

        return None

    def _compute_person_attribute_similarity(
        self,
        attrs1: dict[str, Any],
        attrs2: dict[str, Any],
    ) -> float | None:
        """Compute PERSON attribute similarity.

        Checks email (exact), phone, and title similarity.
        """
        scores: list[float] = []

        # Email - exact match is strong signal
        email1 = _normalize_email(attrs1.get("email"))
        email2 = _normalize_email(attrs2.get("email"))
        if email1 and email2:
            if email1 == email2:
                return 1.0  # Exact email match is definitive
            else:
                # Different emails for same name - likely different people
                return 0.3

        # Title similarity
        title1 = attrs1.get("title", "")
        title2 = attrs2.get("title", "")
        if title1 and title2:
            title_sim = levenshtein_similarity(title1.lower(), title2.lower())
            scores.append(title_sim * 0.7)  # Title is a weaker signal

        # Organization match
        org1 = attrs1.get("organization", "")
        org2 = attrs2.get("organization", "")
        if org1 and org2:
            org_sim = levenshtein_similarity(org1.lower(), org2.lower())
            scores.append(org_sim * 0.8)

        if not scores:
            return None

        return sum(scores) / len(scores)

    def _compute_organization_attribute_similarity(
        self,
        attrs1: dict[str, Any],
        attrs2: dict[str, Any],
    ) -> float | None:
        """Compute ORGANIZATION attribute similarity.

        Checks domain/website and industry.
        """
        scores: list[float] = []

        # Domain/website - exact match is strong signal
        domain1 = _normalize_domain(attrs1.get("website") or attrs1.get("domain"))
        domain2 = _normalize_domain(attrs2.get("website") or attrs2.get("domain"))
        if domain1 and domain2:
            if domain1 == domain2:
                return 1.0  # Same domain is definitive
            else:
                # Check for partial domain match (e.g., "google.com" vs "cloud.google.com")
                if domain1 in domain2 or domain2 in domain1:
                    scores.append(0.8)
                else:
                    return 0.3  # Different domains - likely different orgs

        # Industry match
        industry1 = attrs1.get("industry", "")
        industry2 = attrs2.get("industry", "")
        if industry1 and industry2:
            industry_sim = levenshtein_similarity(industry1.lower(), industry2.lower())
            scores.append(industry_sim * 0.6)  # Industry is supporting evidence

        # Type match (company, nonprofit, etc.)
        type1 = attrs1.get("type", "")
        type2 = attrs2.get("type", "")
        if type1 and type2:
            if type1.lower() == type2.lower():
                scores.append(0.7)

        if not scores:
            return None

        return sum(scores) / len(scores)

    def _compute_location_attribute_similarity(
        self,
        attrs1: dict[str, Any],
        attrs2: dict[str, Any],
    ) -> float | None:
        """Compute LOCATION attribute similarity.

        Checks coordinates proximity and address similarity.
        """
        scores: list[float] = []

        # Coordinates proximity
        coords1 = _parse_coordinates(attrs1.get("coordinates"))
        coords2 = _parse_coordinates(attrs2.get("coordinates"))
        if coords1 and coords2:
            distance = _coordinates_distance(coords1, coords2)
            # Within 1km is very likely same place
            if distance < 1.0:
                return 1.0
            elif distance < 5.0:
                scores.append(0.9)
            elif distance < 20.0:
                scores.append(0.7)
            elif distance < 100.0:
                scores.append(0.4)
            else:
                return 0.2  # Too far apart

        # Address similarity
        addr1 = attrs1.get("address", "")
        addr2 = attrs2.get("address", "")
        if addr1 and addr2:
            addr_sim = levenshtein_similarity(addr1.lower(), addr2.lower())
            scores.append(addr_sim)

        # Country match
        country1 = attrs1.get("country", "")
        country2 = attrs2.get("country", "")
        if country1 and country2:
            if country1.lower() == country2.lower():
                scores.append(0.6)
            else:
                # Different countries is a negative signal
                scores.append(0.2)

        if not scores:
            return None

        return sum(scores) / len(scores)

    def _compute_technology_attribute_similarity(
        self,
        attrs1: dict[str, Any],
        attrs2: dict[str, Any],
    ) -> float | None:
        """Compute TECHNOLOGY attribute similarity.

        Checks version compatibility and vendor match.
        """
        scores: list[float] = []

        # Vendor match - same vendor is strong signal
        vendor1 = attrs1.get("vendor", "")
        vendor2 = attrs2.get("vendor", "")
        if vendor1 and vendor2:
            vendor_sim = levenshtein_similarity(vendor1.lower(), vendor2.lower())
            if vendor_sim > 0.9:
                scores.append(0.9)
            elif vendor_sim > 0.7:
                scores.append(0.7)
            else:
                # Different vendors - likely different tech
                return 0.3

        # Type match (language, framework, etc.)
        type1 = attrs1.get("type", "")
        type2 = attrs2.get("type", "")
        if type1 and type2:
            if type1.lower() == type2.lower():
                scores.append(0.8)
            else:
                scores.append(0.4)

        # Version - same major version is compatible
        version1 = attrs1.get("version", "")
        version2 = attrs2.get("version", "")
        if version1 and version2:
            # Extract major version
            major1 = version1.split(".")[0] if version1 else ""
            major2 = version2.split(".")[0] if version2 else ""
            if major1 == major2:
                scores.append(0.8)
            else:
                scores.append(0.5)  # Different versions still same tech

        if not scores:
            return None

        return sum(scores) / len(scores)

    def _compute_product_attribute_similarity(
        self,
        attrs1: dict[str, Any],
        attrs2: dict[str, Any],
    ) -> float | None:
        """Compute PRODUCT attribute similarity.

        Checks vendor and category.
        """
        scores: list[float] = []

        # Vendor match
        vendor1 = attrs1.get("vendor", "")
        vendor2 = attrs2.get("vendor", "")
        if vendor1 and vendor2:
            vendor_sim = levenshtein_similarity(vendor1.lower(), vendor2.lower())
            if vendor_sim > 0.9:
                scores.append(0.9)
            else:
                # Different vendors - could still be same product (acquisitions, etc.)
                scores.append(vendor_sim * 0.6)

        # Category match
        cat1 = attrs1.get("category", "")
        cat2 = attrs2.get("category", "")
        if cat1 and cat2:
            cat_sim = levenshtein_similarity(cat1.lower(), cat2.lower())
            scores.append(cat_sim * 0.7)

        if not scores:
            return None

        return sum(scores) / len(scores)

    @property
    def metrics(self) -> ResolutionMetrics | None:
        """Get resolution metrics if tracking is enabled."""
        return self._metrics

    @classmethod
    async def with_preloaded_cache(
        cls,
        storage: StorageCoordinator,
        embedder: Embedder | None = None,
        namespace_id: UUID | None = None,
        entity_types: list[str] | None = None,
        *,
        exact_match: bool = True,
        alias_match: bool = True,
        attribute_match: bool = True,
        fuzzy_match: bool = True,
        embedding_match: bool = True,
        fuzzy_threshold: float = 0.85,
        embedding_threshold: float = 0.85,
        merge_thresholds: dict[str, float] | None = None,
        track_metrics: bool = False,
        max_entities_per_type: int = 5000,
    ) -> EntityResolver:
        """Create resolver with pre-loaded entity cache for batch operations.

        This is useful when processing many documents in a batch, as it avoids
        repeated entity list fetches. The cache is pre-warmed with entities
        for all specified types (or default types if none specified).

        Args:
            storage: Storage coordinator for entity access
            embedder: Embedder for semantic matching
            namespace_id: Namespace to pre-load entities from
            entity_types: Entity types to pre-load (defaults to common types)
            exact_match: Enable exact name matching
            alias_match: Enable alias matching
            attribute_match: Enable attribute-level matching
            fuzzy_match: Enable fuzzy string matching
            embedding_match: Enable embedding-based matching
            fuzzy_threshold: Minimum fuzzy match ratio
            embedding_threshold: Minimum embedding similarity
            merge_thresholds: Per-type merge thresholds
            track_metrics: Enable resolution metrics collection
            max_entities_per_type: Maximum entities to load per type

        Returns:
            EntityResolver with pre-populated cache
        """
        resolver = cls(
            storage,
            embedder,
            exact_match=exact_match,
            alias_match=alias_match,
            attribute_match=attribute_match,
            fuzzy_match=fuzzy_match,
            embedding_match=embedding_match,
            fuzzy_threshold=fuzzy_threshold,
            embedding_threshold=embedding_threshold,
            merge_thresholds=merge_thresholds,
            track_metrics=track_metrics,
        )

        if namespace_id is None:
            return resolver

        types_to_load = entity_types or cls.DEFAULT_ENTITY_TYPES

        # Load entities for all types in parallel for efficiency
        import asyncio

        async def load_type(entity_type: str) -> tuple[str, list[Entity]]:
            try:
                entities = await storage.list_entities(
                    namespace_id,
                    entity_type=entity_type,
                    limit=max_entities_per_type,
                )
                return entity_type, entities
            except Exception as e:
                logger.debug(f"Failed to preload entities for type {entity_type}: {e}")
                return entity_type, []

        results = await asyncio.gather(*[load_type(t) for t in types_to_load])

        total_loaded = 0
        for entity_type, entities in results:
            cache_key = f"{namespace_id}:{entity_type}"
            resolver._entity_cache[cache_key] = entities
            total_loaded += len(entities)

        if total_loaded > 0:
            logger.debug(
                f"Pre-loaded {total_loaded} entities across {len(types_to_load)} types for namespace {namespace_id}"
            )

        return resolver

    def share_cache_from(self, other: EntityResolver) -> None:
        """Share entity cache from another resolver instance.

        This allows multiple resolvers to share a pre-warmed cache,
        useful when processing documents in parallel.

        Args:
            other: Another EntityResolver to copy cache from
        """
        self._entity_cache = other._entity_cache

    def add_to_cache(self, namespace_id: UUID, entity: Entity) -> None:
        """Add a newly created entity to the cache.

        This keeps the cache in sync when new entities are created,
        avoiding the need to re-fetch from storage.

        Args:
            namespace_id: Namespace the entity belongs to
            entity: Entity to add to cache
        """
        from khora.core.models.entity import entity_type_str

        entity_type = entity_type_str(entity.entity_type)
        cache_key = f"{namespace_id}:{entity_type}"

        if cache_key in self._entity_cache:
            # Check if entity already exists in cache (by ID)
            existing_ids = {e.id for e in self._entity_cache[cache_key]}
            if entity.id not in existing_ids:
                self._entity_cache[cache_key].append(entity)
        else:
            self._entity_cache[cache_key] = [entity]

    async def resolve(
        self,
        name: str,
        entity_type: str,
        namespace_id: UUID,
        *,
        description: str = "",
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> ResolutionResult:
        """Resolve a potential entity against existing entities.

        Args:
            name: Entity name
            entity_type: Entity type (PERSON, ORGANIZATION, etc.)
            namespace_id: Namespace to check
            description: Optional description for semantic matching
            aliases: Optional aliases for the entity
            attributes: Optional attributes for attribute-level matching

        Returns:
            ResolutionResult indicating if entity is a duplicate
        """
        aliases = aliases or []
        attributes = attributes or {}
        candidates: list[ResolutionCandidate] = []

        # Get per-type merge threshold
        merge_threshold = self._get_merge_threshold(entity_type)

        # Load entities of same type from cache or storage
        cache_key = f"{namespace_id}:{entity_type}"
        if cache_key not in self._entity_cache:
            try:
                entities = await self._storage.list_entities(
                    namespace_id,
                    entity_type=entity_type,
                    limit=1000,
                )
                self._entity_cache[cache_key] = entities
            except Exception as e:
                logger.debug(f"Failed to load entities for resolution: {e}")
                self._entity_cache[cache_key] = []

        existing_entities = self._entity_cache.get(cache_key, [])

        # 1. Exact name match
        if self._exact_match:
            for entity in existing_entities:
                if entity.name.lower() == name.lower():
                    result = ResolutionResult(
                        is_duplicate=True,
                        existing_entity=entity,
                        match_type="exact",
                        match_score=1.0,
                        should_merge=True,
                    )
                    self._record_metrics(entity_type, "exact", 1.0, True)
                    return result

        # 2. Alias match
        if self._alias_match:
            name_lower = name.lower()
            aliases_lower = [a.lower() for a in aliases]

            for entity in existing_entities:
                # Check if name matches any existing alias
                entity_aliases = entity.metadata.get("aliases", [])
                entity_aliases_lower = [a.lower() for a in entity_aliases]

                if name_lower in entity_aliases_lower:
                    candidates.append(ResolutionCandidate(entity, "alias", 0.95))
                    continue

                # Check if any new alias matches existing name or alias
                if entity.name.lower() in aliases_lower:
                    candidates.append(ResolutionCandidate(entity, "alias", 0.95))
                    continue

                for alias in aliases_lower:
                    if alias in entity_aliases_lower:
                        candidates.append(ResolutionCandidate(entity, "alias", 0.9))
                        break

        # 3. Attribute-level match
        if self._attribute_match and attributes:
            for entity in existing_entities:
                # Skip if already matched
                if entity in [c.entity for c in candidates]:
                    continue

                # Compute attribute similarity
                attr_score = self._compute_attribute_similarity(
                    attributes,
                    entity.attributes,
                    entity_type,
                )
                if attr_score is not None and attr_score >= merge_threshold:
                    candidates.append(ResolutionCandidate(entity, "attribute", attr_score))

        # 4. Fuzzy name match
        if self._fuzzy_match:
            name_lower = name.lower()
            for entity in existing_entities:
                # Skip if already matched
                if entity in [c.entity for c in candidates]:
                    continue

                ratio = sequence_match_ratio(name_lower, entity.name.lower())
                if ratio >= self._fuzzy_threshold:
                    candidates.append(ResolutionCandidate(entity, "fuzzy", ratio))

        # 5. Embedding similarity match
        if self._embedding_match and self._embedder and description:
            try:
                # Create text for embedding
                search_text = f"{entity_type}: {name}. {description}"
                embedding = await self._embedder.embed(search_text)

                # Search for similar entities
                results = await self._storage.search_similar_entities(
                    namespace_id,
                    embedding,
                    limit=5,
                    min_similarity=self._embedding_threshold,
                )

                for entity_id, score in results:
                    entity = await self._storage.get_entity(entity_id)
                    if entity and entity_type_str(entity.entity_type) == entity_type:
                        # Skip if already matched
                        if entity in [c.entity for c in candidates]:
                            continue
                        candidates.append(ResolutionCandidate(entity, "embedding", score))

            except Exception as e:
                logger.debug(f"Embedding match failed: {e}")

        # Select best match
        if candidates:
            candidates.sort(key=lambda c: c.score, reverse=True)
            best = candidates[0]

            # Use per-type threshold for merge decision
            should_merge = best.score >= merge_threshold

            self._record_metrics(entity_type, best.match_type, best.score, should_merge)

            return ResolutionResult(
                is_duplicate=True,
                existing_entity=best.entity,
                match_type=best.match_type,
                match_score=best.score,
                should_merge=should_merge,
            )

        # No match found - this is a new entity
        self._record_metrics(entity_type, "", 0.0, False)
        return ResolutionResult(is_duplicate=False)

    def _record_metrics(
        self,
        entity_type: str,
        match_type: str,
        score: float,
        merged: bool,
    ) -> None:
        """Record resolution metrics if tracking is enabled."""
        if self._metrics is not None:
            self._metrics.record_resolution(entity_type, match_type, score, merged)

    def invalidate_cache(self, namespace_id: UUID | None = None) -> None:
        """Invalidate the entity cache.

        Args:
            namespace_id: Optional namespace to invalidate.
                         If None, invalidates all.
        """
        if namespace_id is None:
            self._entity_cache.clear()
        else:
            keys_to_remove = [k for k in self._entity_cache if k.startswith(str(namespace_id))]
            for key in keys_to_remove:
                del self._entity_cache[key]


async def resolve_and_merge_entity(
    name: str,
    entity_type: str,
    namespace_id: UUID,
    storage: StorageCoordinator,
    embedder: Embedder | None = None,
    *,
    description: str = "",
    aliases: list[str] | None = None,
    attributes: dict[str, Any] | None = None,
    source_document_id: UUID | None = None,
    source_chunk_id: UUID | None = None,
    source_tool: str = "",
    resolver: EntityResolver | None = None,
    track_provenance: bool = True,
) -> tuple[Entity, bool]:
    """Resolve and optionally merge an entity.

    Convenience function that resolves an entity and either returns
    the existing entity (merged) or indicates a new entity should be created.

    Args:
        name: Entity name
        entity_type: Entity type
        namespace_id: Namespace ID
        storage: Storage coordinator
        embedder: Optional embedder
        description: Entity description
        aliases: Entity aliases
        attributes: Entity attributes
        source_document_id: Source document ID
        source_chunk_id: Source chunk ID
        source_tool: Source tool that produced this entity (e.g., "slack", "gmail")
        resolver: Optional pre-configured EntityResolver (for cache sharing)
        track_provenance: Track merge provenance in metadata (default True)

    Returns:
        Tuple of (entity, is_new) where is_new indicates if a new entity
        should be created (False means use the returned existing entity)
    """
    from khora.core.models import Entity, EntityType

    if resolver is None:
        resolver = EntityResolver(storage, embedder)
    result = await resolver.resolve(
        name,
        entity_type,
        namespace_id,
        description=description,
        aliases=aliases or [],
        attributes=attributes,
    )

    if result.is_duplicate and result.existing_entity and result.should_merge:
        # Merge into existing entity
        existing = result.existing_entity

        # Track merge provenance
        if track_provenance:
            merge_sources = existing.metadata.get("merge_sources", [])
            merge_source = MergeSource(
                entity_id=str(existing.id),
                score=result.match_score,
                match_type=result.match_type,
                source_tool=source_tool,
            )
            merge_sources.append(merge_source.to_dict())
            existing.metadata["merge_sources"] = merge_sources

        # Update source tracking
        if source_document_id and source_document_id not in existing.source_document_ids:
            existing.source_document_ids.append(source_document_id)
        if source_chunk_id and source_chunk_id not in existing.source_chunk_ids:
            existing.source_chunk_ids.append(source_chunk_id)

        # Increment mention count
        existing.mention_count += 1

        # Merge attributes (prefer existing values, add new ones)
        if attributes:
            for key, value in attributes.items():
                if key not in existing.attributes:
                    existing.attributes[key] = value

        # Merge aliases
        existing_aliases = existing.metadata.get("aliases", [])
        if aliases:
            for alias in aliases:
                if alias not in existing_aliases:
                    existing_aliases.append(alias)
            existing.metadata["aliases"] = existing_aliases

        # Update description if empty
        if not existing.description and description:
            existing.description = description

        logger.debug(
            f"Merged entity '{name}' with existing '{existing.name}' "
            f"(match: {result.match_type}, score: {result.match_score:.2f})"
        )

        return existing, False

    # Create new entity
    try:
        etype: EntityType | str = EntityType(entity_type.upper())
    except ValueError:
        etype = entity_type.upper() or "CONCEPT"

    new_entity = Entity(
        namespace_id=namespace_id,
        name=name,
        entity_type=etype,
        description=description,
        attributes=attributes or {},
        source_tool=source_tool,
        source_document_ids=[source_document_id] if source_document_id else [],
        source_chunk_ids=[source_chunk_id] if source_chunk_id else [],
        mention_count=1,
        metadata={"aliases": aliases or []},
    )

    return new_entity, True
