"""Base extractor protocol and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from khora.extraction.skills import ExpertiseConfig


@dataclass
class TemporalInfo:
    """Temporal information for entities, relationships, or events."""

    mentioned_at: str | None = None  # When mentioned in context
    occurred_at: str | None = None  # When event occurred
    valid_from: str | None = None  # Start of validity period
    valid_until: str | None = None  # End of validity period


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""

    name: str
    entity_type: str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    # Aliases for entity resolution
    aliases: list[str] = field(default_factory=list)

    # Temporal information
    temporal: TemporalInfo | None = None

    # Source provenance — canonical SaaS tool that produced this entity
    source_tool: str = ""

    # Source tracking
    source_text: str = ""
    start_char: int = 0
    end_char: int = 0


@dataclass
class ExtractedRelationship:
    """A relationship extracted from text."""

    source_entity: str
    target_entity: str
    relationship_type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    # Temporal information
    temporal: TemporalInfo | None = None


@dataclass
class ExtractedEvent:
    """An event extracted from text."""

    description: str
    event_type: str = "EVENT"
    occurred_at: str | None = None
    participants: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    """Result of entity extraction from text."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    events: list[ExtractedEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def sanitize_extraction_result(result: ExtractionResult) -> None:
    """Strip NUL bytes (#1528) from every text field of ``result`` in place.

    Extracted text originates from untrusted document content (PDFs, scraped
    HTML, OCR output) and may carry ``0x00``, which PostgreSQL text/jsonb
    columns reject. Mutating the result before it is staged into the storage
    models ensures every backend receives clean data and that entity-name
    matching (dedup keys, relationship endpoints) stays consistent.

    ``strip_nul_json`` is used for every field because extractor output is
    loosely typed: a text field may legitimately be ``None`` (not just the
    ``""`` default), and the JSON-aware helper passes non-string values
    through untouched instead of raising.
    """
    from khora.core.text import strip_nul_json

    for entity in result.entities:
        entity.name = strip_nul_json(entity.name)
        entity.description = strip_nul_json(entity.description)
        entity.attributes = strip_nul_json(entity.attributes)
        entity.aliases = strip_nul_json(entity.aliases)
    for rel in result.relationships:
        rel.source_entity = strip_nul_json(rel.source_entity)
        rel.target_entity = strip_nul_json(rel.target_entity)
        rel.relationship_type = strip_nul_json(rel.relationship_type)
        rel.description = strip_nul_json(rel.description)
        rel.properties = strip_nul_json(rel.properties)
    for event in result.events:
        event.description = strip_nul_json(event.description)
        event.participants = strip_nul_json(event.participants)


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from
            entity_types: Optional list of entity types to extract
            relationship_types: Optional list of relationship types to extract
            expertise: Optional ExpertiseConfig for domain-specific extraction
            context: Optional context dict for prompt template rendering

        Returns:
            ExtractionResult containing entities and relationships
        """
        ...

    @abstractmethod
    async def extract_batch(
        self,
        texts: list[str],
        *,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[ExtractionResult]:
        """Extract from multiple texts.

        Args:
            texts: List of texts to extract from
            entity_types: Optional list of entity types to extract
            relationship_types: Optional list of relationship types to extract
            expertise: Optional ExpertiseConfig for domain-specific extraction
            context: Optional context dict for prompt template rendering

        Returns:
            List of ExtractionResult objects
        """
        ...
