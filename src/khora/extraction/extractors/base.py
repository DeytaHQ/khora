"""Base extractor protocol and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""

    name: str
    entity_type: str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

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


@dataclass
class ExtractionResult:
    """Result of entity extraction from text."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    async def extract(
        self,
        text: str,
        *,
        entity_types: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from
            entity_types: Optional list of entity types to extract

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
    ) -> list[ExtractionResult]:
        """Extract from multiple texts.

        Args:
            texts: List of texts to extract from
            entity_types: Optional list of entity types to extract

        Returns:
            List of ExtractionResult objects
        """
        ...
