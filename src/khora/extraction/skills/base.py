"""Base extraction skill definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionSkill:
    """Configurable extraction skill definition.

    Skills define what types of entities and relationships to extract
    from documents. They can be customized per namespace or document type.
    """

    name: str
    description: str = ""

    # Entity extraction configuration
    entity_types: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)

    # Custom extraction prompt (optional)
    custom_prompt: str | None = None

    # Processing configuration
    extract_entities: bool = True
    extract_relationships: bool = True

    # Confidence thresholds
    min_entity_confidence: float = 0.5
    min_relationship_confidence: float = 0.5

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def general_entities(cls) -> ExtractionSkill:
        """Create a general entity extraction skill."""
        return cls(
            name="general_entities",
            description="Extract general entities like people, organizations, and concepts",
            entity_types=["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION"],
            relationship_types=["WORKS_FOR", "KNOWS", "RELATES_TO", "LOCATED_IN"],
        )

    @classmethod
    def technical_docs(cls) -> ExtractionSkill:
        """Create a skill for technical documentation."""
        return cls(
            name="technical_docs",
            description="Extract technical entities from documentation",
            entity_types=["TECHNOLOGY", "CONCEPT", "PRODUCT", "ORGANIZATION"],
            relationship_types=["DEPENDS_ON", "IMPLEMENTS", "PART_OF", "RELATES_TO"],
        )

    @classmethod
    def business_intel(cls) -> ExtractionSkill:
        """Create a skill for business intelligence."""
        return cls(
            name="business_intel",
            description="Extract business entities and relationships",
            entity_types=["PERSON", "ORGANIZATION", "PRODUCT", "EVENT", "LOCATION"],
            relationship_types=[
                "WORKS_FOR",
                "MANAGES",
                "OWNS",
                "COMPETES_WITH",
                "PARTNERS_WITH",
                "HEADQUARTERED_IN",
            ],
        )

    @classmethod
    def research_papers(cls) -> ExtractionSkill:
        """Create a skill for research papers."""
        return cls(
            name="research_papers",
            description="Extract entities from academic research",
            entity_types=["PERSON", "ORGANIZATION", "CONCEPT", "TECHNOLOGY", "EVENT"],
            relationship_types=[
                "COLLABORATES_WITH",
                "DERIVED_FROM",
                "IMPLEMENTS",
                "RELATES_TO",
                "PRECEDES",
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert skill to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "entity_types": self.entity_types,
            "relationship_types": self.relationship_types,
            "custom_prompt": self.custom_prompt,
            "extract_entities": self.extract_entities,
            "extract_relationships": self.extract_relationships,
            "min_entity_confidence": self.min_entity_confidence,
            "min_relationship_confidence": self.min_relationship_confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionSkill:
        """Create skill from dictionary."""
        return cls(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            entity_types=data.get("entity_types", []),
            relationship_types=data.get("relationship_types", []),
            custom_prompt=data.get("custom_prompt"),
            extract_entities=data.get("extract_entities", True),
            extract_relationships=data.get("extract_relationships", True),
            min_entity_confidence=data.get("min_entity_confidence", 0.5),
            min_relationship_confidence=data.get("min_relationship_confidence", 0.5),
            metadata=data.get("metadata", {}),
        )
