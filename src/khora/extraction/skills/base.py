"""Base extraction skill definition and expertise configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConfidenceLevel(str, Enum):
    """Confidence level thresholds for extraction."""

    HIGH = "high"  # 0.8+
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"  # 0.3-0.5


@dataclass
class EntityTypeConfig:
    """Configurable entity type definition.

    Defines an entity type that the expertise system recognizes,
    including its attributes and identifiers for cross-tool matching.
    """

    name: str
    description: str = ""
    attributes: dict[str, list[str]] = field(default_factory=dict)  # required, optional
    identifiers: list[str] = field(default_factory=list)  # For cross-tool matching
    aliases: list[str] = field(default_factory=list)  # Alternative names for this type

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "attributes": self.attributes,
            "identifiers": self.identifiers,
            "aliases": self.aliases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityTypeConfig:
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            attributes=data.get("attributes", {}),
            identifiers=data.get("identifiers", []),
            aliases=data.get("aliases", []),
        )


@dataclass
class RelationshipTypeConfig:
    """Configurable relationship type definition.

    Defines a relationship type with source and target entity constraints.
    """

    name: str
    description: str = ""
    source_types: list[str] = field(default_factory=list)  # "*" means any
    target_types: list[str] = field(default_factory=list)  # "*" means any
    bidirectional: bool = False
    properties: list[str] = field(default_factory=list)  # Expected properties

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "source_types": self.source_types,
            "target_types": self.target_types,
            "bidirectional": self.bidirectional,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipTypeConfig:
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            source_types=data.get("source_types", []),
            target_types=data.get("target_types", []),
            bidirectional=data.get("bidirectional", False),
            properties=data.get("properties", []),
        )


@dataclass
class CorrelationRule:
    """Rule for cross-tool entity correlation.

    Defines how entities from different tools should be matched and unified.
    """

    name: str
    description: str = ""
    pattern: str | None = None  # Regex pattern for matching references
    match_fields: list[str] = field(default_factory=list)  # Fields to match on (e.g., email)
    entity_types: list[str] = field(default_factory=list)  # Entity types this rule applies to
    creates_relationship: str | None = None  # Relationship type created when matched
    confidence: float = 0.9  # Confidence of matches from this rule

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern,
            "match_fields": self.match_fields,
            "entity_types": self.entity_types,
            "creates_relationship": self.creates_relationship,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrelationRule:
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            pattern=data.get("pattern"),
            match_fields=data.get("match_fields", []),
            entity_types=data.get("entity_types", []),
            creates_relationship=data.get("creates_relationship"),
            confidence=data.get("confidence", 0.9),
        )


@dataclass
class InferenceCondition:
    """Condition for relationship inference rule."""

    relationship: str  # Relationship type to match
    source_type: str | None = None  # Source entity type (optional filter)
    target_type: str | None = None  # Target entity type (optional filter)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "relationship": self.relationship,
            "source_type": self.source_type,
            "target_type": self.target_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceCondition:
        """Create from dictionary."""
        return cls(
            relationship=data.get("relationship", ""),
            source_type=data.get("source_type"),
            target_type=data.get("target_type"),
        )


@dataclass
class InferenceRule:
    """Rule for relationship inference.

    Defines logical rules for inferring new relationships from existing ones.
    """

    name: str
    description: str = ""
    when: list[InferenceCondition] = field(default_factory=list)  # Conditions that must be met
    then_relationship: str = ""  # Relationship type to create
    then_source: str = "first.source"  # Source entity reference (first.source, first.target, etc.)
    then_target: str = "second.target"  # Target entity reference
    confidence: float = 0.5  # Confidence of inferred relationships

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "when": [c.to_dict() for c in self.when],
            "then": {
                "relationship": self.then_relationship,
                "source": self.then_source,
                "target": self.then_target,
            },
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceRule:
        """Create from dictionary."""
        when_data = data.get("when", [])
        when = [InferenceCondition.from_dict(c) if isinstance(c, dict) else c for c in when_data]

        then = data.get("then", {})
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            when=when,
            then_relationship=then.get("relationship", ""),
            then_source=then.get("source", "first.source"),
            then_target=then.get("target", "second.target"),
            confidence=data.get("confidence", 0.5),
        )


@dataclass
class ConfidenceConfig:
    """Confidence threshold configuration."""

    min_entity: float = 0.5
    min_relationship: float = 0.5
    min_inferred: float = 0.3

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "min_entity": self.min_entity,
            "min_relationship": self.min_relationship,
            "min_inferred": self.min_inferred,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfidenceConfig:
        """Create from dictionary."""
        return cls(
            min_entity=data.get("min_entity", 0.5),
            min_relationship=data.get("min_relationship", 0.5),
            min_inferred=data.get("min_inferred", 0.3),
        )


@dataclass
class ExpansionConfig:
    """Configuration for semantic expansion."""

    enabled: bool = True
    depth: int = 2
    cross_tool_unification: bool = True
    relationship_inference: bool = True
    max_entities_per_expansion: int = 100
    # Inference mode: "batch" (after all docs), "incremental" (per doc with graph query), "none"
    inference_mode: str = "incremental"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "enabled": self.enabled,
            "depth": self.depth,
            "cross_tool_unification": self.cross_tool_unification,
            "relationship_inference": self.relationship_inference,
            "max_entities_per_expansion": self.max_entities_per_expansion,
            "inference_mode": self.inference_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpansionConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            depth=data.get("depth", 2),
            cross_tool_unification=data.get("cross_tool_unification", True),
            relationship_inference=data.get("relationship_inference", True),
            max_entities_per_expansion=data.get("max_entities_per_expansion", 100),
            inference_mode=data.get("inference_mode", "incremental"),
        )


@dataclass
class ExpertiseConfig:
    """Complete configurable expertise definition.

    Expertise configurations define domain-specific knowledge for entity
    extraction, including entity types, relationship types, correlation rules,
    and inference rules. All expertise is loaded from configuration (YAML/JSON)
    or defined programmatically - no hard-coded domain knowledge.

    Example usage:
        # Load from file
        loader = ExpertiseLoader()
        expertise = loader.load_file("saas_expert.yaml")

        # Use with MemoryLake
        async with MemoryLake() as lake:
            result = await lake.remember(content, expertise=expertise)

        # Or define programmatically
        expertise = ExpertiseConfig(
            name="custom",
            system_prompt="You are an expert in...",
            entity_types=[EntityTypeConfig(name="CUSTOM", description="...")],
        )
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    extends: list[str] = field(default_factory=list)  # Inherit from other configs

    # LLM prompts (Jinja2 templates supported)
    system_prompt: str | None = None
    extraction_prompt: str | None = None

    # Type definitions
    entity_types: list[EntityTypeConfig] = field(default_factory=list)
    relationship_types: list[RelationshipTypeConfig] = field(default_factory=list)

    # Tool-specific knowledge (arbitrary dict for schema info)
    tool_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Cross-tool correlation rules
    correlation_rules: list[CorrelationRule] = field(default_factory=list)

    # Inference rules for semantic expansion
    inference_rules: list[InferenceRule] = field(default_factory=list)

    # Confidence thresholds
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)

    # Expansion settings
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_entity_type_names(self) -> list[str]:
        """Get list of entity type names."""
        return [et.name for et in self.entity_types]

    def get_relationship_type_names(self) -> list[str]:
        """Get list of relationship type names."""
        return [rt.name for rt in self.relationship_types]

    def get_entity_type(self, name: str) -> EntityTypeConfig | None:
        """Get entity type config by name."""
        for et in self.entity_types:
            if et.name == name:
                return et
        return None

    def get_relationship_type(self, name: str) -> RelationshipTypeConfig | None:
        """Get relationship type config by name."""
        for rt in self.relationship_types:
            if rt.name == name:
                return rt
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "extends": self.extends,
            "system_prompt": self.system_prompt,
            "extraction_prompt": self.extraction_prompt,
            "entity_types": [et.to_dict() for et in self.entity_types],
            "relationship_types": [rt.to_dict() for rt in self.relationship_types],
            "tool_schemas": self.tool_schemas,
            "correlation_rules": [cr.to_dict() for cr in self.correlation_rules],
            "inference_rules": [ir.to_dict() for ir in self.inference_rules],
            "confidence": self.confidence.to_dict(),
            "expansion": self.expansion.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpertiseConfig:
        """Create expertise config from dictionary."""
        entity_types = [
            EntityTypeConfig.from_dict(et) if isinstance(et, dict) else et for et in data.get("entity_types", [])
        ]
        relationship_types = [
            RelationshipTypeConfig.from_dict(rt) if isinstance(rt, dict) else rt
            for rt in data.get("relationship_types", [])
        ]
        correlation_rules = [
            CorrelationRule.from_dict(cr) if isinstance(cr, dict) else cr for cr in data.get("correlation_rules", [])
        ]
        inference_rules = [
            InferenceRule.from_dict(ir) if isinstance(ir, dict) else ir for ir in data.get("inference_rules", [])
        ]

        confidence_data = data.get("confidence", {})
        confidence = (
            ConfidenceConfig.from_dict(confidence_data) if isinstance(confidence_data, dict) else confidence_data
        )

        expansion_data = data.get("expansion", {})
        expansion = ExpansionConfig.from_dict(expansion_data) if isinstance(expansion_data, dict) else expansion_data

        return cls(
            name=data.get("name", "custom"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            extends=data.get("extends", []),
            system_prompt=data.get("system_prompt"),
            extraction_prompt=data.get("extraction_prompt"),
            entity_types=entity_types,
            relationship_types=relationship_types,
            tool_schemas=data.get("tool_schemas", {}),
            correlation_rules=correlation_rules,
            inference_rules=inference_rules,
            confidence=confidence if isinstance(confidence, ConfidenceConfig) else ConfidenceConfig(),
            expansion=expansion if isinstance(expansion, ExpansionConfig) else ExpansionConfig(),
            metadata=data.get("metadata", {}),
        )

    def to_extraction_skill(self) -> ExtractionSkill:
        """Convert to a legacy ExtractionSkill for backward compatibility."""
        return ExtractionSkill(
            name=self.name,
            description=self.description,
            entity_types=self.get_entity_type_names(),
            relationship_types=self.get_relationship_type_names(),
            custom_prompt=self.extraction_prompt,
            min_entity_confidence=self.confidence.min_entity,
            min_relationship_confidence=self.confidence.min_relationship,
            metadata=self.metadata,
        )


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
