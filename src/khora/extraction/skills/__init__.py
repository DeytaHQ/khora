"""Configurable extraction skills and expertise for Khora Memory Lake.

This module provides the expertise configuration system for controlling
entity and relationship extraction. All domain knowledge is configurable
through YAML/JSON files or programmatic configuration.

Example usage:
    from khora.extraction.skills import ExpertiseConfig, ExpertiseLoader

    # Load expertise from file
    loader = ExpertiseLoader()
    expertise = loader.load_file("./config/saas_expert.yaml")

    # Or define programmatically
    expertise = ExpertiseConfig(
        name="custom",
        entity_types=[
            EntityTypeConfig(name="CUSTOM", description="Custom entity"),
        ],
    )

    # Use with MemoryLake
    async with MemoryLake() as lake:
        result = await lake.remember(content, expertise=expertise)
"""

from __future__ import annotations

from .base import (
    ConfidenceConfig,
    ConfidenceLevel,
    CorrelationRule,
    EntityTypeConfig,
    EventExtractionConfig,
    ExpansionConfig,
    ExpertiseConfig,
    ExtractionSkill,
    FactExtractionConfig,
    InferenceCondition,
    InferenceRule,
    RelationshipTypeConfig,
)
from .composer import ExpertiseComposer
from .loader import (
    ExpertiseLoader,
    ExpertiseLoadError,
    get_default_loader,
    load_expertise,
)
from .registry import (
    SkillRegistry,
    get_default_registry,
    load_and_register_expertise,
    register_expertise,
)

__all__ = [
    # Core skill class (legacy)
    "ExtractionSkill",
    # Expertise configuration (new)
    "ExpertiseConfig",
    "EntityTypeConfig",
    "RelationshipTypeConfig",
    "CorrelationRule",
    "InferenceRule",
    "InferenceCondition",
    "ConfidenceConfig",
    "ConfidenceLevel",
    "ExpansionConfig",
    # Chronicle engine extraction toggles (Chronicle #1)
    "EventExtractionConfig",
    "FactExtractionConfig",
    # Loading and composition
    "ExpertiseLoader",
    "ExpertiseLoadError",
    "ExpertiseComposer",
    "get_default_loader",
    "load_expertise",
    # Registry
    "SkillRegistry",
    "get_default_registry",
    "register_expertise",
    "load_and_register_expertise",
]
