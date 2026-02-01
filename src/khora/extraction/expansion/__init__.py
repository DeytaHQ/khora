"""Semantic expansion for knowledge graph enhancement.

This module provides capabilities for expanding and enriching extracted
knowledge graphs through:
- Cross-tool entity unification
- Relationship inference
- Configurable rule-based expansion

Example usage:
    from khora.extraction.expansion import SemanticExpander
    from khora.extraction.skills import ExpertiseConfig

    expertise = load_expertise("saas_expert.yaml")
    expander = SemanticExpander(expertise=expertise)

    # Expand entities and relationships
    expanded_entities, new_relationships = await expander.expand(
        entities=entities,
        relationships=relationships,
    )
"""

from __future__ import annotations

from .cross_tool_unifier import CrossToolUnifier, UnificationResult
from .entity_index import EntityIndex
from .expander import ExpansionResult, SemanticExpander
from .relationship_inferrer import InferredRelationship, RelationshipInferrer
from .rule_engine import RuleEngine, RuleEvaluationContext, RuleMatch

__all__ = [
    # Main expander
    "SemanticExpander",
    "ExpansionResult",
    # Entity index (smart mode)
    "EntityIndex",
    # Cross-tool unification
    "CrossToolUnifier",
    "UnificationResult",
    # Relationship inference
    "RelationshipInferrer",
    "InferredRelationship",
    # Rule engine
    "RuleEngine",
    "RuleMatch",
    "RuleEvaluationContext",
]
