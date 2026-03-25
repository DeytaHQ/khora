"""Ontology validation: schema checks and reference integrity."""

from __future__ import annotations

from dataclasses import dataclass, field

from khora.extraction.skills.base import ExpertiseConfig


@dataclass
class ValidationResult:
    """Result of ontology validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_ontology(config: ExpertiseConfig) -> ValidationResult:
    """Validate an ExpertiseConfig for structural correctness.

    Checks:
    - Entity types have names
    - Relationship source/target types reference valid entity types (or *)
    - Correlation rules reference valid entity types
    - Inference rules reference valid relationship types
    - No duplicate type names
    """
    result = ValidationResult()
    entity_names = {et.name for et in config.entity_types}
    rel_names = {rt.name for rt in config.relationship_types}

    # Check entity types
    seen_entity_names: set[str] = set()
    for et in config.entity_types:
        if not et.name:
            result.errors.append("Entity type has empty name.")
        elif et.name in seen_entity_names:
            result.errors.append(f"Duplicate entity type name: '{et.name}'.")
        seen_entity_names.add(et.name)

        if not et.description:
            result.warnings.append(f"Entity type '{et.name}' has no description.")

    # Check relationship types
    seen_rel_names: set[str] = set()
    for rt in config.relationship_types:
        if not rt.name:
            result.errors.append("Relationship type has empty name.")
        elif rt.name in seen_rel_names:
            result.errors.append(f"Duplicate relationship type name: '{rt.name}'.")
        seen_rel_names.add(rt.name)

        for src in rt.source_types:
            if src != "*" and src not in entity_names:
                result.errors.append(f"Relationship '{rt.name}' references unknown source type '{src}'.")
        for tgt in rt.target_types:
            if tgt != "*" and tgt not in entity_names:
                result.errors.append(f"Relationship '{rt.name}' references unknown target type '{tgt}'.")

    # Check correlation rules
    for cr in config.correlation_rules:
        for et in cr.entity_types:
            if et not in entity_names:
                result.warnings.append(f"Correlation rule '{cr.name}' references unknown entity type '{et}'.")

    # Check inference rules
    for ir in config.inference_rules:
        for cond in ir.when:
            if cond.relationship and cond.relationship not in rel_names:
                result.warnings.append(
                    f"Inference rule '{ir.name}' references unknown relationship '{cond.relationship}'."
                )

    # Check system prompt
    if not config.system_prompt:
        result.warnings.append("No system prompt defined.")

    return result
