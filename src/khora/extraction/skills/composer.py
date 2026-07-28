"""Expertise configuration composer.

Handles merging and inheritance of expertise configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jinja2.exceptions import SecurityError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from loguru import logger

from .base import (
    ConfidenceConfig,
    CorrelationRule,
    EntityTypeConfig,
    ExpansionConfig,
    ExpertiseConfig,
    InferenceRule,
    RelationshipTypeConfig,
)

if TYPE_CHECKING:
    from .loader import ExpertiseLoader

# Shared sandboxed environment for rendering untrusted prompt templates.
# ImmutableSandboxedEnvironment blocks attribute access to dunders / mutating
# operations, closing the SSTI -> RCE surface of raw ``jinja2.Template``.
_SANDBOX_ENV = ImmutableSandboxedEnvironment()


class ExpertiseComposer:
    """Compose and merge expertise configurations.

    Handles:
    - Merging multiple configurations
    - Resolving 'extends' inheritance chains
    - Combining entity types, relationship types, rules
    - Jinja2 template rendering

    Merge strategy:
    - Entity types: Later configs add new types or override existing by name
    - Relationship types: Same as entity types
    - Correlation rules: Combined, later takes precedence on name conflict
    - Inference rules: Combined, later takes precedence on name conflict
    - Prompts: Later config's prompts override earlier
    - Tool schemas: Deep merged, later values take precedence
    - Confidence: Later config overrides
    - Expansion: Later config overrides
    """

    def __init__(self, loader: ExpertiseLoader | None = None) -> None:
        """Initialize the composer.

        Args:
            loader: ExpertiseLoader for resolving 'extends' references
        """
        self._loader = loader

    def merge(self, configs: list[ExpertiseConfig]) -> ExpertiseConfig:
        """Merge multiple expertise configurations.

        Configurations are merged in order, with later configs taking
        precedence for conflicting values.

        Args:
            configs: List of configurations to merge

        Returns:
            Merged ExpertiseConfig
        """
        if not configs:
            raise ValueError("Cannot merge empty list of configurations")

        if len(configs) == 1:
            return configs[0]

        # Start with first config
        result = self._copy_config(configs[0])

        # Merge each subsequent config
        for config in configs[1:]:
            result = self._merge_two(result, config)

        return result

    def resolve_and_merge(self, config: ExpertiseConfig) -> ExpertiseConfig:
        """Resolve 'extends' references and merge with parents.

        Args:
            config: Configuration with potential 'extends' references

        Returns:
            Resolved and merged ExpertiseConfig
        """
        if not config.extends or not self._loader:
            return config

        # Load and resolve parent configs
        parents = []
        for parent_source in config.extends:
            try:
                parent = self._loader.load_source(parent_source)
                # Recursively resolve parent's extends
                parent = self.resolve_and_merge(parent)
                parents.append(parent)
            except Exception as e:
                logger.warning(f"Failed to load parent expertise {parent_source}: {e}")

        if not parents:
            return config

        # Merge parents first
        base = self.merge(parents) if len(parents) > 1 else parents[0]

        # Merge current config on top of base (excluding extends)
        config_without_extends = self._copy_config(config)
        config_without_extends.extends = []

        return self._merge_two(base, config_without_extends)

    def render_prompt(
        self,
        template: str | None,
        *,
        expertise: ExpertiseConfig | None = None,
        context: dict[str, Any] | None = None,
        parent_prompt: str | None = None,
    ) -> str:
        """Render a Jinja2 prompt template.

        Supported template variables:
        - expertise: The ExpertiseConfig object
        - entity_types: List of entity type configs
        - relationship_types: List of relationship type configs
        - tool_schemas: Tool schema dict
        - tools: List of tool names
        - parent_prompt: Parent's system prompt (for inheritance)
        - source_tool: Name of the source tool for the content, if any
        - tool_context: Source/tool field-context block ("" when absent)
        - attribute_schema: Per-type ATTRIBUTE SCHEMA block naming each entity
          type's required/optional attribute keys, supplied on the extraction
          path. A custom ``extraction_prompt`` must interpolate
          ``{{ attribute_schema }}`` to surface these per-type keys — unlike the
          general "emit attributes" nudge, which is baked into the built-in
          prompt text and is always present. Renders "" when no extracted type
          declares attributes.
        - context: Any additional context passed in (e.g. ``text``)

        Args:
            template: Jinja2 template string
            expertise: ExpertiseConfig for template context
            context: Additional context variables
            parent_prompt: Parent's prompt for {{ parent_prompt }} variable

        Returns:
            Rendered prompt string
        """
        if not template:
            return ""

        user_ctx = context or {}
        ctx = {
            "expertise": expertise,
            "entity_types": expertise.entity_types if expertise else [],
            "relationship_types": expertise.relationship_types if expertise else [],
            "tool_schemas": expertise.tool_schemas if expertise else {},
            "tools": list(expertise.tool_schemas.keys()) if expertise else [],
            "parent_prompt": parent_prompt or "",
            "source_tool": user_ctx.get("source_tool", ""),
            "tool_context": user_ctx.get("tool_context", ""),
            **user_ctx,
        }

        try:
            return _SANDBOX_ENV.from_string(template).render(**ctx)

        except SecurityError as e:
            # Sandbox blocked an unsafe construct (SSTI attempt) - never fail
            # open by returning the raw template; surface the attempt loudly.
            logger.warning(f"Blocked unsafe construct in prompt template: {e}")
            raise

        except Exception as e:
            logger.warning(f"Failed to render prompt template: {e}")
            return template

    def _merge_two(self, base: ExpertiseConfig, overlay: ExpertiseConfig) -> ExpertiseConfig:
        """Merge two configurations, with overlay taking precedence."""
        return ExpertiseConfig(
            name=overlay.name or base.name,
            version=overlay.version if overlay.version != "1.0.0" else base.version,
            description=overlay.description or base.description,
            extends=[],  # Resolved, no need to keep extends
            system_prompt=overlay.system_prompt or base.system_prompt,
            extraction_prompt=overlay.extraction_prompt or base.extraction_prompt,
            entity_types=self._merge_entity_types(base.entity_types, overlay.entity_types),
            relationship_types=self._merge_relationship_types(base.relationship_types, overlay.relationship_types),
            tool_schemas=self._deep_merge_dicts(base.tool_schemas, overlay.tool_schemas),
            correlation_rules=self._merge_correlation_rules(base.correlation_rules, overlay.correlation_rules),
            inference_rules=self._merge_inference_rules(base.inference_rules, overlay.inference_rules),
            confidence=self._merge_confidence(base.confidence, overlay.confidence),
            expansion=self._merge_expansion(base.expansion, overlay.expansion),
            metadata=self._deep_merge_dicts(base.metadata, overlay.metadata),
        )

    def _merge_entity_types(
        self,
        base: list[EntityTypeConfig],
        overlay: list[EntityTypeConfig],
    ) -> list[EntityTypeConfig]:
        """Merge entity type lists, overlay takes precedence on name conflict."""
        result: dict[str, EntityTypeConfig] = {et.name: et for et in base}
        for et in overlay:
            result[et.name] = et
        return list(result.values())

    def _merge_relationship_types(
        self,
        base: list[RelationshipTypeConfig],
        overlay: list[RelationshipTypeConfig],
    ) -> list[RelationshipTypeConfig]:
        """Merge relationship type lists, overlay takes precedence on name conflict."""
        result: dict[str, RelationshipTypeConfig] = {rt.name: rt for rt in base}
        for rt in overlay:
            result[rt.name] = rt
        return list(result.values())

    def _merge_correlation_rules(
        self,
        base: list[CorrelationRule],
        overlay: list[CorrelationRule],
    ) -> list[CorrelationRule]:
        """Merge correlation rule lists, overlay takes precedence on name conflict."""
        result: dict[str, CorrelationRule] = {cr.name: cr for cr in base}
        for cr in overlay:
            result[cr.name] = cr
        return list(result.values())

    def _merge_inference_rules(
        self,
        base: list[InferenceRule],
        overlay: list[InferenceRule],
    ) -> list[InferenceRule]:
        """Merge inference rule lists, overlay takes precedence on name conflict."""
        result: dict[str, InferenceRule] = {ir.name: ir for ir in base}
        for ir in overlay:
            result[ir.name] = ir
        return list(result.values())

    def _merge_confidence(self, base: ConfidenceConfig, overlay: ConfidenceConfig) -> ConfidenceConfig:
        """Merge confidence configs, overlay overrides non-default values."""
        return ConfidenceConfig(
            min_entity=overlay.min_entity if overlay.min_entity != 0.5 else base.min_entity,
            min_relationship=overlay.min_relationship if overlay.min_relationship != 0.5 else base.min_relationship,
            min_inferred=overlay.min_inferred if overlay.min_inferred != 0.3 else base.min_inferred,
        )

    def _merge_expansion(self, base: ExpansionConfig, overlay: ExpansionConfig) -> ExpansionConfig:
        """Merge expansion configs, overlay overrides non-default values.

        The booleans default to True, so an overlay left at its default cannot
        be distinguished from an explicit True. We treat the default value as
        "unset" and only let the overlay win when it differs from the default -
        otherwise a child that simply omits ``expansion`` would silently
        re-enable a flag the parent explicitly disabled (#1126).
        """
        return ExpansionConfig(
            enabled=overlay.enabled if overlay.enabled is not True else base.enabled,
            depth=overlay.depth if overlay.depth != 2 else base.depth,
            cross_tool_unification=(
                overlay.cross_tool_unification
                if overlay.cross_tool_unification is not True
                else base.cross_tool_unification
            ),
            relationship_inference=(
                overlay.relationship_inference
                if overlay.relationship_inference is not True
                else base.relationship_inference
            ),
            max_entities_per_expansion=(
                overlay.max_entities_per_expansion
                if overlay.max_entities_per_expansion != 100
                else base.max_entities_per_expansion
            ),
            inference_mode=overlay.inference_mode if overlay.inference_mode != "smart" else base.inference_mode,
            preload_existing=overlay.preload_existing
            if overlay.preload_existing is not True
            else base.preload_existing,
            batch_storage_size=(
                overlay.batch_storage_size if overlay.batch_storage_size != 50 else base.batch_storage_size
            ),
        )

    def _deep_merge_dicts(self, base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries, overlay takes precedence."""
        result = base.copy()
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _copy_config(self, config: ExpertiseConfig) -> ExpertiseConfig:
        """Create a shallow copy of a configuration."""
        return ExpertiseConfig(
            name=config.name,
            version=config.version,
            description=config.description,
            extends=config.extends.copy(),
            system_prompt=config.system_prompt,
            extraction_prompt=config.extraction_prompt,
            entity_types=config.entity_types.copy(),
            relationship_types=config.relationship_types.copy(),
            tool_schemas=config.tool_schemas.copy(),
            correlation_rules=config.correlation_rules.copy(),
            inference_rules=config.inference_rules.copy(),
            confidence=config.confidence,
            expansion=config.expansion,
            metadata=config.metadata.copy(),
        )
