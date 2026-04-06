"""Skill registry for managing extraction skills and expertise configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from .base import ExpertiseConfig, ExtractionSkill

if TYPE_CHECKING:
    pass


class SkillRegistry:
    """Registry for managing extraction skills and expertise configurations.

    Provides a centralized way to register, retrieve, and manage
    extraction skills that can be applied to different document types
    or namespaces.

    Supports both legacy ExtractionSkill objects and new ExpertiseConfig
    configurations. ExpertiseConfig objects are automatically converted
    to ExtractionSkill when retrieved via get() or get_or_default().
    """

    def __init__(self) -> None:
        """Initialize the skill registry."""
        self._skills: dict[str, ExtractionSkill] = {}
        self._expertise: dict[str, ExpertiseConfig] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in extraction skills."""
        self.register(ExtractionSkill.general_entities())
        self.register(ExtractionSkill.technical_docs())
        self.register(ExtractionSkill.business_intel())
        self.register(ExtractionSkill.research_papers())

    def register(self, skill: ExtractionSkill | ExpertiseConfig) -> None:
        """Register an extraction skill or expertise configuration.

        Args:
            skill: ExtractionSkill or ExpertiseConfig to register
        """
        if isinstance(skill, ExpertiseConfig):
            self._expertise[skill.name] = skill
            # Also register as ExtractionSkill for backward compatibility
            self._skills[skill.name] = skill.to_extraction_skill()
            logger.debug(f"Registered expertise config: {skill.name}")
        else:
            self._skills[skill.name] = skill
            logger.debug(f"Registered extraction skill: {skill.name}")

    def get(self, name: str) -> ExtractionSkill | None:
        """Get a skill by name.

        Args:
            name: Skill name

        Returns:
            ExtractionSkill or None if not found
        """
        return self._skills.get(name)

    def get_or_default(self, name: str) -> ExtractionSkill:
        """Get a skill by name, falling back to general_entities.

        Args:
            name: Skill name

        Returns:
            ExtractionSkill (general_entities if not found)
        """
        return self._skills.get(name) or self._skills["general_entities"]

    def list_skills(self) -> list[str]:
        """List all registered skill names.

        Returns:
            List of skill names
        """
        return list(self._skills.keys())

    def all_skills(self) -> list[ExtractionSkill]:
        """Get all registered skills.

        Returns:
            List of ExtractionSkill objects
        """
        return list(self._skills.values())

    def unregister(self, name: str) -> bool:
        """Unregister a skill or expertise config.

        Args:
            name: Skill/expertise name to unregister

        Returns:
            True if removed, False if not found
        """
        removed = False
        if name in self._skills:
            del self._skills[name]
            removed = True
        if name in self._expertise:
            del self._expertise[name]
            removed = True
        if removed:
            logger.debug(f"Unregistered: {name}")
        return removed

    def get_expertise(self, name: str) -> ExpertiseConfig | None:
        """Get an expertise configuration by name.

        Args:
            name: Expertise name

        Returns:
            ExpertiseConfig or None if not found
        """
        return self._expertise.get(name)

    def get_expertise_or_default(self, name: str) -> ExpertiseConfig:
        """Get expertise by name, falling back to general.

        Args:
            name: Expertise name

        Returns:
            ExpertiseConfig (general if not found)
        """
        if name in self._expertise:
            return self._expertise[name]

        # Try to load from builtin
        try:
            from .loader import get_default_loader

            loader = get_default_loader()
            expertise = loader.load_builtin(name)
            self.register(expertise)
            return expertise
        except Exception as e:
            logger.debug(f"Failed to load builtin expertise '{name}': {e}")

        # Fall back to general
        if "general" in self._expertise:
            return self._expertise["general"]

        # Generate a basic general expertise
        from .loader import get_default_loader

        try:
            loader = get_default_loader()
            expertise = loader.load_builtin("general")
            self.register(expertise)
            return expertise
        except Exception:
            # Return a minimal expertise config
            return ExpertiseConfig(
                name="general",
                description="General entity extraction",
            )

    def list_expertise(self) -> list[str]:
        """List all registered expertise names.

        Returns:
            List of expertise names
        """
        return list(self._expertise.keys())

    def all_expertise(self) -> list[ExpertiseConfig]:
        """Get all registered expertise configs.

        Returns:
            List of ExpertiseConfig objects
        """
        return list(self._expertise.values())

    def register_from_config(self, config: list[dict[str, Any]]) -> None:
        """Register skills from configuration.

        Args:
            config: List of skill configuration dictionaries
        """
        for skill_config in config:
            skill = ExtractionSkill.from_dict(skill_config)
            self.register(skill)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Export all skills as a dictionary.

        Returns:
            Dictionary of skill name to skill config
        """
        return {name: skill.to_dict() for name, skill in self._skills.items()}


# Global skill registry instance
_default_registry: SkillRegistry | None = None


def get_default_registry() -> SkillRegistry:
    """Get the default global skill registry.

    Returns:
        Default SkillRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry()
    return _default_registry


def register_expertise(expertise: ExpertiseConfig) -> None:
    """Register an expertise configuration in the default registry.

    Convenience function for registering expertise without manually
    getting the registry first.

    Args:
        expertise: ExpertiseConfig to register

    Example:
        expertise = ExpertiseConfig(name="custom", ...)
        register_expertise(expertise)
    """
    registry = get_default_registry()
    registry.register(expertise)


def load_and_register_expertise(source: str) -> ExpertiseConfig:
    """Load expertise from a source and register it.

    Args:
        source: Source specification (file path, "builtin:name", etc.)

    Returns:
        The loaded and registered ExpertiseConfig
    """
    from .loader import get_default_loader

    loader = get_default_loader()
    expertise = loader.load_source(source)
    register_expertise(expertise)
    return expertise
