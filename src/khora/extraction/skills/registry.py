"""Skill registry for managing extraction skills."""

from __future__ import annotations

from typing import Any

from loguru import logger

from .base import ExtractionSkill


class SkillRegistry:
    """Registry for managing extraction skills.

    Provides a centralized way to register, retrieve, and manage
    extraction skills that can be applied to different document types
    or namespaces.
    """

    def __init__(self) -> None:
        """Initialize the skill registry."""
        self._skills: dict[str, ExtractionSkill] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in extraction skills."""
        self.register(ExtractionSkill.general_entities())
        self.register(ExtractionSkill.technical_docs())
        self.register(ExtractionSkill.business_intel())
        self.register(ExtractionSkill.research_papers())

    def register(self, skill: ExtractionSkill) -> None:
        """Register an extraction skill.

        Args:
            skill: ExtractionSkill to register
        """
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
        """Unregister a skill.

        Args:
            name: Skill name to unregister

        Returns:
            True if skill was removed, False if not found
        """
        if name in self._skills:
            del self._skills[name]
            logger.debug(f"Unregistered extraction skill: {name}")
            return True
        return False

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
