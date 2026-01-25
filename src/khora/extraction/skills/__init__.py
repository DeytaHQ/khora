"""Configurable extraction skills for Khora Memory Lake."""

from __future__ import annotations

from .base import ExtractionSkill
from .registry import SkillRegistry

__all__ = [
    "ExtractionSkill",
    "SkillRegistry",
]
