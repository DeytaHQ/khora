"""Extraction module for Khora Memory Lake.

Provides text chunking, embedding generation, and entity extraction
capabilities using LLM-based processing.
"""

from __future__ import annotations

from .chunkers import Chunker, FixedChunker, RecursiveChunker, SemanticChunker, create_chunker
from .embedders import Embedder, LiteLLMEmbedder
from .entity_resolution import EntityResolver, ResolutionResult, resolve_and_merge_entity
from .extractors import EntityExtractor, LLMEntityExtractor
from .extractors.base import (
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelationship,
    ExtractionResult,
    TemporalInfo,
)
from .skills import ExtractionSkill, SkillRegistry

__all__ = [
    # Chunkers
    "Chunker",
    "FixedChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "create_chunker",
    # Embedders
    "Embedder",
    "LiteLLMEmbedder",
    # Extractors
    "EntityExtractor",
    "LLMEntityExtractor",
    # Extraction types
    "ExtractedEntity",
    "ExtractedRelationship",
    "ExtractedEvent",
    "ExtractionResult",
    "TemporalInfo",
    # Entity resolution
    "EntityResolver",
    "ResolutionResult",
    "resolve_and_merge_entity",
    # Skills
    "ExtractionSkill",
    "SkillRegistry",
]
