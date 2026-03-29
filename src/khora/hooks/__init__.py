"""Semantic hooks and triggers for Khora extraction events.

Provides a multi-level event subscription system that allows users to
define semantic filters and subscribe to async callbacks during document
ingestion.

Example usage::

    from khora import MemoryLake
    from khora.hooks import EventType, SemanticFilter

    async with MemoryLake(db_url) as lake:
        # Simple: subscribe to all entity creation events
        async def on_entity(event):
            print(f"New entity: {event.data.get('name')}")

        lake.subscribe(EventType.ENTITY_CREATED, on_entity)

        # Advanced: semantic filter with embedding pre-screen
        filter = SemanticFilter(
            name="competitor_mention",
            description="Any mention of a competitor company",
            entity_types=["ORGANIZATION"],
        )
        lake.subscribe(EventType.ENTITY_CREATED, on_entity, filter=filter)

        # Ingest — callbacks fire automatically
        await lake.remember("Acme Corp announced a new product...")
"""

from __future__ import annotations

from .dispatcher import HookDispatcher
from .embedding_filter import EmbeddingFilterCache, cosine_similarity
from .llm_evaluator import LLMFilterEvaluator
from .models import FilterMatch, HookSubscription, SemanticFilter, SemanticHooksConfig

__all__ = [
    "EmbeddingFilterCache",
    "FilterMatch",
    "HookDispatcher",
    "HookSubscription",
    "LLMFilterEvaluator",
    "SemanticFilter",
    "SemanticHooksConfig",
    "cosine_similarity",
]
