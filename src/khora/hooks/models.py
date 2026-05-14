"""Data models for semantic hooks and triggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SemanticHooksConfig(BaseSettings):
    """Configuration for semantic hooks.

    Env vars: ``KHORA_HOOKS_ENABLED``, ``KHORA_HOOKS_FILTER_MODEL``, etc.
    """

    model_config = SettingsConfigDict(env_prefix="KHORA_HOOKS_", case_sensitive=False)

    enabled: bool = Field(default=True, description="Enable semantic hooks")

    # LLM model for semantic filter evaluation (Phase 3).
    # Defaults to a cheap nano model. Override via config or per-filter.
    filter_model: str = Field(
        default="gpt-4.1-nano",
        description="LLM model for semantic filter yes/no evaluation. "
        "Use a cheap, fast model (gpt-5-nano, gpt-4.1-nano, gemini-2.5-flash-lite).",
    )

    # Embedding similarity threshold for pre-screening (Phase 2)
    default_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default cosine similarity threshold for embedding pre-filter",
    )

    # Batch settings for LLM evaluation (Phase 3)
    llm_batch_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of entity-filter pairs to evaluate per LLM call",
    )
    llm_batch_flush_ms: float = Field(
        default=100.0,
        ge=10.0,
        description="Max milliseconds to wait before flushing an incomplete batch",
    )

    # Level 2 (nano-LLM) evaluation — default OFF. When false, the dispatcher
    # only runs Level 0 + Level 1 even for filters that supplied examples.
    # Issue #576 Phase 1, Item 7.
    llm_evaluation_enabled: bool = Field(
        default=False,
        description=(
            "Enable Level 2 (LLM yes/no) evaluation for filters with examples. "
            "Default OFF — opt in via KHORA_HOOKS_LLM_EVALUATION_ENABLED=true."
        ),
    )
    # Hard cap on Level 2 tokens (input + output) per namespace per rolling hour.
    # When breached, the evaluator fails open (returns True so the Level 1 match
    # is preserved) and emits ``khora.hooks.llm.throttled_total``.
    llm_max_tokens_per_namespace_per_hour: int = Field(
        default=10_000,
        ge=0,
        description=(
            "Per-namespace hourly token budget for Level 2. 0 disables the cap. "
            "When breached, evaluations fail open and a throttle counter fires."
        ),
    )
    # Per-subscription split of the namespace cap. Prevents one noisy filter
    # from draining the whole namespace's hourly budget — the namespace cap
    # remains the global backstop. Default 0 = no per-subscription split.
    # Issue #601.
    llm_max_tokens_per_subscription_per_hour: int = Field(
        default=0,
        ge=0,
        description=(
            "Per-subscription hourly token budget for Level 2 (Issue #601). "
            "0 disables. When set, charged in addition to the namespace cap. "
            "Recommended: namespace_cap / expected_subscription_count."
        ),
    )
    # Cache for repeated (event_summary, filter) evaluations. The event
    # summary is hashed (no raw text retained) so repeated bulk-upsert events
    # that share entity name/type/description short-circuit to the cached
    # decision instead of paying for an LLM call. Issue #601.
    llm_cache_size: int = Field(
        default=2048,
        ge=0,
        description=("Max cached (event_summary, filter) → decision entries. 0 disables the cache."),
    )
    llm_cache_ttl_seconds: float = Field(
        default=3600.0,
        ge=0.0,
        description=(
            "Cache entry TTL. Stale entries are evicted lazily on lookup. "
            "0 disables expiry — entries live until LRU eviction."
        ),
    )

    # Callback settings
    max_concurrent_callbacks: int = Field(
        default=10,
        ge=1,
        description="Max concurrent hook callbacks to prevent thundering herd",
    )
    callback_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Timeout for individual hook callbacks",
    )


# ---------------------------------------------------------------------------
# Semantic filter
# ---------------------------------------------------------------------------


@dataclass
class SemanticFilter:
    """A user-defined semantic filter for extraction events.

    Defines what the user is interested in. Applied to extracted entities
    and relationships during ingestion.

    The filter operates as a cascade:
    - Level 0: entity_type / relationship_type pre-filter (free)
    - Level 1: embedding similarity pre-screen (Phase 2, sub-ms)
    - Level 2: LLM yes/no evaluation (Phase 3, configurable model)

    Example::

        filter = SemanticFilter(
            name="competitor_mention",
            description="Any mention of a competitor company or product",
            entity_types=["ORGANIZATION", "PRODUCT"],
            examples=["Acme Corp released a new widget"],
        )
    """

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""

    # Type pre-filters (Level 0, free). Empty = match all types.
    entity_types: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)

    # EventBridge-style structural filter (Level 0, free). When set, the
    # dispatcher evaluates this pattern against ``event.data`` after the
    # entity_types / relationship_types checks. See ``khora.hooks.match_dsl``
    # for operator reference. None = no structural filter.
    match: dict[str, Any] | None = None

    # Examples for LLM evaluation (Level 2)
    examples: list[str] = field(default_factory=list)
    anti_examples: list[str] = field(default_factory=list)

    # Embeddings (populated at registration time by the filter engine)
    embedding: list[float] | None = None
    example_embeddings: list[list[float]] | None = None

    # Thresholds
    similarity_threshold: float = 0.5  # Level 1 (embedding)
    llm_confidence_threshold: float = 0.5  # Level 2 (LLM)

    # Per-filter LLM model override. None = use config default.
    filter_model: str | None = None

    # Scope
    namespace_id: UUID | None = None  # None = all namespaces

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Filter match result
# ---------------------------------------------------------------------------


@dataclass
class FilterMatch:
    """Result of a semantic filter evaluation against an extraction event.

    Produced when an entity or relationship passes a filter's cascade.
    Delivered to subscribed callbacks.
    """

    filter_id: UUID = field(default_factory=uuid4)
    filter_name: str = ""

    # What matched
    entity_id: UUID | None = None
    relationship_id: UUID | None = None
    chunk_id: UUID | None = None

    # Match details (populated by whichever level triggered)
    similarity_score: float | None = None  # Level 1
    llm_confidence: float | None = None  # Level 2
    llm_reasoning: str = ""  # Level 2 (optional)

    # Which level confirmed the match (0=type, 1=embedding, 2=LLM)
    matched_at_level: int = 0

    # Event context
    event_data: dict[str, Any] = field(default_factory=dict)
    namespace_id: UUID | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Hook subscription
# ---------------------------------------------------------------------------


@dataclass
class HookSubscription:
    """A registered callback for extraction events.

    Links an event type + optional semantic filter to an async callback.
    """

    id: UUID = field(default_factory=uuid4)
    event_type: str = ""  # EventType value (e.g., "entity.created")
    callback: Any = None  # Callable[[MemoryEvent | FilterMatch], Awaitable[None]]
    filter: SemanticFilter | None = None  # Optional semantic filter
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
