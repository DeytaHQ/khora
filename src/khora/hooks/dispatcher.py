"""Hook dispatcher — routes extraction events to subscribed callbacks."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from loguru import logger

from khora.core.models.event import EventType, MemoryEvent

from .embedding_filter import EmbeddingFilterCache
from .models import HookSubscription, SemanticFilter, SemanticHooksConfig


class HookDispatcher:
    """Manages hook subscriptions and dispatches extraction events.

    This is the Phase 1 (MVP) implementation: lightweight in-process
    pub/sub on ``MemoryEvent`` during ingestion. Consumers register
    callbacks by ``EventType`` and optionally attach a ``SemanticFilter``
    for Phase 2/3 filtering.

    Thread safety: all methods are safe to call from any asyncio task.
    The dispatcher holds no locks — subscriptions are append-only and
    dispatch uses ``asyncio.gather`` for concurrent callback execution.

    Example::

        dispatcher = HookDispatcher()

        async def on_entity(event: MemoryEvent) -> None:
            print(f"New entity: {event.data.get('name')}")

        sub_id = dispatcher.subscribe(EventType.ENTITY_CREATED, on_entity)

        # During ingestion, the pipeline calls:
        await dispatcher.dispatch(event)

        # Cleanup
        dispatcher.unsubscribe(sub_id)
    """

    def __init__(
        self,
        *,
        max_concurrent: int = 10,
        callback_timeout_seconds: float = 30.0,
        config: SemanticHooksConfig | None = None,
    ) -> None:
        self._subscriptions: dict[str, list[HookSubscription]] = defaultdict(list)
        self._sub_by_id: dict[UUID, HookSubscription] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._embedding_cache = EmbeddingFilterCache()
        self._callback_timeout_seconds = callback_timeout_seconds
        # Filters waiting for description embedding. Populated on subscribe;
        # drained by ``Khora.connect()`` via ``embed_pending_filters``.
        # Closes the gap where operators write ``SemanticFilter(description=...)``
        # per the docs and Level 1 silently never engages because no code
        # ever populated ``filter.embedding``. Issue #576 Phase 1, Item 2.
        self._pending_filters: dict[UUID, SemanticFilter] = {}
        # Level 2 (LLM yes/no) — opt-in via SemanticHooksConfig. Issue #576
        # Phase 1, Item 7. The evaluator is built lazily on first need so
        # the LiteLLM import only happens when Level 2 actually runs.
        self._config: SemanticHooksConfig | None = config
        self._llm_evaluator: Any = None

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: EventType | str,
        callback: Callable[[MemoryEvent], Awaitable[None]],
        *,
        filter: SemanticFilter | None = None,
        namespace_id: UUID | None = None,
    ) -> UUID:
        """Register a callback for an event type.

        Args:
            event_type: The event type to subscribe to (e.g., EventType.ENTITY_CREATED).
            callback: Async function called with the MemoryEvent when it fires.
            filter: Optional semantic filter. In Phase 1, filters by entity_type
                only (Level 0). Phase 2 adds embedding similarity, Phase 3 adds LLM.
            namespace_id: Scope to a specific namespace. None = all namespaces.

        Returns:
            Subscription UUID for later unsubscribe.
        """
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)

        if filter and namespace_id:
            filter.namespace_id = namespace_id

        sub = HookSubscription(
            event_type=key,
            callback=callback,
            filter=filter,
        )

        self._subscriptions[key].append(sub)
        self._sub_by_id[sub.id] = sub

        # Register embedding for Level 1 filtering. If the filter has a
        # description but no embedding yet, queue it for auto-embedding by
        # ``embed_pending_filters`` (called from Khora.connect()). Without
        # this, operators following the docs (which show
        # ``SemanticFilter(description=...)`` with no manual embed step)
        # get silently degraded to Level 0. Issue #576 Phase 1, Item 2.
        if filter:
            if filter.embedding is not None:
                self._embedding_cache.register_filter(filter)
            elif filter.description:
                self._pending_filters[filter.id] = filter

        logger.debug(
            "Hook registered: {} → {} (filter={})",
            key,
            callback.__name__ if hasattr(callback, "__name__") else "callback",
            filter.name if filter else "none",
        )
        return sub.id

    def unsubscribe(self, subscription_id: UUID) -> bool:
        """Remove a subscription by ID.

        Returns True if found and removed, False if not found.
        """
        sub = self._sub_by_id.pop(subscription_id, None)
        if sub is None:
            return False

        subs = self._subscriptions.get(sub.event_type, [])
        self._subscriptions[sub.event_type] = [s for s in subs if s.id != subscription_id]
        logger.debug("Hook unsubscribed: {}", subscription_id)
        return True

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._subscriptions.clear()
        self._sub_by_id.clear()
        self._pending_filters.clear()

    @property
    def subscription_count(self) -> int:
        """Total number of active subscriptions."""
        return len(self._sub_by_id)

    async def embed_pending_filters(self, embedder: Any) -> int:
        """Embed any filters that were registered with a description but
        no precomputed embedding. Drains ``self._pending_filters``.

        Called from ``Khora.connect()`` so operators who write
        ``SemanticFilter(description="...", similarity_threshold=0.7)``
        per the docs actually get Level 1 (embedding similarity) gating —
        prior to Issue #576 Phase 1 they silently fell back to Level 0.

        Filters subscribed after connect() also drain through this when a
        matching event fires, by re-invoking ``embed_pending_filters``
        opportunistically. Safe to call multiple times; idempotent.

        Args:
            embedder: Anything with ``async embed_batch(texts) -> list[list[float]]``.
                ``khora.extraction.embedders.LiteLLMEmbedder`` is the
                standard implementation.

        Returns:
            Number of filters that had their embedding populated this call.
        """
        if not self._pending_filters:
            return 0

        # Snapshot under no lock — registration is append-only; the worst
        # case is we miss a filter registered concurrently and pick it up
        # the next call.
        pending = list(self._pending_filters.values())
        descriptions = [f.description for f in pending]
        try:
            embeddings = await embedder.embed_batch(descriptions)
        except Exception as exc:
            logger.warning(
                "Failed to auto-embed {} pending hook filters: {}. "
                "Level 1 (cosine similarity) gating will remain inactive "
                "for these filters until the next call.",
                len(pending),
                exc,
            )
            return 0

        embedded = 0
        for filt, emb in zip(pending, embeddings, strict=True):
            filt.embedding = emb
            self._embedding_cache.register_filter(filt)
            self._pending_filters.pop(filt.id, None)
            embedded += 1
        logger.debug("Auto-embedded {} hook filters for Level 1 gating", embedded)
        return embedded

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, event: MemoryEvent) -> int:
        """Dispatch an event to all matching subscribers.

        Callbacks run concurrently (up to max_concurrent). Failures in
        one callback do not affect others — errors are logged but not
        propagated.

        Args:
            event: The extraction event to dispatch.

        Returns:
            Number of callbacks that were invoked (including failures).
        """
        key = event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type)
        subs = self._subscriptions.get(key, [])

        if not subs:
            return 0

        # Filter subscriptions by namespace and semantic filter (Level 0)
        matching = []
        for sub in subs:
            if not sub.enabled:
                continue

            # Namespace scope check
            if sub.filter and sub.filter.namespace_id:
                if event.namespace_id != sub.filter.namespace_id:
                    continue

            # Level 0: entity_type / relationship_type pre-filter
            if sub.filter and not self._passes_type_filter(event, sub.filter):
                continue

            # Level 1: embedding similarity pre-screen (Phase 2)
            if sub.filter and sub.filter.embedding is not None:
                entity_embedding = event.data.get("embedding")
                if entity_embedding is not None:
                    passes, _score = self._embedding_cache.passes_embedding_gate(
                        entity_embedding,
                        sub.filter,
                    )
                    if not passes:
                        continue

            # Level 2: nano-LLM yes/no (Issue #576 Phase 1, Item 7).
            # Only engages when:
            #   - operator opted in via KHORA_HOOKS_LLM_EVALUATION_ENABLED
            #   - filter supplied positive examples (anchors the prompt)
            # Fails open on any infrastructure trouble — Level 1 already
            # cosine-matched, so a flaky nano tier must not drop the match.
            if self._config is not None and self._config.llm_evaluation_enabled and sub.filter and sub.filter.examples:
                if self._llm_evaluator is None:
                    from .llm_evaluator import LLMFilterEvaluator

                    self._llm_evaluator = LLMFilterEvaluator(self._config)
                passes_llm = await self._llm_evaluator.evaluate(event, sub.filter)
                if not passes_llm:
                    continue

            matching.append(sub)

        if not matching:
            return 0

        # Fire callbacks concurrently with semaphore. Each callback is
        # wrapped in ``asyncio.wait_for`` so a hung operator-supplied
        # callback can't stall ingest indefinitely — honors the
        # previously-dead ``callback_timeout_seconds`` config field
        # (Issue #576 Phase 1, Item 3).
        async def _safe_invoke(sub: HookSubscription) -> None:
            async with self._semaphore:
                try:
                    await asyncio.wait_for(
                        sub.callback(event),
                        timeout=self._callback_timeout_seconds,
                    )
                except TimeoutError:
                    cb_name = getattr(sub.callback, "__name__", "callback")
                    logger.warning(
                        "Hook callback {} timed out after {}s for event {}.{}",
                        cb_name,
                        self._callback_timeout_seconds,
                        event.event_type.value if isinstance(event.event_type, EventType) else event.event_type,
                        event.resource_id,
                    )
                except Exception:
                    cb_name = getattr(sub.callback, "__name__", "callback")
                    logger.warning(
                        "Hook callback {} failed for event {}.{}",
                        cb_name,
                        event.event_type.value if isinstance(event.event_type, EventType) else event.event_type,
                        event.resource_id,
                    )

        await asyncio.gather(*[_safe_invoke(sub) for sub in matching])
        return len(matching)

    # ------------------------------------------------------------------
    # Phase 1: Type-based filtering (Level 0)
    # ------------------------------------------------------------------

    @staticmethod
    def _passes_type_filter(event: MemoryEvent, filter: SemanticFilter) -> bool:
        """Check if an event passes the filter's type constraints.

        Level 0 filtering — zero cost, just list membership checks.
        """
        data = event.data

        # Entity events: check entity_type
        if filter.entity_types and event.resource_type == "entity":
            entity_type = data.get("entity_type", "")
            if entity_type and entity_type not in filter.entity_types:
                return False

        # Relationship events: check relationship_type
        if filter.relationship_types and event.resource_type == "relationship":
            rel_type = data.get("relationship_type", "")
            if rel_type and rel_type not in filter.relationship_types:
                return False

        return True
