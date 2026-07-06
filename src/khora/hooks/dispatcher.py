"""Hook dispatcher — routes extraction events to subscribed callbacks."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from loguru import logger

from khora.core.diagnostics import Degradation
from khora.core.models.event import EventType, MemoryEvent
from khora.telemetry import metric_counter, trace_span

from .embedding_filter import EmbeddingFilterCache
from .match_dsl import matches as _match_dsl
from .models import HookSubscription, SemanticFilter, SemanticHooksConfig
from .subscription_store import PersistentSubscription

# Module-level OTel instruments (#599). Created once per process; no-op
# when no MeterProvider is installed. NO namespace_id label — cardinality.
_LOAD_SPAN = "khora.hooks.subscription.load"
_PERSISTENT_COUNTER = metric_counter(
    "khora.hooks.subscription.persistent_count",
    description="Persistent hook subscriptions loaded from storage on startup (#599).",
)
_PERSIST_DEGRADED_COUNTER = metric_counter(
    "khora.hooks.subscription.persist_degraded_total",
    description="Persistent-subscription writes that fell back to in-memory after a store failure (#599).",
)


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
        subscription_store: Any = None,
        delivery_sink: Callable[[PersistentSubscription, MemoryEvent], Awaitable[None]] | None = None,
        namespace_resolver: Callable[[UUID], Awaitable[UUID]] | None = None,
    ) -> None:
        # #1399: a namespace scope check compares ``event.namespace_id`` against
        # the subscription/filter ``namespace_id``. But ingest emits events with
        # the *row-level* id (``remember()`` resolves the public stable
        # namespace_id to the active version's row id before extraction), while
        # ``subscribe(namespace_id=...)`` stores the *stable* id the caller
        # passed — so the two never matched and every scoped subscription fired
        # zero callbacks. ``namespace_resolver`` (the coordinator's idempotent
        # ``resolve_namespace``, which maps either a stable id OR a row id to the
        # row id) lets the scope check normalize BOTH sides to the row-id space
        # before comparing, so it matches regardless of which space each side
        # started in. None (e.g. a pure in-memory test, or a graph-/vector-only
        # stack with no namespace table) falls back to a direct compare.
        self._namespace_resolver = namespace_resolver
        self._ns_resolve_cache: dict[UUID, UUID] = {}
        self._subscriptions: dict[str, list[HookSubscription]] = defaultdict(list)
        self._sub_by_id: dict[UUID, HookSubscription] = {}
        # Phase 3 (#599): durable subscriptions. Default off — when
        # ``subscription_store`` is None the dispatcher is pure in-process
        # and never touches the DB, preserving the single-process fast path
        # and zero new cost for in-memory-only callers.
        self._subscription_store = subscription_store
        self._delivery_sink = delivery_sink
        self._persistent: dict[str, list[PersistentSubscription]] = defaultdict(list)
        self._persistent_by_id: dict[UUID, PersistentSubscription] = {}
        # Last ADR-001 degradation from a store persist/load failure, for
        # observability assertions / callers that surface it on a result.
        self._last_persist_degradation: Degradation | None = None
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

        if namespace_id:
            # Fold the scope onto the filter so the cascade's namespace check
            # engages. HookSubscription has no namespace_id field, so without
            # this an in-memory sub scoped to a namespace but carrying no
            # filter would match EVERY namespace's events (cross-tenant leak).
            if filter is None:
                filter = SemanticFilter(namespace_id=namespace_id)
            else:
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
        """Remove all subscriptions (in-process and the in-memory copy of
        persistent ones — does NOT delete persisted rows from storage)."""
        self._subscriptions.clear()
        self._sub_by_id.clear()
        self._pending_filters.clear()
        self._persistent.clear()
        self._persistent_by_id.clear()

    @property
    def subscription_count(self) -> int:
        """Total number of active in-process callback subscriptions."""
        return len(self._sub_by_id)

    @property
    def persistent_count(self) -> int:
        """Total number of loaded persistent subscriptions (#599)."""
        return len(self._persistent_by_id)

    # ------------------------------------------------------------------
    # Phase 3 (#599): persistent subscriptions
    # ------------------------------------------------------------------

    async def register_persistent(
        self,
        event_type: EventType | str,
        delivery: dict[str, Any],
        *,
        filter: SemanticFilter | None = None,
        namespace_id: UUID | None = None,
    ) -> UUID:
        """Register a durable subscription with a delivery target.

        Unlike :meth:`subscribe` (an in-process callback that dies with the
        process), a persistent subscription records its ``delivery`` config
        (webhook URL / queue identifier) to storage so a worker re-subscribes
        on restart. The record is also kept in memory so dispatch matches it
        without a DB round-trip.

        Requires a ``subscription_store`` at construction time. If the store
        write fails the subscription is still registered in memory and a
        ``Degradation`` is recorded (ADR-001) — the durability guarantee is
        lost for this row but the subscription is not silently dropped.

        Args:
            event_type: The event type to subscribe to.
            delivery: Webhook URL / queue config the worker resolves.
            filter: Optional semantic filter (same cascade as ``subscribe``).
            namespace_id: Scope. None = all namespaces.

        Returns:
            Subscription UUID.
        """
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)

        if filter and namespace_id:
            filter.namespace_id = namespace_id

        sub = PersistentSubscription(
            event_type=key,
            delivery=delivery,
            namespace_id=namespace_id,
            filter=filter,
        )

        if self._subscription_store is None:
            raise RuntimeError(
                "register_persistent requires a subscription_store; "
                "construct HookDispatcher with subscription_store=... or use subscribe() for in-process hooks."
            )

        try:
            await self._subscription_store.persist(sub)
        except Exception as exc:
            # ADR-001: a persistence failure degrades to in-memory-only
            # rather than dropping the subscription or crashing register.
            _PERSIST_DEGRADED_COUNTER.add(1, {"channel": "subscription_store", "reason": "persist_failed"})
            logger.warning(
                "Failed to persist hook subscription {} ({}). It is registered in "
                "memory but will NOT survive a restart.",
                sub.id,
                key,
                exc_info=True,
            )
            degradation: Degradation = {
                "component": "hooks.subscription_store",
                "reason": "persist_failed",
                "detail": f"event_type={key}",
                "exception": repr(exc),
            }
            self._last_persist_degradation = degradation

        self._add_persistent_in_memory(sub)
        if filter:
            self._register_filter_embedding(filter)

        logger.debug("Persistent hook registered: {} → {} (filter={})", key, sub.id, filter.name if filter else "none")
        return sub.id

    async def unregister_persistent(self, subscription_id: UUID) -> bool:
        """Remove a persistent subscription from memory AND storage.

        Returns True if found (in memory); the storage delete is best-effort
        and a failure is logged but does not flip the return value.
        """
        sub = self._persistent_by_id.pop(subscription_id, None)
        if sub is None:
            return False

        subs = self._persistent.get(sub.event_type, [])
        self._persistent[sub.event_type] = [s for s in subs if s.id != subscription_id]

        if self._subscription_store is not None:
            try:
                await self._subscription_store.delete(subscription_id)
            except Exception as exc:
                # ADR-001: the durable row survives and will resurrect this
                # subscription on the next load_persistent(). Record the
                # degradation + metric so the lingering row is observable
                # rather than silently masked by the True return.
                _PERSIST_DEGRADED_COUNTER.add(1, {"channel": "subscription_store", "reason": "delete_failed"})
                logger.warning(
                    "Failed to delete persistent hook subscription {} from storage; "
                    "it is removed from memory but the row may linger and will "
                    "resurrect on the next load_persistent().",
                    subscription_id,
                    exc_info=True,
                )
                self._last_persist_degradation = {
                    "component": "hooks.subscription_store",
                    "reason": "delete_failed",
                    "detail": f"subscription_id={subscription_id}",
                    "exception": repr(exc),
                }
        return True

    async def load_persistent(self) -> int:
        """Load persistent subscriptions from storage into memory (#599).

        Called on startup so events delivered after a restart still find
        the subscriber. Idempotent — keyed on subscription id, so a second
        call replaces rather than duplicates. A store read failure degrades
        to zero loaded subscriptions (ADR-001) rather than crashing startup.

        Returns:
            Number of persistent subscriptions loaded.
        """
        if self._subscription_store is None:
            return 0

        with trace_span(_LOAD_SPAN):
            try:
                records = await self._subscription_store.load_all()
            except Exception as exc:
                _PERSIST_DEGRADED_COUNTER.add(1, {"channel": "subscription_store", "reason": "load_failed"})
                logger.warning(
                    "Failed to load persistent hook subscriptions from storage; "
                    "persistent hooks are inactive until the next load. {}",
                    exc,
                    exc_info=True,
                )
                self._last_persist_degradation = {
                    "component": "hooks.subscription_store",
                    "reason": "load_failed",
                    "detail": None,
                    "exception": repr(exc),
                }
                return 0

            self._persistent.clear()
            self._persistent_by_id.clear()
            for sub in records:
                self._add_persistent_in_memory(sub)
                if sub.filter:
                    self._register_filter_embedding(sub.filter)

        _PERSISTENT_COUNTER.add(len(records))
        logger.debug("Loaded {} persistent hook subscriptions from storage", len(records))
        return len(records)

    def _add_persistent_in_memory(self, sub: PersistentSubscription) -> None:
        # Replace any prior copy with the same id (idempotent reload).
        existing = self._persistent_by_id.get(sub.id)
        if existing is not None:
            self._persistent[existing.event_type] = [
                s for s in self._persistent.get(existing.event_type, []) if s.id != sub.id
            ]
        self._persistent[sub.event_type].append(sub)
        self._persistent_by_id[sub.id] = sub

    def _register_filter_embedding(self, filter: SemanticFilter) -> None:
        if filter.embedding is not None:
            self._embedding_cache.register_filter(filter)
        elif filter.description:
            self._pending_filters[filter.id] = filter

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
        persistent = self._persistent.get(key, [])

        if not subs and not persistent:
            return 0

        # Filter subscriptions by namespace and semantic filter cascade.
        matching = []
        for sub in subs:
            if not sub.enabled:
                continue
            if not await self._passes_filter_cascade(event, sub.filter):
                continue
            matching.append(sub)

        # Phase 3 (#599): persistent subscriptions take the same cascade,
        # then hand off to the delivery sink (the worker path) instead of an
        # in-process callback.
        if persistent:
            await self._dispatch_persistent(event, key, persistent)

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

    async def _resolve_ns(self, namespace_id: UUID, *, refresh: bool = False) -> UUID:
        """Normalize a namespace id to the row-id space for scope comparison (#1399).

        Uses the (idempotent) ``namespace_resolver`` so a stable id and a
        row-level id both map to the row id. Cached because the mapping is
        stable for the namespace's active version; ``refresh=True`` bypasses
        the cache (and overwrites it on success) - the #1427 verification
        path uses it when a stale stable→row mapping is suspected. Falls back
        to the input unchanged when no resolver is wired or resolution fails
        (e.g. a graph-/vector-only stack with no namespace table), preserving
        the pre-#1399 direct-compare behavior rather than dropping events.
        """
        if self._namespace_resolver is None:
            return namespace_id
        if not refresh:
            cached = self._ns_resolve_cache.get(namespace_id)
            if cached is not None:
                return cached
        try:
            resolved = await self._namespace_resolver(namespace_id)
        except Exception:  # noqa: BLE001 - resolver is best-effort; never drop an event on lookup failure
            return namespace_id
        self._ns_resolve_cache[namespace_id] = resolved
        return resolved

    async def _namespace_scope_matches(self, event: MemoryEvent, scope_ns: UUID | None) -> bool:
        """True if an event is in scope for a subscription/filter namespace (#1399).

        ``None`` scope = all namespaces. Otherwise normalize both the event's
        namespace and the scope namespace to the row-id space before comparing,
        so a subscription scoped by the stable id matches events emitted with
        the resolved row id (and vice versa).

        #1427: ``create_namespace_version()`` activates a new row id for the
        same stable id, so a cached stable→row mapping goes stale and the
        comparison false-negatives forever (scoped subscriptions go silent).
        The cache self-heals here: on a failed comparison, any side whose
        value came from a *previously cached non-identity* mapping (only a
        stable id resolves to a different id - row ids are self-mapping and
        never remap) is re-resolved once, then compared again. A genuinely
        foreign event costs at most one extra indexed lookup per scoped
        subscription; identity-mapped (row-id) sides and cache misses that
        were freshly resolved in this call are never re-resolved.
        """
        if scope_ns is None:
            return True
        event_ns = event.namespace_id
        # Snapshot which sides are stale-suspects BEFORE resolving: a value
        # served from a pre-existing non-identity cache entry may predate a
        # re-versioning; a value resolved fresh in this call cannot be stale.
        stale_suspects = [ns for ns in (event_ns, scope_ns) if self._ns_resolve_cache.get(ns) not in (None, ns)]
        if await self._resolve_ns(event_ns) == await self._resolve_ns(scope_ns):
            return True
        if not stale_suspects:
            return False
        for ns in stale_suspects:
            await self._resolve_ns(ns, refresh=True)
        return await self._resolve_ns(event_ns) == await self._resolve_ns(scope_ns)

    async def _passes_filter_cascade(self, event: MemoryEvent, filter: SemanticFilter | None) -> bool:
        """Run the Level 0/1/2 filter cascade for one subscription.

        Shared by the in-process callback path and the Phase-3 persistent
        path so both honor identical namespace / type / embedding / LLM
        gating. ``None`` filter = match everything for the event type.
        """
        if filter is None:
            return True

        # Namespace scope check (#1399: normalize both sides to row-id space).
        if not await self._namespace_scope_matches(event, filter.namespace_id):
            return False

        # Level 0: entity_type / relationship_type pre-filter
        if not self._passes_type_filter(event, filter):
            return False

        # Level 1: embedding similarity pre-screen (Phase 2)
        if filter.embedding is not None:
            entity_embedding = event.data.get("embedding")
            if entity_embedding is not None:
                passes, _score = self._embedding_cache.passes_embedding_gate(entity_embedding, filter)
                if not passes:
                    return False

        # Level 2: nano-LLM yes/no (Issue #576 Phase 1, Item 7). Only
        # engages when opted in and the filter supplied positive examples.
        # Fails open on infrastructure trouble.
        if self._config is not None and self._config.llm_evaluation_enabled and filter.examples:
            if self._llm_evaluator is None:
                from .llm_evaluator import LLMFilterEvaluator

                self._llm_evaluator = LLMFilterEvaluator(self._config)
            if not await self._llm_evaluator.evaluate(event, filter):
                return False

        return True

    async def _dispatch_persistent(self, event: MemoryEvent, key: str, subs: list[PersistentSubscription]) -> int:
        """Match persistent subscriptions and hand off to the delivery sink.

        The actual webhook/queue worker is out of scope for #599; the
        dispatcher matches and routes to ``delivery_sink`` when one is wired
        (the test harness supplies a stub; production wires the worker). When
        no sink is configured, matches are logged at DEBUG and dropped — the
        durable rows still exist for a worker to drain on its own schedule.
        """
        matching = []
        for sub in subs:
            if sub.paused_at is not None:
                continue
            # Namespace isolation (independent of the filter). A subscription
            # scoped to a namespace must never see another namespace's events,
            # even when it carries no filter — the filter cascade returns True
            # for ``filter is None``, so the scope check cannot live there.
            # #1399: normalize both sides to the row-id space before comparing.
            if not await self._namespace_scope_matches(event, sub.namespace_id):
                continue
            if not await self._passes_filter_cascade(event, sub.filter):
                continue
            matching.append(sub)

        if not matching:
            return 0

        if self._delivery_sink is None:
            logger.debug(
                "{} persistent hook subscription(s) matched {} but no delivery_sink is wired; "
                "a worker must drain them from storage.",
                len(matching),
                key,
            )
            return len(matching)

        async def _safe_deliver(sub: PersistentSubscription) -> None:
            async with self._semaphore:
                try:
                    await asyncio.wait_for(
                        self._delivery_sink(sub, event),
                        timeout=self._callback_timeout_seconds,
                    )
                except Exception:
                    logger.warning(
                        "Persistent hook delivery failed for subscription {} on event {}.{}",
                        sub.id,
                        key,
                        event.resource_id,
                        exc_info=True,
                    )

        await asyncio.gather(*[_safe_deliver(sub) for sub in matching])
        return len(matching)

    # ------------------------------------------------------------------
    # Phase 1: Type-based filtering (Level 0)
    # ------------------------------------------------------------------

    @staticmethod
    def _passes_type_filter(event: MemoryEvent, filter: SemanticFilter) -> bool:
        """Check if an event passes the filter's type constraints.

        Level 0 filtering — zero cost, just list membership checks plus
        (Phase 2) the optional EventBridge-style ``match`` pattern.
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

        # Dream-phase events: check op_type / decision (Issue #666).
        # Resource type is "dream" for events emitted via the dream event
        # sink; matching when either list is set lets subscribers narrow
        # by op kind or decision without re-implementing pattern logic.
        if event.resource_type == "dream":
            if filter.dream_op_types:
                op_type = data.get("op_type", "")
                if op_type and op_type not in filter.dream_op_types:
                    return False
            if filter.dream_decisions:
                decision = data.get("decision", "")
                if decision and decision not in filter.dream_decisions:
                    return False

        # Phase 2 (Item A, Issue #579): structural match DSL.
        if filter.match is not None:
            if not _match_dsl(filter.match, data):
                return False

        return True
