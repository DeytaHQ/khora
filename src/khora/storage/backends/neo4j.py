"""Neo4j backend for knowledge graph storage.

Handles storage and traversal of entities, relationships, and episodes
in Neo4j graph database.
"""

from __future__ import annotations

import asyncio
import re as _re
import time as _time
import weakref as _weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import copy
from datetime import UTC, datetime
from typing import Any, TypedDict
from uuid import UUID, uuid4

from loguru import logger
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction, unit_of_work
from neo4j.exceptions import ClientError, ConnectionAcquisitionTimeoutError

from khora.core.models import Entity, Episode, Relationship
from khora.storage.backends.mixins import (
    GraphBackendBase,
)
from khora.storage.backends.mixins import deserialize_dict as _deserialize_dict
from khora.storage.backends.mixins import element_to_dict as _element_to_dict
from khora.storage.backends.mixins import serialize_dict as _serialize_dict
from khora.telemetry import trace, trace_span

from .._log_safe import _safe_url_for_log

# Neo4j relationship labels must be valid identifiers: letters, digits, underscores.
# LLM-generated types like "at-risk" or "works for" need sanitizing.
_NEO4J_LABEL_RE = _re.compile(r"[^A-Za-z0-9_]")

# Default concurrency limits for Neo4j write transactions.
_DEFAULT_ENTITY_WRITE_CONCURRENCY = 16
_DEFAULT_RELATIONSHIP_WRITE_CONCURRENCY = 8

# Density thresholds for automatic batch-size reduction.
# High-density batches hold Neo4j locks longer, causing deadlock retries.
# Reducing sub-batch size for dense data shortens lock windows without
# touching concurrency — low-density sources are completely unaffected.
_HIGH_DENSITY_ENTITY_THRESHOLD = 80  # entities per upsert call
_HIGH_DENSITY_ENTITY_BATCH_SIZE = 25  # smaller sub-batches for dense data
_HIGH_DENSITY_REL_THRESHOLD = 400  # relationships per call
_HIGH_DENSITY_REL_BATCH_SIZE = 50  # smaller sub-batches for dense data

# Hub-entity overlap threshold for relationship type grouping.
# Relationship types sharing >30% of source/target entities are serialized
# to prevent concurrent MERGE deadlocks on shared hub nodes.
_HUB_OVERLAP_THRESHOLD = 0.3  # Jaccard similarity

# Neo4j 5.x timeout error codes — same tuple used in dual_nodes.py.
_NEO4J_TIMEOUT_CODES = (
    "Neo.ClientError.Transaction.TransactionTimedOut",
    "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
)


class _EntityKeyGate:
    """Key-aware concurrency gate for Neo4j MERGE transactions.

    Tracks in-flight entity keys (namespace_id, name, entity_type).
    Allows non-overlapping batches to proceed concurrently.
    Serializes overlapping batches to prevent Neo4j lock contention.
    """

    def __init__(self, max_concurrent: int) -> None:
        self._condition = asyncio.Condition()
        self._in_flight: set[tuple[str, str, str]] = set()
        self._active = 0
        self._max_concurrent = max_concurrent

    @asynccontextmanager
    async def acquire(self, entities: list, *, bypass: bool = False) -> AsyncIterator[None]:
        if bypass:
            yield
            return
        keys = {
            (
                str(e.namespace_id),
                e.name,
                str(e.entity_type),
            )
            for e in entities
        }
        async with self._condition:
            while (keys & self._in_flight) or self._active >= self._max_concurrent:
                await self._condition.wait()
            self._in_flight |= keys
            self._active += 1
        try:
            yield
        finally:
            async with self._condition:
                self._in_flight -= keys
                self._active -= 1
                self._condition.notify_all()


def _cancel_sampler_task_on_gc(task: asyncio.Task[None]) -> None:
    """Cancel a sampler task from a ``weakref.finalize`` callback.

    Never raises: finalizers run at interpreter shutdown too, where logging
    may no longer be available. If the loop is closed or the task is done,
    silently no-op. Defense in depth — the primary cleanup path is
    ``disconnect()``.
    """
    if task.done():
        return
    try:
        task.cancel()
    except Exception:  # pragma: no cover  # noqa: S110 — finalizer must never raise
        pass


class _InstrumentedSession:
    """Proxy around an ``AsyncSession`` that records real pool acquire time.

    ``AsyncDriver.session()`` is lazy — the pool connection is not bound
    until the session calls its internal ``_connect`` method (invoked by
    ``run``, ``_open_transaction``, and each *retry* inside
    ``_run_transaction``). We wrap ``session._connect`` on the proxy
    instance so each real pool acquire is one histogram observation — not
    "session entry to first run", which would conflate query time + retry
    sleeps with acquire latency.

    **Why wrap ``_connect`` (not ``run`` / ``execute_read`` / ``execute_write``):**
    ``execute_read`` / ``execute_write`` retry internally via
    ``_run_transaction``, and each retry that hits a fresh connection
    calls ``_connect`` again. Wrapping at the entry-method level would
    fold those retries into the first acquire observation, inflating
    ``acquire_duration`` by up to (retries × retry_sleep + query time).
    Hooking ``_connect`` itself produces one observation per real pool
    bind, which is the correct semantic.

    The wrap lives on the instance's ``_connect`` attribute, not on the
    class, so the original method is unaffected. The session is about to
    be awaited and closed inside our ``async with``, so there is no
    state-leak risk and no unpatch is needed.

    All other attribute/method access is forwarded via ``__getattr__``.
    """

    __slots__ = (
        "_inner",
        "_acquire_histogram",
        "_timeout_counter",
        "counted_timeout",
        "last_acquire",
        "slow_acquire_threshold_exceeded",
    )

    _SLOW_ACQUIRE_THRESHOLD_S = 5.0

    def __init__(
        self,
        inner: Any,
        *,
        acquire_histogram: Any,
        timeout_counter: Any,
    ) -> None:
        self._inner = inner
        self._acquire_histogram = acquire_histogram
        self._timeout_counter = timeout_counter
        self.counted_timeout = False
        self.last_acquire = 0.0
        self.slow_acquire_threshold_exceeded = False
        self._install_connect_wrap()

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._inner, name)

    def _install_connect_wrap(self) -> None:
        """Patch ``_connect`` on the wrapped session instance only.

        Each real pool acquire becomes one ``acquire_duration`` observation.
        Timeouts increment the counter and are *not* recorded on the
        histogram (to keep the latency distribution honest).
        """
        inner = self._inner
        original_connect = getattr(inner, "_connect", None)
        if original_connect is None or not callable(original_connect):
            # Session stub / future SDK change — fall through; metrics stay
            # zero for this session rather than crashing.
            return

        acquire_histogram = self._acquire_histogram
        timeout_counter = self._timeout_counter
        proxy = self

        async def _timed_connect(*args: Any, **kwargs: Any) -> Any:
            t0 = _time.monotonic()
            try:
                result = await original_connect(*args, **kwargs)
            except ConnectionAcquisitionTimeoutError:
                timeout_counter.add(1)
                proxy.counted_timeout = True
                raise
            elapsed = _time.monotonic() - t0
            acquire_histogram.record(elapsed, attributes={"status": "ok"})
            proxy.last_acquire = elapsed
            if elapsed > _InstrumentedSession._SLOW_ACQUIRE_THRESHOLD_S:
                proxy.slow_acquire_threshold_exceeded = True
            return result

        try:
            inner._connect = _timed_connect  # instance-level shadow
        except (AttributeError, TypeError):
            # Inner object disallowed attribute assignment (e.g. __slots__
            # without _connect). Skip instrumentation for this session.
            return


# Bidirectional relationship types and their inverses
BIDIRECTIONAL_TYPES: dict[str, str] = {
    "MANAGES": "MANAGED_BY",
    "MANAGED_BY": "MANAGES",
    "WORKS_FOR": "EMPLOYS",
    "EMPLOYS": "WORKS_FOR",
    "PART_OF": "CONTAINS",
    "CONTAINS": "PART_OF",
    "DEPENDS_ON": "DEPENDENCY_OF",
    "DEPENDENCY_OF": "DEPENDS_ON",
    "COLLABORATES_WITH": "COLLABORATES_WITH",
    "REPORTS_TO": "HAS_REPORT",
    "HAS_REPORT": "REPORTS_TO",
    "OWNS": "OWNED_BY",
    "OWNED_BY": "OWNS",
    "LEADS": "LED_BY",
    "LED_BY": "LEADS",
    "ASSIGNED_TO": "HAS_ASSIGNEE",
    "HAS_ASSIGNEE": "ASSIGNED_TO",
}


def _sanitize_neo4j_label(label: str) -> str:
    """Sanitize a string for use as a Neo4j relationship type label."""
    sanitized = _NEO4J_LABEL_RE.sub("_", label.strip())
    return sanitized.upper() if sanitized else "RELATES_TO"


def _derive_version_valid_from(entity: Entity) -> str:
    """Derive the bi-temporal version_valid_from timestamp for an entity.

    Resolution order:
    1. ``occurred_at`` from entity metadata (chunk-level event time)
    2. ``created_at`` from entity metadata (document creation time)
    3. The entity's own ``created_at`` field
    4. ``datetime.now(UTC)`` as last resort
    """
    meta = entity.metadata or {}
    for key in ("occurred_at", "created_at"):
        val = meta.get(key)
        if val is not None:
            if isinstance(val, datetime):
                return val.isoformat()
            if isinstance(val, str):
                return val  # Already ISO-formatted
    if entity.created_at:
        return entity.created_at.isoformat()
    return datetime.now(UTC).isoformat()


def _entity_to_cypher_params(entity: Entity) -> dict[str, Any]:
    """Convert Entity to Cypher-compatible parameter dict."""
    return {
        "id": str(entity.id),
        "namespace_id": str(entity.namespace_id),
        "name": entity.name,
        "entity_type": entity.entity_type,
        "description": entity.description,
        "attributes": _serialize_dict(entity.attributes),
        "source_document_ids": [str(d) for d in entity.source_document_ids],
        "source_chunk_ids": [str(c) for c in entity.source_chunk_ids],
        "mention_count": entity.mention_count,
        "valid_from": entity.valid_from.isoformat() if entity.valid_from else None,
        "valid_until": entity.valid_until.isoformat() if entity.valid_until else None,
        "confidence": entity.confidence,
        "metadata": _serialize_dict(entity.metadata),
        "created_at": entity.created_at.isoformat(),
        "updated_at": entity.updated_at.isoformat(),
        "version_valid_from": _derive_version_valid_from(entity),
        "version_valid_to": None,  # Current version by default
    }


def _relationship_to_cypher_params(rel: Relationship) -> dict[str, Any]:
    """Convert Relationship to Cypher-compatible parameter dict."""
    return {
        "id": str(rel.id),
        "namespace_id": str(rel.namespace_id),
        "source_id": str(rel.source_entity_id),
        "target_id": str(rel.target_entity_id),
        "description": rel.description,
        "properties": _serialize_dict(rel.properties),
        "source_document_ids": [str(d) for d in rel.source_document_ids],
        "source_chunk_ids": [str(c) for c in rel.source_chunk_ids],
        "valid_from": rel.valid_from.isoformat() if rel.valid_from else None,
        "valid_until": rel.valid_until.isoformat() if rel.valid_until else None,
        "confidence": rel.confidence,
        "weight": rel.weight,
        "metadata": _serialize_dict(rel.metadata),
        "created_at": rel.created_at.isoformat(),
        "updated_at": rel.updated_at.isoformat(),
    }


class Neo4jBackend(GraphBackendBase):
    """Neo4j backend for knowledge graph operations.

    Stores entities as nodes and relationships as edges in Neo4j,
    enabling efficient graph traversal and pattern matching.

    Emitted metrics (all are no-ops unless ``logfire`` is installed):

    * ``khora.neo4j.pool.acquire_duration`` (histogram, seconds) —
      duration of each real pool acquire, recorded on every invocation
      of the session's internal ``_connect`` (once per ``run``, once
      per transaction open, and once per retry inside a managed
      transaction). Successful acquires are tagged ``status=ok``;
      timeouts are *not* recorded here — they increment the
      ``khora.neo4j.pool.timeout`` counter only. This keeps the p99
      tail free of ``connection_acquisition_timeout`` deadline
      clusters. **Use for**: alerting on p99 acquire latency
      regressions, sizing pool cap.
    * ``khora.neo4j.pool.acquisition_time`` (histogram, seconds) —
      *legacy* metric kept for backward compatibility. Recorded at
      session construction (not after real acquire), so it captures
      only driver object-allocation overhead. Prefer
      ``acquire_duration`` for new dashboards and alerts.
    * ``khora.neo4j.session.duration`` (histogram, seconds) — total
      wall-clock time a session object was held. **Use for**:
      investigation of slow queries holding connections.
    * ``khora.neo4j.pool.timeout`` (counter) — every
      ``ConnectionAcquisitionTimeoutError``. Incremented exactly once
      per timeout event regardless of which entry method or retry
      surfaced it. **Use for**: paging on pool saturation.
    * ``khora.neo4j.pool.connections.{active,idle,total,creating}``
      (OTel observable gauges) — sampled at the exporter's native
      cadence (~60 s). **Use for**: long-term capacity planning.
    * ``khora.neo4j.pool.utilization`` (gauge) — active / max ratio.
    * ``khora.neo4j.pool.sampled.{active,idle,total,creating,utilization}``
      (histograms) — emitted by the optional high-frequency sampler
      (``pool_sampler_enabled=True``). Interval configurable via
      ``pool_sampler_interval_ms`` (default 500 ms). **Use for**:
      sub-minute ramp/burst investigation. Zero-cost when disabled.
    """

    def __init__(
        self,
        url: str,
        *,
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
        max_connection_pool_size: int = 100,
        connection_acquisition_timeout: float = 60.0,
        retry_delay_jitter_factor: float = 0.5,
        max_connection_lifetime: int = 900,
        liveness_check_timeout: float | None = 30.0,
        query_timeout: float | None = 5.0,
        entity_write_concurrency: int = _DEFAULT_ENTITY_WRITE_CONCURRENCY,
        relationship_write_concurrency: int = _DEFAULT_RELATIONSHIP_WRITE_CONCURRENCY,
        pool_sampler_enabled: bool = False,
        pool_sampler_interval_ms: int = 500,
    ) -> None:
        """Initialize the Neo4j backend.

        Args:
            url: Neo4j connection URL (bolt:// or neo4j://)
            user: Database user
            password: Database password
            database: Database name
            max_connection_pool_size: Maximum connection pool size
            connection_acquisition_timeout: Timeout waiting for a connection from the pool
            retry_delay_jitter_factor: Jitter factor for transaction retry delays (0.0-1.0)
            max_connection_lifetime: Max seconds a connection stays in the pool before rotation
            liveness_check_timeout: Seconds of idle time before liveness check (None disables)
            query_timeout: Per-transaction timeout in seconds for bounded read queries (None disables)
            entity_write_concurrency: Max concurrent entity write transactions
            relationship_write_concurrency: Max concurrent relationship write transactions
            pool_sampler_enabled: When True, start a background task that samples
                Neo4j pool state every ``pool_sampler_interval_ms`` milliseconds
                and records the observations on the
                ``khora.neo4j.pool.sampled.*`` histograms. Zero cost when False.
            pool_sampler_interval_ms: Sampler interval in milliseconds. Clamped to
                [50, 60000]. Only relevant when ``pool_sampler_enabled`` is True.
        """
        self._url = url
        self._user = user
        self._password = password
        self._database = database
        self._max_connection_pool_size = max_connection_pool_size
        self._connection_acquisition_timeout = connection_acquisition_timeout
        self._retry_delay_jitter_factor = retry_delay_jitter_factor
        self._max_connection_lifetime = max_connection_lifetime
        self._liveness_check_timeout = liveness_check_timeout
        self._query_timeout = query_timeout
        self._timed_unit_of_work = unit_of_work(timeout=query_timeout) if query_timeout is not None else None
        self._driver: AsyncDriver | None = None
        self._owns_driver: bool = True
        self._entity_key_gate = _EntityKeyGate(max_concurrent=entity_write_concurrency)
        self._relationship_write_sem = asyncio.Semaphore(relationship_write_concurrency)

        # Apply operator-requested neo4j driver verbosity so services
        # that configure their own loguru sinks (bypassing khora's
        # setup_logging) still get driver logs when they build a backend.
        # Imported lazily to avoid any import-cycle surprises.
        from khora.logging_config import apply_neo4j_log_level_from_env

        apply_neo4j_log_level_from_env()

        self._pool_sampler_enabled = pool_sampler_enabled
        self._pool_sampler_interval_ms = max(50, min(60_000, pool_sampler_interval_ms))
        self._sampler_task: asyncio.Task[None] | None = None
        self._sampler_warned = False
        self._sampler_finalizer: _weakref.finalize | None = None
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialise OTel metric instruments (no-op when logfire absent)."""
        from khora.telemetry.metrics import metric_counter, metric_histogram

        self._pool_metrics_registered = False
        # Legacy histogram — kept for backward compatibility. Recorded at
        # session-object construction, which is lazy, so values are near
        # zero. ``acquire_duration`` is the correct metric to monitor.
        self._acquisition_histogram = metric_histogram(
            "khora.neo4j.pool.acquisition_time",
            unit="s",
            description="Time to construct a Neo4j session object (lazy, legacy).",
        )
        # Correct acquire-duration histogram — recorded on first
        # run/execute_read/execute_write after the pool binds a real
        # connection to the session.
        self._acquire_duration_histogram = metric_histogram(
            "khora.neo4j.pool.acquire_duration",
            unit="s",
            description="Time until a real Neo4j connection is bound to the session.",
        )
        self._session_duration_histogram = metric_histogram(
            "khora.neo4j.session.duration",
            unit="s",
            description="Total time a Neo4j session is held",
        )
        self._timeout_counter = metric_counter(
            "khora.neo4j.pool.timeout",
            description="Neo4j ConnectionAcquisitionTimeoutError occurrences",
        )
        # High-frequency sampler histograms (opt-in).
        self._sampled_active_histogram = metric_histogram(
            "khora.neo4j.pool.sampled.active",
            description="Active Neo4j connections, high-frequency sample.",
        )
        self._sampled_idle_histogram = metric_histogram(
            "khora.neo4j.pool.sampled.idle",
            description="Idle Neo4j connections, high-frequency sample.",
        )
        self._sampled_total_histogram = metric_histogram(
            "khora.neo4j.pool.sampled.total",
            description="Total Neo4j connections, high-frequency sample.",
        )
        self._sampled_creating_histogram = metric_histogram(
            "khora.neo4j.pool.sampled.creating",
            description="Neo4j connections being created, high-frequency sample.",
        )
        self._sampled_utilization_histogram = metric_histogram(
            "khora.neo4j.pool.sampled.utilization",
            description="Neo4j pool utilization (0.0-1.0), high-frequency sample.",
        )

    @classmethod
    def from_config(cls, config: Any) -> Neo4jBackend:
        """Create a Neo4jBackend from a Neo4jConfig object.

        ``config.password`` and ``config.url`` are ``pydantic.SecretStr``
        (or a plain ``str`` if a non-Pydantic dataclass is passed). Unwrap
        exactly here so the driver receives plaintext.
        """
        from pydantic import SecretStr

        password = config.password
        if isinstance(password, SecretStr):
            password = password.get_secret_value()
        url = config.url
        if isinstance(url, SecretStr):
            url = url.get_secret_value()
        return cls(
            url=url or "",
            user=config.user,
            password=password,
            database=config.database,
            max_connection_pool_size=getattr(config, "max_connection_pool_size", 100),
            connection_acquisition_timeout=getattr(config, "connection_acquisition_timeout", 60.0),
            retry_delay_jitter_factor=getattr(config, "retry_delay_jitter_factor", 0.5),
            max_connection_lifetime=getattr(config, "max_connection_lifetime", 900),
            liveness_check_timeout=getattr(config, "liveness_check_timeout", 30.0),
            query_timeout=getattr(config, "query_timeout", 5.0),
            entity_write_concurrency=getattr(config, "entity_write_concurrency", _DEFAULT_ENTITY_WRITE_CONCURRENCY),
            relationship_write_concurrency=getattr(
                config, "relationship_write_concurrency", _DEFAULT_RELATIONSHIP_WRITE_CONCURRENCY
            ),
            pool_sampler_enabled=getattr(config, "pool_sampler_enabled", False),
            pool_sampler_interval_ms=getattr(config, "pool_sampler_interval_ms", 500),
        )

    @classmethod
    def from_driver(
        cls,
        driver: AsyncDriver,
        *,
        database: str = "neo4j",
        entity_write_concurrency: int = _DEFAULT_ENTITY_WRITE_CONCURRENCY,
        relationship_write_concurrency: int = _DEFAULT_RELATIONSHIP_WRITE_CONCURRENCY,
        query_timeout: float | None = 5.0,
    ) -> Neo4jBackend:
        """Create a Neo4jBackend from an existing AsyncDriver.

        The backend will NOT close the driver on disconnect, since
        it does not own it.

        **Sampler note**: the high-frequency pool sampler is *not* started
        by this constructor, even if the caller mutates
        ``instance._pool_sampler_enabled`` afterwards. ``from_driver()`` is
        typically used by shared-driver integrations (tests, fixtures,
        connection-multiplexing setups) where the owner of the driver is
        responsible for metric lifecycle. If you need the sampler on a
        shared-driver backend, use the standard ``Neo4jBackend(...)``
        constructor with a dedicated driver instead.

        Args:
            driver: An existing Neo4j async driver
            database: Database name
            entity_write_concurrency: Max concurrent entity write transactions
            relationship_write_concurrency: Max concurrent relationship write transactions
            query_timeout: Per-transaction timeout in seconds for bounded read queries (None disables)

        Returns:
            Neo4jBackend wrapping the shared driver
        """
        instance = cls.__new__(cls)
        instance._url = ""
        instance._user = ""
        instance._password = ""
        instance._database = database
        instance._max_connection_pool_size = 0
        instance._connection_acquisition_timeout = 60.0
        instance._retry_delay_jitter_factor = 0.5
        instance._max_connection_lifetime = 900
        instance._liveness_check_timeout = 30.0
        instance._query_timeout = query_timeout
        instance._timed_unit_of_work = unit_of_work(timeout=query_timeout) if query_timeout is not None else None
        instance._driver = driver
        instance._owns_driver = False
        instance._entity_key_gate = _EntityKeyGate(max_concurrent=entity_write_concurrency)
        instance._relationship_write_sem = asyncio.Semaphore(relationship_write_concurrency)
        instance._pool_sampler_enabled = False
        instance._pool_sampler_interval_ms = 500
        instance._sampler_task = None
        instance._sampler_warned = False
        instance._sampler_finalizer = None
        instance._init_metrics()
        return instance

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is not None:
            # Already connected (either by connect() or from_driver())
            await self._create_indexes()
            self._register_pool_metrics()
            self._start_pool_sampler()
            return

        logger.info("Connecting to Neo4j at {url}...", url=_safe_url_for_log(self._url))
        self._driver = AsyncGraphDatabase.driver(
            self._url,
            auth=(self._user, self._password),
            max_connection_pool_size=self._max_connection_pool_size,
            connection_acquisition_timeout=self._connection_acquisition_timeout,
            retry_delay_jitter_factor=self._retry_delay_jitter_factor,
            max_connection_lifetime=self._max_connection_lifetime,
            liveness_check_timeout=self._liveness_check_timeout,
            keep_alive=True,
        )
        # Verify connectivity
        await self._driver.verify_connectivity()

        # Create indexes for performance
        await self._create_indexes()
        self._register_pool_metrics()
        self._start_pool_sampler()
        logger.info("Connected to Neo4j")

    async def disconnect(self) -> None:
        """Close Neo4j connections."""
        await self._stop_pool_sampler()
        if self._driver is not None:
            if self._owns_driver:
                logger.info("Disconnecting from Neo4j...")
                await self._driver.close()
                logger.info("Disconnected from Neo4j")
            self._driver = None

    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        if self._driver is None:
            return False
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False

    async def _create_indexes(self) -> None:
        """Create indexes for common queries."""
        if self._driver is None:
            return

        indexes = [
            # Entity indexes
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_namespace IF NOT EXISTS FOR (e:Entity) ON (e.namespace_id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            # Unique constraint on MERGE key (namespace_id, name, entity_type) —
            # prevents duplicate entities and implicitly creates the composite index.
            "CREATE CONSTRAINT entity_ns_name_type_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.namespace_id, e.name, e.entity_type) IS UNIQUE",
            # Composite: namespace + type (for list queries filtering by type without name)
            "CREATE INDEX entity_ns_type IF NOT EXISTS FOR (e:Entity) ON (e.namespace_id, e.entity_type)",
            # Composite: namespace + type + created_at (issue #569 typed-entity recency
            # fast path — supports "latest action items" without scanning all entities
            # of the namespace).
            "CREATE INDEX entity_type_recency IF NOT EXISTS "
            "FOR (e:Entity) ON (e.namespace_id, e.entity_type, e.created_at)",
            # Composite: namespace + valid_from/valid_until (for temporal queries)
            "CREATE INDEX entity_ns_valid_from IF NOT EXISTS FOR (e:Entity) ON (e.namespace_id, e.valid_from)",
            "CREATE INDEX entity_ns_valid_until IF NOT EXISTS FOR (e:Entity) ON (e.namespace_id, e.valid_until)",
            # Entity source_tool (for source-aware queries)
            "CREATE INDEX entity_source_tool IF NOT EXISTS FOR (e:Entity) ON (e.source_tool)",
            # Entity confidence (for threshold filtering: min_entity_confidence)
            "CREATE INDEX entity_confidence IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
            # Bi-temporal entity versioning indexes (on :Entity)
            "CREATE INDEX entity_version_valid_from IF NOT EXISTS FOR (e:Entity) ON (e.version_valid_from)",
            "CREATE INDEX entity_version_valid_to IF NOT EXISTS FOR (e:Entity) ON (e.version_valid_to)",
            "CREATE INDEX entity_ns_version IF NOT EXISTS "
            "FOR (e:Entity) ON (e.namespace_id, e.version_valid_from, e.version_valid_to)",
            # Bi-temporal entity versioning indexes (on :EntityVersion snapshot nodes)
            "CREATE INDEX ev_id IF NOT EXISTS FOR (ev:EntityVersion) ON (ev.id)",
            "CREATE INDEX ev_namespace IF NOT EXISTS FOR (ev:EntityVersion) ON (ev.namespace_id)",
            "CREATE INDEX ev_name IF NOT EXISTS FOR (ev:EntityVersion) ON (ev.name)",
            "CREATE INDEX ev_version_valid_from IF NOT EXISTS FOR (ev:EntityVersion) ON (ev.version_valid_from)",
            "CREATE INDEX ev_version_valid_to IF NOT EXISTS FOR (ev:EntityVersion) ON (ev.version_valid_to)",
            "CREATE INDEX ev_ns_version IF NOT EXISTS "
            "FOR (ev:EntityVersion) ON (ev.namespace_id, ev.version_valid_from, ev.version_valid_to)",
            # Episode indexes
            "CREATE INDEX episode_id IF NOT EXISTS FOR (ep:Episode) ON (ep.id)",
            "CREATE INDEX episode_namespace IF NOT EXISTS FOR (ep:Episode) ON (ep.namespace_id)",
            "CREATE INDEX episode_occurred_at IF NOT EXISTS FOR (ep:Episode) ON (ep.occurred_at)",
        ]

        # Relationship property indexes require Neo4j ≥5.7 or Enterprise Edition
        #
        # Pre-build namespace_id indexes for all known relationship types so that
        # queries don't hit full-scan penalties on the first encounter.  The
        # dynamic _ensure_relationship_type_indexes() remains as a fallback for
        # any LLM-generated types not listed here.
        #
        # Sources: general.yaml, slack.yaml, extraction skills (base.py),
        #          LLM prompt examples, expansion modules (relationship_inferrer,
        #          cross_tool_unifier).
        rel_indexes = [
            # --- id on all known relationship types (used by remap, get, delete) ---
            # Neo4j RANGE indexes require an explicit relationship type — the
            # bare ()-[r]-() form is only valid for LOOKUP indexes, so we expand
            # the same type set used for namespace_id below.
            # Core / general
            "CREATE INDEX rel_relates_to_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.id)",
            "CREATE INDEX rel_collaborates_id IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.id)",
            "CREATE INDEX rel_associated_id IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.id)",
            "CREATE INDEX rel_depends_id IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.id)",
            "CREATE INDEX rel_owns_id IF NOT EXISTS FOR ()-[r:OWNS]-() ON (r.id)",
            "CREATE INDEX rel_works_for_id IF NOT EXISTS FOR ()-[r:WORKS_FOR]-() ON (r.id)",
            "CREATE INDEX rel_implements_id IF NOT EXISTS FOR ()-[r:IMPLEMENTS]-() ON (r.id)",
            "CREATE INDEX rel_part_of_id IF NOT EXISTS FOR ()-[r:PART_OF]-() ON (r.id)",
            # People & org relationships
            "CREATE INDEX rel_knows_id IF NOT EXISTS FOR ()-[r:KNOWS]-() ON (r.id)",
            "CREATE INDEX rel_manages_id IF NOT EXISTS FOR ()-[r:MANAGES]-() ON (r.id)",
            "CREATE INDEX rel_reports_to_id IF NOT EXISTS FOR ()-[r:REPORTS_TO]-() ON (r.id)",
            # Location
            "CREATE INDEX rel_located_in_id IF NOT EXISTS FOR ()-[r:LOCATED_IN]-() ON (r.id)",
            "CREATE INDEX rel_headquartered_in_id IF NOT EXISTS FOR ()-[r:HEADQUARTERED_IN]-() ON (r.id)",
            # Temporal ordering
            "CREATE INDEX rel_precedes_id IF NOT EXISTS FOR ()-[r:PRECEDES]-() ON (r.id)",
            "CREATE INDEX rel_follows_id IF NOT EXISTS FOR ()-[r:FOLLOWS]-() ON (r.id)",
            # Business
            "CREATE INDEX rel_competes_with_id IF NOT EXISTS FOR ()-[r:COMPETES_WITH]-() ON (r.id)",
            "CREATE INDEX rel_partners_with_id IF NOT EXISTS FOR ()-[r:PARTNERS_WITH]-() ON (r.id)",
            "CREATE INDEX rel_uses_id IF NOT EXISTS FOR ()-[r:USES]-() ON (r.id)",
            "CREATE INDEX rel_created_by_id IF NOT EXISTS FOR ()-[r:CREATED_BY]-() ON (r.id)",
            # Slack / messaging
            "CREATE INDEX rel_messaged_id IF NOT EXISTS FOR ()-[r:MESSAGED]-() ON (r.id)",
            "CREATE INDEX rel_sent_message_to_id IF NOT EXISTS FOR ()-[r:SENT_MESSAGE_TO]-() ON (r.id)",
            "CREATE INDEX rel_mentioned_id IF NOT EXISTS FOR ()-[r:MENTIONED]-() ON (r.id)",
            "CREATE INDEX rel_posted_in_id IF NOT EXISTS FOR ()-[r:POSTED_IN]-() ON (r.id)",
            "CREATE INDEX rel_member_of_id IF NOT EXISTS FOR ()-[r:MEMBER_OF]-() ON (r.id)",
            # Project / task
            "CREATE INDEX rel_works_on_id IF NOT EXISTS FOR ()-[r:WORKS_ON]-() ON (r.id)",
            "CREATE INDEX rel_assigned_to_id IF NOT EXISTS FOR ()-[r:ASSIGNED_TO]-() ON (r.id)",
            # Research / derivation
            "CREATE INDEX rel_derived_from_id IF NOT EXISTS FOR ()-[r:DERIVED_FROM]-() ON (r.id)",
            # Expansion-generated types
            "CREATE INDEX rel_co_occurs_with_id IF NOT EXISTS FOR ()-[r:CO_OCCURS_WITH]-() ON (r.id)",
            "CREATE INDEX rel_cross_referenced_id IF NOT EXISTS FOR ()-[r:CROSS_REFERENCED]-() ON (r.id)",
            # Entity-to-chunk / event participation
            "CREATE INDEX rel_mentioned_in_id IF NOT EXISTS FOR ()-[r:MENTIONED_IN]-() ON (r.id)",
            "CREATE INDEX rel_participated_in_id IF NOT EXISTS FOR ()-[r:PARTICIPATED_IN]-() ON (r.id)",
            # Bi-temporal entity versioning: SUPERSEDES edges
            "CREATE INDEX rel_supersedes_id IF NOT EXISTS FOR ()-[r:SUPERSEDES]-() ON (r.id)",
            # --- namespace_id on all known relationship types ---
            # Core / general
            "CREATE INDEX rel_namespace IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.namespace_id)",
            "CREATE INDEX rel_collaborates_ns IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.namespace_id)",
            "CREATE INDEX rel_associated_ns IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.namespace_id)",
            "CREATE INDEX rel_depends_ns IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.namespace_id)",
            "CREATE INDEX rel_owns_ns IF NOT EXISTS FOR ()-[r:OWNS]-() ON (r.namespace_id)",
            "CREATE INDEX rel_works_for_ns IF NOT EXISTS FOR ()-[r:WORKS_FOR]-() ON (r.namespace_id)",
            "CREATE INDEX rel_implements_ns IF NOT EXISTS FOR ()-[r:IMPLEMENTS]-() ON (r.namespace_id)",
            "CREATE INDEX rel_part_of_ns IF NOT EXISTS FOR ()-[r:PART_OF]-() ON (r.namespace_id)",
            # People & org relationships
            "CREATE INDEX rel_knows_ns IF NOT EXISTS FOR ()-[r:KNOWS]-() ON (r.namespace_id)",
            "CREATE INDEX rel_manages_ns IF NOT EXISTS FOR ()-[r:MANAGES]-() ON (r.namespace_id)",
            "CREATE INDEX rel_reports_to_ns IF NOT EXISTS FOR ()-[r:REPORTS_TO]-() ON (r.namespace_id)",
            # Location
            "CREATE INDEX rel_located_in_ns IF NOT EXISTS FOR ()-[r:LOCATED_IN]-() ON (r.namespace_id)",
            "CREATE INDEX rel_headquartered_in_ns IF NOT EXISTS FOR ()-[r:HEADQUARTERED_IN]-() ON (r.namespace_id)",
            # Temporal ordering
            "CREATE INDEX rel_precedes_ns IF NOT EXISTS FOR ()-[r:PRECEDES]-() ON (r.namespace_id)",
            "CREATE INDEX rel_follows_ns IF NOT EXISTS FOR ()-[r:FOLLOWS]-() ON (r.namespace_id)",
            # Business
            "CREATE INDEX rel_competes_with_ns IF NOT EXISTS FOR ()-[r:COMPETES_WITH]-() ON (r.namespace_id)",
            "CREATE INDEX rel_partners_with_ns IF NOT EXISTS FOR ()-[r:PARTNERS_WITH]-() ON (r.namespace_id)",
            "CREATE INDEX rel_uses_ns IF NOT EXISTS FOR ()-[r:USES]-() ON (r.namespace_id)",
            "CREATE INDEX rel_created_by_ns IF NOT EXISTS FOR ()-[r:CREATED_BY]-() ON (r.namespace_id)",
            # Slack / messaging
            "CREATE INDEX rel_messaged_ns IF NOT EXISTS FOR ()-[r:MESSAGED]-() ON (r.namespace_id)",
            "CREATE INDEX rel_sent_message_to_ns IF NOT EXISTS FOR ()-[r:SENT_MESSAGE_TO]-() ON (r.namespace_id)",
            "CREATE INDEX rel_mentioned_ns IF NOT EXISTS FOR ()-[r:MENTIONED]-() ON (r.namespace_id)",
            "CREATE INDEX rel_posted_in_ns IF NOT EXISTS FOR ()-[r:POSTED_IN]-() ON (r.namespace_id)",
            "CREATE INDEX rel_member_of_ns IF NOT EXISTS FOR ()-[r:MEMBER_OF]-() ON (r.namespace_id)",
            # Project / task
            "CREATE INDEX rel_works_on_ns IF NOT EXISTS FOR ()-[r:WORKS_ON]-() ON (r.namespace_id)",
            "CREATE INDEX rel_assigned_to_ns IF NOT EXISTS FOR ()-[r:ASSIGNED_TO]-() ON (r.namespace_id)",
            # Research / derivation
            "CREATE INDEX rel_derived_from_ns IF NOT EXISTS FOR ()-[r:DERIVED_FROM]-() ON (r.namespace_id)",
            # Expansion-generated types
            "CREATE INDEX rel_co_occurs_with_ns IF NOT EXISTS FOR ()-[r:CO_OCCURS_WITH]-() ON (r.namespace_id)",
            "CREATE INDEX rel_cross_referenced_ns IF NOT EXISTS FOR ()-[r:CROSS_REFERENCED]-() ON (r.namespace_id)",
            # Entity-to-chunk / event participation
            "CREATE INDEX rel_mentioned_in_ns IF NOT EXISTS FOR ()-[r:MENTIONED_IN]-() ON (r.namespace_id)",
            "CREATE INDEX rel_participated_in_ns IF NOT EXISTS FOR ()-[r:PARTICIPATED_IN]-() ON (r.namespace_id)",
            # Bi-temporal entity versioning: SUPERSEDES edges
            "CREATE INDEX rel_supersedes_ns IF NOT EXISTS FOR ()-[r:SUPERSEDES]-() ON (r.namespace_id)",
            "CREATE INDEX rel_supersedes_at IF NOT EXISTS FOR ()-[r:SUPERSEDES]-() ON (r.superseded_at)",
            # confidence on highest-volume relationship types
            "CREATE INDEX rel_collaborates_conf IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.confidence)",
            "CREATE INDEX rel_associated_conf IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.confidence)",
            "CREATE INDEX rel_depends_conf IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.confidence)",
            # temporal indexes on relationship valid_from (for "what existed at time T?" queries)
            "CREATE INDEX rel_relates_to_valid_from IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.valid_from)",
            "CREATE INDEX rel_collaborates_valid_from IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.valid_from)",
            "CREATE INDEX rel_associated_valid_from IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.valid_from)",
            "CREATE INDEX rel_depends_valid_from IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.valid_from)",
            "CREATE INDEX rel_owns_valid_from IF NOT EXISTS FOR ()-[r:OWNS]-() ON (r.valid_from)",
            "CREATE INDEX rel_works_for_valid_from IF NOT EXISTS FOR ()-[r:WORKS_FOR]-() ON (r.valid_from)",
            "CREATE INDEX rel_implements_valid_from IF NOT EXISTS FOR ()-[r:IMPLEMENTS]-() ON (r.valid_from)",
            "CREATE INDEX rel_part_of_valid_from IF NOT EXISTS FOR ()-[r:PART_OF]-() ON (r.valid_from)",
            # temporal indexes on relationship created_at (for "when was this edge created?" queries)
            "CREATE INDEX rel_relates_to_created_at IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.created_at)",
            "CREATE INDEX rel_collaborates_created_at IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.created_at)",
            "CREATE INDEX rel_associated_created_at IF NOT EXISTS FOR ()-[r:ASSOCIATED_WITH]-() ON (r.created_at)",
            "CREATE INDEX rel_depends_created_at IF NOT EXISTS FOR ()-[r:DEPENDS_ON]-() ON (r.created_at)",
        ]

        async with self._session() as session:
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.debug(f"Index creation: {e}")

            for index in rel_indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.warning(f"Relationship index creation skipped: {e}")

    def _get_driver(self) -> AsyncDriver:
        """Get the Neo4j driver."""
        if self._driver is None:
            raise RuntimeError("Backend not connected. Call connect() first.")
        return self._driver

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        """Acquire a Neo4j session with pool metrics instrumentation.

        ``AsyncDriver.session()`` is lazy — no pool connection is bound
        until the session's internal ``_connect`` is called (by ``run``,
        ``_open_transaction``, or each retry inside ``_run_transaction``).
        :class:`_InstrumentedSession` wraps ``_connect`` on the instance so
        each real acquire is one ``khora.neo4j.pool.acquire_duration``
        observation. Timeouts bump ``khora.neo4j.pool.timeout`` but are
        excluded from the histogram to keep the latency tail honest.

        The legacy ``khora.neo4j.pool.acquisition_time`` metric is still
        recorded here for backward compatibility, but it only captures the
        session-object construction time (near zero).

        Overhead: a handful of ``time.monotonic()`` calls (~8 µs) per session
        plus one attribute-lookup indirection per ``_connect`` call.
        """
        t0 = _time.monotonic()
        instrumented: _InstrumentedSession | None = None
        try:
            async with self._get_driver().session(database=self._database) as session:
                # Legacy histogram: records object-construction overhead.
                legacy_elapsed = _time.monotonic() - t0
                self._acquisition_histogram.record(legacy_elapsed)
                instrumented = _InstrumentedSession(
                    session,
                    acquire_histogram=self._acquire_duration_histogram,
                    timeout_counter=self._timeout_counter,
                )
                try:
                    yield instrumented
                finally:
                    self._session_duration_histogram.record(_time.monotonic() - t0)
                    if instrumented.slow_acquire_threshold_exceeded:
                        logger.warning(f"Neo4j pool acquisition took {instrumented.last_acquire:.1f}s")
        except ConnectionAcquisitionTimeoutError:
            # Only bump the counter here when the timeout came from a
            # non-_connect path (e.g. a mocked driver raising from
            # __aenter__). The _connect wrap already counted any real
            # pool-acquire timeout. We never record the timeout on the
            # acquire_duration histogram — counter is enough, and keeping
            # the histogram tail free of 10–60s deadline clusters keeps p99
            # honest without dashboard-side filtering.
            if instrumented is None or not instrumented.counted_timeout:
                self._timeout_counter.add(1)
            raise

    def _register_pool_metrics(self) -> None:
        """Register OTel gauge callbacks for Neo4j connection pool state.

        Relies on ``driver._pool`` internals (``connections``,
        ``connections_reservations``), verified stable in neo4j 5.x–6.1.
        Degrades gracefully via ``getattr`` fallbacks if internals change.
        """
        if self._pool_metrics_registered:
            return

        from khora.telemetry.metrics import metric_gauge_callback

        pool = getattr(self._driver, "_pool", None)
        if pool is None:
            logger.debug("Neo4j pool internals not accessible; skipping pool gauges")
            return

        max_size = self._max_connection_pool_size or 100

        def _pool_snapshot() -> tuple[int, int, int, int]:
            """Return (active, idle, total, creating) from the pool.

            Best-effort read; the OTel exporter invokes gauge callbacks
            synchronously so we cannot acquire ``pool.lock`` here (the
            async-cooperative lock forbids await-while-held and the OTel
            callback protocol is sync-only in some backends). A torn read
            is acceptable at the ~60 s exporter cadence — prefer the
            ``khora.neo4j.pool.sampled.*`` histograms (lock-exact) when
            sub-second accuracy is required.
            """
            conns = getattr(pool, "connections", {})
            all_conns = [c for deq in conns.values() for c in deq]
            active = sum(1 for c in all_conns if getattr(c, "in_use", False))
            idle = len(all_conns) - active
            creating = sum(getattr(pool, "connections_reservations", {}).values())
            return active, idle, len(all_conns), creating

        def _observe_active(_options: Any) -> Any:
            from opentelemetry.metrics import Observation

            active, _, _, _ = _pool_snapshot()
            yield Observation(active)

        def _observe_idle(_options: Any) -> Any:
            from opentelemetry.metrics import Observation

            _, idle, _, _ = _pool_snapshot()
            yield Observation(idle)

        def _observe_total(_options: Any) -> Any:
            from opentelemetry.metrics import Observation

            _, _, total, _ = _pool_snapshot()
            yield Observation(total)

        def _observe_creating(_options: Any) -> Any:
            from opentelemetry.metrics import Observation

            _, _, _, creating = _pool_snapshot()
            yield Observation(creating)

        def _observe_utilization(_options: Any) -> Any:
            from opentelemetry.metrics import Observation

            active, _, _, _ = _pool_snapshot()
            yield Observation(active / max_size if max_size else 0.0)

        metric_gauge_callback(
            "khora.neo4j.pool.connections.active",
            [_observe_active],
            description="Neo4j connections currently in use",
        )
        metric_gauge_callback(
            "khora.neo4j.pool.connections.idle",
            [_observe_idle],
            description="Neo4j connections idle in the pool",
        )
        metric_gauge_callback(
            "khora.neo4j.pool.connections.total",
            [_observe_total],
            description="Total Neo4j connections in the pool",
        )
        metric_gauge_callback(
            "khora.neo4j.pool.connections.creating",
            [_observe_creating],
            description="Neo4j connections currently being created",
        )
        metric_gauge_callback(
            "khora.neo4j.pool.utilization",
            [_observe_utilization],
            description="Neo4j connection pool utilization ratio (active / max_pool_size)",
        )
        self._pool_metrics_registered = True

    # =========================================================================
    # High-frequency pool sampler
    # =========================================================================

    def _sample_pool_once(self) -> dict[str, float] | None:
        """Return a single snapshot of pool state, or ``None`` if unavailable.

        Takes ``pool.lock`` (an ``AsyncCooperativeRLock``, used by the driver
        as a sync reentrant lock — no ``await`` permitted while held) so the
        snapshot is consistent with driver mutations. The locked region is
        kept tight: we copy the ``(address, deque)`` pairs and the
        reservation counts under the lock, then release and compute derived
        values outside it.
        """
        driver = self._driver
        if driver is None:
            return None
        pool = getattr(driver, "_pool", None)
        if pool is None:
            return None
        try:
            in_use_fn = getattr(pool, "in_use_connection_count", None)
            pool_lock = getattr(pool, "lock", None)

            # Short critical section — read-only snapshot of the two maps
            # and per-address in_use counts. No await inside the lock.
            if pool_lock is not None:
                with pool_lock:
                    conns_snapshot: list[tuple[Any, list[Any]]] = [
                        (addr, list(deq)) for addr, deq in getattr(pool, "connections", {}).items()
                    ]
                    reservations_snapshot = {
                        addr: int(v) for addr, v in getattr(pool, "connections_reservations", {}).items()
                    }
                    in_use_counts: dict[Any, int] | None = None
                    if in_use_fn is not None:
                        in_use_counts = {}
                        for addr, _deq in conns_snapshot:
                            try:
                                in_use_counts[addr] = int(in_use_fn(addr))
                            except Exception:
                                in_use_counts[addr] = -1  # sentinel → fall back below
            else:
                # Best-effort read without a lock (driver shape changed).
                conns_snapshot = [(addr, list(deq)) for addr, deq in (getattr(pool, "connections", {}) or {}).items()]
                reservations_snapshot = {
                    addr: int(v) for addr, v in (getattr(pool, "connections_reservations", {}) or {}).items()
                }
                in_use_counts = None
                if in_use_fn is not None:
                    in_use_counts = {}
                    for addr, _deq in conns_snapshot:
                        try:
                            in_use_counts[addr] = int(in_use_fn(addr))
                        except Exception:
                            in_use_counts[addr] = -1

            active = 0
            total = 0
            for address, deq in conns_snapshot:
                total += len(deq)
                if in_use_counts is not None and in_use_counts.get(address, -1) >= 0:
                    active += in_use_counts[address]
                else:
                    active += sum(1 for c in deq if getattr(c, "in_use", False))
            # Clamp: driver state can be transiently inconsistent during churn.
            idle = max(0, total - active)
            creating = sum(reservations_snapshot.values())

            pool_config = getattr(pool, "pool_config", None)
            max_size = getattr(pool_config, "max_connection_pool_size", 0) or self._max_connection_pool_size
            utilization = (active / max_size) if max_size and max_size > 0 else 0.0
        except Exception as exc:
            if not self._sampler_warned:
                logger.warning(f"Neo4j pool sampler read failed; disabling further warnings: {exc}")
                self._sampler_warned = True
            return None
        return {
            "active": float(active),
            "idle": float(idle),
            "total": float(total),
            "creating": float(creating),
            "utilization": float(utilization),
        }

    async def _run_sampler(self) -> None:
        """Sample pool state at configured cadence until cancelled."""
        interval_s = self._pool_sampler_interval_ms / 1000.0
        try:
            while True:
                sample = self._sample_pool_once()
                if sample is not None:
                    self._sampled_active_histogram.record(sample["active"])
                    self._sampled_idle_histogram.record(sample["idle"])
                    self._sampled_total_histogram.record(sample["total"])
                    self._sampled_creating_histogram.record(sample["creating"])
                    self._sampled_utilization_histogram.record(sample["utilization"])
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return

    def _start_pool_sampler(self) -> None:
        """Start the background sampler task if enabled and not already running.

        Idempotent — calling ``connect()`` twice (or ``_start_pool_sampler``
        directly) will not spawn a second task.
        """
        if not self._pool_sampler_enabled:
            return
        if self._sampler_task is not None and not self._sampler_task.done():
            return
        task = asyncio.create_task(self._run_sampler())
        self._sampler_task = task
        # Register a GC-triggered finalizer that cancels the task if the
        # backend is dropped without an awaited ``disconnect()`` (crash path,
        # test teardown). ``_cancel_sampler_task_on_gc`` swallows
        # loop-closed / no-loop errors — it is defense in depth only.
        if self._sampler_finalizer is not None:
            self._sampler_finalizer.detach()
        self._sampler_finalizer = _weakref.finalize(self, _cancel_sampler_task_on_gc, task)

    async def _stop_pool_sampler(self) -> None:
        """Cancel and await the sampler task, if any."""
        task = self._sampler_task
        if task is None:
            return
        self._sampler_task = None
        if self._sampler_finalizer is not None:
            self._sampler_finalizer.detach()
            self._sampler_finalizer = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # Sampler is instrumentation-only; never propagate cleanup errors.
            logger.debug(f"Neo4j pool sampler shutdown raised: {exc}")

    # =========================================================================
    # Entity operations
    # =========================================================================

    async def create_entity(self, entity: Entity) -> Entity:
        """Create an entity node in the graph."""

        params = _entity_to_cypher_params(entity)

        async def _create(tx: AsyncManagedTransaction) -> None:
            query = """
            CREATE (e:Entity {
                id: $id,
                namespace_id: $namespace_id,
                name: $name,
                entity_type: $entity_type,
                description: $description,
                attributes: $attributes,
                source_document_ids: $source_document_ids,
                source_chunk_ids: $source_chunk_ids,
                mention_count: $mention_count,
                valid_from: $valid_from,
                valid_until: $valid_until,
                confidence: $confidence,
                metadata: $metadata,
                created_at: $created_at,
                updated_at: $updated_at,
                version_valid_from: $version_valid_from,
                version_valid_to: $version_valid_to
            })
            """
            await tx.run(query, **params)

        async with self._session() as session:
            await session.execute_write(_create)

        return entity

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id`` (IGR-223)."""

        async with self._session() as session:
            result = await session.run(
                "MATCH (e:Entity {id: $id, namespace_id: $namespace_id}) RETURN e",
                id=str(entity_id),
                namespace_id=str(namespace_id),
            )
            record = await result.single()
            if record:
                return self._record_to_entity(record["e"])
            return None

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        """Get an entity by name and type (for deduplication)."""

        async with self._session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {namespace_id: $namespace_id, name: $name, entity_type: $entity_type})
                RETURN e
                LIMIT 1
                """,
                namespace_id=str(namespace_id),
                name=name,
                entity_type=entity_type,
            )
            record = await result.single()
            if record:
                return self._record_to_entity(record["e"])
            return None

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        """Fetch multiple entities in a single query, scoped to ``namespace_id``.

        Entities belonging to any other namespace are silently dropped from
        the result to prevent cross-tenant IDOR (IGR-223).

        Args:
            entity_ids: List of entity IDs to fetch
            namespace_id: Caller's namespace; entities in any other namespace
                are dropped.

        Returns:
            Dictionary mapping entity ID to Entity object
        """
        if not entity_ids:
            return {}

        id_strings = [str(eid) for eid in entity_ids]

        async with self._session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.id IN $ids AND e.namespace_id = $namespace_id
                RETURN e
                """,
                ids=id_strings,
                namespace_id=str(namespace_id),
            )
            records = await result.data()
            return {UUID(r["e"]["id"]): self._record_to_entity(r["e"]) for r in records}

    async def update_entity(self, entity: Entity) -> Entity:
        """Update an entity."""

        params = _entity_to_cypher_params(entity)

        async def _update(tx: AsyncManagedTransaction) -> None:
            query = """
            MATCH (e:Entity {id: $id})
            SET e.name = $name,
                e.description = $description,
                e.attributes = $attributes,
                e.source_document_ids = $source_document_ids,
                e.source_chunk_ids = $source_chunk_ids,
                e.mention_count = $mention_count,
                e.valid_from = $valid_from,
                e.valid_until = $valid_until,
                e.confidence = $confidence,
                e.metadata = $metadata,
                e.updated_at = $updated_at
            """
            await tx.run(query, **params)

        async with self._session() as session:
            await session.execute_write(_update)

        return entity

    async def delete_entity(self, entity_id: UUID) -> bool:
        """Delete an entity and its relationships."""

        async def _delete(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                MATCH (e:Entity {id: $id})
                DETACH DELETE e
                RETURN count(e) as deleted
                """,
                id=str(entity_id),
            )
            record = await result.single()
            return record["deleted"] if record else 0

        async with self._session() as session:
            deleted = await session.execute_write(_delete)
            return deleted > 0

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities in a namespace."""

        query = "MATCH (e:Entity {namespace_id: $namespace_id})"
        params: dict[str, Any] = {"namespace_id": str(namespace_id)}

        if entity_type:
            query += " WHERE e.entity_type = $entity_type"
            params["entity_type"] = entity_type

        query += " RETURN e ORDER BY e.name SKIP $offset LIMIT $limit"
        params["offset"] = offset
        params["limit"] = limit

        async with self._session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._record_to_entity(r["e"]) for r in records]

    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 100,
        bulk_mode: bool = False,
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities using UNWIND + MERGE.

        Matches on (namespace_id, name, entity_type).  Creates if new,
        updates if existing.  Returns (entity, is_new) tuples.

        For high-density batches (>{threshold} entities), automatically
        reduces sub-batch size to shorten Neo4j lock windows and reduce
        deadlock retries.  Low-density batches are unaffected.

        **Bi-temporal versioning**: When an existing entity's attributes
        change, the old node is closed (``version_valid_to`` set) and a new
        versioned node is created with a ``[:SUPERSEDES]`` edge pointing
        from the new version to the old one.  When attributes are unchanged,
        the entity is updated in-place as before.
        """
        if not entities:
            return []

        # Density-based batch size: reduce sub-batch size for high-density
        # data to shorten Neo4j lock windows.  Low-density data is unaffected.
        density_reduced = False
        if len(entities) >= _HIGH_DENSITY_ENTITY_THRESHOLD and batch_size > _HIGH_DENSITY_ENTITY_BATCH_SIZE:
            logger.debug(
                f"High-density entity batch ({len(entities)} entities): "
                f"reducing sub-batch size {batch_size} -> {_HIGH_DENSITY_ENTITY_BATCH_SIZE}"
            )
            batch_size = _HIGH_DENSITY_ENTITY_BATCH_SIZE
            density_reduced = True

        _total_gate_wait_ms = 0.0
        _total_prefetch_merge_ms = 0.0
        _total_versioning_ms = 0.0
        _entities_new = 0
        _entities_updated = 0

        # Phase 1: MERGE to create-or-detect existing entities.
        # Returns whether the entity already existed and if attributes changed.
        _UPSERT_CYPHER = """
            UNWIND $rows AS row
            MERGE (e:Entity {namespace_id: row.namespace_id, name: row.name, entity_type: row.entity_type})
            ON CREATE SET
                e.id = row.id,
                e.description = row.description,
                e.attributes = row.attributes,
                e.source_document_ids = row.source_document_ids,
                e.source_chunk_ids = row.source_chunk_ids,
                e.mention_count = row.mention_count,
                e.valid_from = row.valid_from,
                e.valid_until = row.valid_until,
                e.confidence = row.confidence,
                e.metadata = row.metadata,
                e.created_at = row.created_at,
                e.updated_at = row.updated_at,
                e.version_valid_from = row.version_valid_from,
                e.version_valid_to = null
            ON MATCH SET
                e.description = CASE WHEN size(row.description) > size(coalesce(e.description, ''))
                    THEN row.description ELSE e.description END,
                e.source_document_ids = (e.source_document_ids + row.source_document_ids)[-100..],
                e.source_chunk_ids = (e.source_chunk_ids + row.source_chunk_ids)[-250..],
                e.mention_count = e.mention_count + row.mention_count,
                e.confidence = CASE WHEN row.confidence > e.confidence THEN row.confidence ELSE e.confidence END,
                e.updated_at = row.updated_at,
                e.version_valid_from = coalesce(e.version_valid_from, row.version_valid_from),
                e.attributes = row.attributes
            RETURN e.id AS id, e.name AS name, row.id AS input_id,
                   CASE WHEN e.id = row.id THEN true ELSE false END AS is_new
        """

        # Phase 2: For entities that existed AND had attribute changes,
        # create a versioned snapshot and SUPERSEDES edge.
        # This runs as a separate pass after the main MERGE.
        #
        # Snapshot nodes use the :EntityVersion label (not :Entity) to avoid
        # violating the unique constraint on (namespace_id, name, entity_type).
        # They retain the same properties for bi-temporal point-in-time queries.
        _VERSION_CYPHER = """
            UNWIND $version_rows AS vr
            MATCH (current:Entity {id: vr.current_id})
            WITH current, vr
            CREATE (old:EntityVersion {
                id: vr.old_version_id,
                namespace_id: current.namespace_id,
                name: current.name,
                entity_type: current.entity_type,
                description: vr.old_description,
                attributes: vr.old_attributes,
                source_document_ids: vr.old_source_document_ids,
                source_chunk_ids: vr.old_source_chunk_ids,
                mention_count: vr.old_mention_count,
                valid_from: current.valid_from,
                valid_until: current.valid_until,
                confidence: vr.old_confidence,
                metadata: vr.old_metadata,
                created_at: current.created_at,
                updated_at: vr.superseded_at,
                version_valid_from: vr.old_version_valid_from,
                version_valid_to: vr.superseded_at
            })
            CREATE (current)-[:SUPERSEDES {superseded_at: vr.superseded_at}]->(old)
            SET current.version_valid_from = vr.new_version_valid_from
            RETURN current.id AS id
        """

        # Pre-fetch query: capture attributes of existing entities before MERGE
        # so we can detect attribute changes and create versioned snapshots.
        _PREFETCH_CYPHER = """
            UNWIND $keys AS key
            MATCH (e:Entity {namespace_id: key.namespace_id, name: key.name, entity_type: key.entity_type})
            RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
                   e.namespace_id AS namespace_id,
                   e.attributes AS attributes, e.description AS description,
                   e.source_document_ids AS source_document_ids,
                   e.source_chunk_ids AS source_chunk_ids,
                   e.mention_count AS mention_count,
                   e.confidence AS confidence, e.metadata AS metadata,
                   e.version_valid_from AS version_valid_from
        """

        results: list[tuple[Entity, bool]] = []

        # Sort entities by MERGE key to ensure deterministic lock ordering
        # across concurrent transactions, preventing deadlocks.
        sorted_entities = sorted(
            entities,
            key=lambda e: (
                str(e.namespace_id),
                e.name,
                e.entity_type,
            ),
        )

        for start in range(0, len(sorted_entities), batch_size):
            batch = sorted_entities[start : start + batch_size]
            rows = [_entity_to_cypher_params(e) for e in batch]

            if bulk_mode:
                # Bulk mode: skip prefetch+versioning, bypass gate.
                # Used for --rewrite (new namespace) where no existing entities
                # can conflict and no prior versions need tracking.
                async def _merge_only_tx(
                    tx: AsyncManagedTransaction,
                ) -> list[dict[str, Any]]:
                    merge_result = await tx.run(_UPSERT_CYPHER, rows=rows)
                    return await merge_result.data()

                _gate_t0 = _time.perf_counter()
                async with self._entity_key_gate.acquire(batch, bypass=True):
                    _total_gate_wait_ms += (_time.perf_counter() - _gate_t0) * 1000
                    _merge_t0 = _time.perf_counter()
                    async with self._session() as session:
                        records = await session.execute_write(_merge_only_tx)
                    _total_prefetch_merge_ms += (_time.perf_counter() - _merge_t0) * 1000
            else:
                # Normal mode: prefetch + MERGE + versioning
                # Phase 0: Snapshot existing entities before MERGE
                prefetch_keys = [
                    {
                        "namespace_id": str(e.namespace_id),
                        "name": e.name,
                        "entity_type": e.entity_type,
                    }
                    for e in batch
                ]

                # Combined prefetch + MERGE in a single write transaction.
                # The prefetch read runs first (seeing pre-merge state), then
                # the MERGE runs — one round trip instead of two separate sessions.
                async def _prefetch_and_upsert_tx(
                    tx: AsyncManagedTransaction,
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                    pre_result = await tx.run(_PREFETCH_CYPHER, keys=prefetch_keys)
                    pre_data = await pre_result.data()
                    merge_result = await tx.run(_UPSERT_CYPHER, rows=rows)
                    merge_data = await merge_result.data()
                    return pre_data, merge_data

                _gate_t0 = _time.perf_counter()
                async with self._entity_key_gate.acquire(batch):
                    _total_gate_wait_ms += (_time.perf_counter() - _gate_t0) * 1000

                    _merge_t0 = _time.perf_counter()
                    async with self._session() as session:
                        pre_existing, records = await session.execute_write(_prefetch_and_upsert_tx)
                    _total_prefetch_merge_ms += (_time.perf_counter() - _merge_t0) * 1000

                # Index pre-existing entities by (namespace_id, name, entity_type)
                pre_map: dict[tuple[str, str, str], dict[str, Any]] = {}
                for rec in pre_existing:
                    key = (rec["namespace_id"], rec["name"], rec["entity_type"])
                    pre_map[key] = rec

                # Phase 2: Create versioned snapshots for entities with changed attributes
                now_iso = datetime.now(UTC).isoformat()
                version_rows: list[dict[str, Any]] = []
                input_id_to_row = {r["id"]: r for r in rows}

                for record in records:
                    if record["is_new"]:
                        continue
                    # This was a MATCH (existing entity) — check if attributes changed
                    neo4j_id = record["id"]
                    # Find the corresponding input row
                    input_id = record["input_id"]
                    input_row = input_id_to_row.get(input_id)
                    if not input_row:
                        continue

                    pre_key = (input_row["namespace_id"], input_row["name"], input_row["entity_type"])
                    pre = pre_map.get(pre_key)
                    if not pre:
                        continue

                    # Compare serialized attributes to detect real changes
                    if pre["attributes"] == input_row["attributes"]:
                        continue

                    version_rows.append(
                        {
                            "current_id": neo4j_id,
                            "old_version_id": str(uuid4()),
                            "old_attributes": pre["attributes"],
                            "old_description": pre.get("description", ""),
                            "old_source_document_ids": pre.get("source_document_ids", []),
                            "old_source_chunk_ids": pre.get("source_chunk_ids", []),
                            "old_mention_count": pre.get("mention_count", 1),
                            "old_confidence": pre.get("confidence", 1.0),
                            "old_metadata": pre.get("metadata"),
                            "old_version_valid_from": pre.get("version_valid_from") or now_iso,
                            "superseded_at": now_iso,
                            "new_version_valid_from": input_row["version_valid_from"],
                        }
                    )

                if version_rows:

                    async def _version_tx(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
                        result = await tx.run(_VERSION_CYPHER, version_rows=version_rows)
                        return await result.data()

                    _ver_t0 = _time.perf_counter()
                    async with self._session() as session:
                        version_records = await session.execute_write(_version_tx)
                    _total_versioning_ms += (_time.perf_counter() - _ver_t0) * 1000
                    logger.debug(
                        f"Bi-temporal versioning: created {len(version_records)} "
                        f"SUPERSEDES snapshots for {len(version_rows)} changed entities"
                    )

            # Build result mapping - each input entity should get exactly one result
            input_id_to_entity = {str(e.id): e for e in batch}
            logger.debug(f"Neo4j batch: {len(batch)} entities, {len(records)} records returned")

            # Count new vs updated for telemetry
            for record in records:
                if record["is_new"]:
                    _entities_new += 1
                else:
                    _entities_updated += 1

            # De-duplicate: MERGE can match multiple nodes if duplicates exist
            # in the DB (no unique constraint on the MERGE key). Keep only the
            # first result per input entity.
            seen_input_ids: set[str] = set()
            for record in records:
                input_id = record["input_id"]
                if input_id in seen_input_ids:
                    continue
                seen_input_ids.add(input_id)
                entity = input_id_to_entity.get(input_id)
                if entity is not None:
                    neo4j_id = record["id"]
                    if neo4j_id != input_id:
                        entity.id = UUID(neo4j_id)
                        logger.debug(f"Entity '{entity.name}' ID synced: {input_id} -> {neo4j_id}")
                    results.append((entity, record["is_new"]))
                else:
                    logger.warning(
                        f"Neo4j returned input_id '{input_id}' not in batch. "
                        f"Batch IDs: {list(input_id_to_entity.keys())[:5]}..."
                    )

        _sub_batch_count = (len(entities) + batch_size - 1) // batch_size
        with trace_span(
            "khora.neo4j.upsert_entities_batch",
            entity_count=len(entities),
            batch_size=batch_size,
            sub_batch_count=_sub_batch_count,
            density_reduced=density_reduced,
            entities_new=_entities_new,
            entities_updated=_entities_updated,
            gate_wait_ms=round(_total_gate_wait_ms, 2),
            prefetch_merge_ms=round(_total_prefetch_merge_ms, 2),
            versioning_ms=round(_total_versioning_ms, 2),
        ):
            pass  # span records the accumulated timing as attributes
        logger.debug(
            f"Batch upserted {len(results)} entities "
            f"({_entities_new} new, {_entities_updated} updated, "
            f"gate_wait={_total_gate_wait_ms:.0f}ms, "
            f"merge={_total_prefetch_merge_ms:.0f}ms, "
            f"versioning={_total_versioning_ms:.0f}ms)"
        )
        return results

    async def retire_orphaned_entities_batch(
        self,
        retirement_rows: list[dict[str, str]],
    ) -> int:
        """Snapshot orphaned entities into :EntityVersion nodes on document replacement.

        Each dict in *retirement_rows* must contain keys ``current_id``
        (str UUID of the live :Entity), ``namespace_id`` (str UUID of the
        namespace), ``snapshot_id`` (str UUID for the new :EntityVersion
        node), and ``retired_at`` (ISO-8601 datetime string).

        Unlike ``_VERSION_CYPHER`` (which snapshots *pre-fetch* values on
        attribute change), this copies the *current* node properties — correct
        for document replacement where the entity itself hasn't changed, only
        its source document is being replaced.

        .. note::
           This method does **not** acquire ``_entity_key_gate``.  The
           coordinator (``replace_document_extraction``) must serialize
           retirement with concurrent ``upsert_entities_batch`` calls on
           overlapping entity keys.

        Returns the number of entities retired (may be less than
        ``len(retirement_rows)`` if some ``current_id`` values don't match).
        """
        if not retirement_rows:
            return 0

        _RETIRE_CYPHER = """\
UNWIND $retirement_rows AS r
MATCH (current:Entity {id: r.current_id, namespace_id: r.namespace_id})
CREATE (old:EntityVersion {
    id: r.snapshot_id,
    namespace_id: current.namespace_id,
    name: current.name,
    entity_type: current.entity_type,
    description: current.description,
    attributes: current.attributes,
    source_document_ids: current.source_document_ids,
    source_chunk_ids: current.source_chunk_ids,
    mention_count: current.mention_count,
    valid_from: current.valid_from,
    valid_until: current.valid_until,
    confidence: current.confidence,
    metadata: current.metadata,
    created_at: current.created_at,
    updated_at: r.retired_at,
    version_valid_from: coalesce(current.version_valid_from, r.retired_at),
    version_valid_to: r.retired_at,
    retirement_reason: 'document_replaced'
})
CREATE (current)-[:SUPERSEDES {
    superseded_at: r.retired_at,
    reason: 'document_replaced'
}]->(old)
SET current.valid_until = r.retired_at,
    current.version_valid_to = r.retired_at,
    current.updated_at = r.retired_at
RETURN current.id AS id
"""

        async def _retire_tx(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(_RETIRE_CYPHER, retirement_rows=retirement_rows)
            return await result.data()

        _t0 = _time.perf_counter()
        async with self._session() as session:
            records = await session.execute_write(_retire_tx)
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000

        retired_count = len(records)
        with trace_span(
            "khora.neo4j.retire_orphaned_entities_batch",
            input_count=len(retirement_rows),
            retired_count=retired_count,
            elapsed_ms=round(_elapsed_ms, 2),
        ):
            pass
        logger.debug(
            f"Retired {retired_count} orphaned entities ({len(retirement_rows)} requested, {_elapsed_ms:.0f}ms)"
        )
        return retired_count

    async def reset_entity_source_chunk_ids_batch(
        self,
        namespace_id: UUID,
        rows: list[dict[str, Any]],
    ) -> int:
        """Overwrite ``source_chunk_ids`` on entities to the supplied values.

        The Phase 2 replace lifecycle handles survivor entities via
        ``remap_source_document_ids_batch`` (source_document_ids only) and
        net-new entities via ``upsert_entities_batch`` whose ``ON MATCH`` clause
        *appends* `source_chunk_ids` with a 250-element tail. On a document
        replace where the old chunks are hard-deleted, this leaves survivor
        entities pointing at retired chunk UUIDs and leaves net-new-upserted
        entities carrying stale chunk UUIDs from prior documents. Callers
        (``VectorCypherEngine._remember_via_replace``) use this method after
        the coordinator finishes to SET each entity's ``source_chunk_ids``
        exclusively to the current extraction's chunk UUIDs.

        Each row must contain:
        - ``name`` (str): entity name
        - ``entity_type`` (str): entity type
        - ``source_chunk_ids`` (list[str]): the authoritative new chunk UUIDs

        Returns the number of entities whose ``source_chunk_ids`` were updated.
        """
        if not rows:
            return 0

        _RESET_CYPHER = """\
UNWIND $rows AS row
MATCH (e:Entity {namespace_id: $namespace_id, name: row.name, entity_type: row.entity_type})
SET e.source_chunk_ids = row.source_chunk_ids,
    e.updated_at = $updated_at
RETURN count(e) AS updated
"""

        updated_at = datetime.now(UTC).isoformat()
        ns_str = str(namespace_id)

        async def _reset_tx(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(_RESET_CYPHER, rows=rows, namespace_id=ns_str, updated_at=updated_at)
            record = await result.single()
            return record["updated"] if record else 0

        async with self._session() as session:
            count = await session.execute_write(_reset_tx)
        logger.debug(f"Reset source_chunk_ids on {count} entities in namespace {namespace_id}")
        return count

    async def reset_relationship_source_chunk_ids_batch(
        self,
        namespace_id: UUID,
        rows: list[dict[str, Any]],
    ) -> int:
        """Overwrite ``source_chunk_ids`` on relationships to the supplied values.

        Sibling to :meth:`reset_entity_source_chunk_ids_batch` for the
        relationship side of the replace lifecycle. Same
        ``(old + new)[-250..]`` append issue at ``create_relationships_batch``'s
        ``ON MATCH`` clause (see ``neo4j.py`` relationship MERGE).

        Matching is by entity name+type (not entity UUID) so survivor
        relationships whose endpoints are survivor entities still resolve
        correctly — Neo4j MERGE keys entities on ``(namespace_id, name,
        entity_type)``, so name+type is stable across replaces while the
        entity UUID may not be.

        Each row must contain:
        - ``source_name`` / ``source_type`` (str): source entity key
        - ``target_name`` / ``target_type`` (str): target entity key
        - ``rel_type`` (str): already-sanitized Neo4j relationship type
        - ``source_chunk_ids`` (list[str]): the authoritative new chunk UUIDs

        Returns the number of relationships whose ``source_chunk_ids`` were updated.
        """
        if not rows:
            return 0

        _RESET_CYPHER = """\
UNWIND $rows AS row
MATCH (s:Entity {namespace_id: $namespace_id, name: row.source_name, entity_type: row.source_type})
MATCH (t:Entity {namespace_id: $namespace_id, name: row.target_name, entity_type: row.target_type})
MATCH (s)-[r {namespace_id: $namespace_id}]->(t)
WHERE type(r) = row.rel_type
SET r.source_chunk_ids = row.source_chunk_ids,
    r.updated_at = $updated_at
RETURN count(r) AS updated
"""

        updated_at = datetime.now(UTC).isoformat()
        ns_str = str(namespace_id)

        async def _reset_tx(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(_RESET_CYPHER, rows=rows, namespace_id=ns_str, updated_at=updated_at)
            record = await result.single()
            return record["updated"] if record else 0

        async with self._session() as session:
            count = await session.execute_write(_reset_tx)
        logger.debug(f"Reset source_chunk_ids on {count} relationships in namespace {namespace_id}")
        return count

    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 200,
    ) -> int:
        """Batch create relationships using UNWIND with sequential type processing.

        Relationships (including inverse/bidirectional) are grouped by type
        and each type group is processed sequentially in sorted order to
        ensure deterministic lock ordering and prevent Neo4j deadlocks.
        Uses MERGE to prevent duplicate edges (matched on source/target + namespace).
        """
        if not relationships:
            return 0

        _method_t0 = _time.perf_counter()

        # Density-based batch size for relationships
        rel_density_reduced = False
        if len(relationships) >= _HIGH_DENSITY_REL_THRESHOLD and batch_size > _HIGH_DENSITY_REL_BATCH_SIZE:
            logger.debug(
                f"High-density relationship batch ({len(relationships)} rels): "
                f"reducing sub-batch size {batch_size} -> {_HIGH_DENSITY_REL_BATCH_SIZE}"
            )
            batch_size = _HIGH_DENSITY_REL_BATCH_SIZE
            rel_density_reduced = True

        _per_type_ms: dict[str, float] = {}

        # Build inverse relationships upfront so they share the same pass,
        # eliminating a second round of write transactions on overlapping nodes.
        all_rels = list(relationships)
        for rel in relationships:
            rel_type_str = _sanitize_neo4j_label(rel.relationship_type)
            inverse_type = BIDIRECTIONAL_TYPES.get(rel_type_str)
            if inverse_type and inverse_type != rel_type_str:
                inv = copy(rel)
                inv.id = uuid4()
                inv.source_entity_id = rel.target_entity_id
                inv.target_entity_id = rel.source_entity_id
                inv.relationship_type = inverse_type
                inv.description = f"Inverse of: {rel.description}" if rel.description else ""
                all_rels.append(inv)

        # Group by relationship type (required for dynamic rel type in Cypher)
        type_groups: dict[str, list[Relationship]] = {}
        for rel in all_rels:
            rel_type = _sanitize_neo4j_label(rel.relationship_type)
            type_groups.setdefault(rel_type, []).append(rel)

        async def _create_type_group(rel_type: str, rels: list[Relationship]) -> int:
            """Create all batches for a single relationship type sequentially."""
            # Sort by (source_entity_id, target_entity_id, relationship_type) to ensure
            # deterministic lock ordering across concurrent transactions.
            sorted_rels = sorted(
                rels,
                key=lambda r: (str(r.source_entity_id), str(r.target_entity_id), r.relationship_type),
            )
            _type_t0 = _time.perf_counter()
            type_total = 0
            for start in range(0, len(sorted_rels), batch_size):
                batch = sorted_rels[start : start + batch_size]
                rows = [_relationship_to_cypher_params(r) for r in batch]
                query = f"""
                UNWIND $rows AS row
                MATCH (source:Entity {{id: row.source_id}})
                MATCH (target:Entity {{id: row.target_id}})
                MERGE (source)-[r:{rel_type} {{namespace_id: row.namespace_id}}]->(target)
                ON CREATE SET
                    r.id = row.id,
                    r.description = row.description,
                    r.properties = row.properties,
                    r.source_document_ids = row.source_document_ids,
                    r.source_chunk_ids = row.source_chunk_ids,
                    r.valid_from = row.valid_from,
                    r.valid_until = row.valid_until,
                    r.confidence = row.confidence,
                    r.weight = row.weight,
                    r.metadata = row.metadata,
                    r.created_at = row.created_at,
                    r.updated_at = row.updated_at
                ON MATCH SET
                    r.description = CASE WHEN size(row.description) > size(coalesce(r.description, ''))
                        THEN row.description ELSE r.description END,
                    r.source_document_ids = (r.source_document_ids + row.source_document_ids)[-100..],
                    r.source_chunk_ids = (r.source_chunk_ids + row.source_chunk_ids)[-250..],
                    r.confidence = CASE WHEN row.confidence > r.confidence THEN row.confidence ELSE r.confidence END,
                    r.weight = CASE WHEN row.weight > r.weight THEN row.weight ELSE r.weight END,
                    r.updated_at = row.updated_at
                RETURN count(r) AS created
                """

                async def _tx(tx: AsyncManagedTransaction) -> int:
                    result = await tx.run(query, rows=rows)
                    record = await result.single()
                    return record["created"] if record else 0

                async with self._session() as session:
                    type_total += await session.execute_write(_tx)
            _per_type_ms[rel_type] = (_time.perf_counter() - _type_t0) * 1000
            return type_total

        # ---------------------------------------------------------------
        # Hub-entity grouping: When multiple relationship types share
        # the same source/target entities, concurrent MERGE operations
        # deadlock on those hub nodes.  For high-density batches, we
        # detect shared hub entities via Jaccard overlap and serialize
        # overlapping type groups.  Low-density batches skip this
        # (the overhead isn't worth it for a few relationship types).
        # ---------------------------------------------------------------
        _hub_grouping_ms = 0.0
        _execution_group_count = 0
        if len(all_rels) >= _HIGH_DENSITY_REL_THRESHOLD and len(type_groups) > 1:
            # Build entity reference sets per type group
            _hub_t0 = _time.perf_counter()
            type_entity_sets: dict[str, set[str]] = {}
            for rt, rels_list in type_groups.items():
                entity_ids: set[str] = set()
                for r in rels_list:
                    entity_ids.add(str(r.source_entity_id))
                    entity_ids.add(str(r.target_entity_id))
                type_entity_sets[rt] = entity_ids

            # Group overlapping types into serial execution groups
            execution_groups: list[list[str]] = []
            assigned: set[str] = set()
            for rt in sorted(type_groups):
                if rt in assigned:
                    continue
                group = [rt]
                assigned.add(rt)
                group_entities = set(type_entity_sets[rt])
                for other_rt in sorted(type_groups):
                    if other_rt in assigned:
                        continue
                    other_entities = type_entity_sets[other_rt]
                    if not group_entities or not other_entities:
                        continue
                    intersection = len(group_entities & other_entities)
                    union = len(group_entities | other_entities)
                    jaccard = intersection / union if union else 0.0
                    if jaccard >= _HUB_OVERLAP_THRESHOLD:
                        group.append(other_rt)
                        assigned.add(other_rt)
                        group_entities |= other_entities
                execution_groups.append(group)

            async def _run_hub_group(group: list[str]) -> int:
                total = 0
                for rt in group:
                    total += await _create_type_group(rt, type_groups[rt])
                return total

            async def _limited_hub_run(group: list[str]) -> int:
                async with self._relationship_write_sem:
                    return await _run_hub_group(group)

            _hub_grouping_ms = (_time.perf_counter() - _hub_t0) * 1000
            _execution_group_count = len(execution_groups)
            logger.debug(f"Hub grouping: {len(type_groups)} types -> {_execution_group_count} execution groups")
            results = await asyncio.gather(*[_limited_hub_run(g) for g in execution_groups])
        else:
            # Low-density: simple bounded parallelism (original behavior)
            async def _limited_create(rel_type: str, rels: list[Relationship]) -> int:
                async with self._relationship_write_sem:
                    return await _create_type_group(rel_type, rels)

            results = await asyncio.gather(*[_limited_create(rt, type_groups[rt]) for rt in sorted(type_groups)])
        total_created = sum(results)

        _method_elapsed_ms = (_time.perf_counter() - _method_t0) * 1000
        inverse_count = len(all_rels) - len(relationships)
        if inverse_count > 0:
            logger.debug(f"Included {inverse_count} inverse relationships")

        # Find slowest type group for quick diagnosis
        _slowest_type = max(_per_type_ms, key=lambda k: _per_type_ms[k], default="") if _per_type_ms else ""
        _slowest_ms = _per_type_ms.get(_slowest_type, 0.0)

        with trace_span(
            "khora.neo4j.create_relationships_batch",
            relationship_count=len(relationships),
            total_with_inverses=len(all_rels),
            type_group_count=len(type_groups),
            execution_group_count=_execution_group_count,
            density_reduced=rel_density_reduced,
            total_created=total_created,
            total_ms=round(_method_elapsed_ms, 2),
            hub_grouping_ms=round(_hub_grouping_ms, 2),
            slowest_type=_slowest_type,
            slowest_type_ms=round(_slowest_ms, 2),
        ):
            pass

        logger.debug(
            f"Batch created {total_created} relationships "
            f"({len(type_groups)} types, "
            f"total={_method_elapsed_ms:.0f}ms, "
            f"slowest={_slowest_type}@{_slowest_ms:.0f}ms)"
        )

        # Dynamically create indexes for observed relationship types
        await self._ensure_relationship_type_indexes(set(type_groups.keys()))

        return total_created

    _indexed_rel_types: set[str] = set()  # Per-process cache of already-indexed types

    async def _ensure_relationship_type_indexes(self, relationship_types: set[str]) -> None:
        """Dynamically create namespace_id indexes for observed relationship types.

        Uses a per-process cache to avoid repeated CREATE INDEX calls for types
        that have already been indexed in this process lifetime.
        """
        if not self._driver or not relationship_types:
            return
        new_types = relationship_types - self._indexed_rel_types
        if not new_types:
            return
        async with self._session() as session:
            for rel_type in new_types:
                sanitized = _NEO4J_LABEL_RE.sub("_", rel_type).upper()
                if not sanitized:
                    continue
                index_name = f"rel_{sanitized.lower()}_ns_dyn"
                query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR ()-[r:{sanitized}]-() ON (r.namespace_id)"
                try:
                    await session.run(query)
                    self._indexed_rel_types.add(rel_type)
                except Exception as e:
                    logger.debug(f"Dynamic index creation for {sanitized}: {e}")

    def _record_to_entity(self, node: dict[str, Any]) -> Entity:
        """Convert a Neo4j node to a domain Entity.

        Bi-temporal version properties (``version_valid_from``,
        ``version_valid_to``) are stored in ``entity.metadata`` under the
        keys ``"version_valid_from"`` and ``"version_valid_to"`` so that
        callers can inspect version boundaries without a model change.
        """
        meta = _deserialize_dict(node.get("metadata"))
        # Propagate bi-temporal version properties into metadata
        if node.get("version_valid_from"):
            meta["version_valid_from"] = node["version_valid_from"]
        if node.get("version_valid_to"):
            meta["version_valid_to"] = node["version_valid_to"]

        return Entity(
            id=UUID(node["id"]),
            namespace_id=UUID(node["namespace_id"]),
            name=node["name"],
            entity_type=node["entity_type"],
            description=node.get("description", ""),
            attributes=_deserialize_dict(node.get("attributes")),
            source_document_ids=[UUID(d) for d in node.get("source_document_ids", [])],
            source_chunk_ids=[UUID(c) for c in node.get("source_chunk_ids", [])],
            mention_count=node.get("mention_count", 1),
            valid_from=datetime.fromisoformat(node["valid_from"]) if node.get("valid_from") else None,
            valid_until=datetime.fromisoformat(node["valid_until"]) if node.get("valid_until") else None,
            confidence=node.get("confidence", 1.0),
            metadata=meta,
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else datetime.now(),
        )

    # =========================================================================
    # Relationship operations
    # =========================================================================

    async def create_relationship(self, relationship: Relationship) -> Relationship:
        """Create a relationship between entities."""

        params = _relationship_to_cypher_params(relationship)

        rel_type = _sanitize_neo4j_label(relationship.relationship_type)

        async def _create(tx: AsyncManagedTransaction) -> None:
            # Use dynamic relationship type with MERGE to prevent duplicates
            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            MERGE (source)-[r:{rel_type} {{namespace_id: $namespace_id}}]->(target)
            ON CREATE SET
                r.id = $id,
                r.description = $description,
                r.properties = $properties,
                r.source_document_ids = $source_document_ids,
                r.source_chunk_ids = $source_chunk_ids,
                r.valid_from = $valid_from,
                r.valid_until = $valid_until,
                r.confidence = $confidence,
                r.weight = $weight,
                r.metadata = $metadata,
                r.created_at = $created_at,
                r.updated_at = $updated_at
            ON MATCH SET
                r.description = CASE WHEN size($description) > size(coalesce(r.description, ''))
                    THEN $description ELSE r.description END,
                r.source_document_ids = (r.source_document_ids + [x IN $source_document_ids WHERE NOT x IN r.source_document_ids])[-100..],
                r.source_chunk_ids = (r.source_chunk_ids + [x IN $source_chunk_ids WHERE NOT x IN r.source_chunk_ids])[-250..],
                r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
                r.weight = CASE WHEN $weight > r.weight THEN $weight ELSE r.weight END,
                r.updated_at = $updated_at
            """
            await tx.run(query, **params)

        async with self._session() as session:
            await session.execute_write(_create)

        return relationship

    async def get_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> Relationship | None:
        """Get a relationship by ID, scoped to ``namespace_id`` (IGR-223).

        Source and target nodes must both belong to ``namespace_id`` so the
        result never leaks edges that cross into another namespace.
        """

        async with self._session() as session:
            result = await session.run(
                """
                MATCH (source:Entity {namespace_id: $namespace_id})-[r {id: $id, namespace_id: $namespace_id}]->(target:Entity {namespace_id: $namespace_id})
                RETURN r, source.id as source_id, target.id as target_id, type(r) as rel_type
                """,
                id=str(relationship_id),
                namespace_id=str(namespace_id),
            )
            record = await result.single()
            if record:
                return self._record_to_relationship(
                    record["r"],
                    record["source_id"],
                    record["target_id"],
                    record["rel_type"],
                )
            return None

    async def delete_relationship(self, relationship_id: UUID) -> bool:
        """Delete a relationship."""

        async def _delete(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                MATCH ()-[r {id: $id}]->()
                DELETE r
                RETURN count(r) as deleted
                """,
                id=str(relationship_id),
            )
            record = await result.single()
            return record["deleted"] if record else 0

        async with self._session() as session:
            deleted = await session.execute_write(_delete)
            return deleted > 0

    class RetirementRow(TypedDict):
        """A row for retire_orphaned_relationships_batch()."""

        relationship_id: UUID
        old_doc_id: UUID
        retired_at: datetime

    @trace(
        "khora.neo4j.retire_orphaned_relationships_batch",
        result=lambda r: {"retired_count": r},
    )
    async def retire_orphaned_relationships_batch(
        self,
        retirement_rows: list[RetirementRow],
    ) -> int:
        """Soft-retire orphaned relationships by stamping valid_until in-place.

        Only mutates relationships whose sole source was the replaced document
        (size(source_document_ids) = 1). Multi-sourced relationships are left
        untouched for separate survivor cleanup.

        Each row must contain:
            relationship_id: UUID — the relationship to retire
            old_doc_id: UUID — the document being replaced
            retired_at: datetime — timestamp for valid_until / updated_at

        Returns:
            Number of relationships actually retired (sole-sourced matches only),
            not the number of rows attempted.
        """
        if not retirement_rows:
            return 0
        rows = [
            {
                "relationship_id": str(r["relationship_id"]),
                "old_doc_id": str(r["old_doc_id"]),
                "retired_at": r["retired_at"].isoformat(),
            }
            for r in retirement_rows
        ]

        async def _retire(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                UNWIND $retirement_rows AS r
                MATCH ()-[rel {id: r.relationship_id}]-()
                WHERE size(rel.source_document_ids) = 1
                  AND r.old_doc_id IN rel.source_document_ids
                SET rel.valid_until = r.retired_at,
                    rel.updated_at = r.retired_at
                RETURN count(DISTINCT rel) AS retired
                """,
                retirement_rows=rows,
            )
            record = await result.single()
            return record["retired"] if record else 0

        async with self._session() as session:
            return await session.execute_write(_retire)

    @trace(
        "khora.neo4j.get_entity_relationships",
        include={"entity_id", "direction"},
        result=lambda r: {"result_count": len(r)},
    )
    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity, scoped to ``namespace_id`` (IGR-223).

        Both endpoint nodes are constrained to ``namespace_id`` so edges that
        cross into other namespaces never surface.
        """

        # Build relationship type filter
        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(_sanitize_neo4j_label(rt) for rt in relationship_types)

        # Build direction query — both endpoint nodes constrained to namespace.
        if direction == "outgoing":
            pattern = f"(e:Entity {{namespace_id: $namespace_id}})-[r{rel_filter}]->(other:Entity {{namespace_id: $namespace_id}})"
        elif direction == "incoming":
            pattern = f"(other:Entity {{namespace_id: $namespace_id}})-[r{rel_filter}]->(e:Entity {{namespace_id: $namespace_id}})"
        else:  # both
            pattern = f"(e:Entity {{namespace_id: $namespace_id}})-[r{rel_filter}]-(other:Entity {{namespace_id: $namespace_id}})"

        query = f"""
        MATCH {pattern}
        WHERE e.id = $entity_id
        RETURN properties(r) as r, e.id as source_id, other.id as target_id, type(r) as rel_type
        LIMIT $limit
        """

        async with self._session() as session:
            result = await session.run(
                query,
                entity_id=str(entity_id),
                namespace_id=str(namespace_id),
                limit=limit,
            )
            records = await result.data()
            return [
                self._record_to_relationship(r["r"], r["source_id"], r["target_id"], r["rel_type"]) for r in records
            ]

    def _record_to_relationship(
        self, rel: dict[str, Any], source_id: str, target_id: str, rel_type: str
    ) -> Relationship:
        """Convert a Neo4j relationship to a domain Relationship."""
        rel_id = rel.get("id")
        rel_ns = rel.get("namespace_id")
        if rel_id is None or rel_ns is None:
            logger.warning(
                f"Relationship missing id/namespace_id (type={rel_type}, "
                f"{source_id}->{target_id}); using synthesized identity"
            )
        return Relationship(
            id=UUID(rel_id) if rel_id else uuid4(),
            namespace_id=UUID(rel_ns) if rel_ns else uuid4(),
            source_entity_id=UUID(source_id),
            target_entity_id=UUID(target_id),
            relationship_type=rel_type,
            description=rel.get("description", ""),
            properties=_deserialize_dict(rel.get("properties")),
            source_document_ids=[UUID(d) for d in rel.get("source_document_ids", [])],
            source_chunk_ids=[UUID(c) for c in rel.get("source_chunk_ids", [])],
            valid_from=datetime.fromisoformat(rel["valid_from"]) if rel.get("valid_from") else None,
            valid_until=datetime.fromisoformat(rel["valid_until"]) if rel.get("valid_until") else None,
            confidence=rel.get("confidence", 1.0),
            weight=rel.get("weight", 1.0),
            metadata=_deserialize_dict(rel.get("metadata")),
            created_at=datetime.fromisoformat(rel["created_at"]) if rel.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(rel["updated_at"]) if rel.get("updated_at") else datetime.now(),
        )

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        """List all relationships in a namespace."""

        # Build relationship type filter
        rel_filter = f":{_sanitize_neo4j_label(relationship_type)}" if relationship_type else ""

        query = f"""
        MATCH (source)-[r{rel_filter}]->(target)
        WHERE r.namespace_id = $namespace_id
        RETURN properties(r) as rel_props, source.id as source_id, target.id as target_id, type(r) as rel_type
        ORDER BY r.created_at DESC
        SKIP $offset
        LIMIT $limit
        """

        async with self._session() as session:
            result = await session.run(
                query,
                namespace_id=str(namespace_id),
                offset=offset,
                limit=limit,
            )
            records = await result.data()
            return [
                self._record_to_relationship(r["rel_props"], r["source_id"], r["target_id"], r["rel_type"])
                for r in records
            ]

    # =========================================================================
    # Episode operations
    # =========================================================================

    async def create_episode(self, episode: Episode) -> Episode:
        """Create an episode node."""

        async def _create(tx: AsyncManagedTransaction) -> None:
            query = """
            CREATE (ep:Episode {
                id: $id,
                namespace_id: $namespace_id,
                name: $name,
                description: $description,
                occurred_at: $occurred_at,
                duration_seconds: $duration_seconds,
                entity_ids: $entity_ids,
                source_document_ids: $source_document_ids,
                source_chunk_ids: $source_chunk_ids,
                metadata: $metadata,
                created_at: $created_at,
                updated_at: $updated_at
            })
            """
            await tx.run(
                query,
                id=str(episode.id),
                namespace_id=str(episode.namespace_id),
                name=episode.name,
                description=episode.description,
                occurred_at=episode.occurred_at.isoformat(),
                duration_seconds=episode.duration_seconds,
                entity_ids=[str(e) for e in episode.entity_ids],
                source_document_ids=[str(d) for d in episode.source_document_ids],
                source_chunk_ids=[str(c) for c in episode.source_chunk_ids],
                metadata=_serialize_dict(episode.metadata),
                created_at=episode.created_at.isoformat(),
                updated_at=episode.updated_at.isoformat(),
            )

            # Create links to entities
            if episode.entity_ids:
                link_query = """
                MATCH (ep:Episode {id: $episode_id})
                MATCH (e:Entity) WHERE e.id IN $entity_ids
                CREATE (ep)-[:INVOLVES]->(e)
                """
                await tx.run(
                    link_query,
                    episode_id=str(episode.id),
                    entity_ids=[str(e) for e in episode.entity_ids],
                )

        async with self._session() as session:
            await session.execute_write(_create)

        return episode

    async def get_episode(self, episode_id: UUID, *, namespace_id: UUID) -> Episode | None:
        """Get an episode by ID, scoped to ``namespace_id`` (IGR-223)."""

        async with self._session() as session:
            result = await session.run(
                "MATCH (ep:Episode {id: $id, namespace_id: $namespace_id}) RETURN ep",
                id=str(episode_id),
                namespace_id=str(namespace_id),
            )
            record = await result.single()
            if record:
                return self._record_to_episode(record["ep"])
            return None

    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        """List episodes in a time range."""

        query = "MATCH (ep:Episode {namespace_id: $namespace_id})"
        params: dict[str, Any] = {"namespace_id": str(namespace_id)}
        conditions = []

        if start_time:
            conditions.append("ep.occurred_at >= $start_time")
            params["start_time"] = start_time.isoformat()
        if end_time:
            conditions.append("ep.occurred_at <= $end_time")
            params["end_time"] = end_time.isoformat()

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " RETURN ep ORDER BY ep.occurred_at DESC LIMIT $limit"
        params["limit"] = limit

        async with self._session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [self._record_to_episode(r["ep"]) for r in records]

    def _record_to_episode(self, node: dict[str, Any]) -> Episode:
        """Convert a Neo4j node to a domain Episode."""
        return Episode(
            id=UUID(node["id"]),
            namespace_id=UUID(node["namespace_id"]),
            name=node["name"],
            description=node.get("description", ""),
            occurred_at=datetime.fromisoformat(node["occurred_at"]),
            duration_seconds=node.get("duration_seconds"),
            entity_ids=[UUID(e) for e in node.get("entity_ids", [])],
            source_document_ids=[UUID(d) for d in node.get("source_document_ids", [])],
            source_chunk_ids=[UUID(c) for c in node.get("source_chunk_ids", [])],
            metadata=_deserialize_dict(node.get("metadata")),
            created_at=datetime.fromisoformat(node["created_at"]) if node.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(node["updated_at"]) if node.get("updated_at") else datetime.now(),
        )

    # =========================================================================
    # Graph traversal
    # =========================================================================

    @trace(
        "khora.neo4j.find_paths",
        include={"source_entity_id", "target_entity_id", "max_depth"},
        result=lambda r: {"path_count": len(r)},
    )
    async def find_paths(
        self,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        namespace_id: UUID,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find paths between two entities."""

        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(_sanitize_neo4j_label(rt) for rt in relationship_types)

        # All nodes on the path — endpoints AND intermediates — must share
        # $namespace_id so the traversal never crosses tenants (IGR-223).
        query = f"""
        MATCH path = shortestPath(
            (source:Entity {{id: $source_id, namespace_id: $namespace_id}})-[r{rel_filter}*1..{max_depth}]-(target:Entity {{id: $target_id, namespace_id: $namespace_id}})
        )
        WHERE ALL(n IN nodes(path) WHERE n.namespace_id = $namespace_id)
        RETURN path
        LIMIT 10
        """

        async with self._session() as session:
            result = await session.run(
                query,
                source_id=str(source_entity_id),
                target_id=str(target_entity_id),
                namespace_id=str(namespace_id),
            )
            records = await result.data()

            paths = []
            for record in records:
                path = record["path"]
                path_elements = []
                for element in path:
                    if hasattr(element, "items"):  # Node
                        path_elements.append({"type": "node", "data": _element_to_dict(element)})
                    else:  # Relationship
                        path_elements.append({"type": "relationship", "data": _element_to_dict(element)})
                paths.append(path_elements)

            return paths

    @trace(
        "khora.neo4j.get_neighborhood",
        include={"entity_id", "depth"},
        result=lambda r: {"node_count": len(r.get("entities", [])), "rel_count": len(r.get("relationships", []))},
    )
    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity up to a certain depth, scoped to
        ``namespace_id``.

        The traversal MUST NOT cross into other namespaces (IGR-223). Because
        APOC's ``subgraphAll`` does not let us constrain expanded nodes by
        property — relationshipFilter cannot filter destination labels by
        namespace — we use vanilla Cypher with the namespace predicate baked
        into the pattern. Returns ``{"entities": [], "relationships": []}``
        on query timeout.
        """

        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(_sanitize_neo4j_label(rt) for rt in relationship_types)

        # Vanilla Cypher (no APOC) — the namespace predicate has to hold for
        # *both* endpoint nodes at every hop. We can't express that cleanly
        # inside apoc.path.subgraphAll, so we stick to the pattern form.
        query = f"""
        MATCH (center:Entity {{id: $entity_id, namespace_id: $namespace_id}})-[r{rel_filter}*1..{depth}]-(other:Entity {{namespace_id: $namespace_id}})
        RETURN collect(DISTINCT other) as nodes, collect(DISTINCT [rel IN r | {{props: properties(rel), type: type(rel)}}]) as relationships
        LIMIT $limit
        """

        async def _work(tx):
            result = await tx.run(
                query,
                entity_id=str(entity_id),
                namespace_id=str(namespace_id),
                limit=limit,
            )
            return await result.single()

        if self._timed_unit_of_work is not None:
            _work = self._timed_unit_of_work(_work)

        try:
            async with self._session() as session:
                record = await session.execute_read(_work)
        except ClientError as exc:
            if exc.code in _NEO4J_TIMEOUT_CODES:
                with trace_span(
                    "khora.neo4j.get_neighborhood.timeout",
                    timeout_s=self._query_timeout,
                    entity_id=str(entity_id),
                    code=exc.code,
                    timeout_occurred=True,
                ):
                    pass
                logger.warning(
                    "Neo4j get_neighborhood timed out after {timeout}s "
                    "(entity_id={eid}, code={code}); returning empty neighborhood",
                    timeout=self._query_timeout,
                    eid=entity_id,
                    code=exc.code,
                    timeout_occurred=True,
                )
                return {"entities": [], "relationships": []}
            raise

        if record:
            nodes = [_element_to_dict(n) for n in record.get("nodes", [])]
            relationships = []
            for rel_list in record.get("relationships", []):
                if rel_list:
                    for r in rel_list if isinstance(rel_list, list) else [rel_list]:
                        if r:
                            relationships.append({**r.get("props", {}), "relationship_type": r.get("type")})
            return {"entities": nodes, "relationships": relationships}

        return {"entities": [], "relationships": []}

    @trace(
        "khora.neo4j.get_neighborhoods_batch",
        include={"entity_ids", "depth"},
        result=lambda r: {"result_count": len(r)},
    )
    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        """Get neighborhoods for multiple entities in parallel, scoped to
        ``namespace_id``.

        Seed entities and every node reached during traversal are constrained
        to ``namespace_id`` so the result never crosses into another
        namespace (IGR-223).

        Args:
            entity_ids: List of entity IDs
            namespace_id: Caller's namespace; seeds in any other namespace
                are silently dropped, and traversal never crosses out.
            depth: Max traversal depth
            relationship_types: Optional relationship type filter
            limit_per_entity: Max nodes per entity neighborhood

        Returns:
            Dictionary mapping entity ID to neighborhood data.
            Returns ``{}`` on query timeout when ``query_timeout`` is set.
        """
        if not entity_ids:
            return {}

        id_strings = [str(eid) for eid in entity_ids]

        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(_sanitize_neo4j_label(rt) for rt in relationship_types)

        # UNWIND seeds, expand only nodes that share namespace_id. Both the
        # center and every node hit during traversal must be in $namespace_id.
        query = f"""
        UNWIND $entity_ids AS eid
        MATCH (center:Entity {{id: eid, namespace_id: $namespace_id}})
        OPTIONAL MATCH (center)-[r{rel_filter}*1..{depth}]-(other:Entity {{namespace_id: $namespace_id}})
        With eid, center, collect(DISTINCT other)[0..$limit] as neighbors, collect(DISTINCT [rel IN r | {{props: properties(rel), type: type(rel)}}])[0..$limit] as rels
        RETURN eid, neighbors, rels
        """

        async def _work(tx):
            result = await tx.run(
                query,
                entity_ids=id_strings,
                namespace_id=str(namespace_id),
                limit=limit_per_entity,
            )
            return await result.data()

        if self._timed_unit_of_work is not None:
            _work = self._timed_unit_of_work(_work)

        try:
            async with self._session() as session:
                records = await session.execute_read(_work)
        except ClientError as exc:
            if exc.code in _NEO4J_TIMEOUT_CODES:
                with trace_span(
                    "khora.neo4j.get_neighborhoods_batch.timeout",
                    timeout_s=self._query_timeout,
                    entity_count=len(entity_ids),
                    code=exc.code,
                    timeout_occurred=True,
                ):
                    pass
                logger.warning(
                    "Neo4j get_neighborhoods_batch timed out after {timeout}s "
                    "(entity_count={n}, code={code}); returning empty dict",
                    timeout=self._query_timeout,
                    n=len(entity_ids),
                    code=exc.code,
                    timeout_occurred=True,
                )
                return {}
            raise

        neighborhoods = {}
        for record in records:
            eid = UUID(record["eid"])
            nodes = [_element_to_dict(n) for n in (record.get("neighbors") or []) if n]
            relationships = []
            for rel_list in record.get("rels") or []:
                if rel_list:
                    for r in rel_list if isinstance(rel_list, list) else [rel_list]:
                        if r:
                            relationships.append({**r.get("props", {}), "relationship_type": r.get("type")})
            neighborhoods[eid] = {"entities": nodes, "relationships": relationships}

        return neighborhoods

    async def get_temporal_neighbors(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        valid_after: datetime | None = None,
        valid_before: datetime | None = None,
        max_hops: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get neighboring entities connected via relationships within a time window.

        Traverses 1..max_hops relationships and filters by their valid_from/valid_until
        properties to return only temporally relevant neighbors.

        Args:
            entity_id: Starting entity ID
            namespace_id: Namespace to restrict traversal to
            valid_after: Only traverse relationships where valid_from >= this value
            valid_before: Only traverse relationships where valid_until <= this value
            max_hops: Maximum path length (1–4 recommended)
            limit: Maximum neighbor entities to return

        Returns:
            List of neighbor entity property dicts
        """

        params: dict[str, Any] = {
            "entity_id": str(entity_id),
            "namespace_id": str(namespace_id),
            "limit": limit,
        }

        rel_conditions: list[str] = []
        if valid_after is not None:
            rel_conditions.append("(rel.valid_from IS NULL OR rel.valid_from >= $valid_after)")
            params["valid_after"] = valid_after.isoformat()
        if valid_before is not None:
            rel_conditions.append("(rel.valid_until IS NULL OR rel.valid_until <= $valid_before)")
            params["valid_before"] = valid_before.isoformat()

        temporal_filter = ""
        if rel_conditions:
            conditions_str = " AND ".join(rel_conditions)
            temporal_filter = f"\n  AND ALL(rel IN r WHERE {conditions_str})"

        query = f"""
        MATCH (e:Entity {{id: $entity_id, namespace_id: $namespace_id}})
        MATCH (e)-[r*1..{max_hops}]-(neighbor:Entity)
        WHERE neighbor.namespace_id = $namespace_id
          AND neighbor.id <> $entity_id{temporal_filter}
        RETURN DISTINCT properties(neighbor) AS props
        LIMIT $limit
        """

        async def _read(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(query, **params)
            records = await result.data()
            return [r["props"] for r in records]

        async with self._session() as session:
            return await session.execute_read(_read)

    async def create_session_links(
        self,
        namespace_id: UUID,
    ) -> int:
        """Create NEXT_SESSION edges between consecutive session chunks.

        Reads Chunk nodes from the namespace, groups them by session_id stored
        in their metadata, orders sessions by earliest chunk timestamp, and
        creates NEXT_SESSION relationships from the last chunk of each session
        to the first chunk of the next session.

        Args:
            namespace_id: Namespace to process

        Returns:
            Number of NEXT_SESSION edges created
        """

        # Step 1: Fetch all chunk IDs, timestamps, and metadata
        async def _fetch_chunks(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(
                """
                MATCH (c:Chunk {namespace_id: $namespace_id})
                RETURN c.id AS id,
                       coalesce(c.occurred_at, c.created_at) AS ts,
                       c.metadata AS metadata
                ORDER BY ts
                """,
                namespace_id=str(namespace_id),
            )
            return await result.data()

        async with self._session() as session:
            rows = await session.execute_read(_fetch_chunks)

        if not rows:
            return 0

        # Step 2: Group chunks by session_id (stored in serialized metadata JSON)
        sessions: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            metadata = _deserialize_dict(row.get("metadata"))
            session_id = (metadata or {}).get("session_id")
            if not session_id:
                continue
            sessions.setdefault(str(session_id), []).append(row)

        if len(sessions) < 2:
            return 0

        # Step 3: Sort sessions by earliest chunk timestamp
        ordered_sessions = sorted(
            sessions.values(),
            key=lambda chunks: min(
                (c["ts"] for c in chunks if c.get("ts")),
                default="",
            ),
        )

        # Step 4: Build (last_of_session_A, first_of_session_B) link pairs
        links = [
            {"from_id": ordered_sessions[i][-1]["id"], "to_id": ordered_sessions[i + 1][0]["id"]}
            for i in range(len(ordered_sessions) - 1)
        ]

        if not links:
            return 0

        async def _create_links(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                UNWIND $links AS link
                MATCH (a:Chunk {id: link.from_id})
                MATCH (b:Chunk {id: link.to_id})
                MERGE (a)-[r:NEXT_SESSION]->(b)
                RETURN count(r) AS created
                """,
                links=links,
            )
            record = await result.single()
            return record["created"] if record else 0

        async with self._session() as session:
            created = await session.execute_write(_create_links)

        logger.debug(f"Created {created} NEXT_SESSION edges for namespace {namespace_id}")
        return created

    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        """Search entities by attribute value."""

        query = """
        MATCH (e:Entity {namespace_id: $namespace_id})
        WHERE e.attributes[$attribute_name] = $attribute_value
        RETURN e
        LIMIT $limit
        """

        async with self._session() as session:
            result = await session.run(
                query,
                namespace_id=str(namespace_id),
                attribute_name=attribute_name,
                attribute_value=attribute_value,
                limit=limit,
            )
            records = await result.data()
            return [self._record_to_entity(r["e"]) for r in records]

    @trace("khora.neo4j.count_relationships", include={"namespace_id"})
    async def count_relationships(self, namespace_id: UUID) -> int:
        """Count relationships in a namespace using sampling estimation.

        Counts total entities via index, then samples up to 1000 in storage
        order and extrapolates outbound degree.  Exact when <=1000 entities.
        """
        query = """
        MATCH (e:Entity {namespace_id: $namespace_id})
        WITH count(e) AS total_n
        MATCH (e2:Entity {namespace_id: $namespace_id})
        WITH total_n, e2 LIMIT 1000
        OPTIONAL MATCH (e2)-[out]->()
        WITH total_n, count(out) AS sampled_out, count(DISTINCT e2) AS sampled_n
        RETURN CASE WHEN sampled_n = 0 THEN 0
               ELSE toInteger(total_n * (toFloat(sampled_out) / sampled_n)) END AS estimate
        """

        async def _work(tx):
            result = await tx.run(query, namespace_id=str(namespace_id))
            record = await result.single()
            return record["estimate"] or 0 if record else 0

        if self._timed_unit_of_work is not None:
            _work = self._timed_unit_of_work(_work)

        async with self._session() as session:
            return await session.execute_read(_work)

    async def remap_source_document_ids_batch(
        self,
        *,
        entity_survivors: list[dict[str, str]],
        relationship_survivors: list[dict[str, str]],
    ) -> None:
        """Remap source_document_ids for entities and relationships after dedup.

        Idempotent on repeated application against the same
        ``(old_doc_id, new_doc_id)`` pair: if ``new_doc_id`` is already
        present in the array, the append is suppressed so a retry never
        duplicates it. Self-heal can retry the
        replace lifecycle and the caller's remap payload will re-include
        surviving entity/relationship rows, so this method must be
        safe to run repeatedly.
        """

        if entity_survivors:

            async def _remap_entities(tx: AsyncManagedTransaction) -> None:
                # Single-branch, tolerant form:
                #   filtered = array with every occurrence of old_doc_id removed
                #   append new_doc_id only if not already present
                # Idempotent on retry AND tolerant of pre-existing duplicates
                # (e.g. ``[old, old, new]`` from a partially-written prior run
                # collapses cleanly to ``[new]`` instead of ``[new, new]``).
                query = """
                UNWIND $survivors AS s
                MATCH (e:Entity {id: s.entity_id})
                WITH e, s, [x IN coalesce(e.source_document_ids, []) WHERE x <> s.old_doc_id] AS filtered
                SET e.source_document_ids = filtered +
                    CASE WHEN s.new_doc_id IN filtered THEN [] ELSE [s.new_doc_id] END
                """
                await tx.run(query, survivors=entity_survivors)

            async with self._session() as session:
                await session.execute_write(_remap_entities)

        if relationship_survivors:

            async def _remap_relationships(tx: AsyncManagedTransaction) -> None:
                # Same idempotent/tolerant shape as the entity variant.
                # ``WITH DISTINCT rel`` protects self-loops: an undirected
                # ``()-[rel]-()`` match returns the same edge twice when
                # source == target, which without DISTINCT would apply the
                # SET twice and duplicate ``new_doc_id``.
                query = """
                UNWIND $survivors AS s
                MATCH ()-[rel {id: s.relationship_id}]-()
                WITH DISTINCT rel, s
                WITH rel, s, [x IN coalesce(rel.source_document_ids, []) WHERE x <> s.old_doc_id] AS filtered
                SET rel.source_document_ids = filtered +
                    CASE WHEN s.new_doc_id IN filtered THEN [] ELSE [s.new_doc_id] END
                """
                await tx.run(query, survivors=relationship_survivors)

            async with self._session() as session:
                await session.execute_write(_remap_relationships)

    async def delete_entities_batch(
        self,
        entity_ids: list[UUID],
        namespace_id: UUID,
    ) -> int:
        """Hard-delete entities by id, scoped to a namespace.

        Used by the forget cascade to remove orphan entities (those whose
        only ``source_document_ids`` value is the document being forgotten).
        Unlike ``retire_orphaned_entities_batch`` this does not snapshot
        into :EntityVersion — forget is a GDPR-style removal, not an audit
        retirement.

        Returns the number of nodes deleted.
        """
        if not entity_ids:
            return 0

        async def _delete(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                UNWIND $entity_ids AS eid
                MATCH (e:Entity {id: eid, namespace_id: $namespace_id})
                DETACH DELETE e
                RETURN count(e) AS deleted
                """,
                entity_ids=[str(eid) for eid in entity_ids],
                namespace_id=str(namespace_id),
            )
            record = await result.single()
            return record["deleted"] if record else 0

        async with self._session() as session:
            return await session.execute_write(_delete)

    async def delete_relationships_batch(self, relationship_ids: list[UUID]) -> int:
        """Hard-delete relationships by their ``id`` property.

        Sibling to :meth:`delete_entities_batch` for the relationship side
        of the forget cascade. Matches edges across any direction since
        Cypher's ``-[r]->`` is asymmetric; we use ``()-[r]-()`` and rely on
        ``WITH DISTINCT`` to dedupe self-loop double-matches (same shape as
        :meth:`remap_source_document_ids_batch`).

        Returns the number of relationships deleted.
        """
        if not relationship_ids:
            return 0

        async def _delete(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                UNWIND $relationship_ids AS rid
                MATCH ()-[r {id: rid}]-()
                WITH DISTINCT r
                DELETE r
                RETURN count(r) AS deleted
                """,
                relationship_ids=[str(rid) for rid in relationship_ids],
            )
            record = await result.single()
            return record["deleted"] if record else 0

        async with self._session() as session:
            return await session.execute_write(_delete)

    async def remove_document_from_entity_sources_batch(
        self,
        entity_ids: list[UUID],
        document_id: UUID,
        namespace_id: UUID,
    ) -> int:
        """Strip ``document_id`` from survivor entities' source arrays.

        Survivors are multi-source entities whose ``source_document_ids``
        contains more than just the forgotten document; the forget cascade
        keeps the node but removes the doc_id from ``source_document_ids``.

        Returns the number of entities updated.
        """
        if not entity_ids:
            return 0

        async def _update(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                UNWIND $entity_ids AS eid
                MATCH (e:Entity {id: eid, namespace_id: $namespace_id})
                SET e.source_document_ids =
                        [d IN coalesce(e.source_document_ids, []) WHERE d <> $doc_id]
                RETURN count(e) AS updated
                """,
                entity_ids=[str(eid) for eid in entity_ids],
                namespace_id=str(namespace_id),
                doc_id=str(document_id),
            )
            record = await result.single()
            return record["updated"] if record else 0

        async with self._session() as session:
            return await session.execute_write(_update)

    async def remove_document_from_relationship_sources_batch(
        self,
        relationship_ids: list[UUID],
        document_id: UUID,
    ) -> int:
        """Strip ``document_id`` from survivor relationships' source arrays.

        Sibling to :meth:`remove_document_from_entity_sources_batch` for
        the edge side of the cascade.

        Returns the number of relationships updated.
        """
        if not relationship_ids:
            return 0

        async def _update(tx: AsyncManagedTransaction) -> int:
            result = await tx.run(
                """
                UNWIND $relationship_ids AS rid
                MATCH ()-[r {id: rid}]-()
                WITH DISTINCT r
                SET r.source_document_ids =
                        [d IN coalesce(r.source_document_ids, []) WHERE d <> $doc_id]
                RETURN count(r) AS updated
                """,
                relationship_ids=[str(rid) for rid in relationship_ids],
                doc_id=str(document_id),
            )
            record = await result.single()
            return record["updated"] if record else 0

        async with self._session() as session:
            return await session.execute_write(_update)

    async def fetch_document_extraction_state(
        self, document_id: UUID, *, namespace_id: UUID
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fetch the entity and relationship state linked to a document.

        Used by ``StorageCoordinator.replace_document_extraction()`` to compute
        retirement / survivor sets in Python before issuing graph mutations.
        Namespace-scoped to match the rest of the backend's traversal helpers
        and to keep reads on the namespace-partitioned indexes.

        ``relationship_type`` in the result is the Cypher-sanitized label
        (``_sanitize_neo4j_label`` — upper-cased, non-alphanumerics →
        underscore), not the raw Python string.  Callers that compare against
        ``Relationship.relationship_type`` must apply the same sanitizer.

        Returns:
            (entities, relationships) where each record carries the identity
            fields and the current ``source_document_ids`` length sufficient to
            apply the retirement filter.
        """
        doc_id_str = str(document_id)
        ns_id_str = str(namespace_id)

        async def _fetch_entities(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(
                """
                MATCH (e:Entity {namespace_id: $namespace_id})
                WHERE $doc_id IN e.source_document_ids
                RETURN e.id AS id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       e.namespace_id AS namespace_id,
                       size(e.source_document_ids) AS source_document_count
                """,
                doc_id=doc_id_str,
                namespace_id=ns_id_str,
            )
            return await result.data()

        async def _fetch_relationships(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
            result = await tx.run(
                """
                MATCH (src:Entity {namespace_id: $namespace_id})-[rel]->(tgt:Entity {namespace_id: $namespace_id})
                WHERE $doc_id IN rel.source_document_ids
                RETURN rel.id AS id,
                       src.id AS source_entity_id,
                       tgt.id AS target_entity_id,
                       type(rel) AS relationship_type,
                       size(rel.source_document_ids) AS source_document_count
                """,
                doc_id=doc_id_str,
                namespace_id=ns_id_str,
            )
            return await result.data()

        async with self._session() as session:
            entities = await session.execute_read(_fetch_entities)
            relationships = await session.execute_read(_fetch_relationships)
        return entities, relationships
