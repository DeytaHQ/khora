"""Primary API for Khora — the top-level facade class.

This is the main entry point for using Khora as a library.
Provides a simple, unified interface for memory storage and retrieval.

The Khora class is a thin facade that delegates to pluggable engines.
The default engine is "vectorcypher" which uses knowledge graphs, vectors, and LLM extraction.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import warnings
from collections.abc import AsyncGenerator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger

from khora.config import KhoraConfig, load_config
from khora.core.diagnostics import SkipReason
from khora.core.models import Chunk, CommunityNode, Document, Entity, MemoryNamespace
from khora.query import SearchMode
from khora.telemetry import bounded_text_hash, trace_span
from khora.telemetry.metrics import metric_counter

# Module-level counter — a dangling ref means an engine emitted a chunk /
# entity / relationship pointing at a document_id that the relational store
# could not resolve (deleted, namespace mismatch, or replication lag). No
# namespace_id label by cardinality contract (see CLAUDE.md).
_RECALL_DANGLING_REF_COUNTER = metric_counter(
    "khora.recall.dangling_ref",
    description="Dangling document references in recall results, by referrer kind.",
)

_ORPHANS_RECLAIMED_COUNTER = metric_counter(
    "khora.documents.orphans_reclaimed_total",
    unit="1",
    description="Stale orphaned documents reclaimed by the pending processor's crash-recovery scan, "
    "by their prior status (pending or processing).",
)

# #932: a queued document identity could not be re-loaded at dequeue time
# (the row was deleted / forgotten between enqueue and dequeue). No
# namespace_id label by cardinality contract (see CLAUDE.md).
_PROCESSOR_DOC_MISSING_COUNTER = metric_counter(
    "khora.documents.processor.degraded_total",
    unit="1",
    description="Queued documents skipped by the pending processor because the row was gone at dequeue time, "
    "by reason.",
)


def _coerce_session_id_from_dict(metadata: dict[str, Any] | None) -> UUID | None:
    """Pull ``session_id`` out of a metadata dict and coerce to UUID (#620).

    Mirrors the ingest pipeline helper. Returns ``None`` for missing /
    malformed values rather than raising so bad upstream metadata doesn't
    crash a batch submit.
    """
    if not metadata:
        return None
    value = metadata.get("session_id")
    if value is None or value == "":
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def _normalize_recall_bound(value: datetime | None) -> datetime | None:
    """Normalize a deprecated recall bound to UTC (naive → UTC, aware → UTC)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_exc_summary(exc: BaseException, *, max_len: int = 200) -> str:
    """Return a bounded, safe one-line summary of *exc* for logging.

    Primarily targets SQLAlchemy's asyncpg-wrapped ``DBAPIError`` family
    (e.g. ``IntegrityError``) — the ``.orig`` / ``__cause__`` preference and
    the ``[SQL:]`` / ``[parameters:]`` stripping below are tailored to that
    exception shape. Plain (non-DB) exceptions pass through the same path
    harmlessly.

    A bare ``{exc}`` (or ``exc_info=True``) on a SQLAlchemy ``IntegrityError``
    / ``DBAPIError`` renders the full failed statement *and its bind-parameter
    tuple* — which for a document INSERT is the entire document content and
    metadata. That is a content-leak and a log-bloat hazard, so we never log
    the raw ``str()`` of a DB exception.

    Prefer the underlying driver exception (SQLAlchemy exposes it as ``.orig``
    / ``__cause__``); the asyncpg message it carries has no bind-param tuple.
    Only when neither is present do we fall back to ``str(exc)``, and even
    then we hard-cut the ``[SQL: ...]`` / ``[parameters: ...]`` tail that
    SQLAlchemy appends so a stray wrapper without ``.orig`` can't leak the
    params. The result is newline-stripped, length-capped, and class-name
    prefixed so the log stays diagnostic.
    """
    underlying = getattr(exc, "orig", None) or exc.__cause__
    message = str(underlying) if underlying is not None else str(exc)
    # Defense in depth: SQLAlchemy appends "\n[SQL: ...]\n[parameters: ...]"
    # to its str(); cut everything from the first such marker regardless of
    # which branch produced ``message``.
    for marker in ("[SQL:", "[parameters:"):
        idx = message.find(marker)
        if idx != -1:
            message = message[:idx]
    message = message.replace("\n", " ").strip()
    if len(message) > max_len:
        message = message[:max_len] + "…"
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _is_undefined_table_error(exc: BaseException) -> bool:
    """Return True if *exc* is (or wraps) a Postgres "undefined table" error.

    SQLSTATE 42P01 is raised by asyncpg as ``UndefinedTableError`` and wrapped
    by SQLAlchemy as ``ProgrammingError``. We don't import either type here to
    avoid a hard dependency on the postgres backend at module-import time —
    duck-type via the SQLSTATE attribute on the underlying driver exception.
    """
    for candidate in (exc, getattr(exc, "orig", None), getattr(exc, "__cause__", None)):
        if candidate is None:
            continue
        if getattr(candidate, "sqlstate", None) == "42P01":
            return True
    return False


class _GlobalChunkSemaphore:
    """Counting semaphore supporting bulk acquire/release for chunk windowing.

    asyncio.Semaphore only supports acquire/release of 1 unit at a time.
    This uses asyncio.Condition to support acquiring N tokens at once,
    ensuring total chunks in flight across all concurrent submit_batch
    calls stay within the global limit.

    If n > capacity in acquire(), n is clamped to capacity to avoid
    permanent deadlock. This can occur when per-call max_chunks_in_flight
    exceeds the semaphore capacity (i.e. conflicting values across calls).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._in_flight = 0
        self._condition = asyncio.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    async def acquire(self, n: int) -> int:
        """Block until n tokens are available, then acquire them. Returns tokens acquired."""
        # Clamp to capacity to avoid permanent deadlock when n > capacity.
        n = min(n, self._capacity)
        async with self._condition:
            while self._in_flight + n > self._capacity:
                await self._condition.wait()
            self._in_flight += n
        return n

    async def release(self, n: int) -> None:
        """Release n tokens and wake any waiters."""
        async with self._condition:
            if self._in_flight < n:
                raise RuntimeError(f"Semaphore release({n}) would underflow _in_flight={self._in_flight}")
            self._in_flight -= n
            self._condition.notify_all()


if TYPE_CHECKING:
    from pathlib import Path

    from khora.core.models import Relationship
    from khora.engines.protocol import MemoryEngineProtocol
    from khora.extraction.chunkers import ChunkStrategy
    from khora.extraction.skills import ExpertiseConfig
    from khora.filter import RecallFilter
    from khora.storage import StorageConfig, StorageCoordinator


# LLMUsage is a public API type consumed by external cost-tracking integrations.
# Changes to field names or types require a coordinated release.
@dataclass(slots=True, frozen=True)
class LLMUsage:
    """A single LLM API call's token usage.

    Read-only value object — Khora produces it, consumers read it.
    """

    operation: str
    """Logical operation name (e.g. "entity_extraction", "embedding")."""
    model: str
    """Model identifier (e.g. "gpt-4o", "text-embedding-3-small")."""
    prompt_tokens: int
    completion_tokens: int
    """0 for embeddings."""
    total_tokens: int
    latency_ms: float
    batch_size: int = 1
    """>1 for embedding batches."""
    cost_usd: float = 0.0
    """Estimated USD cost via litellm pricing tables. 0.0 when unknown."""


def _safe_completion_cost(
    response: Any = None,
    *,
    model: str = "",
    call_type: str = "acompletion",
) -> float:
    """Best-effort USD cost from litellm pricing tables.

    Returns 0.0 on any failure (unknown model, missing litellm, etc.).
    """
    try:
        import litellm

        return float(
            litellm.completion_cost(
                completion_response=response,
                model=model,
                call_type=call_type,
            )
        )
    except Exception:
        return 0.0


@dataclass(slots=True, frozen=True)
class _OperationUsage:
    """Per-operation aggregate."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    call_count: int


@dataclass(slots=True, frozen=True)
class UsageSummary:
    """Aggregate view over a list of :class:`LLMUsage` entries."""

    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: float
    by_operation: dict[str, _OperationUsage]
    by_model: dict[str, _OperationUsage]

    @staticmethod
    def from_usage(entries: Iterable[LLMUsage]) -> UsageSummary:
        """Build a summary from LLMUsage entries."""
        prompt = comp = total = 0
        cost = latency = 0.0
        ops: dict[str, list[LLMUsage]] = {}
        models: dict[str, list[LLMUsage]] = {}
        for u in entries:
            prompt += u.prompt_tokens
            comp += u.completion_tokens
            total += u.total_tokens
            cost += u.cost_usd
            latency += u.latency_ms
            ops.setdefault(u.operation, []).append(u)
            models.setdefault(u.model, []).append(u)

        def _agg(items: list[LLMUsage]) -> _OperationUsage:
            return _OperationUsage(
                prompt_tokens=sum(i.prompt_tokens for i in items),
                completion_tokens=sum(i.completion_tokens for i in items),
                total_tokens=sum(i.total_tokens for i in items),
                cost_usd=sum(i.cost_usd for i in items),
                call_count=len(items),
            )

        return UsageSummary(
            total_prompt_tokens=prompt,
            total_completion_tokens=comp,
            total_tokens=total,
            total_cost_usd=cost,
            total_latency_ms=latency,
            by_operation={k: _agg(v) for k, v in ops.items()},
            by_model={k: _agg(v) for k, v in models.items()},
        )


@dataclass(slots=True, frozen=True)
class RememberResult:
    """Result of a remember operation."""

    document_id: UUID
    namespace_id: UUID
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    # Issue #907 (ADR-001): un-remappable relationships that the ingest
    # pipeline dropped because the source / target entity could not be
    # resolved to a canonical id. ``relationships_created`` counts what
    # was actually persisted; this counts what was silently discarded so
    # callers can detect partial success. Always 0 on engines that do
    # not run the shared ingest pipeline. Also reflected as a Degradation
    # entry under ``metadata["degradations"]`` when non-zero.
    relationships_skipped: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_usage: list[LLMUsage] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class BatchResult:
    """Result of remember_batch() operation."""

    total: int
    processed: int
    skipped: int
    failed: int
    chunks: int
    entities: int
    relationships: int
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_usage: list[LLMUsage] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class Stats:
    """Namespace statistics.

    ``metadata`` carries ADR-001 failure-observability records. When a
    counter could not run (backend lacks the method, or it raised), the
    int field stays ``0`` but ``metadata['errors']`` holds an ``ErrorRecord``
    so callers can distinguish "couldn't count" from "counted zero".
    See ``docs/architecture/failure-observability-contract.md``.
    """

    documents: int
    chunks: int
    entities: int
    relationships: int
    last_activity_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DocumentResult:
    """Result of processing a single document via submit_batch().

    Produced by the background worker and delivered to the on_result callback
    as each document completes (or fails) processing.
    """

    document_id: UUID
    """Row-level ID of the pre-created document record."""
    namespace_id: UUID
    success: bool
    error: str | None = None
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_created: int = 0
    llm_usage: list[LLMUsage] = field(default_factory=list)
    skipped: bool = False
    """True when re-processing was skipped. Set for documents in COMPLETED, PROCESSING,
    or ARCHIVED state (unless reprocess_archived=True). Callers should not treat skipped
    results as errors."""
    external_id: str | None = None
    """Caller-supplied opaque identifier from Document.external_id.
    Allows the caller to map each result back to its source row without
    a separate database lookup (e.g. for incremental checkpoint advancement)."""


@dataclass
class BatchHandle:
    """Handle returned by submit_batch() for tracking deferred batch processing.

    Documents are persisted as PENDING before this handle is returned.
    Background processing runs after return; use wait() to block until done.

    Attributes:
        batch_id: Unique identifier for this batch submission.
        total: Total number of documents in the batch.
        completed: Number of documents processed so far (success or failure).
        is_done: True when all documents have been processed.
    """

    batch_id: UUID
    total: int
    _completed: int = field(default=0, init=False, repr=False)
    _failed: int = field(default=0, init=False, repr=False)
    _done_event: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    @property
    def completed(self) -> int:
        """Number of documents processed (success + failure)."""
        return self._completed

    @property
    def failed(self) -> int:
        """Number of documents that failed processing."""
        return self._failed

    @property
    def is_done(self) -> bool:
        """True when all documents have been processed."""
        return self._done_event.is_set()

    async def wait(self) -> None:
        """Block until all documents in the batch have been processed."""
        await self._done_event.wait()

    def _record_result(self, result: DocumentResult) -> None:
        """Internal: update counters after a document completes."""
        self._completed += 1
        if not result.success:
            self._failed += 1

    def _mark_done(self) -> None:
        """Internal: signal that all documents have been processed."""
        self._done_event.set()


@dataclass
class _BatchRegistration:
    """In-memory tracking for an active submit_batch call.

    Links document IDs to their BatchHandle and on_result callback so the
    unified pending processor can deliver results back to the correct batch.
    """

    handle: BatchHandle
    on_result: Callable[[int, int, DocumentResult], None]
    namespace_id: UUID
    pre_failed_doc_ids: set[UUID] = field(default_factory=set)
    _remaining: int = 0
    # Per-batch concurrency cap (#838). When set, the unified pending processor
    # acquires this semaphore around _process_pending_item_impl for items
    # belonging to this batch, capping in-flight processing of this batch's
    # documents at `max_concurrent` regardless of how many global pool workers
    # are available. None for orphan-recovery items (no batch to limit).
    concurrency_sem: asyncio.Semaphore | None = None

    def fire_result(self, result: DocumentResult) -> None:
        """Record a result and fire the callback. Mark handle done when all results delivered."""
        self.handle._record_result(result)
        try:
            self.on_result(self.handle.completed, self.handle.total, result)
        except Exception as cb_exc:
            logger.warning(f"pending_processor: on_result callback raised: {cb_exc}")
        self._remaining -= 1
        if self._remaining <= 0:
            self.handle._mark_done()


@dataclass(slots=True, frozen=True)
class _ProcessorItem:
    """Work item for the unified pending processor.

    Carries a document *identity* (``doc_id`` + ``namespace_id``), not the
    full :class:`Document`. The document is already persisted as PENDING
    before it is enqueued, so the worker re-loads it from storage at
    dequeue time (#932). This keeps each queued item ~100 bytes regardless
    of document size, so the unbounded queue holding a large backlog is no
    longer a memory problem - peak content RAM is bounded by the number of
    concurrently-draining workers, each holding one re-fetched Document.
    """

    doc_id: UUID
    namespace_id: UUID
    batch_reg: _BatchRegistration | None  # None for orphaned docs


def _resolve_occurred_at(doc: Document, engine: Any, *, is_orphan: bool) -> datetime:
    """Resolve the chunk event time for a staged document (#1121).

    Identical resolution for the normal (batch) and crash-recovery (orphan)
    paths so a recovered document does not silently get a different event
    time than the same document on the non-crash path:

    1. ``metadata['occurred_at']`` parsed via the engine's ``_parse_datetime``
       (persisted on the re-loaded row at submit time, #932), else
    2. ``doc.source_timestamp``, else
    3. a tail fallback that is the only thing differing between paths:
       orphans use ``doc.created_at`` (the persisted ingest time of the
       recovered row), batch items use ``now()`` (no event time known).
    """
    doc_metadata = doc.metadata or {}
    occurred_at_raw = doc_metadata.get("occurred_at")
    parse_dt = getattr(engine, "_parse_datetime", None)
    if occurred_at_raw and parse_dt is not None:
        return parse_dt(occurred_at_raw)
    if doc.source_timestamp is not None:
        return doc.source_timestamp
    return doc.created_at if is_orphan else datetime.now(UTC)


# Imported below the LLMUsage definition (line ~118) to break the cycle:
# khora.core.models.recall imports LLMUsage from khora.khora. Re-exported
# here so external code can keep using ``from khora.khora import RecallResult``.
from khora.core.models.recall import (  # noqa: E402, I001, F401
    DocumentProjection,
    RecallChunk,
    RecallEntity,
    RecallRelationship,
    RecallResult,
)


# Shared-instance machinery for `Khora.shared()` (#619).
#
# Cached by config hash so two callers with identical config share one
# asyncpg pool. The lock serialises concurrent first-callers — without
# it, two awaits hitting `Khora.shared()` at the same time would race to
# instantiate and one of the connections would leak.
#
# #1160: both the lock and the instance cache are scoped per running event
# loop. A bare module-level `asyncio.Lock()` binds to the first loop it is
# acquired on; acquiring it from a second loop (sequential `asyncio.run`,
# per-invocation handlers, pytest-asyncio's per-test loops) raises
# `RuntimeError: ... bound to a different event loop` under contention.
# And a cached `Khora` carries an asyncpg pool tied to the loop it was
# connected on - handing it back on a different loop yields "attached to a
# different loop" failures.
#
# So: lazily create one lock per running loop, and store each cached
# instance alongside the loop it was built on. If the running loop differs
# from (or has closed) the stored loop, the instance is dropped and rebuilt
# on the live loop - mirroring the #790 fork drop-and-rebuild. The cache
# stays keyed by config hash (one entry per distinct config), so the public
# cache-size semantics are unchanged.
_SHARED_INSTANCES: dict[str, _SharedEntry] = {}
_SHARED_LOCKS: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}

# Back-compat alias: the #790 fork-safety tests import `_SHARED_LOCK`. There
# is no single process-wide lock any more (it is per-loop), but the name is
# kept pointing at a lock object so those imports and the fork handler's
# reseat keep working. It is not used for actual serialisation.
_SHARED_LOCK: asyncio.Lock = asyncio.Lock()


@dataclass(slots=True)
class _SharedEntry:
    """A cached shared Khora plus the event loop it was connected on (#1160)."""

    instance: Khora
    loop: asyncio.AbstractEventLoop


def _loop_lock() -> asyncio.Lock:
    """Return the lock for the running loop, creating it lazily (#1160).

    Keyed by the loop object itself (not ``id(loop)``) so a closed loop's
    address being reused by a later loop cannot alias two distinct loops.
    A dead loop's lock entry is pruned opportunistically.
    """
    loop = asyncio.get_running_loop()
    lock = _SHARED_LOCKS.get(loop)
    if lock is None:
        for dead in [lp for lp in _SHARED_LOCKS if lp.is_closed()]:
            _SHARED_LOCKS.pop(dead, None)
        lock = asyncio.Lock()
        _SHARED_LOCKS[loop] = lock
    return lock


def _reset_shared_after_fork() -> None:
    """Drop the parent's shared-Khora cache in a forked child (#790).

    The cached :class:`Khora` instances hold asyncpg pool sockets that
    the parent process also has open. If the child were to reuse them
    it would race the parent on connection state - asyncpg's protocol
    machinery is not fork-safe.

    Also reseat the lock machinery: ``asyncio.Lock`` in 3.10+ binds to a
    loop on first acquire, and the parent's loop is gone in the child.
    Fresh locks re-bind lazily on next acquire from the child.

    Important: do NOT try to ``disconnect()`` the cached instances from
    the at-fork handler. Closing the asyncpg connections in the child
    would also close the fds in the parent (they're the same fd
    numbers). Just discard references and let the parent's instance
    keep running.

    Registered via ``os.register_at_fork(after_in_child=...)``.
    """
    global _SHARED_LOCK
    _SHARED_INSTANCES.clear()
    _SHARED_LOCKS.clear()
    _SHARED_LOCK = asyncio.Lock()


if hasattr(os, "register_at_fork"):  # POSIX-only; Windows is a no-op.
    os.register_at_fork(after_in_child=_reset_shared_after_fork)


def _config_hash(config: KhoraConfig) -> str:
    """Stable identity hash for a KhoraConfig.

    Used to key the `Khora.shared()` cache. Identity != equality: we want
    "did the caller supply the same config object or the same effective
    settings" so callers that read the same env vars share one instance.

    Pydantic's `model_dump_json` is deterministic for the field set and
    handles SecretStr / UUID / Pydantic types out of the box. SHA-1 is
    fine here — this is not a security boundary.
    """
    try:
        payload = config.model_dump_json()
    except Exception:
        # If serialisation fails for any reason, fall back to id() so we
        # still get a cache key. This means two equal-but-distinct
        # configs won't share an instance, which is a correctness-safe
        # degradation (worse cache hit rate, not wrong answers).
        return f"id:{id(config)}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


class _SharedAccessor:
    """Bridge object exposed as `Khora.shared` so callers can write
    both `await Khora.shared()` and `await Khora.shared.clear()`.

    A bare `@classmethod` can't carry a `.clear` sub-method while also
    being awaitable, so we use a small descriptor-like instance.
    """

    __slots__ = ()

    async def __call__(self, config: KhoraConfig | None = None) -> Khora:
        """Return the process-wide cached :class:`Khora` instance.

        Lazily creates and connects on first call; subsequent calls with
        the same effective config return the same instance.

        Args:
            config: Optional :class:`KhoraConfig` override. If absent,
                :func:`khora.config.load_config` is called once per
                cache miss.

        Returns:
            A connected :class:`Khora` ready for `remember` / `recall`
            calls. Callers MUST NOT call `disconnect()` — the lifetime
            is process-wide. Use :meth:`clear` in tests.
        """
        return await Khora._shared_get(config)

    async def clear(self) -> None:
        """Disconnect and drop every cached shared instance.

        Test-only — production code never needs this. Call in test
        teardown / fixture finalisation to keep tests isolated.
        """
        await Khora._shared_clear()


class Khora:
    """Primary interface for Khora.

    Provides a simple API for storing and retrieving memories:
    - remember(): Store content in Khora
    - recall(): Retrieve relevant memories for a query
    - forget(): Remove memories
    - create_namespace(): Create a new memory namespace
    - get_namespace_by_stable_id(): Get a namespace by its stable ID

    Can be used as a context manager for automatic connection handling.

    The Khora is a facade that delegates to pluggable engines.
    The default engine is "vectorcypher" which uses knowledge graphs and vector embeddings.

    Usage:
        # Simplest - from env vars (KHORA_DATABASE_URL)
        async with Khora() as kb:
            await kb.remember("Important fact...", namespace=namespace_id,
                entity_types=["PERSON", "CONCEPT"], relationship_types=["RELATES_TO"])

        # Common - explicit database URL
        async with Khora("postgresql://localhost/mydb") as kb:
            results = await kb.recall("What do I know about...", namespace=namespace_id)

        # With graph backend
        async with Khora("postgresql://...", graph_url="bolt://localhost:7687") as kb:
            ...

        # Explicit engine selection (same as default)
        async with Khora("postgresql://...", engine="vectorcypher") as kb:
            ...

        # Full config
        async with Khora(KhoraConfig(...)) as kb:
            ...
    """

    def __init__(
        self,
        database_url: str | KhoraConfig | None = None,
        *,
        engine: str = "vectorcypher",
        graph_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        storage_config: StorageConfig | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        run_migrations: bool = False,
    ) -> None:
        """Initialize the Khora.

        Args:
            database_url: PostgreSQL URL, or full KhoraConfig, or None (reads KHORA_DATABASE_URL from env)
            engine: Engine to use (default: "vectorcypher")
            graph_url: Optional Neo4j/graph database URL (bolt://user:pass@host:port)
            embedding_model: Embedding model to use (default: text-embedding-3-small)
            storage_config: Storage configuration (derived from config if None) - deprecated
            engine_kwargs: Additional keyword arguments forwarded to the engine constructor
                (e.g., vectorcypher_config=VectorCypherConfig(...))
            run_migrations: If True, run Alembic migrations during connect() (default: False)

        Examples:
            # Simplest - from env vars
            kb = Khora()

            # Common - explicit database
            kb = Khora("postgresql://localhost/mydb")

            # With graph
            kb = Khora("postgresql://...", graph_url="bolt://...")

            # Explicit engine selection
            kb = Khora("postgresql://...", engine="vectorcypher")

            # Full config
            kb = Khora(KhoraConfig(...))
        """
        # Handle overloaded first argument
        if isinstance(database_url, KhoraConfig):
            self._config = database_url
        elif isinstance(database_url, str):
            # Build config from URL parameters
            self._config = KhoraConfig(
                database_url=database_url,
                neo4j_url=graph_url,
            )
            # Override embedding model if non-default
            if embedding_model != "text-embedding-3-small":
                self._config.llm.embedding_model = embedding_model
        else:
            # None - load from env/file
            self._config = load_config()
            # Apply overrides if provided
            if graph_url:
                from pydantic import SecretStr as _SecretStr

                self._config.neo4j_url = _SecretStr(graph_url)
            if embedding_model != "text-embedding-3-small":
                self._config.llm.embedding_model = embedding_model

        # Store for deferred engine creation
        self._engine_name = engine
        self._storage_config = storage_config  # for backwards compat
        self._engine_kwargs = engine_kwargs or {}
        self._run_migrations = run_migrations
        self._engine: MemoryEngineProtocol | None = None
        self._connected = False
        self._bg_tasks: set[asyncio.Task] = set()
        # Global chunk semaphore shared across all concurrent submit_batch calls.
        # Initialized on first submit_batch call that sets max_chunks_in_flight.
        self._chunk_semaphore: _GlobalChunkSemaphore | None = None
        # Unified pending processor: replaces both _submit_batch_worker
        # and _recover_pending_documents with a single mechanism.
        self._processor_queue: asyncio.Queue[_ProcessorItem] = asyncio.Queue()
        self._processor_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting Khora...")

        if self._run_migrations:
            # SurrealDB initialises its schema declaratively (DEFINE IF NOT
            # EXISTS on connect), so there is no Alembic chain to run. Treat
            # run_migrations=True as a no-op rather than reaching for a
            # Postgres DSN that doesn't exist — see #713.
            backend = getattr(self._config.storage, "backend", "postgres")
            if backend == "surrealdb":
                logger.info("Skipping Alembic migrations: backend=surrealdb uses declarative schema")
            else:
                from khora.db.session import run_migrations as _run_migrations

                # For the sqlite_lance embedded backend, derive a sqlite+aiosqlite URL
                # from the configured db_path so Alembic migrations target the same
                # file the adapters use. The migrations are dialect-aware.
                db_url: str | None
                if backend == "sqlite_lance" and self._config.storage.sqlite_lance is not None:
                    db_path = self._config.storage.sqlite_lance.db_path
                    db_url = f"sqlite+aiosqlite:///{db_path}"
                else:
                    # database_url is a SecretStr; unwrap for the Alembic runner
                    # (it forwards into SQLAlchemy create_async_engine).
                    db_url = (
                        self._config.database_url.get_secret_value() if self._config.database_url is not None else None
                    )
                result = await _run_migrations(db_url)
                if not result.success:
                    raise RuntimeError(f"Database migration failed: {result.error}")

        from khora.engines import create_engine

        self._engine = create_engine(
            self._engine_name,
            self._config,
            storage_config=self._storage_config,
            **self._engine_kwargs,
        )
        await self._engine.connect()

        # Wire hook dispatcher into the storage coordinator so the
        # ingestion pipeline can dispatch events without knowing about Khora.
        storage = getattr(self._engine, "_storage", None)
        if storage is not None:
            try:
                storage._hook_dispatcher = self._get_hook_dispatcher()
            except (AttributeError, TypeError):
                pass  # Mock or non-standard engine — hooks won't fire

        # Wire durable hook subscriptions (#599). When the coordinator
        # exposes a SQL session factory, give the dispatcher a store and
        # reload any persisted subscriptions so events delivered after a
        # restart still find their subscriber. Always attaches the store
        # post-connect (so a later subscribe_persistent() has somewhere to
        # write) — no-op on stacks without a SQL backend (e.g.
        # SurrealDB-unified) and never raises into connect.
        if storage is not None:
            try:
                await self._wire_persistent_hooks(storage)
            except Exception as exc:  # pragma: no cover - defensive
                # ADR-001: durable reload is disabled but connect() lives on -
                # record a Degradation on the dispatcher (when it exists) so
                # the fallback isn't silent.
                if hasattr(self, "_hook_dispatcher"):
                    self._hook_dispatcher._last_persist_degradation = {
                        "component": "hooks.subscription_store",
                        "reason": "wire_failed",
                        "detail": None,
                        "exception": repr(exc),
                    }
                logger.warning(
                    "Failed to wire persistent hook subscriptions; persistent hooks are disabled.",
                    exc_info=True,
                )

        # Drain any hook filters that were registered with a description
        # but no precomputed embedding (Issue #576 Phase 1, Item 2). After
        # this, operators who wrote SemanticFilter(description="...") per
        # the docs actually get Level 1 (cosine similarity) gating.
        if hasattr(self, "_hook_dispatcher"):
            embedder = getattr(self._engine, "_embedder", None)
            if embedder is not None:
                try:
                    await self._hook_dispatcher.embed_pending_filters(embedder)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to drain pending hook-filter embeddings: {}", exc)

        self._connected = True
        logger.info("Khora connected")

    def _ensure_processor_running(self) -> None:
        """Start the pending processor if it is not already running."""
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._run_pending_processor())
            self._bg_tasks.add(self._processor_task)
            self._processor_task.add_done_callback(self._bg_tasks.discard)

    def start_pending_processor(self) -> None:
        """Start the pending document processor (idempotent).

        Safe to call multiple times — a second call is a no-op if the processor
        is already running. Call this after connect() on services that write
        documents. Read-only services should not call this.

        Raises:
            RuntimeError: if connect() has not been called yet.
        """
        if not self._connected:
            raise RuntimeError("Khora not connected. Call connect() first.")
        self._ensure_processor_running()

    async def stop_pending_processor(self) -> None:
        """Stop the pending document processor.

        Cancels the background task and waits for it to exit. The processor can
        be restarted by calling start_pending_processor() again.
        """
        if self._processor_task is not None and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

    async def _run_pending_processor(self) -> None:
        """Unified pending processor.

        Replaces both ``_submit_batch_worker`` (inline processing) and
        ``_recover_pending_documents`` (startup recovery) with a single
        mechanism that drains PENDING documents from a shared queue.

        On startup: scans all namespaces for orphaned PENDING documents
        (older than the grace period) and enqueues them for processing.

        Then: runs a pool of workers that drain ``_processor_queue`` with
        bounded concurrency.  Items are added by ``submit_batch`` or by
        the orphan recovery scan.
        """
        # Phase 1: recover orphaned PENDING docs from previous crashes.
        try:
            await self._enqueue_orphaned_pending_docs()
        except Exception as exc:
            if _is_undefined_table_error(exc):
                # Fresh DB — `memory_namespaces` hasn't been created yet, so there
                # are no namespaces and therefore no orphaned PENDING docs. Common
                # path on per-run ephemeral databases.
                logger.debug(
                    "pending_processor: skipping orphan recovery on fresh DB (memory_namespaces table not yet created)"
                )
            else:
                logger.error(f"pending_processor: orphan recovery failed: {exc}")

        # Phase 2: drain the queue with bounded concurrency.
        max_concurrent = self._config.pipelines.pending_processor_max_concurrent

        async def _worker() -> None:
            while True:
                item = await self._processor_queue.get()
                try:
                    # #932: items carry only a document identity; re-load the
                    # full Document (with content) from storage here so the
                    # queue never holds content for the whole backlog. Peak
                    # content RAM is bounded by the number of workers, each
                    # holding one re-fetched Document. The load is its own try
                    # so a load failure path can never reach the processing
                    # error handler below - that would double-fire fire_result.
                    try:
                        doc = await self._load_pending_document(item)
                    except Exception as load_exc:
                        # Case 1: the load itself raised (transient DB error,
                        # e.g. connection exhaustion). The row stays PROCESSING
                        # for orphan recovery to retry. Fire exactly one
                        # failure result for batch items so BatchHandle.wait()
                        # can't hang, then move on - never re-raise into the
                        # worker loop.
                        safe_err = _safe_exc_summary(load_exc)
                        logger.error(f"pending_processor: failed to load doc {item.doc_id} at dequeue: {safe_err}")
                        batch_reg = item.batch_reg
                        if batch_reg is not None:
                            try:
                                batch_reg.fire_result(
                                    DocumentResult(
                                        document_id=item.doc_id,
                                        namespace_id=batch_reg.namespace_id,
                                        success=False,
                                        error=safe_err,
                                    )
                                )
                            except Exception as fire_exc:
                                logger.error(
                                    f"pending_processor: fire_result failed after load fault for doc "
                                    f"{item.doc_id}: {_safe_exc_summary(fire_exc)}"
                                )
                        continue
                    if doc is None:
                        # Case 2: the row was deleted/forgotten between enqueue
                        # and dequeue (ADR-001 skip). _load_pending_document
                        # already logged it and fired the skip result for batch
                        # items; nothing left to process.
                        continue

                    # Case 3: we have the document. Process it in its own try.
                    try:
                        await self._process_pending_item(item, doc)
                    except Exception as exc:
                        # Defense in depth for #869: ``_process_pending_item_impl``
                        # has pre-try work (engine resolution, ``getattr`` calls,
                        # pre-FAILED state cleanup, ``parse_dt``,
                        # ``start_usage_collection``) that runs before its own
                        # ``try``. If any of that raises, the inner block never
                        # fires ``batch_reg.fire_result`` and ``BatchHandle.wait``
                        # blocks forever - ``_remaining`` is only decremented by
                        # ``fire_result``. Fall back here: deliver one failure
                        # result so the batch's ``_done_event`` can fire, and
                        # flip the doc to FAILED so it doesn't linger in
                        # PROCESSING. ``doc`` is guaranteed non-None here (we
                        # only reach this try after a successful load), so we
                        # never re-load - that was the #932 double-fire source.
                        # Both side effects are best-effort and guarded - we
                        # must never re-raise into the worker loop.
                        safe_err = _safe_exc_summary(exc)
                        logger.error(f"pending_processor: unhandled error processing doc {item.doc_id}: {safe_err}")
                        batch_reg = item.batch_reg
                        if batch_reg is not None:
                            try:
                                doc.mark_failed(str(exc))
                                await self.storage.update_document(doc)
                            except Exception as upd_exc:
                                logger.warning(
                                    f"pending_processor: could not update document status after pre-try fault: {_safe_exc_summary(upd_exc)}"
                                )
                            try:
                                batch_reg.fire_result(
                                    DocumentResult(
                                        document_id=item.doc_id,
                                        namespace_id=batch_reg.namespace_id,
                                        success=False,
                                        error=safe_err,
                                        external_id=doc.external_id,
                                    )
                                )
                            except Exception as fire_exc:
                                logger.error(
                                    f"pending_processor: fire_result failed in fallback for doc {item.doc_id}: {_safe_exc_summary(fire_exc)}"
                                )
                finally:
                    self._processor_queue.task_done()

        workers = [asyncio.create_task(_worker()) for _ in range(max_concurrent)]
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            for w in workers:
                w.cancel()
            raise

    async def _enqueue_orphaned_pending_docs(self) -> None:
        """Scan for stale orphaned documents and enqueue them for processing.

        Covers both PENDING documents (never picked up) and PROCESSING documents
        left behind by a crashed worker (#885). Claims are atomic via
        ``claim_orphaned_documents`` - on PostgreSQL it uses ``FOR UPDATE SKIP
        LOCKED`` so two concurrent instances never claim the same doc (#886).
        Each claimed doc is already flipped to PROCESSING by the claim, so it is
        enqueued directly. Extraction parameters are read from the document's
        ``extraction_params`` column; if absent, falls back to config defaults.
        """
        grace_minutes = self._config.pipelines.pending_processor_grace_period_minutes
        # Backwards compat: honour deprecated field if the new one wasn't explicitly set.
        if self._config.pipelines.pending_recovery_grace_period_minutes is not None:
            grace_minutes = self._config.pipelines.pending_recovery_grace_period_minutes
        now = datetime.now(UTC)
        pending_before = now - timedelta(minutes=grace_minutes)
        processing_stale_seconds = self._config.pipelines.pending_processor_orphan_stale_after_seconds
        processing_before = now - timedelta(seconds=processing_stale_seconds)

        storage = getattr(self._engine, "_storage", None)
        if storage is None:
            return

        engine = self._get_engine()
        process_fn = getattr(engine, "process_staged_document", None)
        if process_fn is None:
            logger.debug("pending_processor: engine does not support process_staged_document, skipping orphan recovery")
            return

        total_enqueued = 0

        offset = 0
        while True:
            ns_page = await storage.list_namespaces(active_only=True, limit=100, offset=offset)
            namespaces = ns_page.items
            if not namespaces:
                break

            for ns in namespaces:
                while True:
                    docs = await storage.claim_orphaned_documents(
                        ns.id,
                        pending_before=pending_before,
                        processing_before=processing_before,
                        limit=100,
                    )
                    if not docs:
                        break

                    for doc in docs:
                        prior_status = doc.orphan_prior_status or "pending"
                        _ORPHANS_RECLAIMED_COUNTER.add(1, attributes={"prior_status": prior_status})
                        self._processor_queue.put_nowait(
                            _ProcessorItem(doc_id=doc.id, namespace_id=doc.namespace_id, batch_reg=None)
                        )
                        total_enqueued += 1

                    if len(docs) < 100:
                        break

            offset += len(namespaces)
            if len(namespaces) < 100:
                break

        if total_enqueued:
            logger.info(
                f"pending_processor: reclaimed {total_enqueued} orphaned documents "
                f"for recovery (pending_grace={grace_minutes}m, processing_stale={processing_stale_seconds}s)"
            )
        else:
            logger.debug(
                f"pending_processor: no orphaned documents found "
                f"(pending_grace={grace_minutes}m, processing_stale={processing_stale_seconds}s)"
            )

    async def _load_pending_document(self, item: _ProcessorItem) -> Document | None:
        """Re-load the full Document for a queued item (#932).

        The queue carries only a document identity, so the worker re-loads
        the persisted PENDING/PROCESSING row (with content) here. Scoped to
        ``item.namespace_id`` so the load can't reach across tenants.

        Returns ``None`` when the row is gone (deleted / forgotten between
        enqueue and dequeue). Per ADR-001 this is a deliberate skip, not a
        crash: it is logged at INFO, counted, and - for batch items - the
        batch registration is decremented (via a skipped failure result) so
        a waiting ``BatchHandle.wait()`` can't hang on the missing doc.
        """
        doc = await self.storage.get_document(item.doc_id, namespace_id=item.namespace_id)
        if doc is not None:
            return doc

        # SkipReason (docs/architecture/failure-observability-contract.md).
        skip: SkipReason = {
            "op_kind": "pending_processor.load_document",
            "reason": "document_missing_at_dequeue",
            "detail": f"doc_id={item.doc_id}",
        }
        _PROCESSOR_DOC_MISSING_COUNTER.add(1, attributes={"reason": skip["reason"]})
        logger.info(
            "pending_processor: skipping queued doc {} - row gone at dequeue (deleted/forgotten); {}",
            item.doc_id,
            skip,
        )
        batch_reg = item.batch_reg
        if batch_reg is not None:
            # Decrement the batch so BatchHandle.wait() can't hang. Reported
            # as a skipped (non-error) result - the doc no longer exists.
            batch_reg.fire_result(
                DocumentResult(
                    document_id=item.doc_id,
                    namespace_id=batch_reg.namespace_id,
                    success=True,
                    skipped=True,
                )
            )
        return None

    async def _process_pending_item(self, item: _ProcessorItem, doc: Document) -> None:
        """Process a single PENDING document, gated by the per-batch semaphore.

        Acquires the batch's `concurrency_sem` (if set) around
        `_process_pending_item_impl` so in-flight docs *for this batch* never
        exceed the batch's `max_concurrent`. Concurrent batches each have
        their own semaphore - they don't share state.

        Orphan-recovery items (`batch_reg is None`) carry no per-batch
        semaphore and run unguarded.
        """
        sem = item.batch_reg.concurrency_sem if item.batch_reg is not None else None
        if sem is not None:
            async with sem:
                await self._process_pending_item_impl(item, doc)
        else:
            await self._process_pending_item_impl(item, doc)

    async def _process_pending_item_impl(self, item: _ProcessorItem, doc: Document) -> None:
        """Process a single PENDING document through the engine pipeline.

        Handles both enqueued items (from submit_batch with batch_reg) and
        orphaned items (from crash recovery with batch_reg=None). ``doc`` is
        the full Document re-loaded by the worker (#932).
        """
        from khora.telemetry.context import collect_usage, start_usage_collection

        batch_reg = item.batch_reg
        namespace_id = batch_reg.namespace_id if batch_reg else doc.namespace_id

        storage = self.storage
        engine = self._get_engine()
        process_fn = getattr(engine, "process_staged_document", None)

        if process_fn is None:
            err_msg = f"Engine {type(engine).__name__!r} does not support process_staged_document"
            if batch_reg:
                doc.mark_failed(err_msg)
                try:
                    await storage.update_document(doc)
                except Exception as upd_exc:
                    logger.warning(f"pending_processor: could not update document status: {upd_exc}")
                batch_reg.fire_result(
                    DocumentResult(
                        document_id=doc.id,
                        namespace_id=namespace_id,
                        success=False,
                        error=err_msg,
                        external_id=doc.external_id,
                    )
                )
            else:
                logger.warning(f"pending_processor: {err_msg}, skipping orphan doc {doc.id}")
            return

        # H1: Clear partial extraction state for previously-FAILED/ARCHIVED documents
        # before re-processing to prevent duplicate chunks/entities on retry.
        pre_failed_doc_ids = batch_reg.pre_failed_doc_ids if batch_reg else set()
        if doc.id in pre_failed_doc_ids:
            if storage.vector is not None:
                try:
                    await storage.vector.delete_chunks_by_document(doc.id, namespace_id=namespace_id)
                except Exception as exc:
                    logger.warning(f"pending_processor: could not clear chunks table for {doc.id}: {exc}")
            clear_fn = getattr(engine, "clear_document_extraction_state", None)
            if clear_fn is not None:
                try:
                    await clear_fn(doc.id, namespace_id)
                except Exception as exc:
                    logger.warning(f"pending_processor: could not clear extraction state for {doc.id}: {exc}")

        # Resolve extraction parameters.
        # extraction_params (and metadata, used below for occurred_at) come
        # from the re-loaded document row for both enqueued and orphaned items
        # (#932) - the item itself no longer carries a doc_data payload.
        params = doc.extraction_params or {}
        skill_name = params.get("skill_name", "general_entities")
        entity_types = params.get("entity_types", list(self._config.pipelines.entity_types))
        relationship_types = params.get("relationship_types", [])
        extraction_config_hash = doc.extraction_config_hash
        chunk_strategy = params.get("chunk_strategy")
        max_chunks_in_flight = params.get("max_chunks_in_flight")

        # Reconstruct expertise from stored dict, if present.
        expertise = None
        expertise_data = params.get("expertise")
        if expertise_data is not None:
            try:
                from khora.extraction.skills import ExpertiseConfig

                expertise = ExpertiseConfig.from_dict(expertise_data)
            except Exception:
                logger.warning(f"pending_processor: could not reconstruct ExpertiseConfig for doc {doc.id}")

        # Resolve occurred_at.
        # #932: metadata is read off the re-loaded document row (it is
        # persisted at submit time), not from a queued doc_data dict.
        # #1121: both paths resolve identically via _resolve_occurred_at -
        # the orphan-recovery path (batch_reg is None) must also consult
        # metadata['occurred_at'] before falling back, or recovered docs
        # get chunks stamped with ingest time instead of the event time.
        occurred_at = _resolve_occurred_at(doc, engine, is_orphan=batch_reg is None)

        start_usage_collection()
        try:
            chunks, entities, rels = await process_fn(
                doc,
                skill_name=skill_name,
                occurred_at=occurred_at,
                entity_types=entity_types,
                relationship_types=relationship_types,
                expertise=expertise,
                extraction_config_hash=extraction_config_hash,
                chunk_strategy=chunk_strategy,
                max_chunks_in_flight=max_chunks_in_flight,
                chunk_semaphore=self._chunk_semaphore if max_chunks_in_flight is not None else None,
            )
            if batch_reg:
                batch_reg.fire_result(
                    DocumentResult(
                        document_id=doc.id,
                        namespace_id=namespace_id,
                        success=True,
                        chunks_created=chunks,
                        entities_extracted=entities,
                        relationships_created=rels,
                        llm_usage=collect_usage(),
                        external_id=doc.external_id,
                    )
                )
            else:
                collect_usage()  # discard for orphan recovery
                logger.info(f"pending_processor: recovered orphan doc {doc.id}")
        except Exception as exc:
            partial_usage = collect_usage()
            # _safe_exc_summary, not {exc}: a payload-bearing DBAPIError here
            # would otherwise leak the document content + metadata bind params.
            # (doc.mark_failed below keeps str(exc) — that persists to the
            # error_message column, a separate surface tracked elsewhere.)
            logger.error(f"pending_processor: failed to process document {doc.id}: {_safe_exc_summary(exc)}")
            doc.mark_failed(str(exc))
            try:
                await storage.update_document(doc)
            except Exception as upd_exc:
                logger.warning(f"pending_processor: could not update document status: {_safe_exc_summary(upd_exc)}")
            if batch_reg:
                batch_reg.fire_result(
                    DocumentResult(
                        document_id=doc.id,
                        namespace_id=namespace_id,
                        success=False,
                        error=_safe_exc_summary(exc),
                        llm_usage=partial_usage,
                        external_id=doc.external_id,
                    )
                )

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting Khora...")

        # Cancel the pending processor if running.
        if self._processor_task is not None and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

        if self._engine:
            await self._engine.disconnect()
            self._engine = None

        self._connected = False
        logger.info("Khora disconnected")

    async def __aenter__(self) -> Khora:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit.

        If the body of the ``async with`` raised, suppress any secondary
        exception from disconnect() so the original traceback is what reaches
        the user. Disconnect-time failures are logged at warning level.
        Without this guard, a teardown error (e.g. a SurrealDB shared
        connection in a bad state after the body's exception) replaces the
        real cause — exactly the kind of masking #715 describes for the Rust
        side.
        """
        try:
            await self.disconnect()
        except Exception as disc_exc:  # noqa: BLE001
            if exc_type is None:
                raise
            logger.warning(
                "Khora.disconnect raised during __aexit__; suppressing to preserve original {} traceback: {}",
                exc_type.__name__,
                disc_exc,
            )

    # =========================================================================
    # Process-wide singleton — Khora.shared() (#619)
    #
    # Adapters (CrewAI / LangGraph / OpenAI Agents SDK) run in ephemeral
    # contexts (`Runner.run_sync`, one-shot crew kicks) where allocating a
    # fresh `Khora()` per call would churn the asyncpg pool. `Khora.shared()`
    # returns a process-wide instance, lazily connected, cached by config
    # hash so two callers with the same config share one pool.
    #
    # `Khora.shared.clear()` resets the cache — test-only escape hatch.
    # =========================================================================
    shared: _SharedAccessor  # set below the class definition

    @classmethod
    async def _shared_get(cls, config: KhoraConfig | None = None) -> Khora:
        """Internal: implement the shared() singleton fetch.

        #1160: a `Khora` connected on a now-dead (or simply different) loop
        is unusable - its asyncpg pool is bound to that loop. If the cached
        entry's loop is not the running loop, drop it and rebuild on the
        live loop rather than returning the stale instance.
        """
        loop = asyncio.get_running_loop()
        async with _loop_lock():
            cfg = config if config is not None else load_config()
            key = _config_hash(cfg)
            entry = _SHARED_INSTANCES.get(key)
            if entry is not None and (entry.loop is not loop or entry.loop.is_closed()):
                # Stale loop. Don't disconnect - the owning loop is gone, so
                # awaiting its pool teardown is impossible; drop the
                # reference and rebuild (#790 fork drop-and-rebuild shape).
                _SHARED_INSTANCES.pop(key, None)
                entry = None
            if entry is not None and entry.instance._connected:
                return entry.instance
            if entry is None:
                entry = _SharedEntry(instance=cls(cfg), loop=loop)
                _SHARED_INSTANCES[key] = entry
            if not entry.instance._connected:
                await entry.instance.connect()
            return entry.instance

    @classmethod
    async def _shared_clear(cls) -> None:
        """Internal: disconnect and drop every cached shared instance."""
        async with _loop_lock():
            entries = list(_SHARED_INSTANCES.values())
            _SHARED_INSTANCES.clear()
        for entry in entries:
            try:
                await entry.instance.disconnect()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Khora.shared.clear: disconnect raised: {}", exc)

    def _get_engine(self) -> MemoryEngineProtocol:
        """Get the engine (internal use)."""
        if self._engine is None:
            raise RuntimeError("Khora not connected. Call connect() first.")
        return self._engine

    @property
    def storage(self) -> StorageCoordinator:
        """Get the storage coordinator for admin/management operations.

        Provides direct access to the underlying storage coordinator for
        managing namespaces and other administrative tasks not covered
        by the high-level API.

        For common operations, prefer the Khora convenience methods:
        - kb.get_document() for document retrieval
        - kb.list_documents() for document listing
        - kb.search_entities() for entity search
        - kb.stats() for namespace statistics
        """
        engine = self._get_engine()
        if hasattr(engine, "_storage") and engine._storage:
            return engine._storage  # type: ignore[invalid-return-type]
        raise AttributeError("Current engine does not expose storage")

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def create_namespace(
        self,
        *,
        config_overrides: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace.

        Args:
            config_overrides: Optional configuration overrides
            metadata: Optional namespace metadata

        Returns:
            Created MemoryNamespace
        """
        return await self._get_engine().create_namespace(
            config_overrides=config_overrides,
            metadata=metadata,
        )

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        return await self._get_engine().get_namespace(namespace_id)

    async def get_namespace_by_stable_id(self, namespace_id: str | UUID) -> MemoryNamespace | None:
        """Get a namespace by its stable namespace_id.

        Unlike get_namespace() which takes a row-level id, this accepts
        the stable namespace_id (shared across versions) and resolves it
        to the active version before fetching.

        Args:
            namespace_id: The stable namespace identifier (UUID or string)

        Returns:
            MemoryNamespace, or None if the resolved namespace row is not found

        Raises:
            ValueError: If no active namespace version exists for the given namespace_id
        """
        resolved_id = await self._resolve_namespace(namespace_id)
        return await self._get_engine().get_namespace(resolved_id)

    # =========================================================================
    # Core API: remember, recall, forget
    # =========================================================================

    async def remember(
        self,
        content: str,
        *,
        namespace: str | UUID,
        title: str = "",
        source: str = "",
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        external_id: str | None = None,
        session_id: UUID | None = None,
    ) -> RememberResult:
        """Store content in Khora.

        This is the primary method for adding memories. It:
        1. Creates a document
        2. Chunks the content
        3. Generates embeddings
        4. Extracts entities and relationships

        Args:
            content: Content to remember
            namespace: Namespace UUID (as UUID or string)
            title: Optional title for the content
            source: Optional source identifier
            source_type: Provenance category (default "library").
            source_name: Optional provider-level identifier (e.g. "slack", "linear").
            source_url: Optional original-source URL.
            source_timestamp: Optional original-source timestamp (when the content
                was authored / occurred at the source). When provided, wins over
                any timestamp derivable from ``metadata`` via the metadata-based
                fallback (``sent_at`` / ``occurred_at`` / ``created_at`` / ...).
                When omitted, the existing metadata-derived fallback is preserved.
            metadata: Optional metadata
            skill_name: Extraction skill to use
            entity_types: Required entity types to extract
            relationship_types: Required relationship types to extract
            expertise: Optional expertise config for domain-specific extraction
            extraction_config_hash: Optional hash of the extraction config for change detection
            chunk_strategy: Override chunking strategy for this call only.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.
            external_id: Optional caller-supplied external identifier for the document.
                Must be None or a non-blank string (max 512 chars).
                Raises ValueError if constraints are violated.

        Returns:
            RememberResult with details. ``metadata["extraction_errors"]``
            (int) and ``metadata["degradations"]`` (ADR-001 list) are
            present only when one or more chunks failed LLM extraction
            (#889); on the happy path ``metadata`` stays empty.

        Raises:
            UnsupportedEngineKwargError: When the configured engine
                cannot honor a kwarg. The Skeleton engine raises this
                for non-empty ``entity_types`` / ``relationship_types``
                (it does not perform typed entity extraction - #890);
                see each engine's docstring for the full contract.
        """
        import time as _time

        from khora.pipelines.flows.ingest import coerce_source_timestamp
        from khora.telemetry.aggregate_metrics import record_ingest_duration
        from khora.telemetry.context import (
            clear_trace_id,
            collect_usage,
            ensure_trace_id,
            start_usage_collection,
        )

        ensure_trace_id()
        start_usage_collection()
        _t0 = _time.perf_counter()
        _status = "success"
        try:
            namespace_id = await self._resolve_namespace(namespace)
            # Normalize a possibly-string source_timestamp before it reaches
            # the engine's Document(...) — upstream callers hand us ISO strings
            # despite the datetime-typed kwarg.
            source_timestamp = coerce_source_timestamp(source_timestamp)
            # Stamp session_id into the metadata dict so engines that bypass
            # the ingest pipeline (vectorcypher builds Document directly)
            # can recover it from ``document.metadata["session_id"]``
            # without needing a dedicated kwarg on every engine's remember().
            if session_id is not None:
                metadata = {**(metadata or {}), "session_id": str(session_id)}
            with trace_span("khora.remember", namespace_id=str(namespace_id), content_length=len(content)):
                # NOTE: expertise and extraction_config_hash are always forwarded,
                # even when None. Custom engines registered via register_engine()
                # must accept these kwargs to remain compatible.
                result = await self._get_engine().remember(
                    content,
                    namespace_id,
                    title=title,
                    source=source,
                    source_type=source_type,
                    source_name=source_name,
                    source_url=source_url,
                    source_timestamp=source_timestamp,
                    metadata=metadata,
                    skill_name=skill_name,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                    expertise=expertise,
                    extraction_config_hash=extraction_config_hash,
                    chunk_strategy=chunk_strategy,
                    external_id=external_id,
                )
                return replace(result, llm_usage=collect_usage())
        except Exception:
            _status = "error"
            raise
        finally:
            collect_usage()  # idempotent — drains queue if not already collected
            clear_trace_id()
            record_ingest_duration(
                _time.perf_counter() - _t0,
                stage="end_to_end",
                status=_status,
            )

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str | UUID,
        skill_name: str = "general_entities",
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        max_concurrent: int = 10,
        deduplicate: bool = True,
        infer_relationships: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        extraction_batch_size: int | None = None,
        extraction_max_tokens: int | None = None,
    ) -> BatchResult:
        """Store multiple documents with automatic optimization.

        Handles internally:
        - Shared embedder with LRU cache (reused across batches)
        - Entity deduplication via EntityIndex
        - Multi-phase resolution (smart mode)
        - Relationship inference

        This is more efficient than calling remember() for each document
        as it processes documents in parallel with controlled concurrency
        and shares resources across documents.

        Args:
            documents: List of document dicts with keys:
                - content: str (required)
                - title: str (optional)
                - source: str (optional)
                - source_type: str (optional) — overrides top-level kwarg per doc
                - source_name: str (optional) — overrides top-level kwarg per doc
                - source_url: str (optional) — overrides top-level kwarg per doc
                - source_timestamp: datetime (optional) — overrides top-level kwarg per doc
                - metadata: dict (optional)
                - external_id: str (optional) — caller-supplied external identifier
            namespace: Namespace UUID (as UUID or string)
            skill_name: Extraction skill to use
            source_type: Default provenance category for docs that don't supply one.
            source_name: Default provider identifier for docs that don't supply one.
            source_url: Default original-source URL for docs that don't supply one.
            source_timestamp: Default original-source timestamp for docs that don't
                supply one. When provided (top-level kwarg or per-doc), wins over
                the metadata-derived fallback in the ingest pipeline; when omitted,
                the existing fallback is preserved.
            max_concurrent: Maximum concurrent document processing
            deduplicate: Deduplicate entities across documents (default: True)
            infer_relationships: Infer relationships after ingestion (default: True)
            on_progress: Callback(completed_count, total_count) for progress updates.
                Fired once per document with an incrementing ``completed_count``.
                Note the batched-progress caveat: VectorCypher and Chronicle run
                most work (embedding, entity extraction) as batched stages across
                all documents at once, so the per-document callbacks do not arrive
                strictly one-at-a-time during processing - they arrive in bursts at
                stage / window completion. Streaming per-document progress before a
                stage completes is not possible without restructuring the batched
                pipeline. For more granular progress, set ``max_chunks_in_flight``
                smaller so windows close more often.
            entity_types: Required entity types to extract
            relationship_types: Required relationship types to extract
            expertise: Optional expertise config for domain-specific extraction
            extraction_config_hash: Optional hash of the extraction config for change detection
            chunk_strategy: Override chunking strategy for this call only.
                Valid values: "fixed", "semantic", "recursive", "conversation".
                When None (default), uses the configured pipeline default.

        Returns:
            BatchResult with aggregated statistics
        """
        import time as _time

        from khora.pipelines.flows.ingest import coerce_source_timestamp
        from khora.telemetry.aggregate_metrics import record_ingest_duration
        from khora.telemetry.context import (
            clear_trace_id,
            collect_usage,
            ensure_trace_id,
            start_usage_collection,
        )

        ensure_trace_id()
        start_usage_collection()
        _t0 = _time.perf_counter()
        _status = "success"
        try:
            namespace_id = await self._resolve_namespace(namespace)
            # Normalize a possibly-string default before it propagates to the
            # per-doc dicts and the engine — see remember() for the rationale.
            source_timestamp = coerce_source_timestamp(source_timestamp)
            with trace_span("khora.remember_batch", namespace_id=str(namespace_id), batch_size=len(documents)):
                # Pre-stamp the top-level provenance kwargs onto each doc dict
                # so engines (and the ingest pipeline) read a single source of
                # truth. Per-doc dict values always win — only fill the kwarg
                # default when the doc didn't supply its own.
                for doc_data in documents:
                    if "source_type" not in doc_data:
                        doc_data["source_type"] = source_type
                    if "source_name" not in doc_data:
                        doc_data["source_name"] = source_name
                    if "source_url" not in doc_data:
                        doc_data["source_url"] = source_url
                    if "source_timestamp" not in doc_data:
                        doc_data["source_timestamp"] = source_timestamp
                    else:
                        doc_data["source_timestamp"] = coerce_source_timestamp(doc_data["source_timestamp"])
                # NOTE: see remember() comment re: custom engine compatibility
                batch_kwargs: dict[str, Any] = dict(
                    skill_name=skill_name,
                    max_concurrent=max_concurrent,
                    deduplicate=deduplicate,
                    infer_relationships=infer_relationships,
                    on_progress=on_progress,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                    expertise=expertise,
                    extraction_config_hash=extraction_config_hash,
                    chunk_strategy=chunk_strategy,
                    source_type=source_type,
                    source_name=source_name,
                    source_url=source_url,
                    source_timestamp=source_timestamp,
                )
                if extraction_batch_size is not None:
                    batch_kwargs["extraction_batch_size"] = extraction_batch_size
                if extraction_max_tokens is not None:
                    batch_kwargs["extraction_max_tokens"] = extraction_max_tokens
                result = await self._get_engine().remember_batch(
                    documents,
                    namespace_id,
                    **batch_kwargs,
                )
                return replace(result, llm_usage=collect_usage())
        except Exception:
            _status = "error"
            raise
        finally:
            collect_usage()  # idempotent — drains queue if not already collected
            clear_trace_id()
            record_ingest_duration(
                _time.perf_counter() - _t0,
                stage="end_to_end",
                status=_status,
            )

    async def submit_batch(
        self,
        documents: list[dict[str, Any]],
        *,
        on_result: Callable[[int, int, DocumentResult], None],
        namespace: str | UUID,
        skill_name: str = "general_entities",
        source_type: str = "library",
        source_name: str | None = None,
        source_url: str | None = None,
        source_timestamp: datetime | None = None,
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        max_chunks_in_flight: int | None = None,
        max_concurrent: int = 20,
        reprocess_archived: bool = False,
        session_id: UUID | None = None,
    ) -> BatchHandle:
        """Submit documents for deferred background processing.

        Unlike remember_batch() (which blocks until all documents are processed),
        submit_batch() persists documents as PENDING and returns a BatchHandle
        immediately. Processing continues in the background.

        Contract:
        - Before return: all documents are persisted to the DB with PENDING status
          (durable — survives crashes).
        - After return: documents are processed in bounded windows of
          max_chunks_in_flight chunks. on_result fires per document as each
          completes.
        - Multiple concurrent submit_batch() calls are safe; each has an
          independent BatchHandle and background task.

        Args:
            documents: List of document dicts with 'content', 'title', 'source',
                'source_type', 'source_name', 'source_url', 'source_timestamp',
                'metadata', 'external_id' keys. Per-doc source_type / source_name /
                source_url / source_timestamp override the top-level kwargs for
                that document.
            on_result: Synchronous callback(completed, total, DocumentResult)
                invoked per document as processing completes.
            namespace: Namespace UUID (as UUID or string).
            skill_name: Extraction skill to use.
            source_type: Default provenance category (e.g. "library", "api").
            source_name: Default provider identifier (e.g. "slack", "linear").
            source_url: Default original-source URL.
            source_timestamp: Default original-source timestamp for docs that
                don't supply one. When provided (top-level kwarg or per-doc),
                wins over the metadata-derived fallback when the document is
                processed; when omitted, the existing fallback is preserved.
            entity_types: Required entity types to extract.
            relationship_types: Required relationship types to extract.
            expertise: Optional domain-specific extraction config.
            extraction_config_hash: Optional hash for extraction config change detection.
            chunk_strategy: Override chunking strategy for this batch.
            max_chunks_in_flight: Maximum chunks processed per window. Bounds
                in-flight chunk processing *after* a document is dequeued; it
                does NOT bound the staging-queue depth. Staging memory is
                bounded separately by re-loading document content at dequeue
                time (#932), so the queue itself holds only lightweight
                identities. None = unbounded chunk window.
            max_concurrent: Maximum number of documents from THIS batch being
                processed concurrently (default: 20). Bounded above by the
                global processor pool size, which is set via
                ``KhoraConfig.pipelines.pending_processor_max_concurrent``.
                Two batches submitted concurrently are independently
                rate-limited - their ``max_concurrent`` values do not stack.
            reprocess_archived: If True, ARCHIVED documents are reset to PENDING
                and re-processed like FAILED documents. If False (default), ARCHIVED
                documents are skipped with a warning - preserving intentional
                archival semantics.

        Returns:
            BatchHandle with batch_id, completion status, and wait() method.

        Raises:
            RuntimeError: If the engine does not support staged document processing.
        """
        from khora.core.models.document import Document

        if max_concurrent < 1:
            raise ValueError(f"submit_batch: max_concurrent must be >= 1, got {max_concurrent}")

        if not documents:
            handle = BatchHandle(batch_id=uuid4(), total=0)
            handle._mark_done()
            return handle

        from khora.core.models.document import DocumentStatus
        from khora.pipelines.flows.ingest import coerce_source_timestamp

        namespace_id = await self._resolve_namespace(namespace)
        storage = self.storage

        # Normalize a possibly-string default once; the per-doc override is
        # coerced at each Document(...) site below.
        source_timestamp = coerce_source_timestamp(source_timestamp)

        # Stamp the batch-level session_id into each doc's metadata so the
        # ingest pipeline's ``stage_document`` path picks it up as a
        # first-class ``Document.session_id`` (#620). A per-doc
        # ``metadata.session_id`` always wins so callers can mix sessions
        # within a single submit_batch call.
        if session_id is not None:
            session_id_str = str(session_id)
            for doc_data in documents:
                meta = doc_data.get("metadata") or {}
                if "session_id" not in meta:
                    meta = {**meta, "session_id": session_id_str}
                    doc_data["metadata"] = meta

        # Persist all documents as PENDING before returning the handle.
        # This satisfies the durability contract — if the process crashes after
        # submit_batch() returns, the PENDING records survive for recovery.
        #
        # Self-healing for existing documents:
        # Instead of failing on duplicate external_id, detect and dispatch:
        #   PENDING    → skip insert, re-queue for processing (self-heal stalled docs)
        #   COMPLETED  → skip entirely, report success (already done)
        #   FAILED     → reset to PENDING + update content, re-queue for processing
        #   ARCHIVED   → skip by default (preserves intentional archival); set
        #                reprocess_archived=True to re-activate explicitly
        #   PROCESSING → skip to avoid race with active worker (M1)
        pending_docs: list[Document] = []
        pre_failed_docs: list[tuple[Document, str]] = []
        pre_completed_docs: list[Document] = []

        # Batch-lookup existing documents by external_id to avoid N serial queries.
        all_external_ids = [d.get("external_id") for d in documents if d.get("external_id")]
        existing_by_ext_id: dict[str, Document] = {}
        if all_external_ids:
            try:
                existing_by_ext_id = await storage.get_documents_by_external_ids(
                    all_external_ids, namespace_id=namespace_id
                )
            except Exception as exc:
                # M2: Fall through to the normal insert path if the lookup fails.
                logger.warning(
                    f"submit_batch: could not look up existing documents by external_id "
                    f"({exc}); treating all as new inserts"
                )
                existing_by_ext_id = {}

        # Build extraction parameters payload once, to be stored on each PENDING document.
        expertise_dict = None
        if expertise is not None:
            try:
                expertise_dict = expertise.to_dict()
            except Exception as exc:
                logger.debug(f"submit_batch: could not serialize expertise config: {exc}")
        extraction_params_payload: dict[str, Any] = {
            "skill_name": skill_name,
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "expertise": expertise_dict,
            "extraction_config_hash": extraction_config_hash,
            "chunk_strategy": chunk_strategy,
            "max_chunks_in_flight": max_chunks_in_flight,
        }

        seen_external_ids: set[str] = set()
        pre_failed_doc_ids: set[UUID] = set()

        for doc_data in documents:
            content = doc_data.get("content", "")
            checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
            external_id = doc_data.get("external_id")

            # M4: Skip duplicate external_ids within the same batch.
            if external_id:
                if external_id in seen_external_ids:
                    logger.warning(
                        f"submit_batch: duplicate external_id in batch, skipping subsequent occurrence "
                        f"(external_id={external_id!r})"
                    )
                    continue
                seen_external_ids.add(external_id)

            existing = existing_by_ext_id.get(external_id) if external_id else None

            if existing is not None:
                if existing.status == DocumentStatus.COMPLETED:
                    # Already fully processed — skip re-insertion, report as skipped.
                    logger.debug(
                        f"submit_batch: document already COMPLETED, skipping "
                        f"(external_id={external_id!r}, doc_id={existing.id})"
                    )
                    pre_completed_docs.append(existing)
                    continue

                # M1: PROCESSING means an active worker holds this doc — skip to avoid race.
                if existing.status == DocumentStatus.PROCESSING:
                    logger.warning(
                        f"submit_batch: document is PROCESSING, skipping re-queue to avoid race "
                        f"(external_id={external_id!r}, doc_id={existing.id})"
                    )
                    pre_completed_docs.append(existing)
                    continue

                # ARCHIVED: skip by default to preserve intentional archival semantics.
                # Callers must explicitly pass reprocess_archived=True to re-activate.
                if existing.status == DocumentStatus.ARCHIVED and not reprocess_archived:
                    logger.warning(
                        f"submit_batch: ARCHIVED document skipped — pass reprocess_archived=True "
                        f"to re-activate (external_id={external_id!r}, doc_id={existing.id})"
                    )
                    pre_completed_docs.append(existing)
                    continue

                # PENDING, FAILED, or ARCHIVED (reprocess_archived=True): reset to PENDING and re-process.
                # Update content + metadata so the re-run uses the latest submitted values
                # (fixes empty-source issue observed in soak test).
                prior_status = existing.status
                # H1: Track FAILED and ARCHIVED docs — they may have prior extraction
                # state (chunks, graph entities) that must be cleared before re-processing
                # to prevent duplicate chunks/entities on retry.
                if prior_status in (DocumentStatus.FAILED, DocumentStatus.ARCHIVED):
                    pre_failed_doc_ids.add(existing.id)
                existing.content = content
                existing.title = doc_data.get("title") or None
                existing.source = doc_data.get("source") or None
                existing.source_type = doc_data.get("source_type", source_type)
                existing.source_name = doc_data.get("source_name", source_name) or None
                existing.source_url = doc_data.get("source_url", source_url) or None
                existing.source_timestamp = coerce_source_timestamp(doc_data.get("source_timestamp", source_timestamp))
                existing.checksum = checksum
                existing.size_bytes = len(content.encode("utf-8"))
                existing.metadata = doc_data.get("metadata") or {}
                existing.status = DocumentStatus.PENDING
                existing.extraction_config_hash = extraction_config_hash
                existing.extraction_params = extraction_params_payload
                existing.error_message = None
                # #620: refresh session_id from the new metadata payload so
                # re-queued docs pick up an updated session tag.
                new_sid = _coerce_session_id_from_dict(doc_data.get("metadata"))
                if new_sid is not None:
                    existing.session_id = new_sid
                logger.debug(
                    f"submit_batch: re-queuing existing {prior_status.value} document "
                    f"(external_id={external_id!r}, doc_id={existing.id})"
                )
                try:
                    await storage.update_document(existing)
                    pending_docs.append(existing)
                except Exception as exc:
                    # Never interpolate the raw exc repr — a SQLAlchemy DBAPIError
                    # embeds the failed statement + its bind-param tuple (= the
                    # full document content + metadata). _safe_exc_summary bounds it.
                    safe_err = _safe_exc_summary(exc)
                    logger.warning(
                        f"submit_batch: could not update document record "
                        f"(external_id={external_id!r}, doc_id={existing.id}): {safe_err}"
                    )
                    pre_failed_docs.append((existing, safe_err))
                continue

            # No existing document — normal insert path.
            doc = Document(
                namespace_id=namespace_id,
                content=content,
                title=doc_data.get("title") or None,
                source=doc_data.get("source") or None,
                source_type=doc_data.get("source_type", source_type),
                source_name=doc_data.get("source_name", source_name) or None,
                source_url=doc_data.get("source_url", source_url) or None,
                source_timestamp=coerce_source_timestamp(doc_data.get("source_timestamp", source_timestamp)),
                checksum=checksum,
                size_bytes=len(content.encode("utf-8")),
                metadata=doc_data.get("metadata") or {},
                extraction_config_hash=extraction_config_hash,
                extraction_params=extraction_params_payload,
                external_id=external_id,
                session_id=_coerce_session_id_from_dict(doc_data.get("metadata")),
            )
            try:
                doc = await storage.create_document(doc)
                pending_docs.append(doc)
            except Exception as exc:
                # Never interpolate the raw exc repr — a SQLAlchemy DBAPIError
                # embeds the failed INSERT + its bind-param tuple (= the full
                # document content + metadata). _safe_exc_summary bounds it.
                safe_err = _safe_exc_summary(exc)
                logger.warning(
                    f"submit_batch: could not create document record "
                    f"(external_id={external_id!r}, doc_id={doc.id}): {safe_err}"
                )
                pre_failed_docs.append((doc, safe_err))

        # Initialize (or validate) the global chunk semaphore.
        # The first call that sets max_chunks_in_flight establishes the semaphore capacity.
        # Subsequent calls with a different value log a warning — the first value wins.
        if max_chunks_in_flight is not None:
            if self._chunk_semaphore is None:
                self._chunk_semaphore = _GlobalChunkSemaphore(max_chunks_in_flight)
            elif self._chunk_semaphore.capacity != max_chunks_in_flight:
                logger.warning(
                    f"submit_batch: max_chunks_in_flight={max_chunks_in_flight} conflicts with "
                    f"existing semaphore capacity={self._chunk_semaphore.capacity}; "
                    f"first value wins — using {self._chunk_semaphore.capacity}"
                )

        handle = BatchHandle(
            batch_id=uuid4(),
            total=len(pending_docs) + len(pre_failed_docs) + len(pre_completed_docs),
        )

        # Per-batch concurrency cap (#838): each batch carries its own
        # asyncio.Semaphore so concurrent submit_batch() calls don't stack
        # their limits. The unified pending processor acquires this around
        # _process_pending_item_impl below.
        per_batch_sem = asyncio.Semaphore(max_concurrent)

        # Create batch registration for callback delivery.
        batch_reg = _BatchRegistration(
            handle=handle,
            on_result=on_result,
            namespace_id=namespace_id,
            pre_failed_doc_ids=pre_failed_doc_ids,
            _remaining=len(pending_docs) + len(pre_failed_docs) + len(pre_completed_docs),
            concurrency_sem=per_batch_sem,
        )

        # Fire error results for documents that failed to be created.
        for doc, err in pre_failed_docs:
            batch_reg.fire_result(
                DocumentResult(
                    document_id=doc.id,
                    namespace_id=namespace_id,
                    success=False,
                    error=err,
                    external_id=doc.external_id,
                )
            )

        # Fire skipped results for documents already COMPLETED/PROCESSING/ARCHIVED.
        for doc in pre_completed_docs:
            batch_reg.fire_result(
                DocumentResult(
                    document_id=doc.id,
                    namespace_id=namespace_id,
                    success=True,
                    skipped=True,
                    chunks_created=doc.chunk_count,
                    entities_extracted=doc.entity_count,
                    relationships_created=doc.relationship_count,
                    external_id=doc.external_id,
                )
            )

        if not pending_docs and not pre_failed_docs and not pre_completed_docs:
            # Empty batch — nothing to process at all.
            handle._mark_done()
        elif pending_docs:
            # Enqueue PENDING docs for the unified processor.
            if self._processor_task is None or self._processor_task.done():
                raise RuntimeError(
                    f"submit_batch: pending processor is not running — cannot process {len(pending_docs)} "
                    "doc(s). Call start_pending_processor() before submitting documents that require "
                    "processing."
                )
            for doc in pending_docs:
                self._processor_queue.put_nowait(
                    _ProcessorItem(doc_id=doc.id, namespace_id=doc.namespace_id, batch_reg=batch_reg)
                )

        return handle

    async def recall(
        self,
        query: str,
        *,
        namespace: str | UUID,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.0,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        filter: RecallFilter | dict[str, Any] | None = None,
    ) -> RecallResult:
        """Recall memories relevant to a query.

        This is the primary method for retrieving memories. It:
        1. Uses LLM to understand query (entities, temporal refs, etc.)
        2. Searches across vector, graph, and keyword indexes
        3. Fuses results using Reciprocal Rank Fusion
        4. Returns ranked results

        Args:
            query: Query text
            namespace: Namespace UUID (as UUID or string)
            limit: Maximum results to return
            mode: Search mode (VECTOR, GRAPH, HYBRID, ALL)
            min_similarity: Minimum similarity threshold
            start_time: Deprecated. Optional lower bound (inclusive) for memory
                time. Timezone-aware datetimes are recommended; naive datetimes
                are assumed UTC. Prefer ``filter={"occurred_at": {"$gte": ...}}``.
            end_time: Deprecated. Optional upper bound (EXCLUSIVE — matches the
                legacy temporal window) for memory time. Same timezone semantics
                as start_time. Prefer ``filter={"occurred_at": {"$lt": ...}}``.
            filter: Optional deterministic recall filter — a
                :class:`~khora.filter.RecallFilter` instance or its dict/wire
                form (validated here). Cannot be combined with the deprecated
                ``start_time``/``end_time`` bounds.

        Returns:
            RecallResult with matched memories.  When using the VectorCypher
            engine, ``relationships`` contains scored relationship projections.
            Use ``khora.context_text(result)`` to render chunks, entities, and
            relationships as a formatted LLM context string.

        Raises:
            ValueError: If ``filter`` is combined with ``start_time``/``end_time``.
            RecallFilterValidationError: If ``filter`` fails validation.
        """
        import time as _time

        from khora.core.models.event import EventType, MemoryEvent
        from khora.telemetry.aggregate_metrics import record_recall_duration
        from khora.telemetry.context import (
            clear_trace_id,
            collect_usage,
            ensure_trace_id,
            start_usage_collection,
        )

        ensure_trace_id()
        start_usage_collection()
        _t0 = _time.perf_counter()
        _status = "success"
        _recall_id = uuid4()
        from khora.filter import RecallFilter, canonical_hash, metadata_leaf_count, parse_to_ast

        try:
            # Resolve the recall filter once, at the facade. The public
            # ``filter=`` kwarg and the deprecated ``start_time``/``end_time``
            # bounds are mutually exclusive but feed DIFFERENT axes on purpose:
            #
            #   * ``filter=`` produces a canonical ``filter_ast`` the engines
            #     compile and post-filter, so an ``occurred_at`` predicate is
            #     enforced against the EVENT-time axis (no created_at fallback) —
            #     the documented event-time semantics of the new API.
            #   * ``start_time``/``end_time`` produce ONLY a ``temporal_filter``
            #     recency window. They are a window-axis API: every engine narrows
            #     its channel reads on ``COALESCE(source_timestamp, created_at)``,
            #     so a chunk recent by ingest time survives even when it carries no
            #     event-time anchor. Folding these bounds into an ``occurred_at``
            #     ``filter_ast`` (as an earlier revision did, speculatively) would
            #     AND an event-time post-filter on top of the window and false-empty
            #     every anchor-less chunk a plain ``remember()`` produces. They must
            #     NOT set ``filter_ast``.
            temporal_filter: Any = None
            filter_ast: Any = None
            if filter is not None and (start_time is not None or end_time is not None):
                raise ValueError("Pass either filter= or the deprecated start_time/end_time, not both")
            if filter is not None:
                # Validate once: dict → model_validate, instance → use as-is.
                # RecallFilterValidationError propagates unwrapped.
                recall_filter = filter if isinstance(filter, RecallFilter) else RecallFilter.model_validate(filter)
                filter_ast = parse_to_ast(recall_filter)
            elif start_time is not None or end_time is not None:
                warnings.warn(
                    "start_time/end_time are deprecated; use filter={'occurred_at': {...}}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Normalize tz: naive → UTC, aware → UTC.
                norm_start = _normalize_recall_bound(start_time)
                norm_end = _normalize_recall_bound(end_time)
                from khora.core.temporal import ChunkTemporalFilter as SkeletonTemporalFilter

                # Window-axis only: the recency bounds narrow on
                # ``COALESCE(source_timestamp, created_at)`` inside each engine.
                # ``filter_ast`` stays None so no event-time post-filter is AND-ed
                # on top (see the axis note above).
                temporal_filter = SkeletonTemporalFilter(
                    occurred_after=norm_start,
                    occurred_before=norm_end,
                )
            namespace_id = await self._resolve_namespace(namespace)

            # Emit RECALL_REQUESTED before any embedding/retrieval work.
            try:
                await self._dispatch_hook(
                    MemoryEvent(
                        namespace_id=namespace_id,
                        event_type=EventType.RECALL_REQUESTED,
                        resource_type="recall",
                        resource_id=_recall_id,
                        data={
                            "query": query,
                            "k": limit,
                            "mode": getattr(mode, "value", str(mode)) if mode is not None else "default",
                            "namespace_id": str(namespace_id),
                        },
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Hook dispatch failed for {}: {}", EventType.RECALL_REQUESTED.value, exc)

            with trace_span(
                "khora.recall",
                namespace_id=str(namespace_id),
                query_hash=bounded_text_hash(query),
                query_length=len(query),
            ) as _recall_span:
                # Tag the canonical filter hash only when a filter is present, so
                # the common no-filter recall does not carry a meaningless attribute.
                if filter_ast is not None:
                    _recall_span.set_attribute("filter.canonical_hash", canonical_hash(filter_ast))
                    _recall_span.set_attribute("filter.metadata_leaf_count", metadata_leaf_count(filter_ast))
                result = await self._get_engine().recall(
                    query,
                    namespace_id,
                    limit=limit,
                    mode=mode,
                    min_similarity=min_similarity,
                    temporal_filter=temporal_filter,
                    filter_ast=filter_ast,
                )
                # Engines return DocumentProjection stubs (id + bare
                # essentials). Batch-fetch full source metadata, derive
                # per-chunk entity linkage, and emit a new RecallResult.
                result = await self._upgrade_recall_documents(result, namespace_id)

                # The engine's ``engine_info["filter"]`` is passed through
                # unchanged. The skeleton engine writes the canonical
                # FilterPushdownReport (``report.model_dump(mode="json")``);
                # chronicle currently writes its own hand-rolled
                # ``{pushed_down, post_filtered}`` dict (canonical adoption is a
                # follow-up). The facade no longer synthesizes a ``{engine,
                # supported, pushed_down}`` shape over it (#1069): ``supported`` /
                # the facade-level ``engine`` were not part of the canonical schema
                # and were read nowhere. Engines that do not report filter pushdown
                # simply omit the key, so callers must treat ``"filter"`` as
                # optional (key on its presence, not assume it).

                # Emit RECALL_RESULTS_READY after engine returns, before packaging.
                try:
                    _top_score = result.chunks[0].score if result.chunks else None
                    _entity_ids = [str(e.id) for e in result.entities[:20]]
                    _chunk_ids = [str(c.id) for c in result.chunks[:20]]
                    await self._dispatch_hook(
                        MemoryEvent(
                            namespace_id=namespace_id,
                            event_type=EventType.RECALL_RESULTS_READY,
                            resource_type="recall",
                            resource_id=_recall_id,
                            data={
                                "query": query,
                                "result_count": len(result.chunks),
                                "top_score": _top_score,
                                "entity_ids": _entity_ids,
                                "chunk_ids": _chunk_ids,
                                "abstention_signals": (result.engine_info or {}).get("abstention_signals"),
                            },
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Hook dispatch failed for {}: {}", EventType.RECALL_RESULTS_READY.value, exc)

                packaged = replace(result, llm_usage=collect_usage())

                # Emit RECALL_COMPLETED with end-to-end timing.
                try:
                    await self._dispatch_hook(
                        MemoryEvent(
                            namespace_id=namespace_id,
                            event_type=EventType.RECALL_COMPLETED,
                            resource_type="recall",
                            resource_id=_recall_id,
                            data={
                                "query": query,
                                "latency_ms": (_time.perf_counter() - _t0) * 1000.0,
                                "result_count": len(packaged.chunks) if packaged.chunks else 0,
                                "llm_usage": (packaged.engine_info or {}).get("llm_usage"),
                            },
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Hook dispatch failed for {}: {}", EventType.RECALL_COMPLETED.value, exc)

                return packaged
        except Exception:
            _status = "error"
            raise
        finally:
            collect_usage()  # idempotent — drains queue if not already collected
            clear_trace_id()
            record_recall_duration(
                _time.perf_counter() - _t0,
                engine=self._engine_name,
                mode=getattr(mode, "value", str(mode)),
                status=_status,
            )

    async def forget(
        self,
        document_id: UUID,
        *,
        namespace: str | UUID,
    ) -> bool:
        """Remove a memory.

        Args:
            document_id: ID of the document to remove
            namespace: Namespace UUID (as UUID or string)

        Returns:
            True if deleted, False if not found
        """
        namespace_id = await self._resolve_namespace(namespace)

        with trace_span(
            "khora.forget",
            namespace_id=str(namespace_id),
            document_id=str(document_id),
        ):
            return await self._get_engine().forget(document_id, namespace_id)

    async def forget_session(
        self,
        namespace_id: UUID,
        session_id: UUID,
    ) -> int:
        """Delete every document in *namespace_id* tagged with *session_id*.

        Single transactional ``DELETE FROM documents WHERE namespace_id=?
        AND session_id=?``. The ``ON DELETE CASCADE`` on
        ``chunks.document_id`` propagates the deletion to the chunks table.
        Graph-side cleanup runs afterwards as a best-effort pass over the
        deleted document ids (Neo4j Chunk nodes via ``forget()``). Returns
        the number of documents that were deleted.

        Use this for session-scoped retention (#620): an agentic adapter
        ending a conversation can call ``forget_session(ns, session)`` and
        not leave orphaned chunks behind.

        Args:
            namespace_id: Stable namespace UUID (resolved to the active
                version automatically).
            session_id: Session UUID stamped on the documents at ingest.

        Returns:
            Count of documents deleted (0 if none matched).
        """
        resolved_ns = await self._resolve_namespace(namespace_id)
        storage = self.storage

        with trace_span(
            "khora.forget_session",
            namespace_id=str(resolved_ns),
            session_id=str(session_id),
        ):
            # Snapshot the document ids first so the graph/vector cleanup
            # can target them after the SQL DELETE commits. ``list_documents``
            # has no session_id filter today, so we filter client-side; the
            # partial index ``ix_documents_ns_session`` will pay off when a
            # future kwarg lands on the backend (see ticket follow-ups).
            matching_ids: list[UUID] = []
            page_size = 500
            offset = 0
            while True:
                page = await storage.list_documents(resolved_ns, limit=page_size, offset=offset)
                matching_ids.extend(d.id for d in page if d.session_id == session_id)
                if len(page) < page_size:
                    break
                offset += page_size

            if not matching_ids:
                return 0

            # Route each delete through the engine so chunks / Chunk nodes /
            # extracted entities all get the same cascade ``forget()`` runs.
            # This pays the per-doc cost in exchange for matching the
            # single-document forget contract — graph backends without
            # batched delete (Neo4j Cypher MERGE/DETACH DELETE) end up paying
            # the same cost anyway.
            deleted = 0
            for doc_id in matching_ids:
                ok = await self._get_engine().forget(doc_id, resolved_ns)
                if ok:
                    deleted += 1
            return deleted

    # =========================================================================
    # Dream-phase (#649 / #650 scaffolding — orchestrator stubs)
    # =========================================================================

    async def dream(
        self,
        namespace: str | UUID,
        *,
        mode: str = "dry-run",
        scope: Any = None,
        ops: Any = None,
        config: Any = None,
        expertise: Any = None,
        on_progress: Callable[[Any], None] | None = None,
        resume_from: UUID | None = None,
    ) -> Any:
        """Run a dream-phase pass over ``namespace``.

        Phase 0.1 scaffolding — body raises ``NotImplementedError`` until
        the orchestrator ships in #661. The signature is settled so
        callers can wire against it.

        ``expertise`` is an optional :class:`ExpertiseConfig` forwarded to
        the schema-drift planner (#1036). It is required for the
        ``schema_drift`` op; when omitted, that op is skipped with a
        ``skip_reasons`` entry rather than dropped silently.
        """
        from khora.dream.api import dream as _dream

        return await _dream(
            self,
            namespace,
            mode=mode,
            scope=scope,
            ops=ops,
            config=config,
            expertise=expertise,
            on_progress=on_progress,
            resume_from=resume_from,
        )

    async def dream_status(self, run_id: UUID) -> dict[str, object]:
        """Return live or post-mortem status for a dream run."""
        from khora.dream.api import dream_status as _dream_status

        return await _dream_status(self, run_id)

    async def dream_history(
        self,
        namespace: str | UUID,
        *,
        limit: int = 20,
    ) -> list[Any]:
        """Return recent dream-run results for ``namespace`` (newest first)."""
        from khora.dream.api import dream_history as _dream_history

        return await _dream_history(self, namespace, limit=limit)

    async def dream_undo(
        self,
        op_id: UUID,
        *,
        base_dir: str | Path | None = None,
    ) -> bool:
        """Reverse a previously-applied dream op by ``op_id``.

        Reads the run's ``undo.json`` (schema ``dream-undo/1`` written by
        :class:`khora.dream.report.DreamFileSink`), locates the op, and
        dispatches to the op-type-specific reverse handler inside one
        coordinator transaction. Returns ``True`` when at least one row
        was restored and ``False`` for unknown / already-undone ops.

        See :func:`khora.dream.api.dream_undo` for the full contract.
        """
        from khora.dream.api import dream_undo as _dream_undo

        return await _dream_undo(self, op_id, base_dir=base_dir)

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def get_entity(
        self,
        entity_id: UUID,
        *,
        namespace: str | UUID,
        include_sources: bool = False,
    ) -> Entity | None:
        """Get an entity by ID, scoped to a namespace.

        Args:
            entity_id: Entity UUID to retrieve
            namespace: Namespace UUID (as UUID or string). Required —
                returns ``None`` when the entity belongs to a different
                namespace (prevents cross-tenant IDOR).
            include_sources: If True, populate source document metadata on
                the returned entity (default: False)

        Returns:
            Entity if found in the namespace, else None
        """
        namespace_id = await self._resolve_namespace(namespace)
        entity = await self._get_engine().get_entity(entity_id, namespace_id=namespace_id)
        if entity is not None and include_sources:
            await self._populate_sources([], [entity], [], namespace_id=namespace_id)
        return entity

    async def list_entities(
        self,
        *,
        namespace: str | UUID,
        entity_type: str | None = None,
        limit: int = 100,
        include_sources: bool = False,
    ) -> list[Entity]:
        """List entities in a namespace.

        Args:
            namespace: Namespace UUID (as UUID or string)
            entity_type: Optional entity type filter
            limit: Maximum entities to return
            include_sources: If True, populate source document metadata on
                returned entities (default: False)

        Returns:
            List of Entity objects
        """
        namespace_id = await self._resolve_namespace(namespace)
        entities = await self._get_engine().list_entities(namespace_id, entity_type=entity_type, limit=limit)
        if include_sources:
            await self._populate_sources([], entities, [], namespace_id=namespace_id)
        return entities

    async def find_related_entities(
        self,
        entity_id: UUID,
        *,
        namespace: str | UUID,
        max_depth: int = 2,
        limit: int = 20,
        include_sources: bool = False,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity.

        Args:
            entity_id: Entity UUID to find related entities for
            namespace: Namespace UUID (as UUID or string)
            max_depth: Maximum graph traversal depth
            limit: Maximum entities to return
            include_sources: If True, populate source document metadata on
                returned entities (default: False)

        Returns:
            List of (Entity, score) tuples
        """
        namespace_id = await self._resolve_namespace(namespace)
        results = await self._get_engine().find_related_entities(
            entity_id,
            namespace_id,
            max_depth=max_depth,
            limit=limit,
        )
        if include_sources:
            await self._populate_sources([], results, [], namespace_id=namespace_id)
        return results

    async def get_communities(
        self,
        *,
        namespace: str | UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CommunityNode]:
        """Return materialized dream community summaries for a namespace (#1276).

        The GraphRAG payoff: the dream ``community_summary`` op computes
        LLM-grounded community summaries and the post-commit mirror materializes
        them into the graph as :Community nodes. This read surfaces them at
        recall. Read-only; returns an empty list on a stack without a graph
        backend or without materialized communities.

        Args:
            namespace: Namespace UUID (as UUID or string)
            limit: Maximum communities to return
            offset: Pagination offset

        Returns:
            List of CommunityNode objects (summary text + member ids)
        """
        namespace_id = await self._resolve_namespace(namespace)
        return await self.storage.get_communities(namespace_id, limit=limit, offset=offset)

    async def get_entity_communities(
        self,
        entity_ids: list[UUID],
        *,
        namespace: str | UUID,
    ) -> list[CommunityNode]:
        """Return the dream community summaries the given entities belong to (#1276).

        The entity-anchored leg of the community recall reader: given a recall
        hit's entity set, fetch the community summaries they are members of so a
        caller can surface community context alongside the entity hits.

        Args:
            entity_ids: Entity UUIDs to look up community membership for
            namespace: Namespace UUID (as UUID or string)

        Returns:
            List of CommunityNode objects, deduplicated by community id
        """
        namespace_id = await self._resolve_namespace(namespace)
        return await self.storage.get_entity_communities(entity_ids, namespace_id=namespace_id)

    # =========================================================================
    # Document Operations (Convenience Methods)
    # =========================================================================

    async def get_document(
        self,
        document_id: UUID,
        *,
        namespace: str | UUID,
    ) -> Document | None:
        """Get a document by ID, scoped to ``namespace``.

        Args:
            document_id: Document UUID
            namespace: Namespace UUID (as UUID or string) — the caller's
                namespace; cross-tenant lookups by id return ``None``
                (IDOR).

        Returns:
            Document or None if not found (or not in this namespace)
        """
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_engine().get_document(document_id, namespace_id=namespace_id)

    async def list_documents(
        self,
        *,
        namespace: str | UUID,
        limit: int = 100,
    ) -> list[Document]:
        """List documents in a namespace.

        Args:
            namespace: Namespace UUID (as UUID or string)
            limit: Maximum documents to return

        Returns:
            List of Documents
        """
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_engine().list_documents(namespace_id, limit=limit)

    async def search_entities(
        self,
        query: str,
        *,
        namespace: str | UUID,
        limit: int = 10,
        include_sources: bool = False,
    ) -> list[Entity]:
        """Search entities by query text using embedding similarity.

        Args:
            query: Search query text
            namespace: Namespace UUID (as UUID or string)
            limit: Maximum entities to return
            include_sources: If True, populate source document metadata on
                returned entities (default: False)

        Returns:
            List of matching Entities (most similar first)
        """
        namespace_id = await self._resolve_namespace(namespace)
        entities = await self._get_engine().search_entities(query, namespace_id, limit=limit)
        if include_sources:
            await self._populate_sources([], entities, [], namespace_id=namespace_id)
        return entities

    async def stats(self, *, namespace: str | UUID) -> Stats:
        """Get document/chunk/entity/relationship counts for a namespace.

        Args:
            namespace: Namespace UUID (as UUID or string)

        Returns:
            Stats with document/chunk/entity/relationship counts
        """
        namespace_id = await self._resolve_namespace(namespace)
        return await self._get_engine().stats(namespace_id)

    # =========================================================================
    # Helpers
    # =========================================================================

    async def _populate_sources(
        self,
        chunks: list[tuple[Chunk, float]],
        entities: list[tuple[Entity, float]] | list[Entity],
        relationships: list[tuple[Relationship, float]],
        *,
        namespace_id: UUID,
    ) -> None:
        """Batch-fetch document sources and populate entity/chunk/relationship fields **in-place**.

        ``entities`` accepts either ``list[Entity]`` or
        ``list[tuple[Entity, float]]`` (entity, score pairs).  The method
        unwraps tuples transparently.

        Collects unique document IDs from *chunks*, *entities*, and
        *relationships*, fetches lightweight metadata via batched SELECTs
        (chunked at 1 000 IDs), then populates ``chunk.source_document``,
        ``entity.source_documents``, and ``relationship.source_documents``
        on the provided objects.  No value is returned; callers observe
        changes through the mutated inputs.
        """
        # Collect unique doc IDs
        doc_ids: set[UUID] = set()
        for chunk, _score in chunks:
            doc_ids.add(chunk.document_id)
        for item in entities:
            entity = item[0] if isinstance(item, tuple) else item
            doc_ids.update(entity.source_document_ids)
        for rel, _score in relationships:
            doc_ids.update(rel.source_document_ids)

        if not doc_ids:
            return

        sorted_ids = sorted(doc_ids)
        sources: dict = {}
        for i in range(0, len(sorted_ids), 1000):
            batch = sorted_ids[i : i + 1000]
            sources.update(await self.storage.get_document_sources_batch(batch, namespace_id=namespace_id))

        # Populate chunks
        for chunk, _score in chunks:
            chunk.source_document = sources.get(chunk.document_id)

        # Populate entities
        for item in entities:
            entity = item[0] if isinstance(item, tuple) else item
            entity_sources = {did: sources[did] for did in entity.source_document_ids if did in sources}
            # None means either "sources not fetched" (include_sources=False) or
            # "all source documents deleted".  Callers distinguish via the
            # include_sources flag they passed.
            entity.source_documents = entity_sources if entity_sources else None

        # Populate relationships
        for rel, _score in relationships:
            rel_sources = {did: sources[did] for did in rel.source_document_ids if did in sources}
            rel.source_documents = rel_sources if rel_sources else None

    async def _upgrade_recall_documents(self, result: RecallResult, namespace_id: UUID) -> RecallResult:
        """Batch-fetch document sources and return an upgraded RecallResult.

        Engines return ``RecallResult`` with ``DocumentProjection`` stubs
        that carry an ``id`` plus whatever fields the engine already had
        in hand (typically just ``created_at`` and ``source_type``). This
        helper unions every document id referenced by chunks, entities,
        or relationships, performs a single batched lookup via
        ``storage.get_document_projections_batch``, and produces a fresh
        ``RecallResult`` with:

        - ``documents``: replaced with full ``DocumentProjection`` rows
          where the relational store returned a row; stubs that the
          relational store can't resolve are preserved (and counted as
          dangling refs).
        - ``chunks``: ``connected_entity_ids`` populated by inverting
          ``RecallEntity.source_chunk_ids`` (the canonical chunk → entity
          linkage produced upstream by extraction).

        ``RecallResult`` is frozen, so the return value is a new instance
        via ``dataclasses.replace`` — never an in-place mutation.
        """
        # Collect every document id referenced anywhere in the result.
        doc_ids: set[UUID] = set()
        for chunk in result.chunks:
            doc_ids.add(chunk.document_id)
        for entity in result.entities:
            doc_ids.update(entity.source_document_ids)
        for rel in result.relationships:
            doc_ids.update(rel.source_document_ids)

        if not doc_ids:
            return result

        sorted_ids = sorted(doc_ids)
        projections: dict[UUID, DocumentProjection] = {}
        upgrade_failed_reason: str | None = None
        with trace_span(
            "khora.recall.document_upgrade",
            namespace_id=str(namespace_id),
            doc_count=len(sorted_ids),
        ):
            try:
                for i in range(0, len(sorted_ids), 1000):
                    batch = sorted_ids[i : i + 1000]
                    projections.update(
                        await self.storage.get_document_projections_batch(batch, namespace_id=namespace_id)
                    )
            except Exception as exc:
                # Fail open: the engine call already succeeded; degrading the
                # document-upgrade pass to "stubs only" is strictly better than
                # crashing the recall. Surface the reason in engine_info so
                # consumers can detect the degraded state.
                upgrade_failed_reason = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "recall document upgrade failed; returning engine stubs (reason={}) ns={}",
                    upgrade_failed_reason,
                    namespace_id,
                )
                projections = {}

        # Preserve the engine-emitted document list order: replace stubs
        # whose id resolved against the relational store, leave the rest
        # untouched.
        seen_ids: set[UUID] = set()
        upgraded_documents: list[DocumentProjection] = []
        for stub in result.documents:
            seen_ids.add(stub.id)
            upgraded_documents.append(projections.get(stub.id, stub))

        # Producer invariant: every referenced doc id must appear in
        # ``documents``. Engines mostly emit a stub for every id they
        # touch, but if any slip through, materialise an entry here —
        # using the storage row when available, otherwise a minimal
        # stub — rather than corrupting the invariant.
        for did in sorted(doc_ids - seen_ids):
            proj = projections.get(did)
            if proj is not None:
                upgraded_documents.append(proj)
                continue
            upgraded_documents.append(DocumentProjection(id=did, created_at=datetime.now(UTC), source_type="library"))

        # Dangling refs: ids referenced by some chunk / entity / rel that
        # the relational store could not resolve. Counted by referrer
        # kind for triage; ``namespace_id`` stays on the span / log.
        unresolved = doc_ids - set(projections.keys())
        if unresolved:
            chunk_dangling = sum(1 for c in result.chunks if c.document_id in unresolved)
            entity_dangling = sum(1 for e in result.entities for did in e.source_document_ids if did in unresolved)
            rel_dangling = sum(1 for r in result.relationships for did in r.source_document_ids if did in unresolved)
            if chunk_dangling:
                _RECALL_DANGLING_REF_COUNTER.add(chunk_dangling, attributes={"referrer": "chunk"})
            if entity_dangling:
                _RECALL_DANGLING_REF_COUNTER.add(entity_dangling, attributes={"referrer": "entity"})
            if rel_dangling:
                _RECALL_DANGLING_REF_COUNTER.add(rel_dangling, attributes={"referrer": "relationship"})
            logger.debug(
                "recall: {} dangling document refs (chunks={}, entities={}, rels={}) ns={}",
                len(unresolved),
                chunk_dangling,
                entity_dangling,
                rel_dangling,
                namespace_id,
            )

        # Per-chunk entity linkage: invert ``RecallEntity.source_chunk_ids``
        # — the canonical chunk → entity mapping produced by extraction.
        # ``connected_entity_ids`` reflects only entities that survived
        # the engine's filtering and made it into ``result.entities``;
        # chunks may reference additional entities that fell below the
        # recall threshold (or were never extracted) — those are
        # intentionally omitted. Empty list means "no linkage in this
        # result", not "no edges in the graph". Entity-less stacks
        # (skeleton, chronicle without entity hits) leave every chunk
        # with the default empty list. Ordering follows
        # ``result.entities`` (the engine's score-sorted order).
        chunk_to_entities: dict[UUID, list[UUID]] = {}
        for entity in result.entities:
            for cid in entity.source_chunk_ids:
                chunk_to_entities.setdefault(cid, []).append(entity.id)

        if chunk_to_entities:
            upgraded_chunks = [
                replace(chunk, connected_entity_ids=chunk_to_entities.get(chunk.id, [])) for chunk in result.chunks
            ]
        else:
            upgraded_chunks = list(result.chunks)

        if upgrade_failed_reason is not None:
            engine_info = dict(result.engine_info)
            engine_info["document_upgrade_failed"] = upgrade_failed_reason
            return replace(
                result,
                documents=upgraded_documents,
                chunks=upgraded_chunks,
                engine_info=engine_info,
            )
        return replace(result, documents=upgraded_documents, chunks=upgraded_chunks)

    # ------------------------------------------------------------------
    # Semantic hooks (subscription API)
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: str,
        callback: Callable[..., Any],
        *,
        filter: Any | None = None,
        namespace_id: UUID | None = None,
    ) -> UUID:
        """Subscribe to extraction events with optional semantic filtering.

        Registers an async callback that fires during document ingestion
        when an event of the given type occurs. Optionally, attach a
        ``SemanticFilter`` to narrow matches by entity type, embedding
        similarity, or LLM evaluation.

        Args:
            event_type: Event type string or ``EventType`` enum
                (e.g., ``"entity.created"``, ``EventType.ENTITY_CREATED``).
            callback: Async function ``async def handler(event: MemoryEvent) -> None``.
            filter: Optional ``SemanticFilter`` for type/embedding/LLM gating.
            namespace_id: Scope to a specific namespace (None = all).

        Returns:
            Subscription UUID for later ``unsubscribe()``.

        Example::

            async def on_entity(event):
                print(f"New entity: {event.data.get('name')}")

            sub_id = kb.subscribe("entity.created", on_entity)
            await kb.remember("Acme Corp announced...", ...)
            kb.unsubscribe(sub_id)
        """
        return self._get_hook_dispatcher().subscribe(
            event_type,
            callback,
            filter=filter,
            namespace_id=namespace_id,
        )

    def unsubscribe(self, subscription_id: UUID) -> bool:
        """Remove a hook subscription.

        Returns True if found and removed, False otherwise.
        """
        return self._get_hook_dispatcher().unsubscribe(subscription_id)

    async def subscribe_persistent(
        self,
        event_type: Any,
        delivery: dict[str, Any],
        *,
        filter: Any = None,
        namespace_id: UUID | None = None,
    ) -> UUID:
        """Register a durable hook subscription (#599).

        Unlike :meth:`subscribe` (an in-process callback that dies with the
        process), a persistent subscription records its ``delivery`` config
        (webhook URL / queue identifier) to PostgreSQL so it survives a
        restart. Requires the durable store wired at ``connect()`` (any SQL
        backend); raises ``RuntimeError`` on a store-less stack.

        Returns the subscription UUID.
        """
        return await self._get_hook_dispatcher().register_persistent(
            event_type,
            delivery,
            filter=filter,
            namespace_id=namespace_id,
        )

    async def unsubscribe_persistent(self, subscription_id: UUID) -> bool:
        """Remove a persistent hook subscription from memory and storage (#599)."""
        return await self._get_hook_dispatcher().unregister_persistent(subscription_id)

    async def _wire_persistent_hooks(self, storage: Any) -> None:
        """Attach a durable subscription store to the dispatcher and reload.

        Resolves a SQL session factory from the coordinator (mirrors
        ``coordinator.transaction()``'s factory resolution). When none is
        available (no SQL backend), persistent hooks stay disabled.
        """
        factory = None
        for attr in ("_relational", "_vector", "_event_store"):
            backend = getattr(storage, attr, None)
            sf = getattr(backend, "_session_factory", None)
            if sf is not None:
                factory = sf
                break
        if factory is None:
            return

        from khora.hooks.subscription_store import HookSubscriptionStore

        dispatcher = self._get_hook_dispatcher()
        dispatcher._subscription_store = HookSubscriptionStore(factory)
        await dispatcher.load_persistent()

    def _get_hook_dispatcher(self) -> Any:
        """Lazy-initialize the hook dispatcher."""
        if not hasattr(self, "_hook_dispatcher"):
            from khora.hooks.dispatcher import HookDispatcher

            hooks_config = getattr(self._config, "hooks", None)
            max_concurrent = 10
            callback_timeout = 30.0
            if hooks_config:
                max_concurrent = getattr(hooks_config, "max_concurrent_callbacks", 10)
                callback_timeout = getattr(hooks_config, "callback_timeout_seconds", 30.0)
            self._hook_dispatcher = HookDispatcher(
                max_concurrent=max_concurrent,
                callback_timeout_seconds=callback_timeout,
                config=hooks_config,
            )
        return self._hook_dispatcher

    @property
    def hooks(self) -> Any:
        """Access the hook dispatcher directly for advanced usage."""
        return self._get_hook_dispatcher()

    async def _dispatch_hook(self, event: Any) -> None:
        """Dispatch an event to hook subscribers (internal, called by engines)."""
        dispatcher = self._get_hook_dispatcher()
        if dispatcher.subscription_count > 0:
            await dispatcher.dispatch(event)

    async def _resolve_namespace(self, namespace: str | UUID) -> UUID:
        """Resolve a namespace_id to the active version's row-level id.

        Accepts a stable namespace_id (UUID or string) and resolves it to
        the row-level id of the currently active version via DB lookup.
        """
        if isinstance(namespace, str):
            try:
                namespace = UUID(namespace)
            except ValueError:
                raise ValueError(f"Invalid namespace: {namespace!r}. Must be a valid UUID.")

        return await self.storage.resolve_namespace(namespace)

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components."""
        if not self._connected or self._engine is None:
            return {"status": "disconnected"}

        return await self._engine.health_check()


# Attach the singleton accessor onto Khora. Done outside the class body
# because `_SharedAccessor` exists only after the class definition
# closes (it references Khora) — see `_SharedAccessor.__call__` /
# `clear`. Callers write `await Khora.shared()` and `await Khora.shared.clear()`.
Khora.shared = _SharedAccessor()


# Convenience function for one-off usage
@asynccontextmanager
async def khora(
    config: KhoraConfig | None = None,
) -> AsyncGenerator[Khora]:
    """Context manager for one-off Khora usage.

    Usage:
        async with khora() as kb:
            await kb.remember("Hello, world!")
            result = await kb.recall("greeting")
    """
    kb = Khora(config)
    try:
        await kb.connect()
        yield kb
    finally:
        await kb.disconnect()
