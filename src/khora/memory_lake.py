"""MemoryLake - Primary API for Khora Memory Lake.

This is the main entry point for using Khora as a library.
Provides a simple, unified interface for memory storage and retrieval.

The MemoryLake class is a thin facade that delegates to pluggable engines.
The default engine is "vectorcypher" which uses knowledge graphs, vectors, and LLM extraction.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger

from khora.config import KhoraConfig, load_config
from khora.core.models import Chunk, Document, Entity, MemoryNamespace
from khora.query import SearchMode
from khora.telemetry import trace_span


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

    async def acquire(self, n: int) -> None:
        """Block until n tokens are available, then acquire them."""
        # Clamp to capacity to avoid permanent deadlock when n > capacity.
        n = min(n, self._capacity)
        async with self._condition:
            while self._in_flight + n > self._capacity:
                await self._condition.wait()
            self._in_flight += n

    async def release(self, n: int) -> None:
        """Release n tokens and wake any waiters."""
        async with self._condition:
            self._in_flight -= n
            self._condition.notify_all()


if TYPE_CHECKING:
    from khora.core.models import Relationship
    from khora.engines.protocol import MemoryEngineProtocol
    from khora.extraction.chunkers import ChunkStrategy
    from khora.extraction.skills import ExpertiseConfig
    from khora.storage import StorageConfig, StorageCoordinator


# DYT-645: LLMUsage is a public API type consumed by Poros (DYT-650) and Peras (DYT-651).
# Changes to field names or types require coordination with those projects.
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


@dataclass(slots=True, frozen=True)
class RememberResult:
    """Result of a remember operation."""

    document_id: UUID
    namespace_id: UUID
    chunks_created: int
    entities_extracted: int
    relationships_created: int
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
    """Namespace statistics."""

    documents: int
    chunks: int
    entities: int
    relationships: int
    last_activity_at: datetime | None = None


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


@dataclass(slots=True, frozen=True)
class RecallResult:
    """Result of a recall operation.

    Attributes:
        query: The original query string.
        namespace_id: Namespace the recall was executed against.
        chunks: Scored chunk tuples ``(Chunk, score)``.
        entities: Scored entity tuples ``(Entity, score)``.
        context_text: Pre-formatted text for LLM context.  When relationships
            are present, includes a ``--- Relationships ---`` section.
        metadata: Engine-specific metadata dict.
        relationships: Scored relationship tuples ``(Relationship, score)``.
            Populated only by the VectorCypher engine; empty list for other
            engines.
    """

    query: str
    namespace_id: UUID
    chunks: list[tuple[Chunk, float]]
    entities: list[tuple[Entity, float]]
    context_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    relationships: list[tuple[Relationship, float]] = field(default_factory=list)
    llm_usage: list[LLMUsage] = field(default_factory=list)


class MemoryLake:
    """Primary interface for Khora Memory Lake.

    Provides a simple API for storing and retrieving memories:
    - remember(): Store content in the memory lake
    - recall(): Retrieve relevant memories for a query
    - forget(): Remove memories
    - create_namespace(): Create a new memory namespace
    - get_namespace_by_stable_id(): Get a namespace by its stable ID

    Can be used as a context manager for automatic connection handling.

    The MemoryLake is a facade that delegates to pluggable engines.
    The default engine is "vectorcypher" which uses knowledge graphs and vector embeddings.

    Usage:
        # Simplest - from env vars (KHORA_DATABASE_URL)
        async with MemoryLake() as lake:
            await lake.remember("Important fact...", namespace=namespace_id,
                entity_types=["PERSON", "CONCEPT"], relationship_types=["RELATES_TO"])

        # Common - explicit database URL
        async with MemoryLake("postgresql://localhost/mydb") as lake:
            results = await lake.recall("What do I know about...", namespace=namespace_id)

        # With graph backend
        async with MemoryLake("postgresql://...", graph_url="bolt://localhost:7687") as lake:
            ...

        # Explicit engine selection (same as default)
        async with MemoryLake("postgresql://...", engine="vectorcypher") as lake:
            ...

        # Full config
        async with MemoryLake(KhoraConfig(...)) as lake:
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
        """Initialize the Memory Lake.

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
            lake = MemoryLake()

            # Common - explicit database
            lake = MemoryLake("postgresql://localhost/mydb")

            # With graph
            lake = MemoryLake("postgresql://...", graph_url="bolt://...")

            # Explicit engine selection
            lake = MemoryLake("postgresql://...", engine="vectorcypher")

            # Full config
            lake = MemoryLake(KhoraConfig(...))
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
                self._config.neo4j_url = graph_url
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

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting Memory Lake...")

        if self._run_migrations:
            from khora.db.session import run_migrations as _run_migrations

            # For the sqlite_lance embedded backend, derive a sqlite+aiosqlite URL
            # from the configured db_path so Alembic migrations target the same
            # file the adapters use. DYT-2727 made the migrations dialect-aware.
            db_url: str | None
            if (
                getattr(self._config.storage, "backend", "postgres") == "sqlite_lance"
                and self._config.storage.sqlite_lance is not None
            ):
                db_path = self._config.storage.sqlite_lance.db_path
                db_url = f"sqlite+aiosqlite:///{db_path}"
            else:
                db_url = self._config.database_url
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
        # ingestion pipeline can dispatch events without knowing about MemoryLake.
        storage = getattr(self._engine, "_storage", None)
        if storage is not None:
            try:
                storage._hook_dispatcher = self._get_hook_dispatcher()
            except (AttributeError, TypeError):
                pass  # Mock or non-standard engine — hooks won't fire

        self._connected = True
        logger.info("Memory Lake connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting Memory Lake...")

        if self._engine:
            await self._engine.disconnect()
            self._engine = None

        self._connected = False
        logger.info("Memory Lake disconnected")

    async def __aenter__(self) -> MemoryLake:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def _get_engine(self) -> MemoryEngineProtocol:
        """Get the engine (internal use)."""
        if self._engine is None:
            raise RuntimeError("Memory Lake not connected. Call connect() first.")
        return self._engine

    @property
    def storage(self) -> StorageCoordinator:
        """Get the storage coordinator for admin/management operations.

        Provides direct access to the underlying storage coordinator for
        managing namespaces and other administrative tasks not covered
        by the high-level API.

        For common operations, prefer the MemoryLake convenience methods:
        - lake.get_document() for document retrieval
        - lake.list_documents() for document listing
        - lake.search_entities() for entity search
        - lake.stats() for namespace statistics
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
    ) -> MemoryNamespace:
        """Create a new memory namespace.

        Args:
            config_overrides: Optional configuration overrides

        Returns:
            Created MemoryNamespace
        """
        return await self._get_engine().create_namespace(
            config_overrides=config_overrides,
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
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        external_id: str | None = None,
    ) -> RememberResult:
        """Store content in the memory lake.

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
            RememberResult with details
        """
        from khora.telemetry.context import (
            clear_trace_id,
            collect_usage,
            ensure_trace_id,
            start_usage_collection,
        )

        ensure_trace_id()
        start_usage_collection()
        try:
            namespace_id = await self._resolve_namespace(namespace)
            with trace_span("khora.remember", namespace_id=str(namespace_id), content_length=len(content)):
                # NOTE: expertise and extraction_config_hash are always forwarded,
                # even when None. Custom engines registered via register_engine()
                # must accept these kwargs to remain compatible (ADR-022).
                result = await self._get_engine().remember(
                    content,
                    namespace_id,
                    title=title,
                    source=source,
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
        finally:
            collect_usage()  # idempotent — drains queue if not already collected
            clear_trace_id()

    async def remember_batch(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str | UUID,
        skill_name: str = "general_entities",
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
                - metadata: dict (optional)
                - external_id: str (optional) — caller-supplied external identifier
            namespace: Namespace UUID (as UUID or string)
            skill_name: Extraction skill to use
            max_concurrent: Maximum concurrent document processing
            deduplicate: Deduplicate entities across documents (default: True)
            infer_relationships: Infer relationships after ingestion (default: True)
            on_progress: Callback(processed_count, total_count) for progress updates
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
        from khora.telemetry.context import (
            clear_trace_id,
            collect_usage,
            ensure_trace_id,
            start_usage_collection,
        )

        ensure_trace_id()
        start_usage_collection()
        try:
            namespace_id = await self._resolve_namespace(namespace)
            with trace_span("khora.remember_batch", namespace_id=str(namespace_id), batch_size=len(documents)):
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
        finally:
            collect_usage()  # idempotent — drains queue if not already collected
            clear_trace_id()

    async def submit_batch(
        self,
        documents: list[dict[str, Any]],
        *,
        on_result: Callable[[int, int, DocumentResult], None],
        namespace: str | UUID,
        skill_name: str = "general_entities",
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None = None,
        extraction_config_hash: str | None = None,
        chunk_strategy: ChunkStrategy | None = None,
        max_chunks_in_flight: int | None = None,
        max_concurrent: int = 1,
        reprocess_archived: bool = False,
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
                'metadata', 'external_id' keys.
            on_result: Synchronous callback(completed, total, DocumentResult)
                invoked per document as processing completes.
            namespace: Namespace UUID (as UUID or string).
            skill_name: Extraction skill to use.
            entity_types: Required entity types to extract.
            relationship_types: Required relationship types to extract.
            expertise: Optional domain-specific extraction config.
            extraction_config_hash: Optional hash for extraction config change detection.
            chunk_strategy: Override chunking strategy for this batch.
            max_chunks_in_flight: Maximum chunks processed per window. Controls
                memory usage during background processing. None = unbounded.
            max_concurrent: Maximum documents to process concurrently in background
                (default: 1 — sequential processing).
            reprocess_archived: If True, ARCHIVED documents are reset to PENDING
                and re-processed like FAILED documents. If False (default), ARCHIVED
                documents are skipped with a warning — preserving intentional
                archival semantics.

        Returns:
            BatchHandle with batch_id, completion status, and wait() method.

        Raises:
            RuntimeError: If the engine does not support staged document processing.
        """
        from khora.core.models.document import Document, DocumentMetadata

        if not documents:
            handle = BatchHandle(batch_id=uuid4(), total=0)
            handle._mark_done()
            return handle

        from khora.core.models.document import DocumentStatus

        namespace_id = await self._resolve_namespace(namespace)
        storage = self.storage

        # Persist all documents as PENDING before returning the handle.
        # This satisfies the durability contract — if the process crashes after
        # submit_batch() returns, the PENDING records survive for recovery.
        #
        # ADR-068 Decision 1 — self-healing for existing documents:
        # Instead of failing on duplicate external_id, detect and dispatch:
        #   PENDING    → skip insert, re-queue for processing (self-heal stalled docs)
        #   COMPLETED  → skip entirely, report success (already done)
        #   FAILED     → reset to PENDING + update content, re-queue for processing
        #   ARCHIVED   → skip by default (preserves intentional archival); set
        #                reprocess_archived=True to re-activate explicitly (DYT-3077)
        #   PROCESSING → skip to avoid race with active worker (M1)
        pending_docs: list[Document] = []
        pending_doc_data: list[dict[str, Any]] = []
        pre_failed_docs: list[tuple[Document, str]] = []
        pre_completed_docs: list[Document] = []

        # Batch-lookup existing documents by external_id to avoid N serial queries.
        all_external_ids = [d.get("external_id") for d in documents if d.get("external_id")]
        existing_by_ext_id: dict[str, Document] = {}
        if all_external_ids:
            try:
                existing_by_ext_id = await storage.get_documents_by_external_ids(namespace_id, all_external_ids)
            except Exception as exc:
                # M2: Fall through to the normal insert path if the lookup fails.
                logger.warning(
                    f"submit_batch: could not look up existing documents by external_id "
                    f"({exc}); treating all as new inserts"
                )
                existing_by_ext_id = {}

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
                # (fixes empty-source issue observed in soak test — DYT-3075).
                prior_status = existing.status
                # H1: Track FAILED and ARCHIVED docs — they may have prior extraction
                # state (chunks, graph entities) that must be cleared before re-processing
                # to prevent duplicate chunks/entities on retry.
                if prior_status in (DocumentStatus.FAILED, DocumentStatus.ARCHIVED):
                    pre_failed_doc_ids.add(existing.id)
                existing.content = content
                existing.metadata = DocumentMetadata(
                    title=doc_data.get("title", ""),
                    source=doc_data.get("source", ""),
                    source_type="api",
                    checksum=checksum,
                    size_bytes=len(content.encode("utf-8")),
                    custom=doc_data.get("metadata") or {},
                )
                existing.status = DocumentStatus.PENDING
                existing.extraction_config_hash = extraction_config_hash
                existing.error_message = None
                logger.debug(
                    f"submit_batch: re-queuing existing {prior_status.value} document "
                    f"(external_id={external_id!r}, doc_id={existing.id})"
                )
                try:
                    await storage.update_document(existing)
                    pending_docs.append(existing)
                    pending_doc_data.append(doc_data)
                except Exception as exc:
                    logger.warning(
                        f"submit_batch: could not update document record (external_id={external_id!r}): {exc}"
                    )
                    pre_failed_docs.append((existing, str(exc)))
                continue

            # No existing document — normal insert path.
            doc = Document(
                namespace_id=namespace_id,
                content=content,
                metadata=DocumentMetadata(
                    title=doc_data.get("title", ""),
                    source=doc_data.get("source", ""),
                    source_type="api",
                    checksum=checksum,
                    size_bytes=len(content.encode("utf-8")),
                    custom=doc_data.get("metadata") or {},
                ),
                extraction_config_hash=extraction_config_hash,
                external_id=external_id,
            )
            try:
                doc = await storage.create_document(doc)
                pending_docs.append(doc)
                pending_doc_data.append(doc_data)
            except Exception as exc:
                logger.warning(
                    f"submit_batch: could not create document record "
                    f"(external_id={doc_data.get('external_id')!r}): {exc}"
                )
                pre_failed_docs.append((doc, str(exc)))

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
        task = asyncio.create_task(
            self._submit_batch_worker(
                handle,
                pending_docs,
                pending_doc_data,
                pre_failed_docs,
                pre_completed_docs,
                on_result,
                namespace_id=namespace_id,
                skill_name=skill_name,
                entity_types=entity_types,
                relationship_types=relationship_types,
                expertise=expertise,
                extraction_config_hash=extraction_config_hash,
                chunk_strategy=chunk_strategy,
                max_chunks_in_flight=max_chunks_in_flight,
                max_concurrent=max_concurrent,
                pre_failed_doc_ids=pre_failed_doc_ids,
                chunk_semaphore=self._chunk_semaphore,
            )
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return handle

    async def _submit_batch_worker(
        self,
        handle: BatchHandle,
        pending_docs: list[Document],
        doc_data_list: list[dict[str, Any]],
        pre_failed_docs: list[tuple[Document, str]],
        pre_completed_docs: list[Document],
        on_result: Callable[[int, int, DocumentResult], None],
        *,
        namespace_id: UUID,
        skill_name: str,
        entity_types: list[str],
        relationship_types: list[str],
        expertise: ExpertiseConfig | None,
        extraction_config_hash: str | None,
        chunk_strategy: ChunkStrategy | None,
        max_chunks_in_flight: int | None,
        max_concurrent: int,
        pre_failed_doc_ids: set[UUID],
        chunk_semaphore: _GlobalChunkSemaphore | None = None,
    ) -> None:
        """Background worker that processes pre-staged PENDING documents."""
        storage = self.storage

        def _fire_result(result: DocumentResult) -> None:
            handle._record_result(result)
            try:
                on_result(handle.completed, handle.total, result)
            except Exception as cb_exc:
                logger.warning(f"submit_batch: on_result callback raised: {cb_exc}")

        # Fire error results for documents that failed to be created (e.g. external_id collision).
        for doc, err in pre_failed_docs:
            _fire_result(
                DocumentResult(
                    document_id=doc.id,
                    namespace_id=namespace_id,
                    success=False,
                    error=err,
                    external_id=doc.external_id,
                )
            )

        # Fire skipped results for documents already COMPLETED (ADR-068 self-heal).
        for doc in pre_completed_docs:
            _fire_result(
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

        engine = self._get_engine()
        process_fn = getattr(engine, "process_staged_document", None)
        if process_fn is None:
            # Engine doesn't support staged processing — mark all PENDING docs as FAILED.
            err_msg = f"Engine {type(engine).__name__!r} does not support submit_batch (requires process_staged_document method)"
            for doc in pending_docs:
                doc.mark_failed(err_msg)
                try:
                    await storage.update_document(doc)
                except Exception as upd_exc:
                    logger.warning(f"submit_batch: could not update document status: {upd_exc}")
                _fire_result(
                    DocumentResult(
                        document_id=doc.id,
                        namespace_id=namespace_id,
                        success=False,
                        error=err_msg,
                        external_id=doc.external_id,
                    )
                )
            handle._mark_done()
            return

        sem = asyncio.Semaphore(max_concurrent)

        async def _process_one(doc: Document, doc_data: dict[str, Any]) -> None:
            from khora.telemetry.context import collect_usage, start_usage_collection

            async with sem:
                # H1: Clear partial extraction state for previously-FAILED documents
                # before re-processing to prevent duplicate chunks/entities on retry.
                if doc.id in pre_failed_doc_ids:
                    if storage.vector is not None:
                        try:
                            await storage.vector.delete_chunks_by_document(doc.id)
                        except Exception as exc:
                            logger.warning(f"submit_batch: could not clear chunks table for {doc.id}: {exc}")
                    clear_fn = getattr(engine, "clear_document_extraction_state", None)
                    if clear_fn is not None:
                        try:
                            await clear_fn(doc.id, namespace_id)
                        except Exception as exc:
                            logger.warning(f"submit_batch: could not clear extraction state for {doc.id}: {exc}")

                start_usage_collection()
                try:
                    doc_metadata = doc_data.get("metadata") or {}
                    occurred_at_raw = doc_metadata.get("occurred_at")
                    parse_dt = getattr(engine, "_parse_datetime", None)
                    if occurred_at_raw and parse_dt is not None:
                        occurred_at = parse_dt(occurred_at_raw)
                    else:
                        occurred_at = datetime.now(UTC)

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
                        chunk_semaphore=chunk_semaphore,
                    )
                    result = DocumentResult(
                        document_id=doc.id,
                        namespace_id=namespace_id,
                        success=True,
                        chunks_created=chunks,
                        entities_extracted=entities,
                        relationships_created=rels,
                        llm_usage=collect_usage(),
                        external_id=doc.external_id,
                    )
                except Exception as exc:
                    partial_usage = collect_usage()
                    logger.error(f"submit_batch: failed to process document {doc.id}: {exc}")
                    doc.mark_failed(str(exc))
                    try:
                        await storage.update_document(doc)
                    except Exception as upd_exc:
                        logger.warning(f"submit_batch: could not update document status: {upd_exc}")
                    result = DocumentResult(
                        document_id=doc.id,
                        namespace_id=namespace_id,
                        success=False,
                        error=str(exc),
                        llm_usage=partial_usage,
                        external_id=doc.external_id,
                    )
                _fire_result(result)

        try:
            await asyncio.gather(*[_process_one(doc, data) for doc, data in zip(pending_docs, doc_data_list)])
        finally:
            handle._mark_done()

    async def recall(
        self,
        query: str,
        *,
        namespace: str | UUID,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.0,
        agentic: bool = False,
        raw: bool = False,
        include_sources: bool = False,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> RecallResult:
        """Recall memories relevant to a query.

        This is the primary method for retrieving memories. It:
        1. Uses LLM to understand query (entities, temporal refs, etc.)
        2. Searches across vector, graph, and keyword indexes
        3. Fuses results using Reciprocal Rank Fusion
        4. Returns ranked results

        When agentic=True, uses multi-step exploration:
        1. Initial comprehensive search with query understanding
        2. Executes pre-computed follow-up queries for deeper exploration
        3. Explores under-represented sources
        4. Returns combined results with full trace

        When raw=True, skips all LLM features:
        - Query understanding
        - Entity linking
        - Reranking
        - HyDE expansion
        This is useful for benchmarks and simple searches.

        Args:
            query: Query text
            namespace: Namespace UUID (as UUID or string)
            limit: Maximum results to return
            mode: Search mode (VECTOR, GRAPH, HYBRID, ALL)
            min_similarity: Minimum similarity threshold
            agentic: If True, use multi-step agentic search (default: False)
            raw: If True, skip all LLM features (default: False)
            include_sources: If True, populate source document metadata on
                returned chunks and entities (default: False)
            start_time: Optional lower bound (inclusive) for memory time.
                Timezone-aware datetimes are recommended; naive datetimes are
                assumed UTC. When provided, bypasses NLP temporal detection.
            end_time: Optional upper bound (inclusive) for memory time.
                Same timezone semantics as start_time.

        Returns:
            RecallResult with matched memories.  When using the VectorCypher
            engine, ``relationships`` contains scored relationship tuples and
            ``context_text`` includes a ``--- Relationships ---`` section.

        Raises:
            ValueError: If both ``start_time`` and ``end_time`` are provided
                and ``start_time > end_time``.
        """
        from khora.telemetry.context import (
            clear_trace_id,
            collect_usage,
            ensure_trace_id,
            start_usage_collection,
        )

        ensure_trace_id()
        start_usage_collection()
        try:
            if start_time is not None or end_time is not None:
                if start_time is not None and end_time is not None:
                    try:
                        if start_time > end_time:
                            raise ValueError("start_time must be <= end_time")
                    except TypeError as e:
                        raise ValueError("start_time and end_time must both be timezone-aware or both naive") from e
                from khora.engines.skeleton.backends import TemporalFilter as SkeletonTemporalFilter

                temporal_filter: Any = SkeletonTemporalFilter(
                    occurred_after=start_time,
                    occurred_before=end_time,
                )
            else:
                temporal_filter = None
            namespace_id = await self._resolve_namespace(namespace)
            with trace_span("khora.recall", namespace_id=str(namespace_id), query=query):
                result = await self._get_engine().recall(
                    query,
                    namespace_id,
                    limit=limit,
                    mode=mode,
                    min_similarity=min_similarity,
                    agentic=agentic,
                    raw=raw,
                    temporal_filter=temporal_filter,
                )
                if include_sources:
                    await self._populate_sources(result.chunks, result.entities, result.relationships)
                return replace(result, llm_usage=collect_usage())
        finally:
            collect_usage()  # idempotent — drains queue if not already collected
            clear_trace_id()

    async def forget(
        self,
        document_id: UUID,
        *,
        namespace: str | UUID,
    ) -> bool:
        """Remove a memory from the lake.

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

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def get_entity(
        self,
        entity_id: UUID,
        *,
        include_sources: bool = False,
    ) -> Entity | None:
        """Get an entity by ID.

        Args:
            entity_id: Entity UUID to retrieve
            include_sources: If True, populate source document metadata on
                the returned entity (default: False)

        Returns:
            Entity if found, else None
        """
        entity = await self._get_engine().get_entity(entity_id)
        if entity is not None and include_sources:
            await self._populate_sources([], [entity], [])
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
            await self._populate_sources([], entities, [])
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
            await self._populate_sources([], results, [])
        return results

    # =========================================================================
    # Document Operations (Convenience Methods)
    # =========================================================================

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID.

        Args:
            document_id: Document UUID

        Returns:
            Document or None if not found
        """
        return await self._get_engine().get_document(document_id)

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
            await self._populate_sources([], entities, [])
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
            sources.update(await self.storage.get_document_sources_batch(batch))

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

            sub_id = lake.subscribe("entity.created", on_entity)
            await lake.remember("Acme Corp announced...", ...)
            lake.unsubscribe(sub_id)
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

    def _get_hook_dispatcher(self) -> Any:
        """Lazy-initialize the hook dispatcher."""
        if not hasattr(self, "_hook_dispatcher"):
            from khora.hooks.dispatcher import HookDispatcher

            hooks_config = getattr(self._config, "hooks", None)
            max_concurrent = 10
            if hooks_config:
                max_concurrent = getattr(hooks_config, "max_concurrent_callbacks", 10)
            self._hook_dispatcher = HookDispatcher(max_concurrent=max_concurrent)
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


# Convenience function for one-off usage
@asynccontextmanager
async def memory_lake(
    config: KhoraConfig | None = None,
) -> AsyncGenerator[MemoryLake]:
    """Context manager for one-off Memory Lake usage.

    Usage:
        async with memory_lake() as lake:
            await lake.remember("Hello, world!")
            result = await lake.recall("greeting")
    """
    lake = MemoryLake(config)
    try:
        await lake.connect()
        yield lake
    finally:
        await lake.disconnect()
