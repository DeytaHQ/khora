"""Storage coordinator that orchestrates all backends.

The coordinator provides a unified interface to all storage backends
(PostgreSQL, pgvector, Neo4j) and handles cross-cutting concerns like
transaction coordination and consistency.
"""

from __future__ import annotations

import asyncio
import functools
import time as _time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from khora.core.models import (
    Chunk,
    CommunityNode,
    Document,
    Entity,
    Episode,
    MemoryEvent,
    MemoryNamespace,
    Relationship,
)
from khora.core.models.document import DocumentSource
from khora.core.models.recall import DocumentProjection
from khora.exceptions import GraphMirrorFailedAfterPGCommitError
from khora.storage.backends.base import PaginatedResult
from khora.storage.replace_mirror import apply_replace_mirror_payload, build_replace_mirror_payload
from khora.telemetry import get_collector, metric_counter, trace_span

if TYPE_CHECKING:
    from khora.config.schema import KhoraConfig
    from khora.filter.ast import FilterNode
    from khora.storage.temporal import TemporalVectorStore

    from .backends.base import (
        EventStoreProtocol,
        GraphBackendProtocol,
        RelationalBackendProtocol,
        VectorBackendProtocol,
    )


def _replace_partial_failure_counter() -> Any:
    """Counter for replace graph-mirror divergence (#884) and failed
    reconcile re-attempts (#1430) - same counter for both, mirroring how the
    dream reconciler shares ``khora.dream.graph_mirror.partial_failure``
    between the original mirror failure and its drain (#1272)."""
    return metric_counter(
        "khora.storage.replace_document.partial_failure",
        unit="1",
        description=(
            "PG transaction committed (chunks + COMPLETED) but "
            "the post-commit graph-mirror phase of "
            "replace_document_extraction raised (or a #1430 reconcile "
            "re-attempt of the persisted plan failed). The graph is "
            "in a partial-mirror state; the plan is queued on "
            "documents.graph_mirror_pending for the reconciler."
        ),
    )


def _extract_namespace_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> UUID | None:
    """Best-effort extraction of ``namespace_id`` from a decorated method's call.

    Looks in (in order): ``kwargs["namespace_id"]``, any positional ``UUID``
    arg (after ``self``), the ``.namespace_id`` attribute of a positional
    model arg (e.g. ``Document``, ``Entity``, ``Chunk``, ``Relationship``),
    or the ``.namespace_id`` of the first element of a positional list.
    Returns ``None`` if no source is found.
    """
    ns = kwargs.get("namespace_id")
    if isinstance(ns, UUID):
        return ns

    # Skip the bound `self` (args[0]); inspect remaining positionals.
    for arg in args[1:]:
        if isinstance(arg, UUID):
            return arg
        attr = getattr(arg, "namespace_id", None)
        if isinstance(attr, UUID):
            return attr
        if isinstance(arg, list) and arg:
            head_attr = getattr(arg[0], "namespace_id", None)
            if isinstance(head_attr, UUID):
                return head_attr

    return None


def _record_storage_op(operation: str, backend: str = "postgresql"):
    """Decorator to record telemetry for async storage operations.

    Measures wall-clock time and records success/error via the global
    telemetry collector.  Best-effort extracts ``namespace_id`` from the
    decorated method's arguments via :func:`_extract_namespace_id` so
    ``storage_events.namespace_id`` is populated downstream.
    """

    span_name = f"khora.storage.{operation}"

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            t0 = _time.perf_counter()
            namespace_id = _extract_namespace_id(args, kwargs)
            try:
                with trace_span(span_name, backend=backend) as span:
                    result = await func(*args, **kwargs)
                    elapsed = _time.perf_counter() - t0
                    span.set_attribute("latency_ms", elapsed * 1000)
                    span.set_attribute("status", "success")
                get_collector().record_storage_op(
                    operation=operation,
                    backend=backend,
                    latency_ms=elapsed * 1000,
                    namespace_id=namespace_id,
                )
                return result
            except Exception:
                elapsed = _time.perf_counter() - t0
                get_collector().record_storage_op(
                    operation=operation,
                    backend=backend,
                    latency_ms=elapsed * 1000,
                    status="error",
                    namespace_id=namespace_id,
                )
                raise

        return wrapper

    return decorator


@dataclass
class TransactionContext:
    """Shared session for multi-backend atomic operations.

    Obtained via ``StorageCoordinator.transaction()``.  The coordinator
    commits on successful exit and rolls back on exception.
    """

    session: AsyncSession

    @asynccontextmanager
    async def savepoint(self) -> AsyncGenerator[TransactionContext]:
        """Create a savepoint (nested transaction).

        Usage::

            async with coordinator.transaction() as txn:
                # ... work ...
                async with txn.savepoint() as sp:
                    # rolled back independently on error
                    ...
        """
        async with self.session.begin_nested():
            yield self


@dataclass(frozen=True)
class ReplaceResult:
    """Outcome of ``StorageCoordinator.replace_document_extraction()``.

    Counts describe the write footprint of a successful document-replacement
    lifecycle — chunks hard-replaced in pgvector, new entities/relationships
    persisted to the graph, and orphaned/surviving graph state retired or
    remapped.

    ``degradations`` carries ADR-001 entries from the pre-replace reconciler
    drain (#1430): pending graph-mirror markers from PRIOR failed replaces in
    the same namespace that could not be replayed. Empty on the happy path.
    """

    document_id: UUID
    chunks_deleted: int
    chunks_created: int
    entities_created: int
    entities_updated: int
    entities_retired: int
    relationships_created: int
    relationships_retired: int
    degradations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StorageHealth:
    """Health status of all storage backends."""

    relational: bool = False
    vector: bool = False
    graph: bool = False
    event_store: bool = False

    @property
    def is_healthy(self) -> bool:
        """Check if all backends are healthy."""
        return self.relational and self.vector

    @property
    def summary(self) -> dict[str, bool]:
        """Get health summary as a dictionary."""
        return {
            "relational": self.relational,
            "vector": self.vector,
            "graph": self.graph,
            "event_store": self.event_store,
        }


@dataclass
class StorageCoordinator:
    """Coordinates operations across all storage backends.

    Provides a unified interface for storage operations and handles
    cross-cutting concerns like transaction management and consistency.
    """

    relational: RelationalBackendProtocol | None = None
    vector: VectorBackendProtocol | None = None
    graph: GraphBackendProtocol | None = None
    event_store: EventStoreProtocol | None = None

    _relational: RelationalBackendProtocol | None = field(default=None, init=False, repr=False)
    _vector: VectorBackendProtocol | None = field(default=None, init=False, repr=False)
    _graph: GraphBackendProtocol | None = field(default=None, init=False, repr=False)
    _event_store: EventStoreProtocol | None = field(default=None, init=False, repr=False)

    _connected: bool = field(default=False, init=False)
    _is_unified_backend: bool = field(default=False, init=False)
    _hook_dispatcher: Any = field(default=None, init=False)

    def should_dispatch_hooks(self) -> bool:
        """True when a dispatch would reach at least one subscriber.

        Mirrors the exact guard in :meth:`dispatch_hook`. Callers that build a
        batch of ``MemoryEvent`` objects can check this once and skip the whole
        construction+dispatch loop when it returns False - a no-subscriber
        dispatch is already a no-op, so this only avoids the wasted work.
        """
        return self._hook_dispatcher is not None and self._hook_dispatcher.subscription_count > 0

    async def dispatch_hook(self, event: Any) -> None:
        """Dispatch an event to hook subscribers if a dispatcher is attached.

        Called by the ingestion pipeline after extraction/storage operations.
        No-op if no dispatcher is set (i.e., no hooks subscribed).
        """
        if self.should_dispatch_hooks():
            await self._hook_dispatcher.dispatch(event)

    # Security: tuple of public attr names that proxy through __setattr__.
    # Listed at class level so __setattr__ doesn't pay dict-lookup per call.
    _PROXIED_ROLES: tuple[str, ...] = ("relational", "vector", "graph", "event_store")

    def __setattr__(self, name: str, value: Any) -> None:
        """Wrap public backend attrs in ``NamespaceRequiredProxy`` on assign.

        Routes assignment to ``relational`` / ``vector`` / ``graph`` /
        ``event_store`` (either via dataclass-generated ``__init__`` or
        post-construction) through the proxy so external callers always see
        a deprecation-warning + namespace-enforcing wrapper. Internal
        coordinator code uses ``self._{role}`` directly to bypass the proxy.
        """
        if name in StorageCoordinator._PROXIED_ROLES:
            from khora.storage._namespace_proxy import NamespaceRequiredProxy

            object.__setattr__(self, f"_{name}", value)
            proxy = NamespaceRequiredProxy(value, name) if value is not None else None
            object.__setattr__(self, name, proxy)
            return
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        # Public-attr assignment in dataclass-generated __init__ already routed
        # through __setattr__ above, so self._{role} is populated and self.{role}
        # holds the proxy. Nothing to wrap here.

        # Detect if graph and vector share a SurrealDB connection (unified backend).
        # Some adapters (e.g. SQLiteLance) expose ``_conn`` as a property that
        # raises when the underlying connection isn't open yet — treat any
        # error as "not unified" since the probe is advisory.
        if self._graph is not None and self._vector is not None:
            try:
                graph_conn = getattr(self._graph, "_conn", None)
                vector_conn = getattr(self._vector, "_conn", None)
            except Exception:
                graph_conn = None
                vector_conn = None
            if graph_conn is not None and graph_conn is vector_conn:
                self._is_unified_backend = True
                logger.info("Detected unified SurrealDB backend — entity dual-writes will be collapsed")

    def surrealdb_connection(self) -> Any | None:
        """Return the shared :class:`SurrealDBConnection` on a unified stack (#1280).

        A SurrealDB-unified coordinator wires graph + vector adapters that share
        one ``SurrealDBConnection`` via their ``_conn`` slot - there is NO SQL
        session, so ``transaction()`` raises. Callers (dream-apply) use this to
        route to the SurrealQL-native path instead of the absent SQL session.

        Returns the connection only when graph and vector share the SAME
        ``SurrealDBConnection`` instance; any other shape (PG/Neo4j,
        sqlite_lance, graph-less) returns ``None``.
        """
        graph = self._graph
        vector = self._vector
        if graph is None or vector is None:
            return None
        try:
            graph_conn = getattr(graph, "_conn", None)
            vector_conn = getattr(vector, "_conn", None)
        except Exception:  # pragma: no cover - advisory probe
            logger.warning(
                "surrealdb_connection probe failed; falling back to non-unified path",
                exc_info=True,
            )
            return None
        if graph_conn is None or graph_conn is not vector_conn:
            return None
        if type(graph_conn).__name__ != "SurrealDBConnection":
            return None
        return graph_conn

    async def connect(self) -> None:
        """Connect all configured backends in parallel.

        All backend connections are initiated concurrently using asyncio.gather
        for faster startup when multiple backends are configured.
        """
        if self._connected:
            return

        logger.info("Connecting storage backends...")

        # Build list of connection tasks to run in parallel
        tasks = []
        if self._relational:
            tasks.append(self._relational.connect())
        if self._vector:
            tasks.append(self._vector.connect())
        if self._graph:
            tasks.append(self._graph.connect())
        if self._event_store:
            tasks.append(self._event_store.connect())

        # Connect all backends concurrently
        if tasks:
            await asyncio.gather(*tasks)

        self._connected = True
        logger.info("All storage backends connected")

    async def disconnect(self) -> None:
        """Disconnect all backends in parallel.

        All backend disconnections are initiated concurrently for faster shutdown.
        """
        if not self._connected:
            return

        logger.info("Disconnecting storage backends...")

        # Build list of disconnection tasks to run in parallel
        tasks = []
        if self._event_store:
            tasks.append(self._event_store.disconnect())
        if self._graph:
            tasks.append(self._graph.disconnect())
        if self._vector:
            tasks.append(self._vector.disconnect())
        if self._relational:
            tasks.append(self._relational.disconnect())

        # Disconnect all backends concurrently
        if tasks:
            await asyncio.gather(*tasks)

        self._connected = False
        logger.info("All storage backends disconnected")

    async def health_check(self) -> StorageHealth:
        """Check health of all backends (parallel)."""
        health = StorageHealth()

        # Build list of health check coroutines to run in parallel
        checks: list[tuple[str, Any]] = []
        if self._relational:
            checks.append(("relational", self._relational.is_healthy()))
        if self._vector:
            checks.append(("vector", self._vector.is_healthy()))
        if self._graph:
            checks.append(("graph", self._graph.is_healthy()))
        if self._event_store:
            checks.append(("event_store", self._event_store.is_healthy()))

        if checks:
            results = await asyncio.gather(*[coro for _, coro in checks], return_exceptions=True)
            for (name, _), result in zip(checks, results):
                # Treat exceptions as unhealthy
                setattr(health, name, result is True)

        return health

    # =========================================================================
    # Transaction support
    # =========================================================================

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[TransactionContext]:
        """Create a transaction context for multi-step atomic operations.

        All SQL-backend writes performed with ``txn.session`` share a
        single database transaction.  The session is committed on
        successful exit and rolled back on exception.

        Usage::

            async with coordinator.transaction() as txn:
                await coordinator.create_document(doc, session=txn.session)
                await coordinator.create_chunks_batch(chunks, session=txn.session)
        """
        # Resolve a session factory from the first available SQL backend
        factory = None
        for backend in (self._relational, self._vector, self._event_store):
            sf = getattr(backend, "_session_factory", None)
            if sf is not None:
                factory = sf
                break

        if factory is None:
            raise RuntimeError("No SQL backend connected; cannot create transaction")

        session = factory()
        try:
            yield TransactionContext(session=session)
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def temporal_store(
        self,
        backend: str,
        config: KhoraConfig,
    ) -> TemporalVectorStore:
        """Build a connected ``TemporalVectorStore`` for ``backend``.

        Factory method (not an accessor): every call returns a fresh,
        already-``connect()``-ed store. The instance is intentionally NOT
        cached on the coordinator — callers own its lifecycle and must
        disconnect it.

        Backend selection is delegated to
        :func:`khora.storage.temporal.create_temporal_store` (imported
        lazily so optional backend dependencies stay lazy). The coordinator
        only gathers the per-backend shared resource so the store reuses the
        coordinator's existing connections instead of opening its own:

        - ``pgvector``: shares the coordinator's SQLAlchemy engine (vector
          adapter's, falling back to the relational adapter's) so the
          connection pool is not doubled.
        - ``surrealdb``: shares the coordinator's ``SurrealDBConnection``.
          Embedded ``surrealkv://`` allows only one open handle per
          directory, so a second connection would fail on first write.
        - ``sqlite_lance``: shares the vector adapter's
          ``EmbeddedStorageHandle`` (single aiosqlite + LanceDB pair).
        - ``weaviate`` / ``turbopuffer``: read their connection details
          from ``config.storage.weaviate`` / ``config.storage.turbopuffer``.
          The factory raises ``ValueError`` if the required config is
          missing.
        """
        from khora.storage.temporal import create_temporal_store

        shared_pg_engine = None
        surrealdb_connection = None
        sqlite_lance_handle = None

        if backend == "pgvector":
            if self._vector is not None:
                shared_pg_engine = getattr(self._vector, "_engine", None)
            if shared_pg_engine is None and self._relational is not None:
                shared_pg_engine = getattr(self._relational, "_engine", None)
        elif backend == "surrealdb":
            if self._relational is not None:
                surrealdb_connection = getattr(self._relational, "_conn", None)
        elif backend == "sqlite_lance":
            if self._vector is None:
                raise RuntimeError("sqlite_lance backend requires a vector adapter on the coordinator")
            sqlite_lance_handle = getattr(self._vector, "_handle", None)
            if sqlite_lance_handle is None:
                raise RuntimeError("sqlite_lance vector adapter is missing its EmbeddedStorageHandle")

        store = create_temporal_store(
            backend,
            config,
            surrealdb_config=config.storage.surrealdb if backend == "surrealdb" else None,
            surrealdb_connection=surrealdb_connection,
            engine=shared_pg_engine,
            sqlite_lance_handle=sqlite_lance_handle,
        )
        await store.connect()
        return store

    # =========================================================================
    # Tenancy operations (delegated to relational)
    # =========================================================================

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        """Resolve a stable namespace_id to the active version's row id.

        Args:
            namespace_id: The stable namespace identifier (shared across versions)

        Returns:
            The row-level id of the active version

        Raises:
            ValueError: If no active version exists for the given namespace_id
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.resolve_namespace(namespace_id)

    async def _resolve_read_namespace(self, namespace_id: UUID) -> UUID:
        """Resolve a stable namespace_id to the row id for a backend read.

        ``memory_namespaces`` carries two UUIDs: ``id`` (the row PK every
        child FK and the graph ``(:Entity {namespace_id})`` store reference)
        and ``namespace_id`` (the stable id ``create_namespace`` returns and
        ``MemoryNamespace`` documents as the external identifier). Public
        ``StorageCoordinator`` read methods accept either, so resolve to the
        row id before forwarding to a backend; otherwise a stable id matches
        zero rows and the read silently returns empty.

        Idempotent on row ids (``resolve_namespace`` matches on
        ``namespace_id`` OR ``id``). No-op when no relational backend is
        configured (graph- or vector-only stacks have no namespace table to
        resolve against), preserving the pre-resolution behavior there.
        """
        if not self._relational:
            return namespace_id
        return await self._relational.resolve_namespace(namespace_id)

    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Create a new memory namespace."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.create_namespace(namespace)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.get_namespace(namespace_id)

    async def list_namespaces(
        self, *, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> PaginatedResult[MemoryNamespace]:
        """List namespaces with pagination."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.list_namespaces(active_only=active_only, limit=limit, offset=offset)

    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Update a namespace."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.update_namespace(namespace)

    async def create_namespace_version(
        self,
        *,
        previous_version: MemoryNamespace | None = None,
    ) -> MemoryNamespace:
        """Create a new version of a namespace.

        Args:
            previous_version: The previous version to supersede (if any)

        Returns:
            New namespace version
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.create_namespace_version(previous_version=previous_version)

    async def deactivate_namespace(self, namespace_id: UUID) -> None:
        """Mark a namespace version as inactive.

        Args:
            namespace_id: ID of the namespace to deactivate
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        await self._relational.deactivate_namespace(namespace_id)

    # =========================================================================
    # Document operations (delegated to relational)
    # =========================================================================

    @_record_storage_op("create_document", "postgresql")
    async def create_document(self, document: Document) -> Document:
        """Create a new document."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.create_document(document)

    async def get_document(self, document_id: UUID, *, namespace_id: UUID) -> Document | None:
        """Get a document by ID, scoped to ``namespace_id``.

        Returns ``None`` if the document does not exist OR belongs to a
        different namespace. The ``namespace_id`` filter is applied at the
        backend's SQL layer to prevent cross-tenant document access by id
        (IDOR).
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.get_document(document_id, namespace_id=namespace_id)

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        status: str | None = None,
        updated_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents in a namespace."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.list_documents(
            namespace_id, status=status, updated_before=updated_before, limit=limit, offset=offset
        )

    @_record_storage_op("claim_orphaned_documents", "postgresql")
    async def claim_orphaned_documents(
        self,
        namespace_id: UUID,
        *,
        pending_before: datetime,
        processing_before: datetime,
        limit: int = 100,
    ) -> list[Document]:
        """Atomically claim stale orphaned documents for crash recovery."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.claim_orphaned_documents(
            namespace_id,
            pending_before=pending_before,
            processing_before=processing_before,
            limit=limit,
        )

    async def update_document(self, document: Document) -> Document:
        """Update a document."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.update_document(document)

    async def delete_document(self, document_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a document and its chunks, scoped to ``namespace_id`` (IDOR family)."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")

        # Delete chunks first
        if self._vector:
            await self._vector.delete_chunks_by_document(document_id, namespace_id=namespace_id)

        return await self._relational.delete_document(document_id, namespace_id=namespace_id)

    @_record_storage_op("replace_document_extraction", "coordinator")
    async def replace_document_extraction(
        self,
        *,
        namespace_id: UUID,
        old_document_id: UUID,
        new_document: Document,
        new_chunks: list[Chunk],
        new_entities: list[Entity],
        new_relationships: list[Relationship],
    ) -> ReplaceResult:
        """Replace the extraction footprint of a document.

        Orchestrates the document-replacement lifecycle:

        1. Inside a single pgvector transaction: update the document row,
           hard-delete old chunks, and insert the new pre-embedded chunks.
           Embedding is assumed to have happened before this call.
        2. After Postgres commits, run graph-side retirement / remap against
           the graph backend — best-effort, not part of the PG transaction.
           Orphaned entities are snapshotted into ``:EntityVersion``; orphaned
           relationships get ``valid_until`` stamped in place; survivors have
           their ``source_document_ids`` arrays swapped from old to new doc
           UUID. Net-new entities and relationships are upserted/created via
           the existing MERGE paths.
        3. The document is marked ``COMPLETED`` inside the PG transaction
           (chunk + entity counts; relationship_count left at 0 - see #884
           follow-up). If the PG tx fails, the rollback also reverts the
           status stamp. If the post-tx graph phase fails (#884), the
           document row remains ``COMPLETED`` (PG data is durable and
           consistent), ``khora.storage.replace_document.partial_failure``
           is incremented, and the original exception is wrapped in
           ``GraphMirrorFailedAfterPGCommitError`` so the caller can
           record the divergence on its user-facing result instead of
           presenting the failure as a full rollback. Additionally (#1430,
           modeled on the dream reconciler #1272) the computed graph plan
           is persisted to ``documents.graph_mirror_pending`` so the
           divergence is durable-recoverable: the reconciler drain that
           runs at the start of the next replace in the same namespace
           (or an explicit ``reconcile_replace_graph_mirror()`` call)
           replays it. The status deliberately stays ``COMPLETED`` (#887:
           PG data is durable and consistent; a FAILED stamp would
           contradict fully-written data and re-trigger self-heal paths) -
           the non-NULL ``graph_mirror_pending`` marker is the
           "flagged-completed" signal that the graph is known-diverged.

        Args:
            namespace_id: Namespace owning the document.
            old_document_id: The document being replaced (its chunks are
                deleted and its graph footprint is retired / remapped).
            new_document: The updated ``Document`` row. The same ``id`` may
                be reused across replacements; the row is updated in place.
            new_chunks: Pre-embedded chunks for the replacement content.
            new_entities: Entities extracted from the replacement content.
            new_relationships: Relationships extracted from the replacement
                content.

        Returns:
            A ``ReplaceResult`` with the write-footprint counts.
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        if not self._graph:
            raise RuntimeError("Graph backend not configured")

        # 0. Reconciler drain (#1430, same trigger shape as the dream
        #    reconciler #1272 which drains at the start of the next apply
        #    run): replay any pending graph-mirror markers left by prior
        #    failed replaces in this namespace BEFORE prefetching graph
        #    state, so the retire / survive sets below are computed against
        #    a converged graph. A still-failing marker stays queued and
        #    surfaces as a degradation on the ReplaceResult.
        drain_degradations = await self.reconcile_replace_graph_mirror(namespace_id)

        try:
            # 1. Prefetch old graph state and compute retire / survive sets
            #    BEFORE mutating anything.  Doing this up front keeps the
            #    Python-side filter aligned with the Cypher.
            fetch = getattr(self._graph, "fetch_document_extraction_state", None)
            if fetch is None:
                old_entity_records: list[dict[str, Any]] = []
                old_relationship_records: list[dict[str, Any]] = []
            else:
                old_entity_records, old_relationship_records = await fetch(old_document_id, namespace_id=namespace_id)

            # Relationship types in Neo4j are stored sanitized (upper-case,
            # non-alphanumerics → underscore) — see ``_sanitize_neo4j_label``
            # in ``storage/backends/neo4j.py``.  The prefetch returns the
            # stored (sanitized) label via ``type(rel)``, so we apply the same
            # sanitizer to ``Relationship.relationship_type`` before building
            # the comparison key.  Lazy import keeps this coordinator free of
            # Neo4j-specific imports at module load time.
            from khora.storage.backends.neo4j import _sanitize_neo4j_label

            new_entity_keys = {(e.name, e.entity_type) for e in new_entities}
            new_relationship_keys = {
                (r.source_entity_id, r.target_entity_id, _sanitize_neo4j_label(r.relationship_type))
                for r in new_relationships
            }
            old_doc_str = str(old_document_id)
            new_doc_str = str(new_document.id)
            retired_at_iso = datetime.now(UTC).isoformat()
            retired_at = datetime.now(UTC)

            entity_retirement_rows: list[dict[str, str]] = []
            entity_survivor_keys: set[tuple[str, str]] = set()
            entity_survivor_remap_rows: list[dict[str, str]] = []
            # Co-sourced entities that the NEW extraction no longer mentions:
            # another document still sources them (source_document_count > 1),
            # so they must survive — but the replaced document's id must be
            # stripped from their source_document_ids. Without this they keep a
            # dangling reference to a document that no longer extracts them.
            entity_survivor_strip_ids: list[UUID] = []
            for rec in old_entity_records:
                # Belt-and-suspenders: skip cross-namespace leaks even though
                # the prefetch Cypher is already namespace-scoped.
                if rec.get("namespace_id") != str(namespace_id):
                    continue
                name = rec["name"]
                entity_type = rec["entity_type"]
                key = (name, entity_type)
                if key in new_entity_keys:
                    entity_survivor_keys.add(key)
                    entity_survivor_remap_rows.append(
                        {
                            "entity_id": rec["id"],
                            "old_doc_id": old_doc_str,
                            "new_doc_id": new_doc_str,
                        }
                    )
                elif rec["source_document_count"] == 1:
                    entity_retirement_rows.append(
                        {
                            "current_id": rec["id"],
                            "snapshot_id": str(uuid4()),
                            "namespace_id": rec["namespace_id"],
                            "retired_at": retired_at_iso,
                        }
                    )
                else:
                    # Co-sourced (count > 1) and dropped from the new
                    # extraction: keep the node, strip the old document id.
                    entity_survivor_strip_ids.append(UUID(rec["id"]))

            # For relationships, identity is (src_entity, tgt_entity,
            # sanitized_type).  Both sides of the comparison are now
            # sanitized, so mixed-case / punctuated rel types classify
            # correctly as survivors instead of leaking into net-new + retire.
            relationship_retirement_rows: list[dict[str, Any]] = []
            relationship_survivor_remap_rows: list[dict[str, str]] = []
            # Mirror of entity_survivor_strip_ids for the edge side: co-sourced
            # relationships dropped from the new extraction survive but must
            # lose the replaced document id from their source arrays.
            relationship_survivor_strip_ids: list[UUID] = []
            for rec in old_relationship_records:
                rel_id = rec["id"]
                rel_key = (
                    UUID(rec["source_entity_id"]),
                    UUID(rec["target_entity_id"]),
                    rec["relationship_type"],
                )
                if rel_key in new_relationship_keys:
                    relationship_survivor_remap_rows.append(
                        {
                            "relationship_id": rel_id,
                            "old_doc_id": old_doc_str,
                            "new_doc_id": new_doc_str,
                        }
                    )
                elif rec["source_document_count"] == 1:
                    relationship_retirement_rows.append(
                        {
                            "relationship_id": UUID(rel_id),
                            "old_doc_id": old_document_id,
                            "retired_at": retired_at,
                        }
                    )
                else:
                    # Co-sourced (count > 1) and dropped from the new
                    # extraction: keep the edge, strip the old document id.
                    relationship_survivor_strip_ids.append(UUID(rel_id))

            # Net-new sets are pure Python - computed here (before any
            # mutation) so the #1430 pending-mirror payload can capture them
            # even when the very first graph verb fails. The extracted
            # entities/relationships are not durably stored anywhere else
            # once the graph phase fails.
            net_new_entities = [e for e in new_entities if (e.name, e.entity_type) not in entity_survivor_keys]
            old_relationship_keys = {
                (
                    UUID(rec["source_entity_id"]),
                    UUID(rec["target_entity_id"]),
                    rec["relationship_type"],
                )
                for rec in old_relationship_records
            }
            net_new_relationships = [
                r
                for r in new_relationships
                if (r.source_entity_id, r.target_entity_id, _sanitize_neo4j_label(r.relationship_type))
                not in old_relationship_keys
            ]

            # 2. Postgres transaction: atomic chunk hard-replace.
            #    Embedding (OpenAI roundtrip) happened before this block - the
            #    transaction deliberately wraps only DB work (ADR §Performance).
            #
            #    Status stamp is split (fix for #887, ADR-002 Layer 1):
            #      - In-tx: mark COMPLETED with chunk + entity counts so the
            #        document row's status moves atomically with the
            #        fully-written data. If PG rolls back, the row keeps its
            #        pre-replace status.
            #      - Relationship count is left at 0 here; the graph phase
            #        runs after PG commits and is not part of the SQL tx, so
            #        we do not undercount silently. A follow-up PR (#884)
            #        will introduce a partial-state status for the
            #        graph-pending case.
            #    The shared SQLAlchemy session only covers backends that run
            #    on its engine. The sqlite_lance vector adapter writes through
            #    a raw aiosqlite handle on a separate engine - passing the
            #    session there crashed ``create_chunks_batch`` (#1134) and
            #    left the chunk DELETE pending on the shared handle for a
            #    later unrelated commit to apply outside any controlled
            #    transaction (#1135). The probe mirrors ``transaction()``'s
            #    backend discovery: only session-capable SQL backends expose
            #    ``_session_factory``.
            vector_in_txn = getattr(self._vector, "_session_factory", None) is not None
            async with self.transaction() as txn:
                if vector_in_txn:
                    chunks_deleted = await self._vector.delete_chunks_by_document(
                        old_document_id, namespace_id=namespace_id, session=txn.session
                    )
                    await self._vector.create_chunks_batch(new_chunks, session=txn.session)  # type: ignore[unresolved-attribute]
                else:
                    # Embedded path: the adapter commits its own SQLite work
                    # and compensates LanceDB itself. Cross-store atomicity is
                    # partial on embedded (documented) - the delete + insert
                    # are durable even if the document update below rolls back.
                    chunks_deleted = await self._vector.delete_chunks_by_document(
                        old_document_id, namespace_id=namespace_id
                    )
                    await self._vector.create_chunks_batch(new_chunks)
                new_document.mark_completed(len(new_chunks), len(new_entities), 0)
                await self._relational.update_document(new_document, session=txn.session)  # type: ignore[unresolved-attribute]
                # Clear any stale #1430 pending-mirror marker atomically with
                # the new content. A marker left by a PRIOR failed replace of
                # this document describes a superseded extraction - replaying
                # it after this replace would inject stale entities. The
                # update_document above deliberately does not touch the
                # column (it is not in its column list), so clear explicitly.
                new_document.graph_mirror_pending = None
                partial_update = getattr(self._relational, "partial_update_document", None)
                if partial_update is not None:
                    await partial_update(
                        new_document.id,
                        namespace_id=namespace_id,
                        session=txn.session,
                        graph_mirror_pending=None,
                    )

            # 3. Graph-side retirement / remap (after PG commits).  Order:
            #    retire -> remap -> upsert.  Retirement snapshots the current
            #    source_document_ids before we change them; remap cleanly
            #    swaps old->new on survivors before upsert would append
            #    new_doc_id a second time.  Net-new entities/relationships
            #    are those with keys absent from the old extraction.
            #
            #    Fix for #884: this phase runs OUTSIDE the PG transaction
            #    that just committed at line 670. Any exception here leaves
            #    PG durable (chunks + COMPLETED stamp) but the graph in a
            #    partial-mirror state. We catch, increment a partial-failure
            #    counter so operators can monitor the rate, and re-raise
            #    wrapped in GraphMirrorFailedAfterPGCommitError so the
            #    caller (engine layer) can record the divergence on the
            #    user-facing RememberResult instead of presenting the
            #    failure as if PG also rolled back.
            try:
                entities_retired = 0
                if entity_retirement_rows:
                    entities_retired = await self._graph.retire_orphaned_entities_batch(  # type: ignore[unresolved-attribute]
                        entity_retirement_rows
                    )

                relationships_retired = 0
                if relationship_retirement_rows:
                    relationships_retired = await self._graph.retire_orphaned_relationships_batch(  # type: ignore[unresolved-attribute]
                        relationship_retirement_rows, namespace_id=namespace_id
                    )

                if entity_survivor_remap_rows or relationship_survivor_remap_rows:
                    await self._graph.remap_source_document_ids_batch(  # type: ignore[unresolved-attribute]
                        entity_survivors=entity_survivor_remap_rows,
                        relationship_survivors=relationship_survivor_remap_rows,
                        namespace_id=namespace_id,
                    )

                # Strip the replaced document id from co-sourced survivors the
                # new extraction no longer mentions (entities/relationships with
                # source_document_count > 1). Unlike remap (swap old->new), these
                # keys are absent from the new extraction, so there is no new
                # doc id to swap in — the old id is simply removed.
                if entity_survivor_strip_ids:
                    await self._graph.remove_document_from_entity_sources_batch(  # type: ignore[unresolved-attribute]
                        entity_survivor_strip_ids, old_document_id, namespace_id
                    )
                if relationship_survivor_strip_ids:
                    await self._graph.remove_document_from_relationship_sources_batch(  # type: ignore[unresolved-attribute]
                        relationship_survivor_strip_ids, old_document_id, namespace_id
                    )

                entities_created = 0
                entities_updated = len(entity_survivor_remap_rows)
                if net_new_entities:
                    upsert_results = await self.upsert_entities_batch(namespace_id, net_new_entities)
                    entities_created = sum(1 for _, is_new in upsert_results if is_new)
                    entities_updated += sum(1 for _, is_new in upsert_results if not is_new)

                relationships_created = 0
                if net_new_relationships:
                    relationships_created = len(await self.create_relationships_batch(net_new_relationships))
                # Survivor relationships are accounted for implicitly via remap;
                # their id is preserved from the old graph state.
            except Exception as graph_exc:
                # PG already committed (chunks + COMPLETED stamp are durable).
                # Increment the partial-failure counter so operators can
                # monitor graph-mirror dropouts, then re-raise wrapped so
                # the caller can distinguish "graph mirror partial" from
                # a full-rollback failure. No namespace_id label - cardinality
                # rule (see CLAUDE.md).
                _replace_partial_failure_counter().add(1)
                # #1430: persist the computed graph plan as a durable
                # pending-mirror marker so the reconciler can replay it.
                # Best-effort - a failed marker write degrades back to the
                # #884 behavior (observable via degradation, healed by the
                # next successful replace) instead of masking graph_exc.
                pending_persisted = False
                try:
                    payload = build_replace_mirror_payload(
                        old_document_id=old_document_id,
                        entity_retirement_rows=entity_retirement_rows,
                        relationship_retirement_rows=relationship_retirement_rows,
                        entity_survivor_remap_rows=entity_survivor_remap_rows,
                        relationship_survivor_remap_rows=relationship_survivor_remap_rows,
                        entity_survivor_strip_ids=entity_survivor_strip_ids,
                        relationship_survivor_strip_ids=relationship_survivor_strip_ids,
                        net_new_entities=net_new_entities,
                        net_new_relationships=net_new_relationships,
                        exception=graph_exc,
                    )
                    partial_update = getattr(self._relational, "partial_update_document", None)
                    if partial_update is not None:
                        rows = await partial_update(
                            new_document.id,
                            namespace_id=namespace_id,
                            graph_mirror_pending=payload,
                        )
                        pending_persisted = rows > 0
                        new_document.graph_mirror_pending = payload if pending_persisted else None
                except Exception:
                    logger.warning(
                        "replace_document_extraction: failed to persist the "
                        "graph_mirror_pending marker for document {} in "
                        "namespace {}; divergence is observable via the #884 "
                        "degradation but not durably queued for reconcile "
                        "(#1430)",
                        new_document.id,
                        namespace_id,
                        exc_info=True,
                    )
                raise GraphMirrorFailedAfterPGCommitError(
                    document_id=new_document.id,
                    namespace_id=namespace_id,
                    original=graph_exc,
                    pending_persisted=pending_persisted,
                    # The failure path returns no ReplaceResult, so the
                    # exception is the only channel for the drain's own
                    # degradations (prior documents' markers that could not
                    # be replayed in this call).
                    drain_degradations=drain_degradations,
                ) from graph_exc

            return ReplaceResult(
                document_id=new_document.id,
                chunks_deleted=chunks_deleted,
                chunks_created=len(new_chunks),
                entities_created=entities_created,
                entities_updated=entities_updated,
                entities_retired=entities_retired,
                relationships_created=relationships_created,
                relationships_retired=relationships_retired,
                degradations=drain_degradations,
            )

        except Exception:
            # Fix for #887 (same family as #884): do NOT stamp the document
            # row FAILED here. There are two distinct failure modes and the
            # old code mishandled both:
            #
            #   - PG transaction failed -> the `async with self.transaction()`
            #     block already rolled back the document row to its
            #     pre-replace state. Writing FAILED post-tx would corrupt
            #     status of a row whose data is otherwise consistent.
            #
            #   - Graph step failed AFTER PG committed -> chunks, entities
            #     and the COMPLETED status stamp are all durable in PG.
            #     Marking the doc FAILED would diverge status from the
            #     fully-written data (#887 reproducer). The graph-fail-
            #     after-pg-commit case is now wrapped in
            #     GraphMirrorFailedAfterPGCommitError inside the inner
            #     try/except above so the caller can attach a degradation
            #     to its user-facing result (#884).
            #
            # The exception is re-raised unwrapped (or wrapped, when it
            # comes from the inner graph-phase try/except) so callers see
            # the original error or a typed signal.
            logger.warning(
                "replace_document_extraction failed for document {} in namespace {}; "
                "re-raising without restamping document status (#887)",
                new_document.id,
                namespace_id,
            )
            raise

    async def reconcile_replace_graph_mirror(
        self,
        namespace_id: UUID,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Replay pending replace graph-mirror markers for a namespace (#1430).

        The replace-path equivalent of the dream reconciler's
        ``_drain_graph_mirror_pending`` (#1272): each non-NULL
        ``documents.graph_mirror_pending`` payload is a committed-but-
        unmirrored replace whose graph plan is replayed here (idempotent -
        see ``khora.storage.replace_mirror``) and cleared on success. A
        still-failing marker stays queued, increments
        ``khora.storage.replace_document.partial_failure``, and surfaces an
        ADR-001 degradation entry. Runs automatically at the start of every
        ``replace_document_extraction`` call; safe to invoke directly for
        operator-driven repair.

        Returns the degradation entries for markers that could not be
        replayed (including a failed pending read). Never raises.
        """
        relational = self._relational
        list_pending = getattr(relational, "list_documents_with_graph_mirror_pending", None)
        partial_update = getattr(relational, "partial_update_document", None)
        if relational is None or self._graph is None or list_pending is None or partial_update is None:
            # Markers are only ever written on graph-backed stacks whose
            # relational backend supports them (PostgreSQL); nothing to drain
            # elsewhere.
            return []

        try:
            pending_docs = await list_pending(namespace_id, limit=limit)
        except Exception as exc:
            # A failed pending read hides committed mirror lag - record it
            # rather than silently returning empty (ADR-001, same shape as
            # the dream drain's read guard).
            _replace_partial_failure_counter().add(1)
            logger.warning(
                "replace mirror reconcile: pending read failed for namespace {} (mirror lag may be hidden): {}",
                namespace_id,
                exc,
                exc_info=True,
            )
            return [
                {
                    "component": "coordinator.replace_mirror.reconcile",
                    "reason": "graph_mirror_pending_read_failed",
                    "detail": "list_documents_with_graph_mirror_pending raised",
                    "exception": type(exc).__name__,
                }
            ]
        if not pending_docs:
            return []

        degradations: list[dict[str, Any]] = []
        for doc in pending_docs:
            payload = doc.graph_mirror_pending
            if not payload:
                continue
            try:
                counts = await apply_replace_mirror_payload(self, payload, namespace_id=namespace_id)
            except Exception as exc:
                _replace_partial_failure_counter().add(1)
                logger.warning(
                    "replace mirror reconcile failed for document {} in namespace {} (still queued): {}",
                    doc.id,
                    namespace_id,
                    exc,
                    exc_info=True,
                )
                degradations.append(
                    {
                        "component": "coordinator.replace_mirror.reconcile",
                        "reason": "graph_mirror_reconcile_failed",
                        "detail": f"document_id={doc.id}",
                        "exception": type(exc).__name__,
                        "issue": "1430",
                    }
                )
                continue
            try:
                await partial_update(doc.id, namespace_id=namespace_id, graph_mirror_pending=None)
            except Exception as exc:
                # Replay succeeded but the clear failed: the marker stays
                # queued and the (idempotent) plan is replayed again on the
                # next drain. Record it so the lag is observable (ADR-001).
                _replace_partial_failure_counter().add(1)
                logger.warning(
                    "replace mirror reconciled for document {} in namespace {} but "
                    "clearing the marker failed (will replay again): {}",
                    doc.id,
                    namespace_id,
                    exc,
                    exc_info=True,
                )
                degradations.append(
                    {
                        "component": "coordinator.replace_mirror.reconcile",
                        "reason": "graph_mirror_pending_clear_failed",
                        "detail": f"document_id={doc.id}",
                        "exception": type(exc).__name__,
                        "issue": "1430",
                    }
                )
                continue
            logger.info(
                "replace mirror reconciled for document {} in namespace {}: {}",
                doc.id,
                namespace_id,
                counts,
            )
        return degradations

    async def count_documents(self, namespace_id: UUID) -> int:
        """Count documents in a namespace.

        Args:
            namespace_id: Namespace UUID

        Returns:
            Total number of documents in the namespace (0 if empty)

        Raises:
            RuntimeError: If relational backend is not configured
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.count_documents(namespace_id)

    async def get_last_activity_at(self, namespace_id: UUID) -> datetime | None:
        """Get the most recent document creation timestamp in a namespace.

        Args:
            namespace_id: Namespace UUID

        Returns:
            Timestamp of the most recently created document in the namespace.
            None if the namespace has no documents.

        Raises:
            RuntimeError: If relational backend is not configured
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.get_last_activity_at(namespace_id)

    async def get_document_stats(self, namespace_id: UUID) -> tuple[int, datetime | None]:
        """Get document count and last activity timestamp in a single query."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.get_document_stats(namespace_id)

    async def get_document_by_checksum(
        self, namespace_id: UUID, checksum: str, *, pending_stale_before: datetime | None = None
    ) -> Document | None:
        """Get a document by its content checksum."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.get_document_by_checksum(
            namespace_id, checksum, pending_stale_before=pending_stale_before
        )

    async def get_document_by_external_id(
        self,
        external_id: str | None,
        *,
        namespace_id: UUID,
    ) -> Document | None:
        """Get a document by (namespace_id, external_id).

        Unlike ``get_document_by_checksum``, this lookup returns documents in
        any status (including ``FAILED`` and ``PROCESSING``) so callers can
        self-heal on the next successful replace.
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.get_document_by_external_id(external_id, namespace_id=namespace_id)

    async def get_documents_by_external_ids(
        self,
        external_ids: list[str],
        *,
        namespace_id: UUID,
    ) -> dict[str, Document]:
        """Batch variant of :meth:`get_document_by_external_id`.

        Collapses N serial lookups into one query for ``remember_batch`` replace
        dispatch. Status-agnostic like the single lookup.
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.get_documents_by_external_ids(external_ids, namespace_id=namespace_id)

    async def get_documents_by_checksums(
        self, namespace_id: UUID, checksums: list[str], *, pending_stale_before: datetime | None = None
    ) -> dict[str, Document]:
        """Fetch documents by content checksums in a single query.

        Used for batch deduplication to avoid N serial DB queries.

        Args:
            namespace_id: Namespace to search in
            checksums: List of content checksums to look up
            pending_stale_before: Cutoff for reclaiming stale PENDING half-ingests (#1464)

        Returns:
            Dictionary mapping checksum to Document (only for existing documents)
        """
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.get_documents_by_checksums(  # type: ignore[unresolved-attribute]
            namespace_id, checksums, pending_stale_before=pending_stale_before
        )

    # =========================================================================
    # Chunk operations (delegated to vector)
    # =========================================================================

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        """Create a new chunk with embedding."""
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        return await self._vector.create_chunk(chunk)

    @_record_storage_op("create_chunks_batch", "pgvector")
    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """Create multiple chunks in a batch."""
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        return await self._vector.create_chunks_batch(chunks)

    async def get_chunk(self, chunk_id: UUID, *, namespace_id: UUID) -> Chunk | None:
        """Get a chunk by ID, filtered to the caller's ``namespace_id``.

        Returns ``None`` when the chunk does not exist or belongs to a
        different namespace. The ``namespace_id`` keyword is required to
        prevent cross-tenant chunk access via id (IDOR).
        """
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        return await self._vector.get_chunk(chunk_id, namespace_id=namespace_id)

    async def get_chunks_by_document(self, document_id: UUID, *, namespace_id: UUID) -> list[Chunk]:
        """Get all chunks for a document, filtered to the caller's ``namespace_id``.

        Returns an empty list when the document does not belong to the
        caller's namespace.
        """
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        return await self._vector.get_chunks_by_document(document_id, namespace_id=namespace_id)

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        """Fetch multiple chunks by ID in a single query, filtered to ``namespace_id``.

        Args:
            chunk_ids: List of chunk IDs to fetch.
            namespace_id: Caller's namespace; chunks belonging to any
                other namespace are silently dropped from the result.

        Returns:
            Dictionary mapping chunk ID to Chunk (only for existing
            chunks within ``namespace_id``).
        """
        if not chunk_ids:
            return {}
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        return await self._vector.get_chunks_batch(chunk_ids, namespace_id=namespace_id)

    async def upsert_keyword_chunk_edges(
        self,
        namespace_id: UUID,
        edges: list[tuple[str, UUID, float]],
    ) -> int:
        """Persist keyword -> chunk edges for the keyword_ppr lexical channel (#1391).

        No-op (returns 0) when the vector backend lacks the method (e.g. a
        SurrealDB-unified stack) so the gated ingest step degrades rather than
        crashes. Resolves ``namespace_id`` to the row id so the stored
        ``keyword_chunks.namespace_id`` matches the chunks table's FK value.
        """
        if not edges:
            return 0
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._vector and hasattr(self._vector, "upsert_keyword_chunk_edges"):
            return await self._vector.upsert_keyword_chunk_edges(namespace_id, edges)  # type: ignore[unresolved-attribute]
        return 0

    async def get_keyword_chunk_edges(
        self,
        namespace_id: UUID,
        *,
        limit: int,
    ) -> list[tuple[str, UUID, float]]:
        """Load a namespace's keyword -> chunk edges, capped at ``limit`` (#1391).

        Returns ``[]`` when the vector backend lacks the method so the
        keyword_ppr query channel degrades to an empty lexical slot.
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._vector and hasattr(self._vector, "get_keyword_chunk_edges"):
            return await self._vector.get_keyword_chunk_edges(namespace_id, limit=limit)  # type: ignore[unresolved-attribute]
        return []

    @_record_storage_op("search_similar_chunks", "pgvector")
    async def search_similar_chunks(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        filter_document_ids: list[UUID] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks."""
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        return await self._vector.search_similar(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
            filter_document_ids=filter_document_ids,
            created_after=created_after,
            created_before=created_before,
        )

    @_record_storage_op("search_fulltext_chunks", "pgvector")
    async def search_fulltext_chunks(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        language: str = "english",
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        filter_ast: FilterNode | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search chunks using PostgreSQL full-text search.

        ``filter_ast`` is the canonical recall-filter AST. It is forwarded to
        the relational vector backend's ``search_fulltext`` — which queries the
        legacy ``chunks`` table (no denormalized filter columns) and therefore
        REFUSES to return rows under an active filter rather than risk
        smuggling unfiltered chunks. The filtered BM25 path is the
        ``khora_chunks`` temporal store (``TemporalVectorStore.search_fulltext``).
        """
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        return await self._vector.search_fulltext(
            namespace_id,
            query_text,
            limit=limit,
            language=language,
            created_after=created_after,
            created_before=created_before,
            filter_ast=filter_ast,
        )

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Count chunks in a namespace.

        Chunks live in the vector backend on every topology (sqlite_lance
        writes chunk metadata to the SQLite ``chunks`` table too). A backend
        missing the method raises ``NotImplementedError`` so callers can
        record the gap instead of seeing a confusing ``AttributeError``.
        """
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        impl = getattr(self._vector, "count_chunks", None)
        if impl is None:
            raise NotImplementedError("Vector backend has no count_chunks")
        return await impl(namespace_id)

    async def update_last_accessed(
        self,
        namespace_id: UUID,
        chunk_ids: list[UUID],
        ts: datetime,
    ) -> int:
        """Stamp ``last_accessed_at = ts`` on chunks scoped to a namespace (#855).

        Delegates to the vector backend's single-UPDATE implementation.
        Returns the number of rows affected. Backends that don't implement
        the method (older third-party adapters) silently return 0 so the
        reinforcement path never crashes recall.
        """
        if not chunk_ids:
            return 0
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        impl = getattr(self._vector, "update_last_accessed", None)
        if impl is None:
            return 0
        return await impl(namespace_id, chunk_ids, ts)

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Chunk]:
        """List chunks in a namespace.

        Args:
            namespace_id: Namespace ID
            limit: Maximum chunks to return
            offset: Offset for pagination

        Returns:
            List of chunks
        """
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        return await self._vector.list_chunks(namespace_id, limit=limit, offset=offset)

    async def count_entities(self, namespace_id: UUID) -> int:
        """Count entities in a namespace. Best-effort during active ingestion (non-atomic dual-write).

        Entities are owned by the graph backend when present (the vector
        backend holds a denormalized mirror). Counting from the owner avoids
        topologies where the vector adapter lacks ``count_entities`` (e.g.
        sqlite_lance / SurrealDB), which previously raised ``AttributeError``.
        Falls back to the vector backend only when no graph backend exists
        (PostgreSQL-only chronicle stacks).
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        graph_impl = getattr(self._graph, "count_entities", None) if self._graph else None
        if graph_impl is not None:
            return await graph_impl(namespace_id)
        vector_impl = getattr(self._vector, "count_entities", None) if self._vector else None
        if vector_impl is not None:
            return await vector_impl(namespace_id)
        return 0

    async def count_relationships(self, namespace_id: UUID) -> int:
        """Count relationships in a namespace.

        Relationships are owned by the graph backend when present. Falls back
        to the vector backend's ``relationships`` mirror when no graph backend
        exists (PostgreSQL-only chronicle stacks, #1066) so the count matches
        what ``create_relationships_batch`` persisted there.
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            return await self._graph.count_relationships(namespace_id)
        vector_impl = getattr(self._vector, "count_relationships", None) if self._vector else None
        if vector_impl is not None:
            return await vector_impl(namespace_id)
        return 0

    # =========================================================================
    # Entity operations (cross-backend)
    # =========================================================================

    @_record_storage_op("create_entity", "graph+vector")
    async def create_entity(self, entity: Entity) -> Entity:
        """Create an entity in both graph and vector stores.

        When a unified backend is detected (graph and vector share the same
        connection, e.g. SurrealDB), the vector write is skipped to avoid
        duplicate records in the same database.

        Write ordering (issue #1138, mirroring #868 for the batch path):
        vector commits first, then graph. The prior implementation raced both
        writes via ``asyncio.gather`` with no partial-failure handling, so a
        graph failure after the vector committed (or vice versa) left the
        stores silently diverged. Sequencing vector-first inverts the failure
        asymmetry - a graph failure after a successful vector write leaves a
        vector row reconciled by the next upsert MERGE - and the
        ``khora.storage.create_entity.partial_failure`` counter increments so
        operators can monitor the rate.
        """
        if self._graph and self._vector:
            if self._is_unified_backend:
                # Single DB — graph adapter write is sufficient
                entity = await self._graph.create_entity(entity)
            else:
                await self._vector.create_entity(entity)
                try:
                    entity = await self._graph.create_entity(entity)
                except Exception:
                    metric_counter(
                        "khora.storage.create_entity.partial_failure",
                        unit="1",
                        description=(
                            "Vector commit succeeded but graph create_entity "
                            "raised; the vector row is reconciled by the next "
                            "upsert MERGE."
                        ),
                    ).add(1)
                    raise
        elif self._graph:
            entity = await self._graph.create_entity(entity)
        elif self._vector:
            await self._vector.create_entity(entity)
        return entity

    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id``.

        Returns ``None`` when the entity belongs to a different namespace.
        ``namespace_id`` is required to prevent cross-tenant IDOR — the graph
        backend's underlying ``get_entity`` only filters by ID.
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            # Backend now filters at the Cypher/SQL layer (IDOR family). Keep the
            # post-fetch check as defense-in-depth in case a backend's filter
            # ever regresses.
            entity = await self._graph.get_entity(entity_id, namespace_id=namespace_id)
            if entity is None or entity.namespace_id != namespace_id:
                return None
            return entity
        return None

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        """Get an entity by name and type."""
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            return await self._graph.get_entity_by_name(namespace_id, name, entity_type)
        return None

    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> Entity:
        """Update an entity in both graph and vector stores.

        Scoped to ``namespace_id`` (IDOR family). When a unified backend is
        detected, the vector write is skipped.

        Write ordering (issue #1138, mirroring #868 / ``create_entity``):
        vector commits first, then graph, with the
        ``khora.storage.update_entity.partial_failure`` counter incremented
        when the graph write raises after the vector committed. Replaces the
        prior ``asyncio.gather`` that masked half-applied writes.
        """
        if self._graph and self._vector:
            if self._is_unified_backend:
                return await self._graph.update_entity(entity, namespace_id=namespace_id)
            await self._vector.update_entity(entity, namespace_id=namespace_id)
            try:
                return await self._graph.update_entity(entity, namespace_id=namespace_id)
            except Exception:
                metric_counter(
                    "khora.storage.update_entity.partial_failure",
                    unit="1",
                    description=(
                        "Vector commit succeeded but graph update_entity "
                        "raised; the vector row is reconciled by the next "
                        "upsert MERGE."
                    ),
                ).add(1)
                raise
        if self._graph:
            return await self._graph.update_entity(entity, namespace_id=namespace_id)
        if self._vector:
            await self._vector.update_entity(entity, namespace_id=namespace_id)
        return entity

    async def delete_entity(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete an entity from both graph and vector stores, scoped to ``namespace_id`` (IDOR family).

        On a split stack (e.g. pg + neo4j) entities are dual-written to both the
        graph node and the pgvector ``entities`` row (#928). Deleting only the
        graph node left the vector row as an orphan that kept surfacing via
        similarity search. Mirror the forget-cascade pattern: delete the graph
        node, then the vector mirror. The vector delete is skipped on a unified
        backend (SurrealDB), where the graph adapter owns the single row.
        """
        deleted = False
        if self._graph:
            deleted = await self._graph.delete_entity(entity_id, namespace_id=namespace_id)
        if self._vector is not None and not self._is_unified_backend and hasattr(self._vector, "delete_entities_batch"):
            count = await self._vector.delete_entities_batch([entity_id], namespace_id=namespace_id)
            deleted = deleted or count > 0
        return deleted

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        source_chunk_ids: list[UUID] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities in a namespace.

        Prefers the graph backend when configured; otherwise falls back to
        the vector backend, which owns the entities table on PostgreSQL-only
        stacks (e.g. chronicle engine without Neo4j).
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            return await self._graph.list_entities(
                namespace_id,
                entity_type=entity_type,
                source_chunk_ids=source_chunk_ids,
                limit=limit,
                offset=offset,
            )
        if self._vector and hasattr(self._vector, "list_entities"):
            return await self._vector.list_entities(  # type: ignore[unresolved-attribute]
                namespace_id,
                entity_type=entity_type,
                source_chunk_ids=source_chunk_ids,
                limit=limit,
                offset=offset,
            )
        return []

    async def update_entity_embedding(
        self,
        entity_id: UUID,
        embedding: list[float],
        model: str,
        *,
        namespace_id: UUID,
    ) -> None:
        """Update the embedding for an entity, scoped to ``namespace_id`` (IDOR family)."""
        if self._vector:
            await self._vector.update_entity_embedding(entity_id, embedding, model, namespace_id=namespace_id)

    async def update_entity_embeddings_batch(
        self,
        updates: list[tuple[UUID, list[float], str]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Update embeddings for multiple entities in a single transaction, scoped to ``namespace_id`` (IDOR family)."""
        if self._vector and hasattr(self._vector, "update_entity_embeddings_batch"):
            return await self._vector.update_entity_embeddings_batch(updates, namespace_id=namespace_id)
        # Fallback to individual updates (sequential)
        if self._vector:
            for entity_id, embedding, model in updates:
                await self._vector.update_entity_embedding(entity_id, embedding, model, namespace_id=namespace_id)
            return len(updates)
        return 0

    @_record_storage_op("search_similar_entities", "pgvector")
    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[UUID, float]]:
        """Search for similar entities."""
        if not self._vector:
            raise RuntimeError("Vector backend not configured")
        namespace_id = await self._resolve_read_namespace(namespace_id)
        return await self._vector.search_similar_entities(
            namespace_id,
            query_embedding,
            limit=limit,
            min_similarity=min_similarity,
        )

    @_record_storage_op("upsert_entities_batch", "graph+vector")
    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 200,
        bulk_mode: bool = False,
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities across graph and vector backends.

        Uses MERGE semantics: creates new entities, updates existing ones
        matched by (namespace_id, name, entity_type).

        Write ordering (issue #868): vector backend commits first, then the
        graph backend.  The prior implementation raced both writes via
        ``asyncio.gather`` with no exception handling.  When the vector
        write raised after the graph had already committed (Neo4j
        ``session.execute_write`` commits on coroutine return), the graph
        was left with orphan nodes that the read path could not see.

        Sequencing vector first inverts the failure asymmetry: a graph
        failure after a successful vector write leaves vector rows that the
        next upsert reconciles via MERGE (entity uniqueness keyed on
        ``(namespace_id, name, entity_type)``), and the
        ``khora.storage.upsert_entities_batch.partial_failure`` counter
        increments so operators can monitor the rate.

        Args:
            bulk_mode: When True, skip prefetch/versioning and bypass the
                entity key gate.  Used for --rewrite (new namespace) where
                no existing entities can conflict.

        Returns list of (entity, is_new) tuples.
        """
        if not entities:
            return []

        results: list[tuple[Entity, bool]] = []

        # Upsert in graph and vector backends in parallel
        has_graph = self._graph and hasattr(self._graph, "upsert_entities_batch")
        has_vector = self._vector and hasattr(self._vector, "upsert_entities_batch")
        logger.debug(f"upsert_entities_batch: {len(entities)} entities, has_graph={has_graph}, has_vector={has_vector}")

        if has_graph and has_vector:
            assert self._graph is not None  # narrowed by has_graph
            assert self._vector is not None  # narrowed by has_vector
            if self._is_unified_backend:
                # Single DB — graph adapter upsert is sufficient
                results = await self._graph.upsert_entities_batch(
                    namespace_id,
                    entities,
                    batch_size=batch_size,
                    bulk_mode=bulk_mode,
                )
            else:
                # Vector first, then graph (issue #868). See method
                # docstring for the failure-mode rationale. A graph
                # exception after a successful vector commit raises and
                # bumps the partial_failure counter; the next upsert MERGE
                # reconciles the vector-side rows.
                await self._vector.upsert_entities_batch(  # type: ignore[unresolved-attribute]
                    namespace_id, entities, batch_size=batch_size
                )
                try:
                    results = await self._graph.upsert_entities_batch(
                        namespace_id,
                        entities,
                        batch_size=batch_size,
                        bulk_mode=bulk_mode,
                    )
                except Exception:
                    metric_counter(
                        "khora.storage.upsert_entities_batch.partial_failure",
                        unit="1",
                        description=(
                            "Vector commit succeeded but graph upsert "
                            "raised; vector rows are reconciled by the "
                            "next MERGE."
                        ),
                    ).add(1)
                    raise
        elif has_graph:
            assert self._graph is not None
            results = await self._graph.upsert_entities_batch(
                namespace_id,
                entities,
                batch_size=batch_size,
                bulk_mode=bulk_mode,
            )
            # Split embedded backends (e.g. sqlite_lance): the vector adapter
            # owns only entity *vectors* and has no upsert_entities_batch, so
            # the graph-only branch above persisted entity rows but dropped
            # their embeddings. Write the embeddings through
            # update_entity_embeddings_batch so the entity vector store is
            # populated; otherwise search_similar_entities returns nothing and
            # the VectorCypher GRAPH recall channel silently degrades to
            # vector-only (#1057). The graph upsert just wrote these entity rows
            # to the relational store, so the vector adapter can resolve their
            # namespaces by id.
            await self._persist_entity_embeddings_after_graph_upsert(results, namespace_id)
        elif has_vector:
            results = await self._vector.upsert_entities_batch(namespace_id, entities, batch_size=batch_size)  # type: ignore[unresolved-attribute]

        # Fallback: if no backend returned results, create synthetic results
        # to ensure callers always get one result per input entity
        if not results:
            logger.debug(f"upsert_entities_batch: using fallback synthetic results for {len(entities)} entities")
            results = [(entity, True) for entity in entities]

        logger.debug(f"upsert_entities_batch: returning {len(results)} results for {len(entities)} input entities")

        return results

    async def _persist_entity_embeddings_after_graph_upsert(
        self,
        results: list[tuple[Entity, bool]],
        namespace_id: UUID,
    ) -> None:
        """Write through entity embeddings on graph-only/split backends (#1057).

        When ``upsert_entities_batch`` took the graph-only branch (the vector
        adapter has no ``upsert_entities_batch``), entities that arrived with an
        embedding had it dropped. If the vector adapter can store entity vectors
        via ``update_entity_embeddings_batch``, persist them here so entity
        similarity search works. No-op when the vector adapter can't store entity
        vectors or no entity carries an embedding. Idempotent: the batch update
        uses upsert (delete + add) semantics.
        """
        vector = self._vector
        if vector is None or not hasattr(vector, "update_entity_embeddings_batch"):
            return
        updates = [
            (entity.id, entity.embedding, entity.embedding_model or "")
            for entity, _ in results
            if entity.embedding is not None
        ]
        if updates:
            await vector.update_entity_embeddings_batch(updates, namespace_id=namespace_id)

    @_record_storage_op("create_relationships_batch", "graph")
    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 50,
    ) -> list[tuple[Relationship, bool]]:
        """Batch create relationships in the graph backend.

        Returns one ``(relationship, is_new)`` tuple per persisted edge,
        mirroring ``upsert_entities_batch`` (#1320). Each relationship's ``id``
        is synced in place to the canonical stored edge id; ``is_new``
        distinguishes a genuine create from a dedup-merge so the
        ``relationship.created`` / ``relationship.updated`` hook dispatch
        reports the right event with the stored id. ``len(result)`` is the
        number of relationships written.
        """
        if not relationships:
            return []

        results: list[tuple[Relationship, bool]] = []
        if self._graph and hasattr(self._graph, "create_relationships_batch"):
            results = await self._graph.create_relationships_batch(relationships, batch_size=batch_size)
        elif not self._graph and self._vector and hasattr(self._vector, "create_relationships_batch"):
            # Graph-less stacks (PostgreSQL-only chronicle, #1066): persist
            # extracted relationships to the vector backend's relationships
            # mirror so they are not silently dropped, mirroring the entity
            # write/count fallback.
            results = await self._vector.create_relationships_batch(relationships, batch_size=batch_size)  # type: ignore[unresolved-attribute]

        return results

    # =========================================================================
    # Relationship operations (delegated to graph)
    # =========================================================================

    @_record_storage_op("create_relationship", "graph")
    async def create_relationship(self, relationship: Relationship) -> Relationship:
        """Create a relationship between entities."""
        if not self._graph:
            raise RuntimeError("Graph backend not configured")
        return await self._graph.create_relationship(relationship)

    async def get_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> Relationship | None:
        """Get a relationship by ID, scoped to ``namespace_id``.

        Returns ``None`` when the relationship belongs to a different
        namespace. ``namespace_id`` is required to prevent cross-tenant IDOR.
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            rel = await self._graph.get_relationship(relationship_id, namespace_id=namespace_id)
            if rel is None or rel.namespace_id != namespace_id:
                return None
            return rel
        return None

    async def delete_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a relationship, scoped to ``namespace_id`` (IDOR family)."""
        if self._graph:
            return await self._graph.delete_relationship(relationship_id, namespace_id=namespace_id)
        return False

    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity, scoped to ``namespace_id``.

        Returns an empty list if the entity does not belong to the caller's
        namespace. Edges that cross into other namespaces are excluded
        (IDOR family).
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            return await self._graph.get_entity_relationships(
                entity_id,
                namespace_id=namespace_id,
                direction=direction,
                relationship_types=relationship_types,
                limit=limit,
            )
        return []

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        between_entity_ids: list[UUID] | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        """List all relationships in a namespace.

        Prefers the graph backend when configured; otherwise falls back to
        the vector backend so graph-less stacks no longer raise — they will
        receive an empty list (chronicle on PG-only does not currently write
        the relationships table).
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            return await self._graph.list_relationships(
                namespace_id,
                relationship_type=relationship_type,
                between_entity_ids=between_entity_ids,
                limit=limit,
                offset=offset,
            )
        if self._vector and hasattr(self._vector, "list_relationships"):
            return await self._vector.list_relationships(  # type: ignore[unresolved-attribute]
                namespace_id,
                relationship_type=relationship_type,
                between_entity_ids=between_entity_ids,
                limit=limit,
                offset=offset,
            )
        return []

    # =========================================================================
    # Community summary operations (#1276 - GraphRAG payoff, delegated to graph)
    # =========================================================================

    async def get_communities(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CommunityNode]:
        """Return materialized dream :Community summary nodes for a namespace.

        Read-only. Communities are materialized into the graph by the dream
        community_summary mirror (#1276); a stack without a graph backend (or
        without materialized communities) returns an empty list.
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph and hasattr(self._graph, "get_communities"):
            return await self._graph.get_communities(namespace_id, limit=limit, offset=offset)
        return []

    async def get_entity_communities(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> list[CommunityNode]:
        """Return the dream :Community nodes the given entities are members of.

        The entity-anchored leg of the community recall reader (#1276): map a
        recall hit's entity set to the community summaries they belong to.
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph and hasattr(self._graph, "get_entity_communities"):
            return await self._graph.get_entity_communities(entity_ids, namespace_id=namespace_id)
        return []

    # =========================================================================
    # Episode operations (delegated to graph)
    # =========================================================================

    async def create_episode(self, episode: Episode) -> Episode:
        """Create an episode."""
        if not self._graph:
            raise RuntimeError("Graph backend not configured")
        return await self._graph.create_episode(episode)

    async def get_episode(self, episode_id: UUID, *, namespace_id: UUID) -> Episode | None:
        """Get an episode by ID, scoped to ``namespace_id``.

        Returns ``None`` when the episode belongs to a different namespace.
        ``namespace_id`` is required to prevent cross-tenant IDOR.
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            ep = await self._graph.get_episode(episode_id, namespace_id=namespace_id)
            if ep is None or ep.namespace_id != namespace_id:
                return None
            return ep
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
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            return await self._graph.list_episodes(namespace_id, start_time=start_time, end_time=end_time, limit=limit)
        return []

    # =========================================================================
    # Graph traversal (delegated to graph)
    # =========================================================================

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
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            return await self._graph.find_paths(
                source_entity_id,
                target_entity_id,
                namespace_id=namespace_id,
                max_depth=max_depth,
                relationship_types=relationship_types,
            )
        return []

    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity, scoped to ``namespace_id``.

        Returns ``{"entities": [], "relationships": []}`` if the seed entity
        belongs to a different namespace. Traversal never crosses namespace
        boundaries (IDOR family).
        """
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            return await self._graph.get_neighborhood(
                entity_id,
                namespace_id=namespace_id,
                depth=depth,
                relationship_types=relationship_types,
                limit=limit,
            )
        return {"entities": [], "relationships": []}

    # =========================================================================
    # Batch operations (optimized for parallel fetching)
    # =========================================================================

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        """Fetch multiple entities in a single query, scoped to ``namespace_id``.

        Entities belonging to any other namespace are silently dropped from
        the result (IDOR family).

        Args:
            entity_ids: List of entity IDs to fetch
            namespace_id: Caller's namespace; out-of-namespace rows are dropped.

        Returns:
            Dictionary mapping entity ID to Entity object
        """
        if not entity_ids:
            return {}
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            return await self._graph.get_entities_batch(entity_ids, namespace_id=namespace_id)
        # Fallback to pgvector for engines without graph backend (e.g., Chronicle)
        if self._vector and hasattr(self._vector, "get_entities_batch"):
            return await self._vector.get_entities_batch(  # type: ignore[unresolved-attribute]
                entity_ids, namespace_id=namespace_id
            )
        return {}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Entity]:
        """Fetch entities by name within a namespace.

        Used by Chronicle (graph-less) to resolve event subjects to Entity
        records. Delegates to the vector backend when present (pgvector
        implements this); returns ``{}`` for backends that don't.
        """
        if not names:
            return {}
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._vector and hasattr(self._vector, "get_entities_by_names_batch"):
            return await self._vector.get_entities_by_names_batch(namespace_id, names)
        return {}

    async def get_documents_batch(self, document_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query, scoped to ``namespace_id``.

        Documents belonging to any other namespace are silently dropped from
        the result (IDOR family).

        Args:
            document_ids: List of document IDs to fetch
            namespace_id: Caller's namespace; out-of-namespace rows are dropped.

        Returns:
            Dictionary mapping document ID to Document object
        """
        if not document_ids:
            return {}
        if self._relational:
            return await self._relational.get_documents_batch(document_ids, namespace_id=namespace_id)
        return {}

    async def get_document_sources_batch(
        self, document_ids: list[UUID], *, namespace_id: UUID
    ) -> dict[UUID, DocumentSource]:
        """Fetch lightweight document metadata for source attribution,
        scoped to ``namespace_id``.

        Documents in other namespaces are silently dropped from the result
        (IDOR family).

        Args:
            document_ids: List of document IDs to fetch
            namespace_id: Caller's namespace; out-of-namespace rows are dropped.

        Returns:
            Dictionary mapping document ID to DocumentSource
        """
        if not document_ids:
            return {}
        if self._relational:
            return await self._relational.get_document_sources_batch(document_ids, namespace_id=namespace_id)
        return {}

    async def get_document_projections_batch(
        self,
        document_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> dict[UUID, DocumentProjection]:
        """Fetch full DocumentProjection rows for recall responses.

        Args:
            document_ids: List of document IDs to fetch
            namespace_id: Namespace scope — cross-namespace ids are
                silently dropped from the result (security close-out).

        Returns:
            Dictionary mapping document ID to DocumentProjection
        """
        if not document_ids:
            return {}
        if self._relational:
            return await self._relational.get_document_projections_batch(document_ids, namespace_id=namespace_id)
        return {}

    @_record_storage_op("get_neighborhoods_batch", "graph")
    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
        prefer_current: bool = False,
    ) -> dict[UUID, dict[str, Any]]:
        """Get neighborhoods for multiple entities in a single query.

        Args:
            entity_ids: List of entity IDs
            depth: Max traversal depth
            relationship_types: Optional relationship type filter
            limit_per_entity: Max nodes per entity neighborhood
            prefer_current: When True, exclude paths that traverse any
                edge whose ``valid_until`` has passed. Forwarded to graph
                backends that support it (currently the embedded SQLite
                backend); silently dropped otherwise.

        Returns:
            Dictionary mapping entity ID to neighborhood data
        """
        if not entity_ids:
            return {}
        namespace_id = await self._resolve_read_namespace(namespace_id)
        if self._graph:
            kwargs: dict[str, Any] = {
                "namespace_id": namespace_id,
                "depth": depth,
                "relationship_types": relationship_types,
                "limit_per_entity": limit_per_entity,
            }
            # Only forward prefer_current when set, so backends whose
            # signatures don't accept the kwarg keep working.
            if prefer_current:
                try:
                    return await self._graph.get_neighborhoods_batch(entity_ids, **kwargs, prefer_current=True)
                except TypeError:
                    # Backend doesn't accept prefer_current — fall through.
                    pass
            return await self._graph.get_neighborhoods_batch(entity_ids, **kwargs)
        return {}

    # =========================================================================
    # Event operations (delegated to event store)
    # =========================================================================

    async def append_event(self, event: MemoryEvent) -> MemoryEvent:
        """Append an event to the log."""
        if not self._event_store:
            raise RuntimeError("Event store not configured")
        return await self._event_store.append_event(event)

    async def append_events_batch(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        """Append multiple events in a batch."""
        if not self._event_store:
            raise RuntimeError("Event store not configured")
        return await self._event_store.append_events_batch(events)

    async def get_events(
        self,
        namespace_id: UUID,
        *,
        event_types: list[str] | None = None,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEvent]:
        """Query events from the log."""
        if not self._event_store:
            raise RuntimeError("Event store not configured")
        return await self._event_store.get_events(
            namespace_id,
            event_types=event_types,
            resource_type=resource_type,
            resource_id=resource_id,
            after=after,
            before=before,
            limit=limit,
            offset=offset,
        )

    # =========================================================================
    # Chronicle event / fact operations (delegated to vector backend)
    #
    # Schema lives in chronicle_events / memory_facts (migration 024). Writes
    # for now go through the pgvector backend because it owns the embedding
    # column on chronicle_events. Engine wiring (EventExtractor / FactExtractor)
    # arrives in Chronicle #2/#3.
    # =========================================================================

    def _chronicle_backend(self, method_name: str) -> Any:
        """Pick the backend that implements a chronicle method.

        pgvector exposes chronicle writes on its vector adapter (the same
        backend owns both vector and SQL). sqlite_lance puts them on the
        relational adapter (vector = LanceDB, which has no SQL session).
        Prefer vector for back-compat, fall back to relational.
        """
        if self._vector is not None and hasattr(self._vector, method_name):
            return self._vector
        if self._relational is not None and hasattr(self._relational, method_name):
            return self._relational
        raise RuntimeError(f"No backend supports chronicle method {method_name!r}")

    async def write_events(
        self,
        events: list[Any],
        *,
        namespace_id: UUID,
    ) -> list[UUID]:
        """Persist Chronicle events to the chronicle_events table.

        Returns the list of inserted event IDs in input order.
        """
        if not events:
            return []
        return await self._chronicle_backend("write_events").write_events(events, namespace_id=namespace_id)

    async def write_facts(
        self,
        facts: list[Any],
        *,
        namespace_id: UUID,
    ) -> list[UUID]:
        """Persist memory facts to the memory_facts table.

        Returns the list of inserted fact IDs in input order.
        """
        if not facts:
            return []
        return await self._chronicle_backend("write_facts").write_facts(facts, namespace_id=namespace_id)

    async def query_events(
        self,
        namespace_id: UUID,
        *,
        subject: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """Query chronicle_events filtered by subject and ``referenced_date`` range."""
        return await self._chronicle_backend("query_events").query_events(
            namespace_id,
            subject=subject,
            since=since,
            until=until,
            limit=limit,
        )

    async def query_active_facts_for_subject(
        self,
        namespace_id: UUID,
        subject: str,
    ) -> list[Any]:
        """Return all active (not superseded) memory facts for a subject."""
        return await self._chronicle_backend("query_active_facts_for_subject").query_active_facts_for_subject(
            namespace_id, subject
        )

    async def supersede_fact(self, fact_id: UUID, superseded_by: UUID, *, namespace_id: UUID) -> None:
        """Mark a fact inactive and record the replacement fact ID, scoped to ``namespace_id`` (IDOR family)."""
        await self._chronicle_backend("supersede_fact").supersede_fact(
            fact_id, superseded_by, namespace_id=namespace_id
        )

    async def delete_facts_for_chunks(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> int:
        """Hard-delete memory facts referencing any of ``chunk_ids``.

        Forget-cascade cleanup (#1140): ``memory_facts`` carries chunk
        provenance only in the non-FK ``source_chunk_ids`` array, so document
        deletion never cascades to it. Returns the number of facts deleted.
        """
        if not chunk_ids:
            return 0
        return await self._chronicle_backend("delete_facts_for_chunks").delete_facts_for_chunks(
            chunk_ids, namespace_id=namespace_id
        )

    # =========================================================================
    # Sync checkpoint operations (delegated to relational)
    # =========================================================================

    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        """Get the last sync checkpoint for a source."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        return await self._relational.get_sync_checkpoint(namespace_id, source)

    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        """Set the sync checkpoint for a source."""
        if not self._relational:
            raise RuntimeError("Relational backend not configured")
        await self._relational.set_sync_checkpoint(namespace_id, source, checkpoint)
