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
    Document,
    Entity,
    Episode,
    MemoryEvent,
    MemoryNamespace,
    Relationship,
)
from khora.core.models.document import DocumentSource
from khora.storage.backends.base import PaginatedResult
from khora.telemetry import get_collector, trace_span

if TYPE_CHECKING:
    from .backends.base import (
        EventStoreProtocol,
        GraphBackendProtocol,
        RelationalBackendProtocol,
        VectorBackendProtocol,
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
    """

    document_id: UUID
    chunks_deleted: int
    chunks_created: int
    entities_created: int
    entities_updated: int
    entities_retired: int
    relationships_created: int
    relationships_retired: int


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

    _connected: bool = field(default=False, init=False)
    _is_unified_backend: bool = field(default=False, init=False)
    _hook_dispatcher: Any = field(default=None, init=False)

    async def dispatch_hook(self, event: Any) -> None:
        """Dispatch an event to hook subscribers if a dispatcher is attached.

        Called by the ingestion pipeline after extraction/storage operations.
        No-op if no dispatcher is set (i.e., no hooks subscribed).
        """
        if self._hook_dispatcher is not None and self._hook_dispatcher.subscription_count > 0:
            await self._hook_dispatcher.dispatch(event)

    def __post_init__(self) -> None:
        # Detect if graph and vector share a SurrealDB connection (unified backend).
        # Some adapters (e.g. SQLiteLance) expose ``_conn`` as a property that
        # raises when the underlying connection isn't open yet — treat any
        # error as "not unified" since the probe is advisory.
        if self.graph is not None and self.vector is not None:
            try:
                graph_conn = getattr(self.graph, "_conn", None)
                vector_conn = getattr(self.vector, "_conn", None)
            except Exception:
                graph_conn = None
                vector_conn = None
            if graph_conn is not None and graph_conn is vector_conn:
                self._is_unified_backend = True
                logger.info("Detected unified SurrealDB backend — entity dual-writes will be collapsed")

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
        if self.relational:
            tasks.append(self.relational.connect())
        if self.vector:
            tasks.append(self.vector.connect())
        if self.graph:
            tasks.append(self.graph.connect())
        if self.event_store:
            tasks.append(self.event_store.connect())

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
        if self.event_store:
            tasks.append(self.event_store.disconnect())
        if self.graph:
            tasks.append(self.graph.disconnect())
        if self.vector:
            tasks.append(self.vector.disconnect())
        if self.relational:
            tasks.append(self.relational.disconnect())

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
        if self.relational:
            checks.append(("relational", self.relational.is_healthy()))
        if self.vector:
            checks.append(("vector", self.vector.is_healthy()))
        if self.graph:
            checks.append(("graph", self.graph.is_healthy()))
        if self.event_store:
            checks.append(("event_store", self.event_store.is_healthy()))

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
        for backend in (self.relational, self.vector, self.event_store):
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
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.resolve_namespace(namespace_id)

    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Create a new memory namespace."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.create_namespace(namespace)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_namespace(namespace_id)

    async def list_namespaces(
        self, *, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> PaginatedResult[MemoryNamespace]:
        """List namespaces with pagination."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.list_namespaces(active_only=active_only, limit=limit, offset=offset)

    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Update a namespace."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.update_namespace(namespace)

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
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.create_namespace_version(previous_version=previous_version)

    async def deactivate_namespace(self, namespace_id: UUID) -> None:
        """Mark a namespace version as inactive.

        Args:
            namespace_id: ID of the namespace to deactivate
        """
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        await self.relational.deactivate_namespace(namespace_id)

    # =========================================================================
    # Document operations (delegated to relational)
    # =========================================================================

    @_record_storage_op("create_document", "postgresql")
    async def create_document(self, document: Document) -> Document:
        """Create a new document."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.create_document(document)

    async def get_document(self, document_id: UUID) -> Document | None:
        """Get a document by ID."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_document(document_id)

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
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.list_documents(
            namespace_id, status=status, updated_before=updated_before, limit=limit, offset=offset
        )

    async def update_document(self, document: Document) -> Document:
        """Update a document."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.update_document(document)

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and its chunks."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")

        # Delete chunks first
        if self.vector:
            await self.vector.delete_chunks_by_document(document_id)

        return await self.relational.delete_document(document_id)

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
        3. On success the document is marked ``COMPLETED``; on any exception
           it is marked ``FAILED`` with the error string and the exception is
           re-raised unwrapped, mirroring ``remember()`` (``ingest.py:1356``).
           A ``FAILED`` document self-heals on the next successful replace
           against the same ``external_id``.

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
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        if not self.graph:
            raise RuntimeError("Graph backend not configured")

        try:
            # 1. Prefetch old graph state and compute retire / survive sets
            #    BEFORE mutating anything.  Doing this up front keeps the
            #    Python-side filter aligned with the Cypher.
            fetch = getattr(self.graph, "fetch_document_extraction_state", None)
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

            # For relationships, identity is (src_entity, tgt_entity,
            # sanitized_type).  Both sides of the comparison are now
            # sanitized, so mixed-case / punctuated rel types classify
            # correctly as survivors instead of leaking into net-new + retire.
            relationship_retirement_rows: list[dict[str, Any]] = []
            relationship_survivor_remap_rows: list[dict[str, str]] = []
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

            # 2. Postgres transaction: atomic chunk hard-replace.
            #    Embedding (OpenAI roundtrip) happened before this block — the
            #    transaction deliberately wraps only DB work (ADR §Performance).
            async with self.transaction() as txn:
                await self.relational.update_document(new_document, session=txn.session)  # type: ignore[unresolved-attribute]
                chunks_deleted = await self.vector.delete_chunks_by_document(old_document_id, session=txn.session)
                await self.vector.create_chunks_batch(new_chunks, session=txn.session)  # type: ignore[unresolved-attribute]

            # 3. Graph-side retirement / remap (after PG commits).  Order:
            #    retire -> remap -> upsert.  Retirement snapshots the current
            #    source_document_ids before we change them; remap cleanly
            #    swaps old->new on survivors before upsert would append
            #    new_doc_id a second time.  Net-new entities/relationships
            #    are those with keys absent from the old extraction.
            entities_retired = 0
            if entity_retirement_rows:
                entities_retired = await self.graph.retire_orphaned_entities_batch(  # type: ignore[unresolved-attribute]
                    entity_retirement_rows
                )

            relationships_retired = 0
            if relationship_retirement_rows:
                relationships_retired = await self.graph.retire_orphaned_relationships_batch(  # type: ignore[unresolved-attribute]
                    relationship_retirement_rows
                )

            if entity_survivor_remap_rows or relationship_survivor_remap_rows:
                await self.graph.remap_source_document_ids_batch(  # type: ignore[unresolved-attribute]
                    entity_survivors=entity_survivor_remap_rows,
                    relationship_survivors=relationship_survivor_remap_rows,
                )

            net_new_entities = [e for e in new_entities if (e.name, e.entity_type) not in entity_survivor_keys]
            entities_created = 0
            entities_updated = len(entity_survivor_remap_rows)
            if net_new_entities:
                upsert_results = await self.upsert_entities_batch(namespace_id, net_new_entities)
                entities_created = sum(1 for _, is_new in upsert_results if is_new)
                entities_updated += sum(1 for _, is_new in upsert_results if not is_new)

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
            relationships_created = 0
            if net_new_relationships:
                relationships_created = await self.create_relationships_batch(net_new_relationships)
            # Survivor relationships are accounted for implicitly via remap;
            # their id is preserved from the old graph state.

            new_document.mark_completed(len(new_chunks), len(new_entities), relationships_created)
            await self.relational.update_document(new_document)

            return ReplaceResult(
                document_id=new_document.id,
                chunks_deleted=chunks_deleted,
                chunks_created=len(new_chunks),
                entities_created=entities_created,
                entities_updated=entities_updated,
                entities_retired=entities_retired,
                relationships_created=relationships_created,
                relationships_retired=relationships_retired,
            )

        except Exception as e:
            # Mirrors remember() (pipelines/flows/ingest.py:1356-1359):
            # mark FAILED with error string and re-raise unwrapped so the
            # caller observes the original exception and the next successful
            # replace can self-heal the row (ADR §Decision #8).
            new_document.mark_failed(str(e))
            try:
                await self.relational.update_document(new_document)
            except Exception as persist_error:
                logger.error(
                    f"Failed to persist FAILED status after replace_document_extraction error: {persist_error}"
                )
            raise

    async def count_documents(self, namespace_id: UUID) -> int:
        """Count documents in a namespace.

        Args:
            namespace_id: Namespace UUID

        Returns:
            Total number of documents in the namespace (0 if empty)

        Raises:
            RuntimeError: If relational backend is not configured
        """
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.count_documents(namespace_id)

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
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_last_activity_at(namespace_id)

    async def get_document_stats(self, namespace_id: UUID) -> tuple[int, datetime | None]:
        """Get document count and last activity timestamp in a single query."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_document_stats(namespace_id)

    async def get_document_by_checksum(self, namespace_id: UUID, checksum: str) -> Document | None:
        """Get a document by its content checksum."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_document_by_checksum(namespace_id, checksum)

    async def get_document_by_external_id(self, namespace_id: UUID, external_id: str | None) -> Document | None:
        """Get a document by (namespace_id, external_id).

        Unlike ``get_document_by_checksum``, this lookup returns documents in
        any status (including ``FAILED`` and ``PROCESSING``) so callers can
        self-heal on the next successful replace.
        """
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_document_by_external_id(namespace_id, external_id)

    async def get_documents_by_external_ids(self, namespace_id: UUID, external_ids: list[str]) -> dict[str, Document]:
        """Batch variant of :meth:`get_document_by_external_id`.

        Collapses N serial lookups into one query for ``remember_batch`` replace
        dispatch. Status-agnostic like the single lookup.
        """
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_documents_by_external_ids(namespace_id, external_ids)

    async def get_documents_by_checksums(self, namespace_id: UUID, checksums: list[str]) -> dict[str, Document]:
        """Fetch documents by content checksums in a single query.

        Used for batch deduplication to avoid N serial DB queries.

        Args:
            namespace_id: Namespace to search in
            checksums: List of content checksums to look up

        Returns:
            Dictionary mapping checksum to Document (only for existing documents)
        """
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_documents_by_checksums(namespace_id, checksums)  # type: ignore[unresolved-attribute]

    # =========================================================================
    # Chunk operations (delegated to vector)
    # =========================================================================

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        """Create a new chunk with embedding."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.create_chunk(chunk)

    @_record_storage_op("create_chunks_batch", "pgvector")
    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """Create multiple chunks in a batch."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.create_chunks_batch(chunks)

    async def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        """Get a chunk by ID."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.get_chunk(chunk_id)

    async def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        """Get all chunks for a document."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.get_chunks_by_document(document_id)

    async def get_chunks_batch(self, chunk_ids: list[UUID]) -> dict[UUID, Chunk]:
        """Fetch multiple chunks by ID in a single query.

        Args:
            chunk_ids: List of chunk IDs to fetch

        Returns:
            Dictionary mapping chunk ID to Chunk (only for existing chunks)
        """
        if not chunk_ids:
            return {}
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.get_chunks_batch(chunk_ids)

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
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.search_similar(
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
    ) -> list[tuple[Chunk, float]]:
        """Search chunks using PostgreSQL full-text search."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.search_fulltext(
            namespace_id,
            query_text,
            limit=limit,
            language=language,
            created_after=created_after,
            created_before=created_before,
        )

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Count chunks in a namespace."""
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.count_chunks(namespace_id)

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
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.list_chunks(namespace_id, limit=limit, offset=offset)

    async def count_entities(self, namespace_id: UUID) -> int:
        """Count entities in a namespace. Best-effort during active ingestion (non-atomic dual-write)."""
        if self.vector:
            return await self.vector.count_entities(namespace_id)
        if self.graph:
            return await self.graph.count_entities(namespace_id)
        return 0

    async def count_relationships(self, namespace_id: UUID) -> int:
        """Count relationships in a namespace (graph-only)."""
        if self.graph:
            return await self.graph.count_relationships(namespace_id)
        return 0

    # =========================================================================
    # Entity operations (cross-backend)
    # =========================================================================

    @_record_storage_op("create_entity", "graph+vector")
    async def create_entity(self, entity: Entity) -> Entity:
        """Create an entity in both graph and vector stores (parallel).

        When a unified backend is detected (graph and vector share the same
        connection, e.g. SurrealDB), the vector write is skipped to avoid
        duplicate records in the same database.
        """
        if self.graph and self.vector:
            if self._is_unified_backend:
                # Single DB — graph adapter write is sufficient
                entity = await self.graph.create_entity(entity)
            else:
                graph_result, _ = await asyncio.gather(
                    self.graph.create_entity(entity),
                    self.vector.create_entity(entity),
                )
                entity = graph_result
        elif self.graph:
            entity = await self.graph.create_entity(entity)
        elif self.vector:
            await self.vector.create_entity(entity)
        return entity

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        if self.graph:
            return await self.graph.get_entity(entity_id)
        return None

    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        """Get an entity by name and type."""
        if self.graph:
            return await self.graph.get_entity_by_name(namespace_id, name, entity_type)
        return None

    async def update_entity(self, entity: Entity) -> Entity:
        """Update an entity in both graph and vector stores (parallel).

        When a unified backend is detected, the vector write is skipped.
        """
        if self.graph and self.vector:
            if self._is_unified_backend:
                return await self.graph.update_entity(entity)
            graph_result, _ = await asyncio.gather(
                self.graph.update_entity(entity),
                self.vector.update_entity(entity),
            )
            return graph_result
        if self.graph:
            return await self.graph.update_entity(entity)
        if self.vector:
            await self.vector.update_entity(entity)
        return entity

    async def delete_entity(self, entity_id: UUID) -> bool:
        """Delete an entity."""
        if self.graph:
            return await self.graph.delete_entity(entity_id)
        return False

    async def list_entities(
        self,
        namespace_id: UUID,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities in a namespace."""
        if self.graph:
            return await self.graph.list_entities(namespace_id, entity_type=entity_type, limit=limit, offset=offset)
        return []

    async def update_entity_embedding(self, entity_id: UUID, embedding: list[float], model: str) -> None:
        """Update the embedding for an entity."""
        if self.vector:
            await self.vector.update_entity_embedding(entity_id, embedding, model)

    async def update_entity_embeddings_batch(self, updates: list[tuple[UUID, list[float], str]]) -> int:
        """Update embeddings for multiple entities in a single transaction."""
        if self.vector and hasattr(self.vector, "update_entity_embeddings_batch"):
            return await self.vector.update_entity_embeddings_batch(updates)
        # Fallback to individual updates (sequential)
        if self.vector:
            for entity_id, embedding, model in updates:
                await self.vector.update_entity_embedding(entity_id, embedding, model)
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
        if not self.vector:
            raise RuntimeError("Vector backend not configured")
        return await self.vector.search_similar_entities(
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
        has_graph = self.graph and hasattr(self.graph, "upsert_entities_batch")
        has_vector = self.vector and hasattr(self.vector, "upsert_entities_batch")
        logger.debug(f"upsert_entities_batch: {len(entities)} entities, has_graph={has_graph}, has_vector={has_vector}")

        if has_graph and has_vector:
            assert self.graph is not None  # narrowed by has_graph
            assert self.vector is not None  # narrowed by has_vector
            if self._is_unified_backend:
                # Single DB — graph adapter upsert is sufficient
                results = await self.graph.upsert_entities_batch(
                    namespace_id,
                    entities,
                    batch_size=batch_size,
                    bulk_mode=bulk_mode,
                )
            else:
                graph_results, _ = await asyncio.gather(
                    self.graph.upsert_entities_batch(
                        namespace_id,
                        entities,
                        batch_size=batch_size,
                        bulk_mode=bulk_mode,
                    ),
                    self.vector.upsert_entities_batch(namespace_id, entities, batch_size=batch_size),  # type: ignore[unresolved-attribute]
                )
                results = graph_results
        elif has_graph:
            assert self.graph is not None
            results = await self.graph.upsert_entities_batch(
                namespace_id,
                entities,
                batch_size=batch_size,
                bulk_mode=bulk_mode,
            )
        elif has_vector:
            results = await self.vector.upsert_entities_batch(namespace_id, entities, batch_size=batch_size)  # type: ignore[unresolved-attribute]

        # Fallback: if no backend returned results, create synthetic results
        # to ensure callers always get one result per input entity
        if not results:
            logger.debug(f"upsert_entities_batch: using fallback synthetic results for {len(entities)} entities")
            results = [(entity, True) for entity in entities]

        logger.debug(f"upsert_entities_batch: returning {len(results)} results for {len(entities)} input entities")

        return results

    @_record_storage_op("create_relationships_batch", "graph")
    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 50,
    ) -> int:
        """Batch create relationships in the graph backend.

        Returns the number of relationships created.
        """
        if not relationships:
            return 0

        count = 0
        if self.graph and hasattr(self.graph, "create_relationships_batch"):
            count = await self.graph.create_relationships_batch(relationships, batch_size=batch_size)

        return count

    # =========================================================================
    # Relationship operations (delegated to graph)
    # =========================================================================

    @_record_storage_op("create_relationship", "graph")
    async def create_relationship(self, relationship: Relationship) -> Relationship:
        """Create a relationship between entities."""
        if not self.graph:
            raise RuntimeError("Graph backend not configured")
        return await self.graph.create_relationship(relationship)

    async def get_relationship(self, relationship_id: UUID) -> Relationship | None:
        """Get a relationship by ID."""
        if self.graph:
            return await self.graph.get_relationship(relationship_id)
        return None

    async def delete_relationship(self, relationship_id: UUID) -> bool:
        """Delete a relationship."""
        if self.graph:
            return await self.graph.delete_relationship(relationship_id)
        return False

    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        direction: str = "both",
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        if self.graph:
            return await self.graph.get_entity_relationships(
                entity_id, direction=direction, relationship_types=relationship_types, limit=limit
            )
        return []

    async def list_relationships(
        self,
        namespace_id: UUID,
        *,
        relationship_type: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Relationship]:
        """List all relationships in a namespace."""
        if self.graph:
            return await self.graph.list_relationships(
                namespace_id, relationship_type=relationship_type, limit=limit, offset=offset
            )
        return []

    # =========================================================================
    # Episode operations (delegated to graph)
    # =========================================================================

    async def create_episode(self, episode: Episode) -> Episode:
        """Create an episode."""
        if not self.graph:
            raise RuntimeError("Graph backend not configured")
        return await self.graph.create_episode(episode)

    async def get_episode(self, episode_id: UUID) -> Episode | None:
        """Get an episode by ID."""
        if self.graph:
            return await self.graph.get_episode(episode_id)
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
        if self.graph:
            return await self.graph.list_episodes(namespace_id, start_time=start_time, end_time=end_time, limit=limit)
        return []

    # =========================================================================
    # Graph traversal (delegated to graph)
    # =========================================================================

    async def find_paths(
        self,
        namespace_id: UUID,
        source_entity_id: UUID,
        target_entity_id: UUID,
        *,
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Find paths between two entities."""
        if self.graph:
            return await self.graph.find_paths(
                namespace_id,
                source_entity_id,
                target_entity_id,
                max_depth=max_depth,
                relationship_types=relationship_types,
            )
        return []

    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity."""
        if self.graph:
            return await self.graph.get_neighborhood(
                entity_id, depth=depth, relationship_types=relationship_types, limit=limit
            )
        return {"entities": [], "relationships": []}

    # =========================================================================
    # Batch operations (optimized for parallel fetching)
    # =========================================================================

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
        """Fetch multiple entities in a single query.

        Args:
            entity_ids: List of entity IDs to fetch

        Returns:
            Dictionary mapping entity ID to Entity object
        """
        if not entity_ids:
            return {}
        if self.graph:
            return await self.graph.get_entities_batch(entity_ids)
        # Fallback to pgvector for engines without graph backend (e.g., Chronicle)
        if self.vector and hasattr(self.vector, "get_entities_batch"):
            return await self.vector.get_entities_batch(entity_ids)
        return {}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Entity]:
        """Fetch entities by name within a namespace.

        Used by Chronicle (graph-less) to resolve event subjects to Entity
        records. Delegates to the vector backend when present (pgvector
        implements this); returns ``{}`` for backends that don't.
        """
        if not names:
            return {}
        if self.vector and hasattr(self.vector, "get_entities_by_names_batch"):
            return await self.vector.get_entities_by_names_batch(namespace_id, names)
        return {}

    async def get_documents_batch(self, document_ids: list[UUID]) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            Dictionary mapping document ID to Document object
        """
        if not document_ids:
            return {}
        if self.relational:
            return await self.relational.get_documents_batch(document_ids)
        return {}

    async def get_document_sources_batch(self, document_ids: list[UUID]) -> dict[UUID, DocumentSource]:
        """Fetch lightweight document metadata for source attribution.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            Dictionary mapping document ID to DocumentSource
        """
        if not document_ids:
            return {}
        if self.relational:
            return await self.relational.get_document_sources_batch(document_ids)
        return {}

    @_record_storage_op("get_neighborhoods_batch", "graph")
    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
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
        if self.graph:
            kwargs: dict[str, Any] = {
                "depth": depth,
                "relationship_types": relationship_types,
                "limit_per_entity": limit_per_entity,
            }
            # Only forward prefer_current when set, so backends whose
            # signatures don't accept the kwarg keep working.
            if prefer_current:
                try:
                    return await self.graph.get_neighborhoods_batch(entity_ids, **kwargs, prefer_current=True)
                except TypeError:
                    # Backend doesn't accept prefer_current — fall through.
                    pass
            return await self.graph.get_neighborhoods_batch(entity_ids, **kwargs)
        return {}

    # =========================================================================
    # Event operations (delegated to event store)
    # =========================================================================

    async def append_event(self, event: MemoryEvent) -> MemoryEvent:
        """Append an event to the log."""
        if not self.event_store:
            raise RuntimeError("Event store not configured")
        return await self.event_store.append_event(event)

    async def append_events_batch(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        """Append multiple events in a batch."""
        if not self.event_store:
            raise RuntimeError("Event store not configured")
        return await self.event_store.append_events_batch(events)

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
        if not self.event_store:
            raise RuntimeError("Event store not configured")
        return await self.event_store.get_events(
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
        if self.vector is not None and hasattr(self.vector, method_name):
            return self.vector
        if self.relational is not None and hasattr(self.relational, method_name):
            return self.relational
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

    async def supersede_fact(self, fact_id: UUID, superseded_by: UUID) -> None:
        """Mark a fact inactive and record the replacement fact ID."""
        await self._chronicle_backend("supersede_fact").supersede_fact(fact_id, superseded_by)

    # =========================================================================
    # Sync checkpoint operations (delegated to relational)
    # =========================================================================

    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        """Get the last sync checkpoint for a source."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        return await self.relational.get_sync_checkpoint(namespace_id, source)

    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        """Set the sync checkpoint for a source."""
        if not self.relational:
            raise RuntimeError("Relational backend not configured")
        await self.relational.set_sync_checkpoint(namespace_id, source, checkpoint)
