"""SurrealDB relational adapter for Khora.

Implements RelationalBackendProtocol using SurrealQL, delegating connection
lifecycle to SurrealDBConnection.  Record IDs follow the SurrealDB convention:
``table:⟨uuid⟩``.  All UUIDs are converted to ``str`` at the boundary and
parsed back on read.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger

from khora.core.models import Document, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentSource, DocumentStatus
from khora.core.models.recall import DocumentProjection
from khora.storage.backends.base import (
    DocumentScanKey,
    DocumentScanStep,
    PaginatedResult,
    build_scan_step,
)
from khora.storage.backends.surrealdb._helpers import (
    _parse_dt,
    _parse_uuid,
    _record_id,
)
from khora.storage.backends.surrealdb.connection import SurrealDBConnection

if TYPE_CHECKING:
    from khora.filter.ast import FilterNode

# ---------------------------------------------------------------------------
# Datetime binds
#
# Every ``datetime`` bound into a SurrealQL statement in this module binds as a
# ``datetime`` OBJECT. Never ``.isoformat()``. The timestamp columns are
# ``TYPE datetime`` (``schema.py``) and SurrealDB does not coerce a string
# operand to match one:
#
# * In a COMPARISON a string operand never reaches the stored value. Measured
#   with ``RETURN $dt <op> $str`` over all six operators, the answer is the same
#   for a past string, a future string and the non-timestamp ``'zzz'``: ``<``,
#   ``<=`` and ``=`` are always false, ``>``, ``>=`` and ``!=`` always true.
#   Inference from that value-independence — the engine ranks the two types and
#   stops there. So a string bind does not misorder a predicate, it PINS it:
#   ``updated_at < $cutoff`` matches nothing and ``updated_at >= $cutoff``
#   matches everything. Both directions fail silently; neither raises.
# * In a WRITE a string is REJECTED, not coerced — ``InternalError: ... expected
#   a datetime`` — and the whole statement is discarded: a multi-field ``UPDATE``
#   leaves every field untouched, not just the rejected one.
#
# Measured with python SDK 2.0.0, identical on all three modes: ``memory://``
# and ``surrealkv://`` (file-backed embedded, both on the SDK-bundled core
# 2.3.10 — read off the extension binary, not inferred from the Python package
# version) and ``ws://`` (standalone server 2.3.7). Every leg is therefore
# 2.3.x: the behaviour is measured on 2.3.x only. NOT measured below 2.3, and
# NOT measured against a SurrealDB 3.x engine.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Disjunctions and index selection
#
# **No ``document`` read in this module may contain an ``OR``.** Three of them
# used to, and each one made the statement read every ``document`` row in the
# database — all tenants included, not just the namespace asked for.
#
# The rule, measured on 2.x rather than assumed. SurrealDB *can* answer a
# disjunction as a union of index scans, but only when EVERY disjunct is a
# comparison on a field carried by a **single-field** index. A composite index
# does not qualify, not even on its own leading column:
#
# * ``status = 'a' OR status = 'b'`` with a one-field ``status`` index — two
#   ``Iterate Index`` operations, and it stays indexed when wrapped in an outer
#   ``AND`` (three operations, the extra one serving the conjunct).
# * ``checksum = $a OR checksum = $b`` with only the two-field
#   ``idx_document_ns_checksum`` — ``Iterate Table``.
#
# When one disjunct cannot be served, the fallback is not scoped to that
# disjunct: the whole statement becomes ``Iterate Table`` and the conjuncts
# around it lose their indexes too. That is why the ``namespace_id`` scope
# disappears along with the keyset — the tenant filter still *filters*, it just
# stops *narrowing*, so correctness is unaffected and cost is not.
#
# ``document`` carries composite ``(namespace_id, …)`` indexes and nothing else,
# by design: the one-field ``idx_document_namespace`` was dropped as a redundant
# prefix of the sort index. So on this table the qualifying case cannot arise and
# **every** ``OR`` scans, whatever its shape — verified over eight, including a
# bare ``OR`` of two composite leading columns and an ``OR`` of two full
# composite matches. Defining one-field indexes to buy the union back is not a
# trade worth making: it is permanent write amplification on every ``document``
# insert to salvage a predicate that splits into conjunctive legs for free.
#
# The measurements above are on ``memory://`` with the SDK-bundled core; the
# adapter's three former ``OR`` sites were each confirmed ``Iterate Table``
# before the split and ``Iterate Index`` per leg after. Any predicate added here
# should be checked the same way — the plan tests re-issue every statement this
# adapter sends with ``EXPLAIN`` appended, so a new ``OR`` fails there.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _none_if_empty(v: str | None) -> str | None:
    return v if v else None


def _checksum_reingestable_legs(pending_stale_before: datetime | None) -> list[tuple[str, dict[str, Any]]]:
    """Build the checksum-dedup exclusion as OR-free, index-eligible legs.

    FAILED rows are always excluded (re-ingestable). When ``pending_stale_before``
    is given, PENDING rows older than that cutoff are also excluded so a
    crash-abandoned half-ingest re-ingests; fresh PENDING rows stay a dedup hit,
    preserving the concurrent in-flight guard. When ``None`` only FAILED is
    excluded (legacy behavior).

    **Why legs rather than one clause.** The predicate this replaces was
    ``status != 'failed' AND (status != 'pending' OR updated_at >= $cutoff)``. A
    nested ``OR`` costs the whole statement its indexes on this table, the
    ``namespace_id`` prefix included (see the disjunction note at the top of this
    module), so a dedup probe read every ``document`` row in the database, all
    namespaces included. Since ``pending_stale_before`` is set on the production
    ingest path, that was every dedup probe. The same set is expressed as two
    disjoint conjunctions:

    * ``status != 'failed' AND status != 'pending'`` — neither failed nor pending
    * ``status = 'pending' AND updated_at >= $cutoff`` — pending but still fresh

    Their union is exactly the original predicate and their intersection is empty
    (they disagree on ``status``), so a caller may concatenate leg results with no
    comparator and no dedup. Each leg alone plans ``Iterate Index`` on
    ``idx_document_ns_checksum`` / ``idx_document_ns_status``.

    The cutoff binds as a ``datetime`` object (see the datetime-binds note at the
    top of this module). Bound as an ISO string the ``>=`` compare is
    unconditionally true, so the fresh-PENDING leg would match every PENDING row
    however stale and silently disable the re-ingest.

    Returns:
        ``[(clause, binds), ...]`` — one entry when ``pending_stale_before`` is
        ``None``, two otherwise. Legs are ordered non-pending first, matching the
        precedence a single-row consumer wants: a settled row outranks an
        in-flight one.
    """
    if pending_stale_before is None:
        return [("status != 'failed'", {})]
    return [
        ("status != 'failed' AND status != 'pending'", {}),
        (
            "status = 'pending' AND updated_at >= $pending_stale_before",
            {"pending_stale_before": pending_stale_before},
        ),
    ]


def _scan_key_from_row(row: dict[str, Any]) -> DocumentScanKey:
    """Build a keyset position from a RAW ``document`` row, masking nothing.

    ``_row_to_document`` is the wrong source for a cursor even though it is the
    convenient one: it coalesces a missing ``created_at`` to ``datetime.now(UTC)``
    for the domain object's sake, and a ``now()`` timestamp sorts above the whole
    window, so the next step re-reads the rows it just returned and the walk loops
    instead of advancing. Reading the raw row and raising instead keeps a schema
    violation loud — ``created_at`` is a non-optional ``TYPE datetime`` on this
    table (``schema.py``), so a ``None`` here means the row did not come from this
    schema and there is no position to resume from.

    The record id is parsed with ``_parse_uuid(..., strict=True)`` for the same
    reason. khora writes every ``document`` id as a ``document:⟨uuid⟩`` record id
    through ``_record_id``, so strict mode is a no-op on any row this store
    produces. On a row written directly outside khora with a non-UUID id, the
    non-strict fallback would derive a well-formed ``uuid5`` — a position no row
    holds — and the keyset walk would cycle without terminating (the record-id
    homogeneity precondition documented on
    :meth:`SurrealDBRelationalAdapter.scan_documents`). Strict parsing catches it
    at the raw id, before the derivation, and raises loudly like the ``created_at``
    guard above: a walk with no resumable position stops rather than repeating a
    page forever.
    """
    created_at = _parse_dt(row.get("created_at"))
    if created_at is None:
        raise ValueError(f"document row has no readable created_at; cannot build a scan cursor: {row.get('id')!r}")
    return (created_at, _parse_uuid(row["id"], strict=True))


# Bind names :meth:`SurrealDBRelationalAdapter.scan_documents` owns. A compiled
# filter fragment that emitted any of them would silently overwrite a scan bind —
# a compiled ``ns`` would replace the tenant scope and return another namespace's
# rows with no error anywhere — so the splice checks against this set rather than
# against whichever optional binds happen to be present on the current call.
_SCAN_RESERVED_BINDS: frozenset[str] = frozenset(
    {"ns", "lim", "status", "updated_before", "after_created_at", "after_id"}
)


def _documents_where(
    namespace_id: UUID,
    *,
    status: str | None,
    updated_before: datetime | None,
    filter_ast: FilterNode | None,
) -> tuple[list[str], dict[str, Any], frozenset[str]]:
    """Build the namespace/status/filter conjuncts shared by every scan statement.

    :meth:`SurrealDBRelationalAdapter.scan_documents` issues up to two statements
    per step and both must carry the identical narrowing, so the block lives here
    once instead of being duplicated per statement. The cursor bound is NOT
    included — it is the only part that differs between the two.

    Returns ``(conditions, params, consumed_keys)``. ``conditions`` is a list of
    ``AND``-joinable fragments; every fragment is parenthesized unless it is a
    single comparison, so a caller may append its own conjunct without thinking
    about operator precedence.

    Raises ``ValueError`` when a compiled filter's binds collide with the scan's
    own reserved names (see :data:`_SCAN_RESERVED_BINDS`).
    """
    conditions = ["namespace_id = $ns"]
    params: dict[str, Any] = {"ns": str(namespace_id)}
    if status:
        conditions.append("status = $status")
        params["status"] = status
    if updated_before is not None:
        conditions.append("updated_at < $updated_before")
        params["updated_before"] = updated_before

    consumed_keys: frozenset[str] = frozenset()
    if filter_ast is not None:
        compiler = CompilerRegistry.get("relational.surrealdb", "documents")
        # No ``CompileError`` mapping here. Under this context's
        # ``on_unsupported="split"``, an unrenderable metadata segment is a
        # capability gap the compiler defers (see
        # ``_documents_compile_context``), so the only ``CompileError`` left
        # reaching this line would be a genuine compiler fault — which must
        # escape as itself rather than be relabelled a caller input problem.
        compiled = compiler(filter_ast, _documents_compile_context())
        consumed_keys = compiled.consumed_keys
        conditions.append(f"({compiled.predicate})")
        # Compiled binds are ``{param_namespace}_{n}`` — ``f_0``, ``f_1``, … — and
        # the scan's are the reserved set, so the two families are disjoint by
        # construction. The invariant runs BOTH ways and neither direction is safe
        # to break: never name a scan bind ``f_<n>``, and never set this context's
        # ``param_namespace`` to anything that could produce a reserved name.
        collisions = _SCAN_RESERVED_BINDS & compiled.params.keys()
        if collisions:
            raise ValueError(f"compiled filter binds collide with scan binds: {sorted(collisions)}")
        params.update(compiled.params)

    return conditions, params, consumed_keys


class SurrealDBRelationalAdapter:
    """Relational backend backed by SurrealDB.

    Fulfils :class:`~khora.storage.backends.base.RelationalBackendProtocol`
    without importing SQLAlchemy.  The adapter delegates all I/O to a
    :class:`SurrealDBConnection` instance.
    """

    def __init__(self, connection: SurrealDBConnection) -> None:
        self._conn = connection

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SurrealDBRelationalAdapter:
        """Create an adapter from a configuration dictionary.

        Expected keys mirror :class:`SurrealDBConnection.__init__` kwargs:
        ``mode``, ``path``, ``url``, ``namespace``, ``database``, ``user``,
        ``password``.  All are optional and fall back to SurrealDBConnection
        defaults.
        """
        from pydantic import SecretStr

        password = config.get("password", "root")
        if isinstance(password, SecretStr):
            password = password.get_secret_value()
        conn = SurrealDBConnection(
            mode=config.get("mode", "memory"),
            path=config.get("path"),
            url=config.get("url"),
            namespace=config.get("namespace", "khora"),
            database=config.get("database", "default"),
            user=config.get("user", "root"),
            password=password,
        )
        return cls(connection=conn)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    async def create_tables(self) -> None:
        """Create SurrealDB tables and indexes (idempotent).

        Schema is also auto-initialized on connect(), so this is
        safe to call multiple times.
        """
        from .schema import initialize_schema

        await initialize_schema(self._conn)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to SurrealDB."""
        await self._conn.connect()

    async def disconnect(self) -> None:
        """Close the SurrealDB connection."""
        await self._conn.disconnect()

    async def is_healthy(self) -> bool:
        """Delegate health check to the connection."""
        return await self._conn.is_healthy()

    # ------------------------------------------------------------------
    # Namespace operations
    # ------------------------------------------------------------------

    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        """Resolve a namespace identifier to the ID used by chunk/entity records.

        In SurrealDB, chunks and entities store namespace references using the
        stable ``namespace_id`` (not the row-level ``id``).  This method
        validates that an active namespace exists and returns the stable
        ``namespace_id`` so that search filters match stored data.
        """
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT id, namespace_id FROM memory_namespace "
            "WHERE (namespace_id = $ns OR id = $rid) AND is_active = true "
            "LIMIT 1",
            {"ns": ns_str, "rid": _record_id("memory_namespace", namespace_id)},
        )
        if row is not None:
            # Return the stable namespace_id — this is what chunks/entities
            # use as their namespace record reference.
            return UUID(row["namespace_id"])
        raise ValueError(f"No active namespace found for namespace_id or id={namespace_id}")

    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Create a new memory namespace record."""
        rid = _record_id("memory_namespace", namespace.id)
        now_iso = namespace.created_at
        upd_iso = namespace.updated_at

        row = await self._conn.query_one(
            "CREATE $rid SET "
            "namespace_id = $namespace_id, "
            "tenancy_mode = $tenancy_mode, "
            "version = $version, "
            "is_active = $is_active, "
            "config_overrides = $config_overrides, "
            "sync_checkpoints = $sync_checkpoints, "
            "metadata_ = $metadata_, "
            "created_at = $created_at, "
            "updated_at = $updated_at",
            {
                "rid": rid,
                "namespace_id": str(namespace.namespace_id),
                "tenancy_mode": (
                    namespace.tenancy_mode.value
                    if isinstance(namespace.tenancy_mode, TenancyMode)
                    else namespace.tenancy_mode
                ),
                "version": namespace.version,
                "is_active": namespace.is_active,
                "config_overrides": namespace.config_overrides or {},
                "sync_checkpoints": namespace.sync_checkpoints or {},
                "metadata_": namespace.metadata or {},
                "created_at": now_iso,
                "updated_at": upd_iso,
            },
        )
        if row is None:
            raise RuntimeError(f"Failed to create namespace {namespace.id}")
        return self._row_to_namespace(row)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by its row-level id."""
        rid = _record_id("memory_namespace", namespace_id)
        row = await self._conn.query_one(
            "SELECT * FROM $rid",
            {"rid": rid},
        )
        if row is None:
            return None
        return self._row_to_namespace(row)

    async def list_namespaces(
        self,
        *,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> PaginatedResult[MemoryNamespace]:
        """List namespaces with pagination."""
        where = "WHERE is_active = true" if active_only else ""

        count_row = await self._conn.query_one(
            f"SELECT count() AS total FROM memory_namespace {where} GROUP ALL",  # noqa: S608
        )
        total = count_row["total"] if count_row else 0

        rows = await self._conn.query(
            f"SELECT * FROM memory_namespace {where} ORDER BY id ASC LIMIT $lim START $off",  # noqa: S608
            {"lim": limit, "off": offset},
        )
        items = [self._row_to_namespace(r) for r in rows]
        return PaginatedResult(items=items, total=total, limit=limit, offset=offset)

    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Update mutable namespace fields."""
        rid = _record_id("memory_namespace", namespace.id)
        await self._conn.execute(
            "UPDATE $rid SET "
            "version = $version, "
            "is_active = $is_active, "
            "config_overrides = $config_overrides, "
            "sync_checkpoints = $sync_checkpoints, "
            "metadata_ = $metadata_, "
            "updated_at = $updated_at",
            {
                "rid": rid,
                "version": namespace.version,
                "is_active": namespace.is_active,
                "config_overrides": namespace.config_overrides or {},
                "sync_checkpoints": namespace.sync_checkpoints or {},
                "metadata_": namespace.metadata or {},
                "updated_at": datetime.now(UTC),
            },
        )
        return namespace

    async def create_namespace_version(
        self,
        *,
        previous_version: MemoryNamespace | None = None,
    ) -> MemoryNamespace:
        """Create a new version, deactivating the previous one."""
        new_version = 1

        if previous_version:
            new_version = previous_version.version + 1
            await self.deactivate_namespace(previous_version.id)

        namespace = MemoryNamespace(
            id=uuid4(),
            namespace_id=previous_version.namespace_id if previous_version else uuid4(),
            version=new_version,
            is_active=True,
            config_overrides=previous_version.config_overrides if previous_version else {},
            metadata=previous_version.metadata if previous_version else {},
        )
        return await self.create_namespace(namespace)

    async def deactivate_namespace(self, namespace_id: UUID) -> None:
        """Mark a namespace version as inactive."""
        rid = _record_id("memory_namespace", namespace_id)
        await self._conn.execute(
            "UPDATE $rid SET is_active = false, updated_at = $updated_at",
            {"rid": rid, "updated_at": datetime.now(UTC)},
        )
        logger.info(f"Deactivated namespace {namespace_id}")

    # -- namespace row → domain model --

    def _row_to_namespace(self, row: dict[str, Any]) -> MemoryNamespace:
        tenancy_raw = row.get("tenancy_mode", "shared")
        return MemoryNamespace(
            id=_parse_uuid(row["id"]),
            namespace_id=UUID(row["namespace_id"]) if isinstance(row["namespace_id"], str) else row["namespace_id"],
            tenancy_mode=TenancyMode(tenancy_raw) if isinstance(tenancy_raw, str) else tenancy_raw,
            version=row.get("version", 1),
            is_active=row.get("is_active", True),
            config_overrides=row.get("config_overrides") or {},
            sync_checkpoints=row.get("sync_checkpoints") or {},
            metadata=row.get("metadata_") or {},
            created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
            updated_at=_parse_dt(row.get("updated_at")) or datetime.now(UTC),
        )

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    async def create_document(self, document: Document) -> Document:
        """Create a new document record."""
        rid = _record_id("document", document.id)
        row = await self._conn.query_one(
            "CREATE $rid SET "
            "namespace_id = $namespace_id, "
            "content = $content, "
            "status = $status, "
            "source = $source, "
            "source_type = $source_type, "
            "source_name = $source_name, "
            "source_url = $source_url, "
            "content_type = $content_type, "
            "title = $title, "
            "author = $author, "
            "language = $language, "
            "checksum = $checksum, "
            "size_bytes = $size_bytes, "
            "metadata_ = $metadata_, "
            "chunk_count = $chunk_count, "
            "entity_count = $entity_count, "
            "relationship_count = $relationship_count, "
            "error_message = $error_message, "
            "extraction_config_hash = $extraction_config_hash, "
            "extraction_params = $extraction_params, "
            "external_id = $external_id, "
            "created_at = $created_at, "
            "updated_at = $updated_at, "
            "processed_at = $processed_at, "
            "source_timestamp = $source_timestamp, "
            "session_id = $session_id",
            {
                "rid": rid,
                "namespace_id": str(document.namespace_id),
                "content": document.content,
                "status": document.status.value if isinstance(document.status, DocumentStatus) else document.status,
                "source": document.source,
                "source_type": document.source_type,
                "source_name": document.source_name,
                "source_url": document.source_url,
                "content_type": document.content_type,
                "title": document.title,
                "author": document.author,
                "language": document.language,
                "checksum": document.checksum,
                "size_bytes": document.size_bytes,
                "metadata_": document.metadata or {},
                "chunk_count": document.chunk_count,
                "entity_count": document.entity_count,
                "relationship_count": document.relationship_count,
                "error_message": document.error_message,
                "extraction_config_hash": document.extraction_config_hash,
                "extraction_params": document.extraction_params,
                "external_id": document.external_id,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "processed_at": document.processed_at,
                "source_timestamp": document.source_timestamp,
                "session_id": str(document.session_id) if document.session_id else None,
            },
        )
        if row is None:
            raise RuntimeError(f"Failed to create document {document.id}")
        return self._row_to_document(row)

    async def get_document(self, document_id: UUID, *, namespace_id: UUID) -> Document | None:
        """Get a document by ID, scoped to ``namespace_id``.

        Returns ``None`` if the document does not exist OR belongs to a
        different namespace.  ``RecordID`` lookup is not namespace-scoped on
        its own, so we filter explicitly on the document's ``namespace_id``
        column to prevent cross-tenant IDOR (IDOR family).
        """
        rid = _record_id("document", document_id)
        row = await self._conn.query_one(
            "SELECT * FROM $rid WHERE namespace_id = $ns",
            {"rid": rid, "ns": str(namespace_id)},
        )
        if row is None:
            return None
        return self._row_to_document(row)

    async def list_documents(
        self,
        namespace_id: UUID,
        *,
        status: str | None = None,
        updated_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents in a namespace, newest first, ties broken by descending id."""
        ns_str = str(namespace_id)
        conditions = ["namespace_id = $ns"]
        params: dict = {"ns": ns_str, "lim": limit, "off": offset}
        if status:
            conditions.append("status = $status")
            params["status"] = status
        if updated_before is not None:
            conditions.append("updated_at < $updated_before")
            params["updated_before"] = updated_before
        where = " AND ".join(conditions)
        rows = await self._conn.query(
            f"SELECT * FROM document WHERE {where} ORDER BY created_at DESC, id DESC LIMIT $lim START $off",  # noqa: S608
            params,
        )
        return [self._row_to_document(r) for r in rows]

    async def scan_documents(
        self,
        namespace_id: UUID,
        *,
        filter_ast: FilterNode | None = None,
        status: str | None = None,
        updated_before: datetime | None = None,
        after: DocumentScanKey | None = None,
        scan_limit: int = 100,
    ) -> DocumentScanStep:
        """Scan one bounded window of a namespace's documents by keyset.

        ``@internal``. Not part of the public storage API — the offset-based
        :meth:`list_documents` is. One or two ``SELECT``s on this adapter's own
        connection; a walk is the caller's job, chaining
        :attr:`DocumentScanStep.last_scanned` back in as ``after`` until the step
        reports ``exhausted``. No transaction spans steps and no consistent
        snapshot is claimed.

        **A resumed step is TWO statements, not one.** SurrealQL has no row-value
        comparison, so the keyset bound ``(created_at, id) < (…)`` has to be
        written out — and written as one predicate it needs an ``OR``, which is
        exactly what this store's planner cannot take (see the complexity note
        below). It is issued as two ``OR``-free statements instead, both carrying
        the identical narrowing from ``_documents_where``:

        * **Q1, the tie block** — ``created_at = $ts AND id < $id``, ordered
          ``id DESC``. Every row it returns sits at the cursor instant.
        * **Q2, strictly older** — ``created_at < $ts``, ordered
          ``created_at DESC, id DESC``.

        Their concatenation *is* the merged window, in order, with no comparator
        and no dedup: under ``created_at DESC, id DESC`` every Q1 row (equal to
        ``$ts``) sorts strictly before every Q2 row (below ``$ts``), and the two
        sets are disjoint by the same equality. The result is byte-identical to
        the single disjunctive window it replaces. Q2 runs with
        ``LIMIT scan_limit - len(Q1)`` and is skipped outright when Q1 already
        filled the window, so a step resumed deep inside a tie block costs one
        statement rather than two. The **first** step (``after is None``) has no
        tie block at all and is the single Q2 shape without the cursor bound —
        unchanged from before.

        **No transaction spans the two statements, and none is claimed.** A write
        landing between Q1 and Q2 can change what Q2 sees. Contractually that
        changes nothing: this method already promises no consistent snapshot, no
        transaction spans steps, and a concurrent insert lands *above* the
        descending cursor — in the region the walk has already passed and never
        revisits. The observable difference against the old single statement is
        the width of that race, not its existence.

        **Every conjunct is parenthesized unless it is a single comparison.** With
        the keyset disjunction gone, the compiled-fragment splice in
        ``_documents_where`` is the only remaining site, and it is a tripwire
        rather than a live hazard (``compile_surrealdb`` self-groups every boolean
        node today) — measured at 6 rows becoming 12 if it ever stopped.
        ``created_at`` is a non-optional ``TYPE datetime`` (``schema.py``), so the
        engine rejects ``NONE`` by type and the keyset needs no null guard.

        Cursor and bound operands bind as native objects — ``datetime`` for the
        timestamps, ``RecordID`` for the id — matching the write path and
        :meth:`list_documents`. See the datetime-binds note at the top of this
        module for why the ISO-string form would ship an ``updated_before``
        returning an empty walk that reports itself exhausted on the first call.
        The narrowing is otherwise the shape the raw-SQL siblings ship; its
        NULL-``updated_at`` exclusion does not arise, the field being
        non-optional here.

        ``filter_ast`` is compiled by the compiler registered for this store's
        ``documents`` target. Only the leaves in ``consumed_keys`` were pushed;
        **the caller must still evaluate the full filter over the returned
        rows.** Re-running a pushed leaf can only narrow, because everything
        pushed is a superset filter — which is why resuming past the rows the
        pushdown rejected skips nothing. That belongs to the compiler and its
        compile context, not to this method.

        ``scan_limit`` bounds rows **returned**, not rows **examined**: a
        selective fragment can still make one call read the whole namespace, so
        it is no latency bound. Because ``last_scanned`` is the last *raw* row, a
        resumed step never re-*returns* the rejected gap.

        **A step is namespace-scoped, but it is not O(scan_limit).** Both
        statements plan ``Iterate Index`` on ``idx_document_ns_created``
        (``(namespace_id, created_at)``), so a step examines the namespace, not
        the table: no other tenant's rows are read, and cost no longer tracks
        total corpus size. What the index does *not* buy is a bounded step —
        SurrealDB 2.x still materializes and sorts the qualifying set in memory
        before applying ``LIMIT`` (see :mod:`.schema`), so a step is
        O(rows-below-the-cursor-in-this-namespace), and a full walk is quadratic
        in namespace size. ``scan_limit`` bounds rows returned, never rows
        examined.

        The ``OR``-freedom is the whole mechanism and it is fragile: on this
        table an ``OR`` anywhere in the ``WHERE``, nested inside a conjunct
        included, collapses the plan to ``Iterate Table`` and takes the
        ``namespace_id`` prefix down with it — a cross-tenant scan. The note at
        the top of this module says exactly when a disjunction can be indexed and
        why ``document`` never qualifies. Two near-miss workarounds were measured
        and do **not** restore the index, so do not re-try them: a
        ``WITH INDEX`` hint still plans ``Iterate Table``, and so does a redundant
        ``created_at <= $ts`` range guard bracketing a disjunction. Nor does a
        pure ``created_at <= $ts`` range with the tie resolved client-side: that
        is index-eligible but re-reads the whole tie block on every step, which is
        quadratic inside one.

        **Out of scope, and still true:** a compiled filter fragment containing a
        ``$or`` re-introduces a top-level ``OR`` and table-scans — including on
        the very first step, which carries no cursor. Nothing here can prevent
        that; whether an ``$or`` leaf is pushed down or deferred to the caller's
        post-filter belongs to the compiler and its compile context. The
        EXPLAIN-pinning tests therefore cover the no-filter shapes only.

        The raw-SQLite sibling has no such problem — its row-value keyset is
        consumed as an index range constraint, so there the total-work bound
        holds as stated.

        **Two record-shape preconditions, both satisfied by khora's own writes.**
        They matter because SurrealDB is writable directly, outside khora, and a
        user who does so could otherwise break a walk that has no way to notice.
        The first is now ENFORCED (``_scan_key_from_row`` raises); the second still
        cannot be, and is documented so a caller knows to avoid it.

        First, every ``document`` record id must be an ``Id::Uuid``. khora writes
        them all through ``_record_id``, so this holds. A row created directly with
        a string id (``document:'abc'``) is now rejected up front:
        ``_scan_key_from_row`` parses the raw id with ``_parse_uuid(strict=True)``
        and raises ``ValueError`` rather than seating a cursor. **The failure it
        prevents was not the one it looks like.** ``id < $rid`` and
        ``ORDER BY id DESC`` do NOT disagree — measured on a table of 4 uuid-id and
        4 string-id rows, the set below the cursor by compare matched the set below
        it by sort at all 8 resume positions. The break was in the round trip
        instead: without the guard, ``_parse_uuid`` derives a ``uuid5`` from the
        string id and ``_record_id`` turns that back into a record id no row holds,
        sitting nowhere near the original's position — measured over those 8 rows, a
        walk at ``scan_limit=1`` yielded 5 distinct documents in a cycle, never
        terminated (stopped at 60 steps), and never reached 3 of the 8 at all.
        Strict parsing converts that silent cycle into a loud ``ValueError``.

        Second, ``created_at`` must not carry SUB-MICROSECOND precision. The engine
        stores nanoseconds; the Python SDK truncates to microseconds on read. So a
        cursor read back from a row stored at ``.0000015`` is ``.000001`` — below
        the row's own stored instant — and every row between the two is invisible
        to both statements at once: not Q2's ``created_at < $ts`` (they are above
        it), not Q1's ``created_at = $ts`` (they are not equal to it). The split
        neither introduces nor repairs this; it is the same two comparisons in two
        statements instead of two disjuncts. Measured on three rows at
        ``.0000015`` / ``.0000012`` / ``.000001``, resuming from the first returned
        an EMPTY window: the walk reports itself exhausted and silently drops the
        entire remainder, rather than dropping one tie-mate. Unreachable through
        khora's writes, which bind Python ``datetime`` objects and are therefore
        microsecond-precise at the source.

        Raises ``ValueError`` when ``scan_limit`` is below 1, before anything is
        compiled or executed, so a bad bound is never masked by a compile error;
        also when a compiled filter's binds collide with this method's own (see
        ``_documents_where``), and when the final raw row carries no readable
        ``created_at`` to resume from.

        An unrenderable metadata path segment does NOT raise here. Under this
        store's ``on_unsupported="split"`` context it is a capability gap the
        compiler defers: the leaf stays out of ``consumed_keys`` and reaches the
        caller's post-filter, which handles a hyphenated key correctly. That holds
        in every position — conjunctive as well as inside a deferred ``$or`` —
        so it is no longer sibling-dependent.
        """
        if scan_limit < 1:
            raise ValueError(f"scan_limit must be >= 1, got {scan_limit}")

        conditions, params, consumed_keys = _documents_where(
            namespace_id,
            status=status,
            updated_before=updated_before,
            filter_ast=filter_ast,
        )
        where = " AND ".join(conditions)

        rows: list[dict[str, Any]] = []
        remaining = scan_limit
        if after is not None:
            cursor_created_at, cursor_id = after
            # Q1 — the tie block at the cursor instant. ``ORDER BY id DESC`` is
            # the whole sort: ``created_at`` is pinned by equality, so ordering on
            # it too would be dead weight.
            rows = await self._conn.query(
                f"SELECT * FROM document WHERE {where} "  # noqa: S608
                "AND created_at = $after_created_at AND id < $after_id "
                "ORDER BY id DESC LIMIT $lim",
                {
                    **params,
                    "after_created_at": cursor_created_at,
                    "after_id": _record_id("document", cursor_id),
                    "lim": scan_limit,
                },
            )
            remaining = scan_limit - len(rows)

        if remaining > 0:
            # Q2 — strictly below the cursor instant, or the whole namespace when
            # there is no cursor. Skipped entirely when Q1 already filled the
            # window, so a step resumed inside a large tie block costs one
            # statement. Its own ``LIMIT`` is the shortfall, not ``scan_limit``,
            # which is what makes ``len(Q1) + len(Q2) <= scan_limit`` hold by
            # arithmetic — the concatenation needs no trailing slice.
            older_where = where if after is None else f"{where} AND created_at < $after_created_at"
            older_params: dict[str, Any] = {**params, "lim": remaining}
            if after is not None:
                older_params["after_created_at"] = after[0]
            rows = rows + await self._conn.query(
                f"SELECT * FROM document WHERE {older_where} "  # noqa: S608
                "ORDER BY created_at DESC, id DESC LIMIT $lim",
                older_params,
            )

        # ``last_scanned`` and ``exhausted`` both describe the RAW window — now
        # the MERGED one: the final row scanned across both statements, and
        # whether the pair together ran out of rows filling ``scan_limit``.
        # ``exhausted`` must be read off the merge and not off either leg: Q1
        # returning fewer rows than asked is the normal case (a tie block is
        # usually one row), and calling that exhausted would end every walk after
        # its first resumed step.
        # Neither may be derived from a post-filtered subset — that would re-scan
        # the rejected gap on resume, and would call a full window exhausted. The
        # key comes from ``rows[-1]``, not from ``docs[-1]``: ``_row_to_document``
        # masks a missing ``created_at`` with ``now()``, which as a cursor sorts
        # above the whole window and makes the walk loop. See ``_scan_key_from_row``
        # (it does the ``RecordID`` -> ``UUID`` conversion ``DocumentScanKey``
        # requires, so the raw ``id`` never seats a ``RecordID`` in the tuple).
        docs = [self._row_to_document(r) for r in rows]
        return build_scan_step(
            documents=docs,
            last_scanned=_scan_key_from_row(rows[-1]) if rows else None,
            raw_row_count=len(rows),
            scan_limit=scan_limit,
            consumed_keys=consumed_keys,
        )

    async def claim_orphaned_documents(
        self,
        namespace_id: UUID,
        *,
        pending_before: datetime,
        processing_before: datetime,
        limit: int = 100,
    ) -> list[Document]:
        """Claim stale orphaned documents (no row locking - SurrealDB plain claim).

        **Two statements, one per status/cutoff pair, merged client-side.** As a
        single predicate the two pairs need a top-level ``OR``, which costs the
        statement its indexes on this table, the ``namespace_id`` prefix included
        (see the disjunction note at the top of this module) — the sweep read
        every ``document`` row in the database, all namespaces included. Each leg
        alone plans ``Iterate Index`` on ``idx_document_ns_status``.

        The legs are disjoint by ``status``, so the union is exactly the original
        row set. Unlike the keyset split in :meth:`scan_documents`, the
        concatenation is **not** already ordered — either leg can supply any
        position in ``updated_at`` order — so the merge re-sorts before applying
        ``limit``. Both legs must therefore fetch the full ``limit``: either one
        alone can own the whole claim. ``updated_at`` cannot be ``NONE`` on a
        returned row, the leg predicates having compared it.

        Rows returned, and their ``updated_at`` order, are unchanged. What *can*
        differ from the single-statement form is which of several rows sharing one
        ``updated_at`` instant lands inside ``limit`` — measured on a seeded
        namespace, the two forms returned the same timestamp multiset with one id
        swapped. The old ``ORDER BY updated_at`` had no secondary key either, so
        that choice was already arbitrary and is not a contract; the merge's
        stable sort now makes it at least deterministic, PENDING before
        PROCESSING.

        **The id de-duplication in the merge is not defensive padding — the split
        created the race it closes.** At any single instant the legs are disjoint,
        a row being either PENDING or PROCESSING and never both, so a reader
        checking the predicates concludes no row can arrive twice. That holds for
        one statement and not for two: nothing spans them, so a *concurrent*
        claimer that flips a row PENDING -> PROCESSING between them makes the
        first leg return it as pending and the second return the same row as
        processing. Reproduced deterministically by interleaving that flip: a
        namespace holding one document returned it twice, claiming it twice and
        spending two of ``limit`` on one row. Keyed on the record id and
        first-wins, so the row keeps the earlier leg's ``orphan_prior_status`` —
        the status it actually held when this call observed it.
        """
        ns_str = str(namespace_id)
        legs = (
            (DocumentStatus.PENDING.value, pending_before),
            (DocumentStatus.PROCESSING.value, processing_before),
        )
        by_id: dict[str, dict[str, Any]] = {}
        for status_value, cutoff in legs:
            for row in await self._conn.query(
                "SELECT * FROM document WHERE namespace_id = $ns "
                "AND status = $status AND updated_at < $cutoff "
                "ORDER BY updated_at LIMIT $lim",
                {"ns": ns_str, "status": status_value, "cutoff": cutoff, "lim": limit},
            ):
                by_id.setdefault(str(row["id"]), row)
        rows = sorted(by_id.values(), key=lambda r: _parse_dt(r["updated_at"]) or datetime.max.replace(tzinfo=UTC))
        docs = [self._row_to_document(r) for r in rows[:limit]]
        if not docs:
            return []
        # Both cutoffs above and this stamp bind as ``datetime`` objects. As ISO
        # strings they pinned the SELECT's two ``<`` compares false, so it
        # returned nothing and this UPDATE was never reached. It could not have
        # stored a string had it run: SurrealDB rejects a string into a ``TYPE
        # datetime`` field rather than coercing it. No existing row can hold a
        # string ``updated_at``, so there is nothing to repair; do not go looking.
        now = datetime.now(UTC)
        for doc in docs:
            doc.orphan_prior_status = doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status
            await self._conn.query(
                "UPDATE $rid SET status = $status, updated_at = $updated_at",
                {
                    "rid": _record_id("document", doc.id),
                    "status": DocumentStatus.PROCESSING.value,
                    "updated_at": now,
                },
            )
            doc.status = DocumentStatus.PROCESSING
            # Mirror the stamp the UPDATE just wrote. Without it a returned doc
            # reads PROCESSING with its pre-claim ``updated_at`` — an object
            # that matches no stored row. The SQLAlchemy adapter refreshes the
            # instances it returns; this is the equivalent for this store.
            doc.updated_at = now
        return docs

    async def update_document(self, document: Document) -> Document:
        """Update a document's mutable fields."""
        rid = _record_id("document", document.id)
        await self._conn.execute(
            "UPDATE $rid SET "
            "content = $content, "
            "status = $status, "
            "source = $source, "
            "source_type = $source_type, "
            "source_name = $source_name, "
            "source_url = $source_url, "
            "content_type = $content_type, "
            "title = $title, "
            "author = $author, "
            "language = $language, "
            "checksum = $checksum, "
            "size_bytes = $size_bytes, "
            "metadata_ = $metadata_, "
            "chunk_count = $chunk_count, "
            "entity_count = $entity_count, "
            "relationship_count = $relationship_count, "
            "error_message = $error_message, "
            "extraction_config_hash = $extraction_config_hash, "
            "extraction_params = $extraction_params, "
            "external_id = $external_id, "
            "updated_at = $updated_at, "
            "processed_at = $processed_at, "
            "source_timestamp = $source_timestamp, "
            "session_id = $session_id",
            {
                "rid": rid,
                "content": document.content,
                "status": document.status.value if isinstance(document.status, DocumentStatus) else document.status,
                "source": document.source,
                "source_type": document.source_type,
                "source_name": document.source_name,
                "source_url": document.source_url,
                "content_type": document.content_type,
                "title": document.title,
                "author": document.author,
                "language": document.language,
                "checksum": document.checksum,
                "size_bytes": document.size_bytes,
                "metadata_": document.metadata or {},
                "chunk_count": document.chunk_count,
                "entity_count": document.entity_count,
                "relationship_count": document.relationship_count,
                "error_message": document.error_message,
                "extraction_config_hash": document.extraction_config_hash,
                "extraction_params": document.extraction_params,
                "external_id": document.external_id,
                "updated_at": datetime.now(UTC),
                "processed_at": document.processed_at,
                "source_timestamp": document.source_timestamp,
                "session_id": str(document.session_id) if document.session_id else None,
            },
        )
        return document

    async def delete_document(self, document_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a document, scoped to ``namespace_id`` (IDOR family).

        Returns ``False`` if the document does not exist OR belongs to a
        different namespace.  ``RecordID`` deletion alone is not namespace-
        scoped, so we filter on ``namespace_id`` to prevent cross-tenant
        deletion by id.
        """
        rid = _record_id("document", document_id)
        deleted = await self._conn.query(
            "DELETE $rid WHERE namespace_id = $ns RETURN BEFORE",
            {"rid": rid, "ns": str(namespace_id)},
        )
        return bool(deleted)

    async def count_documents(self, namespace_id: UUID) -> int:
        """Count documents in a namespace."""
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT count() AS cnt FROM document WHERE namespace_id = $ns GROUP ALL",
            {"ns": ns_str},
        )
        return (row["cnt"] or 0) if row else 0

    async def get_last_activity_at(self, namespace_id: UUID) -> datetime | None:
        """Get the most recent document creation timestamp in a namespace."""
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT math::max(created_at) AS latest FROM document WHERE namespace_id = $ns GROUP ALL",
            {"ns": ns_str},
        )
        return row["latest"] if row else None

    async def get_document_stats(self, namespace_id: UUID) -> tuple[int, datetime | None]:
        """Get document count and last activity timestamp in a single query."""
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT count() AS cnt, math::max(created_at) AS latest FROM document WHERE namespace_id = $ns GROUP ALL",
            {"ns": ns_str},
        )
        if not row:
            return 0, None
        return (row["cnt"] or 0), row["latest"]

    async def get_document_by_checksum(
        self, namespace_id: UUID, checksum: str, *, pending_stale_before: datetime | None = None
    ) -> Document | None:
        """Get a document by content checksum within a namespace.

        FAILED documents are always excluded so previously-failed content
        re-ingests. When ``pending_stale_before`` is given, PENDING documents
        older than that cutoff are also excluded so a crash-abandoned
        half-ingest (#1464) re-ingests; fresh PENDING rows stay a dedup hit,
        preserving the concurrent in-flight guard.

        Probes one index-eligible leg at a time (see
        ``_checksum_reingestable_legs``) and stops at the first hit, so the common
        settled-document case costs one statement. The legs are disjoint, so which
        one answers does not change the row set — only which of several rows
        sharing a checksum an unordered ``LIMIT 1`` happens to surface, and that
        was already unordered.
        """
        ns_str = str(namespace_id)
        for where, binds in _checksum_reingestable_legs(pending_stale_before):
            row = await self._conn.query_one(
                f"SELECT * FROM document WHERE namespace_id = $ns AND checksum = $checksum AND {where} LIMIT 1",  # noqa: S608
                {"ns": ns_str, "checksum": checksum, **binds},
            )
            if row is not None:
                return self._row_to_document(row)
        return None

    async def get_document_by_external_id(self, external_id: str | None, *, namespace_id: UUID) -> Document | None:
        """Get a document by (namespace_id, external_id).

        Status is NOT filtered so FAILED rows can self-heal on the next
        successful replace.
        """
        if external_id is None:
            return None
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT * FROM document WHERE namespace_id = $ns AND external_id = $external_id LIMIT 1",
            {"ns": ns_str, "external_id": external_id},
        )
        if row is None:
            return None
        return self._row_to_document(row)

    async def get_documents_by_checksums(
        self, namespace_id: UUID, checksums: list[str], *, pending_stale_before: datetime | None = None
    ) -> dict[str, Document]:
        """Fetch documents by content checksums in a single query.

        Used for batch deduplication to avoid N serial DB queries.
        FAILED documents are always excluded so previously-failed content
        re-ingests. When ``pending_stale_before`` is given, PENDING documents
        older than that cutoff are also excluded so a crash-abandoned
        half-ingest (#1464) re-ingests; fresh PENDING rows stay a dedup hit,
        preserving the concurrent in-flight guard.

        Args:
            namespace_id: Namespace to search in
            checksums: List of content checksums to look up
            pending_stale_before: Cutoff for reclaiming stale PENDING half-ingests

        One index-eligible statement per leg (see
        ``_checksum_reingestable_legs``), accumulated into one mapping. Unlike the
        single-row probe, both legs always run: a batch needs every checksum
        answered, and the two legs cover different rows.

        **First write wins, and that is what keeps this method agreeing with**
        :meth:`get_document_by_checksum`. The legs are disjoint by ``status``, but
        the *checksums* they return are not: ``(namespace_id, checksum)`` carries
        no unique constraint, so one checksum can have both a settled row (leg A)
        and a fresh-PENDING row (leg B). Under plain last-wins the later leg would
        take it — handing the batch caller the in-flight row while the single-row
        probe, which stops at its first hit, hands back the settled one. Two dedup
        entry points disagreeing about the same checksum is the kind of split that
        surfaces as a phantom duplicate ingest much later, so both resolve it the
        same way: leg order is precedence, settled ahead of in-flight.

        Returns:
            Dictionary mapping checksum to Document (only for existing documents)
        """
        if not checksums:
            return {}
        ns_str = str(namespace_id)
        result: dict[str, Document] = {}
        for where, binds in _checksum_reingestable_legs(pending_stale_before):
            rows = await self._conn.query(
                f"SELECT * FROM document WHERE namespace_id = $ns AND checksum IN $checksums AND {where}",  # noqa: S608
                {"ns": ns_str, "checksums": checksums, **binds},
            )
            for r in rows:
                cs = r.get("checksum", "")
                if cs and cs not in result:
                    result[cs] = self._row_to_document(r)
        return result

    async def get_documents_batch(self, document_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query, scoped to ``namespace_id``.

        Documents belonging to a different namespace are silently dropped
        from the result to prevent cross-tenant IDOR (IDOR family).
        """
        if not document_ids:
            return {}
        id_strs = [_record_id("document", uid) for uid in document_ids]
        rows = await self._conn.query(
            "SELECT * FROM document WHERE id IN $ids AND namespace_id = $ns",
            {"ids": id_strs, "ns": str(namespace_id)},
        )
        return {_parse_uuid(r["id"]): self._row_to_document(r) for r in rows}

    async def get_documents_by_external_ids(
        self, external_ids: list[str], *, namespace_id: UUID
    ) -> dict[str, Document]:
        """Batch lookup by ``(namespace_id, external_id)``. Status-agnostic."""
        filtered = [e for e in external_ids if e]
        if not filtered:
            return {}
        ns_str = str(namespace_id)
        rows = await self._conn.query(
            "SELECT * FROM document WHERE namespace_id = $ns AND external_id IN $external_ids",
            {"ns": ns_str, "external_ids": filtered},
        )
        result: dict[str, Document] = {}
        for r in rows:
            ext = r.get("external_id")
            if ext:
                result[ext] = self._row_to_document(r)
        return result

    async def get_document_sources_batch(
        self, document_ids: list[UUID], *, namespace_id: UUID
    ) -> dict[UUID, DocumentSource]:
        """Fetch lightweight document metadata for source attribution,
        scoped to ``namespace_id``.

        Documents belonging to a different namespace are silently dropped
        from the result to prevent cross-tenant IDOR (IDOR family).
        """
        if not document_ids:
            return {}
        id_strs = [_record_id("document", uid) for uid in document_ids]
        rows = await self._conn.query(
            "SELECT id, title, source, source_type, created_at, source_timestamp "
            "FROM document WHERE id IN $ids AND namespace_id = $ns",
            {"ids": id_strs, "ns": str(namespace_id)},
        )
        result: dict[UUID, DocumentSource] = {}
        for r in rows:
            uid = _parse_uuid(r["id"])
            result[uid] = DocumentSource(
                id=uid,
                title=r.get("title", ""),
                source=r.get("source", ""),
                source_type=r.get("source_type", ""),
                created_at=_parse_dt(r.get("created_at")),
                source_timestamp=_parse_dt(r.get("source_timestamp")),
            )
        return result

    async def get_document_projections_batch(
        self,
        document_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> dict[UUID, DocumentProjection]:
        """Fetch full DocumentProjection rows for recall responses.

        Filters by ``namespace_id`` at the SurrealQL layer; cross-namespace
        ids are silently dropped (security close-out).
        """
        if not document_ids:
            return {}
        id_strs = [_record_id("document", uid) for uid in document_ids]
        rows = await self._conn.query(
            "SELECT id, created_at, source_type, title, external_id, source, source_name, "
            "source_url, content_type, source_timestamp, metadata_ FROM document "
            "WHERE id IN $ids AND namespace_id = $ns",
            {"ids": id_strs, "ns": str(namespace_id)},
        )
        result: dict[UUID, DocumentProjection] = {}
        for r in rows:
            uid = _parse_uuid(r["id"])
            result[uid] = DocumentProjection(
                id=uid,
                created_at=_parse_dt(r.get("created_at")) or datetime.now(UTC),
                source_type=r.get("source_type") or "library",
                title=_none_if_empty(r.get("title")),
                external_id=_none_if_empty(r.get("external_id")),
                source=_none_if_empty(r.get("source")),
                source_name=_none_if_empty(r.get("source_name")),
                source_url=_none_if_empty(r.get("source_url")),
                content_type=_none_if_empty(r.get("content_type")),
                source_timestamp=_parse_dt(r.get("source_timestamp")),
                metadata=dict(r.get("metadata_") or {}),
            )
        return result

    # -- document row → domain model --

    def _row_to_document(self, row: dict[str, Any]) -> Document:
        status_raw = row.get("status", "pending")
        return Document(
            id=_parse_uuid(row["id"]),
            namespace_id=UUID(row["namespace_id"]) if isinstance(row["namespace_id"], str) else row["namespace_id"],
            content=row.get("content", ""),
            status=DocumentStatus(status_raw) if isinstance(status_raw, str) else status_raw,
            title=_none_if_empty(row.get("title")),
            source=_none_if_empty(row.get("source")),
            source_type=row.get("source_type") or "library",
            source_name=_none_if_empty(row.get("source_name")),
            source_url=_none_if_empty(row.get("source_url")),
            content_type=_none_if_empty(row.get("content_type")),
            author=_none_if_empty(row.get("author")),
            language=_none_if_empty(row.get("language")),
            checksum=_none_if_empty(row.get("checksum")),
            size_bytes=row.get("size_bytes", 0),
            metadata=dict(row.get("metadata_") or {}),
            chunk_count=row.get("chunk_count", 0),
            entity_count=row.get("entity_count", 0),
            relationship_count=row.get("relationship_count", 0),
            error_message=row.get("error_message"),
            extraction_config_hash=row.get("extraction_config_hash"),
            extraction_params=row.get("extraction_params"),
            created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
            updated_at=_parse_dt(row.get("updated_at")) or datetime.now(UTC),
            processed_at=_parse_dt(row.get("processed_at")),
            source_timestamp=_parse_dt(row.get("source_timestamp")),
            external_id=row.get("external_id"),
            session_id=_parse_uuid(row.get("session_id")) if row.get("session_id") else None,
        )

    # ------------------------------------------------------------------
    # Sync checkpoint operations
    # ------------------------------------------------------------------

    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        """Get the last sync checkpoint for a source."""
        ns_str = str(namespace_id)
        row = await self._conn.query_one(
            "SELECT checkpoint FROM sync_checkpoint WHERE namespace_id = $ns AND source = $source LIMIT 1",
            {"ns": ns_str, "source": source},
        )
        if row is None:
            return None
        return row.get("checkpoint")

    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        """Upsert a sync checkpoint for a namespace+source pair.

        Uses SurrealDB's UPSERT with a deterministic record ID derived
        from namespace and source so that repeated calls overwrite rather
        than duplicate.
        """
        ns_str = str(namespace_id)
        # Deterministic record ID avoids duplicates
        upsert_id = f"sync_checkpoint:⟨{ns_str}_{source}⟩"
        await self._conn.execute(
            "UPSERT $rid SET namespace_id = $ns, source = $source, checkpoint = $checkpoint, updated_at = $updated_at",
            {
                "rid": upsert_id,
                "ns": ns_str,
                "source": source,
                "checkpoint": checkpoint,
                "updated_at": datetime.now(UTC),
            },
        )

    # ------------------------------------------------------------------
    # Chronicle engine: events + facts (issue #712)
    #
    # Mirrors the chronicle methods on the pgvector / sqlite_lance
    # relational adapters. ``StorageCoordinator._chronicle_backend``
    # picks self.vector first, then self.relational — the SurrealDB
    # vector adapter does not carry chronicle methods, so dispatch
    # falls through here.
    #
    # Tables ``chronicle_event`` / ``memory_fact`` are defined in
    # schema.py; they are SurrealDB-side mirrors of the pgvector
    # ChronicleEventModel / MemoryFactModel rows. Returned rows are
    # lightweight namespace objects shaped to MemoryFact / ChronicleEvent
    # attribute access (``id``, ``subject``, ``is_active``, etc.) so the
    # chronicle engine's reconciliation path works without changes.
    # ------------------------------------------------------------------

    async def _write_rows_atomic(
        self,
        statements: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Execute a batch of CREATE statements all-or-nothing (issue #1228).

        Each statement uses row-unique parameter names so the whole batch can
        be sent together. Remote (``ws://``) mode wraps the per-row executes in
        :meth:`SurrealDBConnection.transaction` (BEGIN/COMMIT, CANCEL on error).
        Embedded / memory modes -- where a standalone ``BEGIN`` is rejected by
        surrealkv -- send the rows as a single ``BEGIN TRANSACTION; ...;
        COMMIT TRANSACTION;`` multi-statement query via
        :meth:`SurrealDBConnection.execute_batch`, which surrealkv applies
        atomically and rolls back wholesale on any statement error. Either way a
        mid-write failure leaves the table unchanged rather than half-written.
        """
        if not statements:
            return
        if self._conn.supports_transactions:
            async with self._conn.transaction():
                for sql, bindings in statements:
                    await self._conn.execute(sql, bindings)
            return
        # Embedded / memory: a single BEGIN..COMMIT multi-statement query is
        # the only way to get cross-row atomicity (a bare BEGIN raises on
        # surrealkv, so conn.transaction() is a no-op there).
        batch: list[tuple[str, dict[str, Any] | None]] = [("BEGIN TRANSACTION", None)]
        batch.extend(statements)
        batch.append(("COMMIT TRANSACTION", None))
        await self._conn.execute_batch(batch)

    async def write_events(
        self,
        events: list[Any],
        *,
        namespace_id: UUID,
    ) -> list[UUID]:
        """Insert chronicle_event rows atomically; returns IDs in input order."""
        if not events:
            return []
        now = datetime.now(UTC)
        ns_str = str(namespace_id)
        ids: list[UUID] = []
        statements: list[tuple[str, dict[str, Any]]] = []
        for i, ev in enumerate(events):
            ev_id: UUID = getattr(ev, "id", None) or uuid4()
            ids.append(ev_id)
            statements.append(
                (
                    f"CREATE $rid_{i} SET "
                    f"namespace_id = $ns_{i}, "
                    f"chunk_id = $chunk_id_{i}, "
                    f"subject = $subject_{i}, "
                    f"verb = $verb_{i}, "
                    f"object = $object_{i}, "
                    f"observation_date = $observation_date_{i}, "
                    f"referenced_date = $referenced_date_{i}, "
                    f"relative_offset = $relative_offset_{i}, "
                    f"confidence = $confidence_{i}, "
                    f"source_text = $source_text_{i}, "
                    f"embedding = $embedding_{i}, "
                    f"created_at = $created_at_{i}",
                    {
                        f"rid_{i}": _record_id("chronicle_event", ev_id),
                        f"ns_{i}": ns_str,
                        f"chunk_id_{i}": str(ev.chunk_id) if getattr(ev, "chunk_id", None) else None,
                        f"subject_{i}": ev.subject,
                        f"verb_{i}": ev.verb,
                        f"object_{i}": ev.object or None,
                        f"observation_date_{i}": ev.observation_date or now,
                        f"referenced_date_{i}": ev.referenced_date,
                        f"relative_offset_{i}": ev.relative_offset or None,
                        f"confidence_{i}": float(ev.confidence),
                        f"source_text_{i}": ev.source_text or "",
                        f"embedding_{i}": list(ev.embedding) if getattr(ev, "embedding", None) is not None else None,
                        f"created_at_{i}": now,
                    },
                )
            )
        await self._write_rows_atomic(statements)
        return ids

    async def write_facts(
        self,
        facts: list[Any],
        *,
        namespace_id: UUID,
    ) -> list[UUID]:
        """Insert memory_fact rows atomically; returns IDs in input order."""
        if not facts:
            return []
        now = datetime.now(UTC)
        ns_str = str(namespace_id)
        ids: list[UUID] = []
        statements: list[tuple[str, dict[str, Any]]] = []
        for i, f in enumerate(facts):
            fact_id: UUID = getattr(f, "id", None) or uuid4()
            ids.append(fact_id)
            chunk_ids = [str(cid) for cid in (getattr(f, "source_chunk_ids", None) or [])]
            superseded_by = getattr(f, "superseded_by", None)
            statements.append(
                (
                    f"CREATE $rid_{i} SET "
                    f"namespace_id = $ns_{i}, "
                    f"subject = $subject_{i}, "
                    f"predicate = $predicate_{i}, "
                    f"object = $object_{i}, "
                    f"fact_text = $fact_text_{i}, "
                    f"confidence = $confidence_{i}, "
                    f"is_active = $is_active_{i}, "
                    f"superseded_by = $superseded_by_{i}, "
                    f"source_chunk_ids = $source_chunk_ids_{i}, "
                    f"created_at = $created_at_{i}, "
                    f"updated_at = $updated_at_{i}",
                    {
                        f"rid_{i}": _record_id("memory_fact", fact_id),
                        f"ns_{i}": ns_str,
                        f"subject_{i}": f.subject or "",
                        f"predicate_{i}": f.predicate or "",
                        f"object_{i}": f.object_ or "",
                        f"fact_text_{i}": f.fact_text or "",
                        f"confidence_{i}": float(f.confidence),
                        f"is_active_{i}": bool(getattr(f, "is_active", True)),
                        f"superseded_by_{i}": str(superseded_by) if superseded_by else None,
                        f"source_chunk_ids_{i}": chunk_ids,
                        f"created_at_{i}": now,
                        f"updated_at_{i}": now,
                    },
                )
            )
        await self._write_rows_atomic(statements)
        return ids

    async def query_events(
        self,
        namespace_id: UUID,
        *,
        subject: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[Any]:
        """Query chronicle_event filtered by subject and referenced_date range."""
        ns_str = str(namespace_id)
        conditions = ["namespace_id = $ns"]
        params: dict[str, Any] = {"ns": ns_str, "lim": limit}
        if subject is not None:
            conditions.append("subject = $subject")
            params["subject"] = subject
        if since is not None:
            conditions.append("referenced_date >= $since")
            params["since"] = since
        if until is not None:
            conditions.append("referenced_date <= $until")
            params["until"] = until
        where = " AND ".join(conditions)
        rows = await self._conn.query(
            f"SELECT * FROM chronicle_event WHERE {where} ORDER BY referenced_date DESC LIMIT $lim",  # noqa: S608
            params,
        )
        return [_row_to_chronicle_event(r) for r in rows]

    async def query_active_facts_for_subject(
        self,
        namespace_id: UUID,
        subject: str,
    ) -> list[Any]:
        """Return all active (not superseded) memory facts for a subject."""
        ns_str = str(namespace_id)
        rows = await self._conn.query(
            "SELECT * FROM memory_fact "
            "WHERE namespace_id = $ns AND subject = $subject AND is_active = true "
            "ORDER BY created_at DESC",
            {"ns": ns_str, "subject": subject},
        )
        return [_row_to_memory_fact(r) for r in rows]

    async def supersede_fact(self, fact_id: UUID, superseded_by: UUID, *, namespace_id: UUID) -> None:
        """Mark a fact inactive and link it to its replacement.

        Scoped to ``namespace_id`` (IDOR family) — no-op when the fact belongs
        to a different namespace.
        """
        await self._conn.execute(
            "UPDATE $rid SET is_active = false, superseded_by = $superseded_by, updated_at = $updated_at "
            "WHERE namespace_id = $ns",
            {
                "rid": _record_id("memory_fact", fact_id),
                "superseded_by": str(superseded_by),
                "updated_at": datetime.now(UTC),
                "ns": str(namespace_id),
            },
        )

    async def delete_facts_for_chunks(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> int:
        """Hard-delete memory_fact rows referencing any of ``chunk_ids`` (#1140).

        Forget-cascade cleanup: memory facts carry chunk provenance only in
        the ``source_chunk_ids`` array, so document deletion never cascades
        to them. Scoped to ``namespace_id`` (IDOR family). Returns the
        number of facts deleted.
        """
        if not chunk_ids:
            return 0
        deleted = await self._conn.query(
            "DELETE memory_fact WHERE namespace_id = $ns AND source_chunk_ids CONTAINSANY $chunks RETURN BEFORE",
            {
                "ns": str(namespace_id),
                "chunks": [str(cid) for cid in chunk_ids],
            },
        )
        return len(deleted or [])

    # ------------------------------------------------------------------
    # SQLAlchemy compatibility shim
    # ------------------------------------------------------------------

    def _get_session(self) -> None:
        """No-op — SurrealDB does not use SQLAlchemy sessions."""
        return None


# ---------------------------------------------------------------------------
# Row → dataclass helpers for chronicle methods (issue #712)
#
# Returned objects only need to support attribute access (``row.id``,
# ``row.subject``, ``row.is_active`` etc.) — the chronicle engine consumes
# the rows via ``getattr``. A lightweight type with the same surface as
# ``ChronicleEvent`` / ``MemoryFact`` keeps us from importing
# litellm-heavy modules just to construct the dataclasses.
# ---------------------------------------------------------------------------


class _ChronicleEventRow:
    """Minimal attribute container shaped like ``ChronicleEvent``."""

    __slots__ = (
        "id",
        "namespace_id",
        "chunk_id",
        "subject",
        "verb",
        "object",
        "observation_date",
        "referenced_date",
        "relative_offset",
        "confidence",
        "source_text",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))


class _MemoryFactRow:
    """Minimal attribute container shaped like ``MemoryFact``."""

    __slots__ = (
        "id",
        "namespace_id",
        "subject",
        "predicate",
        "object_",
        "fact_text",
        "confidence",
        "is_active",
        "superseded_by",
        "source_chunk_ids",
        "created_at",
        "updated_at",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs.get(key))


def _row_to_chronicle_event(row: dict[str, Any]) -> _ChronicleEventRow:
    chunk_raw = row.get("chunk_id")
    chunk_id: UUID | None = None
    if chunk_raw:
        try:
            chunk_id = UUID(str(chunk_raw))
        except (ValueError, TypeError):
            chunk_id = None
    return _ChronicleEventRow(
        id=_parse_uuid(row.get("id", "")),
        namespace_id=_parse_uuid(row.get("namespace_id", "")) if row.get("namespace_id") else None,
        chunk_id=chunk_id,
        subject=row.get("subject") or "",
        verb=row.get("verb") or "",
        object=row.get("object") or "",
        observation_date=_parse_dt(row.get("observation_date")),
        referenced_date=_parse_dt(row.get("referenced_date")),
        relative_offset=row.get("relative_offset") or "",
        confidence=float(row.get("confidence", 1.0)),
        source_text=row.get("source_text") or "",
    )


def _row_to_memory_fact(row: dict[str, Any]) -> _MemoryFactRow:
    raw_chunks = row.get("source_chunk_ids") or []
    chunk_ids: list[UUID] = []
    for cid in raw_chunks:
        try:
            chunk_ids.append(UUID(str(cid)))
        except (ValueError, TypeError):
            continue
    superseded_raw = row.get("superseded_by")
    superseded_by: UUID | None = None
    if superseded_raw:
        try:
            superseded_by = UUID(str(superseded_raw))
        except (ValueError, TypeError):
            superseded_by = None
    return _MemoryFactRow(
        id=_parse_uuid(row.get("id", "")),
        namespace_id=_parse_uuid(row.get("namespace_id", "")) if row.get("namespace_id") else None,
        subject=row.get("subject") or "",
        predicate=row.get("predicate") or "",
        object_=row.get("object") or "",
        fact_text=row.get("fact_text") or "",
        confidence=float(row.get("confidence", 1.0)),
        is_active=bool(row.get("is_active", True)),
        superseded_by=superseded_by,
        source_chunk_ids=chunk_ids,
        created_at=_parse_dt(row.get("created_at")),
        updated_at=_parse_dt(row.get("updated_at")),
    )


# --------------------------------------------------------------------------- #
# Documents-tier recall-filter compile context + compiler registration.
# --------------------------------------------------------------------------- #
from khora.filter import (  # noqa: E402
    CompileContext,
    CompilerRegistry,
)
from khora.filter.compilers.surrealdb import compile_surrealdb  # noqa: E402

# The system keys this table backs with a real field — the single source of
# truth for the documents tier. Nine of the ten ``SYSTEM_KEYS``; ``occurred_at``
# is a recall-chunk field and has no ``document`` counterpart (see the
# ``DEFINE FIELD ... ON document`` block in ``surrealdb/schema.py``).
_BACKED_SYSTEM_KEYS: frozenset[str] = frozenset(
    {
        "created_at",
        "source_timestamp",
        "source_type",
        "source_name",
        "source_url",
        "external_id",
        "content_type",
        "source",
        "title",
    }
)


def _documents_compile_context() -> CompileContext:
    """Build the recall-filter :class:`CompileContext` for the ``document`` table.

    ``field_mapping`` declares the nine backed system keys (identity-mapped to
    their bare fields) plus the ``metadata`` root remap to the physical
    ``metadata_`` field. ``on_unsupported="split"`` per the enumeration
    contract: a document enumeration always has an in-memory post-filter
    available, so an unpushable leaf is left unconsumed rather than raising.

    ``occurred_at`` is deliberately absent: no ``document`` row backs it.
    ``compile_surrealdb`` treats the ``field_mapping`` key set as the backend's
    declared+pushable whitelist, so an undeclared system key never emits a
    predicate against a missing field. Under ``"split"`` such a leaf emits the
    match-all placeholder, stays out of ``CompiledFilter.consumed_keys``, and
    reaches the caller's post-filter.

    That whitelist matters more here than on the SQL backends. Were such a
    predicate emitted, SQL would ERROR on a column that does not exist — loud,
    and impossible to mistake for a result — whereas on the SCHEMAFULL
    ``document`` table a missing field reads ``NONE`` and SurrealQL's
    total-false absent-compare would return an empty result set that looks
    exactly like a legitimate no-match. ``compile_surrealdb`` is correspondingly
    stricter about what counts as declared; see its own gate for the difference.

    **``"split"`` is sound here only because of the all-or-nothing gate.** The
    ``true`` placeholder an unpushable leaf emits is superset-safe in positive
    position (``A AND true`` is ``A``) but inverts under ``$not`` / ``$or``:
    ``{"$not": {"$or": [{"title": "x"}, {"occurred_at": {"$gt": ...}}]}}`` would
    compile to ``!((((title = $f_0)) OR (true)))`` — ``!true``, matching ZERO
    rows silently, while ``consumed_keys`` reported ``title`` as pushed and left
    the caller a post-filter that can only narrow further. ``compile_surrealdb``
    therefore pushes an ``OR`` / ``NOT`` node only when its ENTIRE subtree is
    pushable; otherwise the whole node defers to the post-filter, consuming
    nothing. The filter above now compiles to a bare ``true`` with empty
    ``consumed_keys``. This context's ``"split"`` posture depends on that gate:
    relaxing it re-opens the zero-row defect above.
    ``tests/unit/filter/test_documents_compile_contexts.py`` pins the behaviour.

    **An unrenderable metadata segment is a capability gap on this context, not
    an internal fault.** A hyphenated key such as ``metadata.due-date`` is legal
    JSON and common in the wild, and ``compile_surrealdb`` cannot interpolate it
    as a SurrealQL identifier. Under ``"split"`` it therefore routes the leaf to
    the unsupported path — placeholder, unconsumed, post-filtered — rather than
    raising the internal ``CompileError`` (which stays the behaviour under
    ``"raise"``, where it is the injection guard). The emit path and the gate
    predicate were moved together, sharing one ``_segments_safe`` test, so they
    cannot disagree: the leaf defers in conjunctive position and inside an
    ``$or`` / ``$not`` alike. Consequently ``scan_documents`` carries no
    ``CompileError`` mapping — any ``CompileError`` that reaches it now is a
    genuine compiler fault and must escape as one.

    **The enumeration post-filter must evaluate the FULL filter AST
    unconditionally.** Everything the compiler pushes is a superset filter, but
    only because of two specific properties — it is not a free-standing
    guarantee. The all-or-nothing gate above defers a whole subtree it cannot
    fully express, rather than leaving a ``true`` placeholder inside it that
    would invert under negation and wrongly EXCLUDE rows; and an undeclared
    system key defers its own leaf rather than compiling to a ``NONE`` compare
    that would drop every row. Given both, re-running every leaf in memory can
    only narrow. Treat ``consumed_keys`` as a reporting and
    overfetch-sizing signal, not as permission to skip the leaves it names; that
    keeps caller correctness independent of how precisely the compiler tracks
    partial pushdown.

    ``backend_target`` is the SINGULAR ``document``, the real physical table name
    (``surrealdb/schema.py``), not the plural registry key. ``compile_surrealdb``
    never reads ``backend_target``, so the value is inert today; it names the
    physical table (house precedent) so the eventual enumeration caller reads the
    truth rather than a plausible-looking wrong table.
    """
    field_mapping = {key: key for key in _BACKED_SYSTEM_KEYS} | {"metadata": "metadata_"}
    return CompileContext(
        backend_target="document",
        field_mapping=field_mapping,
        # The enumeration contract's mode. Safe only because ``compile_surrealdb``
        # keeps an ``OR`` / ``NOT`` node all-or-nothing — see the docstring; a bare
        # ``true`` placeholder inside a negated subtree would otherwise match zero
        # rows silently.
        on_unsupported="split",
    )


# Register the deterministic recall-filter compiler for this store/target at
# import time (idempotent — same function object). This module is imported
# lazily by ``StorageFactory``, so registration happens when the backend is
# first constructed.
CompilerRegistry.register("relational.surrealdb", "documents", compile_surrealdb)
