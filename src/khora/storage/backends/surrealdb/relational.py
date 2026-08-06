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
from khora.storage.backends.base import DocumentScanKey, DocumentScanStep, PaginatedResult, build_scan_step
from khora.storage.backends.surrealdb._helpers import (
    _parse_dt,
    _parse_uuid,
    _record_id,
)
from khora.storage.backends.surrealdb.connection import SurrealDBConnection

if TYPE_CHECKING:
    from khora.filter.ast import FilterNode

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _none_if_empty(v: str | None) -> str | None:
    return v if v else None


def _checksum_reingestable_clause(pending_stale_before: datetime | None) -> tuple[str, dict[str, Any]]:
    """Build the checksum-dedup exclusion clause + binds for SurrealDB (#1464).

    FAILED rows are always excluded (re-ingestable). When ``pending_stale_before``
    is given, PENDING rows older than that cutoff are also excluded so a
    crash-abandoned half-ingest re-ingests; fresh PENDING rows stay a dedup hit,
    preserving the concurrent in-flight guard. When ``None`` only FAILED is
    excluded (legacy behavior).
    """
    if pending_stale_before is None:
        return "status != 'failed'", {}
    clause = "status != 'failed' AND (status != 'pending' OR updated_at >= $pending_stale_before)"
    return clause, {"pending_stale_before": pending_stale_before.isoformat()}


# Every named bind ``scan_documents`` builds for itself. A compiled filter's
# binds are merged over these, so any overlap is a silent substitution rather
# than an error — see the guard at the merge site. Reserved as a SET (not checked
# against the live keys) so the rejection does not depend on which step of a walk
# it happens on. Keep in sync with ``_documents_where`` + ``scan_documents``.
_SCAN_BIND_NAMES: frozenset[str] = frozenset({"ns", "lim", "status", "updated_before", "after_created_at", "after_id"})


def _documents_where(
    namespace_id: UUID,
    *,
    status: str | None,
    updated_before: datetime | str | None,
) -> tuple[list[str], dict[str, Any]]:
    """The ``document`` narrowing shared by ``list_documents`` and ``scan_documents``.

    Returns SurrealQL conjuncts and their named binds. Extracted because the
    namespace scope was written out verbatim in two methods 124 lines apart, and
    khora #1586's incident was *a deleted namespace predicate that tests did not
    catch*; #1587's response wrote that predicate into two more places. Callers
    append their own further conjuncts (the keyset disjunction, a compiled
    fragment) and their own ``lim`` / ``off`` binds.

    **``updated_before`` is taken already-serialized, and the union in its type is
    deliberate.** The two callers bind it differently on purpose:
    :meth:`~SurrealDBRelationalAdapter.list_documents` passes ``.isoformat()`` and
    :meth:`~SurrealDBRelationalAdapter.scan_documents` passes the ``datetime``
    object. ``updated_at`` is a ``TYPE datetime`` field, so the string form matches
    **no row at all.** Re-measured on a 6-row namespace with a bound every row is
    strictly before: no bound 6 rows, ``datetime`` bind 6, string bind **0** — and
    ``list_documents(updated_before=…)`` itself returns **0** for that same bound.
    So the scan's object bind is the correct one and ``list_documents``' string
    bind is a pre-existing bug in a *public* method.

    Normalizing it here would silently change what ``list_documents`` returns, so
    this helper binds whatever it is handed and the divergence stays visible at the
    two call sites. Fixing ``list_documents`` is a separate, caller-visible change
    and deliberately out of scope.
    """
    conditions = ["namespace_id = $ns"]
    params: dict[str, Any] = {"ns": str(namespace_id)}
    if status:
        conditions.append("status = $status")
        params["status"] = status
    if updated_before is not None:
        conditions.append("updated_at < $updated_before")
        params["updated_before"] = updated_before
    return conditions, params


def _scan_key(row: dict[str, Any]) -> DocumentScanKey:
    """The keyset position of one raw ``document`` row. ``@internal``.

    Strict on purpose, and **split between the two halves** — that split is the
    whole design, not an inconsistency:

    * ``created_at`` is taken **raw**. The row already carries a real
      ``datetime`` (``TYPE datetime``, returned tz-aware UTC by the SDK), so there
      is nothing to parse. It deliberately does NOT go through ``_parse_dt``,
      which returns ``None`` for a shape it cannot read; that ``None`` then
      coalesces to ``datetime.now(UTC)`` upstream, producing a cursor above every
      row and a walk that re-matches the same window instead of advancing.
    * ``id`` **must** be converted. ``DocumentScanKey`` is
      ``tuple[datetime, UUID]`` and the raw ``id`` is a ``RecordID``.

    **Do not "simplify" this by deriving both halves from the raw row wholesale.**
    Measured, seating the ``RecordID`` in the key with no conversion: **7 of the 33
    tests** in
    ``tests/integration/storage/backends/surrealdb/test_relational_scan_documents.py``
    fail, by **two different mechanisms** — and the second is the one that matters:

    * six walk/cursor tests die inside the SDK, ``ValueError: Failed to decode CBOR
      request: Error: Expected a CBOR integer, text, array or map``
      (``surrealdb/connections/async_embedded.py:119``), when the ``RecordID`` is
      bound back in as a cursor operand;
    * ``test_a_non_uuid_record_id_raises_instead_of_inventing_a_position`` fails
      with ``DID NOT RAISE``, because the wholesale form **defeats the strict
      conversion below entirely** rather than merely breaking the driver. That is
      the stronger argument for this function's shape: the failure is not "the SDK
      rejects it" but "the guard stops existing".

    Both counts are re-runnable rather than trustworthy — the denominator moves
    whenever that module grows (it was 6 of 30 before the QA lane added three
    tests). Re-run the mutant rather than citing these numbers second-hand.

    Nor may the id half be taken from the converted ``Document`` instead — that is
    what this function replaced, and it reintroduces the ``created_at`` coalesce
    above.

    The conversion is done locally rather than through ``_helpers._parse_uuid``
    because that helper's documented ``uuid5(NAMESPACE_URL, raw)`` fallback
    **invents** a UUID for a non-UUID record id — the same class of fault as the
    ``now()`` coalesce, and just as silent: the invented key round-trips into this
    same store and compares against nothing real. A non-UUID record id must raise
    here instead. ``_parse_uuid`` itself is left alone; its fallback is
    load-bearing for auto-generated RELATE ids and it has many other callers.

    ``ValueError`` for a malformed stored row: a data-integrity fault, not a
    filter-capability outcome, and deliberately a different class from the
    bind-collision guard's ``RuntimeError`` (an in-tree contract violation).

    **What the id raise does and does not cover — say both halves, because it is
    tempting to over-claim.** It closes the case where the window's **last** row
    (the only row a cursor is ever built from) has a non-UUID record id. It does
    **not** close the record-id homogeneity hazard generally: a foreign-shaped id
    anywhere else in the table still desynchronises ``id < $after_id`` from
    ``ORDER BY id DESC`` without this extractor ever seeing it. So the
    homogeneity precondition in :meth:`SurrealDBRelationalAdapter.scan_documents`
    stays, rescoped to that residue.
    """
    created_at = row.get("created_at")
    if not isinstance(created_at, datetime):
        raise ValueError(
            f"document row {row.get('id')!r} has a non-datetime created_at "
            f"({type(created_at).__name__}); cannot build a scan cursor from it"
        )

    raw_id = row.get("id")
    # A ``RecordID`` exposes the identifier half as ``.id`` — already a ``UUID``
    # for the ids this store writes. The ``getattr`` fallback also accepts the
    # bare ``table:⟨uuid⟩`` string form, whose delimiters and table prefix are
    # stripped below.
    inner = getattr(raw_id, "id", raw_id)
    if isinstance(inner, UUID):
        return (created_at, inner)

    text = str(inner)
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.strip("⟨⟩")
    try:
        return (created_at, UUID(text))
    except ValueError as exc:
        raise ValueError(
            f"document record id {raw_id!r} is not a UUID; refusing to invent a scan cursor position for it"
        ) from exc


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
        # ``.isoformat()`` is preserved verbatim, bug and all — see
        # ``_documents_where``'s docstring for why it is not "corrected" here.
        conditions, params = _documents_where(
            namespace_id,
            status=status,
            updated_before=updated_before.isoformat() if updated_before is not None else None,
        )
        params["lim"] = limit
        params["off"] = offset
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
        :meth:`list_documents` is. One ``SELECT`` on this adapter's own
        connection; a walk is the caller's job, chaining
        :attr:`DocumentScanStep.last_scanned` back in as ``after`` until the step
        reports ``exhausted``. No transaction spans steps and no consistent
        snapshot is claimed.

        SurrealQL has no row-value comparison, so #1586's single
        ``(created_at, id) < (…)`` expands to ``created_at < $ts OR (created_at =
        $ts AND id < $id)``. **Every conjunct is parenthesized unless it is a
        single comparison**, and there are two such sites at different
        severities. This one is **live**: we write the ``OR`` rather than
        inheriting it from a self-parenthesizing compiler, and it is reached on
        every resumed step. Ungrouped, ``AND`` binds tighter, so the disjunction
        splits the conjunct list at its own position: everything appended *before*
        it (namespace, status, ``updated_before``) lands inside the left disjunct,
        and the right one carries only the tie clause plus whatever is appended
        *after* — today just the compiled fragment. Either way the right disjunct
        has no namespace scope, so it returns other tenants' rows tied on the
        cursor instant. Measured with no ``filter_ast``, where the right disjunct
        is bare: 3 rows grouped against 7 ungrouped, 4 of them foreign. With a
        fragment the leak is narrowed by that fragment rather than removed, so
        the magnitude drops but the cross-tenant read does not. **The order of
        the appends is therefore load-bearing to this paragraph, not to the
        defect** — regrouping the list changes which conjuncts leak, never
        whether they do. The compiled
        fragment splice is the second site, a tripwire rather than a live hazard
        (``compile_surrealdb`` self-groups every boolean node today) with an
        identical failure mode, measured at 6 rows becoming 12. The inner
        ``(created_at = $ts AND id < $id)`` keeps its parens too: we do not lean
        on operator precedence for a predicate that fails cross-tenant.
        ``created_at`` is a non-optional ``TYPE datetime`` (``schema.py``), so the
        engine rejects ``NONE`` by type and the keyset needs no null guard.

        Cursor and bound operands bind as native objects — ``datetime`` for the
        timestamps, ``RecordID`` for the id — matching the write path. **That
        makes ``updated_before`` diverge from :meth:`list_documents`,
        deliberately.** ``updated_at`` is a ``TYPE
        datetime`` field and an ISO string compares against no row at all:
        measured on one corpus, no bound 6 rows, ``datetime`` bind 6, string bind
        0. Mirroring that ``.isoformat()`` would ship an ``updated_before``
        returning an empty walk that reports itself exhausted on the first call.
        The narrowing is otherwise the shape #1586 ships; its NULL-``updated_at``
        exclusion does not arise, the field being non-optional here.

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

        **It does not follow that a walk is O(namespace) on this store, and it
        is not.** Any top-level ``OR`` in the ``WHERE`` collapses SurrealDB's
        planner to a full table scan — measured on 2.x, the keyset disjunction
        loses not only the ``created_at`` range but the ``namespace_id`` prefix,
        so ``EXPLAIN`` reports ``Iterate Table`` where the same statement without
        the disjunction reports ``Iterate Index`` on ``idx_document_ns_created``.
        Every resumed step therefore re-examines every ``document`` row in the
        database, **all namespaces included**, and sorts in memory. A full walk
        is O(rows-in-table x steps), not O(namespace): measured 3.5x wall time
        for a 2x namespace, and 13x per step from 5k foreign rows the caller can
        never see. Cost tracks total corpus *bytes*, since a table scan
        materializes whole records.

        Two workarounds do not help, so do not re-try them: a ``WITH INDEX``
        hint still plans ``Iterate Table``, and so does a redundant
        ``created_at <= $ts`` guard alongside the disjunction. Restoring the
        index means getting the ``OR`` out of the ``WHERE`` — two
        index-eligible queries (the tie block by equality, then the strictly
        below range) merged client-side and capped at ``scan_limit``. That buys
        namespace scoping, not an O(scan_limit) step: 2.x still sorts the
        qualifying set in memory (see :mod:`.schema`). Tracked as a follow-up;
        this is a shipped-and-measured limitation, not an unknown.

        The raw-SQLite sibling has no such problem — its row-value keyset is
        consumed as an index range constraint, so there the total-work bound
        holds as stated.

        **Two preconditions on the stored data, neither reachable through khora's
        own writes, both reachable by a user — SurrealDB is a backend people write
        to directly.**

        *Record ids must be homogeneous.* This scan's tie-break leans on ``id <
        $after_id`` agreeing with ``ORDER BY id DESC``, and SurrealDB record ids
        are a tagged union (``Id::Uuid``, ``Id::String``, ``Id::Number``, …) whose
        variants order relative to one another rather than by content. A
        ``document`` table mixing variants is outside what this method is built
        for. **No magnitude is quoted here on purpose:** reproducing the reported
        cursor-vs-``ORDER BY`` disagreement means seeding record-id variants that
        this store's own write path cannot emit (``create_document`` routes every
        id through ``_record_id``), and the shapes reachable from raw DDL bind
        back differently from the shapes khora writes — so a bench number
        collected that way describes the bench, not this store. What IS verified
        is narrower and more useful: :func:`_scan_key` now **raises**
        ``ValueError`` on a non-UUID record id, so such a row fails the scan
        loudly at the cursor instead of being handed to ``_parse_uuid``, which
        would invent a UUID5 for it and produce a cursor pointing at no row at
        all. The silent-wrong-answer form of this hazard is gone; the precondition
        remains because a mixed table is still not something this ordering is
        defined over.

        *Sub-microsecond ``created_at`` values drop rows.* **Measured on the
        embedded engine in this tree:** ``time::now()`` stores nanoseconds — the
        engine's own string rendering of one such value is
        ``'2026-08-06T12:19:03.565752412Z'``, nine fractional digits — while the
        Python SDK hands back ``datetime(..., microsecond=565752)``, truncating
        the trailing ``412``. A cursor built from that row therefore binds an
        instant strictly *below* the value actually stored, so the row's own tie
        block matches **neither** disjunct: ``created_at < $after_created_at`` is
        false (stored is larger) and ``created_at = $after_created_at`` is false
        (stored is not equal). Those rows are skipped with no error. Unreachable
        through khora today because every write sets ``created_at`` from a Python
        ``datetime``, which is microsecond-precision by construction — but a
        ``DEFAULT time::now()`` on the field, or any direct write, produces ns
        immediately.

        **The compiled filter's binds are merged over this method's, and the merge
        is guarded.** ``params.update(compiled.params)`` lets the compiled side
        win, so an overlapping name is not a clash but a silent substitution —
        a compiled bind called ``ns`` replaces the tenant scope in
        ``namespace_id = $ns`` and the statement still executes normally, with a
        plausible row count and a valid ``last_scanned``, so a walk carries on
        over another tenant's rows. The names this method reserves are
        ``_SCAN_BIND_NAMES``, checked as a set rather than against the live keys so
        the rejection cannot depend on which step of a walk it lands on. See the
        guard's own comment for what it does and does not defend against (short
        version: not ``param_namespace``, which structurally cannot produce a
        reserved name; a compiler that does not follow the ``{prefix}_{n}``
        convention at all). Raw-SQLite needs no counterpart — its binds are
        positional.

        The cursor comes from :func:`_scan_key` over the **raw row**, which is
        strict where ``_row_to_document`` is forgiving: the latter coalesces an
        unreadable ``created_at`` to ``datetime.now(UTC)`` and routes the id
        through ``_parse_uuid``'s UUID5 fallback, either of which yields a
        *plausible but fictional* position rather than an error. Only the id half
        is converted; ``created_at`` is taken as-is.

        Raises ``ValueError`` when ``scan_limit`` is below 1, before anything is
        compiled or executed, so a bad bound is never masked by a compile error;
        also from :func:`_scan_key` on a malformed stored row, and from the bind
        guard above.

        **It does not raise on a metadata path this backend cannot render as an
        identifier** (ticket §8). ``metadata.foo-bar`` is legal, common JSON; it
        is treated as an unpushable leaf, deferred to the caller's residual
        filter, and never reported as a capability failure. The old behaviour —
        raise in conjunctive position, work inside a deferred ``$or`` — was
        position-dependent and made this the only one of four stores that could
        fail a filter the other three answered. That split is gone, not
        documented; see :class:`~khora.storage.backends.base.DocumentScanStep`.
        """
        if scan_limit < 1:
            raise ValueError(f"scan_limit must be >= 1, got {scan_limit}")

        # Binds the datetime OBJECT, not ``.isoformat()`` — see the docstring for
        # the measurement behind the deliberate divergence from
        # ``list_documents``, whose string bind matches no row. Do not "correct"
        # it back to match that method.
        conditions, params = _documents_where(namespace_id, status=status, updated_before=updated_before)
        params["lim"] = scan_limit
        if after is not None:
            cursor_created_at, cursor_id = after
            conditions.append("(created_at < $after_created_at OR (created_at = $after_created_at AND id < $after_id))")
            params["after_created_at"] = cursor_created_at
            params["after_id"] = _record_id("document", cursor_id)

        consumed_keys: frozenset[str] = frozenset()
        if filter_ast is not None:
            compiler = CompilerRegistry.get("relational.surrealdb", "documents")
            # No ``CompileError`` mapping here, deliberately (ticket §8 ruling).
            # A metadata leaf whose path segment is not a SurrealQL identifier —
            # ``metadata.foo-bar``, legal and common JSON — is an **unpushable
            # leaf, not an error**: under this context's ``"split"`` mode the
            # compiler leaves it unconsumed and emits the match-all placeholder,
            # so it never reaches the emit walk that would raise. The compiler's
            # own injection guard in ``_Builder._metadata_path`` still exists and
            # still fires under ``"raise"`` mode, but is unreachable through this
            # context — defense in depth, not the live path.
            #
            # The leaf is then the CALLER's to evaluate, like every other
            # unconsumed leaf: it is absent from ``consumed_keys``, which is
            # exactly the signal that says so. Note there is no caller yet —
            # ``scan_documents`` has none in-tree — so this is an obligation the
            # contract places on a future one, not a post-filter that runs today.
            #
            # What that buys is the thing a mapping could not: all four backends
            # return the same ROWS for the same filter, and the old
            # position-dependence (raises in conjunctive position, works inside a
            # deferred ``$or``) is gone rather than merely relabelled.
            #
            # **Rows, not pushdown.** The three other stores push this leaf into
            # SQL — measured, ``consumed_keys == {"metadata.foo-bar"}`` on
            # raw-sqlite, sqlite_lance and postgresql — while this one defers it
            # and reports ``consumed_keys == set()``. Same answer, reached
            # differently, and the difference is visible in the signal that exists
            # to report it. Do not read the parity as "the backends agree".
            compiled = compiler(filter_ast, _documents_compile_context())
            consumed_keys = compiled.consumed_keys
            conditions.append(f"({compiled.predicate})")

            # The bind namespaces must stay disjoint, and the invariant runs in
            # BOTH directions: (a) never name a scan bind ``f_<n>``, which is the
            # compiler's family (``{param_namespace}_{n}``), and (b) never give a
            # compiler a ``param_namespace`` that could collide with a scan bind.
            #
            # ``params.update`` lets the compiled side WIN the merge, so a
            # collision is not a clash but a silent substitution: a compiled bind
            # named ``ns`` replaces the tenant scope in ``namespace_id = $ns`` and
            # the query still executes normally — plausible row count, valid
            # ``last_scanned``, so a walk continues over foreign data. Reproduced:
            # a compiler returning ``params={"f_0": ..., "ns": str(other_ns)}``
            # made a scan of namespace A return namespace B's row, with no error
            # and nothing in the logs.
            #
            # **This is a tripwire, not a fix for a reachable configuration, and
            # the distinction must not be blurred.** ``param_namespace`` cannot
            # produce a collision at any setting: ``compile_surrealdb._bind`` has
            # a single assignment site and names every bind
            # ``f"{param_namespace}_{counter}"``, so every compiled bind ends in
            # ``_<digits>`` and none of the six reserved names has that shape.
            # Swept over ten values including ``"ns"`` and ``"after_id"`` (which
            # yield ``ns_0`` and ``after_id_0``), the intersection is empty every
            # time. The hazard this catches is a compiler that **hand-writes** a
            # bind name outside that convention — the same severity class as the
            # fragment-paren tripwire one line earlier, which was likewise
            # enforced rather than assumed. ``_documents_compile_context`` pins
            # ``param_namespace`` explicitly as the construction-time half.
            #
            # Reserved names rather than live keys, and this is the only correct
            # justification for that choice: against a compiler hand-writing a
            # bare ``{"after_id": ...}``, the live-key form misses when
            # ``after=None`` (the key is simply absent) and fires only on the
            # first RESUMED step. Rejecting the configuration beats a guard whose
            # firing depends on which step of a walk you happen to be on.
            #
            # ``RuntimeError``, matching this file's precedent for an internal
            # invariant failure (``create_document``'s "Failed to create
            # document"). Deliberately NOT ``ValueError`` — nothing the caller
            # passed is invalid, an in-tree compiler/context contract was
            # violated, and ``ValueError`` already means "caller passed a bad
            # bound" in this method and "a stored value is malformed" in
            # ``_scan_key``. Deliberately NOT ``CompileError`` either: that is
            # defined as an error raised *inside the compile step* and is
            # catchable as a filter-capability outcome, so a future caller could
            # downgrade a potential cross-tenant read into a post-filter
            # fallback. A cross-tenant tripwire must not look recoverable, and it
            # is never mapped onto ``RecallFilterUnsupportedError``.
            #
            # Names only in the message, never values — the binds carry document
            # content and user filter values.
            collisions = compiled.params.keys() & (params.keys() | _SCAN_BIND_NAMES)
            if collisions:
                raise RuntimeError(f"compiled filter binds collide with scan binds: {sorted(collisions)}")
            params.update(compiled.params)

        where = " AND ".join(conditions)
        rows = await self._conn.query(
            f"SELECT * FROM document WHERE {where} ORDER BY created_at DESC, id DESC LIMIT $lim",  # noqa: S608
            params,
        )
        # The cursor comes from ``_scan_key`` over the RAW row, which takes
        # ``created_at`` raw and converts only the ``id``. Still worth stating
        # what makes the id half easy to get wrong: a ``RecordID`` seated in a
        # ``DocumentScanKey`` round-trips into this same store, so a type error
        # there is not loud on every path — it is silently green on some. See
        # ``_scan_key`` for what each half does and why they differ.
        return build_scan_step(
            rows,
            scan_limit=scan_limit,
            consumed_keys=consumed_keys,
            key=_scan_key,
            document=self._row_to_document,
        )

    async def claim_orphaned_documents(
        self,
        namespace_id: UUID,
        *,
        pending_before: datetime,
        processing_before: datetime,
        limit: int = 100,
    ) -> list[Document]:
        """Claim stale orphaned documents (no row locking - SurrealDB plain claim)."""
        ns_str = str(namespace_id)
        rows = await self._conn.query(
            "SELECT * FROM document WHERE namespace_id = $ns AND ("
            "(status = $pending AND updated_at < $pending_before) OR "
            "(status = $processing AND updated_at < $processing_before)"
            ") ORDER BY updated_at LIMIT $lim",
            {
                "ns": ns_str,
                "pending": DocumentStatus.PENDING.value,
                "processing": DocumentStatus.PROCESSING.value,
                "pending_before": pending_before.isoformat(),
                "processing_before": processing_before.isoformat(),
                "lim": limit,
            },
        )
        docs = [self._row_to_document(r) for r in rows]
        if not docs:
            return []
        now_iso = datetime.now(UTC).isoformat()
        for doc in docs:
            doc.orphan_prior_status = doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status
            await self._conn.query(
                "UPDATE $rid SET status = $status, updated_at = $updated_at",
                {
                    "rid": _record_id("document", doc.id),
                    "status": DocumentStatus.PROCESSING.value,
                    "updated_at": now_iso,
                },
            )
            doc.status = DocumentStatus.PROCESSING
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
        """
        ns_str = str(namespace_id)
        where, binds = _checksum_reingestable_clause(pending_stale_before)
        row = await self._conn.query_one(
            f"SELECT * FROM document WHERE namespace_id = $ns AND checksum = $checksum AND {where} LIMIT 1",  # noqa: S608
            {"ns": ns_str, "checksum": checksum, **binds},
        )
        if row is None:
            return None
        return self._row_to_document(row)

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

        Returns:
            Dictionary mapping checksum to Document (only for existing documents)
        """
        if not checksums:
            return {}
        ns_str = str(namespace_id)
        where, binds = _checksum_reingestable_clause(pending_stale_before)
        rows = await self._conn.query(
            f"SELECT * FROM document WHERE namespace_id = $ns AND checksum IN $checksums AND {where}",  # noqa: S608
            {"ns": ns_str, "checksums": checksums, **binds},
        )
        result: dict[str, Document] = {}
        for r in rows:
            doc = self._row_to_document(r)
            cs = r.get("checksum", "")
            if cs:
                result[cs] = doc
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

    **Non-identifier metadata segments defer; they do not raise** (ticket §8).
    A hyphenated key such as ``metadata.due-date`` is legal, common JSON, and
    ``compile_surrealdb``'s ``_Builder._metadata_path`` cannot interpolate it as
    a SurrealQL identifier — the backend has no bind form for a field *name*. It
    is therefore an unpushable leaf, exactly like an undeclared system key: under
    this context's ``"split"`` mode the compiler leaves it unconsumed and emits
    the match-all placeholder instead of raising. It stays out of
    ``consumed_keys``, which is the signal telling the caller to evaluate it in
    memory.

    **Stated as an obligation, not as a running mechanism:** ``compile_python``
    does handle hyphenated keys correctly (verified), but nothing wires it to
    this scan today — ``scan_documents`` has no caller in-tree. The rows are only
    equal to the other stores' *after* some future caller applies the full filter
    to the window. Until then this context has made the leaf deferrable, not
    deferred.

    The compiler's emit-time injection guard is deliberately **left in place and
    untouched**. It becomes defense-in-depth: unreachable through this context,
    because the gate defers the leaf before the emit walk runs, but still firing
    on direct compiles that do not go through a documents context. Do NOT loosen
    ``_SAFE_SEGMENT_RE`` — nothing here relaxes what counts as a safe identifier;
    the change is only about *which path* an unsafe one takes.

    This removes the *raise* and the *row* divergence: all four backends return
    the same rows for the same filter, and the old position-dependence (raise in
    conjunctive position, work inside a deferred ``$or``) is gone. Previously the
    same hyphenated key returned rows on raw-SQLite, whose ``compile_lance`` has
    no such guard, and raised here.

    **The pushdown split remains, and is now visible in ``consumed_keys``.**
    Measured on the same filter: the other three documents contexts push the leaf
    into SQL (``consumed_keys == {"metadata.foo-bar"}`` on raw-sqlite,
    sqlite_lance and postgresql) while this one defers it
    (``consumed_keys == set()``). Two consequences worth knowing rather than
    rediscovering: a caller differencing its own leaf keys against
    ``consumed_keys`` to report "what cost a post-filter" gets a different answer
    per store for the same filter; and under a bounded ``scan_limit`` this store
    fills its window with rows the residual will reject, so the same filter costs
    more steps here than on a store that narrowed in SQL.

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
        # Stated rather than inherited, and it is the construction-time half of
        # the scan's bind-disjointness invariant. ``CompileContext`` documents
        # this field as existing precisely "so compiled params cannot collide with
        # the engine's own query parameters", and ``"f"`` is also its default — so
        # naming it changes nothing today and makes the choice deliberate and
        # local instead of silently inherited from another module.
        #
        # The disjointness argument: ``compile_surrealdb._bind`` names every bind
        # ``f"{param_namespace}_{counter}"``, so every compiled bind ends in
        # ``_<digits>``, and no bind ``scan_documents`` builds for itself has that
        # shape (``ns``, ``lim``, ``status``, ``updated_before``,
        # ``after_created_at``, ``after_id`` — see ``_SCAN_BIND_NAMES``). That
        # holds for ANY value of this field, which is why the runtime guard at the
        # merge site is a tripwire against a compiler abandoning the convention
        # rather than a check on this setting.
        param_namespace="f",
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
