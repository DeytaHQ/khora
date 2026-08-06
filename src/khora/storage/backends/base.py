"""Abstract protocols for storage backends.

These protocols define the interface that all storage backends must implement,
enabling dependency injection and easy testing with mocks.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable
from uuid import UUID

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
    from khora.dream.plan import OpKind
    from khora.filter.ast import FilterNode


@dataclass(frozen=True)
class PaginatedResult(Generic[T]):
    """Paginated query result with total count."""

    items: list[T]
    total: int
    limit: int
    offset: int


# The keyset position of a single document in the enumeration order, as
# ``(created_at, id)``. ``@internal``.
#
# Deliberately a bare tuple rather than a named cursor type: it has exactly one
# producer (a scan step) and one consumer (the next scan step's ``after``), both
# in-tree, and no wire form. A cursor *class* belongs wherever an opaque,
# caller-facing encoding is first needed, and would pin an encoding this tier has
# no use for.
#
# **The position is store-local, not a normalized instant.** The round-trip is
# exact by construction on each backend, but **four stores implement
# ``scan_documents`` and they carry three different key shapes.** Never format
# this into a string, and never carry a position from one store to another.
#
# | store | ``created_at`` half | bound back as |
# | --- | --- | --- |
# | ``postgresql`` | aware (``timestamptz`` holds an instant) | ``sa.literal(v, col.type)`` |
# | ``sqlite_lance`` | **naive** — TEXT holding writer wall clock, offset discarded at write | ``sa.literal(v, col.type)`` |
# | ``sqlite`` (raw) | **aware OR naive**, whichever the writer passed | positional bind, ``_dt_to_str`` / ``str(uuid)`` |
# | ``surrealdb`` | aware, UTC (``TYPE datetime``) | native ``datetime`` + ``RecordID`` |
#
# **Only the two SQLAlchemy stores bind back "through the same column type".**
# That phrasing described the #1586 pair and does not generalize: raw-sqlite
# round-trips ``datetime.fromisoformat`` -> ``_dt_to_str`` over a TEXT column
# with no type object anywhere in the path, and SurrealDB binds native objects,
# re-wrapping the id half into a ``RecordID`` on the way in. A rule derived from
# either SQLAlchemy store does not transfer to either raw store.
#
# Raw-sqlite's third shape is the one most likely to be got wrong, because the
# aware-on-PG / naive-on-embedded split above looks like it covers it and does
# not. That store's writer (``_dt_to_str``) preserves the caller's offset with no
# UTC coercion, and its reader (``datetime.fromisoformat``) preserves whatever
# was stored — so the key is aware for a row written from an aware datetime and
# naive for one written from a naive datetime, **within a single table**. Two
# positions from the same namespace are therefore not necessarily mutually
# comparable in Python, even though SQLite orders the underlying TEXT total.
# Anything that wants to compare, normalize or serialize positions across stores
# has to handle all three shapes explicitly; there is no store-independent
# instant here to fall back on.
DocumentScanKey = tuple[datetime, UUID]


@dataclass(frozen=True, slots=True)
class DocumentScanStep:
    """One bounded step of a keyset walk over a namespace's documents.

    ``@internal``. Produced by a relational backend's ``scan_documents``; not part
    of the public storage API and not re-exported from :mod:`khora.storage`.

    A step is one ``SELECT`` in the backend's own session — no transaction spans
    steps, and no consistent snapshot is claimed. A walk chains steps by feeding
    :attr:`last_scanned` back as the next call's ``after`` until
    :attr:`exhausted`.

    * ``documents`` — the window's rows in ``(created_at DESC, id DESC)`` order
      (the total order all four relational stores share since khora #1576),
      already narrowed by whatever the dialect compiler pushed down. **Not
      necessarily the final matches**: a caller with a filter must still evaluate
      the full filter over these rows (see :attr:`consumed_keys`).
    * ``last_scanned`` — the keyset position of the window's **final row**, or
      ``None`` when the window was empty. Resume from this rather than from the
      last row that survived post-filtering, so a resumed walk does not re-scan
      the gap of rows that were scanned and rejected.
    * ``exhausted`` — the window returned fewer rows than the requested bound, so
      nothing remains after :attr:`last_scanned`. This is the **only** sound
      termination signal; a short — or empty — ``documents`` list means nothing on
      its own, because post-filtering can reject an entire window.
    * ``consumed_keys`` — the dotted filter paths the backend pushed into SQL, as
      reported by the compiler. A **reporting signal only**: it exists so a caller
      can tell users which predicate leaves cost a post-filter. It is *not*
      permission to skip those leaves in the post-filter — the compile contexts on
      both stores require the full filter to be re-evaluated in memory
      unconditionally, and everything pushed down is a superset filter, so
      re-running a pushed leaf can only narrow.

    There is deliberately **no residual-AST field.** The compilers report the
    split as a key set, not as a pruned tree, and the caller already holds the
    filter it passed in — so a residual field could only duplicate the caller's
    own input, while inviting the one mistake the compile contexts warn against
    (post-filtering the residual alone). A caller that wants to report which
    predicate leaves cost a post-filter differences its own leaf keys against
    :attr:`consumed_keys`; :func:`khora.filter.execute.filter_leaf_keys` is the
    left-hand side.

    Distinct from :class:`PaginatedResult`, which is offset-shaped and carries a
    total. A keyset walk has neither an offset nor a total.

    All four stores assemble this through :func:`build_scan_step`, which is what
    keeps ``last_scanned`` and ``exhausted`` derived from the raw window rather
    than from ``documents``.

    **No store rejects a filter it cannot fully push.** ``scan_documents`` does
    not raise :class:`~khora.filter.model.RecallFilterUnsupportedError` on any of
    the four: all four documents compile contexts select
    ``on_unsupported="split"``, and these compilers raise that error only under
    ``"raise"``. (``"raise"`` stops on the first node the backend cannot express;
    ``"split"`` compiles what it can and defers the rest to the caller's
    post-filter.) Measured over the 209-case ``khora.filter.conformance`` corpus
    lowered through each store's own documents context: **209/209 compile with no
    exception on all four**. An unpushable leaf is a deferral, never a rejection.

    This was not always true, and the history is worth one line because the
    earlier contract was the opposite: SurrealDB used to raise on a metadata path
    segment it could not render as an identifier, making it the only store that
    could fail a filter the other three answered. See split 1 below.

    All four raise ``ValueError`` for ``scan_limit < 1``, and the two raw stores
    additionally raise ``ValueError`` from their ``_scan_key`` on a stored row
    they cannot turn into a cursor — a data-integrity fault, not a filter
    outcome.

    **One cross-store behavioural split remains documented and NOT fixed; a
    second was closed by ruling rather than by workaround.**

    1. *Hyphenated metadata keys — CLOSED for rows, and the residue is a
       pushdown difference, not a behavioural one.* ``metadata.foo-bar`` is
       legal, common JSON. It once returned rows on raw-sqlite and raised
       ``RecallFilterUnsupportedError`` on SurrealDB — and, worse, the two
       **agreed** when the same key sat inside an ``$or`` the all-or-nothing gate
       deferred, so the split was per-store *and* position-dependent. It is now
       an **unpushable leaf** on SurrealDB rather than an error: the compile
       context reports it non-consumable and the leaf is left for a post-filter,
       in every position.

       **Be precise about what "the same rows" means, because it is not true of
       this method's own return value.** :attr:`documents` is the raw window, and
       the four stores' windows **differ**: the three pushing stores narrow in
       SQL, SurrealDB does not, so its window carries rows the leaf excludes.
       Measured on a 6-document namespace, conjunctive position: SurrealDB
       ``len(documents) == 6``, the pushing stores 3. Equality holds only for the
       set that survives the full-filter post-filter every caller already owes
       (see :attr:`consumed_keys`) — measured against a ``compile_python`` oracle
       over the same corpus: conjunctive **3 == 3**, inside ``$or`` **4 == 4**,
       inside ``$not`` **3 == 3**. So: same final answer, different windows,
       different step counts, different intermediate :attr:`last_scanned` values.

       The ``$not`` row had to be checked rather than assumed — deferring only
       the *emit* side while leaving the consumable gate alone would have
       compiled it to ``!(true)``, matching **zero** rows, and no post-filter can
       recover rows a window never returned.

       **What survives is a pushdown split, and it is visible in
       :attr:`consumed_keys` rather than hidden.** Measured on that filter: the
       three other stores push it into SQL (``consumed_keys ==
       {"metadata.foo-bar"}``) while SurrealDB defers it (``consumed_keys ==
       set()``).

       **In ``$or`` / ``$not`` position the cost is not marginal — SurrealDB
       pushes NOTHING**, because the all-or-nothing gate defers the entire
       enclosing subtree and the pushable sibling goes with it. Measured on
       ``{"$or": [{"metadata.foo-bar": …}, {"source_type": …}]}`` and on the same
       filter wrapped in ``$not``:

       ``postgresql`` / ``raw-sqlite`` / ``sqlite_lance``
       -> ``consumed_keys == {"metadata.foo-bar", "source_type"}``;
       ``surrealdb`` -> ``consumed_keys == set()``, predicate ``true``.

       So the window there is not "narrowed by one fewer leaf", it is the whole
       namespace — a namespace-sized walk against a matches-sized one, compounding
       with the ``Iterate Table`` planner collapse ``scan_documents`` documents
       separately. **Attribute this correctly:** wholesale subtree deferral is the
       pre-existing all-or-nothing gate, not something this ruling introduced.
       What changed is only that a non-identifier metadata segment now *counts* as
       unconsumable, so the set of filters that push nothing on SurrealDB grew to
       include them — where they previously raised instead. A fair trade, and the
       right one, but a caller sizing overfetch off :attr:`consumed_keys` will see
       ``set()`` and should know why.

       Two consequences for a walking caller: differencing its own leaf keys
       against :attr:`consumed_keys` to report "which predicates cost a
       post-filter" yields a different answer per store for the same filter; and
       under a bounded ``scan_limit`` SurrealDB fills its window with rows the
       post-filter then rejects. Both are correct-but-different work, not a
       correctness divergence.
    2. *``updated_before`` on raw-SQLite.* That store compares a wall-clock
       string against a lexicographically-ordered TEXT column whose writer
       preserves the caller's offset, so it drops rows SurrealDB returns, and
       wrongly includes a naive value exactly equal to the bound (the bind is a
       prefix of it). Inherited verbatim from ``list_documents`` and not fixable
       without changing that public method. ``updated_before`` is a direct
       parameter and **no post-filter can recover the dropped rows**, which is
       what makes this worse than a pushdown gap.
    """

    documents: list[Document]
    last_scanned: DocumentScanKey | None
    exhausted: bool
    consumed_keys: frozenset[str]


def build_scan_step[RowT](
    rows: Sequence[RowT],
    *,
    scan_limit: int,
    consumed_keys: frozenset[str],
    key: Callable[[RowT], DocumentScanKey],
    document: Callable[[RowT], Document],
) -> DocumentScanStep:
    """Assemble a :class:`DocumentScanStep` from a raw result window. ``@internal``.

    The single home for the one invariant every ``scan_documents`` shares, and the
    reason this takes ``rows`` plus a ``key`` *callable* rather than a
    pre-extracted key:

    **``last_scanned`` and ``exhausted`` both describe the RAW window** — the
    final row the SQL scanned, and whether the store ran out of rows filling the
    bound. **Neither may be derived from a post-filtered subset.** Deriving
    ``last_scanned`` from the surviving matches would re-scan the rejected gap on
    every resume; deriving ``exhausted`` from them would call a full window
    exhausted and truncate the walk at the first window whose rows all fail the
    caller's filter.

    A helper taking an already-extracted key could not enforce that: a caller
    handing it ``documents[-1]``'s key would be silently green, which is exactly
    the defect this signature exists to make inexpressible. Because the helper
    builds the documents itself, there is no post-filtered list in scope to pass
    by mistake.

    Dialect coupling stays in the two callables, where it belongs — this function
    never sees a datetime format, a hex form, a ``RecordID`` or a column type.
    ``key`` is expected to be **strict**: it reads the stored value and raises on
    a row it cannot read, rather than substituting a default. A key extractor
    that coalesces (``... or datetime.now(UTC)``) turns one malformed row into a
    cursor above every row, i.e. a non-terminating walk rather than a bad value —
    see each store's own ``_scan_key`` for the measured shape of that.

    ``scan_limit`` is NOT validated here. It has to be rejected before anything
    is compiled or executed, and this runs after execution; the two SQLAlchemy
    stores validate it in ``build_documents_scan_query`` and the two raw stores
    inline.
    """
    return DocumentScanStep(
        documents=[document(row) for row in rows],
        last_scanned=key(rows[-1]) if rows else None,
        exhausted=len(rows) < scan_limit,
        consumed_keys=consumed_keys,
    )


@runtime_checkable
class RelationalBackendProtocol(Protocol):
    """Protocol for relational database backends (PostgreSQL).

    Handles storage of documents, tenancy data, ACLs, and sync checkpoints.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connections."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        ...

    # Namespace operations
    @abstractmethod
    async def resolve_namespace(self, namespace_id: UUID) -> UUID:
        """Resolve a stable namespace_id to the active version's row id."""
        ...

    @abstractmethod
    async def create_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Create a new memory namespace."""
        ...

    @abstractmethod
    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        ...

    @abstractmethod
    async def list_namespaces(
        self, *, active_only: bool = True, limit: int = 100, offset: int = 0
    ) -> PaginatedResult[MemoryNamespace]:
        """List namespaces with pagination."""
        ...

    @abstractmethod
    async def update_namespace(self, namespace: MemoryNamespace) -> MemoryNamespace:
        """Update a namespace."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def deactivate_namespace(self, namespace_id: UUID) -> None:
        """Mark a namespace version as inactive.

        Args:
            namespace_id: ID of the namespace to deactivate
        """
        ...

    # Document operations
    @abstractmethod
    async def create_document(self, document: Document) -> Document:
        """Create a new document."""
        ...

    @abstractmethod
    async def get_document(self, document_id: UUID, *, namespace_id: UUID) -> Document | None:
        """Get a document by ID, scoped to ``namespace_id``.

        Returns ``None`` if the document does not exist OR belongs to a
        different namespace — the caller's namespace is the authority.
        The ``namespace_id`` filter prevents cross-tenant document access
        by id (IDOR).
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def claim_orphaned_documents(
        self,
        namespace_id: UUID,
        *,
        pending_before: datetime,
        processing_before: datetime,
        limit: int = 100,
    ) -> list[Document]:
        """Atomically claim stale orphaned documents for crash recovery.

        Selects documents that are either ``pending`` and older than
        ``pending_before`` OR ``processing`` and older than
        ``processing_before``, flips the claimed rows to ``processing`` (with a
        fresh ``updated_at``), and returns them. On PostgreSQL the claim is
        serialized with ``FOR UPDATE SKIP LOCKED`` so concurrent recovery loops
        never claim the same document. SQLite (single-writer) and SurrealDB
        perform the same claim without row locking.

        Unlike :meth:`list_documents` (a pure read), this method mutates state.
        """
        ...

    @abstractmethod
    async def update_document(self, document: Document) -> Document:
        """Update a document."""
        ...

    @abstractmethod
    async def delete_document(self, document_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a document, scoped to ``namespace_id``.

        Returns ``False`` if the document does not exist OR belongs to a
        different namespace — the caller's namespace is the authority. The
        ``namespace_id`` filter prevents cross-tenant deletion by id.
        """
        ...

    @abstractmethod
    async def count_documents(self, namespace_id: UUID) -> int:
        """Count documents in a namespace.

        Args:
            namespace_id: Namespace UUID

        Returns:
            Total number of documents. Returns 0 if namespace is empty.
        """
        ...

    @abstractmethod
    async def get_last_activity_at(self, namespace_id: UUID) -> datetime | None:
        """Get the most recent document creation timestamp in a namespace.

        Args:
            namespace_id: Namespace UUID

        Returns:
            datetime: Timestamp of the most recently created document (UTC)
            None: If the namespace has no documents
        """
        ...

    async def get_document_stats(self, namespace_id: UUID) -> tuple[int, datetime | None]:
        """Count documents and get last activity in a single query.

        Returns (count, last_activity_at). Backends may override for efficiency;
        the default falls back to two separate calls.
        """
        count = await self.count_documents(namespace_id)
        last_activity = await self.get_last_activity_at(namespace_id)
        return count, last_activity

    @abstractmethod
    async def get_document_by_checksum(
        self, namespace_id: UUID, checksum: str, *, pending_stale_before: datetime | None = None
    ) -> Document | None:
        """Get a document by its content checksum (for deduplication).

        FAILED documents are always excluded. When ``pending_stale_before`` is
        given, PENDING documents older than that cutoff are also excluded so a
        crash-abandoned half-ingest (#1464) re-ingests instead of being a
        permanent dedup hit.
        """
        ...

    @abstractmethod
    async def get_documents_by_checksums(
        self, namespace_id: UUID, checksums: list[str], *, pending_stale_before: datetime | None = None
    ) -> dict[str, Document]:
        """Fetch documents by content checksums in a single query (batch dedup).

        Same status semantics as ``get_document_by_checksum``: FAILED is always
        excluded, and when ``pending_stale_before`` is given, PENDING documents
        older than that cutoff are also excluded so a crash-abandoned half-ingest
        (#1464) re-ingests. Returns a dict mapping checksum to Document for the
        documents that exist and pass the filter.
        """
        ...

    @abstractmethod
    async def get_document_by_external_id(
        self,
        external_id: str | None,
        *,
        namespace_id: UUID,
    ) -> Document | None:
        """Get a document by its caller-supplied external_id.

        Unlike ``get_document_by_checksum``, this lookup does NOT filter by
        status — it returns ``COMPLETED``, ``PROCESSING``, and ``FAILED`` rows
        so callers can self-heal a failed extraction on the next replace
        against the same ``external_id``.

        Returns ``None`` immediately if ``external_id`` is ``None`` (guard).
        """
        ...

    @abstractmethod
    async def get_documents_by_external_ids(
        self,
        external_ids: list[str],
        *,
        namespace_id: UUID,
    ) -> dict[str, Document]:
        """Batch equivalent of :meth:`get_document_by_external_id`.

        Returns a mapping of ``external_id -> Document`` for every external_id
        that currently resolves to a row within the namespace. Like the single
        lookup, this does NOT filter by status (self-heal).
        ``None`` / empty entries in ``external_ids`` are skipped.

        Empty input returns ``{}`` immediately.
        """
        ...

    async def get_documents_batch(self, document_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Document]:
        """Fetch multiple documents in a single query, scoped to ``namespace_id``.

        Documents belonging to any other namespace are silently dropped
        from the result to prevent cross-tenant IDOR (IDOR family).

        Returns dictionary mapping document ID to Document object.
        """
        ...

    async def get_document_sources_batch(
        self, document_ids: list[UUID], *, namespace_id: UUID
    ) -> dict[UUID, DocumentSource]:
        """Fetch lightweight document metadata for source attribution,
        scoped to ``namespace_id``.

        Returns a column-limited projection (no content, processing stats,
        or mutable state) for display and linking purposes. Documents in
        other namespaces are silently dropped from the result (IDOR family).

        Args:
            document_ids: List of document IDs to fetch
            namespace_id: Caller's namespace; documents belonging to any
                other namespace are silently dropped from the result.

        Returns:
            Dictionary mapping document ID to DocumentSource
        """
        ...

    async def get_document_projections_batch(
        self,
        document_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> dict[UUID, DocumentProjection]:
        """Fetch full ``DocumentProjection`` rows for recall responses.

        Returns the typed projection shape used by ``Khora.recall()``:
        ``id``, ``created_at``, ``source_type``, ``title``, ``external_id``,
        ``source``, ``source_name``, ``source_url``, ``content_type``,
        ``source_timestamp``, ``metadata``.

        Distinct from ``get_document_sources_batch`` (which returns the
        narrower ``DocumentSource`` for entity-source attribution) so the
        two consumers can evolve their column sets independently.

        Args:
            document_ids: List of document IDs to fetch
            namespace_id: Namespace scope — rows from other namespaces are
                filtered at the query layer (security close-out).

        Returns:
            Dictionary mapping document ID to DocumentProjection
        """
        ...

    # Sync checkpoint operations
    @abstractmethod
    async def get_sync_checkpoint(self, namespace_id: UUID, source: str) -> str | None:
        """Get the last sync checkpoint for a source."""
        ...

    @abstractmethod
    async def set_sync_checkpoint(self, namespace_id: UUID, source: str, checkpoint: str) -> None:
        """Set the sync checkpoint for a source."""
        ...

    def _get_session(self) -> Any:
        """Get a database session (provided by AsyncSessionMixin)."""
        ...


@runtime_checkable
class VectorBackendProtocol(Protocol):
    """Protocol for vector database backends (pgvector).

    Handles storage and retrieval of embeddings for semantic search.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connections."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        ...

    # Chunk operations
    @abstractmethod
    async def create_chunk(self, chunk: Chunk) -> Chunk:
        """Create a new chunk with its embedding."""
        ...

    @abstractmethod
    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """Create multiple chunks in a batch."""
        ...

    @abstractmethod
    async def get_chunk(self, chunk_id: UUID, *, namespace_id: UUID) -> Chunk | None:
        """Get a chunk by ID, scoped to ``namespace_id``.

        Returns ``None`` if the chunk does not exist OR belongs to a
        different namespace — the caller's namespace is the authority.
        The ``namespace_id`` filter prevents cross-tenant chunk access
        by id (IDOR).
        """
        ...

    @abstractmethod
    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        """Get multiple chunks by ID in a single query, scoped to ``namespace_id``.

        Args:
            chunk_ids: List of chunk IDs to fetch.
            namespace_id: Caller's namespace; chunks belonging to any
                other namespace are silently dropped from the result
                to prevent cross-tenant IDOR.

        Returns:
            Dictionary mapping chunk ID to Chunk (only for existing
            chunks within ``namespace_id``).
        """
        ...

    @abstractmethod
    async def get_chunks_by_document(self, document_id: UUID, *, namespace_id: UUID) -> list[Chunk]:
        """Get all chunks for a document, scoped to ``namespace_id``.

        Returns an empty list when the document does not belong to the
        caller's namespace. The namespace filter prevents cross-tenant
        chunk access by document id.
        """
        ...

    @abstractmethod
    async def delete_chunks_by_document(
        self,
        document_id: UUID,
        *,
        namespace_id: UUID,
        session: AsyncSession | None = None,
    ) -> int:
        """Delete all chunks for a document, scoped to ``namespace_id``.

        When *session* is provided the caller owns the transaction —
        no commit is issued.  When ``None``, a private session is used
        and committed automatically. The ``namespace_id`` filter prevents
        cross-tenant deletion by document id.
        """
        ...

    @abstractmethod
    async def search_similar(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        filter_document_ids: list[UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search for similar chunks using vector similarity.

        Returns list of (chunk, similarity_score) tuples.
        """
        ...

    # Entity operations (for vector search via PostgreSQL)
    @abstractmethod
    async def create_entity(self, entity: Entity) -> None:
        """Create an entity record in PostgreSQL for vector search."""
        ...

    @abstractmethod
    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> None:
        """Update an entity record in PostgreSQL, scoped to ``namespace_id``.

        Updates are skipped silently when the entity belongs to a different
        namespace — prevents cross-tenant entity mutation by id.
        """
        ...

    @abstractmethod
    async def entity_exists(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Check if an entity exists in PostgreSQL within ``namespace_id``.

        Returns ``False`` if the entity does not exist OR belongs to a
        different namespace. The ``namespace_id`` filter prevents
        cross-tenant entity-existence enumeration (IDOR).
        """
        ...

    @abstractmethod
    async def update_entity_embedding(
        self,
        entity_id: UUID,
        embedding: list[float],
        model: str,
        *,
        namespace_id: UUID,
    ) -> None:
        """Update the embedding for an entity, scoped to ``namespace_id``.

        Updates are skipped silently when the entity belongs to a different
        namespace — prevents cross-tenant embedding mutation.
        """
        ...

    async def update_entity_embeddings_batch(
        self,
        updates: list[tuple[UUID, list[float], str]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Update embeddings for multiple entities in a single transaction.

        Updates are restricted to the caller's namespace; ids outside it
        are silently skipped from the count.
        """
        ...

    @abstractmethod
    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[UUID, float]]:
        """Search for similar entities by embedding."""
        ...

    @abstractmethod
    async def search_fulltext(
        self,
        namespace_id: UUID,
        query_text: str,
        *,
        limit: int = 10,
        language: str = "english",
        filter_ast: FilterNode | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search chunks using PostgreSQL full-text search.

        Uses ts_rank on the content_tsv generated column.

        Returns list of (chunk, rank_score) tuples.

        ``filter_ast`` is the canonical recall-filter AST. The relational
        ``chunks`` table lacks the denormalized filter columns, so backends
        REFUSE under an active filter (return ``[]``) rather than smuggle
        unfiltered rows; the filtered BM25 path is the ``khora_chunks``
        temporal store.
        """
        ...

    # Aggregate operations (optional — have default implementations in VectorBackendBase)

    async def count_chunks(self, namespace_id: UUID) -> int:
        """Count chunks in a namespace."""
        ...

    async def list_chunks(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        """List chunks in a namespace."""
        ...


@runtime_checkable
class GraphBackendProtocol(Protocol):
    """Protocol for graph database backends (Neo4j).

    Handles storage and traversal of the knowledge graph.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the database."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connections."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and connected."""
        ...

    # Entity operations
    @abstractmethod
    async def create_entity(self, entity: Entity) -> Entity:
        """Create an entity node in the graph."""
        ...

    @abstractmethod
    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None:
        """Get an entity by ID, scoped to ``namespace_id``.

        Returns ``None`` if the entity does not exist OR belongs to a
        different namespace. Prevents cross-tenant entity access by id
        (IDOR).
        """
        ...

    @abstractmethod
    async def get_entity_by_name(self, namespace_id: UUID, name: str, entity_type: str) -> Entity | None:
        """Get an entity by name and type (for deduplication)."""
        ...

    @abstractmethod
    async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> Entity:
        """Update an entity, scoped to ``namespace_id``.

        Updates are skipped when the entity belongs to a different
        namespace — prevents cross-tenant entity mutation by id.
        """
        ...

    @abstractmethod
    async def delete_entity(self, entity_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete an entity and its relationships, scoped to ``namespace_id``.

        Returns ``False`` if the entity does not exist OR belongs to a
        different namespace. Prevents cross-tenant deletion by id.
        """
        ...

    @abstractmethod
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

        ``source_chunk_ids`` filters by chunk provenance (#1448):

        - ``None`` (default): no filter — behavior byte-identical to today.
        - Non-empty list: return only entities whose ``source_chunk_ids``
          contains AT LEAST ONE of the given ids (any-overlap, not subset).
        - Empty list ``[]``: return ``[]`` (matches nothing).

        Composes with AND against all existing conditions (``entity_type``,
        live/``valid_until`` filters) and applies BEFORE ``limit``/``offset``.
        """
        ...

    # Relationship operations
    @abstractmethod
    async def create_relationship(self, relationship: Relationship) -> Relationship:
        """Create a relationship between entities."""
        ...

    @abstractmethod
    async def get_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> Relationship | None:
        """Get a relationship by ID, scoped to ``namespace_id``.

        Returns ``None`` if the relationship does not exist OR belongs to
        a different namespace. Prevents cross-tenant relationship access
        by id (IDOR).
        """
        ...

    @abstractmethod
    async def delete_relationship(self, relationship_id: UUID, *, namespace_id: UUID) -> bool:
        """Delete a relationship, scoped to ``namespace_id``.

        Returns ``False`` if the relationship does not exist OR belongs to
        a different namespace. Prevents cross-tenant deletion by id.
        """
        ...

    @abstractmethod
    async def get_entity_relationships(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        direction: str = "both",  # "outgoing", "incoming", "both"
        relationship_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Get relationships for an entity, scoped to ``namespace_id``.

        Returns an empty list if the entity does not belong to the
        caller's namespace. Edges that cross into other namespaces are
        excluded from the result. Prevents cross-tenant subgraph leakage
        (IDOR family).
        """
        ...

    @abstractmethod
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

        ``between_entity_ids`` filters by endpoint membership (#1451):

        - ``None`` (default): no filter — behavior byte-identical to today.
        - Non-empty list: return only relationships whose BOTH endpoints
          (source AND target) are in the given id set.
        - Empty list ``[]``: return ``[]`` (matches nothing).

        Composes with AND against all existing conditions (``relationship_type``,
        live/tombstone filters) and applies BEFORE ``limit``/``offset``.
        """
        ...

    # Episode operations
    @abstractmethod
    async def create_episode(self, episode: Episode) -> Episode:
        """Create an episode node."""
        ...

    @abstractmethod
    async def get_episode(self, episode_id: UUID, *, namespace_id: UUID) -> Episode | None:
        """Get an episode by ID, scoped to ``namespace_id``.

        Returns ``None`` if the episode does not exist OR belongs to a
        different namespace. Prevents cross-tenant episode access by id
        (IDOR).
        """
        ...

    @abstractmethod
    async def list_episodes(
        self,
        namespace_id: UUID,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        """List episodes in a time range."""
        ...

    # Graph traversal
    @abstractmethod
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
        ...

    @abstractmethod
    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the neighborhood of an entity up to a certain depth,
        scoped to ``namespace_id``.

        The seed entity is verified to belong to ``namespace_id``; the
        traversal MUST NOT cross into other namespaces. Returns an empty
        structure when the seed is in a different namespace. Prevents
        cross-tenant subgraph leakage (IDOR family).
        """
        ...

    @abstractmethod
    async def search_entities_by_attribute(
        self,
        namespace_id: UUID,
        attribute_name: str,
        attribute_value: Any,
        *,
        limit: int = 100,
    ) -> list[Entity]:
        """Search entities by attribute value."""
        ...

    # Batch and aggregate operations (optional — have default implementations in GraphBackendBase)

    async def get_entities_batch(self, entity_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Entity]:
        """Fetch multiple entities in a single query, scoped to ``namespace_id``.

        Entities belonging to any other namespace are silently dropped
        from the result to prevent cross-tenant IDOR (IDOR family).

        Returns dictionary mapping entity ID to Entity object.
        """
        ...

    async def get_neighborhoods_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit_per_entity: int = 20,
    ) -> dict[UUID, dict[str, Any]]:
        """Get neighborhoods for multiple entities, scoped to ``namespace_id``.

        Seed entities outside ``namespace_id`` are silently dropped; the
        traversal MUST NOT cross into other namespaces. Prevents
        cross-tenant subgraph leakage (IDOR family).

        Returns dictionary mapping entity ID to neighborhood data.
        """
        ...

    async def count_entities(self, namespace_id: UUID) -> int:
        """Count entities in a namespace."""
        ...

    async def count_relationships(self, namespace_id: UUID) -> int:
        """Count relationships in a namespace."""
        ...

    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 100,
        bulk_mode: bool = False,
    ) -> list[tuple[Entity, bool]]:
        """Batch upsert entities using MERGE semantics.

        For each entity, creates it if new or updates if existing
        (matched by name + type within namespace). On match the input
        entity's ``id`` MUST be synced in place to the stored id so
        relationship endpoints resolve (the #806 id-remap contract).

        Returns list of (entity, is_new) tuples.
        """
        ...

    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 100,
    ) -> list[tuple[Relationship, bool]]:
        """Batch create relationships using MERGE semantics.

        Returns one ``(relationship, is_new)`` tuple per *persisted* edge,
        mirroring ``upsert_entities_batch``'s ``(entity, is_new)`` contract.
        On a dedup-merge onto an existing edge the input relationship's ``id``
        MUST be synced in place to the stored edge's canonical id (the #806
        id-remap contract, applied to edges) so callers - notably the
        ``relationship.created`` / ``relationship.updated`` semantic-hook
        dispatch (#1320) - report the actually-stored id, never the submitted
        one. ``is_new`` is ``True`` for a genuine create and ``False`` for a
        merge.

        Backends whose write path cannot cheaply distinguish create from
        merge (SurrealDB's bare ``RELATE``, the per-record ``GraphBackendBase``
        default used by Neptune/AGE) return a best-effort ``is_new=True`` with
        the canonical id and document the limitation; the MERGE-by-endpoint
        backends (Neo4j, Memgraph) report the split exactly, and the
        ON-CONFLICT(id) backends (pgvector, sqlite_lance) report it from the
        id-collision they key on.
        """
        ...

    # Dream bi-temporal mirror verbs (#1271) — optional. The dream-apply
    # phase mirrors its PG-side soft-delete / rewrite / relabel to the graph
    # through these. They are dream-predicate-keyed (confidence + chunk
    # liveness, document-independent), NOT the document-replace-shaped
    # ``retire_orphaned_*`` primitives. ``GraphBackendBase`` provides a
    # capability-gated default that raises ``DreamBackendUnsupported`` so a
    # backend without native support degrades to a structured skip_reason
    # rather than silently no-op-ing or hard-deleting. The mirror wiring into
    # the orchestrator is #1272; this seam only declares the contract.

    def supports_dream_mirror(self) -> frozenset[OpKind]:
        """The ``OpKind`` values this backend can mirror to the graph.

        The dream orchestrator (#1272) intersects the plan's op kinds with
        this set; ops outside it record a structured skip_reason instead of
        diverging the two stores. Empty set = no graph-mirror support.
        """
        ...

    async def soft_invalidate_relationships_batch(
        self,
        relationship_ids: list[UUID],
        *,
        namespace_id: UUID,
        invalidated_at: datetime,
    ) -> int:
        """Soft-delete relationships by id by stamping ``valid_until``.

        Mirrors ``prune_edges`` (the dream predicate: low-confidence +
        chunk-dead edges). Idempotent by id (only edges with a null
        ``valid_until`` are touched), namespace-scoped. Never hard-deletes.
        Returns the number of edges actually invalidated.
        """
        ...

    async def soft_retire_entities_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
        retired_at: datetime,
        reason: str = "dream_consolidated",
    ) -> int:
        """Soft-retire entities by id, snapshotting the pre-state.

        Mirrors the absorbed-entity soft-delete in ``dedupe_entities``:
        snapshots the live node into a version record and stamps
        ``valid_until`` / ``version_valid_to`` on the original. Idempotent
        by id (only still-live entities are retired), namespace-scoped.
        Never hard-deletes. Returns the number of entities actually retired.
        """
        ...

    async def rewrite_relationship_endpoints_batch(
        self,
        rewrites: list[dict[str, Any]],
        *,
        namespace_id: UUID,
        rewritten_at: datetime,
    ) -> int:
        """Re-point relationship endpoints by id.

        Mirrors the absorbed-endpoint rewrite in ``dedupe_entities``. Each
        dict carries ``relationship_id``, ``source_entity_id``,
        ``target_entity_id`` (the post-rewrite endpoints), and
        ``relationship_type`` (the Cypher edge label - sanitized by backends
        that store types as labels). Idempotent by id, namespace-scoped.
        Returns the number of edges actually re-pointed.
        """
        ...

    async def rename_types_batch(
        self,
        renames: list[dict[str, str]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Relabel relationship types (Cypher edge labels).

        Mirrors ``normalize_schema``. Each dict carries ``old_type`` and
        ``new_type``. The relationship type is a Cypher edge label and CANNOT
        be ``$``-parameterized, so backends MUST route both ends through the
        shared ``sanitize_cypher_label`` hard-validation. Namespace-scoped.
        Returns the number of edges relabeled.
        """
        ...

    # Dream graph-mirror REVERSE verbs (#1275) — optional. ``dream_undo``
    # reverses the PG soft-deletes; these reverse the matching forward graph
    # mirror so undo restores PG and graph to identical pre-apply live sets
    # rather than a half-revert. Same capability-gated default contract as the
    # forward verbs above (``GraphBackendBase`` raises ``DreamBackendUnsupported``
    # so a backend without a native reverse degrades to a structured skip rather
    # than diverging). Idempotent by id; empty input short-circuits to 0.

    async def restore_entities_batch(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> int:
        """Un-retire entities soft-retired by :meth:`soft_retire_entities_batch`.

        Reverses the absorbed-entity soft-retire in ``dedupe_entities``: clears
        the node's ``valid_until`` / ``version_valid_to`` tombstone AND deletes
        ONLY the :EntityVersion snapshot + [:SUPERSEDES] edge the forward mirror
        created for this retire, so the node returns to the live set without
        losing any pre-existing version chain. The forward mirror stamps the
        snapshot's ``version_valid_to`` with the same retire timestamp it writes
        to the node's ``valid_until``, so the reverse targets the snapshot by
        ``version_valid_to == valid_until`` (no separate snapshot-id plumbing
        needed). Matched by entity id within ``namespace_id`` (IDOR family);
        idempotent (a node already live transitions nothing). Returns the number
        of entities actually restored.
        """
        ...

    async def restore_relationships_batch(
        self,
        relationship_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> int:
        """Un-invalidate relationships invalidated by :meth:`soft_invalidate_relationships_batch`.

        Reverses the prune / self-loop invalidation by clearing the edge's
        ``valid_until``. Matched by relationship id within ``namespace_id``
        (IDOR family); idempotent (an edge already live matches nothing).
        Returns the number of edges actually restored.
        """
        ...

    async def restore_relationship_endpoints_batch(
        self,
        rewrites: list[dict[str, Any]],
        *,
        namespace_id: UUID,
    ) -> int:
        """Re-point relationship endpoints back to their pre-rewrite endpoints.

        Reverses :meth:`rewrite_relationship_endpoints_batch`. Each dict carries
        ``relationship_id``, ``source_entity_id``, ``target_entity_id`` (the
        PRE-rewrite endpoints to restore), and ``relationship_type`` (the Cypher
        edge label — sanitized by backends that store types as labels).
        Idempotent by id, namespace-scoped. Returns the number of edges
        actually re-pointed.
        """
        ...

    # Dream community materialization (#1276) - the GraphRAG payoff. The dream
    # ``community_summary`` op persists LLM-grounded summaries to PG; this verb
    # materializes them into the graph as :Community nodes + [:HAS_MEMBER] edges
    # so they are queryable at recall. Capability-gated through
    # ``supports_dream_mirror()`` (advertises ``VECTORCYPHER_COMMUNITY_SUMMARY``).

    async def materialize_communities_batch(
        self,
        communities: list[CommunityNode],
        *,
        namespace_id: UUID,
        materialized_at: datetime,
    ) -> int:
        """MERGE :Community nodes + [:HAS_MEMBER] edges to member :Entity nodes.

        Mirrors ``community_summary``. Each :class:`CommunityNode` carries the
        community id, summary text, member entity ids, summary depth, and an
        optional embedding. Idempotent on community id (MERGE on the id keeps a
        re-run / reconciler replay from creating duplicates); namespace-scoped.
        HAS_MEMBER edges are only created to member entities that exist in the
        graph within the namespace. Returns the number of communities upserted.
        """
        ...

    async def get_communities(
        self,
        namespace_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CommunityNode]:
        """Return materialized :Community summary nodes for a namespace.

        The community-level recall reader (#1276). Read-only; returns the
        summary text + member ids so callers can surface community context.
        """
        ...

    async def get_entity_communities(
        self,
        entity_ids: list[UUID],
        *,
        namespace_id: UUID,
    ) -> list[CommunityNode]:
        """Return the :Community nodes the given entities are HAS_MEMBER of.

        The entity-anchored leg of the community recall reader (#1276): given a
        recall hit's entity set, fetch the community summaries they belong to.
        Namespace-scoped; deduplicated by community id.
        """
        ...


@runtime_checkable
class EventStoreProtocol(Protocol):
    """Protocol for event store backends.

    Handles the append-only event log for event sourcing.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the store."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connections."""
        ...

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the store is healthy."""
        ...

    @abstractmethod
    async def append_event(self, event: MemoryEvent) -> MemoryEvent:
        """Append an event to the log."""
        ...

    @abstractmethod
    async def append_events_batch(self, events: list[MemoryEvent]) -> list[MemoryEvent]:
        """Append multiple events in a batch."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def get_events_for_resource(
        self,
        resource_type: str,
        resource_id: UUID,
        *,
        namespace_id: UUID,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """Get all events for a specific resource, scoped to ``namespace_id``.

        Returns an empty list if the resource belongs to a different
        namespace. Prevents cross-tenant audit-log leakage (the IDOR family /
        the IDOR family family).
        """
        ...

    @abstractmethod
    async def get_latest_event(
        self,
        resource_type: str,
        resource_id: UUID,
        *,
        namespace_id: UUID,
    ) -> MemoryEvent | None:
        """Get the latest event for a resource, scoped to ``namespace_id``.

        Returns ``None`` if the resource belongs to a different namespace.
        Prevents cross-tenant audit-log leakage (the IDOR family / the IDOR family family).
        """
        ...

    @abstractmethod
    async def count_events(
        self,
        namespace_id: UUID,
        *,
        event_types: list[str] | None = None,
        after: datetime | None = None,
    ) -> int:
        """Count events matching criteria."""
        ...
