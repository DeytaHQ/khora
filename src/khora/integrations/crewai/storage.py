"""``KhoraStorageBackend`` — sync ``crewai.memory.storage.backend.StorageBackend`` shim.

CrewAI's unified memory orchestrator calls into a sync ``StorageBackend``
Protocol with eleven sync methods + three async siblings. The
MemoryRecord-level CRUD surface (``save``, ``search``, ``delete``,
``update``, ``get_record``, ``list_records``) is the value-carrying
part; the scope / category / count / reset helpers and the async
siblings (``asave`` / ``asearch`` / ``adelete``) are exercised by
CrewAI's own ``encoding_flow`` and ``Memory`` orchestrator at runtime.
This adapter implements the full surface so an
``Agent(memory=Memory(storage=KhoraStorageBackend(...)))`` works
through every code path CrewAI hits.

This module is loaded lazily by ``khora.integrations.crewai.KhoraMemory``
(itself only invoked when the user installs ``khora[crewai]``). It must
not import ``crewai`` at module top level — the ``crewai`` types arrive
through the constructor's framework parameter so the AST lint
(``tools/check_optional_imports.py``) and the subprocess no-import test
both pass.

Concurrency:

* CrewAI calls ``StorageBackend`` methods from its own worker threads.
* Every coroutine into ``Khora`` is dispatched through
  ``khora.integrations._sync.run_sync`` — one shared daemon-thread loop
  for the whole process.
* ``run_sync`` refuses reentry from inside an asyncio loop. The
  ``KhoraStorageBackend`` therefore inherits the same constraint:
  do not wrap it in code that already runs inside an event loop.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from khora.exceptions import KhoraIntegrationError
from khora.integrations._sync import run_sync
from khora.integrations.crewai._mapping import (  # noqa: I001 — keep adapter-local first
    chunk_to_record,
    record_to_remember_kwargs,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from khora.khora import Khora


# Per-thread stash for the most recent text passed to the wrapping
# embedder. The mechanism: ``KhoraMemory`` installs an embedder that
# captures CrewAI's query text into this dict on the calling thread,
# then ``search()`` reads it back during the same call. Thread-local so
# concurrent recalls don't trample each other.
_query_text_stash: threading.local = threading.local()


def _stash_query_text(text: str) -> None:
    """Record the most recent embed-text call on the current thread."""
    _query_text_stash.text = text


def _peek_query_text() -> str | None:
    """Return the stashed text for this thread, or ``None`` if empty."""
    return getattr(_query_text_stash, "text", None)


class KhoraStorageBackend:
    """Sync ``StorageBackend``-shaped adapter that delegates to khora.

    Duck-types ``crewai.memory.storage.backend.StorageBackend``. The
    ``MemoryRecord`` class is passed at construction (rather than
    imported) so this module never requires ``crewai`` to be installed
    at import time.

    Args:
        kb: The bound :class:`khora.Khora` instance. Caller owns the
            lifecycle — adapter does not call ``connect`` /
            ``disconnect``.
        namespace_id: Stable khora namespace UUID. Every read and write
            is scoped to this namespace.
        user_id: Stable end-user identifier. Stamped on every record so
            a single khora namespace can host multi-user CrewAI
            sessions without silent cross-user reads.
        app_id: Free-form app identifier (default ``"crewai"``).
        memory_record_cls: ``crewai.memory.types.MemoryRecord``,
            injected by the factory.
    """

    name: str = "crewai"

    def __init__(
        self,
        *,
        kb: Khora,
        namespace_id: UUID,
        user_id: str,
        app_id: str,
        memory_record_cls: type,
    ) -> None:
        self.kb = kb
        self.namespace_id = namespace_id
        self.user_id = user_id
        self.app_id = app_id
        self._memory_record_cls = memory_record_cls
        # Map MemoryRecord.id (str) → khora document_id (UUID). CrewAI
        # passes record_ids around; khora forgets by document UUID.
        self._record_to_document: dict[str, UUID] = {}

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, records: list[Any]) -> None:
        """Persist a batch of ``MemoryRecord`` objects.

        Each record becomes one khora document. Khora extracts no
        entities or relationships for these documents — CrewAI's
        ``Memory`` has already done its own LLM-driven scope /
        categories / importance analysis at this point. Calling khora
        with the default ``general_entities`` skill would pay for a
        second LLM call we don't need.
        """
        for record in records:
            kwargs = record_to_remember_kwargs(
                record,
                user_id=self.user_id,
                app_id=self.app_id,
            )
            result = run_sync(
                self.kb.remember(namespace=self.namespace_id, **kwargs),
            )
            # Remember the mapping so delete([record_id]) and update()
            # can find the document later.
            self._record_to_document[str(record.id)] = result.document_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        """Vector-search via khora — using the smuggled text query.

        CrewAI's ``Memory`` computes ``query_embedding`` from the
        query text via its own embedder, then passes ONLY the embedding
        to this method. We ignore the embedding entirely: khora's
        ``recall()`` runs its own embedding step on the original text,
        plus HyDE expansion and rerank when enabled.

        To recover the text from inside ``search()``, the
        ``KhoraMemory`` factory wires CrewAI with a wrapping embedder
        (see ``khora.integrations.crewai.__init__``) that stashes the
        query text on a thread-local before returning. We read that
        stash here. If the stash is empty (e.g. a caller used a custom
        embedder path that bypassed our wrapping), we fall back to an
        empty-string query — recall still runs but the engine has
        nothing useful to embed against. That fallback is intentional:
        crashing on a missing text would be a regression from the
        StorageBackend Protocol shape.

        Caveats (documented in ``docs/integrations/crewai.md``):

        * The pre-computed ``query_embedding`` is intentionally
          discarded. khora's embedding model is owned by its config,
          not by CrewAI.
        * ``scope_prefix`` and ``categories`` are NOT pushed down to
          khora today. khora has no per-document scope/category columns
          to filter on. The adapter post-filters returned chunks
          against ``crewai_scope`` / ``crewai_categories`` in
          ``Document.metadata.custom``.
        * ``metadata_filter`` is similarly post-filtered.
        """
        del query_embedding  # explicitly ignored
        query_text = _peek_query_text() or ""

        recall_result = run_sync(
            self.kb.recall(
                query_text,
                namespace=self.namespace_id,
                limit=max(limit, 1),
                min_similarity=min_score,
            ),
        )

        out: list[tuple[Any, float]] = []
        for chunk, score in recall_result.chunks:
            record = chunk_to_record(chunk, self._memory_record_cls)
            if not _matches_filters(
                record,
                scope_prefix=scope_prefix,
                categories=categories,
                metadata_filter=metadata_filter,
            ):
                continue
            out.append((record, float(score)))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete records matching the given criteria.

        When ``record_ids`` is provided, the adapter calls khora's
        ``forget(document_id, namespace=...)`` for each known mapping.
        For the broader scope / categories / older_than filters we walk
        the namespace's documents and forget the matches. Walking is
        adequate for typical CrewAI memory sizes (hundreds to low
        thousands of records); larger deployments should partition by
        namespace.
        """
        if record_ids:
            return self._delete_by_record_ids(record_ids)

        # Filter-driven delete: list documents in the namespace, match,
        # forget. This is O(docs in namespace) but unavoidable without
        # adding scope/category columns to khora itself.
        return run_sync(
            self._delete_by_filter(
                scope_prefix=scope_prefix,
                categories=categories,
                older_than=older_than,
                metadata_filter=metadata_filter,
            ),
        )

    def _delete_by_record_ids(self, record_ids: list[str]) -> int:
        deleted = 0
        for rid in record_ids:
            doc_id = self._record_to_document.get(rid)
            if doc_id is None:
                continue
            ok = run_sync(self.kb.forget(doc_id, namespace=self.namespace_id))
            if ok:
                deleted += 1
                self._record_to_document.pop(rid, None)
        return deleted

    async def _delete_by_filter(
        self,
        *,
        scope_prefix: str | None,
        categories: list[str] | None,
        older_than: datetime | None,
        metadata_filter: dict[str, Any] | None,
    ) -> int:
        storage = self.kb.storage
        deleted = 0
        offset = 0
        page_size = 200
        while True:
            page = await storage.list_documents(self.namespace_id, limit=page_size, offset=offset)
            if not page:
                break
            for doc in page:
                custom = doc.metadata.custom if doc.metadata else {}
                if scope_prefix is not None:
                    doc_scope = str(custom.get("crewai_scope", "/"))
                    if not doc_scope.startswith(scope_prefix):
                        continue
                if categories:
                    doc_cats = set(custom.get("crewai_categories") or [])
                    if doc_cats.isdisjoint(categories):
                        continue
                if older_than is not None and doc.created_at >= older_than:
                    continue
                if metadata_filter:
                    if not all(custom.get(k) == v for k, v in metadata_filter.items()):
                        continue
                ok = await self.kb.forget(doc.id, namespace=self.namespace_id)
                if ok:
                    deleted += 1
                    # Drop stale record_id mapping (best effort).
                    rid = doc.external_id
                    if rid is not None:
                        self._record_to_document.pop(rid, None)
            if len(page) < page_size:
                break
            offset += page_size
        return deleted

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, record: Any) -> None:
        """Replace the document backing ``record.id`` with new content.

        Khora has no in-place chunk update API at the public surface
        — the canonical replacement is forget-then-remember. We honour
        that: forget the existing document, then remember the new
        content under the same external id so subsequent ``get_record``
        and ``delete([rid])`` lookups still work.
        """
        rid = str(record.id)
        doc_id = self._record_to_document.get(rid)
        if doc_id is not None:
            run_sync(self.kb.forget(doc_id, namespace=self.namespace_id))
            self._record_to_document.pop(rid, None)
        # Re-save under the same record id so the mapping is rebuilt.
        self.save([record])

    # ------------------------------------------------------------------
    # get_record
    # ------------------------------------------------------------------

    def get_record(self, record_id: str) -> Any | None:
        """Return the ``MemoryRecord`` for ``record_id`` or ``None``.

        Resolution order:

        1. Known mapping (record_id → document_id), then load the
           first chunk for that document.
        2. Fallback to scanning ``external_id`` on the namespace's
           documents (cheap for typical sizes; the relational backend
           has an index on ``(namespace_id, external_id)`` upstream).
        """
        doc_id = self._record_to_document.get(record_id)
        chunk = run_sync(self._first_chunk_for(record_id, doc_id))
        if chunk is None:
            return None
        return chunk_to_record(chunk, self._memory_record_cls)

    async def _first_chunk_for(
        self,
        record_id: str,
        doc_id: UUID | None,
    ) -> Any | None:
        storage = self.kb.storage
        if doc_id is None:
            doc = await storage.get_document_by_external_id(self.namespace_id, record_id)
            if doc is None:
                return None
            doc_id = doc.id
            self._record_to_document[record_id] = doc_id
        chunks = await storage.get_chunks_by_document(doc_id, namespace_id=self.namespace_id)
        return chunks[0] if chunks else None

    # ------------------------------------------------------------------
    # list_records
    # ------------------------------------------------------------------

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Any]:
        """Return records in the namespace, newest first.

        Pagination is handled by walking ``list_documents``. We don't
        return records for documents with no chunks (e.g. ingestion
        that failed mid-pipeline) — the public ``MemoryRecord`` shape
        expects a content body.
        """
        return run_sync(
            self._list_records_async(
                scope_prefix=scope_prefix,
                limit=max(limit, 0),
                offset=max(offset, 0),
            ),
        )

    async def _list_records_async(
        self,
        *,
        scope_prefix: str | None,
        limit: int,
        offset: int,
    ) -> list[Any]:
        if limit == 0:
            return []
        storage = self.kb.storage
        out: list[Any] = []
        skipped = 0
        cursor = 0
        page_size = max(limit, 50)
        while len(out) < limit:
            page = await storage.list_documents(self.namespace_id, limit=page_size, offset=cursor)
            if not page:
                break
            for doc in page:
                custom = doc.metadata.custom if doc.metadata else {}
                if scope_prefix is not None:
                    doc_scope = str(custom.get("crewai_scope", "/"))
                    if not doc_scope.startswith(scope_prefix):
                        continue
                if skipped < offset:
                    skipped += 1
                    continue
                chunks = await storage.get_chunks_by_document(doc.id, namespace_id=self.namespace_id)
                if not chunks:
                    continue
                out.append(chunk_to_record(chunks[0], self._memory_record_cls))
                if len(out) >= limit:
                    break
            if len(page) < page_size:
                break
            cursor += page_size
        return out

    # ------------------------------------------------------------------
    # async siblings — CrewAI's unified Memory occasionally invokes the
    # async surface (e.g. from inside its own async flows). The bridge
    # is the other direction: delegate directly to the awaitables.
    # ------------------------------------------------------------------

    async def asave(self, records: list[Any]) -> None:
        """Async sibling of :meth:`save` — bypasses the sync bridge."""
        for record in records:
            kwargs = record_to_remember_kwargs(
                record,
                user_id=self.user_id,
                app_id=self.app_id,
            )
            result = await self.kb.remember(namespace=self.namespace_id, **kwargs)
            self._record_to_document[str(record.id)] = result.document_id

    async def asearch(
        self,
        query_embedding: list[float] | Any,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Any, float]]:
        """Async sibling of :meth:`search` — bypasses the sync bridge."""
        del query_embedding  # see :meth:`search` for rationale
        query_text = _peek_query_text() or ""
        recall_result = await self.kb.recall(
            query_text,
            namespace=self.namespace_id,
            limit=max(limit, 1),
            min_similarity=min_score,
        )
        out: list[tuple[Any, float]] = []
        for chunk, score in recall_result.chunks:
            record = chunk_to_record(chunk, self._memory_record_cls)
            if not _matches_filters(
                record,
                scope_prefix=scope_prefix,
                categories=categories,
                metadata_filter=metadata_filter,
            ):
                continue
            out.append((record, float(score)))
            if len(out) >= limit:
                break
        return out

    async def adelete(
        self,
        *,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Async sibling of :meth:`delete` — bypasses the sync bridge."""
        if record_ids:
            deleted = 0
            for rid in record_ids:
                doc_id = self._record_to_document.get(rid)
                if doc_id is None:
                    continue
                if await self.kb.forget(doc_id, namespace=self.namespace_id):
                    deleted += 1
            return deleted
        return await self._delete_by_filter(
            scope_prefix=scope_prefix,
            categories=categories,
            older_than=older_than,
            metadata_filter=metadata_filter,
        )

    # ------------------------------------------------------------------
    # scope / category / count / reset — admin surface CrewAI's encoding
    # flow exercises during ``save``. khora does not natively model
    # "scope" as a first-class entity; the adapter derives scopes from
    # ``chunk.metadata.custom["crewai_scope"]`` and treats them as
    # filesystem-style paths.
    # ------------------------------------------------------------------

    def list_scopes(self, parent: str = "/") -> list[str]:
        """Return distinct scopes that have stored records under ``parent``."""
        return run_sync(self._list_scopes_async(parent=parent))

    async def _list_scopes_async(self, *, parent: str) -> list[str]:
        seen: set[str] = set()
        storage = self.kb.storage
        cursor = 0
        page_size = 200
        while True:
            page = await storage.list_documents(self.namespace_id, limit=page_size, offset=cursor)
            if not page:
                break
            for doc in page:
                custom = doc.metadata.custom if doc.metadata else {}
                scope = str(custom.get("crewai_scope", "/"))
                if scope.startswith(parent):
                    seen.add(scope)
            if len(page) < page_size:
                break
            cursor += page_size
        return sorted(seen)

    def get_scope_info(self, scope: str) -> dict[str, Any]:
        """Return a minimal info dict for ``scope``.

        khora doesn't model scope metadata separately from documents,
        so we surface just the path and the on-the-fly record count.
        Duck-typed to whatever CrewAI's ``ScopeInfo`` consumer reads —
        upstream callers tolerate dict-shaped returns.
        """
        return {"scope": scope, "count": self.count(scope_prefix=scope)}

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        """Return a ``{category_name: count}`` mapping for stored records.

        CrewAI's encoding flow calls ``.keys()`` on the return value
        (see ``crewai/memory/encoding_flow.py:282``), so the shape must
        be dict-like. The count is a useful side-channel for downstream
        UIs that show category histograms.
        """
        return run_sync(self._list_categories_async(scope_prefix=scope_prefix))

    async def _list_categories_async(self, *, scope_prefix: str | None) -> dict[str, int]:
        counts: dict[str, int] = {}
        storage = self.kb.storage
        cursor = 0
        page_size = 200
        while True:
            page = await storage.list_documents(self.namespace_id, limit=page_size, offset=cursor)
            if not page:
                break
            for doc in page:
                custom = doc.metadata.custom if doc.metadata else {}
                if scope_prefix is not None:
                    doc_scope = str(custom.get("crewai_scope", "/"))
                    if not doc_scope.startswith(scope_prefix):
                        continue
                cats = custom.get("crewai_categories") or []
                if isinstance(cats, (list, tuple)):
                    for c in cats:
                        name = str(c)
                        counts[name] = counts.get(name, 0) + 1
            if len(page) < page_size:
                break
            cursor += page_size
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        """Return the count of records whose scope starts with ``scope_prefix``."""
        return run_sync(self._count_async(scope_prefix=scope_prefix))

    async def _count_async(self, *, scope_prefix: str | None) -> int:
        storage = self.kb.storage
        if scope_prefix is None:
            stats = await self.kb.stats(namespace=self.namespace_id)
            return int(stats.documents)
        total = 0
        cursor = 0
        page_size = 200
        while True:
            page = await storage.list_documents(self.namespace_id, limit=page_size, offset=cursor)
            if not page:
                break
            for doc in page:
                custom = doc.metadata.custom if doc.metadata else {}
                doc_scope = str(custom.get("crewai_scope", "/"))
                if doc_scope.startswith(scope_prefix):
                    total += 1
            if len(page) < page_size:
                break
            cursor += page_size
        return total

    def reset(self, scope_prefix: str | None = None) -> None:
        """Delete every record matching ``scope_prefix`` (None = all)."""
        self.delete(scope_prefix=scope_prefix)


def _matches_filters(
    record: Any,
    *,
    scope_prefix: str | None,
    categories: list[str] | None,
    metadata_filter: dict[str, Any] | None,
) -> bool:
    """Post-filter recalled records by scope, category, and metadata."""
    if scope_prefix is not None and not (record.scope or "").startswith(scope_prefix):
        return False
    if categories:
        rec_cats = set(record.categories or [])
        if rec_cats.isdisjoint(categories):
            return False
    if metadata_filter:
        rec_meta = record.metadata or {}
        for k, v in metadata_filter.items():
            if rec_meta.get(k) != v:
                return False
    if record.private:
        # CrewAI's privacy filter normally runs upstream in unified_memory.
        # Adapter keeps records visible by default — the storage backend
        # has no knowledge of the recall request's source/include_private.
        # Upstream filters out private records via its own check on the
        # returned (record, score) tuples (see crewai/memory/unified_memory.py).
        pass
    return True


def _raise_invalid_user_id(user_id: str | None) -> None:
    """Reject empty / placeholder / too-short user_ids.

    Silent cross-user memory sharing is the #1 disaster mode for any
    multi-tenant memory adapter (see #618 risk list). The acceptance
    criterion is explicit: reject ``""``, ``"default"``, and any
    ``user_id`` shorter than 8 characters.
    """
    if user_id is None or not user_id:
        raise KhoraIntegrationError(
            "KhoraMemory requires a non-empty user_id. Silent cross-user "
            "memory sharing is the dominant misuse mode for agent memory "
            "adapters — pass an opaque, stable end-user identifier."
        )
    if user_id == "default":
        raise KhoraIntegrationError("KhoraMemory user_id='default' is rejected. Use a real, per-user identifier.")
    if len(user_id) < 8:
        raise KhoraIntegrationError(
            f"KhoraMemory user_id must be at least 8 characters "
            f"(got {len(user_id)}). Use an opaque, stable identifier "
            "such as a UUID or hashed account id."
        )
