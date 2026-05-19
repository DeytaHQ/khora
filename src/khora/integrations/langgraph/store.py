"""``KhoraStore`` — LangGraph ``BaseStore`` implementation backed by khora.

Maps LangGraph's ``(namespace: tuple[str, ...], key: str, value: dict)``
contract onto khora's ``Khora.remember`` / ``Khora.recall`` /
``Khora.forget`` surface. A single ``KhoraStore`` binds one ``Khora``
instance and one logical user (``user_id``), and all LangGraph
namespace tuples map onto the same khora ``namespace_id`` (a UUID5
derived from ``namespace_root`` + ``user_id``). The tuple itself round-
trips through ``Document.metadata["lg_namespace"]``.

Scope (per #624): this module ships ``KhoraStore`` only.
A LangGraph ``Checkpointer`` adapter is intentionally NOT included —
``langgraph-postgres``'s ``PostgresSaver`` already covers that surface
and khora has no differentiator there.

Sync surface (``put`` / ``get`` / ``search`` / ``delete`` /
``list_namespaces`` / ``batch``) routes through
``khora.integrations._sync.run_sync``. That helper rejects reentrant
calls — so sync access from inside a running LangGraph event loop is
not supported. Use the async methods from inside a graph.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from loguru import logger

from khora.integrations._sync import run_sync
from khora.integrations.langgraph._mapping import (
    composite_external_id,
    flatten_namespace,
    item_from_metadata,
    item_metadata,
    namespace_uuid,
    value_to_content,
)

if TYPE_CHECKING:
    # Framework + khora imports kept under TYPE_CHECKING so the module
    # imports cleanly even when langgraph isn't installed. The runtime
    # framework imports live inside method bodies (see ``_ensure_base``).
    from langgraph.store.base import (
        BaseStore,
        IndexConfig,
        Item,
        Op,
        Result,
        SearchItem,
    )

    from khora.khora import Khora


# Minimum length for a non-default user_id. Disaster-mode prevention per
# #618: empty / "default" / very short user_ids all silently cross-share
# memory between LangGraph runs.
_MIN_USER_ID_LEN = 8

# Disallowed user_id values. "default" is the historical CrewAI default
# that the adapter authors learned to reject the hard way.
_BANNED_USER_IDS = frozenset({"", "default", "anon", "anonymous", "user", "test"})


@dataclass(slots=True)
class _StoreInit:
    """Resolved init arguments. Kept as a small struct so the constructor
    body stays a flat sequence of validations."""

    namespace_root: str
    app_id: str
    user_id: str
    namespace_id: UUID
    namespace_sep: str
    skill_name: str
    entity_types: list[str]
    relationship_types: list[str]


def _validate_user_id(user_id: str) -> None:
    """Reject user_id values that risk silent cross-user memory sharing."""
    if not isinstance(user_id, str):
        raise TypeError(f"user_id must be a string, got {type(user_id).__name__}")
    if user_id.strip() != user_id:
        raise ValueError(f"user_id must not have leading/trailing whitespace: {user_id!r}")
    if user_id.lower() in _BANNED_USER_IDS:
        raise ValueError(
            f"user_id {user_id!r} is in the disallowed list "
            f"({sorted(_BANNED_USER_IDS)!r}). Pass a stable, caller-specific "
            f"identifier — empty / generic defaults cause silent memory sharing."
        )
    if len(user_id) < _MIN_USER_ID_LEN:
        raise ValueError(
            f"user_id {user_id!r} is shorter than {_MIN_USER_ID_LEN} chars. "
            f"Short user_ids are easy to collide accidentally; use a UUID or "
            f"the framework's stable user identifier."
        )


def _import_langgraph_base():
    """Lazy import of langgraph.store.base.

    Centralised so all method-body imports route through one helper —
    nicer error message when the extra isn't installed.
    """
    try:
        from langgraph.store import base as _base  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "KhoraStore requires the optional `langgraph` extra. Install with: pip install 'khora[langgraph]'"
        ) from exc
    return _base


def _resolve_base_store_class() -> type[BaseStore]:
    """Return the concrete ``BaseStore`` class at runtime.

    Kept as a module-level helper so the class hierarchy is built once on
    first ``KhoraStore`` instantiation (see ``__new__``).
    """
    return _import_langgraph_base().BaseStore


# We construct the subclass dynamically on first use so this module never
# imports langgraph at top level. ``_KhoraStoreBase`` is the runtime class
# substituted into the MRO on first access.
_KhoraStoreBase: type | None = None


def _get_base_class() -> type:
    """Return (and cache) the dynamically-resolved ``BaseStore`` class."""
    global _KhoraStoreBase
    if _KhoraStoreBase is None:
        _KhoraStoreBase = _resolve_base_store_class()
    return _KhoraStoreBase


class KhoraStore:
    """LangGraph ``BaseStore`` backed by khora.

    Args:
        kb: A connected :class:`khora.Khora` instance. The store does NOT
            own the connection lifecycle — the caller does.
        user_id: Stable identifier for the memory owner. Must be a
            non-default, non-empty, >= 8-char string. Empty / generic
            values are rejected to prevent silent cross-user sharing
            (#618 disaster-mode prevention).
        namespace_root: Sub-key under which this app's LangGraph
            namespaces live. Different roots map onto different khora
            namespaces for the same user. Default ``"user_id"``.
        app_id: Free-form app identifier embedded in stored metadata.
            Default ``"langgraph"``.
        namespace_sep: Separator used to flatten LangGraph tuple
            namespaces into a single string. Must not appear inside any
            tuple segment. Default ``"/"``.
        index_config: Optional LangGraph ``IndexConfig``. Only the
            ``dims`` field is consulted — it MUST match khora's
            configured embedding dimension or construction raises
            ``ValueError`` (fail-fast).
        skill_name: khora extraction skill name (default
            ``"general_entities"``).
        entity_types / relationship_types: extraction whitelist
            forwarded to ``Khora.remember``. Empty lists are valid and
            disable extraction entirely — useful for pure KV blob
            storage.

    Concurrency: every async method touches khora through async
    primitives, so concurrent calls are safe to the same khora connection
    pool. Sync methods bridge through a single-thread daemon loop and
    serialise per call.

    Stability: experimental. The ``KhoraStore`` class name and its
    ``BaseStore`` contract are public, but implementation details
    (metadata field names, composite-key format, namespace UUID5 root)
    may change without a major-version bump until the first stable
    minor.
    """

    name: str = "langgraph"
    """Identifier for ``khora.integrations`` registry / telemetry."""

    # We deliberately do NOT support TTL — khora has no native TTL on
    # documents/chunks (#620's session GC is the closest, but it's
    # session-scoped, not per-item). The base class flips this flag to
    # ``False`` already; redeclaring here makes intent obvious.
    supports_ttl: bool = False
    ttl_config: dict[str, Any] | None = None

    def __init__(
        self,
        kb: Khora,
        *,
        user_id: str,
        namespace_root: str = "user_id",
        app_id: str = "langgraph",
        namespace_sep: str = "/",
        index_config: IndexConfig | dict[str, Any] | None = None,
        skill_name: str = "general_entities",
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
    ) -> None:
        # Resolve the LangGraph base class once and rebind self.__class__
        # to a subclass that has it in the MRO. This is the cleanest way
        # to defer the framework import to first instantiation while
        # still letting LangGraph's internals see ``isinstance(store,
        # BaseStore)``.
        base_cls = _get_base_class()
        if base_cls not in type(self).__mro__:
            # Build a new class that subclasses both. Cached on the
            # KhoraStore class so the first call pays the cost once.
            new_cls = type(
                "KhoraStore",
                (KhoraStore, base_cls),
                {},
            )
            self.__class__ = new_cls

        _validate_user_id(user_id)

        if not namespace_sep:
            raise ValueError("namespace_sep must be a non-empty string")
        if len(namespace_sep) != 1:
            # Multi-char separators are technically fine but invite bugs
            # when callers compose them by accident. Stick to single-char
            # for v1; relax if a real caller asks.
            raise ValueError(f"namespace_sep must be a single character, got {namespace_sep!r}")

        ns_uuid = namespace_uuid(namespace_root=namespace_root, user_id=user_id)

        # Validate IndexConfig.dims against khora's embedder dim (fail-fast).
        if index_config is not None:
            dims = index_config.get("dims") if isinstance(index_config, dict) else getattr(index_config, "dims", None)
            if dims is not None:
                khora_dim = kb._config.llm.embedding_dimension
                if int(dims) != int(khora_dim):
                    raise ValueError(
                        f"IndexConfig.dims={dims} does not match khora's configured "
                        f"embedding_dimension={khora_dim}. Reconfigure khora or pass "
                        f"index_config={{'dims': {khora_dim}, ...}}."
                    )

        self.kb = kb
        self._init = _StoreInit(
            namespace_root=namespace_root,
            app_id=app_id,
            user_id=user_id,
            namespace_id=ns_uuid,
            namespace_sep=namespace_sep,
            skill_name=skill_name,
            entity_types=list(entity_types) if entity_types is not None else [],
            relationship_types=list(relationship_types) if relationship_types is not None else [],
        )
        # Track which LangGraph namespace tuples we've written. The store
        # falls back to a SQL-via-list_documents scan in
        # ``alist_namespaces`` but a successful put updates this set so
        # subsequent listings are O(1) for the common case.
        self._seen_namespaces: set[tuple[str, ...]] = set()
        # One-time TTL warning per Store instance so log output stays
        # bounded even under tight loops.
        self._ttl_warned = False
        # One-time index=False warning per Store instance. We accept the
        # arg but cannot honour it — khora always embeds.
        self._index_false_warned = False

    # ------------------------------------------------------------------
    # Public stable attrs (KhoraIntegration marker Protocol)
    # ------------------------------------------------------------------

    @property
    def namespace_id(self) -> UUID:
        """Stable khora namespace UUID5 derived from (root, user_id)."""
        return self._init.namespace_id

    @property
    def user_id(self) -> str:
        """The validated user_id this store is bound to."""
        return self._init.user_id

    # ------------------------------------------------------------------
    # Async core (the 6 methods every BaseStore implements)
    # ------------------------------------------------------------------

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None = None,
    ) -> None:
        """Store an item. Backing call: ``Khora.remember``.

        ``ttl`` is accepted but ignored — khora has no per-item TTL. A
        one-time warning is emitted per ``KhoraStore`` instance.

        ``index=False`` is accepted but ignored — khora always embeds.
        A one-time warning is emitted per ``KhoraStore`` instance. The
        item is still retrievable via ``aget`` / ``alist_namespaces``.
        """
        self._validate_lg_namespace(namespace)
        if not isinstance(key, str) or not key:
            raise ValueError(f"key must be a non-empty string, got {key!r}")
        if not isinstance(value, dict):
            raise TypeError(f"value must be a dict, got {type(value).__name__}")

        if ttl is not None and not self._ttl_warned:
            warnings.warn(
                "KhoraStore does not honour TTL — khora has no per-item expiry. "
                "Use Khora.forget_session(...) to bulk-delete a session, or "
                "schedule your own cleanup. This warning is shown once per Store.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._ttl_warned = True

        if index is False and not self._index_false_warned:
            warnings.warn(
                "KhoraStore ignores `index=False`: khora always embeds new "
                "documents. Items are still retrievable. This warning is "
                "shown once per Store.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._index_false_warned = True

        await self._ensure_namespace()
        flat = flatten_namespace(namespace, self._init.namespace_sep)
        external_id = composite_external_id(flat, key, self._init.namespace_sep)
        meta = item_metadata(namespace, key, value, sep=self._init.namespace_sep)
        meta["lg_app_id"] = self._init.app_id

        content = value_to_content(value)

        # If a document already exists at this external_id, delete it
        # first — khora's remember() short-circuits on duplicate checksum
        # which would yield a stale entry on overwrite semantics
        # LangGraph callers expect.
        existing = await self.kb.storage.get_document_by_external_id(self._init.namespace_id, external_id)
        if existing is not None:
            await self.kb.forget(existing.id, namespace=self._init.namespace_id)

        await self.kb.remember(
            content,
            namespace=self._init.namespace_id,
            title=key,
            source=f"langgraph:{self._init.app_id}",
            metadata=meta,
            skill_name=self._init.skill_name,
            entity_types=self._init.entity_types,
            relationship_types=self._init.relationship_types,
            external_id=external_id,
        )
        self._seen_namespaces.add(tuple(namespace))

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,  # accepted, ignored
    ) -> Item | None:
        """Retrieve an item by ``(namespace, key)``.

        Backing call: ``Khora.storage.get_document_by_external_id``.
        Returns ``None`` if no such item exists or the matched document
        was not written through this adapter.
        """
        self._validate_lg_namespace(namespace)
        await self._ensure_namespace()
        flat = flatten_namespace(namespace, self._init.namespace_sep)
        external_id = composite_external_id(flat, key, self._init.namespace_sep)

        document = await self.kb.storage.get_document_by_external_id(self._init.namespace_id, external_id)
        if document is None:
            return None
        unpacked = item_from_metadata(document)
        if unpacked is None:
            return None
        ns_tuple, k, value, created_at, updated_at = unpacked
        # Defensive: only return when the stored namespace matches what
        # the caller asked for. Composite-key collisions are unlikely but
        # not impossible (hash-prefixed long namespaces).
        if ns_tuple != tuple(namespace):
            return None
        Item = _import_langgraph_base().Item
        return Item(
            value=value,
            key=k,
            namespace=ns_tuple,
            created_at=created_at,
            updated_at=updated_at,
        )

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,  # accepted, ignored
    ) -> list[SearchItem]:
        """Search items under a namespace prefix.

        When ``query`` is provided, runs ``Khora.recall`` and maps chunks
        back to ``SearchItem`` instances. When ``query`` is ``None``,
        falls back to a ``list_documents`` scan — slower but covers the
        no-query LangGraph filter-only case.

        ``filter`` is applied client-side against the stored
        ``lg_value`` dict (exact match per key). Operator filters
        (``$gt`` etc) are NOT supported in v1; future work could push
        them down to JSONB at the SQL layer.
        """
        await self._ensure_namespace()

        SearchItem = _import_langgraph_base().SearchItem
        results: list[SearchItem] = []

        if query is not None:
            # Vector recall path. Fetch a generous pool then prefix-filter
            # so we still return ``limit`` items after rejecting chunks
            # outside the requested LangGraph prefix.
            pool = max(limit * 4 + offset, limit)
            recall = await self.kb.recall(query, namespace=self._init.namespace_id, limit=pool)
            # Build the doc-id → metadata lookup: per the recall projection
            # contract, per-chunk metadata is no longer surfaced — the
            # custom keys (lg_namespace, lg_key, lg_value) live on the
            # parent ``DocumentProjection.metadata``.
            docs_by_id = {doc.id: doc for doc in recall.documents}
            seen_keys: set[tuple[tuple[str, ...], str]] = set()
            for chunk in recall.chunks:
                doc = docs_by_id.get(chunk.document_id)
                custom = (doc.metadata if doc else None) or {}
                ns_raw = custom.get("lg_namespace")
                key = custom.get("lg_key")
                value = custom.get("lg_value", {})
                if ns_raw is None or key is None:
                    continue
                try:
                    ns_tuple = tuple(str(s) for s in ns_raw)
                except TypeError:
                    continue
                if namespace_prefix and not _has_prefix(ns_tuple, namespace_prefix):
                    continue
                marker = (ns_tuple, str(key))
                if marker in seen_keys:
                    # One document → many chunks. Keep the highest score
                    # (chunks come back score-sorted).
                    continue
                seen_keys.add(marker)
                if not _matches_filter(value if isinstance(value, dict) else {}, filter):
                    continue
                # Reconstruct timestamps from the chunk's parent doc
                # lazily. For perf we use chunk.created_at as a stand-in
                # — it's the same value khora populates on doc creation.
                ts = chunk.created_at
                results.append(
                    SearchItem(
                        namespace=ns_tuple,
                        key=str(key),
                        value=value if isinstance(value, dict) else {},
                        created_at=ts,
                        updated_at=ts,
                        score=float(chunk.score) if chunk.score is not None else None,
                    )
                )
                if len(results) >= limit + offset:
                    break
        else:
            # No-query branch: list documents, project them client-side,
            # filter by prefix and ``filter`` dict. O(N) scan — acceptable
            # for v1; matches LangGraph's InMemoryStore behaviour.
            documents = await self.kb.list_documents(namespace=self._init.namespace_id, limit=max(limit + offset, 100))
            for document in documents:
                projected = item_from_metadata(document)
                if projected is None:
                    continue
                ns_tuple, key, value, created_at, updated_at = projected
                if namespace_prefix and not _has_prefix(ns_tuple, namespace_prefix):
                    continue
                if not _matches_filter(value, filter):
                    continue
                results.append(
                    SearchItem(
                        namespace=ns_tuple,
                        key=key,
                        value=value,
                        created_at=created_at,
                        updated_at=updated_at,
                        score=None,
                    )
                )
                if len(results) >= limit + offset:
                    break

        return results[offset : offset + limit]

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item by ``(namespace, key)``.

        No-op (no error) if the item doesn't exist, matching LangGraph's
        ``InMemoryStore`` semantics.
        """
        self._validate_lg_namespace(namespace)
        await self._ensure_namespace()
        flat = flatten_namespace(namespace, self._init.namespace_sep)
        external_id = composite_external_id(flat, key, self._init.namespace_sep)
        document = await self.kb.storage.get_document_by_external_id(self._init.namespace_id, external_id)
        if document is None:
            return
        await self.kb.forget(document.id, namespace=self._init.namespace_id)

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List distinct LangGraph namespaces.

        Implementation: lists all documents in the bound khora namespace
        and aggregates distinct ``lg_namespace`` tuples from metadata.

        Cost note: this is an O(N_documents) scan. khora does not expose
        a SQL-pushdown helper for ``SELECT DISTINCT metadata->'lg_namespace'
        FROM documents`` yet; adding one is tracked as future work. For
        bounded LangGraph workloads (one user, dozens of session
        namespaces) the scan is fine. At >= O(10⁴) documents, callers
        should switch to a dedicated tracking table.
        """
        await self._ensure_namespace()
        # Pull a wide page — khora has no DISTINCT helper, so we
        # over-fetch to compensate. ``limit`` here gates the document
        # page size, not the returned namespace count.
        fetch = max(limit * 8 + offset, 200)
        documents = await self.kb.list_documents(namespace=self._init.namespace_id, limit=fetch)
        seen: set[tuple[str, ...]] = set()
        for document in documents:
            projected = item_from_metadata(document)
            if projected is None:
                continue
            ns_tuple = projected[0]
            if max_depth is not None and max_depth >= 0:
                ns_tuple = ns_tuple[:max_depth]
            if prefix and not _has_prefix(ns_tuple, prefix, wildcard="*"):
                continue
            if suffix and not _has_suffix(ns_tuple, suffix, wildcard="*"):
                continue
            seen.add(ns_tuple)
        # In-memory namespaces from this session (e.g. just-written
        # via ``aput``) should also surface even before they hit the DB
        # read scope.
        for ns_tuple in self._seen_namespaces:
            if max_depth is not None and max_depth >= 0:
                ns_tuple = ns_tuple[:max_depth]
            if prefix and not _has_prefix(ns_tuple, prefix, wildcard="*"):
                continue
            if suffix and not _has_suffix(ns_tuple, suffix, wildcard="*"):
                continue
            seen.add(ns_tuple)
        # Stable ordering: lexicographic on the joined form.
        ordered = sorted(seen, key=lambda t: self._init.namespace_sep.join(t))
        return ordered[offset : offset + limit]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Dispatch loop over the per-op async methods.

        v1 implementation: serial dispatch, one op at a time. LangGraph's
        ``InMemoryStore`` runs ops in parallel; we keep it serial for
        determinism and so a partial-failure raises on the first failed
        op rather than after-the-fact. Future work: batch consecutive
        ``PutOp`` calls into a single ``remember_batch``.
        """
        base = _import_langgraph_base()
        GetOp = base.GetOp
        PutOp = base.PutOp
        SearchOp = base.SearchOp
        ListNamespacesOp = base.ListNamespacesOp

        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self.aget(op.namespace, op.key))
            elif isinstance(op, PutOp):
                if op.value is None:
                    await self.adelete(op.namespace, op.key)
                    results.append(None)
                else:
                    await self.aput(op.namespace, op.key, op.value, op.index)
                    results.append(None)
            elif isinstance(op, SearchOp):
                results.append(
                    await self.asearch(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                prefix = None
                suffix = None
                if op.match_conditions:
                    for mc in op.match_conditions:
                        if mc.match_type == "prefix":
                            prefix = mc.path
                        elif mc.match_type == "suffix":
                            suffix = mc.path
                results.append(
                    await self.alist_namespaces(
                        prefix=prefix,
                        suffix=suffix,
                        max_depth=op.max_depth,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            else:
                raise TypeError(f"Unsupported LangGraph Op type: {type(op).__name__}")
        return results

    # ------------------------------------------------------------------
    # Sync surface (bridged through khora.integrations._sync.run_sync)
    # ------------------------------------------------------------------

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None = None,
    ) -> None:
        run_sync(self.aput(namespace, key, value, index, ttl=ttl))

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return run_sync(self.aget(namespace, key, refresh_ttl=refresh_ttl))

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return run_sync(
            self.asearch(
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )
        )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        run_sync(self.adelete(namespace, key))

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return run_sync(
            self.alist_namespaces(
                prefix=prefix,
                suffix=suffix,
                max_depth=max_depth,
                limit=limit,
                offset=offset,
            )
        )

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        # Materialise ops up front — the abatch dispatch is async and we
        # can't yield mid-loop on a sync caller's iterator without
        # deadlock surface.
        return run_sync(self.abatch(list(ops)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_lg_namespace(self, namespace: tuple[str, ...]) -> None:
        """Delegate to LangGraph's own validator + reject our sep char."""
        base = _import_langgraph_base()
        base._validate_namespace(namespace)
        for segment in namespace:
            if self._init.namespace_sep in segment:
                raise ValueError(
                    f"LangGraph namespace segment {segment!r} contains the "
                    f"configured separator {self._init.namespace_sep!r}. "
                    f"Change KhoraStore(namespace_sep=...) or rename the segment."
                )

    async def _ensure_namespace(self) -> None:
        """Lazily create the khora namespace row.

        Idempotent: a second call after a successful first call is a
        no-op. Falls back gracefully if the row already exists (race-
        safe enough for v1 — concurrent constructors converge on the
        same UUID5).
        """
        try:
            await self.kb._resolve_namespace(self._init.namespace_id)
            return
        except ValueError:
            pass
        # No active namespace yet — create one with our deterministic
        # namespace_id. Set ``id`` equal to ``namespace_id`` so the row-
        # level resolver finds it (v1 has one version only).
        from khora.core.models.tenancy import MemoryNamespace  # noqa: PLC0415

        ns = MemoryNamespace(
            id=self._init.namespace_id,
            namespace_id=self._init.namespace_id,
            metadata={
                "source": "khora.integrations.langgraph",
                "user_id": self._init.user_id,
                "namespace_root": self._init.namespace_root,
                "app_id": self._init.app_id,
            },
        )
        try:
            await self.kb.storage.create_namespace(ns)
        except Exception as exc:  # pragma: no cover - race-safe creation
            # Another caller may have raced us. If the namespace now
            # resolves, swallow; otherwise re-raise so the caller sees
            # the real DB error.
            try:
                await self.kb._resolve_namespace(self._init.namespace_id)
            except ValueError:
                raise exc from None
            logger.debug("KhoraStore namespace creation race resolved cleanly: {}", exc)


def _has_prefix(
    ns: tuple[str, ...],
    prefix: tuple[str, ...],
    *,
    wildcard: str = "*",
) -> bool:
    """Return True if ``ns`` starts with ``prefix`` (``*`` matches any segment)."""
    if len(prefix) > len(ns):
        return False
    for got, want in zip(ns[: len(prefix)], prefix):
        if want != wildcard and got != want:
            return False
    return True


def _has_suffix(
    ns: tuple[str, ...],
    suffix: tuple[str, ...],
    *,
    wildcard: str = "*",
) -> bool:
    """Return True if ``ns`` ends with ``suffix`` (``*`` matches any segment)."""
    if len(suffix) > len(ns):
        return False
    for got, want in zip(ns[-len(suffix) :], suffix):
        if want != wildcard and got != want:
            return False
    return True


def _matches_filter(value: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    """Client-side LangGraph filter — exact match on each key.

    Operator filters (``$gt`` etc) are NOT supported in v1 and silently
    fall back to value-equality (so ``{"score": {"$gt": 5}}`` will only
    match items where ``value["score"] == {"$gt": 5}``). Adding operator
    support is a clean future addition — gate it behind a feature flag
    on the constructor when a caller needs it.
    """
    if not filter:
        return True
    for key, expected in filter.items():
        if value.get(key) != expected:
            return False
    return True
