"""Turbopuffer backend for the Skeleton engine.

Serverless vector + BM25 store backed by object storage. Pitched at the
"large-scale recall with pay-per-query" niche above what we'd run on
pgvector or self-hosted Weaviate. Async-native SDK, Apache-2.0.

Issue #824 verdict was YELLOW (clean SDK + filters + license; SaaS-only;
client-side RRF for hybrid). Ship as opt-in, not default.

Three frictions vs the Weaviate backend's contract:

1. **Hybrid is client-side RRF.** turbopuffer's hybrid story is
   "multi-query": one HTTP request batches N independent ranked queries
   server-side, but results fuse on the client. The `TemporalVectorStore`
   protocol takes a `hybrid_alpha: float | None` linear-blend parameter;
   we ignore the alpha value (document this clearly) and apply RRF over
   the two channels' rank lists. Users who need true server-blended
   linear scores should stay on Weaviate.

2. **No reserved-character grammar for namespace names.** UUID hex is
   inside the safe set. Use `f"khora_{namespace_id.hex}"` so a
   `GET /v2/namespaces?prefix=khora_` enumerates only khora's tenants
   (helpful for GDPR delete-by-tenant).

3. **No native ALL-tags filter.** turbopuffer has `Contains` (one
   element) + `ContainsAny`. ALL-tags semantics fold into an `And` of
   N `Contains` clauses.

Wire details (verified against turbopuffer-python 2.x README + docs):

- Query: ``ns.query(rank_by=("vector","ANN",[...]), top_k=10, filters=("And",((...),(...))), include_attributes=[...])``
- BM25 query: ``ns.query(rank_by=("text","BM25","query text"), top_k=10, filters=...)``
- Write: ``client.namespaces.write(namespace=name, distance_metric="cosine_distance", upsert_rows=[...])``
- Delete by filter: ``client.namespaces.write(namespace=name, delete_by_filter={...})``
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger
from pydantic import SecretStr

from khora.engines.skeleton.backends import (
    TemporalChunk,
    TemporalFilter,
    TemporalSearchResult,
    TemporalVectorStore,
)

if TYPE_CHECKING:
    from khora.config import KhoraConfig
    from khora.filter.ast import FilterNode


# Distance metric we always use - matches our pre-normalized embeddings
# convention (CLAUDE.md: "L2-normalized at ingest"). Cosine distance with
# normalized vectors is equivalent to (1 - dot product) and is what every
# other backend resolves to on the score side.
_DISTANCE_METRIC = "cosine_distance"

# Tunable for client-side RRF on the hybrid path. 60 matches Weaviate's
# RELATIVE_SCORE fusion default and the literature norm (Cormack et al.).
_RRF_K = 60


@dataclass(frozen=True)
class TurbopufferBackendConfig:
    """Connection config for ``TurbopufferTemporalStore``.

    Attributes:
        api_key: API key for the turbopuffer API. Required (no
            anonymous access). Accepts ``str`` or ``SecretStr``.
        region: turbopuffer region slug, e.g. ``"gcp-us-central1"``.
            See https://turbopuffer.com/docs/regions for the current
            list. Cost + latency depend on the region; pick one near
            your khora workload.
        base_url: Override the default API endpoint. Almost never
            needed - turbopuffer routes via the region. Useful for
            mocking and for self-routed proxies.
        namespace_prefix: Prefix prepended to every namespace name
            (default ``"khora_"``). Lets ``GET /v2/namespaces?prefix=khora_``
            enumerate only the tenants this khora deployment writes,
            which simplifies cross-tenant audit and GDPR delete-by-tenant.
        ann_distance_threshold: Maximum vector distance (cosine distance,
            so 0=identical, 2=opposite) below which a search result is
            kept. ``None`` disables the threshold. Tied to ``min_similarity``
            on ``search()``: ``min_similarity=0.0`` keeps everything
            irrespective of this; values above 0 are translated to
            ``distance < 1 - min_similarity`` server-side when possible.
    """

    api_key: SecretStr | str
    region: str = "gcp-us-central1"
    base_url: str | None = None
    namespace_prefix: str = "khora_"
    ann_distance_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "TurbopufferBackendConfig requires an `api_key`. turbopuffer rejects anonymous connections."
            )

    def secret_api_key(self) -> str:
        """Return the API key as plain text."""
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key


def _coerce_backend_config(value: str | TurbopufferBackendConfig) -> TurbopufferBackendConfig:
    """Accept either a bare API-key string (back-compat) or a config object."""
    if isinstance(value, TurbopufferBackendConfig):
        return value
    if isinstance(value, str):
        return TurbopufferBackendConfig(api_key=value)
    raise TypeError(
        f"TurbopufferTemporalStore requires an api-key str or TurbopufferBackendConfig; got {type(value).__name__}"
    )


class TurbopufferTemporalStore(TemporalVectorStore):
    """Turbopuffer implementation of ``TemporalVectorStore``.

    One turbopuffer namespace per khora ``namespace_id``. Hybrid search
    is client-side RRF over a multi-query batch.
    """

    def __init__(
        self,
        config: KhoraConfig,
        turbopuffer_config: str | TurbopufferBackendConfig,
    ) -> None:
        """Initialize the backend.

        Args:
            config: Khora configuration (read for embedding dimension).
            turbopuffer_config: Either an API-key str (back-compat with
                the other backends' constructor shape) or a
                :class:`TurbopufferBackendConfig` for non-default region
                / namespace prefix / threshold tuning.
        """
        self._config = config
        self._tp_config = _coerce_backend_config(turbopuffer_config)
        self._client: Any = None  # AsyncTurbopuffer when connected
        self._connected = False
        self._embedding_dimension = config.llm.embedding_dimension or 1536

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to turbopuffer.

        turbopuffer doesn't carry a schema or collection bootstrap step
        like Weaviate does: namespaces are lazily created on first write,
        and attributes are auto-typed from the first row that introduces
        them. So ``connect()`` here just instantiates the async client
        and pins it.
        """
        if self._connected:
            return

        try:
            from turbopuffer import AsyncTurbopuffer
        except ImportError as exc:
            raise ImportError(
                "turbopuffer is required for the turbopuffer backend. "
                "Install with: pip install turbopuffer>=2.1.0 "
                "or: pip install khora[turbopuffer]"
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": self._tp_config.secret_api_key(),
            "region": self._tp_config.region,
        }
        if self._tp_config.base_url:
            kwargs["base_url"] = self._tp_config.base_url

        self._client = AsyncTurbopuffer(**kwargs)
        self._connected = True
        logger.info(
            "TurbopufferTemporalStore connected (region={region}, prefix={prefix!r})",
            region=self._tp_config.region,
            prefix=self._tp_config.namespace_prefix,
        )

    async def disconnect(self) -> None:
        """Disconnect from turbopuffer."""
        if self._client is not None:
            try:
                # AsyncTurbopuffer exposes ``close()`` to drain the underlying
                # httpx pool. Best-effort: some SDK versions name it ``aclose``.
                close = getattr(self._client, "close", None) or getattr(self._client, "aclose", None)
                if close is not None:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"AsyncTurbopuffer close raised: {exc}")
            self._client = None
        self._connected = False
        logger.info("TurbopufferTemporalStore disconnected")

    # ------------------------------------------------------------------
    # Namespace handle
    # ------------------------------------------------------------------

    def _namespace_name(self, namespace_id: UUID) -> str:
        """Map a khora namespace UUID to a turbopuffer namespace name."""
        return f"{self._tp_config.namespace_prefix}{namespace_id.hex}"

    def _namespace(self, namespace_id: UUID) -> Any:
        """Get the (async) namespace handle. No I/O - just a reference."""
        if not self._client:
            raise RuntimeError("TurbopufferTemporalStore is not connected")
        return self._client.namespace(self._namespace_name(namespace_id))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create_chunk(self, chunk: TemporalChunk) -> TemporalChunk:
        """Store a chunk with temporal metadata."""
        chunk.id = chunk.id or uuid4()
        await self._client.namespaces.write(
            namespace=self._namespace_name(chunk.namespace_id),
            distance_metric=_DISTANCE_METRIC,
            upsert_rows=[_chunk_to_row(chunk)],
        )
        return chunk

    async def create_chunks_batch(self, chunks: list[TemporalChunk]) -> list[TemporalChunk]:
        """Store multiple chunks.

        turbopuffer takes upserts in a single HTTP call - no need for
        per-chunk fan-out like Weaviate. Each namespace = one call.
        """
        if not chunks:
            return []

        # Group by namespace - one HTTP write per tenant.
        by_namespace: dict[UUID, list[TemporalChunk]] = {}
        for chunk in chunks:
            chunk.id = chunk.id or uuid4()
            by_namespace.setdefault(chunk.namespace_id, []).append(chunk)

        for namespace_id, ns_chunks in by_namespace.items():
            await self._client.namespaces.write(
                namespace=self._namespace_name(namespace_id),
                distance_metric=_DISTANCE_METRIC,
                upsert_rows=[_chunk_to_row(c) for c in ns_chunks],
            )

        return chunks

    async def get_chunk(self, chunk_id: UUID, namespace_id: UUID) -> TemporalChunk | None:
        """Get a chunk by ID.

        turbopuffer has no ``get_by_id`` primitive - the read path is the
        query path with an ``id Eq`` filter. We attach a constant ``rank_by``
        because the API requires one even on equality lookups.
        """
        ns = self._namespace(namespace_id)
        try:
            result = await ns.query(
                rank_by=("id", "asc"),
                top_k=1,
                filters=("id", "Eq", str(chunk_id)),
                include_attributes=_INCLUDE_ATTRS,
            )
            rows = getattr(result, "rows", None) or []
            if not rows:
                return None
            return _row_to_chunk(rows[0], namespace_id)
        except Exception as exc:
            logger.debug(f"Failed to get chunk {chunk_id}: {exc}")
            return None

    async def delete_chunk(self, chunk_id: UUID, namespace_id: UUID) -> bool:
        """Delete a chunk by ID."""
        try:
            await self._client.namespaces.write(
                namespace=self._namespace_name(namespace_id),
                delete_by_filter=("id", "Eq", str(chunk_id)),
            )
            return True
        except Exception as exc:
            logger.debug(f"Failed to delete chunk {chunk_id}: {exc}")
            return False

    async def delete_chunks_by_document(self, document_id: UUID, namespace_id: UUID) -> int:
        """Delete all chunks for a document.

        turbopuffer's ``delete_by_filter`` returns the row count in the
        response; surface it so callers can confirm.
        """
        try:
            response = await self._client.namespaces.write(
                namespace=self._namespace_name(namespace_id),
                delete_by_filter=("document_id", "Eq", str(document_id)),
            )
        except Exception as exc:
            logger.debug(f"Failed to delete chunks for document {document_id}: {exc}")
            return 0
        # The response shape carries ``rows_affected`` in 2.x.
        return int(getattr(response, "rows_affected", 0) or 0)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        temporal_filter: TemporalFilter | None = None,
        hybrid_alpha: float | None = None,
        query_text: str | None = None,
        filter_ast: FilterNode | None = None,
    ) -> list[TemporalSearchResult]:
        """Vector or hybrid (vector + BM25) search.

        Hybrid path:
            turbopuffer doesn't expose a server-blended linear alpha
            score; instead it offers multi-query batches that the client
            fuses with RRF. ``hybrid_alpha`` is **ignored** here - the
            blend is rank-based, not score-weighted. Document this in
            the operator docs so users picking turbopuffer know
            ``hybrid_alpha`` is a no-op on this backend.

        ``filter_ast`` is accepted for protocol parity; this backend does not
        compile the recall-filter AST yet, so it is ignored.
        """
        ns = self._namespace(namespace_id)
        tp_filter = _build_turbopuffer_filter(temporal_filter)

        # We always pull more than ``limit`` from each channel so RRF
        # has headroom. 3x is a common heuristic.
        per_channel_k = max(limit * 3, 20)

        if hybrid_alpha is not None and query_text:
            # Hybrid path: issue two independent ranked queries and fuse
            # by reciprocal rank. asyncio.gather sends them concurrently
            # over the same client; the SDK reuses the httpx connection.
            vec_task = ns.query(
                rank_by=("vector", "ANN", query_embedding),
                top_k=per_channel_k,
                filters=tp_filter,
                include_attributes=_INCLUDE_ATTRS,
            )
            text_task = ns.query(
                rank_by=("content", "BM25", query_text),
                top_k=per_channel_k,
                filters=tp_filter,
                include_attributes=_INCLUDE_ATTRS,
            )
            vec_result, text_result = await asyncio.gather(vec_task, text_task)
            fused = _rrf_fuse(
                vector_rows=getattr(vec_result, "rows", None) or [],
                text_rows=getattr(text_result, "rows", None) or [],
                k=_RRF_K,
                limit=limit,
            )
            search_results = []
            for row, rrf_score, vec_dist in fused:
                similarity = 1.0 - vec_dist if vec_dist is not None else rrf_score
                if similarity < min_similarity:
                    continue
                chunk = _row_to_chunk(row, namespace_id)
                search_results.append(
                    TemporalSearchResult(
                        chunk=chunk,
                        similarity=similarity,
                        bm25_score=row.get("$bm25_score") if isinstance(row, dict) else None,
                        combined_score=rrf_score,
                    )
                )
            return search_results

        # Vector-only path: single query, distance directly drives similarity.
        result = await ns.query(
            rank_by=("vector", "ANN", query_embedding),
            top_k=limit,
            filters=tp_filter,
            include_attributes=_INCLUDE_ATTRS,
        )
        rows = getattr(result, "rows", None) or []
        search_results = []
        for row in rows:
            dist = _row_get(row, "$dist")
            similarity = 1.0 - float(dist) if dist is not None else 0.0
            if similarity < min_similarity:
                continue
            chunk = _row_to_chunk(row, namespace_id)
            search_results.append(
                TemporalSearchResult(
                    chunk=chunk,
                    similarity=similarity,
                    bm25_score=None,
                    combined_score=similarity,
                )
            )
        return search_results

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Check backend health.

        turbopuffer exposes no ``/health`` route; "are we connected" is
        the strongest claim we can make without paying for a real query.
        """
        if not self._connected or not self._client:
            return {"status": "disconnected", "backend": "turbopuffer"}
        return {"status": "healthy", "backend": "turbopuffer"}


# ---------------------------------------------------------------------------
# Helpers (module-level for testability)
# ---------------------------------------------------------------------------


# Attribute names we want returned on every search/get. Excludes ``vector``
# (large; toggle separately if needed) and the SDK-private ``$dist``.
_INCLUDE_ATTRS = [
    "content",
    "document_id",
    "namespace_id",
    "occurred_at",
    "created_at",
    "source_system",
    "author",
    "channel",
    "tags",
    "confidence",
    "metadata_json",
]


def _chunk_to_row(chunk: TemporalChunk) -> dict[str, Any]:
    """Translate a TemporalChunk to turbopuffer's upsert-row dict shape."""
    occurred = chunk.occurred_at.isoformat() if chunk.occurred_at else None
    created = (chunk.created_at or datetime.now(UTC)).isoformat()
    return {
        "id": str(chunk.id),
        "vector": list(chunk.embedding) if chunk.embedding is not None else None,
        "content": chunk.content,
        "document_id": str(chunk.document_id),
        "namespace_id": str(chunk.namespace_id),
        "occurred_at": occurred,
        "created_at": created,
        "source_system": chunk.source_system,
        "author": chunk.author,
        "channel": chunk.channel,
        "tags": list(chunk.tags or []),
        "confidence": float(chunk.confidence) if chunk.confidence is not None else 1.0,
        "metadata_json": json.dumps(chunk.metadata or {}),
    }


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Tolerant accessor: rows arrive as dicts or as SDK Row objects."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_to_chunk(row: Any, namespace_id: UUID) -> TemporalChunk:
    """Translate a turbopuffer row to a ``TemporalChunk``."""
    metadata: dict[str, Any] = {}
    metadata_json = _row_get(row, "metadata_json")
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except (json.JSONDecodeError, TypeError):
            pass

    row_id = _row_get(row, "id")
    document_id = _row_get(row, "document_id")
    vector = _row_get(row, "vector")

    return TemporalChunk(
        id=UUID(str(row_id)),
        namespace_id=namespace_id,
        document_id=UUID(str(document_id)) if document_id else namespace_id,
        content=_row_get(row, "content", "") or "",
        embedding=list(vector) if vector else None,
        occurred_at=_coerce_datetime(_row_get(row, "occurred_at")),
        created_at=_coerce_datetime(_row_get(row, "created_at")),
        source_system=_row_get(row, "source_system"),
        author=_row_get(row, "author"),
        channel=_row_get(row, "channel"),
        tags=list(_row_get(row, "tags") or []),
        confidence=float(_row_get(row, "confidence", 1.0) or 1.0),
        metadata=metadata,
    )


def _coerce_datetime(value: Any) -> datetime | None:
    """ISO-string-or-datetime → datetime; ``None`` on unparseable input."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _build_turbopuffer_filter(f: TemporalFilter | None) -> Any:
    """Translate a ``TemporalFilter`` to turbopuffer's filter tuple grammar.

    turbopuffer's filter language is a recursive tuple form:
        ("And", (clause1, clause2, ...))
        ("Or",  (clause1, clause2, ...))
        ("Not", clause)
        (field, op, value)         # leaf clause, op ∈ Eq, Gte, Lte, ...

    Returns ``None`` if no predicates fire (the SDK treats ``None`` as
    "no filter"). Returns a single leaf tuple when only one predicate
    fires (saves an empty ``And`` wrapper). Returns an ``And`` of
    leaves otherwise.

    The ``tags`` filter expands to ``ALL tags must be present`` — N
    leaves under an ``And`` (turbopuffer has no native ``ContainsAll``).
    """
    if f is None:
        return None

    clauses: list[Any] = []

    if f.occurred_after:
        clauses.append(("occurred_at", "Gte", f.occurred_after.isoformat()))
    if f.occurred_before:
        clauses.append(("occurred_at", "Lt", f.occurred_before.isoformat()))
    if f.created_after:
        clauses.append(("created_at", "Gte", f.created_after.isoformat()))
    if f.created_before:
        clauses.append(("created_at", "Lt", f.created_before.isoformat()))
    if f.source_system:
        clauses.append(("source_system", "Eq", f.source_system))
    if f.author:
        clauses.append(("author", "Eq", f.author))
    if f.channel:
        clauses.append(("channel", "Eq", f.channel))
    if f.tags:
        # ALL-tags semantics — turbopuffer has no ContainsAll, so fold into N Contains.
        for tag in f.tags:
            clauses.append(("tags", "Contains", tag))

    for key, value in (f.additional or {}).items():
        if isinstance(value, dict):
            for op, val in value.items():
                clauses.append((key, _OP_MAP.get(op, op), val))
        else:
            clauses.append((key, "Eq", value))

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return ("And", tuple(clauses))


# Translate khora's filter op shorthand to turbopuffer's PascalCase ops.
# Anything not listed here is forwarded verbatim so callers can pass
# turbopuffer-native ops directly when they need to (e.g. "Glob", "Regex").
_OP_MAP = {
    "eq": "Eq",
    "neq": "NotEq",
    "ne": "NotEq",
    "gte": "Gte",
    "lte": "Lte",
    "gt": "Gt",
    "lt": "Lt",
    "in": "In",
    "not_in": "NotIn",
    "contains": "Contains",
    "contains_any": "ContainsAny",
}


def _rrf_fuse(
    *,
    vector_rows: list[Any],
    text_rows: list[Any],
    k: int,
    limit: int,
) -> list[tuple[Any, float, float | None]]:
    """Reciprocal Rank Fusion over two ranked lists.

    RRF formula (Cormack et al. 2009):
        score(row) = sum over channels of 1 / (k + rank_in_channel)

    Higher score = better. Returns ``[(row, rrf_score, vector_distance_if_any), ...]``
    truncated to ``limit``, sorted by rrf_score desc. The vector distance
    is preserved so the caller can keep computing a "similarity" field
    that's grounded in the cosine measure, not just the RRF rank score.
    """
    scores: dict[str, float] = {}
    rows_by_id: dict[str, Any] = {}
    vec_dist: dict[str, float | None] = {}

    for rank, row in enumerate(vector_rows, start=1):
        row_id = str(_row_get(row, "id"))
        scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (k + rank)
        rows_by_id[row_id] = row
        dist = _row_get(row, "$dist")
        vec_dist[row_id] = float(dist) if dist is not None else None

    for rank, row in enumerate(text_rows, start=1):
        row_id = str(_row_get(row, "id"))
        scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (k + rank)
        # If the row wasn't in the vector channel we still want it; use
        # the text-channel copy. Vector-distance stays ``None`` so the
        # caller knows the similarity field is RRF-derived only.
        rows_by_id.setdefault(row_id, row)
        vec_dist.setdefault(row_id, None)

    fused = sorted(
        ((rows_by_id[rid], scores[rid], vec_dist[rid]) for rid in scores),
        key=lambda t: t[1],
        reverse=True,
    )
    return fused[:limit]


__all__ = ["TurbopufferBackendConfig", "TurbopufferTemporalStore"]
