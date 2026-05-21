"""Weaviate backend for the Skeleton engine.

This backend provides:
- Native hybrid search (BM25 + vector in single query)
- Rich filtering on any property (timestamps, keywords, custom fields)
- Multi-tenancy with tenant isolation
- Horizontal scaling for large datasets

Async / auth / cloud (issue #783):

The Weaviate client is the v4 ``WeaviateAsyncClient`` (not the sync
``WeaviateClient`` we used until v0.16.2). Every storage method here
awaits the underlying client so the Skeleton engine event loop stays
unblocked under load.

Three deployment shapes are supported through ``WeaviateBackendConfig``:

- **Local**: pass ``WeaviateBackendConfig(url="http://localhost:8090")``
  or just ``WeaviateTemporalStore(config, "http://localhost:8090")``.
  Used by ``compose.yaml``'s ``weaviate`` profile.
- **Cloud**: ``WeaviateBackendConfig(cluster_url="https://...weaviate.network", api_key="...")``.
  Auth via Weaviate Cloud API key.
- **Custom / self-hosted**: pass ``url=`` plus optional ``grpc_port``,
  ``http_secure``, ``grpc_secure``, ``additional_headers``. API key
  optional. Use this when you run Weaviate behind a reverse proxy or on
  non-default ports.

The legacy string-only constructor ``WeaviateTemporalStore(config, "http://...")``
is preserved for back-compat; it wraps the URL in a default
``WeaviateBackendConfig``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

from loguru import logger
from pydantic import SecretStr

from khora.engines.skeleton.backends import (
    TemporalChunk,
    TemporalFilter,
    TemporalSearchResult,
    TemporalVectorStore,
)
from khora.storage._log_safe import _safe_url_for_log

if TYPE_CHECKING:
    from khora.config import KhoraConfig

# Collection name for temporal chunks
COLLECTION_NAME = "KhoraChunk"


@dataclass(frozen=True)
class WeaviateBackendConfig:
    """Connection config for ``WeaviateTemporalStore``.

    One of ``url`` (self-hosted) or ``cluster_url`` (Weaviate Cloud) must
    be set. Combining both is rejected at validation time.

    Attributes:
        url: HTTP endpoint for self-hosted Weaviate (e.g.
            ``http://localhost:8090``). Mutually exclusive with
            ``cluster_url``.
        cluster_url: Weaviate Cloud cluster URL (e.g.
            ``https://my-cluster.weaviate.network``). Mutually exclusive
            with ``url``. Requires ``api_key``.
        api_key: API key for auth. Required for ``cluster_url`` mode;
            optional for self-hosted (if the cluster is configured with
            ``AUTHENTICATION_APIKEY_ENABLED=true``). Accepts ``str`` or
            ``SecretStr``; stored as ``SecretStr``.
        grpc_port: gRPC port (self-hosted / custom only). Default 50051
            matches Weaviate's stock port; ``compose.yaml`` offsets to
            ``50061`` - pass it explicitly when running against the
            project's compose profile.
        http_secure: ``True`` to use TLS for the HTTP channel
            (self-hosted only; cloud is always TLS).
        grpc_secure: ``True`` to use TLS for the gRPC channel
            (self-hosted only).
        additional_headers: Optional headers to send with every request
            (e.g. ``{"X-OpenAI-Api-Key": "sk-..."}`` if a Weaviate module
            needs vendor credentials forwarded). Khora itself does not
            use vectorizer modules so this is rarely needed.
        skip_init_checks: When ``True`` the client skips its
            startup-readiness probe. Use only when the cluster is on a
            slow link and the readiness check times out spuriously.
    """

    url: str | None = None
    cluster_url: str | None = None
    api_key: SecretStr | str | None = None
    grpc_port: int = 50051
    http_secure: bool = False
    grpc_secure: bool = False
    additional_headers: dict[str, str] | None = None
    skip_init_checks: bool = False

    def __post_init__(self) -> None:
        if self.url and self.cluster_url:
            raise ValueError(
                "WeaviateBackendConfig: pass either `url` (self-hosted) or `cluster_url` (Weaviate Cloud), not both."
            )
        if not self.url and not self.cluster_url:
            raise ValueError(
                "WeaviateBackendConfig requires either `url` (self-hosted) or `cluster_url` (Weaviate Cloud)."
            )
        if self.cluster_url and self.api_key is None:
            raise ValueError(
                "WeaviateBackendConfig: `cluster_url` requires an `api_key`. "
                "Weaviate Cloud rejects anonymous connections."
            )

    @property
    def is_cloud(self) -> bool:
        """``True`` when the config targets Weaviate Cloud."""
        return bool(self.cluster_url)

    def secret_api_key(self) -> str | None:
        """Return the API key as plain text (or ``None`` if unset)."""
        if self.api_key is None:
            return None
        if isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return self.api_key

    def log_safe_endpoint(self) -> str:
        """Endpoint string suitable for log lines (no credentials)."""
        return _safe_url_for_log(self.cluster_url or self.url or "")


def _coerce_backend_config(value: str | WeaviateBackendConfig) -> WeaviateBackendConfig:
    if isinstance(value, WeaviateBackendConfig):
        return value
    if isinstance(value, str):
        return WeaviateBackendConfig(url=value)
    raise TypeError(f"WeaviateTemporalStore requires a URL str or WeaviateBackendConfig; got {type(value).__name__}")


class WeaviateTemporalStore(TemporalVectorStore):
    """Weaviate implementation of TemporalVectorStore.

    Provides native hybrid search combining BM25 and vector similarity
    with rich filtering capabilities. Uses the v4 async client so
    Skeleton engine I/O does not block the event loop.
    """

    def __init__(self, config: KhoraConfig, weaviate_url: str | WeaviateBackendConfig):
        """Initialize the backend.

        Args:
            config: Khora configuration.
            weaviate_url: Either a connection URL (back-compat,
                self-hosted) or a :class:`WeaviateBackendConfig` for
                cloud / authenticated / custom-port deployments.
        """
        self._config = config
        self._weaviate_config = _coerce_backend_config(weaviate_url)
        self._client: Any = None  # weaviate.WeaviateAsyncClient when connected
        self._connected = False
        self._tenants_seen: set[str] = set()
        self._embedding_dimension = config.llm.embedding_dimension or 1536

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Weaviate and ensure schema exists."""
        if self._connected:
            return

        try:
            import weaviate
            from weaviate.classes.config import Configure, DataType, Property
            from weaviate.classes.init import Auth
        except ImportError as exc:
            raise ImportError(
                "weaviate-client is required for the Weaviate backend. "
                "Install it with: pip install weaviate-client>=4.21.0 "
                "or: pip install khora[weaviate]"
            ) from exc

        cfg = self._weaviate_config
        api_key = cfg.secret_api_key()
        auth = Auth.api_key(api_key) if api_key else None

        if cfg.is_cloud:
            self._client = weaviate.use_async_with_weaviate_cloud(
                cluster_url=cfg.cluster_url,
                auth_credentials=auth,
                headers=cfg.additional_headers,
                skip_init_checks=cfg.skip_init_checks,
            )
        else:
            host, http_port = _parse_host_port(cfg.url or "")
            self._client = weaviate.use_async_with_custom(
                http_host=host,
                http_port=http_port,
                http_secure=cfg.http_secure,
                grpc_host=host,
                grpc_port=cfg.grpc_port,
                grpc_secure=cfg.grpc_secure,
                auth_credentials=auth,
                headers=cfg.additional_headers,
                skip_init_checks=cfg.skip_init_checks,
            )

        # v4 async clients require an explicit ``connect()`` before use.
        # The sync ``connect_to_*`` helpers do this implicitly; the async
        # ``use_async_with_*`` helpers do not.
        await self._client.connect()

        # Create collection if it doesn't exist
        if not await self._client.collections.exists(COLLECTION_NAME):
            await self._client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),  # We provide embeddings
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="document_id", data_type=DataType.UUID),
                    Property(name="namespace_id", data_type=DataType.UUID),
                    Property(name="occurred_at", data_type=DataType.DATE),
                    Property(name="created_at", data_type=DataType.DATE),
                    Property(name="source_system", data_type=DataType.TEXT),
                    Property(name="author", data_type=DataType.TEXT),
                    Property(name="channel", data_type=DataType.TEXT),
                    Property(name="tags", data_type=DataType.TEXT_ARRAY),
                    Property(name="confidence", data_type=DataType.NUMBER),
                    Property(name="metadata_json", data_type=DataType.TEXT),  # JSON string
                ],
                multi_tenancy_config=Configure.multi_tenancy(enabled=True),
            )
            logger.info(f"Created Weaviate collection: {COLLECTION_NAME}")

        self._connected = True
        logger.info(
            "WeaviateTemporalStore connected ({mode}) to {url}",
            mode="cloud" if cfg.is_cloud else "self-hosted",
            url=cfg.log_safe_endpoint(),
        )

    async def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"WeaviateAsyncClient close raised: {exc}")
            self._client = None
        self._tenants_seen.clear()
        self._connected = False
        logger.info("WeaviateTemporalStore disconnected")

    # -------------------------------------------------------------------
    # Tenant-scoped collection accessor
    # -------------------------------------------------------------------

    async def _get_collection(self, namespace_id: UUID) -> Any:
        """Get collection with tenant context.

        Ensures the tenant exists in Weaviate before returning the
        tenant-scoped collection handle. Tenant existence is cached
        in-process so we don't pay the create RTT on every call.
        """
        if not self._client:
            raise RuntimeError("WeaviateTemporalStore is not connected")

        from weaviate.classes.tenants import Tenant

        collection = self._client.collections.get(COLLECTION_NAME)
        tenant_name = str(namespace_id)

        if tenant_name not in self._tenants_seen:
            try:
                await collection.tenants.create([Tenant(name=tenant_name)])
            except Exception as exc:
                # ``tenants.create`` raises when the tenant already
                # exists. Cache the result either way - the caller only
                # needs to know it's safe to ``with_tenant`` on it.
                logger.debug(f"Tenant create skipped for {tenant_name}: {exc}")
            self._tenants_seen.add(tenant_name)

        return collection.with_tenant(tenant_name)

    # -------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------

    async def create_chunk(self, chunk: TemporalChunk) -> TemporalChunk:
        """Store a chunk with temporal metadata."""
        chunk_id = chunk.id or uuid4()
        chunk.id = chunk_id

        collection = await self._get_collection(chunk.namespace_id)
        await collection.data.insert(
            uuid=chunk_id,
            properties=_chunk_to_properties(chunk),
            vector=chunk.embedding,
        )
        return chunk

    async def create_chunks_batch(self, chunks: list[TemporalChunk]) -> list[TemporalChunk]:
        """Store multiple chunks.

        The v4 async client does not expose the ``batch.dynamic()``
        helper that the sync client carries. For correctness we issue
        per-chunk inserts; for parallelism we fan them out within a
        single namespace via ``asyncio.gather``. Native async batching
        can land in a follow-up once weaviate-client exposes the API.
        """
        import asyncio

        if not chunks:
            return []

        # Group by namespace - each tenant resolution is a single RTT
        by_namespace: dict[UUID, list[TemporalChunk]] = {}
        for chunk in chunks:
            chunk.id = chunk.id or uuid4()
            by_namespace.setdefault(chunk.namespace_id, []).append(chunk)

        for namespace_id, ns_chunks in by_namespace.items():
            collection = await self._get_collection(namespace_id)
            await asyncio.gather(
                *(
                    collection.data.insert(
                        uuid=chunk.id,
                        properties=_chunk_to_properties(chunk),
                        vector=chunk.embedding,
                    )
                    for chunk in ns_chunks
                )
            )

        return chunks

    async def get_chunk(self, chunk_id: UUID, namespace_id: UUID) -> TemporalChunk | None:
        """Get a chunk by ID."""
        collection = await self._get_collection(namespace_id)
        try:
            obj = await collection.query.fetch_object_by_id(chunk_id, include_vector=True)
            if not obj:
                return None
            return self._object_to_chunk(obj, namespace_id)
        except Exception as exc:
            logger.debug(f"Failed to get chunk {chunk_id}: {exc}")
            return None

    async def delete_chunk(self, chunk_id: UUID, namespace_id: UUID) -> bool:
        """Delete a chunk by ID."""
        collection = await self._get_collection(namespace_id)
        try:
            await collection.data.delete_by_id(chunk_id)
            return True
        except Exception as exc:
            logger.debug(f"Failed to delete chunk {chunk_id}: {exc}")
            return False

    async def delete_chunks_by_document(self, document_id: UUID, namespace_id: UUID) -> int:
        """Delete all chunks for a document."""
        import asyncio

        from weaviate.classes.query import Filter

        collection = await self._get_collection(namespace_id)
        result = await collection.query.fetch_objects(
            filters=Filter.by_property("document_id").equal(str(document_id)),
            limit=10000,
        )
        if not result.objects:
            return 0

        await asyncio.gather(*(collection.data.delete_by_id(obj.uuid) for obj in result.objects))
        return len(result.objects)

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
    ) -> list[TemporalSearchResult]:
        """Search for similar chunks with temporal filtering.

        Weaviate provides native hybrid search with alpha blending:
        - alpha=1: Pure vector search
        - alpha=0: Pure BM25 search
        - 0 < alpha < 1: Blend of both
        """
        from weaviate.classes.query import HybridFusion, MetadataQuery

        collection = await self._get_collection(namespace_id)
        weaviate_filter = self._build_weaviate_filter(temporal_filter) if temporal_filter else None

        if hybrid_alpha is not None and query_text:
            result = await collection.query.hybrid(
                query=query_text,
                vector=query_embedding,
                alpha=hybrid_alpha,
                filters=weaviate_filter,
                limit=limit,
                return_metadata=MetadataQuery(score=True, distance=True),
                include_vector=True,
                fusion_type=HybridFusion.RELATIVE_SCORE,
            )
        else:
            result = await collection.query.near_vector(
                near_vector=query_embedding,
                filters=weaviate_filter,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
                include_vector=True,
            )

        search_results = []
        for obj in result.objects:
            chunk = self._object_to_chunk(obj, namespace_id)
            distance = obj.metadata.distance if obj.metadata else 0
            similarity = 1 - distance if distance else 0
            if similarity >= min_similarity:
                search_results.append(
                    TemporalSearchResult(
                        chunk=chunk,
                        similarity=similarity,
                        bm25_score=obj.metadata.score if obj.metadata and hybrid_alpha else None,
                        combined_score=obj.metadata.score if obj.metadata else similarity,
                    )
                )

        return search_results

    def _build_weaviate_filter(self, f: TemporalFilter) -> Any:
        """Build Weaviate filter from TemporalFilter."""
        from weaviate.classes.query import Filter

        filters = []

        if f.occurred_after:
            filters.append(Filter.by_property("occurred_at").greater_or_equal(f.occurred_after.isoformat()))
        if f.occurred_before:
            filters.append(Filter.by_property("occurred_at").less_than(f.occurred_before.isoformat()))
        if f.created_after:
            filters.append(Filter.by_property("created_at").greater_or_equal(f.created_after.isoformat()))
        if f.created_before:
            filters.append(Filter.by_property("created_at").less_than(f.created_before.isoformat()))

        if f.source_system:
            filters.append(Filter.by_property("source_system").equal(f.source_system))
        if f.author:
            filters.append(Filter.by_property("author").equal(f.author))
        if f.channel:
            filters.append(Filter.by_property("channel").equal(f.channel))

        if f.tags:
            for tag in f.tags:
                filters.append(Filter.by_property("tags").contains_any([tag]))

        for key, value in f.additional.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == "eq":
                        filters.append(Filter.by_property(key).equal(val))
                    elif op == "gte":
                        filters.append(Filter.by_property(key).greater_or_equal(val))
                    elif op == "lte":
                        filters.append(Filter.by_property(key).less_or_equal(val))
                    elif op == "gt":
                        filters.append(Filter.by_property(key).greater_than(val))
                    elif op == "lt":
                        filters.append(Filter.by_property(key).less_than(val))
                    elif op == "in":
                        filters.append(Filter.by_property(key).contains_any(val))
            else:
                filters.append(Filter.by_property(key).equal(value))

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]

        combined = filters[0]
        for f_extra in filters[1:]:
            combined = combined & f_extra
        return combined

    def _object_to_chunk(self, obj: Any, namespace_id: UUID) -> TemporalChunk:
        """Convert a Weaviate object to a TemporalChunk."""
        props = obj.properties

        occurred_at = None
        if props.get("occurred_at"):
            try:
                occurred_at = datetime.fromisoformat(props["occurred_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        created_at = None
        if props.get("created_at"):
            try:
                created_at = datetime.fromisoformat(props["created_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        metadata: dict[str, Any] = {}
        if props.get("metadata_json"):
            try:
                metadata = json.loads(props["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        return TemporalChunk(
            id=UUID(str(obj.uuid)),
            namespace_id=namespace_id,
            document_id=UUID(props["document_id"]),
            content=props.get("content", ""),
            embedding=list(obj.vector["default"]) if obj.vector else None,
            occurred_at=occurred_at,
            created_at=created_at,
            source_system=props.get("source_system"),
            author=props.get("author"),
            channel=props.get("channel"),
            tags=props.get("tags") or [],
            confidence=props.get("confidence", 1.0),
            metadata=metadata,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""
        if not self._connected or not self._client:
            return {"status": "disconnected", "backend": "weaviate"}
        try:
            ready = await self._client.is_ready()
            if ready:
                return {"status": "healthy", "backend": "weaviate"}
            return {"status": "unhealthy", "backend": "weaviate", "error": "Not ready"}
        except Exception as exc:
            return {"status": "unhealthy", "backend": "weaviate", "error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_host_port(url: str) -> tuple[str, int]:
    """Split a URL into ``(host, port)``.

    Falls back to port 8080 when the URL has no explicit port. Used to
    feed ``weaviate.use_async_with_custom(http_host=..., http_port=...)``.
    """
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    port = parsed.port if parsed.port else 8080
    return host, port


def _chunk_to_properties(chunk: TemporalChunk) -> dict[str, Any]:
    """Translate a TemporalChunk into the property dict Weaviate expects."""
    return {
        "content": chunk.content,
        "document_id": str(chunk.document_id),
        "namespace_id": str(chunk.namespace_id),
        "occurred_at": chunk.occurred_at.isoformat() if chunk.occurred_at else None,
        "created_at": (chunk.created_at or datetime.now(UTC)).isoformat(),
        "source_system": chunk.source_system,
        "author": chunk.author,
        "channel": chunk.channel,
        "tags": chunk.tags or [],
        "confidence": chunk.confidence,
        "metadata_json": json.dumps(chunk.metadata or {}),
    }


__all__ = ["WeaviateBackendConfig", "WeaviateTemporalStore"]
