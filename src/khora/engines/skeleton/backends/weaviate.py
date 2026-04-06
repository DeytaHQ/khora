"""Weaviate backend for the Skeleton engine.

This backend provides:
- Native hybrid search (BM25 + vector in single query)
- Rich filtering on any property (timestamps, keywords, custom fields)
- Multi-tenancy with tenant isolation
- Horizontal scaling for large datasets
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger

from khora.engines.skeleton.backends import (
    TemporalChunk,
    TemporalFilter,
    TemporalSearchResult,
    TemporalVectorStore,
)

if TYPE_CHECKING:
    from khora.config import KhoraConfig

# Collection name for temporal chunks
COLLECTION_NAME = "KhoraChunk"


class WeaviateTemporalStore(TemporalVectorStore):
    """Weaviate implementation of TemporalVectorStore.

    Provides native hybrid search combining BM25 and vector similarity
    with rich filtering capabilities.
    """

    def __init__(self, config: KhoraConfig, weaviate_url: str):
        """Initialize the backend.

        Args:
            config: Khora configuration
            weaviate_url: Weaviate server URL (e.g., "http://localhost:8080")
        """
        self._config = config
        self._weaviate_url = weaviate_url
        self._client = None
        self._connected = False
        self._embedding_dimension = config.llm.embedding_dimension or 1536

    async def connect(self) -> None:
        """Connect to Weaviate and ensure schema exists."""
        if self._connected:
            return

        try:
            import weaviate
            from weaviate.classes.config import Configure, DataType, Property
        except ImportError:
            raise ImportError(
                "weaviate-client is required for the Weaviate backend. "
                "Install it with: pip install weaviate-client>=4.10.0 "
                "or: pip install khora[weaviate]"
            )

        # Connect to Weaviate (sync client, async operations below)
        self._client = weaviate.connect_to_local(
            host=self._weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(self._weaviate_url.split(":")[-1]) if ":" in self._weaviate_url else 8080,
        )

        # Create collection if it doesn't exist
        if not self._client.collections.exists(COLLECTION_NAME):
            self._client.collections.create(
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
        logger.info(f"WeaviateTemporalStore connected to {self._weaviate_url}")

    async def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self._client:
            self._client.close()
            self._client = None
        self._connected = False
        logger.info("WeaviateTemporalStore disconnected")

    def _get_collection(self, namespace_id: UUID):
        """Get collection with tenant context."""
        if not self._client:
            raise RuntimeError("Not connected")

        collection = self._client.collections.get(COLLECTION_NAME)

        # Ensure tenant exists
        tenant_name = str(namespace_id)
        try:
            from weaviate.classes.tenants import Tenant

            collection.tenants.create([Tenant(name=tenant_name)])
        except Exception as e:
            logger.debug(f"Tenant creation skipped (likely already exists): {e}")

        return collection.with_tenant(tenant_name)

    async def create_chunk(self, chunk: TemporalChunk) -> TemporalChunk:
        """Store a chunk with temporal metadata."""
        chunk_id = chunk.id or uuid4()
        chunk.id = chunk_id

        collection = self._get_collection(chunk.namespace_id)

        import json

        properties = {
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

        collection.data.insert(
            uuid=chunk_id,
            properties=properties,
            vector=chunk.embedding,
        )

        return chunk

    async def create_chunks_batch(self, chunks: list[TemporalChunk]) -> list[TemporalChunk]:
        """Store multiple chunks in batch."""
        if not chunks:
            return []

        import json

        # Group by namespace for batch insert
        by_namespace: dict[UUID, list[TemporalChunk]] = {}
        for chunk in chunks:
            chunk.id = chunk.id or uuid4()
            by_namespace.setdefault(chunk.namespace_id, []).append(chunk)

        for namespace_id, ns_chunks in by_namespace.items():
            collection = self._get_collection(namespace_id)

            with collection.batch.dynamic() as batch:
                for chunk in ns_chunks:
                    properties = {
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
                    batch.add_object(
                        uuid=chunk.id,
                        properties=properties,
                        vector=chunk.embedding,
                    )

        return chunks

    async def get_chunk(self, chunk_id: UUID, namespace_id: UUID) -> TemporalChunk | None:
        """Get a chunk by ID."""
        collection = self._get_collection(namespace_id)

        try:
            obj = collection.query.fetch_object_by_id(chunk_id, include_vector=True)
            if not obj:
                return None
            return self._object_to_chunk(obj, namespace_id)
        except Exception as e:
            logger.debug(f"Failed to get chunk {chunk_id}: {e}")
            return None

    async def delete_chunk(self, chunk_id: UUID, namespace_id: UUID) -> bool:
        """Delete a chunk by ID."""
        collection = self._get_collection(namespace_id)

        try:
            collection.data.delete_by_id(chunk_id)
            return True
        except Exception as e:
            logger.debug(f"Failed to delete chunk {chunk_id}: {e}")
            return False

    async def delete_chunks_by_document(self, document_id: UUID, namespace_id: UUID) -> int:
        """Delete all chunks for a document."""
        from weaviate.classes.query import Filter

        collection = self._get_collection(namespace_id)

        # Query for matching chunks
        result = collection.query.fetch_objects(
            filters=Filter.by_property("document_id").equal(str(document_id)),
            limit=10000,
        )

        count = 0
        for obj in result.objects:
            collection.data.delete_by_id(obj.uuid)
            count += 1

        return count

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

        collection = self._get_collection(namespace_id)

        # Build filters
        weaviate_filter = self._build_weaviate_filter(temporal_filter) if temporal_filter else None

        # Perform search
        if hybrid_alpha is not None and query_text:
            # Hybrid search
            result = collection.query.hybrid(
                query=query_text,
                vector=query_embedding,
                alpha=hybrid_alpha,
                filters=weaviate_filter,
                limit=limit,
                return_metadata=MetadataQuery(score=True, distance=True),
                include_vector=True,
                fusion_type=HybridFusion.RELATIVE_SCORE,  # Better fusion
            )
        else:
            # Vector-only search
            result = collection.query.near_vector(
                near_vector=query_embedding,
                filters=weaviate_filter,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
                include_vector=True,
            )

        # Convert results
        search_results = []
        for obj in result.objects:
            chunk = self._object_to_chunk(obj, namespace_id)

            # Calculate similarity from distance (cosine: similarity = 1 - distance)
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

    def _build_weaviate_filter(self, f: TemporalFilter):
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
            # Weaviate TEXT_ARRAY contains filter
            for tag in f.tags:
                filters.append(Filter.by_property("tags").contains_any([tag]))

        # Handle additional filters
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

        # Combine filters with AND
        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]

        combined = filters[0]
        for f in filters[1:]:
            combined = combined & f
        return combined

    def _object_to_chunk(self, obj, namespace_id: UUID) -> TemporalChunk:
        """Convert a Weaviate object to a TemporalChunk."""
        import json
        from datetime import datetime

        props = obj.properties

        # Parse dates
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

        # Parse metadata
        metadata = {}
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
            if self._client.is_ready():
                return {"status": "healthy", "backend": "weaviate"}
            else:
                return {"status": "unhealthy", "backend": "weaviate", "error": "Not ready"}
        except Exception as e:
            return {"status": "unhealthy", "backend": "weaviate", "error": str(e)}


__all__ = ["WeaviateTemporalStore"]
