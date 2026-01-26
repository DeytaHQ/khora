"""MemoryLake - Primary API for Khora Memory Lake.

This is the main entry point for using Khora as a library.
Provides a simple, unified interface for memory storage and retrieval.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.config import KhoraConfig, LiteLLMConfig, load_config
from khora.core.models import Document, DocumentMetadata, Entity, MemoryNamespace, Organization, Workspace
from khora.extraction.embedders import LiteLLMEmbedder
from khora.query import HybridQueryEngine, QueryConfig, SearchMode
from khora.storage import StorageConfig, StorageCoordinator, create_storage_coordinator

if TYPE_CHECKING:
    pass


@dataclass
class RememberResult:
    """Result of a remember operation."""

    document_id: UUID
    namespace_id: UUID
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecallResult:
    """Result of a recall operation."""

    query: str
    namespace_id: UUID
    chunks: list[tuple[Any, float]]
    entities: list[tuple[Any, float]]
    context_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryLake:
    """Primary interface for Khora Memory Lake.

    Provides a simple API for storing and retrieving memories:
    - remember(): Store content in the memory lake
    - recall(): Retrieve relevant memories for a query
    - forget(): Remove memories

    Can be used as a context manager for automatic connection handling.

    Usage:
        async with MemoryLake() as lake:
            await lake.remember("Important fact...", namespace="my-ns")
            results = await lake.recall("What do I know about...", namespace="my-ns")
    """

    def __init__(
        self,
        config: KhoraConfig | None = None,
        storage_config: StorageConfig | None = None,
    ) -> None:
        """Initialize the Memory Lake.

        Args:
            config: Khora configuration (loads from env if None)
            storage_config: Storage configuration (derived from config if None)
        """
        self._config = config or load_config()

        # Set up storage config
        if storage_config:
            self._storage_config = storage_config
        else:
            self._storage_config = StorageConfig(
                postgresql_url=self._config.get_postgresql_url(),
                neo4j_url=self._config.get_neo4j_url(),
                neo4j_user=self._config.get_neo4j_user(),
                neo4j_password=self._config.get_neo4j_password(),
                neo4j_database=self._config.get_neo4j_database(),
                pgvector_embedding_dimension=self._config.storage.embedding_dimension,
            )

        self._storage: StorageCoordinator | None = None
        self._embedder: LiteLLMEmbedder | None = None
        self._query_engine: HybridQueryEngine | None = None
        self._connected = False

        # Default namespace for simple usage
        self._default_namespace_id: UUID | None = None

    async def connect(self) -> None:
        """Connect to all storage backends."""
        if self._connected:
            return

        logger.info("Connecting Memory Lake...")

        # Create and connect storage
        self._storage = create_storage_coordinator(self._storage_config)
        await self._storage.connect()

        # Create embedder
        llm_config = LiteLLMConfig(
            model=self._config.llm.model,
            embedding_model=self._config.llm.embedding_model,
            embedding_dimension=self._config.llm.embedding_dimension,
            timeout=self._config.llm.timeout,
            max_retries=self._config.llm.max_retries,
        )
        self._embedder = LiteLLMEmbedder.from_config(llm_config)

        # Create query engine
        self._query_engine = HybridQueryEngine(
            storage=self._storage,
            embedder=self._embedder,
        )

        self._connected = True
        logger.info("Memory Lake connected")

    async def disconnect(self) -> None:
        """Disconnect from all storage backends."""
        if not self._connected:
            return

        logger.info("Disconnecting Memory Lake...")

        if self._storage:
            await self._storage.disconnect()
            self._storage = None

        self._embedder = None
        self._query_engine = None
        self._connected = False

        logger.info("Memory Lake disconnected")

    async def __aenter__(self) -> MemoryLake:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    @property
    def storage(self) -> StorageCoordinator:
        """Get the storage coordinator."""
        if self._storage is None:
            raise RuntimeError("Memory Lake not connected. Call connect() first.")
        return self._storage

    @property
    def query_engine(self) -> HybridQueryEngine:
        """Get the query engine."""
        if self._query_engine is None:
            raise RuntimeError("Memory Lake not connected. Call connect() first.")
        return self._query_engine

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def create_namespace(
        self,
        name: str,
        workspace_id: UUID,
        *,
        description: str = "",
        config_overrides: dict[str, Any] | None = None,
    ) -> MemoryNamespace:
        """Create a new memory namespace.

        Args:
            name: Namespace name
            workspace_id: Parent workspace ID
            description: Optional description
            config_overrides: Optional configuration overrides

        Returns:
            Created MemoryNamespace
        """
        namespace = MemoryNamespace(
            workspace_id=workspace_id,
            name=name,
            description=description,
            config_overrides=config_overrides or {},
        )
        return await self.storage.create_namespace(namespace)

    async def get_namespace(self, namespace_id: UUID) -> MemoryNamespace | None:
        """Get a namespace by ID."""
        return await self.storage.get_namespace(namespace_id)

    async def get_or_create_default_namespace(self) -> UUID:
        """Get or create a default namespace for simple usage."""
        if self._default_namespace_id:
            return self._default_namespace_id

        # Try to find existing default namespace
        # For simplicity, we'll create a default org/workspace/namespace
        default_org = await self.storage.get_organization_by_slug("default")
        if not default_org:
            default_org = await self.storage.create_organization(Organization(name="Default", slug="default"))

        workspaces = await self.storage.list_workspaces(default_org.id)
        if workspaces:
            default_workspace = workspaces[0]
        else:
            default_workspace = await self.storage.create_workspace(
                Workspace(
                    organization_id=default_org.id,
                    name="Default",
                    slug="default",
                )
            )

        namespaces = await self.storage.list_namespaces(default_workspace.id)
        if namespaces:
            default_namespace = namespaces[0]
        else:
            default_namespace = await self.storage.create_namespace(
                MemoryNamespace(
                    workspace_id=default_workspace.id,
                    name="Default",
                    slug="default",
                )
            )

        self._default_namespace_id = default_namespace.id
        return self._default_namespace_id

    # =========================================================================
    # Core API: remember, recall, forget
    # =========================================================================

    async def remember(
        self,
        content: str,
        *,
        namespace: str | UUID | None = None,
        title: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
        skill_name: str = "general_entities",
    ) -> RememberResult:
        """Store content in the memory lake.

        This is the primary method for adding memories. It:
        1. Creates a document
        2. Chunks the content
        3. Generates embeddings
        4. Extracts entities and relationships

        Args:
            content: Content to remember
            namespace: Namespace name, ID, or None for default
            title: Optional title for the content
            source: Optional source identifier
            metadata: Optional metadata
            skill_name: Extraction skill to use

        Returns:
            RememberResult with details
        """
        # Resolve namespace
        namespace_id = await self._resolve_namespace(namespace)

        # Compute checksum
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Check for duplicate
        existing = await self.storage.get_document_by_checksum(namespace_id, checksum)
        if existing and existing.is_processed:
            logger.debug(f"Document already exists (checksum={checksum[:8]}...)")
            return RememberResult(
                document_id=existing.id,
                namespace_id=namespace_id,
                chunks_created=existing.chunk_count,
                entities_extracted=existing.entity_count,
                relationships_created=0,
                metadata={"duplicate": True},
            )

        # Create document
        doc_metadata = DocumentMetadata(
            title=title,
            source=source,
            source_type="api",
            checksum=checksum,
            size_bytes=len(content.encode("utf-8")),
            custom=metadata or {},
        )
        document = Document(
            namespace_id=namespace_id,
            content=content,
            metadata=doc_metadata,
        )
        document = await self.storage.create_document(document)

        # Process through pipeline
        from khora.pipelines.flows.ingest import process_document

        result = await process_document(
            document,
            self.storage,
            skill_name=skill_name,
            embedding_model=self._config.llm.embedding_model,
            extraction_model=self._config.llm.model,
        )

        return RememberResult(
            document_id=document.id,
            namespace_id=namespace_id,
            chunks_created=result["chunks"],
            entities_extracted=result["entities"],
            relationships_created=result["relationships"],
        )

    async def recall(
        self,
        query: str,
        *,
        namespace: str | UUID | None = None,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        min_similarity: float = 0.5,
    ) -> RecallResult:
        """Recall memories relevant to a query.

        This is the primary method for retrieving memories. It:
        1. Embeds the query
        2. Searches across vector, graph, and keyword indexes
        3. Fuses results using Reciprocal Rank Fusion
        4. Returns ranked results

        Args:
            query: Query text
            namespace: Namespace name, ID, or None for default
            limit: Maximum results to return
            mode: Search mode (VECTOR, GRAPH, HYBRID, ALL)
            min_similarity: Minimum similarity threshold

        Returns:
            RecallResult with matched memories
        """
        namespace_id = await self._resolve_namespace(namespace)

        config = QueryConfig(
            mode=mode,
            max_chunks=limit,
            max_entities=limit,
            min_chunk_similarity=min_similarity,
            min_entity_similarity=min_similarity,
        )

        result = await self.query_engine.query(query, namespace_id, config=config)

        return RecallResult(
            query=query,
            namespace_id=namespace_id,
            chunks=result.chunks,
            entities=result.entities,
            context_text=result.get_context_text(max_chunks=limit),
            metadata=result.metadata,
        )

    async def forget(
        self,
        document_id: UUID,
        *,
        namespace: str | UUID | None = None,
    ) -> bool:
        """Remove a memory from the lake.

        Args:
            document_id: ID of the document to remove
            namespace: Namespace for verification (optional)

        Returns:
            True if deleted, False if not found
        """
        # Verify namespace if provided
        if namespace:
            namespace_id = await self._resolve_namespace(namespace)
            document = await self.storage.get_document(document_id)
            if document and document.namespace_id != namespace_id:
                logger.warning(f"Document {document_id} not in namespace {namespace_id}")
                return False

        return await self.storage.delete_document(document_id)

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID."""
        return await self.storage.get_entity(entity_id)

    async def list_entities(
        self,
        *,
        namespace: str | UUID | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[Entity]:
        """List entities in a namespace."""
        namespace_id = await self._resolve_namespace(namespace)
        return await self.storage.list_entities(namespace_id, entity_type=entity_type, limit=limit)

    async def find_related_entities(
        self,
        entity_id: UUID,
        *,
        namespace: str | UUID | None = None,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Find entities related to a given entity."""
        namespace_id = await self._resolve_namespace(namespace)
        return await self.query_engine.find_related_entities(
            entity_id,
            namespace_id,
            max_depth=max_depth,
            limit=limit,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    async def _resolve_namespace(self, namespace: str | UUID | None) -> UUID:
        """Resolve a namespace reference to a UUID."""
        if namespace is None:
            return await self.get_or_create_default_namespace()

        if isinstance(namespace, UUID):
            return namespace

        # Try to parse as UUID
        try:
            return UUID(namespace)
        except ValueError:
            pass

        # Look up by slug in default workspace
        default_ns_id = await self.get_or_create_default_namespace()
        default_ns = await self.storage.get_namespace(default_ns_id)
        if default_ns:
            ns = await self.storage.get_namespace_by_slug(default_ns.workspace_id, namespace)
            if ns:
                return ns.id

        raise ValueError(f"Namespace not found: {namespace}")

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components."""
        if not self._connected:
            return {"status": "disconnected"}

        storage_health = await self.storage.health_check()

        return {
            "status": "healthy" if storage_health.is_healthy else "degraded",
            "storage": storage_health.summary,
        }


# Convenience function for one-off usage
@asynccontextmanager
async def memory_lake(
    config: KhoraConfig | None = None,
) -> AsyncGenerator[MemoryLake]:
    """Context manager for one-off Memory Lake usage.

    Usage:
        async with memory_lake() as lake:
            await lake.remember("Hello, world!")
            result = await lake.recall("greeting")
    """
    lake = MemoryLake(config=config)
    try:
        await lake.connect()
        yield lake
    finally:
        await lake.disconnect()
