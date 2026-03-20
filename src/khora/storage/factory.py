"""Storage factory for creating storage backends and coordinator.

Creates and configures storage backends based on configuration.
Supports registry-based dispatch for multiple graph and vector backend types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .backends.pgvector import PgVectorBackend
from .backends.postgresql import PostgreSQLBackend
from .coordinator import StorageCoordinator

if TYPE_CHECKING:
    from .backends.base import EventStoreProtocol, GraphBackendProtocol, VectorBackendProtocol


# ---------------------------------------------------------------------------
# Backend registries: backend_name → (module_path, class_name)
# ---------------------------------------------------------------------------

_GRAPH_REGISTRY: dict[str, tuple[str, str]] = {
    "neo4j": ("khora.storage.backends.neo4j", "Neo4jBackend"),
    "kuzu": ("khora.storage.backends.kuzu", "KuzuBackend"),
    "memgraph": ("khora.storage.backends.memgraph", "MemgraphBackend"),
    "arcadedb": ("khora.storage.backends.arcadedb", "ArcadeDBBackend"),
    "surrealdb": ("khora.storage.backends.surrealdb.graph", "SurrealDBGraphAdapter"),
}

_VECTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "pgvector": ("khora.storage.backends.pgvector", "PgVectorBackend"),
    "arcadedb": ("khora.storage.backends.arcadedb", "ArcadeDBBackend"),
    "surrealdb": ("khora.storage.backends.surrealdb.vector", "SurrealDBVectorAdapter"),
}


def _import_backend_class(module_path: str, class_name: str) -> type | None:
    """Lazily import a backend class. Returns None if the dependency is missing."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        logger.warning(f"Cannot import {class_name} from {module_path}: {e}")
        return None
    except AttributeError:
        logger.warning(f"Class {class_name} not found in {module_path}")
        return None


@dataclass
class StorageConfig:
    """Configuration for storage backends.

    Supports both legacy flat fields (neo4j_url, pgvector_url, etc.) and
    new-style discriminated union configs (graph_config, vector_config).
    """

    # PostgreSQL configuration
    postgresql_url: str | None = None
    postgresql_echo: bool = False
    postgresql_pool_size: int = 10
    postgresql_max_overflow: int = 20
    postgresql_pool_pre_ping: bool = False

    # pgvector configuration (can share PostgreSQL URL) — legacy
    pgvector_url: str | None = None
    pgvector_embedding_dimension: int = 1536
    pgvector_use_halfvec: bool = True

    # Neo4j configuration — legacy
    neo4j_url: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # New-style backend configs (Pydantic models from config/schema.py)
    graph_config: Any = None  # GraphConfig union type
    vector_config: Any = None  # VectorConfig union type

    # Unified backend selector
    backend: str = "postgres"  # "postgres" (traditional) or "surrealdb" (unified)
    surrealdb_config: Any = None  # SurrealDBConfig from config/schema.py

    # Event store configuration (uses PostgreSQL by default)
    event_store_url: str | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> StorageConfig:
        """Create configuration from a dictionary."""
        storage_config = config.get("storage", {})

        # Extract relational config
        relational = storage_config.get("relational", {})
        postgresql_url = relational.get("url") or config.get("database_url")

        # Extract vector config
        vector = storage_config.get("vector", {})
        pgvector_url = vector.get("url") or postgresql_url  # Default to same as PostgreSQL
        embedding_dimension = vector.get("embedding_dimension", 1536)

        # Extract graph config
        graph = storage_config.get("graph", {})
        neo4j_url = graph.get("url")
        neo4j_user = graph.get("user", "neo4j")
        neo4j_password = graph.get("password", "")
        neo4j_database = graph.get("database", "neo4j")

        return cls(
            postgresql_url=postgresql_url,
            postgresql_echo=relational.get("echo", False),
            postgresql_pool_size=relational.get("pool_size", 5),
            postgresql_max_overflow=relational.get("max_overflow", 10),
            pgvector_url=pgvector_url,
            pgvector_embedding_dimension=embedding_dimension,
            neo4j_url=neo4j_url,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            event_store_url=storage_config.get("event_store", {}).get("url") or postgresql_url,
        )


# Cache for ArcadeDB dual-role instance sharing
_arcadedb_instances: dict[str, Any] = {}

# Cache for SurrealDB dual-role instance sharing
_surrealdb_instances: dict[str, Any] = {}


def _normalize_url(url: str) -> str:
    """Normalize a database URL for cache-key comparison.

    Strips trailing slashes and lowercases the scheme+host so that
    ``postgresql://HOST/db`` and ``postgresql://host/db/`` share a pool.
    """
    parsed = urlparse(url)
    # Rebuild with lowercased scheme+host, stripped trailing slash on path
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip("/") or "/",
    )
    return normalized.geturl()


@dataclass
class StorageFactory:
    """Factory for creating storage backends."""

    config: StorageConfig = field(default_factory=StorageConfig)
    _engine_cache: dict[str, AsyncEngine] = field(default_factory=dict, repr=False)

    def get_or_create_engine(
        self,
        url: str,
        *,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_pre_ping: bool = False,
    ) -> AsyncEngine:
        """Get a cached engine or create a new one for the given URL.

        Engines are cached by normalized URL so that backends sharing the
        same database (e.g. PostgreSQLBackend and PgVectorBackend) reuse
        a single connection pool.
        """
        # Convert to async URL if needed
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)

        key = _normalize_url(url)
        if key not in self._engine_cache:
            self._engine_cache[key] = create_async_engine(
                url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=pool_pre_ping,
            )
            logger.debug(f"Created shared engine for {key}")
        return self._engine_cache[key]

    async def dispose_engines(self) -> None:
        """Dispose all cached engines."""
        for key, engine in self._engine_cache.items():
            await engine.dispose()
            logger.debug(f"Disposed shared engine for {key}")
        self._engine_cache.clear()

    def create_relational_backend(self) -> PostgreSQLBackend | None:
        """Create the PostgreSQL relational backend."""
        if not self.config.postgresql_url:
            logger.warning("PostgreSQL URL not configured, relational backend disabled")
            return None

        engine = self.get_or_create_engine(
            self.config.postgresql_url,
            echo=self.config.postgresql_echo,
            pool_size=self.config.postgresql_pool_size,
            max_overflow=self.config.postgresql_max_overflow,
            pool_pre_ping=self.config.postgresql_pool_pre_ping,
        )
        return PostgreSQLBackend(
            self.config.postgresql_url,
            echo=self.config.postgresql_echo,
            pool_size=self.config.postgresql_pool_size,
            max_overflow=self.config.postgresql_max_overflow,
            pool_pre_ping=self.config.postgresql_pool_pre_ping,
            engine=engine,
        )

    def create_vector_backend(self) -> VectorBackendProtocol | None:
        """Create the vector backend based on config."""
        vector_config = self.config.vector_config

        # New-style config dispatch
        if vector_config is not None:
            backend_name = getattr(vector_config, "backend", None)
            if backend_name == "pgvector":
                # PgVectorBackend has a specialized constructor
                url = getattr(vector_config, "url", None)
                if not url:
                    logger.warning("pgvector URL not configured, vector backend disabled")
                    return None
                dim = getattr(vector_config, "embedding_dimension", 1536)
                engine = self.get_or_create_engine(
                    url,
                    echo=self.config.postgresql_echo,
                    pool_size=self.config.postgresql_pool_size,
                    max_overflow=self.config.postgresql_max_overflow,
                    pool_pre_ping=self.config.postgresql_pool_pre_ping,
                )
                return PgVectorBackend(
                    url,
                    embedding_dimension=dim,
                    echo=self.config.postgresql_echo,
                    pool_size=self.config.postgresql_pool_size,
                    max_overflow=self.config.postgresql_max_overflow,
                    pool_pre_ping=self.config.postgresql_pool_pre_ping,
                    use_halfvec=self.config.pgvector_use_halfvec,
                    engine=engine,
                )
            elif backend_name and backend_name in _VECTOR_REGISTRY:
                return self._create_from_registry(_VECTOR_REGISTRY, backend_name, vector_config, "vector")

        # Legacy pgvector path
        if not self.config.pgvector_url:
            logger.warning("pgvector URL not configured, vector backend disabled")
            return None

        engine = self.get_or_create_engine(
            self.config.pgvector_url,
            echo=self.config.postgresql_echo,
            pool_size=self.config.postgresql_pool_size,
            max_overflow=self.config.postgresql_max_overflow,
            pool_pre_ping=self.config.postgresql_pool_pre_ping,
        )
        return PgVectorBackend(
            self.config.pgvector_url,
            embedding_dimension=self.config.pgvector_embedding_dimension,
            echo=self.config.postgresql_echo,
            pool_size=self.config.postgresql_pool_size,
            max_overflow=self.config.postgresql_max_overflow,
            pool_pre_ping=self.config.postgresql_pool_pre_ping,
            use_halfvec=self.config.pgvector_use_halfvec,
            engine=engine,
        )

    def create_graph_backend(self) -> GraphBackendProtocol | None:
        """Create the graph backend based on config."""
        graph_config = self.config.graph_config

        # New-style config dispatch
        if graph_config is not None:
            backend_name = getattr(graph_config, "backend", None)
            if backend_name and backend_name in _GRAPH_REGISTRY:
                return self._create_from_registry(_GRAPH_REGISTRY, backend_name, graph_config, "graph")

        # Legacy Neo4j path
        if not self.config.neo4j_url:
            logger.warning("Neo4j URL not configured, graph backend disabled")
            return None

        try:
            from .backends.neo4j import Neo4jBackend

            return Neo4jBackend(
                self.config.neo4j_url,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database,
                max_connection_pool_size=100,
            )
        except ImportError:
            logger.warning("neo4j package not installed, graph backend disabled")
            return None

    def _create_from_registry(
        self,
        registry: dict[str, tuple[str, str]],
        backend_name: str,
        config: Any,
        role: str,
    ) -> Any | None:
        """Create a backend instance from registry via lazy import + from_config().

        For ArcadeDB, reuses the same instance when it serves both graph and vector roles.
        """
        if backend_name == "arcadedb":
            # ArcadeDB dual-role: reuse instance for same URL
            url = getattr(config, "url", None) or ""
            cache_key = f"arcadedb:{url}"
            if cache_key in _arcadedb_instances:
                logger.info(f"Reusing ArcadeDB instance for {role} role (url={url})")
                return _arcadedb_instances[cache_key]

        if backend_name == "surrealdb":
            # SurrealDB dual-role: reuse instance for same endpoint
            url = getattr(config, "url", None) or ""
            path = getattr(config, "path", None) or ""
            mode = getattr(config, "mode", "memory")
            cache_key = f"surrealdb:{mode}:{url}:{path}"
            if cache_key in _surrealdb_instances:
                logger.info(f"Reusing SurrealDB instance for {role} role (mode={mode})")
                return _surrealdb_instances[cache_key]

        module_path, class_name = registry[backend_name]
        cls = _import_backend_class(module_path, class_name)
        if cls is None:
            logger.warning(f"{backend_name} backend not available (missing dependency), {role} backend disabled")
            return None

        if not hasattr(cls, "from_config"):
            raise ValueError(f"Backend class {class_name} does not implement from_config()")

        instance = cls.from_config(config)

        if backend_name == "arcadedb":
            url = getattr(config, "url", None) or ""
            _arcadedb_instances[f"arcadedb:{url}"] = instance

        if backend_name == "surrealdb":
            url = getattr(config, "url", None) or ""
            path = getattr(config, "path", None) or ""
            mode = getattr(config, "mode", "memory")
            _surrealdb_instances[f"surrealdb:{mode}:{url}:{path}"] = instance

        return instance

    def create_event_store(self) -> EventStoreProtocol | None:
        """Create the event store backend."""
        if not self.config.event_store_url:
            logger.warning("Event store URL not configured, event store disabled")
            return None

        # Import here to avoid circular dependency
        try:
            from .event_store import PostgreSQLEventStore

            engine = self.get_or_create_engine(
                self.config.event_store_url,
                echo=self.config.postgresql_echo,
                pool_size=self.config.postgresql_pool_size,
                max_overflow=self.config.postgresql_max_overflow,
                pool_pre_ping=self.config.postgresql_pool_pre_ping,
            )
            return PostgreSQLEventStore(
                self.config.event_store_url,
                echo=self.config.postgresql_echo,
                pool_size=self.config.postgresql_pool_size,
                max_overflow=self.config.postgresql_max_overflow,
                engine=engine,
            )
        except ImportError:
            logger.warning("Event store not available")
            return None

    def create_coordinator(self) -> StorageCoordinator:
        """Create a storage coordinator with all configured backends.

        When backend='surrealdb', the same SurrealDB instance is used for
        graph and vector roles (relational and event_store are set to None
        since SurrealDB handles those internally).
        """
        if self.config.backend == "surrealdb":
            surreal_config = self.config.surrealdb_config
            if surreal_config is None:
                raise ValueError("SurrealDB backend selected but surrealdb_config is not set")

            # Create a shared SurrealDB connection for all four adapters
            try:
                from .backends.surrealdb.connection import SurrealDBConnection
                from .backends.surrealdb.event_store import SurrealDBEventStoreAdapter
                from .backends.surrealdb.graph import SurrealDBGraphAdapter
                from .backends.surrealdb.relational import SurrealDBRelationalAdapter
                from .backends.surrealdb.vector import SurrealDBVectorAdapter

                conn = SurrealDBConnection(
                    mode=getattr(surreal_config, "mode", "memory"),
                    path=getattr(surreal_config, "path", None),
                    url=getattr(surreal_config, "url", None),
                    namespace=getattr(surreal_config, "namespace", "khora"),
                    database=getattr(surreal_config, "database", "default"),
                    user=getattr(surreal_config, "user", "root"),
                    password=getattr(surreal_config, "password", "root"),
                    sync_data=getattr(surreal_config, "sync_data", True),
                )
                return StorageCoordinator(
                    relational=SurrealDBRelationalAdapter(conn),
                    vector=SurrealDBVectorAdapter(conn),
                    graph=SurrealDBGraphAdapter(conn),
                    event_store=SurrealDBEventStoreAdapter(conn),
                )
            except ImportError:
                raise ValueError(
                    "SurrealDB backend selected but surrealdb package is not installed. "
                    "Install with: pip install khora[surrealdb]"
                )

        return StorageCoordinator(
            relational=self.create_relational_backend(),
            vector=self.create_vector_backend(),
            graph=self.create_graph_backend(),
            event_store=self.create_event_store(),
        )


def create_storage_coordinator(config: dict[str, Any] | StorageConfig | None = None) -> StorageCoordinator:
    """Convenience function to create a storage coordinator.

    Args:
        config: Configuration dictionary, StorageConfig instance, or None for defaults

    Returns:
        Configured StorageCoordinator
    """
    if config is None:
        storage_config = StorageConfig()
    elif isinstance(config, StorageConfig):
        storage_config = config
    else:
        storage_config = StorageConfig.from_dict(config)

    factory = StorageFactory(config=storage_config)
    return factory.create_coordinator()
