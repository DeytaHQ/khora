"""Storage factory for creating storage backends and coordinator.

Creates and configures storage backends based on configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from .backends.pgvector import PgVectorBackend
from .backends.postgresql import PostgreSQLBackend
from .coordinator import StorageCoordinator

if TYPE_CHECKING:
    from .backends.base import EventStoreProtocol, GraphBackendProtocol


@dataclass
class StorageConfig:
    """Configuration for storage backends."""

    # PostgreSQL configuration
    postgresql_url: str | None = None
    postgresql_echo: bool = False
    postgresql_pool_size: int = 5
    postgresql_max_overflow: int = 10

    # pgvector configuration (can share PostgreSQL URL)
    pgvector_url: str | None = None
    pgvector_embedding_dimension: int = 1536

    # Neo4j configuration
    neo4j_url: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

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


@dataclass
class StorageFactory:
    """Factory for creating storage backends."""

    config: StorageConfig = field(default_factory=StorageConfig)

    def create_relational_backend(self) -> PostgreSQLBackend | None:
        """Create the PostgreSQL relational backend."""
        if not self.config.postgresql_url:
            logger.warning("PostgreSQL URL not configured, relational backend disabled")
            return None

        return PostgreSQLBackend(
            self.config.postgresql_url,
            echo=self.config.postgresql_echo,
            pool_size=self.config.postgresql_pool_size,
            max_overflow=self.config.postgresql_max_overflow,
        )

    def create_vector_backend(self) -> PgVectorBackend | None:
        """Create the pgvector backend."""
        if not self.config.pgvector_url:
            logger.warning("pgvector URL not configured, vector backend disabled")
            return None

        return PgVectorBackend(
            self.config.pgvector_url,
            embedding_dimension=self.config.pgvector_embedding_dimension,
            echo=self.config.postgresql_echo,
            pool_size=self.config.postgresql_pool_size,
            max_overflow=self.config.postgresql_max_overflow,
        )

    def create_graph_backend(self) -> GraphBackendProtocol | None:
        """Create the Neo4j graph backend."""
        if not self.config.neo4j_url:
            logger.warning("Neo4j URL not configured, graph backend disabled")
            return None

        # Import here to avoid dependency if not used
        try:
            from .backends.neo4j import Neo4jBackend

            return Neo4jBackend(
                self.config.neo4j_url,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database,
            )
        except ImportError:
            logger.warning("neo4j package not installed, graph backend disabled")
            return None

    def create_event_store(self) -> EventStoreProtocol | None:
        """Create the event store backend."""
        if not self.config.event_store_url:
            logger.warning("Event store URL not configured, event store disabled")
            return None

        # Import here to avoid circular dependency
        try:
            from .event_store import PostgreSQLEventStore

            return PostgreSQLEventStore(
                self.config.event_store_url,
                echo=self.config.postgresql_echo,
                pool_size=self.config.postgresql_pool_size,
                max_overflow=self.config.postgresql_max_overflow,
            )
        except ImportError:
            logger.warning("Event store not available")
            return None

    def create_coordinator(self) -> StorageCoordinator:
        """Create a storage coordinator with all configured backends."""
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
