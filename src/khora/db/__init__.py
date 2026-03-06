"""Database module for Khora.

Uses SQLAlchemy for ORM and Alembic for migrations.
"""

from .models import (
    Base,
    ChunkModel,
    DocumentModel,
    EntityModel,
    EpisodeModel,
    MemoryEventModel,
    MemoryNamespaceModel,
    PermissionModel,
    RelationshipModel,
    SyncCheckpointModel,
)
from .session import close_db, get_db, get_engine, init_db, run_migrations

__all__ = [
    # Base
    "Base",
    # Models
    "MemoryNamespaceModel",
    "DocumentModel",
    "ChunkModel",
    "EntityModel",
    "RelationshipModel",
    "EpisodeModel",
    "MemoryEventModel",
    "PermissionModel",
    "SyncCheckpointModel",
    # Session utilities
    "close_db",
    "get_db",
    "get_engine",
    "init_db",
    "run_migrations",
]
