"""Multi-tenancy models for Khora Memory Lake.

Namespace is the sole data isolation boundary. All memories, entities,
and relationships are scoped to a namespace_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class TenancyMode(str, Enum):
    """Tenancy isolation mode."""

    SHARED = "shared"  # Shared database with namespace_id filtering + row-level security
    ISOLATED = "isolated"  # Separate database instances per tenant


@dataclass
class MemoryNamespace:
    """Memory namespace for isolating memories.

    A namespace is the primary unit of memory isolation. All memories,
    entities, and relationships are scoped to a namespace.
    Every query is filtered by namespace_id for multi-tenancy.

    Two ID fields serve different purposes:
    - id: Row-level identifier for this specific version (changes per version)
    - namespace_id: Stable identifier shared across all versions of a namespace

    Use ``namespace_id`` for external references and API calls.
    Use ``id`` for internal versioning logic and child-table FK lookups.

    Supports versioning for data replacement workflows:
    - version: Incremental version number (starts at 1)
    - is_active: Whether this is the current active version
    - previous_version_id: Reference to the previous version (if any)
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    tenancy_mode: TenancyMode = TenancyMode.SHARED

    # Versioning fields
    version: int = 1
    is_active: bool = True
    previous_version_id: UUID | None = None

    # Configuration overrides for this namespace
    config_overrides: dict[str, Any] = field(default_factory=dict)

    # Sync checkpoints for incremental updates
    sync_checkpoints: dict[str, str] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
