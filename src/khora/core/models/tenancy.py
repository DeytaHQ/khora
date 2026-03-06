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

    Supports versioning for data replacement workflows:
    - version: Incremental version number (starts at 1)
    - is_active: Whether this is the current active version
    - previous_version_id: Reference to the previous version (if any)
    """

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    slug: str = ""
    description: str = ""
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

    def __post_init__(self) -> None:
        if not self.slug and self.name:
            self.slug = self.name.lower().replace(" ", "-")
