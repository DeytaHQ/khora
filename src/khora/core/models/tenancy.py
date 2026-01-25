"""Multi-tenancy models for Khora Memory Lake.

Supports two modes:
- Shared: namespace_id filtering with row-level security
- Isolated: Separate database instances per tenant
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
class Organization:
    """Top-level tenant organization.

    An organization can have multiple workspaces and represents the billing
    and administrative boundary for tenants.
    """

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    slug: str = ""  # URL-friendly identifier
    tenancy_mode: TenancyMode = TenancyMode.SHARED
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.slug and self.name:
            self.slug = self.name.lower().replace(" ", "-")


@dataclass
class Workspace:
    """Workspace within an organization.

    Workspaces provide logical separation of projects or teams within
    an organization. Each workspace can have multiple memory namespaces.
    """

    id: UUID = field(default_factory=uuid4)
    organization_id: UUID = field(default_factory=uuid4)
    name: str = ""
    slug: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.slug and self.name:
            self.slug = self.name.lower().replace(" ", "-")


@dataclass
class MemoryNamespace:
    """Memory namespace for isolating memories.

    A namespace is the primary unit of memory isolation. All memories,
    entities, and relationships are scoped to a namespace.
    Every query is filtered by namespace_id for multi-tenancy.
    """

    id: UUID = field(default_factory=uuid4)
    workspace_id: UUID = field(default_factory=uuid4)
    name: str = ""
    slug: str = ""
    description: str = ""

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

    @property
    def full_path(self) -> str:
        """Get the full path identifier for this namespace."""
        return f"{self.workspace_id}/{self.slug}"
