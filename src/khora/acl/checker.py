"""ACL checker with permission inheritance for Khora Memory Lake."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from loguru import logger


class Permission(str, Enum):
    """Permission levels in the ACL system."""

    READ = "read"  # Can read/query data
    WRITE = "write"  # Can create/update data
    ADMIN = "admin"  # Can manage permissions
    OWNER = "owner"  # Full control including deletion

    def __ge__(self, other: Permission) -> bool:
        """Check if this permission is >= another."""
        order = [Permission.READ, Permission.WRITE, Permission.ADMIN, Permission.OWNER]
        return order.index(self) >= order.index(other)

    def __gt__(self, other: Permission) -> bool:
        """Check if this permission is > another."""
        return self != other and self >= other

    def __le__(self, other: Permission) -> bool:
        """Check if this permission is <= another."""
        return other >= self

    def __lt__(self, other: Permission) -> bool:
        """Check if this permission is < another."""
        return other > self


@dataclass
class Principal:
    """A principal (user, role, or API key) that can have permissions."""

    principal_type: str  # user, role, api_key
    principal_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, user_id: str) -> Principal:
        """Create a user principal."""
        return cls(principal_type="user", principal_id=user_id)

    @classmethod
    def role(cls, role_name: str) -> Principal:
        """Create a role principal."""
        return cls(principal_type="role", principal_id=role_name)

    @classmethod
    def api_key(cls, key_id: str) -> Principal:
        """Create an API key principal."""
        return cls(principal_type="api_key", principal_id=key_id)

    @classmethod
    def system(cls) -> Principal:
        """Create a system principal with full access."""
        return cls(principal_type="system", principal_id="system")


@dataclass
class PermissionGrant:
    """A permission grant for a principal on a resource."""

    principal: Principal
    resource_type: str  # namespace
    resource_id: UUID
    permission: Permission


class ACLChecker:
    """ACL checker for namespace-only permissions."""

    # Resource type hierarchy (namespace is the sole boundary)
    HIERARCHY = {
        "namespace": None,
    }

    def __init__(self) -> None:
        """Initialize the ACL checker."""
        # In-memory permission store for quick lookups
        # In production, this would be backed by the database
        self._grants: list[PermissionGrant] = []

    def grant(
        self,
        principal: Principal,
        resource_type: str,
        resource_id: UUID,
        permission: Permission,
    ) -> PermissionGrant:
        """Grant a permission to a principal.

        Args:
            principal: Principal to grant permission to
            resource_type: Type of resource
            resource_id: ID of resource
            permission: Permission to grant

        Returns:
            PermissionGrant object
        """
        grant = PermissionGrant(
            principal=principal,
            resource_type=resource_type,
            resource_id=resource_id,
            permission=permission,
        )
        self._grants.append(grant)
        logger.debug(
            f"Granted {permission.value} on {resource_type}/{resource_id} to {principal.principal_type}:{principal.principal_id}"
        )
        return grant

    def revoke(
        self,
        principal: Principal,
        resource_type: str,
        resource_id: UUID,
        permission: Permission | None = None,
    ) -> int:
        """Revoke permissions from a principal.

        Args:
            principal: Principal to revoke from
            resource_type: Type of resource
            resource_id: ID of resource
            permission: Specific permission to revoke (None = all)

        Returns:
            Number of grants revoked
        """
        before = len(self._grants)
        self._grants = [
            g
            for g in self._grants
            if not (
                g.principal.principal_type == principal.principal_type
                and g.principal.principal_id == principal.principal_id
                and g.resource_type == resource_type
                and g.resource_id == resource_id
                and (permission is None or g.permission == permission)
            )
        ]
        revoked = before - len(self._grants)
        if revoked:
            logger.debug(f"Revoked {revoked} grants from {principal.principal_type}:{principal.principal_id}")
        return revoked

    def check(
        self,
        principal: Principal,
        resource_type: str,
        resource_id: UUID,
        required_permission: Permission,
    ) -> bool:
        """Check if a principal has a permission on a resource.

        Args:
            principal: Principal to check
            resource_type: Type of resource
            resource_id: ID of resource
            required_permission: Permission required

        Returns:
            True if permission is granted
        """
        # System principal always has access
        if principal.principal_type == "system":
            return True

        # Check direct grant
        return self._has_direct_grant(principal, resource_type, resource_id, required_permission)

    def _has_direct_grant(
        self,
        principal: Principal,
        resource_type: str,
        resource_id: UUID,
        required_permission: Permission,
    ) -> bool:
        """Check for a direct grant (no inheritance)."""
        for grant in self._grants:
            if (
                grant.principal.principal_type == principal.principal_type
                and grant.principal.principal_id == principal.principal_id
                and grant.resource_type == resource_type
                and grant.resource_id == resource_id
                and grant.permission >= required_permission
            ):
                return True
        return False

    def get_effective_permissions(
        self,
        principal: Principal,
        resource_type: str,
        resource_id: UUID,
    ) -> list[PermissionGrant]:
        """Get all effective permissions for a principal on a resource.

        Args:
            principal: Principal to check
            resource_type: Type of resource
            resource_id: ID of resource

        Returns:
            List of effective PermissionGrant objects
        """
        effective = []

        for grant in self._grants:
            if (
                grant.principal.principal_type == principal.principal_type
                and grant.principal.principal_id == principal.principal_id
                and grant.resource_type == resource_type
                and grant.resource_id == resource_id
            ):
                effective.append(grant)

        return effective

    def list_grants(
        self,
        *,
        principal: Principal | None = None,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
    ) -> list[PermissionGrant]:
        """List grants with optional filtering.

        Args:
            principal: Filter by principal
            resource_type: Filter by resource type
            resource_id: Filter by resource ID

        Returns:
            List of matching grants
        """
        grants = self._grants

        if principal:
            grants = [
                g
                for g in grants
                if g.principal.principal_type == principal.principal_type
                and g.principal.principal_id == principal.principal_id
            ]

        if resource_type:
            grants = [g for g in grants if g.resource_type == resource_type]

        if resource_id:
            grants = [g for g in grants if g.resource_id == resource_id]

        return grants
