"""Cross-layer ACL enforcement for Khora."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from loguru import logger

from .checker import ACLChecker, Permission, Principal

if TYPE_CHECKING:
    pass

F = TypeVar("F", bound=Callable[..., Any])


class ACLError(Exception):
    """Raised when an ACL check fails."""

    def __init__(
        self,
        message: str,
        *,
        principal: Principal | None = None,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        required_permission: Permission | None = None,
    ) -> None:
        super().__init__(message)
        self.principal = principal
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.required_permission = required_permission


@dataclass
class ACLContext:
    """Context for ACL enforcement containing the current principal."""

    principal: Principal
    namespace_id: UUID | None = None


class ACLEnforcer:
    """Enforces ACL checks across all storage layers.

    Provides a unified way to check and enforce permissions
    before operations on any storage backend.
    """

    def __init__(self, checker: ACLChecker | None = None) -> None:
        """Initialize the ACL enforcer.

        Args:
            checker: ACLChecker instance (creates new one if None)
        """
        self._checker = checker or ACLChecker()
        self._enabled = True

    @property
    def checker(self) -> ACLChecker:
        """Get the underlying ACL checker."""
        return self._checker

    @property
    def enabled(self) -> bool:
        """Check if ACL enforcement is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable ACL enforcement."""
        self._enabled = True
        logger.info("ACL enforcement enabled")

    def disable(self) -> None:
        """Disable ACL enforcement (for testing/development)."""
        self._enabled = False
        logger.warning("ACL enforcement disabled")

    def check_permission(
        self,
        context: ACLContext,
        resource_type: str,
        resource_id: UUID,
        required_permission: Permission,
    ) -> bool:
        """Check if the current principal has permission.

        Args:
            context: ACL context with principal
            resource_type: Type of resource
            resource_id: ID of resource
            required_permission: Permission required

        Returns:
            True if permitted

        Raises:
            ACLError: If permission denied
        """
        if not self._enabled:
            return True

        has_permission = self._checker.check(
            context.principal,
            resource_type,
            resource_id,
            required_permission,
        )

        if not has_permission:
            raise ACLError(
                f"Permission denied: {context.principal.principal_type}:{context.principal.principal_id} "
                f"requires {required_permission.value} on {resource_type}/{resource_id}",
                principal=context.principal,
                resource_type=resource_type,
                resource_id=resource_id,
                required_permission=required_permission,
            )

        return True

    def check_namespace_read(self, context: ACLContext, namespace_id: UUID) -> bool:
        """Check read permission on a namespace."""
        return self.check_permission(context, "namespace", namespace_id, Permission.READ)

    def check_namespace_write(self, context: ACLContext, namespace_id: UUID) -> bool:
        """Check write permission on a namespace."""
        return self.check_permission(context, "namespace", namespace_id, Permission.WRITE)

    def check_namespace_admin(self, context: ACLContext, namespace_id: UUID) -> bool:
        """Check admin permission on a namespace."""
        return self.check_permission(context, "namespace", namespace_id, Permission.ADMIN)

    def require(
        self,
        resource_type: str,
        permission: Permission,
        *,
        resource_id_param: str = "namespace_id",
        context_param: str = "context",
    ) -> Callable[[F], F]:
        """Decorator to require a permission for a function.

        Usage:
            @enforcer.require("namespace", Permission.WRITE)
            async def create_document(context: ACLContext, namespace_id: UUID, ...):
                ...

        Args:
            resource_type: Type of resource to check
            permission: Required permission
            resource_id_param: Name of parameter containing resource ID
            context_param: Name of parameter containing ACLContext

        Returns:
            Decorator function
        """

        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Get context and resource_id from kwargs
                context = kwargs.get(context_param)
                resource_id = kwargs.get(resource_id_param)

                if context is None or resource_id is None:
                    raise ValueError(f"Missing required parameter: {context_param} or {resource_id_param}")

                # Check permission
                self.check_permission(context, resource_type, resource_id, permission)

                return await func(*args, **kwargs)

            return wrapper  # type: ignore

        return decorator


# Default enforcer instance
_default_enforcer: ACLEnforcer | None = None


def get_enforcer() -> ACLEnforcer:
    """Get the default ACL enforcer.

    Returns:
        Default ACLEnforcer instance
    """
    global _default_enforcer
    if _default_enforcer is None:
        _default_enforcer = ACLEnforcer()
    return _default_enforcer
