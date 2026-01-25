"""FastAPI dependencies for Khora Memory Lake."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request, status

from khora.acl import ACLContext, ACLEnforcer, Principal
from khora.config import KhoraConfig
from khora.memory_lake import MemoryLake

# Global instances (set during app startup)
_memory_lake: MemoryLake | None = None
_acl_enforcer: ACLEnforcer | None = None


def set_memory_lake(lake: MemoryLake) -> None:
    """Set the global MemoryLake instance."""
    global _memory_lake
    _memory_lake = lake


def set_acl_enforcer(enforcer: ACLEnforcer) -> None:
    """Set the global ACL enforcer."""
    global _acl_enforcer
    _acl_enforcer = enforcer


async def get_memory_lake() -> MemoryLake:
    """Get the MemoryLake instance."""
    if _memory_lake is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory Lake not initialized",
        )
    return _memory_lake


async def get_config(request: Request) -> KhoraConfig:
    """Get the application configuration."""
    return request.app.state.config


async def get_acl_enforcer() -> ACLEnforcer:
    """Get the ACL enforcer."""
    if _acl_enforcer is None:
        # Return a disabled enforcer if not configured
        enforcer = ACLEnforcer()
        enforcer.disable()
        return enforcer
    return _acl_enforcer


async def get_current_principal(
    x_user_id: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Extract the current principal from request headers.

    Supports:
    - X-User-ID header for user identification
    - X-API-Key header for API key authentication
    - Authorization header (Bearer token)
    """
    if x_user_id:
        return Principal.user(x_user_id)

    if x_api_key:
        return Principal.api_key(x_api_key)

    if authorization and authorization.startswith("Bearer "):
        # Extract user from token (simplified - real impl would validate)
        token = authorization[7:]
        return Principal.user(f"token:{token[:8]}")

    # Anonymous/system principal for unauthenticated requests
    return Principal.system()


async def get_acl_context(
    principal: Annotated[Principal, Depends(get_current_principal)],
    namespace_id: UUID | None = None,
) -> ACLContext:
    """Create an ACL context for the current request."""
    return ACLContext(
        principal=principal,
        namespace_id=namespace_id,
    )


def require_namespace_read(
    namespace_id: UUID,
    context: ACLContext,
    enforcer: ACLEnforcer,
) -> None:
    """Require read permission on a namespace."""
    enforcer.check_namespace_read(context, namespace_id)


def require_namespace_write(
    namespace_id: UUID,
    context: ACLContext,
    enforcer: ACLEnforcer,
) -> None:
    """Require write permission on a namespace."""
    enforcer.check_namespace_write(context, namespace_id)


# Type aliases for dependency injection
MemoryLakeDep = Annotated[MemoryLake, Depends(get_memory_lake)]
ConfigDep = Annotated[KhoraConfig, Depends(get_config)]
PrincipalDep = Annotated[Principal, Depends(get_current_principal)]
ACLEnforcerDep = Annotated[ACLEnforcer, Depends(get_acl_enforcer)]
