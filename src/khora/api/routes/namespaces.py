"""Namespace management API routes for Khora Memory Lake."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from khora.api.deps import ACLEnforcerDep, MemoryLakeDep, PrincipalDep
from khora.core.models import MemoryNamespace, Organization, Workspace

router = APIRouter(prefix="/namespaces", tags=["namespaces"])


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateOrganizationRequest(BaseModel):
    """Request to create an organization."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str | None = Field(None, min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrganizationResponse(BaseModel):
    """Organization response."""

    id: str
    name: str
    slug: str
    tenancy_mode: str
    created_at: str


class CreateWorkspaceRequest(BaseModel):
    """Request to create a workspace."""

    organization_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    slug: str | None = Field(None, min_length=1, max_length=255)
    description: str = Field("")
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceResponse(BaseModel):
    """Workspace response."""

    id: str
    organization_id: str
    name: str
    slug: str
    description: str
    created_at: str


class CreateNamespaceRequest(BaseModel):
    """Request to create a namespace."""

    workspace_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    slug: str | None = Field(None, min_length=1, max_length=255)
    description: str = Field("")
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NamespaceResponse(BaseModel):
    """Namespace response."""

    id: str
    workspace_id: str
    name: str
    slug: str
    description: str
    config_overrides: dict[str, Any]
    created_at: str


# =============================================================================
# Organization Routes
# =============================================================================


@router.post("/organizations", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(
    request: CreateOrganizationRequest,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    enforcer: ACLEnforcerDep,
) -> OrganizationResponse:
    """Create a new organization."""
    org = Organization(
        name=request.name,
        slug=request.slug or request.name.lower().replace(" ", "-"),
        metadata=request.metadata,
    )

    try:
        created = await lake.storage.create_organization(org)
        return OrganizationResponse(
            id=str(created.id),
            name=created.name,
            slug=created.slug,
            tenancy_mode=created.tenancy_mode.value,
            created_at=created.created_at.isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create organization: {str(e)}",
        )


@router.get("/organizations/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: UUID,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> OrganizationResponse:
    """Get an organization by ID."""
    org = await lake.storage.get_organization(org_id)
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization not found: {org_id}",
        )

    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        slug=org.slug,
        tenancy_mode=org.tenancy_mode.value,
        created_at=org.created_at.isoformat(),
    )


# =============================================================================
# Workspace Routes
# =============================================================================


@router.post("/workspaces", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    request: CreateWorkspaceRequest,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    enforcer: ACLEnforcerDep,
) -> WorkspaceResponse:
    """Create a new workspace."""
    workspace = Workspace(
        organization_id=request.organization_id,
        name=request.name,
        slug=request.slug or request.name.lower().replace(" ", "-"),
        description=request.description,
        metadata=request.metadata,
    )

    try:
        created = await lake.storage.create_workspace(workspace)
        return WorkspaceResponse(
            id=str(created.id),
            organization_id=str(created.organization_id),
            name=created.name,
            slug=created.slug,
            description=created.description,
            created_at=created.created_at.isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workspace: {str(e)}",
        )


@router.get("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: UUID,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> WorkspaceResponse:
    """Get a workspace by ID."""
    workspace = await lake.storage.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace not found: {workspace_id}",
        )

    return WorkspaceResponse(
        id=str(workspace.id),
        organization_id=str(workspace.organization_id),
        name=workspace.name,
        slug=workspace.slug,
        description=workspace.description,
        created_at=workspace.created_at.isoformat(),
    )


@router.get("/organizations/{org_id}/workspaces", response_model=list[WorkspaceResponse])
async def list_workspaces(
    org_id: UUID,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> list[WorkspaceResponse]:
    """List workspaces in an organization."""
    workspaces = await lake.storage.list_workspaces(org_id)

    return [
        WorkspaceResponse(
            id=str(w.id),
            organization_id=str(w.organization_id),
            name=w.name,
            slug=w.slug,
            description=w.description,
            created_at=w.created_at.isoformat(),
        )
        for w in workspaces
    ]


# =============================================================================
# Namespace Routes
# =============================================================================


@router.post("/", response_model=NamespaceResponse, status_code=status.HTTP_201_CREATED)
async def create_namespace(
    request: CreateNamespaceRequest,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
    enforcer: ACLEnforcerDep,
) -> NamespaceResponse:
    """Create a new memory namespace."""
    namespace = MemoryNamespace(
        workspace_id=request.workspace_id,
        name=request.name,
        slug=request.slug or request.name.lower().replace(" ", "-"),
        description=request.description,
        config_overrides=request.config_overrides,
        metadata=request.metadata,
    )

    try:
        created = await lake.storage.create_namespace(namespace)
        return NamespaceResponse(
            id=str(created.id),
            workspace_id=str(created.workspace_id),
            name=created.name,
            slug=created.slug,
            description=created.description,
            config_overrides=created.config_overrides,
            created_at=created.created_at.isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create namespace: {str(e)}",
        )


@router.get("/{namespace_id}", response_model=NamespaceResponse)
async def get_namespace(
    namespace_id: UUID,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> NamespaceResponse:
    """Get a namespace by ID."""
    namespace = await lake.storage.get_namespace(namespace_id)
    if not namespace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Namespace not found: {namespace_id}",
        )

    return NamespaceResponse(
        id=str(namespace.id),
        workspace_id=str(namespace.workspace_id),
        name=namespace.name,
        slug=namespace.slug,
        description=namespace.description,
        config_overrides=namespace.config_overrides,
        created_at=namespace.created_at.isoformat(),
    )


@router.get("/workspaces/{workspace_id}/namespaces", response_model=list[NamespaceResponse])
async def list_namespaces(
    workspace_id: UUID,
    lake: MemoryLakeDep,
    principal: PrincipalDep,
) -> list[NamespaceResponse]:
    """List namespaces in a workspace."""
    namespaces = await lake.storage.list_namespaces(workspace_id)

    return [
        NamespaceResponse(
            id=str(n.id),
            workspace_id=str(n.workspace_id),
            name=n.name,
            slug=n.slug,
            description=n.description,
            config_overrides=n.config_overrides,
            created_at=n.created_at.isoformat(),
        )
        for n in namespaces
    ]
