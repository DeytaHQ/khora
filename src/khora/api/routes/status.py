"""Status endpoints for Khora API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/status")
async def status_check(request: Request) -> dict[str, Any]:
    """Basic status check endpoint.

    Returns:
        Status with timestamp and version
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "0.0.8",
        "service": "khora",
    }


@router.get("/health")
async def health_check(request: Request) -> dict[str, Any]:
    """Health check endpoint for orchestration systems.

    Returns:
        Health status with timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "0.0.8",
    }


@router.get("/health/ready")
async def readiness_check(request: Request) -> dict[str, Any]:
    """Readiness check for Kubernetes/orchestration.

    Checks that all required services are available.

    Returns:
        Readiness status with component checks
    """
    config = request.app.state.config
    checks: dict[str, bool] = {}

    # TODO: Add actual health checks for:
    # - Database connections
    # - External services

    # For now, return basic status
    checks["config_loaded"] = config is not None

    all_healthy = all(checks.values())

    return {
        "status": "ready" if all_healthy else "not_ready",
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": checks,
    }


@router.get("/health/live")
async def liveness_check() -> dict[str, Any]:
    """Liveness check for Kubernetes/orchestration.

    Simple check that the application is running.

    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(UTC).isoformat(),
    }
