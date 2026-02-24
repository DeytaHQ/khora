"""Separate async engine for the telemetry database."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_telemetry_engine(database_url: str) -> AsyncEngine:
    """Create an async engine for the telemetry database.

    The engine is completely independent from khora's main DB engine.

    Args:
        database_url: PostgreSQL connection URL.

    Returns:
        AsyncEngine configured for telemetry writes.
    """
    url = database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    return create_async_engine(
        url,
        echo=False,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        connect_args={"ssl": False},
    )
