"""Database session management for Khora."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def get_database_url() -> str:
    """Get database URL from environment.

    Converts postgresql:// to postgresql+asyncpg:// for async support.
    """
    url = os.getenv("KHORA_DATABASE_URL", "")
    if not url:
        raise ValueError("KHORA_DATABASE_URL environment variable not set")

    # Convert to async URL if needed
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    return url


class DatabaseManager:
    """Manages database engine and session factory lifecycle.

    Encapsulates what was previously module-level global state,
    enabling proper isolation in tests and multi-database scenarios.
    """

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    def get_engine(self) -> AsyncEngine:
        """Get or create the database engine."""
        if self._engine is None:
            self._engine = create_async_engine(
                get_database_url(),
                echo=os.getenv("KHORA_DEBUG", "").lower() == "true",
                pool_size=20,
                max_overflow=30,
                connect_args={"ssl": False},
            )
        return self._engine

    def get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.get_engine(),
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._session_factory

    @asynccontextmanager
    async def get_db(self) -> AsyncGenerator[AsyncSession]:
        """Get a database session."""
        session = self.get_session_factory()()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def init_db(self) -> None:
        """Initialize database tables."""
        from .models import Base

        engine = self.get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close_db(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    def reset(self) -> None:
        """Reset state without async disposal. For test cleanup."""
        self._engine = None
        self._session_factory = None


_default_manager: DatabaseManager | None = None


def get_default_manager() -> DatabaseManager:
    """Get the default DatabaseManager instance (lazy singleton)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = DatabaseManager()
    return _default_manager


# Backward-compatible module-level functions


def get_engine() -> AsyncEngine:
    """Get or create the database engine."""
    return get_default_manager().get_engine()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    return get_default_manager().get_session_factory()


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession]:
    """Get a database session.

    Usage:
        async with get_db() as db:
            result = await db.execute(...)
    """
    async with get_default_manager().get_db() as session:
        yield session


async def init_db() -> None:
    """Initialize database tables.

    For development/testing only. Use Alembic migrations in production.
    """
    await get_default_manager().init_db()


async def close_db() -> None:
    """Close database connections."""
    await get_default_manager().close_db()


def _run_migrations_sync() -> None:
    """Internal function to run migrations synchronously."""
    from pathlib import Path

    from loguru import logger

    from alembic import command
    from alembic.config import Config

    # Check if database URL is configured
    url = os.getenv("KHORA_DATABASE_URL", "")
    if not url:
        logger.warning("KHORA_DATABASE_URL not set, skipping migrations")
        return

    # Find alembic.ini - look in common locations
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "alembic.ini",  # src/khora/db -> root
        Path.cwd() / "alembic.ini",
        Path("/app/alembic.ini"),  # Docker container path
    ]

    alembic_cfg_path = None
    for path in possible_paths:
        if path.exists():
            alembic_cfg_path = path
            break

    if alembic_cfg_path is None:
        logger.warning("alembic.ini not found, skipping migrations")
        return

    logger.info(f"Running database migrations from {alembic_cfg_path}")

    alembic_cfg = Config(str(alembic_cfg_path))
    # Override the script location to be relative to alembic.ini
    alembic_cfg.set_main_option("script_location", str(alembic_cfg_path.parent / "alembic"))

    command.upgrade(alembic_cfg, "head")
    logger.info("Database migrations completed")


async def run_migrations() -> None:
    """Run database migrations using Alembic.

    Runs migrations in a thread pool to avoid conflicts with the running event loop.
    """
    import asyncio
    import concurrent.futures

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, _run_migrations_sync)
