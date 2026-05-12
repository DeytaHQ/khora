"""Database session management for Khora."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

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


@dataclass(slots=True, frozen=True)
class MigrationResult:
    """Result of running database migrations."""

    success: bool
    target_revision: str | None
    current_revision: str | None
    elapsed_seconds: float
    skipped: bool = False
    error: str | None = None


class _DatabaseAheadError(Exception):
    """Raised when the DB revision is not recognized by the local migration scripts.

    Used by ``env.py`` to signal ``_run_migrations_sync`` that the database
    is ahead of this Khora version and migrations should be skipped.
    """

    def __init__(self, current_revision: str) -> None:
        self.current_revision = current_revision
        super().__init__(f"Database at unrecognized revision: {current_revision}")


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
        """Initialize database tables.

        .. deprecated::
            Use ``run_migrations()`` instead. ``init_db()`` bypasses Alembic
            and masks missing migrations. Will be removed in a future release.
        """
        import warnings

        warnings.warn(
            "init_db() is deprecated. Use khora.db.run_migrations() instead. "
            "init_db() bypasses Alembic and masks missing migrations.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    .. deprecated::
        Use ``run_migrations()`` instead. ``init_db()`` bypasses Alembic
        and masks missing migrations. Will be removed in a future release.
    """
    await get_default_manager().init_db()


async def close_db() -> None:
    """Close database connections."""
    await get_default_manager().close_db()


def _run_migrations_sync(database_url: str | None = None) -> MigrationResult:
    """Run Alembic migrations synchronously.

    Args:
        database_url: PostgreSQL URL. Falls back to KHORA_DATABASE_URL env var.

    Returns:
        MigrationResult with outcome details.
    """
    import time
    from pathlib import Path

    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from loguru import logger

    url = database_url or os.getenv("KHORA_DATABASE_URL", "")
    if not url:
        return MigrationResult(
            success=False,
            target_revision=None,
            current_revision=None,
            elapsed_seconds=0.0,
            error="No database URL. Set KHORA_DATABASE_URL or pass database_url.",
        )

    start = time.monotonic()

    # Build programmatic Config — no alembic.ini needed
    alembic_cfg = Config()
    migrations_dir = str(Path(__file__).parent / "migrations")
    alembic_cfg.set_main_option("script_location", migrations_dir)
    alembic_cfg.set_main_option("sqlalchemy.url", "")  # unused, env.py reads attributes
    alembic_cfg.attributes["database_url"] = url

    try:
        script = ScriptDirectory.from_config(alembic_cfg)
        head = script.get_current_head()
        logger.info("Running khora database migrations...")
        command.upgrade(alembic_cfg, "head")
        elapsed = time.monotonic() - start

        logger.info("Migrations completed in {:.2f}s", elapsed)
        return MigrationResult(
            success=True,
            target_revision=head,
            current_revision=head,
            elapsed_seconds=elapsed,
        )
    except _DatabaseAheadError as exc:
        elapsed = time.monotonic() - start
        logger.warning(
            "Migrations skipped (database is ahead at {}) in {:.2f}s",
            exc.current_revision,
            elapsed,
        )
        return MigrationResult(
            success=True,
            target_revision=head,
            current_revision=None,
            elapsed_seconds=elapsed,
            skipped=True,
        )
    except Exception as e:
        # ADR-084 §⚠️: scrub plaintext DSN userinfo from error messages before
        # they hit logs or the returned MigrationResult.error string.
        from khora.config._secrets import redact_dsn

        elapsed = time.monotonic() - start
        redacted_msg = redact_dsn(str(e))
        logger.error("Migration failed: {}", redacted_msg)
        return MigrationResult(
            success=False,
            target_revision=None,
            current_revision=None,
            elapsed_seconds=elapsed,
            error=f"{type(e).__name__}: {redacted_msg}",
        )


async def run_migrations(database_url: str | None = None) -> MigrationResult:
    """Run database migrations using Alembic.

    Runs in a thread pool to avoid blocking the event loop.

    Args:
        database_url: PostgreSQL URL. Falls back to KHORA_DATABASE_URL env var.

    Returns:
        MigrationResult with outcome details.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_migrations_sync, database_url)
