"""Schema synchronisation helpers.

Ensures PostgreSQL enum types stay in sync with their Python counterparts
across library upgrades, even when ``Base.metadata.create_all`` is used
(which only creates a type if it doesn't already exist).
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from khora.core.models.document import DocumentStatus

# Map of PostgreSQL enum type name → Python enum members.
_ENUM_SYNC: dict[str, list[str]] = {
    "document_status": [s.value for s in DocumentStatus],
}


async def sync_enum_values(engine: AsyncEngine) -> None:
    """Add any missing values to PostgreSQL enum types.

    ``ALTER TYPE … ADD VALUE IF NOT EXISTS`` cannot run inside a
    transaction, so we obtain a connection in *autocommit* mode.
    """
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        for type_name, values in _ENUM_SYNC.items():
            for value in values:
                stmt = text(f"ALTER TYPE {type_name} ADD VALUE IF NOT EXISTS :val")
                try:
                    await conn.execute(stmt, {"val": value})
                except Exception:
                    # Type may not exist yet (first run) — create_all will
                    # create it with all values.  Log and continue.
                    logger.debug(f"Skipping enum sync for {type_name}.{value}")
                    break
