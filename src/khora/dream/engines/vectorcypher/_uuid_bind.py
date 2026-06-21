"""Dialect-aware UUID binding for the vectorcypher apply handlers (#1277).

The vectorcypher mutation handlers (dedupe / prune_edges / normalize_schema)
issue ``session.execute(text(...))`` statements that bind ``uuid.UUID`` values
into the WHERE clause and SET list. PostgreSQL's asyncpg driver binds those
natively; SQLite's aiosqlite driver raises
``sqlite3.ProgrammingError: type 'UUID' is not supported``. That is the #875
"UUID-bind unsafety" class that kept these ops Postgres-only.

SQLAlchemy's ``UUID(as_uuid=True)`` serializes UUIDs to 32-char hex **without**
dashes on SQLite (see ``khora.storage.backends.sqlite_lance._helpers``), so the
fix is to bind ``uuid.hex`` whenever the active session speaks SQLite and the
raw ``UUID`` everywhere else. ``uuid_bind(session)`` reads the dialect once and
returns the appropriate converter.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID


def _is_sqlite(session: Any) -> bool:
    """Best-effort read of the session dialect; True only for SQLite."""
    bind = getattr(session, "bind", None)
    dialect = getattr(bind, "dialect", None)
    name = getattr(dialect, "name", None)
    return name == "sqlite"


def uuid_bind(session: Any):
    """Return a converter that makes a ``UUID`` safe to bind on this session.

    On SQLite, UUID columns store as 32-char hex (no dashes), so the
    converter returns ``value.hex``. On every other dialect (notably
    PostgreSQL) the raw ``UUID`` binds natively and is returned unchanged.
    ``None`` passes through untouched.
    """
    sqlite = _is_sqlite(session)

    def convert(value: UUID | None) -> Any:
        if value is None:
            return None
        if sqlite:
            return value.hex
        return value

    return convert
