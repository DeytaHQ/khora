"""Small utilities shared by the sqlite_lance adapters.

Kept intentionally minimal — expand in later tickets as adapters land.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID


def uuid_to_text(u: UUID | str) -> str:
    """Return the UUID format used by the SQLite schema.

    SQLAlchemy's ``UUID(as_uuid=True)`` serializes to 32-char hex **without**
    dashes on SQLite.  The raw-aiosqlite adapters share the same ``entities``
    / ``chunks`` / ``documents`` tables as the SQLAlchemy-based relational
    adapter and must produce the same on-disk format — otherwise cross-store
    foreign keys (``chunks.document_id → documents.id``) never match and
    ``UUID(as_uuid=True)`` readback on the relational side fails on 36-char
    dashed strings.
    """
    if isinstance(u, UUID):
        return u.hex
    # Already-stringified input: strip dashes defensively so a caller passing
    # ``str(uuid)`` still lands in the canonical hex-no-dashes form.
    return u.replace("-", "")


def text_to_uuid(s: str) -> UUID:
    """Parse a TEXT UUID value back into a ``uuid.UUID``.

    Accepts both dashed (36-char) and non-dashed (32-char) forms so
    historical databases written by the older dashed-UUID adapter still
    read back.
    """
    return UUID(s)


def to_json_text(obj: Any) -> str:
    """Serialize ``obj`` to a compact JSON string for TEXT columns."""
    return json.dumps(obj, separators=(",", ":"), default=str)


def from_json_text(s: str) -> dict[str, Any]:
    """Parse a JSON TEXT column back into a dict."""
    result = json.loads(s)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result).__name__}")
    return result


def iso8601(dt: datetime | None) -> str | None:
    """Format a datetime as ISO-8601, or None if the input is None."""
    if dt is None:
        return None
    return dt.isoformat()
