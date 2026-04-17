"""Small utilities shared by the sqlite_lance adapters.

Kept intentionally minimal — expand in later tickets as adapters land.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID


def uuid_to_text(u: UUID | str) -> str:
    """Return the canonical string form of a UUID (for TEXT columns)."""
    if isinstance(u, UUID):
        return str(u)
    return u


def text_to_uuid(s: str) -> UUID:
    """Parse a TEXT UUID value back into a ``uuid.UUID``."""
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
