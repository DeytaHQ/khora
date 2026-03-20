"""Shared helper utilities for SurrealDB adapters."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from khora.core.models import Entity

# Regex to extract a UUID from a SurrealDB record ID.
# Handles both ``table:uuid`` and ``table:⟨uuid⟩`` forms.
_RECORD_ID_RE = re.compile(r"[^:]+:\u27e8?([0-9a-fA-F\-]{36})\u27e9?")

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _rid(table: str, uid: UUID) -> str:
    """Build a SurrealDB record-link literal ``table:⟨uuid⟩``."""
    return f"{table}:\u27e8{uid}\u27e9"


# Alias used by graph/vector adapters
_record_id = _rid


def _parse_uuid(record_id: str | dict | UUID | Any) -> UUID:
    """Extract a UUID from a SurrealDB record ID.

    Handles strings like ``chunk:018f...``, ``chunk:⟨018f...⟩``,
    bare UUID strings, and ``uuid.UUID`` objects.
    """
    if isinstance(record_id, UUID):
        return record_id
    raw = str(record_id)
    m = _RECORD_ID_RE.match(raw)
    if m:
        return UUID(m.group(1))
    # Fall back: try treating the whole string as a UUID
    return UUID(raw)


def _iso(dt: datetime | None) -> str | None:
    """Convert a datetime to an ISO-8601 string or *None*."""
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(val: Any) -> datetime | None:
    """Best-effort parse of a SurrealDB datetime value."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        raw = str(val)
        # SurrealDB may return ISO strings
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _dt_to_iso(dt: datetime | None) -> str | None:
    """Serialise a datetime to ISO-8601 for SurrealDB, or ``None``."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_dt(value: str | datetime | None) -> datetime | None:
    """Deserialise an ISO-8601 string (or pass-through datetime) from SurrealDB."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


_SAFE_FIELD_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")


def _sanitize_field_name(name: str) -> str:
    """Validate a field/attribute name for safe use in SurrealQL queries.

    Only allows alphanumeric characters, underscores, and dots (for nested access).
    Raises ValueError if the name contains unsafe characters.
    """
    if not name or not _SAFE_FIELD_RE.match(name):
        raise ValueError(f"Unsafe field name for SurrealQL query: {name!r}")
    return name


def _row_to_entity(row: dict[str, Any]) -> Entity:
    """Map a SurrealDB result row to a domain :class:`Entity`."""
    entity_id = _parse_uuid(row.get("id", ""))
    namespace_id = _parse_uuid(row.get("namespace", ""))

    raw_embedding = row.get("embedding")
    if raw_embedding is not None:
        if _HAS_NUMPY:
            embedding: list[float] | Any = np.asarray(raw_embedding, dtype=np.float32)
        else:
            embedding = [float(v) for v in raw_embedding]
    else:
        embedding = None

    src_doc_ids = [UUID(s) for s in (row.get("source_document_ids") or [])]
    src_chunk_ids = [UUID(s) for s in (row.get("source_chunk_ids") or [])]

    return Entity(
        id=entity_id,
        namespace_id=namespace_id,
        name=row.get("name", ""),
        entity_type=row.get("entity_type", "CONCEPT"),
        description=row.get("description", ""),
        attributes=row.get("attributes") or {},
        source_tool=row.get("source_tool", ""),
        source_document_ids=src_doc_ids,
        source_chunk_ids=src_chunk_ids,
        mention_count=int(row.get("mention_count", 1)),
        embedding=embedding,
        embedding_model=row.get("embedding_model", ""),
        valid_from=_parse_dt(row.get("valid_from")),
        valid_until=_parse_dt(row.get("valid_until")),
        confidence=float(row.get("confidence", 1.0)),
        metadata=row.get("metadata_") or {},
        created_at=_parse_dt(row.get("created_at")) or datetime.now(UTC),
        updated_at=_parse_dt(row.get("updated_at")) or datetime.now(UTC),
    )


def _entity_to_bindings(entity: Entity) -> dict[str, Any]:
    """Convert an :class:`Entity` to SurrealQL parameter bindings."""
    return {
        "id": str(entity.id),
        "ns": str(entity.namespace_id),
        "name": entity.name,
        "entity_type": entity.entity_type,
        "description": entity.description,
        "attributes": entity.attributes or {},
        "source_document_ids": [str(uid) for uid in entity.source_document_ids],
        "source_chunk_ids": [str(uid) for uid in entity.source_chunk_ids],
        "source_tool": entity.source_tool,
        "mention_count": entity.mention_count,
        "embedding": list(entity.embedding) if entity.embedding is not None else None,
        "embedding_model": entity.embedding_model,
        "valid_from": _iso(entity.valid_from),
        "valid_until": _iso(entity.valid_until),
        "confidence": entity.confidence,
        "metadata_": entity.metadata or {},
        "created_at": _iso(entity.created_at),
        "updated_at": _iso(entity.updated_at),
    }
