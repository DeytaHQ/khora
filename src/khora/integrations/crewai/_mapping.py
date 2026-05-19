"""Translation helpers between CrewAI's ``MemoryRecord`` and khora's models.

CrewAI's memory record (``crewai.memory.types.MemoryRecord``) is a flat
record with hierarchical scope + free-form categories + arbitrary
metadata. khora speaks in documents, chunks, and entities scoped by a
single ``namespace_id``. This module owns every place the two shapes
meet — keeping it in one file makes it obvious where new fields need
threading through when CrewAI adds them upstream.

Mapping summary:

* ``MemoryRecord.id``           ↔ khora document ``external_id`` (stable round-trip)
* ``MemoryRecord.content``      ↔ khora ``Document.content`` / ``Chunk.content``
* ``MemoryRecord.scope``        ↔ stamped on ``Document.metadata["crewai_scope"]``
*                                 + parsed for a trailing UUID-shaped tail
*                                   that becomes ``Chunk.session_id``
* ``MemoryRecord.categories``   ↔ ``Document.metadata["crewai_categories"]``
* ``MemoryRecord.importance``   ↔ ``Document.metadata["crewai_importance"]``
* ``MemoryRecord.metadata``     ↔ merged into ``Document.metadata``
* ``MemoryRecord.created_at``   ↔ best-effort from ``Chunk.created_at``
* ``MemoryRecord.source``       ↔ ``Document.metadata["crewai_source"]``
* ``MemoryRecord.private``      ↔ ``Document.metadata["crewai_private"]``

The translation is one-way enriched: round-tripping a record through
``record_to_remember_kwargs`` then ``chunk_to_record`` preserves the
public fields (id, content, scope, categories, importance, metadata,
source, private). Embedding is intentionally dropped — CrewAI computes
its own at recall-time anyway, and storing two embedding columns per
chunk would waste space on the embedded sqlite_lance backend.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:  # pragma: no cover - typing only
    from khora.core.models.document import Chunk


# Sentinel keys we own under ``Document.metadata``. Prefix with
# ``crewai_`` so adapter writes don't collide with the user's own
# metadata, which is merged in alongside.
_KEY_SCOPE = "crewai_scope"
_KEY_CATEGORIES = "crewai_categories"
_KEY_IMPORTANCE = "crewai_importance"
_KEY_SOURCE = "crewai_source"
_KEY_PRIVATE = "crewai_private"
_KEY_CREATED_AT = "crewai_created_at"
_KEY_LAST_ACCESSED = "crewai_last_accessed"
_KEY_USER_ID = "crewai_user_id"
_KEY_APP_ID = "crewai_app_id"


def session_id_from_scope(scope: str) -> UUID | None:
    """Return the trailing UUID-shaped segment of ``scope`` if present.

    CrewAI's scope tree is hierarchical (``/crew/research/<session>``)
    and the adapter convention is to put a UUID at the tail when the
    caller wants session-scoped retention. This helper is a no-op for
    scopes that don't include a UUID — most non-session contexts use
    semantic paths like ``/crew/research/ai``.
    """
    if not scope:
        return None
    tail = scope.rstrip("/").rsplit("/", 1)[-1]
    if not tail:
        return None
    try:
        return UUID(tail)
    except ValueError:
        return None


def record_to_remember_kwargs(
    record: Any,
    *,
    user_id: str,
    app_id: str,
) -> dict[str, Any]:
    """Translate a CrewAI ``MemoryRecord`` into ``Khora.remember(...)`` kwargs.

    The returned dict is keyword-only and is intended to be unpacked
    directly into ``Khora.remember(**kwargs)``. ``namespace`` is the
    caller's responsibility — adapters pass the bound namespace UUID
    separately so this function stays stateless.

    Args:
        record: A ``crewai.memory.types.MemoryRecord``-shaped object.
            Duck-typed to keep this module import-free at module load.
        user_id: Stable end-user identifier from the adapter factory.
            Stamped on every record so a single khora namespace can
            host multi-user CrewAI sessions without silent cross-talk.
        app_id: Adapter app identifier (defaults to ``"crewai"``).

    Returns:
        A dict with ``content``, ``title`` (empty), ``source`` (empty),
        ``metadata``, ``external_id``, ``session_id``, plus the
        ``entity_types`` / ``relationship_types`` extraction-required
        kwargs (set to empty lists — the adapter intentionally bypasses
        extraction; CrewAI's encoding flow has already analysed the
        content and we don't pay for a second LLM call).
    """
    user_metadata = dict(record.metadata or {})
    # Carry the CrewAI public fields into our own ``crewai_*`` keys so
    # ``chunk_to_record`` can rebuild a faithful MemoryRecord later.
    crewai_meta = {
        _KEY_SCOPE: record.scope or "/",
        _KEY_CATEGORIES: list(record.categories or []),
        _KEY_IMPORTANCE: float(record.importance),
        _KEY_SOURCE: record.source,
        _KEY_PRIVATE: bool(record.private),
        _KEY_CREATED_AT: _isoformat(record.created_at),
        _KEY_LAST_ACCESSED: _isoformat(record.last_accessed),
        _KEY_USER_ID: user_id,
        _KEY_APP_ID: app_id,
    }
    merged: dict[str, Any] = {**user_metadata, **crewai_meta}

    return {
        "content": record.content,
        "title": "",
        "source": "",
        "metadata": merged,
        "external_id": str(record.id),
        "session_id": session_id_from_scope(record.scope or ""),
        # Empty extraction directive: CrewAI's Memory has already analysed
        # the content (scope, categories, importance) — we don't trigger a
        # second LLM call to extract entities khora doesn't surface back
        # through the StorageBackend Protocol anyway.
        "entity_types": [],
        "relationship_types": [],
    }


def _strip_tz(value: datetime | None) -> datetime | None:
    """Return a naive datetime (drops tzinfo).

    CrewAI's internal scoring (``types.py:371``) computes recency via
    ``datetime.now() - record.created_at`` using a NAIVE
    ``datetime.now()``. khora stores timestamps as tz-aware UTC. The
    subtraction raises ``TypeError: can't subtract offset-naive and
    offset-aware datetimes`` unless we hand CrewAI naive values. We
    drop tzinfo on the round-trip to match CrewAI's expectations;
    on-disk khora chunks keep their UTC tagging unchanged.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    # Convert to UTC then drop tzinfo so timestamps remain comparable.
    return value.astimezone(UTC).replace(tzinfo=None)


def chunk_to_record(
    chunk: Chunk,
    memory_record_cls: type,
) -> Any:
    """Rebuild a ``MemoryRecord`` from a khora ``Chunk``.

    ``memory_record_cls`` is passed in (rather than imported at module
    scope) so this module stays free of any top-level ``crewai`` import
    — the adapter loads it lazily inside ``KhoraStorageBackend``.

    Args:
        chunk: A ``khora.core.models.document.Chunk``.
        memory_record_cls: The ``crewai.memory.types.MemoryRecord``
            class — passed as a runtime parameter so this module never
            imports crewai at top level.

    Returns:
        A populated ``MemoryRecord``.
    """
    custom = chunk.metadata or {}
    # Strip our internal keys so the round-tripped ``metadata`` dict
    # contains only the user's own keys.
    user_metadata = {k: v for k, v in custom.items() if not k.startswith("crewai_")}

    record_id = custom.get("external_id") or str(chunk.document_id)
    created_at = _strip_tz(_parse_isoformat(custom.get(_KEY_CREATED_AT)) or chunk.created_at)
    last_accessed = _strip_tz(_parse_isoformat(custom.get(_KEY_LAST_ACCESSED)) or chunk.created_at)
    return memory_record_cls(
        id=record_id,
        content=chunk.content,
        scope=custom.get(_KEY_SCOPE, "/"),
        categories=list(custom.get(_KEY_CATEGORIES) or []),
        metadata=user_metadata,
        importance=float(custom.get(_KEY_IMPORTANCE, 0.5)),
        created_at=created_at,
        last_accessed=last_accessed,
        source=custom.get(_KEY_SOURCE),
        private=bool(custom.get(_KEY_PRIVATE, False)),
    )


def _isoformat(value: Any) -> str | None:
    """Render a datetime as an ISO-8601 string for JSON-safe storage."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    # Defensive: caller may already have stringified
    return str(value)


def _parse_isoformat(value: Any) -> datetime | None:
    """Parse an ISO-8601 string back into a tz-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
