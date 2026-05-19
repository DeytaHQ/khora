"""Translation helpers between Google ADK's ``Event`` / ``Content`` and khora.

ADK identifies a memory entry by ``MemoryEntry(content: google.genai.types.Content,
author, timestamp, ...)`` plus the surrounding ``Session(app_name, user_id,
id, events: list[Event])``. khora speaks documents + chunks scoped by a
single ``namespace_id`` with a first-class ``session_id`` column (#620).
This module is the single place those shapes meet.

Mapping summary:

* ``Session.app_name`` + ``Session.user_id`` → ``khora namespace_id`` via
  :func:`namespace_uuid` (UUID5 of ``"adk:{app}:{user}"``).
* ``Session.id`` → ``khora session_id`` via :func:`session_uuid` (UUID5 so
  ADK session id strings round-trip stably; raw UUIDs pass through).
* ``Event.content.parts`` → ``Document.content`` (text parts concatenated
  with ``"\n"``). Non-text parts (``function_call``, ``function_response``,
  ``inline_data``) are JSON-encoded into ``metadata.custom["adk_parts"]``;
  ``inline_data.data`` bytes are dropped (mime type + bounded-hash kept).
* ``Event.id`` → ``Document.external_id`` (deduplication key for
  ``add_events_to_memory``).
* ``Event.author`` → ``metadata.custom["adk_author"]`` (round-trips back
  to ``MemoryEntry.author``).
* ``Event.timestamp`` (float epoch) → ``Document.source_timestamp``
  (tz-aware UTC) AND ``MemoryEntry.timestamp`` (ISO 8601 on read-back).

Private to the adapter — not part of the public ``khora.integrations`` API.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_DNS, UUID, uuid5

if TYPE_CHECKING:  # pragma: no cover - typing only
    from khora.core.models.document import Chunk

# Stable UUID5 root for ADK namespaces. Versioned so a future scheme change
# can rotate without colliding with shipped data.
_NAMESPACE_ROOT = uuid5(NAMESPACE_DNS, "khora.integrations.google_adk.v1")
_SESSION_ROOT = uuid5(NAMESPACE_DNS, "khora.integrations.google_adk.session.v1")

# Document.external_id is VARCHAR(512). ADK Event ids fit comfortably; this
# guard is defensive in case ADK changes its id generator upstream.
_EXTERNAL_ID_MAX_LEN = 512

# Metadata keys this adapter owns under ``Document.metadata``. Prefix
# with ``adk_`` so caller-supplied metadata cannot collide.
KEY_APP_ID = "adk_app_id"
KEY_USER_ID = "adk_user_id"
KEY_SESSION_ID = "adk_session_id"  # original ADK session string id
KEY_EVENT_ID = "adk_event_id"  # original ADK event id (also stamped on external_id)
KEY_AUTHOR = "adk_author"
KEY_TIMESTAMP = "adk_timestamp"  # ISO 8601 of the original event.timestamp
KEY_PARTS = "adk_parts"  # JSON-encoded list of non-text parts


def namespace_uuid(*, app_name: str, user_id: str) -> UUID:
    """Derive the khora namespace UUID for an ADK (app_name, user_id) pair.

    Two ``KhoraMemoryService`` calls with the same (app_name, user_id)
    map to the same khora namespace, per the #618 canonical mapping
    (``adk:{app_name}:{user_id}``). Used to allocate per-user long-term
    memory without storing a separate registry.
    """
    return uuid5(_NAMESPACE_ROOT, f"adk:{app_name}:{user_id}")


def session_uuid(session_id: str) -> UUID:
    """Map an ADK session id (free-form string) to a stable khora session UUID.

    ADK uses arbitrary strings for ``Session.id`` (e.g. ``"persistent-session-123"``).
    khora's ``session_id`` column is a UUID. We derive a deterministic UUID5
    so calling ``forget_session`` for the same string always hits the right
    rows. If the caller already supplied a UUID string, we honour it
    verbatim — that's the common case for callers that mint UUIDs upstream.
    """
    try:
        return UUID(session_id)
    except (ValueError, AttributeError):
        return uuid5(_SESSION_ROOT, session_id)


def event_external_id(event_id: str) -> str:
    """Map an ADK ``Event.id`` to a khora ``Document.external_id``.

    Prefixed so foreign documents in the same namespace don't shadow ADK
    events, and clamped to 512 chars to fit the DB column.
    """
    raw = f"adk_event:{event_id}"
    if len(raw) <= _EXTERNAL_ID_MAX_LEN:
        return raw
    digest = hashlib.sha1(event_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"adk_event:h{digest}"


def content_to_text(content: Any) -> str:
    """Concatenate the text parts of a ``google.genai.types.Content`` instance.

    Non-text parts (function_call, function_response, inline_data,
    executable_code, ...) are intentionally not surfaced here — they live
    in ``adk_parts`` metadata and are reconstructed by :func:`event_to_memory_entry`.
    Returns ``""`` if the content has no text parts at all.
    """
    if content is None:
        return ""
    parts = getattr(content, "parts", None) or []
    pieces: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            pieces.append(str(text))
    return "\n".join(pieces)


def serialise_non_text_parts(content: Any) -> list[dict[str, Any]]:
    """JSON-safe projection of every non-text Part for round-trip storage.

    Each entry preserves enough to rebuild a useful ``Part`` on read-back:

    * ``function_call`` — ``{"function_call": {"name", "args"}}``
    * ``function_response`` — ``{"function_response": {"name", "response"}}``
    * ``inline_data`` — ``{"inline_data": {"mime_type", "data_sha1"}}``
      (raw bytes intentionally dropped — they bloat the document store
      and aren't useful as long-term memory)
    * other typed parts — best-effort ``pydantic.BaseModel.model_dump``

    Unknown Part variants fall through to a stringified placeholder so
    we never crash on a future ADK type.
    """
    if content is None:
        return []
    parts = getattr(content, "parts", None) or []
    out: list[dict[str, Any]] = []
    for part in parts:
        if getattr(part, "text", None):
            continue  # text already lives in Document.content
        fc = getattr(part, "function_call", None)
        if fc is not None:
            out.append(
                {
                    "function_call": {
                        "name": getattr(fc, "name", None),
                        "args": _to_jsonable(getattr(fc, "args", None)),
                    }
                }
            )
            continue
        fr = getattr(part, "function_response", None)
        if fr is not None:
            out.append(
                {
                    "function_response": {
                        "name": getattr(fr, "name", None),
                        "response": _to_jsonable(getattr(fr, "response", None)),
                    }
                }
            )
            continue
        inline = getattr(part, "inline_data", None)
        if inline is not None:
            raw = getattr(inline, "data", None)
            digest = hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:16] if isinstance(raw, bytes) else None
            out.append(
                {
                    "inline_data": {
                        "mime_type": getattr(inline, "mime_type", None),
                        "data_sha1": digest,
                    }
                }
            )
            continue
        # Fallback: try the pydantic model_dump escape hatch; otherwise stringify.
        dump = getattr(part, "model_dump", None)
        if callable(dump):
            try:
                out.append({"raw": dump(exclude_none=True)})
                continue
            except Exception:  # noqa: BLE001, S110 — last-ditch fallback, repr() below covers any failure
                pass
        out.append({"raw": repr(part)})
    return out


def event_to_remember_kwargs(
    event: Any,
    *,
    session: Any,
    app_id: str,
) -> dict[str, Any] | None:
    """Translate one ADK ``Event`` into ``Khora.remember(**kwargs)``.

    Returns ``None`` for events with no usable content (no text and no
    non-text parts). ADK ingests partial / control-flow events that hold
    only ``actions``; persisting them as memory entries would clutter
    recall results.

    Args:
        event: A ``google.adk.events.Event``-shaped object. Duck-typed.
        session: The owning ``google.adk.sessions.Session``. Provides
            ``app_name`` / ``user_id`` / ``id``.
        app_id: Free-form adapter app identifier, stamped into metadata.

    Returns:
        Keyword dict ready for ``Khora.remember(**kwargs)``, or ``None``.
    """
    content = getattr(event, "content", None)
    text = content_to_text(content)
    non_text = serialise_non_text_parts(content)
    if not text and not non_text:
        return None

    event_id = str(getattr(event, "id", "") or "")
    if not event_id:
        # Defensive: ADK always assigns an id in ``model_post_init``.
        # Falling back to a deterministic hash of (session, timestamp,
        # author, text) keeps re-ingestion idempotent even in that case.
        seed = f"{session.id}:{getattr(event, 'timestamp', 0)}:{getattr(event, 'author', '')}:{text}"
        event_id = hashlib.sha1(seed.encode("utf-8"), usedforsecurity=False).hexdigest()

    timestamp = getattr(event, "timestamp", None)
    source_ts = _epoch_to_utc(timestamp)
    iso_ts = source_ts.isoformat() if source_ts else None

    metadata: dict[str, Any] = {
        KEY_APP_ID: app_id,
        KEY_USER_ID: session.user_id,
        KEY_SESSION_ID: session.id,
        KEY_EVENT_ID: event_id,
        KEY_AUTHOR: getattr(event, "author", None),
        KEY_TIMESTAMP: iso_ts,
    }
    if non_text:
        # JSON-encode so the dict survives the JSONB round-trip even for
        # nested function arguments with custom types.
        metadata[KEY_PARTS] = json.dumps(non_text, default=_to_jsonable)

    # Title kept short; recall results show it in the chunk metadata.
    author = metadata[KEY_AUTHOR] or "event"
    title = f"adk:{author}:{event_id[:12]}"

    return {
        "content": text or _placeholder_for_non_text(non_text),
        "title": title,
        "source": f"google_adk:{app_id}",
        "metadata": metadata,
        "external_id": event_external_id(event_id),
        "session_id": session_uuid(session.id),
        # Empty extraction directive: ADK events are conversation turns,
        # not corpus documents. Extracting entities from every chat turn
        # is wasteful and pollutes the graph. Callers who want extraction
        # can rewrite ingested memory through `Khora.remember` directly.
        "entity_types": [],
        "relationship_types": [],
        "source_timestamp_iso": iso_ts,  # consumed by the service, not by remember()
    }


def chunk_to_memory_entry(
    chunk: Chunk,
    memory_entry_cls: type,
    content_cls: type,
    part_cls: type,
) -> Any:
    """Rebuild a ``MemoryEntry`` from a khora ``Chunk``.

    The ADK classes (``MemoryEntry``, ``Content``, ``Part``) are passed
    as runtime arguments so this module never imports ``google.adk``
    / ``google.genai`` at top level.

    Args:
        chunk: A ``khora.core.models.document.Chunk``.
        memory_entry_cls: ``google.adk.memory.MemoryEntry``.
        content_cls: ``google.genai.types.Content``.
        part_cls: ``google.genai.types.Part``.

    Returns:
        A populated ``MemoryEntry``.
    """
    custom = chunk.metadata or {}
    text = chunk.content or ""

    parts: list[Any] = []
    if text:
        parts.append(part_cls(text=text))

    raw_non_text = custom.get(KEY_PARTS)
    if raw_non_text:
        try:
            decoded = json.loads(raw_non_text) if isinstance(raw_non_text, str) else raw_non_text
        except (TypeError, ValueError):
            decoded = []
        for entry in decoded or []:
            part = _deserialise_part(entry, part_cls)
            if part is not None:
                parts.append(part)

    content = content_cls(parts=parts, role=custom.get(KEY_AUTHOR) or "user")
    return memory_entry_cls(
        content=content,
        author=custom.get(KEY_AUTHOR),
        timestamp=custom.get(KEY_TIMESTAMP),
        custom_metadata={
            k: v
            for k, v in custom.items()
            # Hide our own internal bookkeeping from the LLM-facing metadata.
            if not k.startswith("adk_") or k in {KEY_APP_ID, KEY_USER_ID, KEY_SESSION_ID}
        },
    )


def _deserialise_part(entry: dict[str, Any], part_cls: type) -> Any | None:
    """Recreate a ``google.genai.types.Part`` from a serialised entry."""
    if not isinstance(entry, dict):
        return None
    if "function_call" in entry:
        fc = entry["function_call"] or {}
        try:
            return part_cls(function_call={"name": fc.get("name"), "args": fc.get("args") or {}})
        except Exception:
            return None
    if "function_response" in entry:
        fr = entry["function_response"] or {}
        try:
            return part_cls(function_response={"name": fr.get("name"), "response": fr.get("response") or {}})
        except Exception:
            return None
    if "inline_data" in entry:
        # Bytes were dropped on write; reconstruct an empty-bytes Part
        # so downstream code sees the mime type without crashing.
        idata = entry["inline_data"] or {}
        try:
            return part_cls(inline_data={"mime_type": idata.get("mime_type"), "data": b""})
        except Exception:
            return None
    return None


def _epoch_to_utc(timestamp: Any) -> datetime | None:
    """Convert ADK's float epoch seconds to tz-aware UTC. ``None`` on bad input."""
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(float(timestamp), tz=UTC)
    except (TypeError, ValueError, OSError):
        return None


def _placeholder_for_non_text(non_text: list[dict[str, Any]]) -> str:
    """Render a one-line description for events that carry only tool calls.

    Lets vector recall still surface the event by some matching tokens
    (function name, mime type) instead of empty content. Cheap and
    deterministic.
    """
    if not non_text:
        return ""
    bits: list[str] = []
    for entry in non_text:
        if "function_call" in entry:
            fc = entry["function_call"] or {}
            bits.append(f"tool call: {fc.get('name')}")
        elif "function_response" in entry:
            fr = entry["function_response"] or {}
            bits.append(f"tool response: {fr.get('name')}")
        elif "inline_data" in entry:
            idata = entry["inline_data"] or {}
            bits.append(f"inline data: {idata.get('mime_type')}")
        else:
            bits.append("event payload")
    return " | ".join(bits)


def _to_jsonable(value: Any) -> Any:
    """Coerce arbitrary values into a JSON-safe representation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _to_jsonable(dump(exclude_none=True))
        except Exception:  # noqa: BLE001, S110 — last-ditch fallback, repr() below covers any failure
            pass
    return repr(value)


__all__ = [
    "KEY_APP_ID",
    "KEY_AUTHOR",
    "KEY_EVENT_ID",
    "KEY_PARTS",
    "KEY_SESSION_ID",
    "KEY_TIMESTAMP",
    "KEY_USER_ID",
    "chunk_to_memory_entry",
    "content_to_text",
    "event_external_id",
    "event_to_remember_kwargs",
    "namespace_uuid",
    "serialise_non_text_parts",
    "session_uuid",
]
