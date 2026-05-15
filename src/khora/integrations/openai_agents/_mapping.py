"""Translation helpers between OpenAI Agents SDK items and khora.

OpenAI Agents SDK identifies a conversation turn by ``TResponseInputItem``
(a JSON-serialisable ``ResponseInputItemParam`` TypedDict union from the
OpenAI Python SDK). A ``Session`` (or ``SessionABC``) stores an ordered
list of these items keyed by ``session_id``. ``SQLiteSession`` — the
reference upstream backend — serialises items via ``json.dumps`` /
``json.loads`` and recovers them in chronological order.

khora speaks documents + chunks scoped by a single ``namespace_id`` with
a first-class ``session_id`` column (#620). This module owns the two
directions:

* ``item_to_remember_kwargs(item, session_id, app_id, seq)`` — build the
  ``Khora.remember(**kwargs)`` payload for one item.
* ``chunk_to_item(chunk)`` — recover the original SDK item from a stored
  chunk so ``get_items`` can return it verbatim.

Each item becomes one khora document. Round-trip preserves the JSON
exactly: ``json.dumps`` is run at write time, and ``json.loads`` is run
at read time on the same bytes. Non-text items (function calls, tool
responses, refusals, etc.) round-trip because they live inside the JSON
payload rather than being projected to ``Document.content``.

Private to the adapter — not part of the public ``khora.integrations`` API.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_DNS, UUID, uuid5

if TYPE_CHECKING:  # pragma: no cover - typing only
    from khora.core.models.document import Chunk

# Stable UUID5 root for openai-agents session ids. Versioned so a future
# scheme change can rotate without colliding with shipped data.
_SESSION_ROOT = uuid5(NAMESPACE_DNS, "khora.integrations.openai_agents.session.v1")

# Document.external_id is VARCHAR(512). We stamp it with the SDK session
# id plus a monotonic sequence number so iteration order is preserved
# even when documents share a millisecond timestamp.
_EXTERNAL_ID_MAX_LEN = 512

# Metadata keys this adapter owns under ``Document.metadata.custom``.
# Prefix with ``oai_`` so caller-supplied metadata cannot collide.
KEY_APP_ID = "oai_app_id"
KEY_SESSION_ID = "oai_session_id"  # original SDK session string id
KEY_SEQ = "oai_seq"  # monotonic in-session ordering (0, 1, 2, ...)
KEY_ITEM_JSON = "oai_item"  # the verbatim JSON-encoded TResponseInputItem
KEY_ROLE = "oai_role"  # convenience: role/type for filtering, redundant with item
KEY_TYPE = "oai_type"  # message / function_call / function_call_output / ...


def session_uuid(session_id: str) -> UUID:
    """Map an SDK session id (free-form string) to a stable khora session UUID.

    The SDK uses arbitrary strings for ``Session.session_id`` (e.g.
    ``"conversation_123"``). khora's ``session_id`` column is a UUID.
    UUID-shaped session ids round-trip verbatim; anything else maps via
    UUID5 so the same string always lands on the same khora session.
    """
    try:
        return UUID(session_id)
    except (ValueError, AttributeError):
        return uuid5(_SESSION_ROOT, session_id)


def event_external_id(session_id: str, seq: int) -> str:
    """Build the khora ``Document.external_id`` for one item in a session.

    Format: ``oai:<session_id>:<seq>``. The seq number is the monotonic
    in-session ordering; combined with the session id it is globally
    unique per khora namespace. Long session ids are hashed so we always
    fit under the 512-char DB column cap.
    """
    raw = f"oai:{session_id}:{seq}"
    if len(raw) <= _EXTERNAL_ID_MAX_LEN:
        return raw
    digest = hashlib.sha1(session_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"oai:h{digest}:{seq}"


def item_text(item: Any) -> str:
    """Best-effort plain-text projection of a ``TResponseInputItem``.

    Used as the ``Document.content`` body so vector recall has something
    meaningful to embed against. The verbatim JSON still lives in
    ``metadata.custom[KEY_ITEM_JSON]`` so this projection is purely a
    side channel for the embedder.

    Heuristics, in order:

    * ``role`` + ``content`` (string) → ``"role: content"`` — covers the
      simple ``{"role": "user", "content": "..."}`` message shape.
    * ``content`` (list of part dicts with ``text``) → concatenated text.
    * Function call: ``"tool call: <name>(<args>)"``.
    * Function output: ``"tool result: <output>"`` (truncated).
    * Fallback: ``str(item)`` so we never embed an empty string.
    """
    if not isinstance(item, dict):
        return str(item)

    role = item.get("role")
    raw_content = item.get("content")

    if isinstance(raw_content, str):
        return f"{role}: {raw_content}" if role else raw_content
    if isinstance(raw_content, list):
        pieces: list[str] = []
        for part in raw_content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                pieces.append(text)
        if pieces:
            joined = "\n".join(pieces)
            return f"{role}: {joined}" if role else joined

    item_type = item.get("type")
    if item_type == "function_call":
        name = item.get("name", "?")
        args = item.get("arguments", "")
        return f"tool call: {name}({args})"
    if item_type == "function_call_output":
        output = item.get("output", "")
        # Truncate noisy tool outputs so we don't embed thousands of
        # characters of stack trace into a single chunk.
        if isinstance(output, str) and len(output) > 2000:
            output = output[:2000] + "…"
        return f"tool result: {output}"

    # Fallback — embed a stringified form so chunks aren't empty.
    return json.dumps(item, default=str)


def item_to_remember_kwargs(
    item: Any,
    *,
    session_id: str,
    app_id: str,
    seq: int,
) -> dict[str, Any]:
    """Translate one SDK item into ``Khora.remember(**kwargs)``.

    Args:
        item: A ``TResponseInputItem``-shaped dict (the SDK's union of
            message / function_call / function_call_output / etc.).
        session_id: The SDK session id this item belongs to.
        app_id: Free-form adapter app identifier, stamped into metadata.
        seq: Monotonic in-session ordering (0-based).

    Returns:
        Keyword dict ready for ``Khora.remember(**kwargs)``.
    """
    payload = json.dumps(item, default=str)
    text = item_text(item)
    role: str | None = None
    item_type: str | None = None
    if isinstance(item, dict):
        role = item.get("role") if isinstance(item.get("role"), str) else None
        item_type = item.get("type") if isinstance(item.get("type"), str) else None

    metadata: dict[str, Any] = {
        KEY_APP_ID: app_id,
        KEY_SESSION_ID: session_id,
        KEY_SEQ: seq,
        KEY_ITEM_JSON: payload,
        KEY_ROLE: role,
        KEY_TYPE: item_type,
    }

    label = role or item_type or "item"
    title = f"oai:{label}:{seq}"

    return {
        "content": text or json.dumps(item, default=str),
        "title": title,
        "source": f"openai_agents:{app_id}",
        "metadata": metadata,
        "external_id": event_external_id(session_id, seq),
        "session_id": session_uuid(session_id),
        # Conversation turns aren't corpus documents — entity extraction
        # on every chat turn is wasteful and pollutes the graph. Callers
        # who want extraction can rewrite ingested memory through
        # ``Khora.remember`` directly.
        "entity_types": [],
        "relationship_types": [],
    }


def chunk_to_item(chunk: Chunk) -> Any | None:
    """Recover the original ``TResponseInputItem`` from a stored chunk.

    Returns ``None`` if the chunk wasn't written by this adapter (no
    ``oai_item`` key in ``metadata.custom``) or if the stored JSON has
    been corrupted. The SDK silently skips invalid items in its reference
    ``SQLiteSession`` and we mirror that behaviour.
    """
    custom = (chunk.metadata.custom if chunk.metadata else {}) or {}
    raw = custom.get(KEY_ITEM_JSON)
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None
    # Already-decoded (some storage backends may store JSONB natively).
    return raw


def chunk_seq(chunk: Chunk) -> int | None:
    """Return the monotonic in-session sequence number stamped at write."""
    custom = (chunk.metadata.custom if chunk.metadata else {}) or {}
    seq = custom.get(KEY_SEQ)
    if isinstance(seq, int):
        return seq
    if isinstance(seq, str) and seq.isdigit():
        return int(seq)
    return None


__all__ = [
    "KEY_APP_ID",
    "KEY_ITEM_JSON",
    "KEY_ROLE",
    "KEY_SEQ",
    "KEY_SESSION_ID",
    "KEY_TYPE",
    "chunk_seq",
    "chunk_to_item",
    "event_external_id",
    "item_text",
    "item_to_remember_kwargs",
    "session_uuid",
]
