"""Translation helpers between Hermes ``MemoryProvider`` turns and khora.

Hermes drives a long-running agent session and calls
``MemoryProvider.initialize(session_id, **kwargs)`` once per session. The
``kwargs`` always include ``agent_identity`` (a stable string per agent)
plus ``hermes_home`` and ``platform``. Together with ``session_id``,
``agent_identity`` is the natural tenancy key: two sessions for the same
agent share long-term memory, while sessions across different agents
stay isolated. khora speaks documents + chunks scoped by a single
``namespace_id`` with a first-class ``session_id`` column (#620); this
module is the single place those shapes meet.

Mapping summary:

* ``(agent_identity, session_id)`` → ``khora namespace_id`` via
  :func:`derive_namespace_uuid` (UUID5 of ``"hermes:{agent_identity}:{session_id}"``).
* One Hermes conversation turn = one ``Document`` whose ``content`` is
  ``"USER: {user}\\n\\nASSISTANT: {assistant}"``. The verbatim user and
  assistant text are preserved under ``metadata.custom`` so the original
  turn can be reconstructed at recall time (mirrors the ``oai_item``
  pattern in the openai_agents adapter).
* ``Document.external_id`` is ``"hermes:{session_id}:{turn_seq}"`` so
  re-ingesting the same turn is idempotent.

Private to the adapter — not part of the public ``khora.integrations`` API.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_OID, UUID, uuid5

from khora.core.models.document import Document

if TYPE_CHECKING:
    from khora.core.models.recall import RecallChunk, RecallEntity

# Stable UUID5 root for hermes namespaces. Derived from the OID namespace
# against the literal module path so it is reproducible without storing
# a hard-coded UUID literal in source.
UUID_NAMESPACE_HERMES: UUID = uuid5(NAMESPACE_OID, "khora.integrations.hermes")

# Metadata keys this adapter owns under ``Document.metadata``. Prefix
# with ``hermes_`` so caller-supplied metadata cannot collide.
KEY_SOURCE = "source"
KEY_EXTERNAL_ID = "external_id"
KEY_SESSION_ID = "session_id"
KEY_TURN_SEQ = "turn_seq"
KEY_USER_CONTENT = "hermes_user_content"
KEY_ASSISTANT_CONTENT = "hermes_assistant_content"
KEY_OAI_SEQ = "oai_seq"  # ordering key, mirrors openai_agents convention
KEY_OCCURRED_AT = "occurred_at"


def derive_namespace_uuid(agent_identity: str, session_id: str) -> UUID:
    """Derive the khora namespace UUID for a Hermes (agent_identity, session_id) pair.

    Two ``KhoraMemoryProvider`` calls with the same ``agent_identity``
    and ``session_id`` map to the same khora namespace. This is the
    Hermes equivalent of the ADK ``namespace_uuid(app_name, user_id)``
    helper — agent identity is the tenancy key, session id is the
    conversation scope.
    """
    return uuid5(UUID_NAMESPACE_HERMES, f"hermes:{agent_identity}:{session_id}")


def turn_external_id(session_id: str, turn_seq: int) -> str:
    """Build the khora ``Document.external_id`` for one turn in a session.

    Format: ``hermes:{session_id}:{turn_seq}``. Combined with the session
    id, the monotonic turn sequence is globally unique per khora
    namespace and enables idempotent dedup via
    ``Document.metadata.custom["external_id"]`` in
    ``KhoraMemoryProvider.sync_turn``.
    """
    return f"hermes:{session_id}:{turn_seq}"


def turn_to_document(
    user_content: str,
    assistant_content: str,
    *,
    session_id: str,
    turn_seq: int,
    namespace_id: UUID,
) -> Document:
    """Translate one Hermes conversation turn into a khora ``Document``.

    The user and assistant texts are concatenated into the embedded
    content so vector recall has a single coherent turn to match
    against; the verbatim originals live in ``metadata.custom`` so the
    turn can be reconstructed losslessly on read.
    """
    content = f"USER: {user_content}\n\nASSISTANT: {assistant_content}"
    occurred_at = datetime.now(UTC).isoformat()
    custom: dict[str, Any] = {
        KEY_EXTERNAL_ID: turn_external_id(session_id, turn_seq),
        KEY_SOURCE: "hermes",
        KEY_SESSION_ID: session_id,
        KEY_TURN_SEQ: turn_seq,
        KEY_USER_CONTENT: user_content,
        KEY_ASSISTANT_CONTENT: assistant_content,
        KEY_OAI_SEQ: turn_seq,
        KEY_OCCURRED_AT: occurred_at,
    }
    return Document(
        namespace_id=namespace_id,
        content=content,
        source_type="conversation",
        metadata={"custom": custom},
    )


def format_memory_context(
    chunks: list[RecallChunk],
    entities: list[RecallEntity] | None = None,
) -> str:
    """Render recalled memory into the ``<memory-context>`` block Hermes prefetches.

    The output is consumed directly by the LLM via Hermes's prefetch
    path. Format choices:

    * Top 5 chunks max — further truncation is the caller's job.
    * Each chunk lists ``[score, ISO date]`` then the (truncated) content,
      capped at 500 chars with an ellipsis. The score nudges the model
      to weigh higher-confidence memories more heavily; the date helps
      with recency questions.
    * Entity section is omitted when ``entities`` is empty or ``None``.
    * Empty ``chunks`` returns an explicit "No prior memories found."
      payload — Hermes propagates this back to the LLM as the abstention
      signal so the model knows not to hallucinate context.
    """
    if not chunks:
        return "<memory-context>\nNo prior memories found.\n</memory-context>"

    lines: list[str] = [
        "The user has prior context about this conversation. Relevant memories:",
        "",
    ]
    for idx, chunk in enumerate(chunks[:5], start=1):
        score = getattr(chunk, "score", 0.0) or 0.0
        created_at = getattr(chunk, "created_at", None)
        date_str = created_at.strftime("%Y-%m-%d") if created_at is not None else "unknown"
        content = getattr(chunk, "content", "") or ""
        if len(content) > 500:
            content = content[:500].rstrip() + "..."
        lines.append(f"{idx}. [score: {score:.2f}, {date_str}] {content}")

    if entities:
        lines.append("")
        lines.append("Key entities mentioned:")
        for entity in entities:
            name = getattr(entity, "name", "")
            etype = getattr(entity, "entity_type", "")
            mentions = getattr(entity, "mention_count", 0)
            lines.append(f"- {name} ({etype}, mentions: {mentions})")

    body = "\n".join(lines)
    return f"<memory-context>\n{body}\n</memory-context>"


def message_pair_iter(messages: list[dict]) -> Iterator[tuple[str, str]]:
    """Iterate Hermes ``on_pre_compress`` messages as (user, assistant) pairs.

    System and tool messages are skipped. A user message with no
    following assistant message yields ``(user, "")``; an assistant
    message with no preceding user message yields ``("", assistant)``
    (rare opener case in some Hermes flows). Anything that isn't a
    dict with the expected ``role`` / ``content`` shape is skipped.
    """
    pending_user: str | None = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if role == "user":
            if pending_user is not None:
                # Two user messages in a row — flush the first as a dangling pair.
                yield (pending_user, "")
            pending_user = content
        else:  # assistant
            if pending_user is None:
                yield ("", content)
            else:
                yield (pending_user, content)
                pending_user = None
    if pending_user is not None:
        yield (pending_user, "")


__all__ = [
    "KEY_ASSISTANT_CONTENT",
    "KEY_EXTERNAL_ID",
    "KEY_OAI_SEQ",
    "KEY_OCCURRED_AT",
    "KEY_SESSION_ID",
    "KEY_SOURCE",
    "KEY_TURN_SEQ",
    "KEY_USER_CONTENT",
    "UUID_NAMESPACE_HERMES",
    "derive_namespace_uuid",
    "format_memory_context",
    "message_pair_iter",
    "turn_external_id",
    "turn_to_document",
]
