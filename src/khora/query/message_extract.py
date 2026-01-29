"""Utilities for extracting individual messages from conversation chunks.

Conversation chunks produced by :class:`ConversationChunker` embed
per-message metadata (author, timestamp, character offsets) so that
individual messages can be retrieved from search results.
"""

from __future__ import annotations

from typing import Any


def extract_messages_from_chunk(chunk_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the per-message metadata list from a chunk.

    Args:
        chunk_metadata: The ``metadata`` dict of a chunk or search result.

    Returns:
        List of message dicts (each with ``id``, ``author``, ``timestamp``,
        ``start_char``, ``end_char``).  Returns an empty list when the
        metadata does not contain conversation message data.
    """
    return chunk_metadata.get("messages", [])


def extract_message_text(chunk_content: str, start_char: int, end_char: int) -> str:
    """Slice a message's text from the chunk content using character offsets.

    Args:
        chunk_content: The full chunk text.
        start_char: Start character offset (inclusive).
        end_char: End character offset (exclusive).

    Returns:
        The substring corresponding to the message.
    """
    return chunk_content[start_char:end_char]


def find_message_in_chunk(
    chunk_content: str,
    chunk_metadata: dict[str, Any],
    message_id: str,
) -> dict[str, Any] | None:
    """Find a specific message by ID within a conversation chunk.

    Args:
        chunk_content: The full chunk text.
        chunk_metadata: The chunk's metadata dict.
        message_id: The ``message_id`` to look up.

    Returns:
        A dict with ``id``, ``author``, ``timestamp``, and ``text`` keys,
        or ``None`` if the message was not found.
    """
    for msg in extract_messages_from_chunk(chunk_metadata):
        if msg.get("id") == message_id:
            text = extract_message_text(chunk_content, msg["start_char"], msg["end_char"])
            return {
                "id": msg["id"],
                "author": msg["author"],
                "timestamp": msg["timestamp"],
                "text": text,
            }
    return None
