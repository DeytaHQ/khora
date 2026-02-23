"""Conversation chunker for Slack message grouping.

Groups Slack messages into coherent conversation chunks using
thread-awareness, temporal proximity, and semantic similarity.
Individual messages remain retrievable via per-message metadata
with character offsets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import Chunker, ChunkResult


@dataclass
class SlackMessage:
    """A single Slack message."""

    text: str
    author: str
    timestamp: datetime
    message_id: str
    thread_ts: str | None = None
    channel: str | None = None
    reactions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlackMessage:
        """Create from a dictionary.

        Args:
            data: Dictionary with message fields. The ``timestamp`` value
                  can be an ISO-format string or a :class:`datetime` instance.

        Returns:
            SlackMessage instance
        """
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            text=data["text"],
            author=data["author"],
            timestamp=ts,
            message_id=data["message_id"],
            thread_ts=data.get("thread_ts"),
            channel=data.get("channel"),
            reactions=data.get("reactions", []),
        )


@dataclass
class ConversationChunkerConfig:
    """Configuration for the conversation chunker."""

    time_gap_minutes: int = 15
    session_gap_minutes: int = 30
    max_group_size: int = 50
    min_group_size: int = 2
    semantic_threshold: float | None = None
    include_message_metadata: bool = True


class ConversationChunker(Chunker):
    """Chunker that groups Slack messages into conversation chunks.

    Three-layer grouping strategy:
    1. **Thread grouping** – messages sharing a ``thread_ts`` are kept together.
    2. **Temporal windowing** – top-level messages are split when the gap
       between consecutive messages exceeds ``time_gap_minutes``.
    3. **Semantic similarity** *(optional)* – further splits groups when
       cosine similarity drops below ``semantic_threshold``.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        config: ConversationChunkerConfig | None = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.config = config or ConversationChunkerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[ChunkResult]:
        """Parse a JSON array of message dicts and chunk them.

        Args:
            text: JSON string containing a list of SlackMessage dicts.

        Returns:
            List of ChunkResult objects.
        """
        if not text or not text.strip():
            return []
        raw = json.loads(text)
        messages = [SlackMessage.from_dict(m) for m in raw]
        return self.chunk_messages(messages)

    def chunk_messages(self, messages: list[SlackMessage]) -> list[ChunkResult]:
        """Group messages into conversation chunks.

        Args:
            messages: List of SlackMessage objects.

        Returns:
            List of ChunkResult objects with per-message metadata
            including ``session_id`` for session boundary tracking.
        """
        if not messages:
            return []

        # Sort by timestamp
        messages = sorted(messages, key=lambda m: m.timestamp)

        # Step 1: separate threads from top-level
        threads, top_level = self._group_by_threads(messages)

        # Step 2: split top-level messages by time gaps
        top_groups = self._split_by_time_gaps(top_level, self.config.time_gap_minutes)

        # Combine all groups: threads first, then top-level groups
        all_groups: list[list[SlackMessage]] = list(threads.values()) + top_groups

        # Step 3: enforce size limits
        all_groups = self._enforce_size_limits(all_groups)

        # Step 4: assign session IDs based on session boundary gaps
        session_ids = self._assign_session_ids(all_groups)

        # Build ChunkResults
        results: list[ChunkResult] = []
        for idx, group in enumerate(all_groups):
            if group:
                results.append(self._build_chunk_result(group, idx, session_id=session_ids[idx]))

        # Re-index sequentially
        for i, r in enumerate(results):
            r.index = i

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_by_threads(
        self, messages: list[SlackMessage]
    ) -> tuple[dict[str, list[SlackMessage]], list[SlackMessage]]:
        """Separate threaded messages from top-level messages.

        Returns:
            (threads_dict, top_level_messages) where threads_dict maps
            thread_ts to message lists.
        """
        threads: dict[str, list[SlackMessage]] = {}
        top_level: list[SlackMessage] = []

        for msg in messages:
            if msg.thread_ts:
                threads.setdefault(msg.thread_ts, []).append(msg)
            else:
                top_level.append(msg)

        # Sort each thread by timestamp
        for ts in threads:
            threads[ts].sort(key=lambda m: m.timestamp)

        return threads, top_level

    def _split_by_time_gaps(self, messages: list[SlackMessage], gap_minutes: int) -> list[list[SlackMessage]]:
        """Split sorted messages when the time gap exceeds the threshold.

        Args:
            messages: Sorted list of messages.
            gap_minutes: Gap threshold in minutes.

        Returns:
            List of message groups.
        """
        if not messages:
            return []

        groups: list[list[SlackMessage]] = [[messages[0]]]
        gap_seconds = gap_minutes * 60

        for msg in messages[1:]:
            prev = groups[-1][-1]
            delta = (msg.timestamp - prev.timestamp).total_seconds()
            if delta > gap_seconds:
                groups.append([msg])
            else:
                groups[-1].append(msg)

        return groups

    def _enforce_size_limits(self, groups: list[list[SlackMessage]]) -> list[list[SlackMessage]]:
        """Split groups exceeding max_group_size and merge tiny groups.

        Args:
            groups: List of message groups.

        Returns:
            Size-adjusted groups.
        """
        max_size = self.config.max_group_size
        min_size = self.config.min_group_size

        # Split oversized groups
        split: list[list[SlackMessage]] = []
        for group in groups:
            if len(group) <= max_size:
                split.append(group)
            else:
                for i in range(0, len(group), max_size):
                    split.append(group[i : i + max_size])

        # Merge undersized groups with their nearest neighbour
        if len(split) <= 1:
            return split

        merged: list[list[SlackMessage]] = [split[0]]
        for group in split[1:]:
            if len(merged[-1]) < min_size or len(group) < min_size:
                merged[-1].extend(group)
            else:
                merged.append(group)

        # Final pass: if last group is undersized, merge with previous
        if len(merged) > 1 and len(merged[-1]) < min_size:
            merged[-2].extend(merged[-1])
            merged.pop()

        return merged

    def _assign_session_ids(self, groups: list[list[SlackMessage]]) -> list[int]:
        """Assign session IDs to groups based on time gaps between them.

        Groups separated by more than ``session_gap_minutes`` are assigned
        different session IDs.  Groups within the same session share an ID.

        Args:
            groups: Ordered list of message groups.

        Returns:
            List of session IDs (one per group), starting from 0.
        """
        if not groups:
            return []

        session_gap_seconds = self.config.session_gap_minutes * 60
        session_ids: list[int] = [0]
        current_session = 0

        for i in range(1, len(groups)):
            prev_group = groups[i - 1]
            curr_group = groups[i]

            if prev_group and curr_group:
                prev_end = max(m.timestamp for m in prev_group)
                curr_start = min(m.timestamp for m in curr_group)
                gap = (curr_start - prev_end).total_seconds()

                if gap > session_gap_seconds:
                    current_session += 1

            session_ids.append(current_session)

        return session_ids

    def _format_group(self, messages: list[SlackMessage]) -> str:
        """Render messages as ``[HH:MM] author: text`` lines.

        Args:
            messages: Messages to format.

        Returns:
            Formatted string.
        """
        lines: list[str] = []
        for msg in messages:
            time_str = msg.timestamp.strftime("%H:%M")
            lines.append(f"[{time_str}] {msg.author}: {msg.text}")
        return "\n".join(lines)

    def _build_chunk_result(self, messages: list[SlackMessage], index: int, *, session_id: int = 0) -> ChunkResult:
        """Build a ChunkResult with per-message metadata.

        Args:
            messages: Messages in this chunk.
            index: Chunk index.
            session_id: Session boundary identifier for cross-session retrieval.

        Returns:
            ChunkResult with content and metadata including ``session_id``.
        """
        content = self._format_group(messages)

        # Compute per-message character offsets
        message_meta: list[dict[str, Any]] = []
        offset = 0
        for i, msg in enumerate(messages):
            time_str = msg.timestamp.strftime("%H:%M")
            line = f"[{time_str}] {msg.author}: {msg.text}"
            start = offset
            end = offset + len(line)
            message_meta.append(
                {
                    "id": msg.message_id,
                    "author": msg.author,
                    "timestamp": msg.timestamp.isoformat(),
                    "start_char": start,
                    "end_char": end,
                }
            )
            # +1 for the newline separator (except after last message)
            offset = end + (1 if i < len(messages) - 1 else 0)

        # Collect metadata
        authors = list(dict.fromkeys(msg.author for msg in messages))
        channel = next((m.channel for m in messages if m.channel), None)
        thread_ts = next((m.thread_ts for m in messages if m.thread_ts), None)

        metadata: dict[str, Any] = {
            "source_type": "slack_conversation",
            "channel": channel,
            "thread_ts": thread_ts,
            "session_id": session_id,
            "message_count": len(messages),
            "time_start": messages[0].timestamp.isoformat(),
            "time_end": messages[-1].timestamp.isoformat(),
            "authors": authors,
        }

        if self.config.include_message_metadata:
            metadata["messages"] = message_meta

        return ChunkResult(
            content=content,
            index=index,
            start_char=0,
            end_char=len(content),
            token_count=self.count_tokens(content),
            metadata=metadata,
        )
