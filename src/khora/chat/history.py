"""Chat history management with compression."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

if TYPE_CHECKING:
    pass


@dataclass
class ChatMessage:
    """A single message in the conversation."""

    id: UUID = field(default_factory=uuid4)
    role: Literal["user", "assistant", "system"] = "user"
    content: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)

    # Search results that informed this response
    search_context: list[dict] | None = None


@dataclass
class ConversationHistory:
    """Manages conversation history with compression."""

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID | None = None
    messages: list[ChatMessage] = field(default_factory=list)

    # Compressed summary of older messages
    compressed_summary: str = ""
    compressed_turn_count: int = 0

    # Configuration
    max_turns: int = 20
    compress_after: int = 10
    keep_recent: int = 3


class HistoryManager:
    """Manages chat history with automatic compression."""

    def __init__(
        self,
        max_turns: int = 20,
        compress_after: int = 10,
        keep_recent: int = 3,
        compression_model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize the history manager.

        Args:
            max_turns: Maximum conversation turns to maintain
            compress_after: Compress history after this many turns
            keep_recent: Keep this many recent turns uncompressed
            compression_model: LLM model to use for compression
        """
        self.max_turns = max_turns
        self.compress_after = compress_after
        self.keep_recent = keep_recent
        self.compression_model = compression_model
        self._histories: dict[UUID, ConversationHistory] = {}

    def get_or_create(self, conversation_id: UUID, namespace_id: UUID) -> ConversationHistory:
        """Get existing history or create new one.

        Args:
            conversation_id: Conversation identifier
            namespace_id: Namespace for the conversation

        Returns:
            ConversationHistory instance
        """
        if conversation_id not in self._histories:
            self._histories[conversation_id] = ConversationHistory(
                id=conversation_id,
                namespace_id=namespace_id,
                max_turns=self.max_turns,
                compress_after=self.compress_after,
                keep_recent=self.keep_recent,
            )
        return self._histories[conversation_id]

    def add_message(
        self,
        conversation_id: UUID,
        role: Literal["user", "assistant", "system"],
        content: str,
        search_context: list[dict] | None = None,
    ) -> ChatMessage:
        """Add a message to conversation history.

        Args:
            conversation_id: Conversation identifier
            role: Message role
            content: Message content
            search_context: Search results that informed this message

        Returns:
            Created ChatMessage
        """
        history = self._histories.get(conversation_id)
        if not history:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = ChatMessage(
            role=role,
            content=content,
            search_context=search_context,
        )
        history.messages.append(message)
        return message

    async def compress_if_needed(
        self,
        conversation_id: UUID,
    ) -> bool:
        """Compress history if it exceeds threshold.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if compression was performed
        """
        import litellm

        history = self._histories.get(conversation_id)
        if not history:
            return False

        turn_count = len(history.messages) // 2  # user + assistant = 1 turn

        if turn_count <= history.compress_after:
            return False

        # Messages to compress (all except recent)
        keep_count = history.keep_recent * 2
        to_compress = history.messages[:-keep_count] if keep_count else history.messages

        if not to_compress:
            return False

        # Generate summary
        summary_prompt = self._build_compression_prompt(to_compress, history.compressed_summary)

        response = await litellm.acompletion(
            model=self.compression_model,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=500,
        )

        new_summary = response.choices[0].message.content

        # Update history
        history.compressed_summary = new_summary
        history.compressed_turn_count += len(to_compress) // 2
        history.messages = history.messages[-keep_count:] if keep_count else []

        return True

    def _build_compression_prompt(
        self,
        messages: list[ChatMessage],
        existing_summary: str,
    ) -> str:
        """Build prompt for compressing messages.

        Args:
            messages: Messages to compress
            existing_summary: Existing summary to incorporate

        Returns:
            Compression prompt
        """
        parts = ["Summarize this conversation concisely, preserving key topics and decisions:"]

        if existing_summary:
            parts.append(f"\nPrevious summary:\n{existing_summary}")

        parts.append("\nNew messages to incorporate:")
        for msg in messages:
            content_preview = msg.content[:500]
            parts.append(f"\n{msg.role.upper()}: {content_preview}")

        parts.append("\n\nProvide a concise summary (2-3 sentences):")
        return "\n".join(parts)

    def get_context_messages(self, conversation_id: UUID) -> tuple[str, list[ChatMessage]]:
        """Get compressed summary and recent messages for context.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Tuple of (compressed_summary, recent_messages)
        """
        history = self._histories.get(conversation_id)
        if not history:
            return "", []

        return history.compressed_summary, list(history.messages)

    def clear(self, conversation_id: UUID) -> None:
        """Clear conversation history.

        Args:
            conversation_id: Conversation identifier
        """
        if conversation_id in self._histories:
            del self._histories[conversation_id]
