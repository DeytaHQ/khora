"""Unit tests for khora.chat.history — ChatMessage, ConversationHistory, HistoryManager."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.chat.history import ChatMessage, ConversationHistory, HistoryManager

# ---------------------------------------------------------------------------
# ChatMessage dataclass
# ---------------------------------------------------------------------------


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        msg = ChatMessage()
        assert isinstance(msg.id, UUID)
        assert msg.role == "user"
        assert msg.content == ""
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}
        assert msg.search_context is None

    def test_all_fields(self) -> None:
        """All fields can be set explicitly."""
        msg_id = uuid4()
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        ctx = [{"content": "result", "score": 0.9}]

        msg = ChatMessage(
            id=msg_id,
            role="assistant",
            content="Hello!",
            timestamp=ts,
            metadata={"key": "value"},
            search_context=ctx,
        )

        assert msg.id == msg_id
        assert msg.role == "assistant"
        assert msg.content == "Hello!"
        assert msg.timestamp == ts
        assert msg.metadata == {"key": "value"}
        assert msg.search_context == ctx

    def test_system_role(self) -> None:
        """System role is accepted."""
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"


# ---------------------------------------------------------------------------
# ConversationHistory dataclass
# ---------------------------------------------------------------------------


class TestConversationHistory:
    """Tests for ConversationHistory dataclass."""

    def test_defaults(self) -> None:
        """Default values are set correctly."""
        conv = ConversationHistory()
        assert isinstance(conv.id, UUID)
        assert conv.namespace_id is None
        assert conv.messages == []
        assert conv.compressed_summary == ""
        assert conv.compressed_turn_count == 0
        assert conv.max_turns == 20
        assert conv.compress_after == 10
        assert conv.keep_recent == 3

    def test_custom_fields(self) -> None:
        """Custom fields can be set."""
        conv_id = uuid4()
        ns_id = uuid4()
        conv = ConversationHistory(
            id=conv_id,
            namespace_id=ns_id,
            max_turns=50,
            compress_after=25,
            keep_recent=5,
        )
        assert conv.id == conv_id
        assert conv.namespace_id == ns_id
        assert conv.max_turns == 50
        assert conv.compress_after == 25
        assert conv.keep_recent == 5


# ---------------------------------------------------------------------------
# HistoryManager
# ---------------------------------------------------------------------------


class TestHistoryManagerGetOrCreate:
    """Tests for HistoryManager.get_or_create()."""

    def test_creates_new_conversation(self) -> None:
        """Creates a new ConversationHistory when ID is new."""
        mgr = HistoryManager(max_turns=30, compress_after=15, keep_recent=4)
        conv_id = uuid4()
        ns_id = uuid4()

        result = mgr.get_or_create(conv_id, ns_id)

        assert isinstance(result, ConversationHistory)
        assert result.id == conv_id
        assert result.namespace_id == ns_id
        assert result.max_turns == 30
        assert result.compress_after == 15
        assert result.keep_recent == 4
        assert result.messages == []

    def test_returns_existing_conversation(self) -> None:
        """Returns existing conversation on second call."""
        mgr = HistoryManager()
        conv_id = uuid4()
        ns_id = uuid4()

        first = mgr.get_or_create(conv_id, ns_id)
        first.messages.append(ChatMessage(content="test"))

        second = mgr.get_or_create(conv_id, ns_id)
        assert second is first
        assert len(second.messages) == 1


class TestHistoryManagerAddMessage:
    """Tests for HistoryManager.add_message()."""

    def test_adds_message_to_conversation(self) -> None:
        """Adds a message to the correct conversation."""
        mgr = HistoryManager()
        conv_id = uuid4()
        ns_id = uuid4()
        mgr.get_or_create(conv_id, ns_id)

        msg = mgr.add_message(conv_id, "user", "Hello!")

        assert isinstance(msg, ChatMessage)
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.search_context is None

    def test_adds_message_with_search_context(self) -> None:
        """Search context is stored on the message."""
        mgr = HistoryManager()
        conv_id = uuid4()
        ns_id = uuid4()
        mgr.get_or_create(conv_id, ns_id)

        ctx = [{"content": "result", "score": 0.9}]
        msg = mgr.add_message(conv_id, "assistant", "Answer", search_context=ctx)

        assert msg.search_context == ctx

    def test_raises_for_unknown_conversation(self) -> None:
        """Raises ValueError for unknown conversation ID."""
        mgr = HistoryManager()

        with pytest.raises(ValueError, match="not found"):
            mgr.add_message(uuid4(), "user", "Hello")

    def test_multiple_messages_accumulate(self) -> None:
        """Messages accumulate in conversation history."""
        mgr = HistoryManager()
        conv_id = uuid4()
        ns_id = uuid4()
        mgr.get_or_create(conv_id, ns_id)

        mgr.add_message(conv_id, "user", "Q1")
        mgr.add_message(conv_id, "assistant", "A1")
        mgr.add_message(conv_id, "user", "Q2")

        _, messages = mgr.get_context_messages(conv_id)
        assert len(messages) == 3


class TestHistoryManagerGetContextMessages:
    """Tests for HistoryManager.get_context_messages()."""

    def test_returns_summary_and_messages(self) -> None:
        """Returns compressed summary and recent messages."""
        mgr = HistoryManager()
        conv_id = uuid4()
        ns_id = uuid4()
        conv = mgr.get_or_create(conv_id, ns_id)

        mgr.add_message(conv_id, "user", "Hello")
        mgr.add_message(conv_id, "assistant", "Hi there")
        conv.compressed_summary = "Previous chat about greetings."

        summary, messages = mgr.get_context_messages(conv_id)

        assert summary == "Previous chat about greetings."
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there"

    def test_unknown_conversation_returns_empty(self) -> None:
        """Returns empty summary and empty list for unknown conversation."""
        mgr = HistoryManager()

        summary, messages = mgr.get_context_messages(uuid4())

        assert summary == ""
        assert messages == []

    def test_returns_copies_not_references(self) -> None:
        """Returned messages list is a copy, not the internal list."""
        mgr = HistoryManager()
        conv_id = uuid4()
        ns_id = uuid4()
        mgr.get_or_create(conv_id, ns_id)
        mgr.add_message(conv_id, "user", "Hello")

        _, messages = mgr.get_context_messages(conv_id)
        messages.clear()

        # Internal list should be unchanged
        _, messages2 = mgr.get_context_messages(conv_id)
        assert len(messages2) == 1


class TestHistoryManagerClear:
    """Tests for HistoryManager.clear()."""

    def test_removes_conversation(self) -> None:
        """Clears the conversation from the manager."""
        mgr = HistoryManager()
        conv_id = uuid4()
        ns_id = uuid4()
        mgr.get_or_create(conv_id, ns_id)
        mgr.add_message(conv_id, "user", "Hello")

        mgr.clear(conv_id)

        summary, messages = mgr.get_context_messages(conv_id)
        assert summary == ""
        assert messages == []

    def test_clear_unknown_conversation_is_noop(self) -> None:
        """Clearing unknown conversation doesn't raise."""
        mgr = HistoryManager()
        mgr.clear(uuid4())  # Should not raise


class TestHistoryManagerCompression:
    """Tests for HistoryManager.compress_if_needed()."""

    async def test_compresses_when_over_threshold(self) -> None:
        """Calls LLM to compress when turn count exceeds threshold."""
        mgr = HistoryManager(compress_after=2, keep_recent=1)
        conv_id = uuid4()
        ns_id = uuid4()
        mgr.get_or_create(conv_id, ns_id)

        # Add enough messages to trigger compression (> 2 turns = > 4 messages)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            mgr.add_message(conv_id, role, f"Message {i}")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Compressed summary"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_acomp:
            result = await mgr.compress_if_needed(conv_id)

        assert result is True
        mock_acomp.assert_awaited_once()

        # Verify compression updated the history
        summary, messages = mgr.get_context_messages(conv_id)
        assert summary == "Compressed summary"
        # Only keep_recent * 2 messages should remain
        assert len(messages) == 2

    async def test_no_compression_when_under_threshold(self) -> None:
        """Does NOT call LLM when turn count is under threshold."""
        mgr = HistoryManager(compress_after=10, keep_recent=3)
        conv_id = uuid4()
        ns_id = uuid4()
        mgr.get_or_create(conv_id, ns_id)

        # Add only 2 turns (4 messages) - under threshold of 10
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            mgr.add_message(conv_id, role, f"Message {i}")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
            result = await mgr.compress_if_needed(conv_id)

        assert result is False
        mock_acomp.assert_not_awaited()

    async def test_compression_unknown_conversation(self) -> None:
        """Returns False for unknown conversation."""
        mgr = HistoryManager()
        result = await mgr.compress_if_needed(uuid4())
        assert result is False

    async def test_compression_preserves_recent_messages(self) -> None:
        """After compression, only keep_recent * 2 messages remain."""
        mgr = HistoryManager(compress_after=2, keep_recent=2)
        conv_id = uuid4()
        ns_id = uuid4()
        mgr.get_or_create(conv_id, ns_id)

        # Add 8 messages (4 turns) - exceeds compress_after=2
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            mgr.add_message(conv_id, role, f"Message {i}")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of early messages"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            await mgr.compress_if_needed(conv_id)

        _, messages = mgr.get_context_messages(conv_id)
        # keep_recent=2, so 2*2=4 messages should remain
        assert len(messages) == 4
        # The remaining messages should be the last 4
        assert messages[0].content == "Message 4"
        assert messages[-1].content == "Message 7"

    async def test_compression_increments_turn_count(self) -> None:
        """Compression increments compressed_turn_count."""
        mgr = HistoryManager(compress_after=2, keep_recent=1)
        conv_id = uuid4()
        ns_id = uuid4()
        conv = mgr.get_or_create(conv_id, ns_id)

        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            mgr.add_message(conv_id, role, f"Message {i}")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            await mgr.compress_if_needed(conv_id)

        # 4 messages compressed = 2 turns
        assert conv.compressed_turn_count == 2


class TestBuildCompressionPrompt:
    """Tests for HistoryManager._build_compression_prompt()."""

    def test_without_existing_summary(self) -> None:
        """Builds prompt without existing summary."""
        mgr = HistoryManager()
        messages = [
            ChatMessage(role="user", content="What is X?"),
            ChatMessage(role="assistant", content="X is Y."),
        ]

        prompt = mgr._build_compression_prompt(messages, "")

        assert "Summarize this conversation" in prompt
        assert "USER: What is X?" in prompt
        assert "ASSISTANT: X is Y." in prompt
        assert "Previous summary" not in prompt

    def test_with_existing_summary(self) -> None:
        """Builds prompt with existing summary included."""
        mgr = HistoryManager()
        messages = [ChatMessage(role="user", content="Follow-up question")]

        prompt = mgr._build_compression_prompt(messages, "We discussed topic Z.")

        assert "Previous summary" in prompt
        assert "We discussed topic Z." in prompt
        assert "USER: Follow-up question" in prompt
